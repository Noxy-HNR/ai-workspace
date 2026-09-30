"""The tuner's memory: everything it has measured, in one SQLite file.

Tuning is slow (every load is seconds, every measurement is a request) so the tuner must not repeat work it has
already done, and a new model should start from what worked on the same architecture. `state/tuner.db` holds:

- runs          one per tuning job: the model, mode, power state, baseline and winner
- trials        one per configuration tried: projection, load result, outcome
- samples       speed measurements, tagged with workload and power state (samples are only comparable when their
                power keys match)
- quality       quality-suite results per trial
- calibrations  the measured gap between llama-fit-params and reality, per family, server build and MTP use; the
                strategy reserves it before it trusts a projection
- best          the winning configuration per model file, the starting point for the next run

Model files are identified by `fingerprint()` rather than by name, so a renamed or re-downloaded file is still the
same model and a changed file is a new one. A `schema_version` row lets a later version migrate an old file forward
instead of starting over.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import statistics
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Iterator, Sequence

from modeltuner import paths
from modeltuner.backend import LoadResult, QualityResult, Sample
from modeltuner.fitparams import Projection
from modeltuner.knobs import Config

SCHEMA_VERSION = 1
HASH_WINDOW = 16 * 1024 * 1024        # only the first and last 16 MiB are hashed; see fingerprint()
HASH_CHUNK = 4 * 1024 * 1024
CALIBRATION_WINDOW = 10               # how many recent projections a calibration is averaged over
# Outcomes that mean "this configuration does not work here". Mirrors the list the strategy refuses to retry.
FAILURE_OUTCOMES = ("oom", "crash", "spill", "refused", "ram", "verify", "timeout")
_TUPLE_FIELDS = ("passthrough",)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id                   TEXT PRIMARY KEY,
    model_file           TEXT,
    model_fingerprint    TEXT,
    family               TEXT,
    mode                 TEXT,
    started              REAL,
    finished             REAL,
    status               TEXT,
    power_key            TEXT,
    baseline_config_json TEXT,
    best_config_json     TEXT,
    summary_json         TEXT
);

CREATE TABLE IF NOT EXISTS trials (
    id               INTEGER PRIMARY KEY,
    run_id           TEXT,
    model_fingerprint TEXT,
    config_key       TEXT,
    config_json      TEXT,
    phase            TEXT,
    projection_json  TEXT,
    load_json        TEXT,
    outcome          TEXT,
    reason           TEXT,
    created          REAL
);

CREATE TABLE IF NOT EXISTS samples (
    id             INTEGER PRIMARY KEY,
    trial_id       INTEGER,
    workload       TEXT,
    gen_tps        REAL,
    prompt_tps     REAL,
    prompt_n       INTEGER,
    predicted_n    INTEGER,
    draft_n        INTEGER,
    draft_accepted INTEGER,
    wall_s         REAL,
    power_key      TEXT,
    created        REAL
);

CREATE TABLE IF NOT EXISTS quality (
    id           INTEGER PRIMARY KEY,
    trial_id     INTEGER,
    suite        TEXT,
    passed       INTEGER,
    total        INTEGER,
    tokens       INTEGER,
    seconds      REAL,
    details_json TEXT,
    created      REAL
);

CREATE TABLE IF NOT EXISTS calibrations (
    id            INTEGER PRIMARY KEY,
    family        TEXT,
    server_binary TEXT,
    uses_mtp      INTEGER,
    projected_mib INTEGER,
    actual_mib    INTEGER,
    created       REAL
);

CREATE TABLE IF NOT EXISTS best (
    model_fingerprint TEXT PRIMARY KEY,
    model_file        TEXT,
    family            TEXT,
    config_json       TEXT,
    metrics_json      TEXT,
    power_key         TEXT,
    run_id            TEXT,
    updated           REAL
);

CREATE INDEX IF NOT EXISTS trials_lookup ON trials (model_fingerprint, config_key);
CREATE INDEX IF NOT EXISTS trials_run ON trials (run_id, id);
CREATE INDEX IF NOT EXISTS samples_trial ON samples (trial_id, workload, power_key);
CREATE INDEX IF NOT EXISTS quality_trial ON quality (trial_id);
CREATE INDEX IF NOT EXISTS calibrations_lookup ON calibrations (family, server_binary, uses_mtp, id);
CREATE INDEX IF NOT EXISTS best_family ON best (family, updated);
CREATE INDEX IF NOT EXISTS runs_started ON runs (started);
"""


def fingerprint(path: Path) -> str:
    """Identify a model file by its size, mtime and a SHA-256 of its first and last 16 MiB.

    Hashing 16 GB before every run would cost more than the run itself, and a re-quantised file always differs at the
    header and at the tail, so the two windows are enough. mtime is in the digest as well, so a file that was
    replaced or touched is treated as a new one even if the sampled bytes happen to match.
    """
    path = Path(path)
    st = path.stat()
    size = st.st_size
    h = hashlib.sha256()
    h.update(f"{size}:{st.st_mtime_ns}".encode())
    with path.open("rb") as f:
        _hash_window(f, h, 0, min(size, HASH_WINDOW))
        if size > HASH_WINDOW:
            _hash_window(f, h, size - HASH_WINDOW, HASH_WINDOW)
    return h.hexdigest()


def _hash_window(f: Any, h: Any, start: int, length: int) -> None:
    f.seek(start)
    left = length
    while left > 0:
        block = f.read(min(left, HASH_CHUNK))
        if not block:
            break
        h.update(block)
        left -= len(block)


def config_to_json(cfg: Config) -> str:
    """Config -> canonical JSON: equal configs always produce identical text, so rows compare cleanly."""
    return _dumps(asdict(cfg))


def config_from_json(text: str) -> Config:
    """JSON -> Config. Tuple fields come back as tuples, so the result is equal to the config that went in.

    Fields this version does not know about are ignored, which is what lets a newer tuner add a knob and still read
    rows written by an older one (a missing field simply keeps its default).
    """
    known = {f.name for f in fields(Config)}
    kwargs: dict[str, Any] = {}
    for name, value in json.loads(text).items():
        if name not in known:
            continue
        kwargs[name] = tuple(value) if name in _TUPLE_FIELDS and value is not None else value
    return Config(**kwargs)


def _dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


class History:
    """The tuner's SQLite memory. One connection, guarded by a lock; every write is one transaction.

    The file is `paths.DB_PATH` (`state/tuner.db`) unless another one is given. It is a log of measurements rather
    than the user's data, so WAL with normal durability is enough: a run that dies mid-write loses at most its last
    row, and the next run measures that again.
    """

    def __init__(self, db_path: Path | None = None):
        self.path = Path(db_path) if db_path is not None else paths.DB_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db: sqlite3.Connection | None = sqlite3.connect(str(self.path), check_same_thread=False,
                                                            isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        try:
            self._create()
        except BaseException:
            self.close()          # never leave a half-open handle on a file we refuse to use
            raise

    def close(self) -> None:
        """Close the connection. The file stays valid, so a later History() on the same path reopens it."""
        with self.lock:
            if self.db is not None:
                self.db.close()
                self.db = None

    # ------------------------------------------------------------------ plumbing
    def _open(self) -> sqlite3.Connection:
        if self.db is None:
            raise RuntimeError(f"history for {self.path} is closed")
        return self.db

    @contextmanager
    def _txn(self) -> Iterator[sqlite3.Connection]:
        """One transaction per write, so a run's rows land together and two threads never interleave."""
        with self.lock:
            db = self._open()
            db.execute("BEGIN IMMEDIATE")
            try:
                yield db
            except BaseException:
                db.execute("ROLLBACK")
                raise
            db.execute("COMMIT")

    def _rows(self, sql: str, params: Sequence[Any] = ()) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self._open().execute(sql, params).fetchall()]

    def _create(self) -> None:
        with self.lock:
            db = self._open()
            db.executescript(_SCHEMA)
            row = db.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
            found = row["v"] if row else None
            if found is None:
                db.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
            elif found > SCHEMA_VERSION:
                raise RuntimeError(f"{self.path} was written by a newer Model Tuner (schema {found} > {SCHEMA_VERSION})")
            elif found < SCHEMA_VERSION:
                self._migrate(found)

    def _migrate(self, from_version: int) -> None:
        """Bring an older file up to SCHEMA_VERSION. Version 1 is the first schema, so there is nothing to migrate
        yet; later versions add their ALTER TABLE steps here, one `if from_version < n` block each."""

    # ------------------------------------------------------------------ runs
    def start_run(self, run_id: str, model_file: str, fingerprint: str, family: str, mode: str, power_key: str,
                  baseline: Config) -> None:
        """Open a run. Re-entering an existing run id replaces the row, so a resumed run starts clean."""
        with self._txn() as db:
            db.execute(
                "INSERT OR REPLACE INTO runs (id, model_file, model_fingerprint, family, mode, started, finished,"
                " status, power_key, baseline_config_json, best_config_json, summary_json)"
                " VALUES (?,?,?,?,?,?,NULL,?,?,?,NULL,NULL)",
                (run_id, model_file, fingerprint, family, mode, time.time(), "running", power_key,
                 config_to_json(baseline)),
            )

    def finish_run(self, run_id: str, status: str, best: Config | None, summary: dict) -> None:
        """Close a run with its outcome, the winning config (None when the run found nothing) and a summary dict."""
        with self._txn() as db:
            db.execute(
                "UPDATE runs SET finished=?, status=?, best_config_json=?, summary_json=? WHERE id=?",
                (time.time(), status, config_to_json(best) if best is not None else None, _dumps(summary or {}), run_id),
            )

    def runs(self, limit: int = 20) -> list[dict]:
        """Recent runs, newest first, with their configs and summary decoded. Each dict carries the row's `id` and
        the same value as `run_id`, plus the baseline and best as Configs."""
        rows = self._rows("SELECT * FROM runs ORDER BY started DESC, id DESC LIMIT ?", (limit,))
        out: list[dict] = []
        for r in rows:
            out.append({
                "id": r["id"],
                "run_id": r["id"],
                "model_file": r["model_file"],
                "model_fingerprint": r["model_fingerprint"],
                "family": r["family"],
                "mode": r["mode"],
                "status": r["status"],
                "power_key": r["power_key"],
                "started": r["started"],
                "finished": r["finished"],
                "seconds": round(r["finished"] - r["started"], 1) if r["started"] and r["finished"] else None,
                "baseline": config_from_json(r["baseline_config_json"]) if r["baseline_config_json"] else None,
                "best": config_from_json(r["best_config_json"]) if r["best_config_json"] else None,
                "summary": json.loads(r["summary_json"] or "{}"),
            })
        return out

    # ------------------------------------------------------------------ trials
    def add_trial(self, run_id: str, fingerprint: str, cfg: Config, phase: str,
                  projection: Projection | None) -> int:
        """Record a configuration about to be tried. Returns the trial id that samples, quality and outcomes hang off."""
        with self._txn() as db:
            cur = db.execute(
                "INSERT INTO trials (run_id, model_fingerprint, config_key, config_json, phase, projection_json,"
                " load_json, outcome, reason, created) VALUES (?,?,?,?,?,?,NULL,'pending','',?)",
                (run_id, fingerprint, cfg.key(), config_to_json(cfg), phase,
                 _dumps(projection.as_dict()) if projection is not None else None, time.time()),
            )
            return int(cur.lastrowid)

    def set_trial_outcome(self, trial_id: int, outcome: str, reason: str = "",
                          load: LoadResult | None = None) -> None:
        """Record how a trial ended. A load result is kept when given; without one the stored one is left alone."""
        with self._txn() as db:
            if load is None:
                db.execute("UPDATE trials SET outcome=?, reason=? WHERE id=?", (outcome, reason, trial_id))
            else:
                db.execute("UPDATE trials SET outcome=?, reason=?, load_json=? WHERE id=?",
                           (outcome, reason, _dumps(asdict(load)), trial_id))

    def known_failure(self, fingerprint: str, cfg: Config) -> str | None:
        """The most recent failure outcome ("oom", "crash", "spill", ...) recorded for this exact file and config.

        The tuner uses this to never retry a configuration that already failed: an OOM is expensive to rediscover
        and a crash usually repeats. Successes are not failures, so a config that was measured is never reported
        here; whether a stale failure is still believed is the caller's call, not the database's.
        """
        marks = ",".join("?" * len(FAILURE_OUTCOMES))
        rows = self._rows(
            f"SELECT outcome FROM trials WHERE model_fingerprint=? AND config_key=? AND outcome IN ({marks})"
            " ORDER BY id DESC LIMIT 1",
            (fingerprint, cfg.key(), *FAILURE_OUTCOMES),
        )
        return rows[0]["outcome"] if rows else None

    # ------------------------------------------------------------------ measurements
    def add_samples(self, trial_id: int, samples: list[Sample]) -> None:
        """Store a batch of speed measurements for one trial (one workload, one power state, one load)."""
        if not samples:
            return
        now = time.time()
        with self._txn() as db:
            db.executemany(
                "INSERT INTO samples (trial_id, workload, gen_tps, prompt_tps, prompt_n, predicted_n, draft_n,"
                " draft_accepted, wall_s, power_key, created) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                [(trial_id, s.workload, s.gen_tps, s.prompt_tps, s.prompt_n, s.predicted_n, s.draft_n,
                  s.draft_accepted, s.wall_s, s.power_key, now) for s in samples],
            )

    def samples_for(self, fingerprint: str, cfg: Config, workload: str, power_key: str) -> list[Sample]:
        """Every stored measurement of one config, on one workload, in one power state; oldest first.

        The power key is part of the lookup on purpose: a sample taken on battery or in a quiet power mode is not
        comparable with one taken plugged in, so the tuner only ever asks for its own power state.
        """
        rows = self._rows(
            "SELECT s.workload, s.gen_tps, s.prompt_tps, s.prompt_n, s.predicted_n, s.draft_n, s.draft_accepted,"
            " s.wall_s, s.power_key FROM samples s JOIN trials t ON t.id = s.trial_id"
            " WHERE t.model_fingerprint=? AND t.config_key=? AND s.workload=? AND s.power_key=?"
            " ORDER BY s.created, s.id",
            (fingerprint, cfg.key(), workload, power_key),
        )
        return [Sample(**r) for r in rows]

    def add_quality(self, trial_id: int, result: QualityResult) -> None:
        """Store one quality suite's result for a trial."""
        with self._txn() as db:
            db.execute(
                "INSERT INTO quality (trial_id, suite, passed, total, tokens, seconds, details_json, created)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (trial_id, result.suite, result.passed, result.total, result.tokens, result.seconds,
                 _dumps(result.details), time.time()),
            )

    # ------------------------------------------------------------------ calibration
    def add_calibration(self, family: str, server_binary: str, uses_mtp: bool, projected_mib: int,
                        actual_mib: int) -> None:
        """Record one projected-versus-measured comparison, so the next run trusts the projection less blindly."""
        with self._txn() as db:
            db.execute(
                "INSERT INTO calibrations (family, server_binary, uses_mtp, projected_mib, actual_mib, created)"
                " VALUES (?,?,?,?,?,?)",
                (family, server_binary, int(bool(uses_mtp)), projected_mib, actual_mib, time.time()),
            )

    def calibration_error(self, family: str, server_binary: str, uses_mtp: bool) -> tuple[float, int] | None:
        """Median (actual - projected) MiB over the most recent matching loads, and how many loads that was.

        The strategy reserves this before it trusts a projection: llama-fit-params cannot see an MTP draft context,
        and the architecture can change. `None` means this combination has never been measured, so the caller keeps
        its default overhead. The median ignores the occasional wild load; the count says how much to trust it.
        """
        rows = self._rows(
            "SELECT projected_mib, actual_mib FROM calibrations"
            " WHERE family=? AND server_binary=? AND uses_mtp=? ORDER BY id DESC LIMIT ?",
            (family, server_binary, int(bool(uses_mtp)), CALIBRATION_WINDOW),
        )
        if not rows:
            return None
        errors = [r["actual_mib"] - r["projected_mib"] for r in rows]
        return float(statistics.median(errors)), len(errors)

    # ------------------------------------------------------------------ best configs
    def set_best(self, fingerprint: str, model_file: str, family: str, cfg: Config, metrics: dict,
                 power_key: str, run_id: str) -> None:
        """Record the winning config for a model file, replacing any earlier winner.

        Written once per run and by a resumed run, so a repeat must be an upsert, never an error.
        """
        with self._txn() as db:
            db.execute(
                "INSERT INTO best (model_fingerprint, model_file, family, config_json, metrics_json, power_key,"
                " run_id, updated) VALUES (?,?,?,?,?,?,?,?)"
                " ON CONFLICT(model_fingerprint) DO UPDATE SET model_file=excluded.model_file,"
                " family=excluded.family, config_json=excluded.config_json, metrics_json=excluded.metrics_json,"
                " power_key=excluded.power_key, run_id=excluded.run_id, updated=excluded.updated",
                (fingerprint, model_file, family, config_to_json(cfg), _dumps(metrics), power_key, run_id, time.time()),
            )

    def best_for(self, fingerprint: str) -> tuple[Config, dict] | None:
        """The last winning config and metrics for one exact model file, or None if it has never been tuned."""
        rows = self._rows("SELECT config_json, metrics_json FROM best WHERE model_fingerprint=?", (fingerprint,))
        if not rows:
            return None
        return config_from_json(rows[0]["config_json"]), json.loads(rows[0]["metrics_json"] or "{}")

    def best_for_family(self, family: str, exclude_fingerprint: str | None = None) -> list[tuple[str, Config, dict]]:
        """Winning configs of the other models in one family, newest first: where a new model starts from.

        `exclude_fingerprint` drops the model being tuned, so a model never inherits its own previous answer.
        """
        rows = self._rows(
            "SELECT model_file, config_json, metrics_json FROM best WHERE family=?"
            " AND (? IS NULL OR model_fingerprint <> ?) ORDER BY updated DESC, rowid DESC",
            (family, exclude_fingerprint, exclude_fingerprint),
        )
        return [(r["model_file"], config_from_json(r["config_json"]), json.loads(r["metrics_json"] or "{}"))
                for r in rows]
