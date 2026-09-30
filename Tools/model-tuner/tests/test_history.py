"""Tests for modeltuner.history: the tuner's memory of what it measured.

Each test pins one thing the next run depends on: that rows survive the round trip with their configs intact
(tuples and nulls included), that calibration and failure memory only ever look at the recent and the exact, that a
model file is identified by its bytes, and that two threads can write at once.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from dataclasses import asdict
from pathlib import Path

import pytest

from modeltuner import paths
from modeltuner.backend import LoadResult, QualityResult, Sample
from modeltuner.fitparams import Projection
from modeltuner.history import SCHEMA_VERSION, History, config_from_json, config_to_json, fingerprint
from modeltuner.knobs import Config

FP_A = "a" * 64
FP_B = "b" * 64
POWER = "ac|best_performance|saver-off"
OTHER_POWER = "dc|balanced|saver-on"
TABLES = ("schema_version", "runs", "trials", "samples", "quality", "calibrations", "best")


def base_cfg(**overrides) -> Config:
    cfg = Config(ctx=32768, ngl=999, threads=8, batch=1024, ubatch=256)
    return cfg.with_(**overrides) if overrides else cfg


def full_cfg() -> Config:
    """Every field set, including the ones that are usually None and the verbatim passthrough tuple."""
    return Config(ctx=131072, ngl=24, threads=12, batch=2048, ubatch=512, cache_k="f16", cache_v="q4_0",
                  flash_attn=True, spec="draft-mtp,ngram-mod", draft_n_max=3, ngram_n_match=16,
                  reasoning_budget=4096, reasoning_effort="high", threads_batch=6, n_cpu_moe=4, parallel=2,
                  jinja=False, passthrough=("--rope-scaling", "none", "--no-mmap"))


def a_projection(context_mib: int = 800) -> Projection:
    return Projection(device_model_mib=5000, device_context_mib=context_mib, device_compute_mib=250,
                      host_model_mib=1000, host_context_mib=32, host_compute_mib=16, device_total_mib=9000,
                      fitted_ngl=999, fitted_ctx=32768, seconds=2.1)


def a_load(**overrides) -> LoadResult:
    fields = dict(seconds=4.2, vram_used_mib=6200, vram_other_mib=1500, host_avail_gb=8.5, pid=4321, n_ctx=32768)
    return LoadResult(**{**fields, **overrides})


def a_sample(workload: str = "prose", gen: float = 42.0, power: str = POWER, **overrides) -> Sample:
    fields = dict(workload=workload, gen_tps=gen, prompt_tps=None, prompt_n=128, predicted_n=64, draft_n=None,
                  draft_accepted=None, wall_s=1.5, power_key=power)
    return Sample(**{**fields, **overrides})


def stored(h: History, sql: str, params: tuple = ()) -> list[dict]:
    """Read a row straight from the table: these tests check the columns named in the schema, not just the API."""
    with h.lock:
        return [dict(r) for r in h.db.execute(sql, params).fetchall()]


def make_file(path: Path, mib: int) -> Path:
    chunk = bytes(1 << 20)
    with path.open("wb") as f:
        for _ in range(mib):
            f.write(chunk)
    return path


@pytest.fixture
def hist(tmp_path: Path):
    h = History(tmp_path / "tuner.db")
    yield h
    h.close()


class TestSetup:
    def test_wal_mode_and_schema_version(self, hist: History):
        assert stored(hist, "PRAGMA journal_mode")[0]["journal_mode"].lower() == "wal"
        assert stored(hist, "SELECT MAX(version) AS v FROM schema_version")[0]["v"] == SCHEMA_VERSION

    def test_every_table_exists(self, hist: History):
        names = {r["name"] for r in stored(hist, "SELECT name FROM sqlite_master WHERE type='table'")}
        assert set(TABLES) <= names

    def test_default_path_is_the_state_db(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        target = tmp_path / "state" / "tuner.db"
        monkeypatch.setattr(paths, "DB_PATH", target)
        h = History()
        try:
            assert h.path == target
            assert target.exists()          # the state directory is created on demand
        finally:
            h.close()

    def test_closed_history_refuses_work(self, hist: History):
        hist.close()
        hist.close()                        # closing twice is not an error
        with pytest.raises(RuntimeError):
            hist.runs()


class TestConfigJson:
    def test_round_trip_is_equal(self):
        cfg = full_cfg()
        assert config_from_json(config_to_json(cfg)) == cfg

    def test_passthrough_comes_back_as_a_tuple(self):
        back = config_from_json(config_to_json(full_cfg()))
        assert isinstance(back.passthrough, tuple)
        assert back.passthrough == ("--rope-scaling", "none", "--no-mmap")

    def test_none_fields_stay_none(self):
        cfg = Config(ctx=16384, threads=8, batch=1024, ubatch=256, draft_n_max=None, ngram_n_match=None,
                     reasoning_budget=None, threads_batch=None, n_cpu_moe=None, passthrough=())
        back = config_from_json(config_to_json(cfg))
        assert back == cfg
        assert back.draft_n_max is None
        assert back.reasoning_budget is None
        assert back.n_cpu_moe is None
        assert back.passthrough == ()

    def test_key_survives_the_round_trip(self):
        # Trials and stored winners are looked up by Config.key(), so a round trip must not change identity.
        cfg = full_cfg()
        assert config_from_json(config_to_json(cfg)).key() == cfg.key()

    def test_text_is_canonical(self):
        # The same config always produces the same text, whichever order its fields arrive in.
        text = config_to_json(base_cfg())
        shuffled = json.dumps(dict(sorted(asdict(base_cfg()).items(), reverse=True)))
        assert json.loads(text) == json.loads(shuffled)
        assert config_to_json(base_cfg()) == text

    def test_unknown_fields_are_ignored(self):
        # A newer tuner may add a knob; an old row must still load, keeping the new field's default.
        raw = json.loads(config_to_json(base_cfg()))
        raw["flash_attn_v2"] = True
        assert config_from_json(json.dumps(raw)) == base_cfg()


class TestRuns:
    def test_round_trip(self, hist: History):
        baseline = base_cfg()
        hist.start_run("run-1", "C:/AI/models/qwen.gguf", FP_A, "qwen3", "quick", POWER, baseline)
        best = full_cfg()
        hist.finish_run("run-1", "ok", best, {"improvement": 0.12, "loads": 7})

        [row] = hist.runs()
        assert row["id"] == row["run_id"] == "run-1"
        assert row["model_file"] == "C:/AI/models/qwen.gguf"
        assert row["model_fingerprint"] == FP_A
        assert row["family"] == "qwen3"
        assert row["mode"] == "quick"
        assert row["status"] == "ok"
        assert row["power_key"] == POWER
        assert row["baseline"] == baseline
        assert row["best"] == best
        assert row["summary"] == {"improvement": 0.12, "loads": 7}
        assert row["finished"] >= row["started"] > 0
        assert row["seconds"] >= 0

    def test_an_open_run_has_no_winner_yet(self, hist: History):
        hist.start_run("run-1", "m.gguf", FP_A, "qwen3", "quick", POWER, base_cfg())
        assert [r["status"] for r in hist.runs()] == ["running"]
        assert hist.runs()[0]["best"] is None
        assert hist.runs()[0]["summary"] == {}

    def test_finish_without_a_winner(self, hist: History):
        hist.start_run("run-1", "m.gguf", FP_A, "qwen3", "quick", POWER, base_cfg())
        hist.finish_run("run-1", "stopped", None, {})
        [row] = hist.runs()
        assert row["best"] is None
        assert row["status"] == "stopped"

    def test_newest_first_and_limited(self, hist: History):
        # Two runs started in the same clock tick still order newest first: `id DESC` breaks the tie.
        for name in ("r1", "r2", "r3"):
            hist.start_run(name, f"{name}.gguf", FP_A, "qwen3", "quick", POWER, base_cfg())
        assert [r["run_id"] for r in hist.runs()] == ["r3", "r2", "r1"]
        assert [r["run_id"] for r in hist.runs(limit=2)] == ["r3", "r2"]
        assert hist.runs(limit=0) == []

    def test_restarting_a_run_replaces_the_row(self, hist: History):
        hist.start_run("run-1", "m.gguf", FP_A, "qwen3", "quick", POWER, base_cfg())
        hist.finish_run("run-1", "ok", full_cfg(), {"loads": 3})
        hist.start_run("run-1", "m.gguf", FP_A, "qwen3", "full", OTHER_POWER, base_cfg())
        [row] = hist.runs()
        assert row["status"] == "running"
        assert row["mode"] == "full"
        assert row["power_key"] == OTHER_POWER
        assert row["best"] is None

    def test_finish_run_of_an_unknown_run_is_not_an_error(self, hist: History):
        hist.finish_run("nope", "ok", base_cfg(), {})
        assert hist.runs() == []


class TestTrials:
    def test_round_trip(self, hist: History):
        hist.start_run("run-1", "m.gguf", FP_A, "qwen3", "quick", POWER, base_cfg())
        cfg = base_cfg(ctx=65536)
        proj = a_projection()
        trial_id = hist.add_trial("run-1", FP_A, cfg, "fit", proj)

        [row] = stored(hist, "SELECT * FROM trials WHERE id=?", (trial_id,))
        assert row["run_id"] == "run-1"
        assert row["model_fingerprint"] == FP_A
        assert row["config_key"] == cfg.key()
        assert config_from_json(row["config_json"]) == cfg
        assert row["phase"] == "fit"
        assert json.loads(row["projection_json"]) == proj.as_dict()
        assert row["load_json"] is None
        assert row["outcome"] == "pending"
        assert row["reason"] == ""
        assert row["created"] > 0

    def test_trial_without_projection(self, hist: History):
        trial_id = hist.add_trial("run-1", FP_A, base_cfg(), "spec", None)
        [row] = stored(hist, "SELECT * FROM trials WHERE id=?", (trial_id,))
        assert row["projection_json"] is None

    def test_outcome_and_load_result(self, hist: History):
        trial_id = hist.add_trial("run-1", FP_A, base_cfg(), "fit", a_projection())
        load = a_load()
        hist.set_trial_outcome(trial_id, "measured", "43.9 tok/s on prose", load)

        [row] = stored(hist, "SELECT * FROM trials WHERE id=?", (trial_id,))
        assert row["outcome"] == "measured"
        assert row["reason"] == "43.9 tok/s on prose"
        assert json.loads(row["load_json"]) == asdict(load)

    def test_outcome_without_a_load_keeps_the_earlier_one(self, hist: History):
        trial_id = hist.add_trial("run-1", FP_A, base_cfg(), "fit", a_projection())
        hist.set_trial_outcome(trial_id, "measured", "", a_load())
        hist.set_trial_outcome(trial_id, "measured", "second look", None)
        [row] = stored(hist, "SELECT outcome, reason, load_json FROM trials WHERE id=?", (trial_id,))
        assert row["reason"] == "second look"
        assert json.loads(row["load_json"]) == asdict(a_load())

    def test_ids_are_distinct_and_ordered(self, hist: History):
        ids = [hist.add_trial("run-1", FP_A, base_cfg(ctx=c), "fit", None) for c in (8192, 16384, 32768)]
        assert len(set(ids)) == 3
        assert ids == sorted(ids)
        assert hist.known_failure(FP_A, base_cfg(ctx=8192)) is None


class TestKnownFailure:
    def test_returns_the_latest_failure(self, hist: History):
        cfg = base_cfg(ctx=65536)
        first = hist.add_trial("run-1", FP_A, cfg, "fit", None)
        hist.set_trial_outcome(first, "oom", "cudaMalloc failed")
        second = hist.add_trial("run-2", FP_A, cfg, "fit", None)
        hist.set_trial_outcome(second, "crash", "exited with -1073741819")

        assert hist.known_failure(FP_A, cfg) == "crash"

    def test_only_the_exact_config(self, hist: History):
        bad = base_cfg(ctx=65536)
        good = base_cfg(ctx=32768)
        oomed = hist.add_trial("run-1", FP_A, bad, "fit", None)
        hist.set_trial_outcome(oomed, "oom", "cudaMalloc failed")
        measured = hist.add_trial("run-1", FP_A, good, "baseline", None)
        hist.set_trial_outcome(measured, "measured", "")

        assert hist.known_failure(FP_A, good) is None
        assert hist.known_failure(FP_A, base_cfg(ctx=131072)) is None      # never tried
        assert hist.known_failure(FP_A, base_cfg(ctx=65536, ngl=24)) is None   # one field apart

    def test_only_the_exact_file(self, hist: History):
        cfg = base_cfg()
        trial_id = hist.add_trial("run-1", FP_A, cfg, "fit", None)
        hist.set_trial_outcome(trial_id, "spill", "WDDM paged the weights out")

        assert hist.known_failure(FP_A, cfg) == "spill"
        assert hist.known_failure(FP_B, cfg) is None                      # same config, different file

    def test_pending_and_skipped_are_not_failures(self, hist: History):
        cfg = base_cfg()
        hist.add_trial("run-1", FP_A, cfg, "fit", None)                    # still pending
        skipped = hist.add_trial("run-1", FP_A, base_cfg(ctx=8192), "spec", None)
        hist.set_trial_outcome(skipped, "skipped", "would not fit in VRAM")

        assert hist.known_failure(FP_A, cfg) is None
        assert hist.known_failure(FP_A, base_cfg(ctx=8192)) is None

    def test_empty_history(self, hist: History):
        assert hist.known_failure(FP_A, base_cfg()) is None


class TestSamples:
    def test_round_trip_and_filters(self, hist: History):
        cfg = base_cfg()
        trial_id = hist.add_trial("run-1", FP_A, cfg, "fit", None)
        prose = [a_sample("prose", 42.0), a_sample("prose", 43.5, draft_n=2, draft_accepted=1)]
        code = [a_sample("code", 18.0)]
        hist.add_samples(trial_id, prose + code)

        assert hist.samples_for(FP_A, cfg, "prose", POWER) == prose
        assert hist.samples_for(FP_A, cfg, "code", POWER) == code
        assert hist.samples_for(FP_A, cfg, "prefill", POWER) == []
        assert hist.samples_for(FP_A, cfg, "prose", OTHER_POWER) == []     # a different power state is not comparable
        assert hist.samples_for(FP_A, base_cfg(ctx=8192), "prose", POWER) == []
        assert hist.samples_for(FP_B, cfg, "prose", POWER) == []

    def test_null_and_draft_fields_survive(self, hist: History):
        trial_id = hist.add_trial("run-1", FP_A, base_cfg(), "fit", None)
        prefill = a_sample("prefill", gen=None, prompt_tps=1800.0, prompt_n=4096, predicted_n=1)
        draft = a_sample("code", 22.0, draft_n=4, draft_accepted=3, gen_tps=22.0)
        hist.add_samples(trial_id, [prefill, draft])

        [got_prefill] = hist.samples_for(FP_A, base_cfg(), "prefill", POWER)
        assert got_prefill == prefill
        assert got_prefill.gen_tps is None
        [got_draft] = hist.samples_for(FP_A, base_cfg(), "code", POWER)
        assert got_draft.draft_n == 4
        assert got_draft.draft_accepted == 3
        assert got_draft.acceptance == pytest.approx(0.75)

    def test_samples_from_several_trials_are_oldest_first(self, hist: History):
        cfg = base_cfg()
        first = hist.add_trial("run-1", FP_A, cfg, "fit", None)
        hist.add_samples(first, [a_sample(gen=40.0), a_sample(gen=41.0)])
        second = hist.add_trial("run-2", FP_A, cfg, "confirm", None)
        hist.add_samples(second, [a_sample(gen=42.0)])

        assert [s.gen_tps for s in hist.samples_for(FP_A, cfg, "prose", POWER)] == [40.0, 41.0, 42.0]

    def test_empty_batch_is_a_no_op(self, hist: History):
        assert hist.add_samples(999, []) is None
        assert stored(hist, "SELECT COUNT(*) AS n FROM samples")[0]["n"] == 0

    def test_samples_are_attached_to_their_trial(self, hist: History):
        trial_id = hist.add_trial("run-1", FP_A, base_cfg(), "fit", None)
        hist.add_samples(trial_id, [a_sample()])
        [row] = stored(hist, "SELECT trial_id, workload, wall_s, power_key, created FROM samples")
        assert row["trial_id"] == trial_id
        assert row["workload"] == "prose"
        assert row["wall_s"] == 1.5
        assert row["power_key"] == POWER
        assert row["created"] > 0


class TestQuality:
    def test_round_trip(self, hist: History):
        trial_id = hist.add_trial("run-1", FP_A, base_cfg(), "quality", None)
        result = QualityResult(suite="reasoning", passed=7, total=8, tokens=4321, seconds=95.5,
                               details=[{"id": 3, "ok": False, "note": "ran out of tokens"}])
        hist.add_quality(trial_id, result)

        [row] = stored(hist, "SELECT * FROM quality WHERE trial_id=?", (trial_id,))
        assert row["suite"] == "reasoning"
        assert (row["passed"], row["total"], row["tokens"]) == (7, 8, 4321)
        assert row["seconds"] == 95.5
        assert json.loads(row["details_json"]) == result.details
        assert row["created"] > 0

    def test_details_that_are_not_json_survive(self, hist: History):
        # `default=str` keeps a suite from losing its row because one detail was not plain JSON.
        trial_id = hist.add_trial("run-1", FP_A, base_cfg(), "quality", None)
        hist.add_quality(trial_id, QualityResult(suite="tools", passed=1, total=2, tokens=10, seconds=1.0,
                                                 details=[{"server": object()}]))
        [row] = stored(hist, "SELECT details_json FROM quality WHERE trial_id=?", (trial_id,))
        assert isinstance(json.loads(row["details_json"])[0]["server"], str)

    def test_suites_accumulate_per_trial(self, hist: History):
        trial_id = hist.add_trial("run-1", FP_A, base_cfg(), "quality", None)
        hist.add_quality(trial_id, QualityResult(suite="reasoning", passed=7, total=8, tokens=1, seconds=1.0))
        hist.add_quality(trial_id, QualityResult(suite="tools", passed=5, total=6, tokens=2, seconds=2.0))
        suites = {r["suite"] for r in stored(hist, "SELECT suite FROM quality WHERE trial_id=?", (trial_id,))}
        assert suites == {"reasoning", "tools"}


class TestCalibration:
    def test_empty(self, hist: History):
        assert hist.calibration_error("qwen3", "server.exe", True) is None

    def test_median_of_the_error(self, hist: History):
        for projected, actual in ((1000, 1400), (2000, 2300), (3000, 3300)):
            hist.add_calibration("qwen3", "server.exe", True, projected, actual)

        assert hist.calibration_error("qwen3", "server.exe", True) == (300.0, 3)

    def test_negative_errors_are_kept(self, hist: History):
        hist.add_calibration("gemma3", "", False, 5000, 4800)      # the projection over-estimated
        assert hist.calibration_error("gemma3", "", False) == (-200.0, 1)

    def test_only_the_latest_ten_loads(self, hist: History):
        for i in range(12):
            hist.add_calibration("qwen3", "server.exe", False, 1000, 1000 + (i + 1))

        # errors 1..12 are stored in order; the window keeps 3..12, whose median is 7.5
        assert hist.calibration_error("qwen3", "server.exe", False) == (7.5, 10)

    def test_matching_is_exact(self, hist: History):
        hist.add_calibration("qwen3", "server.exe", True, 1000, 1500)
        assert hist.calibration_error("qwen3", "server.exe", True) == (500.0, 1)
        assert hist.calibration_error("qwen3", "other.exe", True) is None
        assert hist.calibration_error("qwen3", "", True) is None
        assert hist.calibration_error("gemma3", "server.exe", True) is None
        assert hist.calibration_error("qwen3", "server.exe", False) is None

    def test_two_keys_coexist(self, hist: History):
        hist.add_calibration("qwen3", "", True, 1000, 1500)         # MTP draft context, invisible to fit-params
        hist.add_calibration("qwen3", "", False, 1000, 1040)
        assert hist.calibration_error("qwen3", "", True) == (500.0, 1)
        assert hist.calibration_error("qwen3", "", False) == (40.0, 1)

    def test_a_wild_load_does_not_move_the_median(self, hist: History):
        for _ in range(4):
            hist.add_calibration("qwen3", "", False, 1000, 1100)
        hist.add_calibration("qwen3", "", False, 1000, 9000)        # another app grabbed the VRAM
        assert hist.calibration_error("qwen3", "", False) == (100.0, 5)

    def test_rows_are_stored_as_given(self, hist: History):
        hist.add_calibration("qwen3", "server.exe", True, 4000, 4500)
        columns = "family, server_binary, uses_mtp, projected_mib, actual_mib, created"
        [row] = stored(hist, f"SELECT {columns} FROM calibrations")
        assert row["family"] == "qwen3"
        assert row["server_binary"] == "server.exe"
        assert row["uses_mtp"] == 1
        assert (row["projected_mib"], row["actual_mib"]) == (4000, 4500)
        assert row["created"] > 0


class TestBest:
    def test_round_trip(self, hist: History):
        cfg = full_cfg()
        metrics = {"prose": 44.2, "code": 21.0, "prompt_tps": 1800.0, "score": 1.12}
        hist.set_best(FP_A, "C:/AI/models/qwen.gguf", "qwen3", cfg, metrics, POWER, "run-1")

        got_cfg, got_metrics = hist.best_for(FP_A)
        assert got_cfg == cfg
        assert isinstance(got_cfg.passthrough, tuple)
        assert got_metrics == metrics
        assert [r["run_id"] for r in stored(hist, "SELECT run_id FROM best")] == ["run-1"]

    def test_unknown_file(self, hist: History):
        assert hist.best_for(FP_A) is None

    def test_repeating_never_raises_and_replaces(self, hist: History):
        hist.set_best(FP_A, "m.gguf", "qwen3", base_cfg(), {"prose": 40.0}, POWER, "run-1")
        winner = base_cfg(ctx=131072, ngl=999)
        hist.set_best(FP_A, "m.gguf", "qwen3", winner, {"prose": 46.0}, OTHER_POWER, "run-2")

        [row] = stored(hist, "SELECT * FROM best")
        assert row["run_id"] == "run-2"
        assert row["power_key"] == OTHER_POWER
        assert config_from_json(row["config_json"]) == winner
        assert json.loads(row["metrics_json"]) == {"prose": 46.0}
        assert hist.best_for(FP_A) == (winner, {"prose": 46.0})

    def test_family_ordering_and_exclusion(self, hist: History):
        # Two models in one family, the later winner first.
        hist.set_best(FP_A, "a.gguf", "qwen3", base_cfg(ctx=8192), {"prose": 40.0}, POWER, "run-1")
        hist.set_best(FP_B, "b.gguf", "qwen3", base_cfg(ctx=32768), {"prose": 44.0}, POWER, "run-2")

        got = hist.best_for_family("qwen3")
        assert [(m, c.ctx) for m, c, _ in got] == [("b.gguf", 32768), ("a.gguf", 8192)]

        excluded = hist.best_for_family("qwen3", exclude_fingerprint=FP_B)
        assert [(m, c.ctx) for m, c, _ in excluded] == [("a.gguf", 8192)]
        assert hist.best_for_family("qwen3", exclude_fingerprint=FP_A)[0][0] == "b.gguf"
        assert hist.best_for_family("qwen3", exclude_fingerprint=FP_B + "c") == got

    def test_a_new_win_moves_the_model_to_the_front(self, hist: History):
        hist.set_best(FP_A, "a.gguf", "qwen3", base_cfg(), {"prose": 40.0}, POWER, "run-1")
        hist.set_best(FP_B, "b.gguf", "qwen3", base_cfg(), {"prose": 44.0}, POWER, "run-2")
        time.sleep(0.05)                       # `updated` is wall-clock; let the two wins be distinct
        hist.set_best(FP_A, "a.gguf", "qwen3", full_cfg(), {"prose": 52.0}, POWER, "run-3")

        got = hist.best_for_family("qwen3")
        assert [m for m, _, _ in got] == ["a.gguf", "b.gguf"]
        assert got[0][1] == full_cfg()

    def test_family_mismatch_and_empty(self, hist: History):
        hist.set_best(FP_A, "a.gguf", "qwen3", base_cfg(), {"prose": 40.0}, POWER, "run-1")
        assert hist.best_for_family("gemma3") == []
        assert hist.best_for_family("gemma3", exclude_fingerprint=FP_A) == []


class TestFingerprint:
    def test_stable_for_an_untouched_file(self, tmp_path: Path):
        p = make_file(tmp_path / "model.gguf", 1)
        assert fingerprint(p) == fingerprint(p)
        assert fingerprint(str(p)) == fingerprint(p)      # the CLI knows models by name, the runner by path

    def test_different_files_differ(self, tmp_path: Path):
        a = tmp_path / "a.gguf"
        a.write_bytes(b"GGUF" + b"\x00" * 4092)
        b = tmp_path / "b.gguf"
        b.write_bytes(b"GGUF" + b"\x01" * 4092)
        assert fingerprint(a) != fingerprint(b)

    def test_the_last_byte_counts(self, tmp_path: Path):
        p = make_file(tmp_path / "model.gguf", 64)
        before = fingerprint(p)
        st = p.stat()
        with p.open("r+b") as f:
            f.seek(-1, 2)
            f.write(b"\x01")
        # Size and mtime are put back, so only the bytes can explain a different value.
        os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))
        assert p.stat().st_size == st.st_size
        assert fingerprint(p) != before

    def test_size_and_mtime_count(self, tmp_path: Path):
        p = tmp_path / "model.gguf"
        p.write_bytes(b"GGUF" + b"\x00" * 4092)
        before = fingerprint(p)
        with p.open("ab") as f:
            f.write(b"\x00")                    # same bytes, one more of them
        assert fingerprint(p) != before

        p2 = tmp_path / "touched.gguf"
        p2.write_bytes(b"GGUF" + b"\x00" * 4092)
        before = fingerprint(p2)
        os.utime(p2, ns=(0, 0))                 # same bytes, same size, different mtime
        assert fingerprint(p2) != before

    def test_the_middle_is_deliberately_not_read(self, tmp_path: Path):
        # The two 16 MiB windows are a trade: a change between them is not noticed, but a 16 GB file is fingerprinted
        # in a fraction of a second, which is the whole point.
        p = make_file(tmp_path / "model.gguf", 64)
        before = fingerprint(p)
        st = p.stat()
        with p.open("r+b") as f:
            f.seek(32 * 1024 * 1024)
            f.write(b"\xff")
        os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))
        assert fingerprint(p) == before

    def test_files_smaller_than_a_window(self, tmp_path: Path):
        p = tmp_path / "small.gguf"
        p.write_bytes(b"GGUF" + b"\x00" * 1000)
        before = fingerprint(p)
        assert len(before) == 64
        int(before, 16)                              # hex, as the schema stores it
        p.write_bytes(b"GGUF" + b"\x00" * 999 + b"\x01")
        assert fingerprint(p) != before
        empty = tmp_path / "empty.gguf"
        empty.write_bytes(b"")
        assert fingerprint(empty) == fingerprint(empty)
        assert fingerprint(empty) != before

    def test_fast_enough_on_a_real_sized_file(self, tmp_path: Path):
        # 64 MB stands in for the 16 GB files in C:/AI/models: the same 32 MiB are read either way.
        p = make_file(tmp_path / "model.gguf", 64)
        t0 = time.perf_counter()
        fingerprint(p)
        assert time.perf_counter() - t0 < 1.0


class TestReopen:
    def test_data_survives_reopening(self, tmp_path: Path):
        db = tmp_path / "tuner.db"
        h = History(db)
        h.start_run("run-1", "m.gguf", FP_A, "qwen3", "quick", POWER, base_cfg())
        winner = full_cfg()
        h.finish_run("run-1", "ok", winner, {"loads": 7})
        trial_id = h.add_trial("run-1", FP_A, base_cfg(), "fit", a_projection())
        h.set_trial_outcome(trial_id, "measured", "", a_load())
        h.add_samples(trial_id, [a_sample()])
        h.add_quality(trial_id, QualityResult(suite="tools", passed=5, total=6, tokens=99, seconds=12.0))
        h.add_calibration("qwen3", "", True, 4000, 4500)
        h.set_best(FP_A, "m.gguf", "qwen3", winner, {"prose": 44.0}, POWER, "run-1")
        h.close()

        again = History(db)
        try:
            assert again.path == db
            [row] = again.runs()
            assert (row["status"], row["best"], row["summary"]) == ("ok", winner, {"loads": 7})
            assert again.best_for(FP_A) == (winner, {"prose": 44.0})
            assert again.calibration_error("qwen3", "", True) == (500.0, 1)
            assert again.samples_for(FP_A, base_cfg(), "prose", POWER) == [a_sample()]
            assert stored(again, "SELECT MAX(version) AS v FROM schema_version")[0]["v"] == SCHEMA_VERSION
            assert stored(again, "SELECT COUNT(*) AS n FROM quality")[0]["n"] == 1
        finally:
            again.close()

    def test_a_newer_schema_is_refused(self, tmp_path: Path):
        db = tmp_path / "tuner.db"
        History(db).close()
        conn = sqlite3.connect(str(db))
        try:
            conn.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION + 1,))
            conn.commit()
        finally:
            conn.close()
        with pytest.raises(RuntimeError, match="newer Model Tuner"):
            History(db)


class TestConcurrency:
    def test_two_threads_write(self, hist: History):
        h = hist
        per_thread = 25
        gate = threading.Barrier(2)
        errors: list[BaseException] = []

        def writer(name: str) -> None:
            try:
                gate.wait(timeout=10)
                for i in range(per_thread):
                    cfg = base_cfg(ctx=8192 * (i + 1))
                    run_id = f"run-{name}"
                    h.start_run(run_id, f"{name}.gguf", FP_A, "qwen3", "quick", POWER, cfg)
                    trial_id = h.add_trial(run_id, FP_A, cfg, "fit", a_projection())
                    h.set_trial_outcome(trial_id, "measured", "", a_load())
                    h.add_samples(trial_id, [a_sample(gen=float(i))])
                    h.add_quality(trial_id, QualityResult(suite="reasoning", passed=i, total=i + 1, tokens=i,
                                                          seconds=float(i)))
                    h.add_calibration("qwen3", "", False, 1000, 1100 + i)
                    if i == 0:
                        h.set_best(FP_A, f"{name}.gguf", "qwen3", cfg, {"prose": 40.0}, POWER, run_id)
            except BaseException as e:      # noqa: BLE001 - the test asserts the list is empty
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(name,)) for name in ("a", "b")]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        assert errors == []
        assert stored(h, "SELECT COUNT(*) AS n FROM trials")[0]["n"] == 2 * per_thread
        assert stored(h, "SELECT COUNT(*) AS n FROM samples")[0]["n"] == 2 * per_thread
        assert stored(h, "SELECT COUNT(*) AS n FROM quality")[0]["n"] == 2 * per_thread
        assert stored(h, "SELECT COUNT(*) AS n FROM calibrations")[0]["n"] == 2 * per_thread
        assert stored(h, "SELECT COUNT(*) AS n FROM best")[0]["n"] == 1      # upsert, not a duplicate
        assert stored(h, "SELECT COUNT(DISTINCT id) AS n FROM trials")[0]["n"] == 2 * per_thread
        assert len(h.samples_for(FP_A, base_cfg(ctx=8192), "prose", POWER)) == 2
        assert h.calibration_error("qwen3", "", False)[1] == 10
        assert {r["run_id"] for r in h.runs()} == {"run-a", "run-b"}

    def test_reads_during_writes(self, hist: History):
        h = hist
        h.start_run("run-1", "m.gguf", FP_A, "qwen3", "quick", POWER, base_cfg())
        trial_id = h.add_trial("run-1", FP_A, base_cfg(), "fit", a_projection())
        h.add_samples(trial_id, [a_sample()])
        stop = threading.Event()
        seen: list[int] = []

        def writer() -> None:
            for i in range(60):
                h.add_samples(trial_id, [a_sample(gen=float(i))])
            stop.set()

        def reader() -> None:
            while not stop.is_set():
                seen.append(len(h.samples_for(FP_A, base_cfg(), "prose", POWER)))
            seen.append(len(h.samples_for(FP_A, base_cfg(), "prose", POWER)))

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        assert seen and all(1 <= n <= 61 for n in seen)
        assert seen[-1] == 61
