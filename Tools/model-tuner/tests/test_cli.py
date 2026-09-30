"""Tests for the non-hardware CLI commands: status, stop, runs, report, apply.

Everything that would touch the real machine is faked. MODELTUNER_STATE_DIR points at a
throwaway temp dir before modeltuner.paths is imported (and the autouse fixture re-points
each individual path at a per-test tmp_path, so the tests also pass when paths was imported
earlier), the run's report.json is a fake, and Dashboard.save_settings is monkeypatched.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import shutil
import tempfile
import tomllib
from pathlib import Path

# Before modeltuner.paths is imported: the state directory goes to a throwaway temp dir, so
# these tests never read or write the real one.
_STATE_TMP = Path(tempfile.mkdtemp(prefix="modeltuner-cli-test-"))
os.environ["MODELTUNER_STATE_DIR"] = str(_STATE_TMP)
atexit.register(shutil.rmtree, _STATE_TMP, True)

import pytest  # noqa: E402

from modeltuner import cli, paths, switchboard_sync  # noqa: E402
from modeltuner.dashboard import Dashboard  # noqa: E402

RUN_ID = "20260930-000000-sim-9b"
MODEL_FILE = "sim-9b.gguf"
PROPOSED = {
    "contextSize": 65536,
    "gpuLayers": 999,
    "threads": 8,
    "batchSize": 1024,
    "ubatchSize": 512,
    "cacheTypeK": "q8_0",
    "cacheTypeV": "q8_0",
    "flashAttn": True,
    "parallel": 1,
    "jinja": True,
    "reasoningEffort": "default",
    "extraArgs": "--spec-type draft-mtp --spec-draft-n-max 2 --load-mode none --cache-ram 0",
    "serverBinary": "C:/AI/Tools/llama-native/bin/llama-server.exe",
    "loraFile": "C:/AI/models/sim-lora.gguf",
}
CHANGES = {"contextSize": [32768, 65536], "cacheTypeK": ["f16", "q8_0"], "cacheTypeV": ["f16", "q8_0"],
           "extraArgs": ["", "--spec-type draft-mtp --spec-draft-n-max 2 --load-mode none --cache-ram 0"]}
MODELS_TOML = '''# Switchboard's model list (fake)
[[model]]
file = "sim-9b.gguf"
id = "sim"
enabled = true

[model.capability]
general = 2.0

[model.measured]
gen_tps = 60.0
gen_tps_source = "hand-tuned"
load_s = 20.0
load_s_source = "hand"

[[model]]
file = "other.gguf"
id = "other"
enabled = true
'''


@pytest.fixture(autouse=True)
def state_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Re-point every path the CLI reads or writes at this test's tmp_path."""
    monkeypatch.setattr(paths, "STATE_DIR", tmp_path)
    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(paths, "CURRENT_RUN", tmp_path / "current.json")
    monkeypatch.setattr(paths, "STOP_FILE", tmp_path / "STOP")


class Recorder:
    """Stands in for Dashboard.save_settings: records calls and replies with a canned response."""

    def __init__(self, response: dict | None = None):
        self.calls: list[tuple[str, dict]] = []
        self.response = {"success": True} if response is None else response

    def __call__(self, model_file: str, settings: dict) -> dict:
        self.calls.append((model_file, settings))
        return self.response


@pytest.fixture
def save_settings(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    rec = Recorder()
    monkeypatch.setattr(Dashboard, "save_settings", rec)
    return rec


@pytest.fixture
def report_dir(tmp_path: Path) -> Path:
    """A run directory with a fake report.json (and report.md, for the report command)."""
    d = paths.RUNS_DIR / RUN_ID
    d.mkdir(parents=True)
    data = {
        "run_id": RUN_ID,
        "model": MODEL_FILE,
        "status": "ok",
        "mode": "quick",
        "power_key": "ac|best_performance|saver-off",
        "started": "2026-09-30 00:00:00",
        "finished": "2026-09-30 01:00:00",
        "improvement": 0.25,
        "baseline": {"ctx": 32768},
        "best": {"ctx": 65536},
        "baseline_metrics": {"prose": 60.0, "code": 30.0, "prefill": 2000.0},
        "best_metrics": {"prose": 80.0, "code": 40.0, "prefill": 2200.0},
        "proposed_settings": PROPOSED,
        "changes": CHANGES,
        "decisions": ["Every layer fits on the GPU.", "Speculation: draft-mtp."],
        "notes": [],
        "quality": {},
        "trials": [{"phase": "confirm", "load_s": 12.3}],
    }
    (d / "report.json").write_text(json.dumps(data), encoding="utf-8")
    (d / "report.md").write_text(f"# Model Tuner report: {MODEL_FILE}\n", encoding="utf-8")
    return d


class TestStatus:
    def test_idle(self, capsys):
        assert cli.main(["status"]) == 0
        assert json.loads(capsys.readouterr().out) == {"status": "idle"}

    def test_the_live_run(self, capsys):
        paths.CURRENT_RUN.write_text(json.dumps({"status": "running", "model": MODEL_FILE, "phase": "spec"}),
                                    encoding="utf-8")
        assert cli.main(["status"]) == 0
        out = capsys.readouterr().out
        assert '"status": "running"' in out and '"phase": "spec"' in out


class TestStop:
    def test_writes_the_stop_file(self, capsys):
        assert cli.main(["stop"]) == 0
        assert paths.RUNS_DIR.exists()  # ensure_state_dirs ran first
        text = paths.STOP_FILE.read_text(encoding="utf-8")
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", text)
        assert "Stop requested" in capsys.readouterr().out


class TestRuns:
    def test_a_run_with_a_report_shows_status_improvement_and_mode(self, capsys, report_dir: Path):
        assert cli.main(["runs"]) == 0
        out = capsys.readouterr().out
        assert RUN_ID in out and "ok" in out and "+25.0%" in out and "quick" in out

    def test_a_run_with_only_state_json_shows_its_status(self, capsys):
        d = paths.RUNS_DIR / "20260930-010000-other"
        d.mkdir(parents=True)
        (d / "state.json").write_text(json.dumps({"status": "running"}), encoding="utf-8")
        assert cli.main(["runs"]) == 0
        out = capsys.readouterr().out
        assert "20260930-010000-other" in out and "running" in out

    def test_a_directory_with_neither_file_is_listed_with_unknown_status(self, capsys):
        (paths.RUNS_DIR / "not-a-run").mkdir(parents=True)
        assert cli.main(["runs"]) == 0
        out = capsys.readouterr().out
        assert "not-a-run" in out and "?" in out


class TestReport:
    def test_prints_the_named_run(self, capsys, report_dir: Path):
        assert cli.main(["report", RUN_ID]) == 0
        assert capsys.readouterr().out == f"# Model Tuner report: {MODEL_FILE}\n\n"  # print() adds a newline

    def test_defaults_to_the_latest_run_by_report_json_mtime(self, capsys):
        older = paths.RUNS_DIR / "20260930-000000-older"
        newer = paths.RUNS_DIR / "20260930-010000-newer"
        older.mkdir(parents=True)
        newer.mkdir(parents=True)
        (older / "report.json").write_text("{}", encoding="utf-8")
        (newer / "report.json").write_text("{}", encoding="utf-8")
        (older / "report.md").write_text("older", encoding="utf-8")
        (newer / "report.md").write_text("newer", encoding="utf-8")
        os.utime(older / "report.json", ns=(1_000_000, 1_000_000))
        os.utime(newer / "report.json", ns=(2_000_000, 2_000_000))
        assert cli.main(["report"]) == 0
        assert capsys.readouterr().out == "newer\n"

    def test_no_report_anywhere(self, capsys):
        assert cli.main(["report"]) == 1
        assert "no report found" in capsys.readouterr().out


class TestApply:
    def test_without_yes_it_asks_and_does_nothing_on_n(self, capsys, report_dir: Path, save_settings: Recorder,
                                                        monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr("builtins.input", lambda *a: "n")
        assert cli.main(["apply", RUN_ID]) == 1
        out = capsys.readouterr().out
        assert f"Proposed changes for {MODEL_FILE}:" in out
        assert "contextSize" in out
        assert "Not applied." in out
        assert save_settings.calls == []
        assert not (report_dir / "applied.json").exists()

    def test_with_yes_it_saves_the_proposed_settings_and_writes_applied_json(self, capsys, report_dir: Path,
                                                                              save_settings: Recorder):
        assert cli.main(["apply", RUN_ID, "--yes"]) == 0
        assert save_settings.calls == [(MODEL_FILE, PROPOSED)]
        applied = json.loads((report_dir / "applied.json").read_text(encoding="utf-8"))
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", applied["applied"])
        assert "Applied to models.json" in capsys.readouterr().out

    def test_a_refusal_from_the_dashboard_is_reported(self, capsys, report_dir: Path,
                                                       monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(Dashboard, "save_settings", Recorder({"success": False, "message": "unknown model"}))
        assert cli.main(["apply", RUN_ID, "--yes"]) == 1
        assert "The dashboard refused" in capsys.readouterr().out
        assert not (report_dir / "applied.json").exists()

    def test_no_changes_saves_nothing(self, capsys, save_settings: Recorder):
        d = paths.RUNS_DIR / "run-nochange"
        d.mkdir(parents=True)
        data = {"run_id": "run-nochange", "model": MODEL_FILE, "status": "ok", "mode": "quick",
                "improvement": 0.0, "proposed_settings": PROPOSED, "changes": {}}
        (d / "report.json").write_text(json.dumps(data), encoding="utf-8")
        assert cli.main(["apply", "run-nochange", "--yes"]) == 0
        assert save_settings.calls == []
        assert "Nothing to change" in capsys.readouterr().out

    def test_switchboard_edits_the_measured_table(self, capsys, report_dir: Path, save_settings: Recorder,
                                                  tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        models_toml = tmp_path / "models.toml"
        models_toml.write_text(MODELS_TOML, encoding="utf-8")
        real_apply = switchboard_sync.apply_report
        monkeypatch.setattr(switchboard_sync, "apply_report", lambda report: real_apply(report, models_toml))
        assert cli.main(["apply", RUN_ID, "--yes", "--switchboard"]) == 0
        data = tomllib.loads(models_toml.read_text(encoding="utf-8"))
        measured = data["model"][0]["measured"]
        assert measured["gen_tps"] == 80.0
        assert measured["prompt_tps"] == 2200.0
        assert measured["load_s"] == 12.3
        assert f"Model Tuner run {RUN_ID}" in measured["gen_tps_source"]
        assert data["model"][1]["file"] == "other.gguf" and "measured" not in data["model"][1]
        assert ("Switchboard models.toml measured speeds updated: "
                "gen_tps=80.0, prompt_tps=2200.0, load_s=12.3") in capsys.readouterr().out
