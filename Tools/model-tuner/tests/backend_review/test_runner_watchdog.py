"""Proposed tests for modeltuner.runner: what still has to happen when something goes wrong.

Run with: uv run --with pytest pytest -q proposed_tests
"""

from __future__ import annotations

import contextlib
import json
import threading
import time

import pytest

from fakes import FakeDash, FakeLlama, FakeMemory, FakeProfile

from modeltuner import paths, runner, safety
from modeltuner.knobs import Config
from modeltuner.strategy import TuneOptions, TuneResult


def _cpu():
    from modeltuner.hardware import CpuTopology
    return CpuTopology("fake", 20, 20, 8, 12)


def _power():
    from modeltuner.hardware import PowerState
    return PowerState(on_ac=True, battery_pct=80, saver_on=False, overlay="balanced")


def _result() -> TuneResult:
    return TuneResult(status="ok", baseline=Config(ctx=8192), best=Config(ctx=16384), baseline_metrics={},
                      best_metrics={}, improvement=0.1, decisions=[], notes=[], trials=[], loads=1, seconds=1.0)


@pytest.fixture
def env(monkeypatch, tmp_path):
    """runner.run() with the model, the machine, the strategy, history and the dashboard all faked."""
    state: dict = {"restored": 0, "reports": 0, "exits": [], "run_state": {}, "events": []}

    monkeypatch.setattr(paths, "STATE_DIR", tmp_path)
    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(paths, "CURRENT_RUN", tmp_path / "current.json")
    monkeypatch.setattr(paths, "STOP_FILE", tmp_path / "STOP")
    monkeypatch.setattr(paths, "TRIP_FLAG", tmp_path / "WATCHDOG_TRIPPED")
    monkeypatch.setattr(paths, "DB_PATH", tmp_path / "tuner.db")
    monkeypatch.setattr(runner.hardware, "memory", lambda: FakeMemory(avail_gb=9.0))
    monkeypatch.setattr(runner.hardware, "cpu", lambda: _cpu())
    monkeypatch.setattr(runner.hardware, "power", lambda: _power())
    monkeypatch.setattr(runner, "build_profile", lambda f: FakeProfile())
    monkeypatch.setattr(runner, "_open_history", lambda *a, **k: (None, None))
    monkeypatch.setattr(runner.Tuner, "__init__", lambda self, *a, **k: None, raising=False)
    monkeypatch.setattr(runner.Tuner, "run", lambda self, topo=None: _result())

    def write(*a, **k):
        state["reports"] += 1
        return {"improvement": 0.1}

    monkeypatch.setattr(runner.report, "write", write)

    original_save = runner.RunState.save

    def save(self, **kw):
        original_save(self, **kw)
        state["run_state"] = dict(self.state)

    monkeypatch.setattr(runner.RunState, "save", save)

    class FakeBackend:
        def __init__(self, profile, **kw):
            self.dash = FakeDash(llama=FakeLlama())

        def begin(self):
            return None

        def restore(self):
            state["restored"] += 1

    monkeypatch.setattr(runner, "RealBackend", FakeBackend)
    return state


def _events(tmp_path) -> list[dict]:
    files = list((tmp_path / "runs").glob("*/events.jsonl"))
    if not files:
        return []
    return [json.loads(line) for line in files[0].read_text(encoding="utf-8").splitlines() if line.strip()]


# ----------------------------------------------------------------------------------------------- #
# A restore that fails must not eat the report
# ----------------------------------------------------------------------------------------------- #

def test_a_failing_restore_still_writes_the_report(env, monkeypatch):
    # restore() runs in runner's finally, before the report is written. A restore that raises -- and
    # real.restore() reads models.json off disk, which can raise anything -- takes the report, the history
    # finish and dog.stop() with it, and replaces the run's real error with an unrelated one.
    class Exploding(runner.RealBackend):
        def restore(self):
            env["restored"] += 1
            raise OSError("models.json is being written by the dashboard")

    monkeypatch.setattr(runner, "RealBackend", Exploding)

    with contextlib.suppress(OSError):
        runner.run("m.gguf", TuneOptions())

    assert env["reports"] == 1, "the report was lost because the restore failed"
    assert env["restored"] == 1, "restore was attempted"


# ----------------------------------------------------------------------------------------------- #
# The RAM watchdog
# ----------------------------------------------------------------------------------------------- #

def test_a_watchdog_trip_is_not_invisible_to_the_run(env, monkeypatch, tmp_path):
    # Watchdog._trip() communicates by killing the process from its own thread (os._exit(3)). The main thread
    # is never told: with the real os._exit the finally never runs, so the user's model is never reloaded, the
    # report is never written and the history run stays "running" for ever. Here the exit is made survivable
    # so the other half is visible: the run simply carries on and reports success after the watchdog stopped
    # llama-server underneath it, with no record of the trip anywhere but the trip flag.
    trip = threading.Event()

    def fake_exit(code):
        env["exits"].append(code)
        trip.set()

    monkeypatch.setattr(safety.os, "_exit", fake_exit)
    monkeypatch.setattr(runner.Tuner, "run", lambda self, topo=None: (trip.wait(10), time.sleep(0.2), _result())[2])
    monkeypatch.setattr(runner.hardware, "memory", lambda: FakeMemory(avail_gb=0.4))

    class Impatient(safety.Watchdog):
        def __init__(self, on_trip, **kw):
            super().__init__(on_trip, min_gb=0.8, seconds=0, interval=0.05)

    monkeypatch.setattr(safety, "Watchdog", Impatient)

    runner.run("m.gguf", TuneOptions())

    assert env["exits"] == [3], "the watchdog did not terminate the process"
    assert env["run_state"].get("status") != "ok", \
        f"the run reported {env['run_state'].get('status')!r} after the watchdog stopped llama-server"
    assert any(e["kind"] == "watchdog" for e in _events(tmp_path)), \
        "the trip left no trace in the run's own event log"


def test_the_trip_flag_is_written_before_the_process_dies(tmp_path, monkeypatch):
    # The one thing the watchdog does get right, and the only trace a killed run leaves: the flag, so the
    # next run refuses to start (runner.run -> safety.refuse_if_tripped).
    monkeypatch.setattr(paths, "STATE_DIR", tmp_path)
    monkeypatch.setattr(paths, "TRIP_FLAG", tmp_path / "WATCHDOG_TRIPPED")
    dog = safety.Watchdog(lambda avail: None, exit_process=False)

    dog._trip(0.4)

    assert (tmp_path / "WATCHDOG_TRIPPED").exists()
    with pytest.raises(safety.SafetyError):
        safety.refuse_if_tripped()
