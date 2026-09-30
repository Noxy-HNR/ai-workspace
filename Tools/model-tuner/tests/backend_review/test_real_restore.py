"""Proposed tests for the restore paths in modeltuner.real: what the user finds when the run ends.

Run with: uv run --with pytest pytest -q proposed_tests
"""

from __future__ import annotations

import json

import pytest

from fakes import FakeDash, FakeLlama, FakeMemory, FakeProfile, FakeClock, fake_projection
from modeltuner import dashboard, real


def _backend(monkeypatch, tmp_path, *, llama=None, dash=None, lines=None, config=None, config_error=None,
             pids=None, alive=True, clock=None):
    """A RealBackend with every outward call faked: no fit-params, no GPU query, no HTTP, no process, and a
    clock that only moves when the code sleeps, so no test waits in real time."""
    monkeypatch.setattr(real, "FitParams", lambda *a, **kw: type("F", (), {"project": lambda s, c: fake_projection()})())
    monkeypatch.setattr(real.hardware, "memory", lambda: FakeMemory(avail_gb=9.0))
    monkeypatch.setattr(real.hardware, "gpu", lambda *a, **k: None)
    lines = lines if lines is not None else []
    dash = dash or FakeDash(llama=llama or FakeLlama(), config=config, config_error=config_error)
    pids = pids if pids is not None else [1111]
    monkeypatch.setattr(real, "llama_pid", lambda *a, **k: pids.pop(0) if pids else None)
    monkeypatch.setattr(real, "process_alive", lambda pid: alive)
    monkeypatch.setattr(real, "llama_log_tail", lambda lines=80: "log")
    monkeypatch.setattr(dashboard, "llama_log_tail", lambda lines=60: "log")
    clock = clock or FakeClock()
    monkeypatch.setattr(real, "time", clock)
    monkeypatch.setattr(dashboard, "time", clock)
    monkeypatch.setattr(dashboard, "llama_pid", lambda *a, **k: 2222)
    b = real.RealBackend(FakeProfile(), dash=dash, log=lines.append, run_dir=tmp_path, use_advisor=False)
    return b, dash, lines


# ----------------------------------------------------------------------------------------------- #
# begin(): a slow /props must not look like "the user had nothing loaded"
# ----------------------------------------------------------------------------------------------- #

def test_a_props_hiccup_does_not_leave_the_users_model_unloaded(monkeypatch, tmp_path):
    # The user is chatting through llama-server, so /props is slow and the 10 s timeout fires.
    # loaded_model() swallows the error and returns None, so the run believes nothing was loaded and, at the
    # end, stops the server: the user's model is never reloaded and never comes back.
    from modeltuner.llama import LlamaError

    llama = FakeLlama(props_error=LlamaError("GET /props failed: timed out"))
    b, dash, lines = _backend(monkeypatch, tmp_path, llama=llama, pids=[1111, None])

    b.begin()                      # stopping the server here is the run's job
    before = len(dash.calls)
    b.restore()
    restore_calls = [c[0] for c in dash.calls[before:]]

    assert "stop_llama" not in restore_calls, f"the restore stopped the user's server: {restore_calls}"
    assert "load_saved" in restore_calls or "trial_load" in restore_calls, \
        f"the user's model was not brought back: {restore_calls}"


def test_begin_uses_the_listening_port_when_props_fails(monkeypatch, tmp_path):
    from modeltuner.llama import LlamaError

    llama = FakeLlama(props_error=LlamaError("GET /props failed: timed out"))
    b, dash, lines = _backend(monkeypatch, tmp_path, llama=llama, pids=[1111, None])

    b.begin()

    assert b._original_running is True, "a listening port was not enough to know a server was running"


# ----------------------------------------------------------------------------------------------- #
# begin(): never carry on while the old server is still exiting
# ----------------------------------------------------------------------------------------------- #

def test_begin_refuses_to_continue_while_the_old_server_is_alive(monkeypatch, tmp_path):
    # AGENTS.md: "If a llama-server won't exit, wait. Starting another one while it's stuck is what caused
    # the crash." begin() waits 120 s and then carries on anyway, so the VRAM baseline is taken with the old
    # model still resident and the run goes on to load a second one.
    b, dash, lines = _backend(monkeypatch, tmp_path, pids=[1111, 1111, 1111])

    with pytest.raises(RuntimeError):
        b.begin()

    assert b._other_mib is None, "a VRAM baseline was taken while the old server was still up"


# ----------------------------------------------------------------------------------------------- #
# restore(): best effort, always
# ----------------------------------------------------------------------------------------------- #

def test_restore_survives_an_unreadable_models_json(monkeypatch, tmp_path):
    # models_config() reads the file the dashboard owns, straight off disk. A truncated file (the dashboard
    # rewriting it, or a sync tool) raises JSONDecodeError, which is not a DashboardError, so it escapes
    # restore() -- and restore() runs in runner's finally: the run's real error is replaced and the report,
    # the history finish and the watchdog stop never happen.
    b, dash, lines = _backend(monkeypatch, tmp_path, llama=FakeLlama(), config_error=json.JSONDecodeError("x", "", 0))
    b._original_model, b._original_running = "m.gguf", True

    b.restore()      # must not raise

    assert any("restore problem" in m for m in lines), "the failure was not reported to the run log"


def test_restore_verifies_the_context_it_actually_asked_for(monkeypatch, tmp_path):
    # A model whose models.json entry has no contextSize is restored with want_ctx = -1, which no server can
    # ever report, so every restore ends in "verify: server reports n_ctx 4096, expected -1" -- noise that
    # hides a restore that really did fail.
    config = {"models": {"m.gguf": {"gpuLayers": 99}}, "activeModel": "m.gguf"}
    b, dash, lines = _backend(monkeypatch, tmp_path, llama=FakeLlama(n_ctx=4096), config=config)
    b._original_model, b._original_running = "m.gguf", True

    b.restore()

    assert not any("restore problem" in m for m in lines), f"a working restore was reported as broken: {lines}"


def test_restore_survives_a_dashboard_that_refuses(monkeypatch, tmp_path):
    # The half that already works, and the model for the fix to finding 2: a DashboardError is caught and
    # logged, so the run still finishes and still writes its report.
    from modeltuner.dashboard import DashboardError

    b, dash, lines = _backend(monkeypatch, tmp_path,
                              dash=FakeDash(llama=FakeLlama(), stop_error=DashboardError("dashboard down")))
    b._original_model, b._original_running = None, False

    b.restore()      # must not raise: the run has to finish and write its report

    assert "stop_llama" in dash.names(), "the test did not reach the restore path it means to check"
    assert any("restore problem" in m for m in lines), f"the failure was not reported: {lines}"


# ----------------------------------------------------------------------------------------------- #
# restore(): a trial-loaded model comes back with the settings it was running with
# ----------------------------------------------------------------------------------------------- #

def test_a_trial_loaded_model_is_restored_with_its_own_settings(monkeypatch, tmp_path):
    # The user's model was running but was not the active model, so restore() re-loads it without persisting
    # anything, using the settings from models.json. This is the control for the test above: it shows the
    # settings only arrive if models.json can be read, and that a restore which cannot read it (finding 2)
    # would hand the server an empty settings dict and silently change the user's context.
    b, dash, lines = _backend(monkeypatch, tmp_path, llama=FakeLlama(n_ctx=32768),
                              config={"models": {"m.gguf": {"contextSize": 32768, "gpuLayers": 20}},
                                      "activeModel": "other.gguf"})
    b._original_model, b._original_running = "m.gguf", True
    b.run_dir = tmp_path

    b.restore()

    trial = [c for c in dash.calls if c[0] == "trial_load"]
    assert trial, "the trial model was not brought back"
    settings = trial[0][2]
    assert settings.get("contextSize") == 32768, f"restored with the wrong context: {settings}"
    assert settings.get("gpuLayers") == 20, f"restored with the wrong layer split: {settings}"
