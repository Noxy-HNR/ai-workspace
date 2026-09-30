"""Proposed tests for modeltuner.real's measurement path: which prompt each repetition gets, how a slow
request is told apart from a dead server, and when the crash log is kept.

Run with: uv run --with pytest pytest -q proposed_tests
"""

from __future__ import annotations

import sys
import types

import pytest

from fakes import (FakeDash, FakeLlama, FakeMemory, FakeProfile, make_samples, no_lock, quality_with_server_errors,
                   fake_projection)

from modeltuner import dashboard, real
from modeltuner.backend import LoadFailed
from modeltuner.dashboard import DashboardError
from modeltuner.knobs import Config
from modeltuner.llama import LlamaError

CFG = Config(ctx=8192, spec="ngram-mod", ngram_n_match=12)


def _backend(monkeypatch, tmp_path, *, llama=None, dash=None, lines=None):
    monkeypatch.setattr(real, "FitParams", lambda *a, **kw: type("F", (), {"project": lambda s, c: fake_projection()})())
    monkeypatch.setattr(real.hardware, "memory", lambda: FakeMemory(avail_gb=9.0))
    monkeypatch.setattr(real.hardware, "gpu", lambda *a, **k: None)
    monkeypatch.setattr(real.safety, "LlamaLock", lambda: no_lock())     # no state/LLAMA_BUSY file
    monkeypatch.setattr(real, "llama_log_tail", lambda lines=80: "ggml: out of memory")
    lines = lines if lines is not None else []
    dash = dash or FakeDash(llama=llama or FakeLlama())
    b = real.RealBackend(FakeProfile(), dash=dash, log=lines.append, run_dir=tmp_path, use_advisor=False)
    b._loaded = CFG
    return b, dash, lines


# ----------------------------------------------------------------------------------------------- #
# Repetition i of every config must see the same prompt
# ----------------------------------------------------------------------------------------------- #

def test_a_config_measured_twice_is_measured_on_the_same_prompts(monkeypatch, tmp_path):
    # strategy.phase_confirm() reloads the winner and measures it again to confirm it against the baseline.
    # The variant counter is keyed by config and never reset, so the second round uses prompts 3,4,5 while
    # the baseline it is compared with used 0,1,2: the numbers the user is asked to approve describe two
    # different texts. (The n-gram speculator also keeps a cross-request cache, so the prompts must not
    # repeat either.)
    starts: list[int] = []

    def fake_measure(client, workload, reps, power_key, *, start=0, timeout=300):
        starts.append(start)
        return make_samples(workload, reps)

    b, dash, lines = _backend(monkeypatch, tmp_path)
    monkeypatch.setattr(real.bench, "measure", fake_measure)

    b.measure("prose", reps=3)      # first round: the trial
    b._on_loaded(CFG)               # phase_confirm reloads the config (what RealBackend.load() does on success)
    b.measure("prose", reps=3)      # ...and measures it again

    assert starts == [0, 0], f"the confirmation round measured different prompts: {starts}"


def test_an_interrupted_measurement_does_not_replay_the_same_prompts(monkeypatch, tmp_path):
    # Repetitions 1 and 2 are measured, repetition 3 times out. The count is only advanced on success, so the
    # retry starts again at prompt 0 -- the same config, the same text, straight after the n-gram speculator
    # has already seen the answer (the 5x inflation from the 2026-09-30 runs).
    starts: list[int] = []

    def flaky(client, workload, reps, power_key, *, start=0, timeout=300):
        starts.append(start)
        if len(starts) == 1:
            make_samples(workload, 2)        # two good repetitions, then the server stops answering
            raise LlamaError("chat failed: timed out")
        return make_samples(workload, reps)

    b, dash, lines = _backend(monkeypatch, tmp_path, llama=FakeLlama(ready=True))
    monkeypatch.setattr(real.bench, "measure", flaky)

    # a request that times out on a live server is a timeout for this config (see the next test)
    with pytest.raises(LoadFailed):
        b.measure("prose", reps=3)
    b.measure("prose", reps=3)      # same server, no reload

    assert starts[1] >= 2, f"the retry replayed prompts {starts[1]}.. that this server had already answered"


# ----------------------------------------------------------------------------------------------- #
# A slow request is not a dead server
# ----------------------------------------------------------------------------------------------- #

def test_a_timed_out_request_on_a_live_server_is_a_timeout_not_a_crash(monkeypatch, tmp_path):
    # A 9B on 2 CPU-bound layers answers a 400-token prose prompt slowly; the request exceeds bench's 300 s
    # timeout. /v1/models still answers, so the raw LlamaError escapes measure(), strategy.measure only
    # catches LoadFailed, and one slow prompt ends the whole run.
    def too_slow(client, workload, reps, power_key, *, start=0, timeout=300):
        raise LlamaError("POST /v1/chat/completions failed: <urlopen error timed out>")

    b, dash, lines = _backend(monkeypatch, tmp_path, llama=FakeLlama(ready=True))
    monkeypatch.setattr(real.bench, "measure", too_slow)

    with pytest.raises(LoadFailed) as ei:
        b.measure("prose", reps=3)

    assert ei.value.kind == "timeout", f"a slow request was reported as {ei.value.kind!r}"


def test_a_slow_dashboard_is_not_recorded_as_a_refusal(monkeypatch, tmp_path):
    # /api/models/load stops the old server and waits for it to exit before it starts the new one, so the reply
    # can take minutes. past 180 s urlopen gives up and the tuner reads the message's prefix, "dashboard POST
    # /api/models/load failed", decides the dashboard *refused*, and stores that in history -- where
    # "refused" is a known_failure the strategy never retries, for this model, ever.
    dash = FakeDash()
    dash.trial_load = lambda *a, **k: (_ for _ in ()).throw(
        DashboardError("dashboard POST /api/models/load failed: <urlopen error timed out>"))
    b, dash, lines = _backend(monkeypatch, tmp_path, dash=dash)
    monkeypatch.setattr(real, "llama_pid", lambda *a, **k: 1111)

    with pytest.raises(LoadFailed) as ei:
        b.load(CFG)

    assert ei.value.kind == "timeout", f"a dashboard that took too long was recorded as {ei.value.kind!r}"


# ----------------------------------------------------------------------------------------------- #
# The crash log
# ----------------------------------------------------------------------------------------------- #

def test_a_load_that_times_out_still_saves_the_log(monkeypatch, tmp_path):
    # The dashboard answers the load call, the server then dies or wedges, and wait_for_model gives up after
    # 360 s. kind is "timeout", so _save_crash_log() is skipped -- and the next load overwrites
    # llama.stderr.log, so the only record of why this config failed is gone.
    b, dash, lines = _backend(monkeypatch, tmp_path)
    monkeypatch.setattr(real, "llama_pid", lambda *a, **k: 1111)
    monkeypatch.setattr(dashboard, "wait_for_model",
                        lambda *a, **k: (_ for _ in ()).throw(DashboardError("timeout: m.gguf not serving after 360s")))

    with pytest.raises(LoadFailed):
        b.load(CFG)

    logs = sorted(p.name for p in tmp_path.glob("crash-*.log"))
    assert logs, f"nothing was kept of the failed load; run dir holds {sorted(p.name for p in tmp_path.iterdir())}"


def test_a_server_that_dies_mid_measurement_keeps_its_log(monkeypatch, tmp_path):
    # The control for the test above: the crash path does save the log.
    b, dash, lines = _backend(monkeypatch, tmp_path, llama=FakeLlama(ready=False))

    def dead(client, workload, reps, power_key, *, start=0, timeout=300):
        raise LlamaError("POST /v1/chat/completions failed: remote end closed")

    monkeypatch.setattr(real.bench, "measure", dead)

    with pytest.raises(LoadFailed) as ei:
        b.measure("prose", reps=3)

    assert ei.value.kind in ("oom", "crash")
    assert list(tmp_path.glob("crash-*.log")), "the log was not kept"


# ----------------------------------------------------------------------------------------------- #
# quality(): a dead server is not a quality score
# ----------------------------------------------------------------------------------------------- #

def test_a_server_that_died_during_a_quality_suite_is_not_scored(monkeypatch, tmp_path):
    # Every suite records a LlamaError as a failed task, so a server that crashed half way through returns
    # 0 of 8. The quality gate then reads a crash as "this config answers worse" and reverts the cache type
    # and the thinking settings; no crash log is kept either.
    module = types.ModuleType("modeltuner.quality.probe")
    module.run = lambda client, limit=None, progress=None: quality_with_server_errors(8)
    monkeypatch.setitem(sys.modules, "modeltuner.quality.probe", module)

    b, dash, lines = _backend(monkeypatch, tmp_path, llama=FakeLlama(ready=False))

    with pytest.raises(LoadFailed) as ei:
        b.quality("probe")

    assert ei.value.kind in ("oom", "crash", "spill")
    assert list(tmp_path.glob("crash-*.log")), "the log of the server that died was not kept"
