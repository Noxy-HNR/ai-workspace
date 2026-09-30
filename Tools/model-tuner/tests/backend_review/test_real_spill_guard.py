"""Proposed tests for the spill guard thread in modeltuner.real: the race between it and the main thread.

Run with: uv run --with pytest pytest -q proposed_tests
"""

from __future__ import annotations

import threading
import time

from fakes import FakeDash, FakeMemory, fake_projection

from modeltuner import real


def _patched(monkeypatch):
    monkeypatch.setattr(real.hardware, "memory", lambda: FakeMemory(avail_gb=9.0))
    monkeypatch.setattr(real, "FitParams", lambda *a, **kw: type("F", (), {"project": lambda s, c: fake_projection()}))


def test_stop_disarms_the_guard(monkeypatch):
    # measure() finishes, calls guard.stop() and moves on to the next config. The guard is still inside its RAM
    # read, so it never sees the stop event, decides to fire, and stops llama-server -- which by then is the
    # *next* config's server, or (after the run) the model the user had. The samples it just returned are
    # accepted as a success and the server is killed underneath the tuner.
    dash = FakeDash()
    entered = threading.Event()
    release = threading.Event()

    def slow_memory():
        entered.set()
        release.wait(10)
        return FakeMemory(avail_gb=0.5)

    _patched(monkeypatch)
    monkeypatch.setattr(real.hardware, "memory", slow_memory)

    g = real._SpillGuard(dash, floor_gb=1.4, seconds=0.0).start()
    assert entered.wait(5), "the guard never read the RAM"
    g.stop()                       # the measurement is over
    release.set()
    g._t.join(5)

    assert not g.fired, "a disarmed guard fired anyway"
    assert "stop_llama" not in dash.names(), f"llama-server was stopped after the measurement ended: {dash.names()}"


def test_stop_waits_for_a_stop_that_is_already_under_way(monkeypatch):
    # The guard fired just before the measurement returned and its stop_llama() HTTP call is still in flight
    # (the dashboard waits for the server to exit, so this takes seconds). stop() joins for 2 s and gives up,
    # so the call lands later -- on whichever server happens to be up by then.
    dash = FakeDash()
    in_call = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    real_stop = dash.stop_llama

    def slow_stop():
        in_call.set()
        release.wait(10)
        real_stop()
        finished.set()

    _patched(monkeypatch)
    monkeypatch.setattr(real.hardware, "memory", lambda: FakeMemory(avail_gb=0.5))
    dash.stop_llama = slow_stop

    g = real._SpillGuard(dash, floor_gb=1.4, seconds=0.0).start()
    assert in_call.wait(5), "the guard never fired"

    t0 = time.perf_counter()
    g.stop()
    waited = time.perf_counter() - t0
    returned_mid_stop = not finished.is_set()
    release.set()
    g._t.join(5)

    assert not returned_mid_stop, \
        f"guard.stop() came back after {waited:.1f} s with its stop_llama() still in flight"
