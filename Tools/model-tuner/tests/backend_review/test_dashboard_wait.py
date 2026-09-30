"""Proposed tests for modeltuner.dashboard: wait_for_model and the HTTP deadline.

Run with: uv run --with pytest pytest -q proposed_tests
"""

from __future__ import annotations

import socket
import threading
import time

import pytest

from fakes import FakeClock, FakeLlama
from modeltuner import dashboard
from modeltuner.dashboard import Dashboard, DashboardError, wait_for_model

MODEL = "m.gguf"


def _dash(n_ctx: int = 8192, model: str = MODEL, ready: bool = True) -> FakeLlama:
    return FakeLlama(model=model, n_ctx=n_ctx, ready=ready)


class _Holder:
    """wait_for_model only needs an object with a .llama client."""

    def __init__(self, llama: FakeLlama):
        self.llama = llama


# ----------------------------------------------------------------------------------------------- #
# A slow load is not a crash: the log still belongs to the previous server
# ----------------------------------------------------------------------------------------------- #

def test_a_slow_load_is_not_called_a_crash_because_of_the_previous_servers_log(monkeypatch):
    # The dashboard only truncates llama.stderr.log when the NEW server starts. While a big model is still
    # loading, the log on disk is the previous run's, and its last line can easily contain "error".
    clock = FakeClock()
    monkeypatch.setattr(dashboard, "time", clock)
    monkeypatch.setattr(dashboard, "llama_pid", lambda *a, **k: None)
    monkeypatch.setattr(dashboard, "process_alive", lambda pid: True)
    monkeypatch.setattr(dashboard, "llama_log_tail",
                        lambda lines=60: "llama_model_loader: loaded meta data\nggml_cuda: error: cudaMalloc failed\n")

    with pytest.raises(DashboardError) as ei:
        wait_for_model(_Holder(_dash()), MODEL, 8192, None, timeout=60)

    # Nothing is listening and no OOM marker from the *new* server exists, so the honest answer is "not up yet",
    # i.e. a timeout, not a verdict about the config.
    assert str(ei.value).startswith("timeout"), f"reported a stale-log failure instead of waiting: {ei.value}"


def test_a_stale_error_line_does_not_fail_a_load_that_is_still_starting(monkeypatch):
    # Same window, milder marker: llama-server's shutdown line contains "error" on many builds.
    clock = FakeClock()
    monkeypatch.setattr(dashboard, "time", clock)
    monkeypatch.setattr(dashboard, "llama_pid", lambda *a, **k: None)
    monkeypatch.setattr(dashboard, "process_alive", lambda pid: True)
    # llama_log_tail() joins splitlines() with "\n" and adds no trailing newline, so the last line is the
    # server's real last line.
    monkeypatch.setattr(dashboard, "llama_log_tail",
                        lambda lines=60: "error: llama-server is shutting down")

    with pytest.raises(DashboardError) as ei:
        wait_for_model(_Holder(_dash()), MODEL, 8192, None, timeout=30)

    assert str(ei.value).startswith("timeout"), f"a shutting-down log line failed a live load: {ei.value}"


# ----------------------------------------------------------------------------------------------- #
# A dead server is not a slow one
# ----------------------------------------------------------------------------------------------- #

def test_a_server_that_exited_is_reported_from_its_log_not_as_a_timeout(monkeypatch):
    # tasklist can fail (a locked-down session, a busy box). process_alive() then answers True for everything,
    # so the "exited while loading" branch is never taken and a server that died 10 s in is reported as a
    # 120 s timeout -- and the log that explains why is thrown away instead of classified.
    clock = FakeClock()
    monkeypatch.setattr(dashboard, "time", clock)
    pids = [4242, 4242, None, None]
    monkeypatch.setattr(dashboard, "llama_pid", lambda *a, **k: pids.pop(0) if pids else None)
    monkeypatch.setattr(dashboard, "process_alive", lambda pid: True)
    monkeypatch.setattr(dashboard, "llama_log_tail",
                        lambda lines=60: "ggml_backend_cuda: failed to allocate 8000 MiB for the KV cache\n")

    # a listener that is not serving our model: verification must not match, the loop must keep going
    other = _Holder(FakeLlama(model="other.gguf", n_ctx=8192))
    with pytest.raises(DashboardError) as ei:
        wait_for_model(other, MODEL, 8192, None, timeout=120)

    assert not str(ei.value).startswith("timeout"), \
        f"a server that exited was reported as slow: {ei.value}"


def test_process_alive_says_yes_when_tasklist_fails(monkeypatch):
    # The conservative answer is right for a *waiting* loop, but it must not be the only signal: this is the
    # reason the previous test cannot pass. Documented here so the trade-off is visible.
    def boom(*a, **k):
        raise OSError("access denied")

    monkeypatch.setattr(dashboard.subprocess, "run", boom)
    assert dashboard.process_alive(4242) is True


# ----------------------------------------------------------------------------------------------- #
# n_ctx verification
# ----------------------------------------------------------------------------------------------- #

def test_a_context_that_is_not_applied_yet_is_not_a_permanent_verification_failure(monkeypatch):
    # llama-server answers /props as soon as it is up; the requested -c can still be the previous server's for
    # a poll or two. Raising "verify" on the first observation marks the config failed in history for good
    # (strategy.ensure_loaded never retries a "verify" outcome).
    clock = FakeClock()
    monkeypatch.setattr(dashboard, "time", clock)
    pids = [777, 777, 777]
    monkeypatch.setattr(dashboard, "llama_pid", lambda *a, **k: pids.pop(0) if pids else 777)
    props = [{"model_path": "C:/AI/models/m.gguf", "default_generation_settings": {"n_ctx": 16384}},
             {"model_path": "C:/AI/models/m.gguf", "default_generation_settings": {"n_ctx": 16384}},
             {"model_path": "C:/AI/models/m.gguf", "default_generation_settings": {"n_ctx": 8192}}]
    llama = FakeLlama(props=props)

    pid, secs = wait_for_model(_Holder(llama), MODEL, 8192, None, timeout=60)

    assert pid == 777


# ----------------------------------------------------------------------------------------------- #
# The HTTP timeout is a socket timeout, not a deadline
# ----------------------------------------------------------------------------------------------- #

def test_dashboard_calls_have_a_total_deadline():
    # urllib's timeout applies to each recv, so a response that dribbles a byte every 300 ms keeps the tuner
    # blocked for as long as the peer wants. AGENTS.md requires a hard timeout on every call.
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]

    def serve():
        try:
            conn, _ = srv.accept()
            conn.recv(4096)
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 20\r\n\r\n{")
            for _ in range(19):
                time.sleep(0.3)
                conn.sendall(b" ")
            conn.close()
        except OSError:
            pass
        finally:
            srv.close()

    threading.Thread(target=serve, daemon=True).start()

    dash = Dashboard(f"http://127.0.0.1:{port}")
    t0 = time.perf_counter()
    with pytest.raises(DashboardError):
        dash._call("GET", "/api/status", timeout=1)
    elapsed = time.perf_counter() - t0

    assert elapsed < 3.0, f"a 1 s timeout took {elapsed:.1f} s: the deadline is per socket read, not per call"
