"""Integration tests for modeltuner.real.RealBackend against fake servers.

Two ThreadingHTTPServer threads on port 0 stand in for the dashboard and
llama-server. Everything that would touch the real machine is monkeypatched:
modeltuner.real.llama_pid / process_alive / llama_log_tail (and the same names
in modeltuner.dashboard, which wait_for_model uses), hardware.gpu / memory /
power, FitParams.__init__ + project, and safety.LlamaLock.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from modeltuner import bench, dashboard, hardware, safety
from modeltuner.backend import LoadFailed
from modeltuner.dashboard import Dashboard
from modeltuner.fitparams import FitParams, Projection
from modeltuner.hardware import GpuState, MemoryState, PowerState
from modeltuner.knobs import Config
from modeltuner.llama import LlamaClient
from modeltuner.profile import ModelProfile

import modeltuner.real as real_mod

MODEL_FILE = "test-model.gguf"
CTX = 32768


class FakeState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.dash_loads: list[dict] = []
        self.dash_stops = 0
        self.pid = 1000
        self.llama_model = MODEL_FILE
        self.llama_ctx = CTX
        self.llama_alive = True
        self.fail_chat = False
        self.chat_delay = 0.0
        self.chats: list[dict] = []
        self.freeze_llama = False


def _send_json(handler: BaseHTTPRequestHandler, obj: dict, code: int = 200) -> None:
    data = json.dumps(obj).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _make_dash_handler(state: FakeState):
    class H(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw or b"{}")
            except ValueError:
                body = {}
            if self.path == "/api/models/load":
                with state.lock:
                    state.dash_loads.append(body)
                    if not state.freeze_llama:
                        state.llama_model = body.get("filename", state.llama_model)
                        settings = body.get("settings") or {}
                        if "contextSize" in settings:
                            try:
                                state.llama_ctx = int(settings["contextSize"])
                            except (TypeError, ValueError):
                                pass
                    state.pid += 1
                if body.get("persist") is False:
                    _send_json(self, {"success": True, "trial": True})
                else:
                    _send_json(self, {"success": True})
            elif self.path == "/api/stop":
                with state.lock:
                    state.dash_stops += 1
                _send_json(self, {"success": True})
            else:
                _send_json(self, {"error": "not found"}, 404)

        def do_GET(self):  # noqa: N802
            if self.path == "/api/status":
                _send_json(self, {"ok": True})
            else:
                _send_json(self, {"error": "not found"}, 404)

        def log_message(self, *a):  # noqa: ANN001
            pass

    return H


def _make_llama_handler(state: FakeState):
    class H(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path == "/props":
                with state.lock:
                    alive = state.llama_alive
                    model = state.llama_model
                    ctx = state.llama_ctx
                if not alive:
                    _send_json(self, {"error": "down"}, 500)
                    return
                _send_json(self, {
                    "model_path": f"C:/AI/models/{model}",
                    "default_generation_settings": {"n_ctx": ctx},
                })
            elif self.path == "/v1/models":
                with state.lock:
                    alive = state.llama_alive
                if not alive:
                    _send_json(self, {"data": []})
                else:
                    _send_json(self, {"data": [{"id": "test"}]})
            else:
                _send_json(self, {"error": "not found"}, 404)

        def do_POST(self):  # noqa: N802
            if self.path != "/v1/chat/completions":
                _send_json(self, {"error": "not found"}, 404)
                return
            length = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw or b"{}")
            except ValueError:
                body = {}
            with state.lock:
                state.chats.append(body)
                fail = state.fail_chat or not state.llama_alive
                delay = state.chat_delay
            if fail:
                _send_json(self, {"error": "server down"}, 500)
                return
            if delay:
                time.sleep(delay)
            _send_json(self, {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "test",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "hello world"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 120, "completion_tokens": 64, "total_tokens": 184},
                "timings": {
                    "prompt_n": 120,
                    "prompt_ms": 600.0,
                    "prompt_per_second": 200.0,
                    "predicted_n": 64,
                    "predicted_ms": 1280.0,
                    "predicted_per_second": 50.0,
                },
            })

        def log_message(self, *a):  # noqa: ANN001
            pass

    return H


@pytest.fixture()
def fakes():
    state = FakeState()
    dash_srv = ThreadingHTTPServer(("127.0.0.1", 0), _make_dash_handler(state))
    llama_srv = ThreadingHTTPServer(("127.0.0.1", 0), _make_llama_handler(state))
    dash_port = dash_srv.server_address[1]
    llama_port = llama_srv.server_address[1]
    threads = [
        threading.Thread(target=dash_srv.serve_forever, daemon=True),
        threading.Thread(target=llama_srv.serve_forever, daemon=True),
    ]
    for t in threads:
        t.start()
    yield {
        "state": state,
        "dash_url": f"http://127.0.0.1:{dash_port}",
        "llama_url": f"http://127.0.0.1:{llama_port}",
        "servers": (dash_srv, llama_srv),
    }
    dash_srv.shutdown()
    llama_srv.shutdown()
    dash_srv.server_close()
    llama_srv.server_close()


def _profile() -> ModelProfile:
    return ModelProfile(
        file=MODEL_FILE,
        path=f"C:/AI/models/{MODEL_FILE}",
        size_bytes=4 * 1024**3,
        arch="llama",
        name="test",
        n_layer=32,
        n_mtp_layers=0,
        train_ctx=131072,
        n_embd=4096,
        n_head=32,
        n_head_kv=8,
        head_dim_k=128,
        head_dim_v=128,
        full_attention_interval=None,
        sliding_window=None,
        n_expert=0,
        weights_bytes=4 * 1024**3,
        layer_bytes={},
        template_supports_tools=False,
        template_has_thinking=False,
        template_honours_effort=False,
        server_binary=None,
        current={},
    )


def _projection() -> Projection:
    return Projection(
        device_model_mib=4000,
        device_context_mib=500,
        device_compute_mib=500,
        host_model_mib=100,
        host_context_mib=50,
        host_compute_mib=10,
        device_total_mib=12227,
        fitted_ngl=None,
        fitted_ctx=None,
        seconds=0.01,
    )


def make_backend(monkeypatch: pytest.MonkeyPatch, fakes, run_dir: Path, *,
                 mem_gb: dict | None = None, log_tail: dict | None = None):
    state: FakeState = fakes["state"]
    mem_gb = mem_gb if mem_gb is not None else {"v": 8.0}
    log_tail = log_tail if log_tail is not None else {"v": ""}

    def fake_pid(port: int = 10000) -> int | None:
        with state.lock:
            return state.pid

    def fake_alive(pid: int) -> bool:
        return False

    def fake_tail(lines: int = 60) -> str:
        return log_tail["v"]

    def fake_gpu(timeout_s: float = 10.0) -> GpuState | None:
        return GpuState(
            name="Fake RTX", total_mib=12227, used_mib=3000, free_mib=9227,
            pstate="P0", util_pct=0, temp_c=40, power_w=50.0,
            power_limit_w=150.0, sm_clock_mhz=2000, mem_clock_mhz=1000, driver="1.0",
        )

    def fake_memory() -> MemoryState:
        return MemoryState(
            total_bytes=16 * 1024**3,
            avail_bytes=int(mem_gb["v"] * 1024**3),
            load_pct=50,
        )

    def fake_power() -> PowerState:
        return PowerState(on_ac=True, battery_pct=80, saver_on=False, overlay="balanced")

    def fake_init(self, model_path, server_binary=None, timeout_s: float = 90.0):  # noqa: ANN001
        self.model_path = Path(model_path)
        self.exe = Path("fake-llama-fit-params.exe")
        self.timeout_s = timeout_s
        self._cache = {}
        self.calls = 0

    def fake_project(self, cfg) -> Projection:  # noqa: ANN001
        return _projection()

    class DummyLock:
        def __init__(self, *a, **k):  # noqa: ANN001
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):  # noqa: ANN001
            return False

    # Touching the real machine: pid / process / log live in both namespaces
    # (RealBackend.begin/load use real_mod's copies, wait_for_model uses
    # dashboard's copies), so patch both.
    monkeypatch.setattr(real_mod, "llama_pid", fake_pid)
    monkeypatch.setattr(dashboard, "llama_pid", fake_pid)
    monkeypatch.setattr(real_mod, "process_alive", fake_alive)
    monkeypatch.setattr(dashboard, "process_alive", fake_alive)
    monkeypatch.setattr(real_mod, "llama_log_tail", fake_tail)
    monkeypatch.setattr(dashboard, "llama_log_tail", fake_tail)
    monkeypatch.setattr(hardware, "gpu", fake_gpu)
    monkeypatch.setattr(hardware, "memory", fake_memory)
    monkeypatch.setattr(hardware, "power", fake_power)
    monkeypatch.setattr(FitParams, "__init__", fake_init)
    monkeypatch.setattr(FitParams, "project", fake_project)
    monkeypatch.setattr(safety, "LlamaLock", DummyLock)

    dash = Dashboard(
        base_url=fakes["dash_url"],
        llama=LlamaClient(base_url=fakes["llama_url"]),
    )
    backend = real_mod.RealBackend(
        _profile(), dash=dash, log=lambda *a: None,
        run_dir=run_dir, use_advisor=False,
    )
    return backend, state


def _chat_contents(state: FakeState) -> list[str]:
    with state.lock:
        bodies = list(state.chats)
    out = []
    for b in bodies:
        msgs = b.get("messages") or []
        out.append(msgs[0].get("content", "") if msgs else "")
    return out


def _clear_chats(state: FakeState) -> None:
    with state.lock:
        state.chats.clear()
        state.dash_loads.clear()


def test_successful_trial_load_and_warmup(fakes, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    backend, state = make_backend(monkeypatch, fakes, tmp_path)
    cfg = Config(ctx=CTX)
    result = backend.load(cfg)
    assert result.n_ctx == CTX
    assert result.pid is not None
    assert result.seconds >= 0
    with state.lock:
        loads = list(state.dash_loads)
    assert loads, "dashboard saw no load"
    assert loads[-1]["filename"] == MODEL_FILE
    assert loads[-1]["persist"] is False
    assert loads[-1]["settings"]["contextSize"] == CTX
    # Warm-up ran (its timing is discarded, the request still happened).
    contents = _chat_contents(state)
    assert any("primary colours" in c for c in contents)
    # Timings parse into real samples.
    samples = backend.measure("prose", reps=2)
    assert len(samples) == 2
    assert all(s.gen_tps == pytest.approx(50.0) for s in samples)
    assert all(s.prompt_tps == pytest.approx(200.0) for s in samples)


def test_load_timeout(fakes, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    backend, state = make_backend(monkeypatch, fakes, tmp_path)
    with state.lock:
        state.freeze_llama = True
        state.llama_model = "something-else.gguf"
    orig_wait = dashboard.wait_for_model

    def short_wait(dash, model_file, want_ctx, old_pid, timeout: float = 4.0):  # noqa: ANN001
        return orig_wait(dash, model_file, want_ctx, old_pid, timeout=timeout)

    monkeypatch.setattr(dashboard, "wait_for_model", short_wait)
    with pytest.raises(LoadFailed) as exc:
        backend.load(Config(ctx=CTX))
    assert exc.value.kind == "timeout"
    assert backend._loaded is None


def test_server_dies_mid_measurement(fakes, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    log_tail = {"v": "CUDA error: device-side assert triggered\ncrash boom"}
    backend, state = make_backend(monkeypatch, fakes, tmp_path, log_tail=log_tail)
    backend.load(Config(ctx=CTX))
    with state.lock:
        state.llama_alive = False
        state.fail_chat = True
    with pytest.raises(LoadFailed) as exc:
        backend.measure("prose", reps=2)
    assert exc.value.kind in ("oom", "crash")
    crash = tmp_path / "crash-1.log"
    assert crash.exists()
    assert "crash boom" in crash.read_text(encoding="utf-8")
    assert backend._loaded is None


def test_spill_guard(fakes, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    mem_gb = {"v": 8.0}
    backend, state = make_backend(monkeypatch, fakes, tmp_path, mem_gb=mem_gb)
    backend.load(Config(ctx=CTX))
    with state.lock:
        state.chat_delay = 0.4
    mem_gb["v"] = 0.5  # VRAM spilling into system RAM: available RAM collapses
    try:
        with pytest.raises(LoadFailed) as exc:
            backend.measure("prose", reps=6)
    finally:
        with state.lock:
            state.chat_delay = 0.0
        mem_gb["v"] = 8.0
    assert exc.value.kind == "spill"
    assert backend._loaded is None
    with state.lock:
        assert state.dash_stops >= 1


def test_prompt_variant_sequence(fakes, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    backend, state = make_backend(monkeypatch, fakes, tmp_path)
    cfg_a = Config(ctx=CTX, threads=8)
    cfg_b = Config(ctx=CTX, threads=4)
    assert cfg_a.key() != cfg_b.key()

    backend.load(cfg_a)
    _clear_chats(state)
    backend.measure("prose", reps=2)
    first = _chat_contents(state)

    backend.measure("prose", reps=2)
    with state.lock:
        bodies = list(state.chats)
    second = [ (b.get("messages") or [{}])[0].get("content", "") for b in bodies[2:4] ]
    assert len(first) == 2 and len(second) == 2
    assert first != second  # same config continues the sequence

    def expected(i: int) -> str:
        msgs, _ = bench._variant("prose", i)
        return msgs[0]["content"]

    assert first == [expected(0), expected(1)]
    assert second == [expected(2), expected(3)]

    backend.load(cfg_b)  # new config restarts the sequence
    _clear_chats(state)
    backend.measure("prose", reps=2)
    third = _chat_contents(state)
    assert third == [expected(0), expected(1)]


def test_restore_uses_trial_load(fakes, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    backend, state = make_backend(monkeypatch, fakes, tmp_path)
    backend._original_model = MODEL_FILE
    backend._original_running = True
    monkeypatch.setattr(
        backend.dash, "models_config",
        lambda: {"activeModel": "other.gguf",
                 "models": {MODEL_FILE: {"contextSize": CTX}}},
    )
    with state.lock:
        state.dash_loads.clear()
    backend.restore()
    with state.lock:
        loads = list(state.dash_loads)
    assert loads, "restore performed no load"
    last = loads[-1]
    assert last["filename"] == MODEL_FILE
    assert last["persist"] is False  # trial load, so models.json keeps its active model
    assert last["settings"]["contextSize"] == CTX
