"""Extra RealBackend tests from the bot queue (job 03, Space Bunny, 2026-09-30), updated for the review fixes.

Integration tests for `modeltuner.real.RealBackend`.

RealBackend is the only module that touches the machine: it asks llama-fit-params for a projection, loads
through the dashboard, warms the model up, measures it, watches RAM for a VRAM spill, copies crash logs and puts
the user's model back. These tests run the real code against two fake `http.server` threads (a dashboard and a
llama-server) plus fake hardware probes, so every branch is exercised without loading a model, stopping a
server or starting a process. Nothing here reaches the network beyond 127.0.0.1.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from modeltuner import bench, dashboard as dashboard_mod, fitparams, hardware, real, safety
from modeltuner.backend import LoadFailed
from modeltuner.dashboard import Dashboard
from modeltuner.fitparams import Projection
from modeltuner.knobs import Config
from modeltuner.llama import LlamaClient
from modeltuner.profile import ModelProfile

MODEL = "tuner-test-9b.gguf"          # the model under test
ORIGINAL = "some-other-model.gguf"    # what the user had running before the tuner started
ORIGINAL_SETTINGS = {
    "contextSize": 8192,
    "gpuLayers": 33,
    "threads": 6,
    "batchSize": 2048,
    "ubatchSize": 512,
    "cacheTypeK": "q8_0",
    "cacheTypeV": "q8_0",
    "flashAttn": True,
    "serverBinary": "llama-server-prism.exe",
    "extraArgs": "--jinja",
}


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #


class VirtualClock:
    """Stands in for the `time` module inside real.py and dashboard.py.

    `wait_for_model` polls once a second and gives up after six minutes; on a virtual clock that whole loop
    costs nothing, and `begin()`'s "wait for the old server to exit" sleep never actually sleeps. A virtual
    clock also never advances on its own, so the spill guard's 2 s of continuous low RAM has to be driven by
    the test (it restores the real `time` module when it wants the guard to fire).
    """

    def __init__(self) -> None:
        self.now = 0.0

    def time(self) -> float:
        return self.now

    def monotonic(self) -> float:
        return self.now

    def perf_counter(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


@dataclass
class FakeMachine:
    """What the two fake servers share: what is loaded, whether it is serving, and what it was asked for."""

    saved: dict[str, Any] = field(default_factory=dict)      # models.json, as the dashboard would read it
    model: str | None = None                                 # the file /props reports
    n_ctx: int = 0
    serving: bool = False
    alive: bool = True
    pid: int | None = None
    log_tail: str = "llama_model_loader: loaded meta data"    # what llama.stderr.log would hold
    hold_s: float = 0.0        # chat requests stay in flight this long (so a spill guard can catch them)
    swallow_loads: bool = False    # accept loads but never serve: the server never comes up
    ctx_override: int | None = None   # report a context the tuner did not ask for
    _next_pid: int = 4200

    loads: list[dict] = field(default_factory=list)   # every /api/models/load body, in order
    stops: int = 0
    chats: list[dict] = field(default_factory=list)   # every /v1/chat/completions body

    def start(self, model: str, n_ctx: int) -> None:
        if self.swallow_loads:
            return
        self._next_pid += 1
        self.model = model
        self.n_ctx = self.ctx_override if self.ctx_override is not None else n_ctx
        self.pid = self._next_pid
        self.serving = True
        self.alive = True

    def stop(self) -> None:
        self.stops += 1
        self.serving = False
        self.alive = False
        self.pid = None

    def kill(self) -> None:
        """The server died on its own: nothing asked the dashboard to stop it."""
        self.serving = False
        self.alive = False
        self.pid = None

    def idle_gpu_mib(self) -> int:
        return 9000 if self.serving else 1500

    def reset(self) -> "FakeMachine":
        """Back to 'the user had another model running'.

        The two servers live for the whole session and this puts their shared state back between tests: binding
        and closing a listening socket per test is slow and, on Windows, a fresh ephemeral port can land on one
        still in TIME_WAIT from the last one.
        """
        self.saved = {"activeModel": "yet-another-model.gguf",
                      "models": {ORIGINAL: dict(ORIGINAL_SETTINGS), MODEL: dict(ORIGINAL_SETTINGS)}}
        self.model = None
        self.n_ctx = 0
        self.pid = None
        self.log_tail = "llama_model_loader: loaded meta data"
        self.hold_s = 0.0
        self.swallow_loads = False
        self.ctx_override = None
        self._next_pid = 4200
        self.loads = []
        self.stops = 0
        self.chats = []
        self.alive = True
        self.start(ORIGINAL, int(ORIGINAL_SETTINGS["contextSize"]))
        return self


class _JSON(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:      # keep the test output clean
        pass

    def _reply(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        """Every POST handler calls this first, even when it does not want the body.

        Answering without draining the request leaves unread bytes on the socket, and Windows resets the
        connection rather than letting it close: the client then sees WinError 10053 on a request the server
        had already handled. Reproducible with a 500-request loop against a handler that skips this.
        """
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        return json.loads(raw) if raw else {}


def _dashboard_handler(machine: FakeMachine) -> type[_JSON]:
    class Handler(_JSON):
        def do_POST(self) -> None:
            if self.path == "/api/models/load":
                body = self._body()
                machine.loads.append(body)
                filename = body.get("filename") or ""
                if body.get("persist") is False:
                    machine.start(filename, int((body.get("settings") or {}).get("contextSize") or 0))
                    self._reply(200, {"success": True, "trial": True})
                else:
                    saved = (machine.saved.get("models") or {}).get(filename) or {}
                    machine.start(filename, int(saved.get("contextSize") or 0))
                    self._reply(200, {"success": True})
                return
            if self.path == "/api/stop":
                self._body()
                machine.stop()
                self._reply(200, {"success": True})
                return
            self._reply(404, {"error": "not found"})

        def do_GET(self) -> None:
            if self.path == "/api/status":
                self._reply(200, {"success": True, "llama": {"running": machine.serving}})
                return
            self._reply(404, {"error": "not found"})

    return Handler


def _llama_handler(machine: FakeMachine) -> type[_JSON]:
    class Handler(_JSON):
        def do_GET(self) -> None:
            if self.path == "/props":
                if not machine.serving:
                    self._reply(503, {"error": "loading"})
                    return
                self._reply(200, {"model_path": f"C:/fake/models/{machine.model}",
                                  "default_generation_settings": {"n_ctx": machine.n_ctx}})
                return
            if self.path == "/v1/models":
                self._reply(200, {"object": "list", "data": [{"id": machine.model}] if machine.serving else []})
                return
            self._reply(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/v1/chat/completions":
                self._reply(404, {"error": "not found"})
                return
            body = self._body()
            machine.chats.append(body)
            deadline = time.monotonic() + machine.hold_s
            while machine.alive and time.monotonic() < deadline:
                time.sleep(0.05)
            if not machine.alive:
                self._reply(503, {"error": "server is not running"})
                return
            prompt_n = max(1, len(json.dumps(body.get("messages") or [])) // 4)
            predicted_n = int(body.get("max_tokens") or 128)
            self._reply(200, {
                "id": "chatcmpl-fake",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "A measured answer."}}],
                "usage": {"prompt_tokens": prompt_n, "completion_tokens": predicted_n},
                "timings": {
                    "prompt_n": prompt_n, "prompt_ms": prompt_n / 1800 * 1000, "prompt_per_second": 1800.0,
                    "predicted_n": predicted_n, "predicted_ms": predicted_n / 45.0 * 1000,
                    "predicted_per_second": 45.0,
                },
            })

    return Handler


def _serve(handler: type[_JSON]) -> tuple[str, ThreadingHTTPServer, threading.Thread]:
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)     # port 0: never collides with the real servers
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05},
                              name="fake-http", daemon=True)
    thread.start()
    return f"http://127.0.0.1:{httpd.server_address[1]}", httpd, thread


def _profile() -> ModelProfile:
    return ModelProfile(
        file=MODEL,
        path=f"C:/fake/models/{MODEL}",
        size_bytes=5_100_000_000,
        arch="qwen35",
        name="Tuner Test 9B",
        n_layer=40,
        n_mtp_layers=1,
        train_ctx=262144,
        n_embd=4096,
        n_head=32,
        n_head_kv=8,
        head_dim_k=128,
        head_dim_v=128,
        full_attention_interval=4,
        sliding_window=None,
        n_expert=0,
        weights_bytes=5_000_000_000,
        layer_bytes={},
        template_supports_tools=True,
        template_has_thinking=True,
        template_honours_effort=True,
        server_binary=None,
        current=dict(ORIGINAL_SETTINGS),
    )


def _gpu(used_mib: int) -> hardware.GpuState:
    return hardware.GpuState(name="Fake RTX 5070 Ti Laptop GPU", total_mib=12227, used_mib=used_mib,
                             free_mib=12227 - used_mib, pstate="P0", util_pct=35, temp_c=61, power_w=82.0,
                             power_limit_w=150.0, sm_clock_mhz=2410, mem_clock_mhz=14001, driver="fake")


def _fake_fit_init(self: Any, model_path: str | Path, server_binary: str | None = None,
                   timeout_s: float = 90.0) -> None:
    """Stand in for the part of FitParams.__init__ that looks for llama-fit-params.exe on this PC."""
    self.model_path = Path(model_path)
    self.calls = 0


def _fake_project(self: Any, cfg: Config) -> Projection:
    """A deterministic stand-in for llama-fit-params: memory grows with the context, and anything not on the
    GPU leaves weights on the host."""
    self.calls += 1
    return Projection(
        device_model_mib=6000,
        device_context_mib=cfg.ctx // 1024,
        device_compute_mib=600,
        host_model_mib=0 if cfg.ngl >= 999 else 4000,
        host_context_mib=0,
        host_compute_mib=128,
        device_total_mib=12227,
        fitted_ngl=None,
        fitted_ctx=cfg.ctx,
        seconds=2.0,
    )


@dataclass
class Rig:
    machine: FakeMachine
    backend: real.RealBackend
    ram: dict[str, float]        # tests move available RAM to trigger preflight failures and the spill guard
    run_dir: Path
    lines: list[str]             # everything the backend logged


@dataclass
class Fakes:
    """The two servers, shared by every test in the session."""

    machine: FakeMachine
    dash_url: str
    llama_url: str


@pytest.fixture(scope="session")
def fakes() -> Fakes:
    machine = FakeMachine().reset()
    dash_url, dash_httpd, dash_thread = _serve(_dashboard_handler(machine))
    llama_url, llama_httpd, llama_thread = _serve(_llama_handler(machine))
    try:
        yield Fakes(machine=machine, dash_url=dash_url, llama_url=llama_url)
    finally:
        for httpd, thread in ((dash_httpd, dash_thread), (llama_httpd, llama_thread)):
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=5)


@pytest.fixture
def rig(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fakes: Fakes):
    machine = fakes.machine.reset()
    clock = VirtualClock()
    ram = {"avail_gb": 9.5}
    lines: list[str] = []
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    # Every probe that would read this PC. real.py and dashboard.py both hold their own name for these, so each
    # has to be replaced in both places.
    monkeypatch.setattr(real, "time", clock)
    monkeypatch.setattr(dashboard_mod, "time", clock)
    monkeypatch.setattr(real, "llama_pid", lambda *a, **k: machine.pid)
    monkeypatch.setattr(dashboard_mod, "llama_pid", lambda *a, **k: machine.pid)
    monkeypatch.setattr(real, "process_alive", lambda *a, **k: False)
    monkeypatch.setattr(dashboard_mod, "process_alive", lambda *a, **k: False)
    monkeypatch.setattr(real, "llama_log_tail", lambda *a, **k: machine.log_tail)
    monkeypatch.setattr(dashboard_mod, "llama_log_tail", lambda *a, **k: machine.log_tail)
    monkeypatch.setattr(hardware, "gpu", lambda *a, **k: _gpu(machine.idle_gpu_mib()))
    monkeypatch.setattr(hardware, "memory", lambda *a, **k: hardware.MemoryState(
        total_bytes=15_400_000_000, avail_bytes=int(ram["avail_gb"] * 1024**3), load_pct=40))
    monkeypatch.setattr(hardware, "power", lambda *a, **k: hardware.PowerState(
        on_ac=True, battery_pct=None, saver_on=False, overlay="best_performance"))
    monkeypatch.setattr(Dashboard, "models_config", lambda self: machine.saved)

    # The real fit oracle shells out to llama-fit-params.exe, which is not here; only the arithmetic matters.
    monkeypatch.setattr(fitparams.FitParams, "__init__", _fake_fit_init)
    monkeypatch.setattr(fitparams.FitParams, "project", _fake_project)

    # measure() takes the busy lock, which points at the real project's state/ directory: keep it out of the way
    # of a tuner that may be running on this machine.
    lock_class = safety.LlamaLock
    monkeypatch.setattr(safety, "LlamaLock", lambda: lock_class(tmp_path / "LLAMA_BUSY"))

    backend = real.RealBackend(_profile(),
                               dash=Dashboard(base_url=fakes.dash_url, llama=LlamaClient(base_url=fakes.llama_url)),
                               log=lines.append, run_dir=run_dir, use_advisor=False)
    return Rig(machine=machine, backend=backend, ram=ram, run_dir=run_dir, lines=lines)


def prompts(machine: FakeMachine) -> list[str]:
    return [c["messages"][0]["content"] for c in machine.chats]


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #


class TestLoad:
    def test_successful_trial_load_and_warm_up(self, rig: Rig):
        assert rig.backend.model_file == MODEL
        rig.backend.begin()
        assert rig.backend._other_mib == 1500            # VRAM of everything else, with the server stopped
        assert rig.backend._original_model == ORIGINAL   # remembered so it can be put back

        result = rig.backend.load(Config(ctx=32768, ngl=999, threads=8, batch=1024, ubatch=256))

        # One trial load, nothing persisted, and the enforced RAM-safety flags rode along.
        assert len(rig.machine.loads) == 1
        body = rig.machine.loads[0]
        assert body["filename"] == MODEL
        assert body["persist"] is False
        assert body["settings"]["contextSize"] == 32768
        assert body["settings"]["gpuLayers"] == 999
        assert body["settings"]["extraArgs"] == "--load-mode none --cache-ram 0"

        # The warm-up ran before anything else, asked for no thinking and no prompt cache, and its timing was
        # thrown away (nothing returned yet).
        assert len(rig.machine.chats) == 1
        warm = rig.machine.chats[0]
        assert warm["messages"] == bench.WARMUP["messages"]
        assert warm["max_tokens"] == bench.WARMUP["max_tokens"]
        assert warm["cache_prompt"] is False
        assert warm["chat_template_kwargs"] == {"enable_thinking": False}

        assert result.n_ctx == 32768
        assert result.pid == rig.machine.pid
        assert result.seconds >= 0.0
        assert result.vram_used_mib == 9000
        assert result.vram_other_mib == 1500
        assert result.vram_self_mib == 7500
        assert result.host_avail_gb == pytest.approx(9.5)
        assert rig.backend.loads == 1

    def test_load_that_never_comes_up_times_out(self, rig: Rig):
        rig.backend.begin()
        rig.machine.swallow_loads = True       # the dashboard accepts the load; no server ever serves it
        rig.machine.log_tail = "llama_model_loader: no new request handled"

        with pytest.raises(LoadFailed) as exc:
            rig.backend.load(Config(ctx=32768))

        assert exc.value.kind == "timeout"
        assert "not serving" in str(exc.value)
        assert rig.backend.loads == 0
        assert rig.backend._loaded is None
        # A timeout is exactly when the log is about to be lost (the next load truncates it), so it is kept.
        assert list(rig.run_dir.glob("crash-*.log"))

    def test_load_verifying_the_wrong_context_is_refused(self, rig: Rig):
        rig.backend.begin()
        rig.machine.ctx_override = 65536        # /props reports a context nobody asked for

        with pytest.raises(LoadFailed) as exc:
            rig.backend.load(Config(ctx=32768))

        assert exc.value.kind == "verify"
        assert "expected 32768" in str(exc.value)
        assert rig.backend._loaded is None

    def test_preflight_refuses_when_ram_is_short(self, rig: Rig):
        rig.backend.begin()
        rig.ram["avail_gb"] = 1.2              # 4000 MiB of host weights plus the 1 GB margin does not fit

        with pytest.raises(LoadFailed) as exc:
            rig.backend.load(Config(ctx=32768, ngl=20))

        assert exc.value.kind == "ram"
        assert rig.machine.loads == []          # refused before the dashboard was asked
        assert rig.backend._loaded is None


# --------------------------------------------------------------------------- #
# measuring
# --------------------------------------------------------------------------- #


class TestMeasure:
    def test_prompt_variants_continue_for_a_config_and_restart_for_the_next(self, rig: Rig):
        rig.backend.begin()
        rig.backend.load(Config(ctx=32768))

        rig.backend.measure("prose", reps=2)
        rig.backend.measure("prose", reps=1)          # same config: continues at variant 2
        rig.backend.measure("code", reps=1)           # other workload: its own sequence, from the start
        rig.backend.load(Config(ctx=65536))           # other config: prose starts again at variant 0
        rig.backend.measure("prose", reps=1)

        assert prompts(rig.machine) == [
            bench.WARMUP["messages"][0]["content"],
            *[bench._variant("prose", i)[0][0]["content"] for i in (0, 1, 2)],
            bench._variant("code", 0)[0][0]["content"],
            bench.WARMUP["messages"][0]["content"],
            bench._variant("prose", 0)[0][0]["content"],
        ]

    def test_samples_come_from_the_servers_own_timings(self, rig: Rig):
        rig.backend.begin()
        rig.backend.load(Config(ctx=32768))
        samples = rig.backend.measure("prefill", reps=3)

        assert [s.workload for s in samples] == ["prefill"] * 3
        assert [s.gen_tps for s in samples] == [45.0] * 3
        assert [s.prompt_tps for s in samples] == [1800.0] * 3
        assert [s.predicted_n for s in samples] == [8] * 3       # the prefill workload wants one word
        assert all(s.prompt_n > 0 for s in samples)
        assert all(s.power_key == "ac|best_performance|saver-off" for s in samples)
        assert [c["cache_prompt"] for c in rig.machine.chats[1:]] == [False] * 3

    @pytest.mark.parametrize("tail,kind", [
        ("ggml_cuda: error: cudaMalloc failed: out of memory", "oom"),
        ("main: build rng: failed to initialize generator", "crash"),
    ])
    def test_server_dying_mid_measurement_saves_the_log(self, rig: Rig, tail: str, kind: str):
        rig.backend.begin()
        rig.backend.load(Config(ctx=32768))
        rig.machine.log_tail = tail
        rig.machine.kill()

        with pytest.raises(LoadFailed) as exc:
            rig.backend.measure("prose", reps=2)

        assert exc.value.kind == kind
        assert f"server died during prose" in str(exc.value)
        assert rig.backend.crashes == 1
        assert (rig.run_dir / "crash-1.log").read_text(encoding="utf-8") == tail
        assert f"log saved as crash-1.log" in "\n".join(rig.lines)
        assert rig.backend._loaded is None                       # the search must not measure a dead config
        with pytest.raises(RuntimeError):
            rig.backend.measure("prose", reps=1)

    def test_a_failing_request_while_the_server_is_up_is_not_a_crash(self, rig: Rig):
        rig.backend.begin()
        rig.backend.load(Config(ctx=32768))
        rig.machine.alive = False                    # /v1/models still answers, so the server is fine

        with pytest.raises(LoadFailed) as exc:
            rig.backend.measure("prose", reps=1)

        # A live server that fails a request is a timeout for this config (the run goes on), not a crash.
        assert exc.value.kind == "timeout"
        assert rig.backend.crashes == 0
        assert not list(rig.run_dir.glob("crash-*.log"))
        assert rig.backend._loaded is not None

    def test_spill_guard_stops_the_server_and_fails_the_config(self, rig: Rig, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(real, "time", time)      # the guard's 2 s of low RAM has to be real time
        rig.backend.begin()
        rig.backend.load(Config(ctx=131072, ngl=999))
        stops_before = rig.machine.stops

        rig.machine.hold_s = 8.0                     # the request stays in flight while the guard watches
        rig.ram["avail_gb"] = 0.9                    # WDDM backed the VRAM overflow with system RAM

        with pytest.raises(LoadFailed) as exc:
            rig.backend.measure("prose", reps=1)

        assert exc.value.kind == "spill"
        assert "0.90 GB" in str(exc.value)
        assert rig.machine.stops == stops_before + 1        # llama-server was stopped through the dashboard
        assert rig.backend._loaded is None
        assert f"VRAM is spilling into system RAM" in "\n".join(rig.lines)


# --------------------------------------------------------------------------- #
# restore
# --------------------------------------------------------------------------- #


class TestRestore:
    def test_restore_uses_a_trial_load_when_the_model_was_not_active(self, rig: Rig):
        rig.backend.begin()
        rig.backend.load(Config(ctx=32768))
        assert rig.machine.model == MODEL

        rig.backend.restore()

        last = rig.machine.loads[-1]
        assert last["filename"] == ORIGINAL
        assert last["persist"] is False, "a saved load would rewrite models.json and make this model active"
        assert last["settings"]["contextSize"] == 8192
        assert last["settings"]["serverBinary"] == "llama-server-prism.exe"
        assert last["settings"]["extraArgs"] == "--jinja"
        assert rig.machine.model == ORIGINAL
        assert rig.machine.n_ctx == 8192

    def test_restore_uses_the_saved_load_when_the_model_was_active(self, rig: Rig):
        rig.machine.saved["activeModel"] = ORIGINAL
        rig.backend.begin()
        rig.backend.load(Config(ctx=32768))

        rig.backend.restore()

        last = rig.machine.loads[-1]
        assert last["filename"] == ORIGINAL
        assert "persist" not in last and "settings" not in last
        assert rig.machine.n_ctx == 8192

    def test_restore_leaves_the_server_stopped_when_nothing_was_running(self, rig: Rig):
        rig.machine.kill()                            # nothing was loaded before the run
        rig.backend.begin()
        assert rig.backend._original_running is False
        rig.backend.load(Config(ctx=32768))
        rig.backend.restore()

        assert rig.machine.loads[-1]["filename"] == MODEL
        assert rig.machine.serving is False
        assert rig.machine.pid is None