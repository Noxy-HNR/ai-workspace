"""In-process stand-ins for llama-server, the dashboard, the hardware probes and the clock.

Nothing here opens a socket, starts a process or reads anything under C:\\AI: every test that uses these fakes
proves a defect in the tuner's own logic, not in the machine.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from modeltuner.backend import QualityResult, Sample
from modeltuner.fitparams import Projection
from modeltuner.knobs import Config
from modeltuner.llama import LlamaError


class FakeClock:
    """A clock that only moves when someone sleeps, so a polling loop runs instantly and deterministically."""

    def __init__(self, now: float = 1_000.0):
        self.now = now

    def time(self) -> float:
        return self.now

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


@dataclass
class FakeMemory:
    avail_gb: float = 8.0
    total_gb: float = 15.4
    load_pct: int = 40


class FakeLlama:
    """LlamaClient stand-in. `props_error` and `chat_error` are the two ways a real server misbehaves."""

    def __init__(self, model: str = "m.gguf", n_ctx: int = 8192, props: list[dict] | None = None,
                 props_error: Exception | None = None, chat_error: Exception | None = None,
                 ready: bool = True, timings: dict | None = None):
        self.model = model
        self.n_ctx = n_ctx
        self._props = props
        self.props_error = props_error
        self.chat_error = chat_error
        self.ready_value = ready
        self.timings = timings or {"predicted_per_second": 40.0, "prompt_per_second": 900.0,
                                   "prompt_n": 300, "predicted_n": 200}
        self.chats: list[dict] = []
        self.ready_calls = 0

    def props(self, timeout: float = 10) -> dict:
        if self.props_error:
            raise self.props_error
        if self._props is not None:
            p = self._props.pop(0) if len(self._props) > 1 else self._props[0]
            if p is None:
                raise LlamaError("GET /props failed: timed out")
            return p
        return {"model_path": str(Path("C:/AI/models") / self.model),
                "default_generation_settings": {"n_ctx": self.n_ctx}}

    def ready(self, timeout: float = 5) -> bool:
        self.ready_calls += 1
        return self.ready_value

    def loaded_model(self) -> str | None:
        try:
            return Path(self.props()["model_path"]).name
        except (LlamaError, KeyError, TypeError):
            return None

    def chat(self, messages, **kw):
        self.chats.append({"messages": messages, **kw})
        if self.chat_error:
            raise self.chat_error
        return _FakeResult(self.timings, wall_s=1.0)


class _FakeResult:
    def __init__(self, timings: dict, wall_s: float):
        self.timings = timings
        self.wall_s = wall_s
        self.content = "ok"
        self.reasoning = ""
        self.tool_calls: list[dict] = []
        self.finish_reason = "stop"
        self.usage: dict[str, int] = {}
        self.raw: dict[str, Any] = {}


class FakeDash:
    """Dashboard stand-in. `calls` is the ordered list of everything the tuner asked the dashboard to do."""

    def __init__(self, llama: FakeLlama | None = None, config: dict | None = None,
                 config_error: Exception | None = None, stop_error: Exception | None = None,
                 models_dir: str = "C:/AI/models"):
        self.llama = llama or FakeLlama()
        self.calls: list[tuple] = []
        self._config = config if config is not None else {
            "models": {"m.gguf": {"contextSize": 4096, "gpuLayers": 99}},
            "activeModel": "m.gguf",
        }
        self._config_error = config_error
        self._stop_error = stop_error
        self.models_dir = models_dir

    # -- the calls real.py / runner.py make -------------------------------------------------
    def trial_load(self, model_file: str, settings: dict) -> dict:
        self.calls.append(("trial_load", model_file, settings))
        return {"success": True, "trial": True}

    def load_saved(self, model_file: str) -> dict:
        self.calls.append(("load_saved", model_file))
        return {"success": True}

    def stop_llama(self) -> None:
        self.calls.append(("stop_llama",))
        if self._stop_error:
            raise self._stop_error

    def models_config(self) -> dict:
        self.calls.append(("models_config",))
        if self._config_error:
            raise self._config_error
        return self._config

    def alive(self) -> bool:
        return True

    def save_settings(self, model_file: str, settings: dict) -> dict:
        self.calls.append(("save_settings", model_file))
        return {"success": True}

    # -- test helpers -----------------------------------------------------------------------
    def names(self) -> list[str]:
        return [c[0] for c in self.calls]


@dataclass
class FakeProfile:
    file: str = "m.gguf"
    path: str = "C:/AI/models/m.gguf"
    server_binary: str | None = None
    current: dict = field(default_factory=dict)
    arch: str = "qwen3"
    train_ctx: int = 32768
    family: str = "qwen3"
    has_mtp: bool = False

    def summary_lines(self) -> list[str]:
        return ["fake profile"]

    def current_config(self) -> Config:
        return Config(ctx=8192)


def fake_projection(device_mib: int = 6000, host_mib: int = 500) -> Projection:
    return Projection(device_model_mib=device_mib, device_context_mib=0, device_compute_mib=0,
                      host_model_mib=host_mib, host_context_mib=0, host_compute_mib=0,
                      device_total_mib=12227, fitted_ngl=None, fitted_ctx=None, seconds=2.0)


class FakeFit:
    """FitParams stand-in: projects instantly and never runs llama-fit-params."""

    def __init__(self, *a, **kw):
        self.calls = 0
        self.projection = fake_projection()

    def project(self, cfg) -> Projection:
        self.calls += 1
        return self.projection


def make_samples(workload: str, n: int, gen: float = 40.0) -> list[Sample]:
    return [Sample(workload=workload, gen_tps=gen, prompt_tps=900.0, prompt_n=300, predicted_n=200,
                   draft_n=None, draft_accepted=None, wall_s=1.0, power_key="ac|balanced|saver-off")
            for _ in range(n)]


def quality_with_server_errors(n: int = 8) -> QualityResult:
    """What every quality suite returns when llama-server died half way through: 0 of n, all 'server_error'."""
    return QualityResult("probe", 0, n, 0, 12.0,
                         [{"id": f"t{i}", "passed": False, "tokens": 0, "seconds": 1.0,
                           "finish_reason": None, "cause": "server_error", "reason": "server error: timed out"}
                          for i in range(n)])


@contextlib.contextmanager
def no_lock():
    """Replaces safety.LlamaLock (whose path default is bound at import time) so tests touch no state dir."""
    yield
