"""The interface between the strategy (what to try) and the machine (loading, measuring, projecting).

`strategy.py` only talks to a `Backend`. The real backend drives the dashboard and llama-server; `sim.SimBackend`
models a GPU, a model and noisy measurements so the whole strategy can be tested offline in seconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from modeltuner.fitparams import Projection
from modeltuner.knobs import Config


class LoadFailed(RuntimeError):
    """A config could not be loaded. `kind` says why, so the strategy can react (e.g. back off on 'oom')."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind  # "oom" | "crash" | "timeout" | "refused" | "ram" | "verify"


@dataclass(frozen=True)
class LoadResult:
    seconds: float                 # load call to verified "ready"
    vram_used_mib: int | None      # nvidia-smi total used after load and warm-up
    vram_other_mib: int | None     # VRAM used by everything except llama-server (baseline), when known
    host_avail_gb: float | None    # available RAM after load
    pid: int | None
    n_ctx: int | None              # as reported by /props

    @property
    def vram_self_mib(self) -> int | None:
        if self.vram_used_mib is None or self.vram_other_mib is None:
            return None
        return self.vram_used_mib - self.vram_other_mib


@dataclass(frozen=True)
class Sample:
    workload: str                  # "prose" | "code" | "prefill" | ...
    gen_tps: float | None          # decode tokens per second (timings.predicted_per_second)
    prompt_tps: float | None       # prefill tokens per second (timings.prompt_per_second)
    prompt_n: int
    predicted_n: int
    draft_n: int | None
    draft_accepted: int | None
    wall_s: float
    power_key: str                 # hardware.PowerState.key() at measurement time

    @property
    def acceptance(self) -> float | None:
        if not self.draft_n:
            return None
        return (self.draft_accepted or 0) / self.draft_n


@dataclass(frozen=True)
class QualityResult:
    suite: str                     # "reasoning" | "tools" | "coding" | "recall"
    passed: int
    total: int
    tokens: int                    # completion tokens spent across the suite
    seconds: float
    details: list[dict] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


class Backend(Protocol):
    """Everything the strategy needs from the outside world."""

    model_file: str

    def project(self, cfg: Config) -> Projection:
        """Memory projection without loading (llama-fit-params)."""

    def vram_budget_mib(self) -> int:
        """VRAM available to llama-server: total minus what other processes use, minus a safety margin."""

    def load(self, cfg: Config) -> LoadResult:
        """Load `cfg` (not persisted), verify it, warm it up. Raises LoadFailed."""

    def measure(self, workload: str, *, reps: int) -> list[Sample]:
        """Run `workload` `reps` times on the loaded config (after a discarded warm-up) and return the samples."""

    def quality(self, suite: str, *, limit: int | None = None) -> QualityResult:
        """Run a quality suite on the loaded config."""

    def power_key(self) -> str:
        """Current power-state key; samples are only comparable when keys match."""

    def should_stop(self) -> bool:
        """True when the user asked the run to stop (cooperative)."""

    def restore(self) -> None:
        """Put the user's original model and settings back."""
