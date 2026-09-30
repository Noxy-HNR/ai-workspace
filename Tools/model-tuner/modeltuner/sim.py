"""Simulated backend: a plausible GPU, model file and noisy measurements.

The strategy only talks to a `Backend`, so `SimBackend` lets the whole search be
tested offline in seconds. The memory model mirrors what `llama-fit-params`
reports (weights split by layer, KV cache scaled by cache type, compute buffer
by ubatch); the speed model mirrors the real trade-offs (offload hurts a lot,
threads matter only offloaded, speculation helps per workload). All randomness
comes from one seeded `random.Random`, so a seed replays exactly; projections
and load decisions are deterministic and never consume that RNG.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass

from modeltuner.backend import LoadFailed, LoadResult, QualityResult, Sample
from modeltuner.fitparams import Projection
from modeltuner.knobs import Config

__all__ = ["SimBackend", "SimGpu", "SimModel"]

_POWER_BASE = "ac|best_performance|saver-off"
_POWER_ALT = "ac|balanced|saver-off"

# KV bytes per unit of context, relative to f16. q8_0 halves it; q4_0 quarters it.
_CACHE_FACTOR = {
    "f16": 1.0,
    "bf16": 1.0,
    "q8_0": 0.53,
    "q5_1": 0.34,
    "q5_0": 0.33,
    "q4_1": 0.29,
    "q4_0": 0.28,
    "iq4_nl": 0.28,
}

# Thread scaling when layers run on the CPU: best near 8 P-cores, worse past 12
# (E-cores), much worse below 4. Anchors are interpolated linearly.
_THREAD_ANCHORS = (
    (1, 0.30),
    (2, 0.40),
    (4, 0.65),
    (6, 0.90),
    (8, 1.00),
    (12, 0.90),
    (16, 0.75),
    (20, 0.60),
)

# ngram match-length scaling around the 8-12 sweet spot for copy-heavy code.
_NMATCH_ANCHORS = (
    (4, 0.85),
    (8, 1.00),
    (12, 1.00),
    (24, 0.75),
)


@dataclass
class SimModel:
    """The simulated model file."""

    name: str = "sim-9b.gguf"
    n_layer: int = 32
    weights_mib: int = 7300  # all layers
    kv_mib_per_1k_ctx_f16: float = 32.0  # per 1024 tokens at f16
    compute_mib_per_ubatch_256: int = 600  # scales ~linearly with ubatch
    has_mtp: bool = True
    train_ctx: int = 262144
    moe: bool = False
    base_gen_tps: float = 60.0  # decode, all layers on GPU, no speculation
    base_prompt_tps: float = 2500.0


@dataclass
class SimGpu:
    total_mib: int = 12227
    other_mib: int = 1500  # VRAM used by other apps
    margin_mib: int = 512


def _cache_factor(kind: str) -> float:
    if kind in _CACHE_FACTOR:
        return _CACHE_FACTOR[kind]
    low = kind.lower()
    if low in _CACHE_FACTOR:
        return _CACHE_FACTOR[low]
    # Unknown quant: quantized types roughly halve, float types do not.
    return 0.53 if low.startswith(("q", "iq")) else 1.0


def _interp(anchors: tuple[tuple[int, float], ...], value: int) -> float:
    if value <= anchors[0][0]:
        return anchors[0][1]
    for (x0, y0), (x1, y1) in zip(anchors, anchors[1:]):
        if value <= x1:
            frac = (value - x0) / (x1 - x0)
            return y0 + frac * (y1 - y0)
    return anchors[-1][1]


def _gpu_frac(cfg: Config, n_layer: int) -> float:
    if cfg.ngl >= 999:
        return 1.0
    return max(0.0, min(1.0, cfg.ngl / n_layer))


def _nmax_factor(draft_n_max: int | None) -> float:
    # n_max 2 is the sweet spot; 3 slightly worse; 1 and 4 clearly worse.
    if draft_n_max is None or draft_n_max == 2:
        return 1.0
    if draft_n_max == 3:
        return 0.97
    if draft_n_max in (1, 4):
        return 0.9
    return 0.85


def _nmatch_factor(ngram_n_match: int | None) -> float:
    if ngram_n_match is None:
        return 1.0
    return _interp(_NMATCH_ANCHORS, ngram_n_match)


def _jitter_mib(key: str) -> int:
    # Small deterministic load variance (driver allocations), 0-15 MiB. Hashed
    # rather than drawn from the RNG so loads never perturb measurement streams.
    digest = hashlib.md5(key.encode("utf-8")).digest()
    return digest[0] % 16


class SimBackend:
    """In-memory `Backend` with a plausible performance and memory model."""

    def __init__(
        self,
        model: SimModel | None = None,
        gpu: SimGpu | None = None,
        *,
        seed: int = 0,
        noise: float = 0.03,
        mtp_overhead_mib: int = 350,
        power_changes_at: int | None = None,
    ):
        self.model = model or SimModel()
        self.gpu = gpu or SimGpu()
        self.seed = seed
        self.noise = noise
        self.mtp_overhead_mib = mtp_overhead_mib
        self.power_changes_at = power_changes_at
        self.model_file: str = self.model.name
        self.loads = 0
        self.measures = 0
        self.projections = 0
        self.log: list[str] = []
        self.restored = False
        self._rng = random.Random(seed)
        self._loaded: Config | None = None
        self._loaded_actual_mib = 0
        self._stop = False
        self._pids = 10000

    # -- memory -------------------------------------------------------- #

    def vram_budget_mib(self) -> int:
        """VRAM the tuner may spend: total minus other apps minus margin."""
        return self.gpu.total_mib - self.gpu.other_mib - self.gpu.margin_mib

    def _projection_for(self, cfg: Config) -> Projection:
        frac = _gpu_frac(cfg, self.model.n_layer)
        device_model = round(self.model.weights_mib * frac)
        host_model = self.model.weights_mib - device_model

        kv_full = self.model.kv_mib_per_1k_ctx_f16 * (cfg.ctx / 1024)
        cache = (_cache_factor(cfg.cache_k) + _cache_factor(cfg.cache_v)) / 2.0
        total_ctx = round(kv_full * cache)
        device_ctx = round(total_ctx * frac)
        host_ctx = total_ctx - device_ctx

        total_compute = round(self.model.compute_mib_per_ubatch_256 * (cfg.ubatch / 256))
        device_compute = round(total_compute * frac)
        host_compute = total_compute - device_compute

        return Projection(
            device_model_mib=device_model,
            device_context_mib=device_ctx,
            device_compute_mib=device_compute,
            host_model_mib=host_model,
            host_context_mib=host_ctx,
            host_compute_mib=host_compute,
            device_total_mib=self.gpu.total_mib,
            fitted_ngl=cfg.ngl if cfg.ngl < 999 else None,
            fitted_ctx=cfg.ctx,
            seconds=0.0,
        )

    def project(self, cfg: Config) -> Projection:
        proj = self._projection_for(cfg)
        self.projections += 1
        self.log.append(
            f"project {cfg.describe()} -> device {proj.device_self_mib} MiB "
            f"(model {proj.device_model_mib}+ctx {proj.device_context_mib}"
            f"+compute {proj.device_compute_mib})"
        )
        return proj

    # -- loading ------------------------------------------------------- #

    def load(self, cfg: Config) -> LoadResult:
        if cfg.batch < cfg.ubatch:
            raise LoadFailed("refused", f"batch {cfg.batch} < ubatch {cfg.ubatch}: invalid")
        if cfg.uses_mtp and not self.model.has_mtp:
            raise LoadFailed("crash", f"model {self.model.name} has no MTP head for spec {cfg.spec!r}")
        proj = self._projection_for(cfg)
        jitter = _jitter_mib(cfg.key())
        actual = proj.device_self_mib + (self.mtp_overhead_mib if cfg.uses_mtp else 0) + jitter
        limit = self.gpu.total_mib - self.gpu.other_mib
        if actual > limit:
            raise LoadFailed(
                "oom",
                f"needs {actual} MiB on device but only {limit} MiB free "
                f"(projection {proj.device_self_mib} + overhead + jitter)",
            )
        self.loads += 1
        self._pids += 1
        self._loaded = cfg
        self._loaded_actual_mib = actual
        self.log.append(f"load {cfg.describe()} -> {actual} MiB on device (jitter {jitter})")
        return LoadResult(
            seconds=2.5 + actual / 4000.0,
            vram_used_mib=actual + self.gpu.other_mib,
            vram_other_mib=self.gpu.other_mib,
            host_avail_gb=8.0,
            pid=self._pids,
            n_ctx=cfg.ctx,
        )

    # -- measuring ----------------------------------------------------- #

    def _decode_tps(self, cfg: Config, workload: str) -> tuple[float, float]:
        """Base decode speed and speculation multiplier (before noise)."""
        frac = _gpu_frac(cfg, self.model.n_layer)
        base = self.model.base_gen_tps
        harmonic = 1.0 / (frac / base + (1.0 - frac) / (base * 0.12))
        if frac >= 0.999:
            # On-GPU runs barely notice threads: <1% across any sane range.
            thread = 1.0 - 0.0005 * abs(cfg.threads - 8)
        else:
            thread = _interp(_THREAD_ANCHORS, cfg.threads)
        if cfg.ctx > 32768:
            ctx_factor = 0.99 ** math.log2(cfg.ctx / 32768)
        else:
            ctx_factor = 1.0
        cache_factor = 0.97 if cfg.cache_k == "q4_0" or cfg.cache_v == "q4_0" else 1.0

        mtp = cfg.uses_mtp
        ngram = cfg.uses_ngram
        if workload == "code":
            if mtp and ngram:
                spec = 3.0
            elif mtp:
                spec = 1.6 * _nmax_factor(cfg.draft_n_max)
            elif ngram:
                spec = 2.6 * _nmatch_factor(cfg.ngram_n_match)
            else:
                spec = 1.0
        else:  # "prose" and anything decode-like
            if mtp and ngram:
                spec = 1.5
            elif mtp:
                spec = 1.45 * _nmax_factor(cfg.draft_n_max)
            elif ngram:
                spec = 1.05
            else:
                spec = 1.0
        return harmonic * thread * ctx_factor * cache_factor, spec

    def measure(self, workload: str, *, reps: int) -> list[Sample]:
        if self._loaded is None:
            raise RuntimeError("measure() with nothing loaded: call load() first")
        if workload not in ("prose", "code", "prefill"):
            raise ValueError(f"unknown workload {workload!r}")
        cfg = self._loaded
        key = self.power_key()
        out: list[Sample] = []
        for _ in range(reps):
            if workload == "prefill":
                prompt = self.model.base_prompt_tps * (min(cfg.ubatch, 512) / 512)
                prompt *= self._rng.gauss(1.0, self.noise) if self.noise else 1.0
                prompt = max(0.1, prompt)
                out.append(
                    Sample(
                        workload=workload,
                        gen_tps=None,
                        prompt_tps=prompt,
                        prompt_n=4096,
                        predicted_n=16,
                        draft_n=None,
                        draft_accepted=None,
                        wall_s=4096 / prompt,
                        power_key=key,
                    )
                )
            else:
                speed, spec = self._decode_tps(cfg, workload)
                gen = speed * spec
                gen *= self._rng.gauss(1.0, self.noise) if self.noise else 1.0
                gen = max(0.1, gen)
                if cfg.spec == "none":
                    draft_n = draft_ok = None
                else:
                    draft_n = 128
                    draft_ok = round(draft_n * (1.0 - 1.0 / spec))
                out.append(
                    Sample(
                        workload=workload,
                        gen_tps=gen,
                        prompt_tps=None,
                        prompt_n=64,
                        predicted_n=128,
                        draft_n=draft_n,
                        draft_accepted=draft_ok,
                        wall_s=128 / gen,
                        power_key=key,
                    )
                )
        self.measures += 1
        self.log.append(f"measure {workload} x{reps} on {cfg.describe()}")
        return out

    # -- quality / power / lifecycle ----------------------------------- #

    def quality(self, suite: str, *, limit: int | None = None) -> QualityResult:
        total = limit if limit else 50
        base = {"reasoning": 0.85, "tools": 0.88, "coding": 0.80, "recall": 0.90}.get(suite, 0.85)
        cfg = self._loaded
        if cfg is not None and (cfg.cache_k == "q4_0" or cfg.cache_v == "q4_0"):
            base -= 0.05  # q4_0 KV loses a little accuracy
        if (
            suite == "reasoning"
            and cfg is not None
            and cfg.reasoning_budget is not None
            and cfg.reasoning_budget < 1024
        ):
            base -= 0.15  # a tight thinking cap hurts reasoning first
        rate = min(1.0, max(0.0, base + (self._rng.gauss(0.0, 0.01) if self.noise else 0.0)))
        passed = min(total, max(0, round(rate * total)))
        self.log.append(f"quality {suite} limit={total} -> {passed}/{total}")
        return QualityResult(
            suite=suite,
            passed=passed,
            total=total,
            tokens=total * 150,
            seconds=total * 0.4,
        )

    def power_key(self) -> str:
        if self.power_changes_at is not None and self.measures >= self.power_changes_at:
            return _POWER_ALT
        return _POWER_BASE

    def should_stop(self) -> bool:
        return self._stop

    def request_stop(self) -> None:
        self._stop = True
        self.log.append("request_stop")

    def restore(self) -> None:
        self.restored = True
        self.log.append("restore original model")
