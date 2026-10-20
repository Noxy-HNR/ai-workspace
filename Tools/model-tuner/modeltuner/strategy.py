"""The brain: decides what to try, in what order, and when to stop.

The approach mirrors how a careful person tunes a model on this machine:
1. Fit on paper first. llama-fit-params projections cost ~2 s and load nothing, so placement (all layers on the GPU or
   a split), the largest context and the cache type are decided by binary search over projections.
2. Validate with one real load, and learn the gap between projection and reality (the MTP draft context, for example,
   is invisible to the projection).
3. Measure the user's current settings as the baseline, in the same power state.
4. Tune the knobs that matter for THIS model, in order of expected impact, and skip the ones that cannot matter
   (threads with every layer on the GPU; MTP without an MTP head).
5. Numeric knobs use a direction search: step one way, keep going while it wins, never continue past a loss.
6. Every comparison is noise-aware (stats.compare): repeated samples, medians, and a bar of max(3%, 2x spread).
7. Quality gates (medium/full modes) reject a faster config that answers worse.
Every decision is emitted with a plain-English reason; the report is built from those.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from modeltuner import stats
from modeltuner.backend import Backend, LoadFailed, LoadResult, QualityResult, Sample
from modeltuner.fitparams import FitError, Projection
from modeltuner.knobs import SPEC_NONE, Config
from modeltuner.profile import ModelProfile

METRIC = {"prose": "gen_tps", "code": "gen_tps", "prefill": "prompt_tps"}
DEFAULT_WEIGHTS = {"prose": 0.45, "code": 0.35, "prefill": 0.20}
SPEC_SIMPLICITY = {"none": 0, "ngram-mod": 1, "draft-mtp": 2, "draft-mtp,ngram-mod": 3}
DEFAULT_MTP_OVERHEAD_MIB = 1000         # until calibrated for a family (measured 896-1210 MiB on the 9B)
MODES = ("quick", "medium", "full")


class Stopped(Exception):
    """The user asked the run to stop."""


class PowerChanged(Exception):
    """The power state changed and did not come back; measurements would not be comparable."""


@dataclass
class TuneOptions:
    mode: str = "quick"
    reps: int = 3
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    allow_q4: bool = False
    min_ctx: int = 8192
    ctx_target: int | None = None       # a fixed context the user wants; None = the largest that fits
    max_loads: int = 45
    power_wait_s: int = 600


@dataclass
class TrialRecord:
    cfg: Config
    phase: str
    outcome: str = "pending"            # measured | oom | crash | refused | skipped | ...
    reason: str = ""
    projection: Projection | None = None
    load: LoadResult | None = None
    samples: dict[str, list[Sample]] = field(default_factory=dict)
    quality: dict[str, QualityResult] = field(default_factory=dict)

    def medians(self) -> dict[str, float]:
        out = {}
        for w, ss in self.samples.items():
            vals = [getattr(s, METRIC[w]) for s in ss if getattr(s, METRIC[w])]
            if vals:
                out[w] = stats.median(vals)
        return out


@dataclass
class TuneResult:
    status: str                         # "ok" | "stopped" | "failed" | "power_changed"
    baseline: Config
    best: Config
    baseline_metrics: dict[str, float]
    best_metrics: dict[str, float]
    improvement: float                  # relative_score(best, baseline)
    decisions: list[str]
    notes: list[str]
    trials: list[TrialRecord]
    loads: int
    seconds: float
    quality: dict[str, dict[str, Any]] = field(default_factory=dict)


def round_ctx(n: int) -> int:
    step = 8192 if n > 65536 else 4096
    return max(step, (n // step) * step)


class Tuner:
    def __init__(self, backend: Backend, profile: ModelProfile, baseline: Config, opts: TuneOptions | None = None,
                 emit: Callable[..., None] | None = None, history: Any = None, fingerprint: str | None = None,
                 run_id: str | None = None):
        self.b = backend
        self.run_id = run_id
        self._tid: dict[str, int] = {}             # config key -> history trial id
        self.p = profile
        self.baseline = baseline
        self.o = opts or TuneOptions()
        self._emit = emit or (lambda kind, msg, **d: None)
        self.history = history
        self.fingerprint = fingerprint
        self.trials: dict[str, TrialRecord] = {}
        self._qmemo: dict[tuple, QualityResult] = {}   # (config key, suite, limit) -> result
        self.order: list[TrialRecord] = []
        self.decisions: list[str] = []
        self.notes: list[str] = []
        self.loaded_key: str | None = None
        self.loads = 0
        self.power_key = backend.power_key()
        self.ref: dict[str, float] = {}           # baseline medians: the yardstick for composite scores
        self.mtp_overhead = DEFAULT_MTP_OVERHEAD_MIB
        self.fit_error = 0.0                      # learned (actual - projected) MiB, all configs
        self.budget = 0

    # ------------------------------------------------------------------ plumbing
    def emit(self, kind: str, msg: str, **data: Any) -> None:
        if kind == "decision":
            self.decisions.append(msg)
        elif kind == "note":
            self.notes.append(msg)
        self._emit(kind, msg, **data)

    def check_stop(self) -> None:
        if self.b.should_stop():
            raise Stopped("stop requested")

    def trial(self, cfg: Config, phase: str) -> TrialRecord:
        k = cfg.key()
        if k not in self.trials:
            t = TrialRecord(cfg=cfg, phase=phase)
            self.trials[k] = t
            self.order.append(t)
        return self.trials[k]

    def overhead(self, cfg: Config) -> float:
        return (self.mtp_overhead if cfg.uses_mtp else 0) + max(0.0, self.fit_error)

    def project(self, cfg: Config) -> Projection | None:
        try:
            return self.b.project(cfg)
        except FitError as e:
            self.emit("warn", f"projection failed for {cfg.describe()}: {e}")
            return None

    def fits(self, cfg: Config) -> bool:
        pr = self.project(cfg)
        return pr is not None and pr.device_self_mib + self.overhead(cfg) <= self.budget

    # ------------------------------------------------------------------ loading and measuring
    # ---- history (optional; every call is guarded so the tuner works without it) ----
    def _hist_trial(self, t: TrialRecord) -> int | None:
        if not (self.history and self.run_id and self.fingerprint):
            return None
        k = t.cfg.key()
        if k not in self._tid:
            try:
                self._tid[k] = self.history.add_trial(self.run_id, self.fingerprint, t.cfg, t.phase, t.projection)
            except Exception as e:  # noqa: BLE001 -- history must never break a run
                self.emit("warn", f"history: {e}")
                return None
        return self._tid[k]

    def _hist(self, method: str, *args) -> Any:
        if not self.history:
            return None
        try:
            return getattr(self.history, method)(*args)
        except Exception as e:  # noqa: BLE001
            self.emit("warn", f"history.{method}: {e}")
            return None

    def family_prior(self, field_name: str) -> Any:
        """The value of `field_name` in the newest best config of another model in the same family, if any."""
        rows = self._hist("best_for_family", self.p.family, self.fingerprint) or []
        for _file, cfg, _metrics in rows:
            v = getattr(cfg, field_name, None)
            if v is not None:
                return v
        return None

    def ensure_loaded(self, t: TrialRecord) -> bool:
        k = t.cfg.key()
        if self.loaded_key == k:
            return True
        if t.outcome in ("oom", "crash", "spill", "refused", "ram", "verify", "timeout"):
            return False
        if self.loads >= self.o.max_loads:
            raise _Budget()
        self.check_stop()
        if self.fingerprint:
            prior = self._hist("known_failure", self.fingerprint, t.cfg)
            if prior in ("oom", "spill", "crash", "refused"):
                t.outcome, t.reason = prior, "failed in an earlier run; not loading it again"
                self.emit("decision", f"Skipped {t.cfg.describe()}: it failed ({prior}) in an earlier run.")
                return False
        t.projection = t.projection or self.project(t.cfg)
        self.emit("load", f"loading: {t.cfg.describe()}", phase=t.phase)
        try:
            t.load = self.b.load(t.cfg)
        except LoadFailed as e:
            self.loads += 1
            self.loaded_key = None
            t.outcome, t.reason = e.kind, str(e)
            self.emit("load_failed", f"{e.kind}: {e}", phase=t.phase, config=t.cfg.describe(), failure=e.kind)
            tid = self._hist_trial(t)
            if tid is not None:
                self._hist("set_trial_outcome", tid, e.kind, str(e)[:500], None)
            return False
        self.loads += 1
        self.loaded_key = k
        tid = self._hist_trial(t)
        if tid is not None:
            self._hist("set_trial_outcome", tid, "loaded", "", t.load)
        self._calibrate(t)
        return True

    def _calibrate(self, t: TrialRecord) -> None:
        actual = t.load.vram_self_mib if t.load else None
        if actual is None or t.projection is None:
            return
        err = actual - t.projection.device_self_mib
        if t.cfg.uses_mtp:
            # what the projection can't see: MTP draft context (plus the generic error, learned separately)
            self.mtp_overhead = max(0.0, err - max(0.0, self.fit_error))
        else:
            self.fit_error = err
        self.emit("calibration", f"projected {t.projection.device_self_mib} MiB, measured {actual} MiB "
                                 f"({err:+d} MiB){' with MTP' if t.cfg.uses_mtp else ''}",
                  projected=t.projection.device_self_mib, actual=actual, uses_mtp=t.cfg.uses_mtp)
        self._hist("add_calibration", self.p.family, self.p.server_binary or "", t.cfg.uses_mtp,
                   t.projection.device_self_mib, actual)

    def measure(self, t: TrialRecord, workloads: list[str], reps: int | None = None) -> bool:
        """Make sure `t` has at least `reps` samples of each workload. False if it can't be loaded."""
        reps = reps or self.o.reps
        for w in workloads:
            have = len(t.samples.get(w, []))
            if have >= reps:
                continue
            if not self.ensure_loaded(t):
                return False
            self.check_stop()
            try:
                new = self._measure_power_safe(w, reps - have)
            except LoadFailed as e:
                t.outcome, t.reason = e.kind, str(e)
                self.loaded_key = None
                self.emit("load_failed", f"{e.kind} during {w}: {e}", phase=t.phase, config=t.cfg.describe(),
                          failure=e.kind)
                tid = self._hist_trial(t)
                if tid is not None:
                    self._hist("set_trial_outcome", tid, e.kind, str(e)[:500], None)
                return False
            t.samples.setdefault(w, []).extend(new)
            t.outcome = "measured"
            tid = self._hist_trial(t)
            if tid is not None:
                self._hist("add_samples", tid, new)
                self._hist("set_trial_outcome", tid, "measured", "", t.load)
            vals = [getattr(s, METRIC[w]) or 0 for s in new]
            acc = [s.acceptance for s in new if s.acceptance is not None]
            self.emit("sample", f"{w}: " + ", ".join(f"{v:.1f}" for v in vals) + " tok/s"
                      + (f" (draft acceptance {stats.median(acc):.0%})" if acc else ""),
                      workload=w, values=vals, config=t.cfg.describe())
        return True

    def _measure_power_safe(self, workload: str, reps: int) -> list[Sample]:
        for attempt in range(3):
            samples = self.b.measure(workload, reps=reps)
            bad = [s for s in samples if s.power_key != self.power_key]
            if not bad:
                return samples
            self.emit("warn", f"power state changed to {bad[0].power_key} (run started on {self.power_key}); "
                              f"those samples are discarded")
            waited = 0
            while self.b.power_key() != self.power_key and waited < self.o.power_wait_s:
                self.check_stop()
                time.sleep(15)
                waited += 15
            if self.b.power_key() != self.power_key:
                raise PowerChanged(f"power state stayed {self.b.power_key()} (run started on {self.power_key})")
        raise PowerChanged("power state kept changing during measurement")

    # ------------------------------------------------------------------ scoring
    def composite(self, t: TrialRecord, workloads: list[str]) -> list[float]:
        """One composite score per repetition: weighted geometric mean of speed ratios to the baseline."""
        ws = {w: self.o.weights.get(w, 0) for w in workloads if self.o.weights.get(w, 0) > 0 and w in self.ref}
        if not ws:
            return []
        n = min(len(t.samples.get(w, [])) for w in ws)
        tot = sum(ws.values())
        out = []
        for i in range(n):
            acc = 0.0
            for w, wt in ws.items():
                v = getattr(t.samples[w][i], METRIC[w]) or 1e-9
                acc += wt * math.log(v / self.ref[w])
            out.append(math.exp(acc / tot))
        return out

    def challenge(self, inc: TrialRecord, cand: TrialRecord, workloads: list[str], label: str) -> stats.Verdict:
        if not self.measure(cand, workloads):
            self.emit("decision", f"{label}: {cand.cfg.describe()} could not load ({cand.outcome}); kept "
                                  f"{inc.cfg.describe()}.")
            return stats.Verdict.WORSE
        if not self.measure(inc, workloads):
            # The incumbent itself no longer loads (e.g. it spilled earlier) and the candidate does: the candidate
            # takes over, but say so -- it wins by default, not by measurement.
            self.emit("decision", f"{label}: the incumbent {inc.cfg.describe()} could not be measured "
                                  f"({inc.outcome}); {cand.cfg.describe()} takes over by default.")
            return stats.Verdict.BETTER
        a, b = self.composite(inc, workloads), self.composite(cand, workloads)
        if not a or not b:
            self.emit("decision", f"{label}: no comparable measurements ({', '.join(workloads)}); kept "
                                  f"{inc.cfg.describe()}.")
            return stats.Verdict.NEED_MORE
        c = stats.compare(a, b)
        if c.verdict is stats.Verdict.NEED_MORE:
            self.measure(cand, workloads, reps=self.o.reps + 2)
            c = stats.compare(self.composite(inc, workloads), self.composite(cand, workloads))
        self.emit("compare", f"{label}: {c.reason}", verdict=c.verdict.value, delta=c.delta_rel,
                  incumbent=inc.cfg.describe(), candidate=cand.cfg.describe())
        return c.verdict

    # ------------------------------------------------------------------ phases
    def phase_fit(self) -> Config:
        self.emit("phase", "Fit on paper (llama-fit-params projections, nothing loaded)")
        self.budget = self.b.vram_budget_mib()
        # Priors from earlier runs on this family: the projection's usual error, and what MTP adds on top of it.
        plain = self._hist("calibration_error", self.p.family, self.p.server_binary or "", False)
        mtp = self._hist("calibration_error", self.p.family, self.p.server_binary or "", True)
        if plain:
            self.fit_error = max(0.0, plain[0])
        if mtp:
            self.mtp_overhead = max(0.0, mtp[0] - self.fit_error)
        if plain or mtp:
            self.emit("note", f"Calibration from earlier runs ({self.p.family}): projection error "
                              f"{self.fit_error:+.0f} MiB, MTP draft context {self.mtp_overhead:.0f} MiB.")
        self.emit("note", f"VRAM budget for llama-server: {self.budget} MiB (after other apps and a safety margin).")
        base = self.baseline
        cap = self.p.train_ctx
        # Projections can't see the MTP draft context, so when the model has an MTP head the fit reserves its
        # (calibrated) cost up front. Otherwise context would fill VRAM and MTP, usually the biggest decode win,
        # could never be tried.
        spec_probe = base.with_(spec="draft-mtp", draft_n_max=base.draft_n_max or 2) if self.p.has_mtp else base

        def max_ctx(cfg: Config) -> int | None:
            lo = round_ctx(self.o.min_ctx)
            if not self.fits(cfg.with_(ctx=lo)):
                return None
            hi = cap
            if self.fits(cfg.with_(ctx=round_ctx(hi))):
                return round_ctx(hi)
            # binary search on the rounded grid
            while round_ctx(hi) - lo > (8192 if lo > 65536 else 4096):
                mid = round_ctx((lo + hi) // 2)
                if mid <= lo:
                    break
                if self.fits(cfg.with_(ctx=mid)):
                    lo = mid
                else:
                    hi = mid
            return lo

        gpu_all = spec_probe.with_(ngl=999)
        caches = [("f16", "f16"), ("q8_0", "q8_0")] + ([("q4_0", "q4_0")] if self.o.allow_q4 else [])
        best_by_cache: dict[tuple[str, str], int | None] = {}
        for ck, cv in caches:
            best_by_cache[(ck, cv)] = max_ctx(gpu_all.with_(cache_k=ck, cache_v=cv))
            self.check_stop()
        want = self.o.ctx_target
        fitting = {k: v for k, v in best_by_cache.items() if v}
        if fitting:
            # all layers on the GPU is possible
            if want:
                choice = next(((k, v) for k, v in fitting.items() if v >= want), None)
                if choice is None:
                    k = max(fitting, key=lambda kk: fitting[kk])
                    self.emit("note", f"The requested context {want:,} does not fit with every layer on the GPU; "
                                      f"the most that fits is {fitting[k]:,}.")
                    choice = (k, fitting[k])
                else:
                    choice = (choice[0], round_ctx(want))
            else:
                f16 = fitting.get(("f16", "f16"))
                if f16 and f16 >= cap:
                    choice = (("f16", "f16"), f16)
                    self.emit("decision", f"f16 cache already reaches the training context ({cap:,}), so there is "
                                          f"no reason to quantise it.")
                else:
                    k = ("q8_0", "q8_0") if ("q8_0", "q8_0") in fitting else max(fitting, key=lambda kk: fitting[kk])
                    choice = (k, fitting[k])
                    if f16:
                        self.emit("decision", f"q8_0 cache fits {fitting[k]:,} tokens vs {f16:,} at f16 with "
                                              f"practically no quality cost, so q8_0 it is.")
            (ck, cv), ctx = choice
            cfg = base.with_(ngl=999, ctx=ctx, cache_k=ck, cache_v=cv)
            summary = ", ".join(f"{k[0]}: {('%s' % f'{v:,}') if v else 'does not fit'}" for k, v in best_by_cache.items())
            self.emit("decision", f"Every layer fits on the GPU. Largest context by cache type: {summary}. "
                                  f"Chose {ctx:,} tokens with a {ck} cache"
                                  f"{' (capped at the training context)' if ctx >= cap else ''}.")
            return cfg
        # offload: pick the context first, then the most GPU layers that fit
        # Every token of context on a split model costs GPU layers, and layers are what decide decode speed, so
        # without an explicit target the context is capped at 64K here (the user's setting if smaller).
        ctx = round_ctx(want or min(base.ctx or 16384, 65536))
        if not want and base.ctx and base.ctx > 65536:
            self.emit("note", f"Your current context ({base.ctx:,}) would push many layers onto the CPU; the plan "
                              f"uses 65,536. Ask for more with --ctx if you need it.")
        cache = (base.cache_k, base.cache_v) if base.cache_k in ("f16", "q8_0") else ("q8_0", "q8_0")
        lo, hi = 0, self.p.n_layer
        probe = spec_probe.with_(ctx=ctx, cache_k=cache[0], cache_v=cache[1])
        if not self.fits(probe.with_(ngl=0)):
            ctx = round_ctx(self.o.min_ctx)
            probe = probe.with_(ctx=ctx)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.fits(probe.with_(ngl=mid)):
                lo = mid
            else:
                hi = mid - 1
        cfg = base.with_(ngl=lo, ctx=probe.ctx, cache_k=cache[0], cache_v=cache[1])
        self.emit("decision", f"The weights ({self.p.weights_mib:,} MiB) do not fit in the {self.budget:,} MiB budget "
                              f"with a useful context, so layers are split: {lo} of {self.p.n_layer} on the GPU at "
                              f"{ctx:,} tokens of context ({cache[0]} cache). Threads now matter and will be tuned.")
        return cfg

    def phase_validate(self, cfg: Config) -> Config:
        self.emit("phase", "Validate the fit with one real load")
        transient_tries = 0
        for attempt in range(8):
            t = self.trial(cfg, "fit")
            if self.ensure_loaded(t):
                used = t.load.vram_self_mib if t.load else None
                if used is None or used <= self.budget or cfg.ngl < 999 or cfg.ctx <= self.o.min_ctx:
                    return cfg
                # Loaded, but over the budget (the projection missed something, e.g. the MTP draft context):
                # WDDM would page the overflow to system RAM and speed would collapse. Trim the context by the
                # excess, using the projection's own context cost per token.
                per_tok = (t.projection.device_context_mib / cfg.ctx) if t.projection and cfg.ctx else 0.02
                cut = int((used - self.budget) / max(per_tok, 1e-3) * 1.1)
                new = cfg.with_(ctx=round_ctx(max(self.o.min_ctx, cfg.ctx - cut)))
                self.emit("decision", f"{cfg.describe()} loaded but used {used} MiB, over the {self.budget} MiB "
                                      f"budget; trimming the context to {new.ctx:,}.")
                cfg = new
                continue
            if t.outcome in TRANSIENT and transient_tries < 2:
                # A slow dashboard, a context read too early, RAM briefly short: none of that says the config does
                # not fit, so the same config is tried again.
                transient_tries += 1
                self.emit("decision", f"{cfg.describe()} did not load ({t.outcome}); trying it again.")
                t.outcome, t.reason = "pending", ""
                continue
            if t.outcome not in ("oom", "crash", "spill"):
                raise _RunFailed(f"could not load {cfg.describe()}: {t.outcome} ({t.reason})")
            # back off: context for a GPU-resident model, a few layers for a split model
            if cfg.ngl >= 999 and cfg.ctx > self.o.min_ctx:
                new = cfg.with_(ctx=round_ctx(int(cfg.ctx * 0.85)))
            elif cfg.ngl < 999 and cfg.ngl > 0:
                new = cfg.with_(ngl=max(0, cfg.ngl - 2))
            else:
                raise _RunFailed(f"nothing fits: {t.reason}")
            self.fit_error += 256
            self.emit("decision", f"{cfg.describe()} failed to load ({t.outcome}); backing off to {new.describe()}.")
            cfg = new
        raise _RunFailed("could not find a config that loads after 8 attempts")

    def phase_baseline(self, fitted: Config) -> TrialRecord:
        self.emit("phase", "Measure the current settings (the baseline)")
        workloads = [w for w in self.o.weights if self.o.weights[w] > 0]
        bt = self.trial(self.baseline, "baseline")
        if self.baseline.key() != fitted.key() and not self.fits(self.baseline):
            # Loading an over-budget config on purpose would spill VRAM into system RAM (how the watchdog tripped
            # on 2026-09-30). Measure the fitted config as the baseline instead.
            pr = self.project(self.baseline)
            self.emit("note", f"Your current settings are projected at "
                              f"{(pr.device_self_mib + self.overhead(self.baseline)) if pr else '?'} MiB, over the "
                              f"{self.budget} MiB VRAM budget right now, so they were not loaded; the fitted config "
                              f"is the baseline instead.")
            bt.outcome, bt.reason = "skipped", "over the VRAM budget"
            ft = self.trial(fitted, "fit")
            self.measure(ft, workloads)
            self.ref = ft.medians()
            return ft
        if self.baseline.key() == fitted.key() or self.measure(bt, workloads):
            if self.baseline.key() == fitted.key():
                self.measure(bt, workloads)
            self.ref = bt.medians()
            return bt
        self.emit("note", f"The current settings did not load right now ({bt.outcome}); the fitted config is the "
                          f"baseline instead. (Other apps may be using more VRAM than when they were chosen.)")
        ft = self.trial(fitted, "fit")
        self.measure(ft, workloads)
        self.ref = ft.medians()
        return ft

    def phase_placement(self, inc: TrialRecord) -> TrialRecord:
        """Dense model split across GPU and CPU: try per-tensor placement against the layer split.

        A layer split leaves whole blocks on the CPU, so their attention and KV cache run there too, and that cost
        grows with context. Keeping every block's attention and KV cache on the GPU and moving only FFN weights to the
        CPU (`-ot`, the `n_cpu_ffn` knob) puts about the same number of weight bytes in system RAM, but the work that
        scales with context stays on the GPU. Which wins depends on the model and the context, so it is measured.
        """
        if self.p.is_moe or inc.cfg.ngl >= 999 or inc.cfg.ngl >= self.p.n_layer or inc.cfg.n_cpu_ffn:
            return inc
        self.emit("phase", "Tensor placement (FFN weights on the CPU, attention and KV cache on the GPU)")
        lo, hi = 1, self.p.n_layer
        probe = inc.cfg.with_(ngl=999, n_cpu_moe=None)
        if not self.fits(probe.with_(n_cpu_ffn=hi)):
            self.emit("decision", "Placement: even with every FFN on the CPU, attention and the KV cache would not "
                                  "fit on the GPU at this context; kept the layer split.")
            return inc
        while lo < hi:                      # fewest FFN blocks on the CPU that still fit
            mid = (lo + hi) // 2
            if self.fits(probe.with_(n_cpu_ffn=mid)):
                hi = mid
            else:
                lo = mid + 1
        cand = self.trial(probe.with_(n_cpu_ffn=lo), "placement")
        workloads = [w for w in self.ref]
        v = self.challenge(inc, cand, workloads, f"{inc.cfg.ngl} layers on GPU -> FFN of {lo} layers on CPU")
        if v is stats.Verdict.BETTER:
            self.emit("decision", f"Placement: FFN weights of {lo} of {self.p.n_layer} layers on the CPU, attention "
                                  f"and KV cache all on the GPU, beat {inc.cfg.ngl} whole layers on the GPU.")
            return cand
        self.emit("decision", f"Placement: kept {inc.cfg.ngl} whole layers on the GPU (FFN-only offload of {lo} "
                              f"layers was not clearly better).")
        return inc

    def phase_spec(self, inc: TrialRecord) -> TrialRecord:
        self.emit("phase", "Speculative decoding")
        options = ["none", "ngram-mod"] + (["draft-mtp", "draft-mtp,ngram-mod"] if self.p.has_mtp else [])
        if not self.p.has_mtp:
            self.emit("decision", "No MTP head in this file, so only n-gram speculation is tried.")
        workloads = [w for w in ("prose", "code") if w in self.ref]
        for spec in sorted(options, key=lambda s: SPEC_SIMPLICITY.get(s, 9)):
            if spec == inc.cfg.spec:
                continue
            cfg = inc.cfg.with_(spec=spec,
                                draft_n_max=(inc.cfg.draft_n_max or 2) if "draft-mtp" in spec else None,
                                ngram_n_match=(inc.cfg.ngram_n_match or 12) if "ngram" in spec else None)
            if not self.fits(cfg):
                self.emit("decision", f"Skipped {spec}: its draft context would not fit in VRAM.")
                continue
            cand = self.trial(cfg, "spec")
            v = self.challenge(inc, cand, workloads, f"spec {inc.cfg.spec} -> {spec}")
            if v is stats.Verdict.BETTER:
                inc = cand
            elif v is stats.Verdict.TIE and SPEC_SIMPLICITY.get(spec, 9) < SPEC_SIMPLICITY.get(inc.cfg.spec, 9):
                inc = cand
        self.emit("decision", f"Speculation: {inc.cfg.spec}.")
        return inc

    def search(self, inc: TrialRecord, label: str, values: list, field_name: str,
               workloads: list[str], phase: str, default=None) -> TrialRecord:
        """Direction search over ordered `values` for one field. An unset field (None) means llama-server's own
        default, so the search starts from `default` rather than from nowhere."""
        cur = getattr(inc.cfg, field_name)
        if cur is None and default is not None:
            cur = default
        if cur not in values:
            values = sorted(set(values) | {cur}, key=lambda v: (v is None, v))
        idx = values.index(cur)
        start = inc
        # Try first in the direction where a sibling model (same family) found its best value.
        directions = (+1, -1)
        prior = self.family_prior(field_name)
        if prior is not None and prior != cur and prior in values:
            if values.index(prior) < idx:
                directions = (-1, +1)
            self.emit("note", f"{label}: a model of the same family did best at {prior}; searching that way first.")
        for direction in directions:
            j = idx + direction
            moved = False
            while 0 <= j < len(values):
                cfg = inc.cfg.with_(**{field_name: values[j]})
                if not self.fits(cfg):
                    self.emit("decision", f"{label} {values[j]}: would not fit in VRAM; not going further this way.")
                    break
                cand = self.trial(cfg, phase)
                v = self.challenge(inc, cand, workloads, f"{label} {getattr(inc.cfg, field_name)} -> {values[j]}")
                if v is stats.Verdict.BETTER:
                    inc, idx, moved = cand, j, True
                    j += direction
                else:
                    break
            if moved:
                break           # found the good direction; the other side is not worth testing
        if inc is start:
            self.emit("decision", f"{label}: kept {cur}{' (the server default)' if getattr(start.cfg, field_name) is None else ''} "
                                  f"(neighbours were not clearly better).")
        else:
            was = f"unset (server default {cur})" if getattr(start.cfg, field_name) is None else str(cur)
            self.emit("decision", f"{label}: {was} -> {getattr(inc.cfg, field_name)}.")
        return inc

    def phase_spec_params(self, inc: TrialRecord) -> TrialRecord:
        workloads = [w for w in ("prose", "code") if w in self.ref]
        if inc.cfg.uses_mtp:
            self.emit("phase", "MTP draft length")
            inc = self.search(inc, "draft-n-max", [1, 2, 3, 4, 5], "draft_n_max", workloads, "spec-params",
                              default=3)
        if inc.cfg.uses_ngram:
            self.emit("phase", "N-gram match length")
            ng_workloads = ["code"] if "code" in self.ref else [w for w in ("prose",) if w in self.ref]
            if ng_workloads:
                inc = self.search(inc, "ngram match", [4, 6, 8, 12, 16, 24], "ngram_n_match", ng_workloads,
                                  "spec-params", default=24)
        return inc

    def phase_batch(self, inc: TrialRecord) -> TrialRecord:
        if "prefill" not in self.ref:
            return inc
        self.emit("phase", "Batch sizes (prompt speed)")
        # ubatch drives the compute buffer and prefill speed; batch must be >= ubatch
        values = [128, 256, 512, 1024]
        cur = inc.cfg.ubatch
        if cur not in values:
            values = sorted(set(values) | {cur})
        idx = values.index(cur)
        start = inc
        for direction in (+1, -1):
            j, moved = idx + direction, False
            while 0 <= j < len(values):
                ub = values[j]
                cfg = inc.cfg.with_(ubatch=ub, batch=max(inc.cfg.batch, ub))
                if not self.fits(cfg):
                    self.emit("decision", f"ubatch {ub}: its compute buffer would not fit at {cfg.ctx:,} context; "
                                          f"the context target wins.")
                    break
                cand = self.trial(cfg, "batch")
                v = self.challenge(inc, cand, ["prefill"], f"ubatch {inc.cfg.ubatch} -> {ub}")
                if v is stats.Verdict.BETTER:
                    inc, idx, moved = cand, j, True
                    j += direction
                else:
                    break
            if moved:
                break
        if inc is start:
            self.emit("decision", f"Batch: kept {cur} (neighbours were not clearly better).")
        else:
            self.emit("decision", f"Batch: ubatch {cur} -> {inc.cfg.ubatch}, batch {inc.cfg.batch}.")
        return inc

    def phase_threads(self, inc: TrialRecord, topo: tuple[int, int] | None) -> TrialRecord:
        offloaded = inc.cfg.ngl < 999 and inc.cfg.ngl < self.p.n_layer
        if not offloaded and not inc.cfg.n_cpu_moe and not inc.cfg.n_cpu_ffn:
            self.emit("decision", f"Threads: skipped. Every layer runs on the GPU, where CPU threads barely matter; "
                                  f"kept {inc.cfg.threads}.")
            return inc
        self.emit("phase", "CPU threads (layers run on the CPU)")
        perf, logical = topo or (8, 20)
        values = sorted({4, 6, perf, perf + 2, perf + 4, logical} & set(range(1, logical + 1)))
        workloads = [w for w in ("prose", "code") if w in self.ref]
        return self.search(inc, "threads", values, "threads", workloads, "threads")

    def phase_quality(self, inc: TrialRecord, base: TrialRecord) -> TrialRecord:
        if self.o.mode == "quick":
            return inc
        self.emit("phase", "Thinking length and quality gates")
        if self.p.template_has_thinking:
            inc = self._tune_thinking(inc)
        suites = ["reasoning", "tools"] + (["coding", "recall"] if self.o.mode == "full" else [])
        limit = self.quality_limit
        if limit is not None and _q4(inc.cfg) and not _q4(base.cfg):
            # A 4-bit KV cache is a known quality risk, and an 8-task screen cannot see a few points of loss (one
            # task is 12.5 points). Gate it on whole suites instead.
            limit = None
            self.emit("note", "The candidate uses a 4-bit KV cache, so the quality gates run the whole suites.")
        for suite in suites:
            if suite == "tools" and not self.p.template_supports_tools:
                continue
            if inc is base:
                break                       # nothing left to gate: the candidate is the baseline
            r_new = self._quality(inc, suite, limit)
            r_old = self._quality(base, suite, limit)
            if r_new is None or r_old is None:
                continue
            self.emit("compare", f"quality {suite}: {r_new.passed}/{r_new.total} vs baseline "
                                 f"{r_old.passed}/{r_old.total}")
            if not self._quality_dropped(r_new, r_old):
                continue
            # DESIGN: a config whose quality drops beyond noise is rejected however fast it is. First try the one
            # repair that keeps the speed gains (the baseline's cache type and thinking settings) and gate it again;
            # if that is impossible or still fails, the candidate is rejected and the baseline stays.
            cfg = inc.cfg.with_(cache_k=base.cfg.cache_k, cache_v=base.cfg.cache_v,
                                reasoning_budget=base.cfg.reasoning_budget,
                                reasoning_effort=base.cfg.reasoning_effort)
            repaired = None
            if cfg.key() != inc.cfg.key() and self.fits(cfg):
                rt = self.trial(cfg, "quality")
                r_rep = self._quality(rt, suite, limit)
                if r_rep is not None and not self._quality_dropped(r_rep, r_old) and self.measure(rt, list(self.ref)):
                    repaired = rt
            if repaired is not None:
                self.emit("decision", f"Quality gate: {suite} dropped ({r_new.passed}/{r_new.total} vs "
                                      f"{r_old.passed}/{r_old.total}); reverting cache type and thinking settings "
                                      f"to the baseline's fixed it ({repaired.cfg.describe()}).")
                inc = repaired
            else:
                self.emit("decision", f"Quality gate: {suite} dropped ({r_new.passed}/{r_new.total} vs "
                                      f"{r_old.passed}/{r_old.total}) and no repair kept the quality; the candidate "
                                      f"is rejected and your current settings stay.")
                inc = base
        return inc

    @property
    def quality_limit(self) -> int | None:
        return None if self.o.mode == "full" else 8

    @staticmethod
    def _quality_dropped(r_new: QualityResult, r_old: QualityResult) -> bool:
        """More than 10% of the suite (at least one task) worse, compared at the same suite size."""
        if r_new.total == r_old.total:
            drop = r_old.passed - r_new.passed
        else:   # long-context tasks skipped at a smaller context: compare rates on the old scale
            drop = round((r_old.rate - r_new.rate) * r_old.total)
        return drop > max(1, round(0.1 * r_old.total))

    def _quality(self, t: TrialRecord, suite: str, limit: int | None) -> QualityResult | None:
        memo = (t.cfg.key(), suite, limit)
        if memo in self._qmemo:
            return self._qmemo[memo]
        if not self.ensure_loaded(t):
            return None
        for attempt in range(2):
            key_before = self.b.power_key()
            try:
                r = self.b.quality(suite, limit=limit)
            except NotImplementedError as e:
                self.emit("note", f"Quality suite {suite} skipped: {e}")
                return None
            why = interruption(r, key_before, self.b.power_key())
            if not why or attempt == 1:
                if why:
                    self.emit("warn", f"{suite} was interrupted again ({why}); keeping the second result")
                break
            # A stall (the laptop slept, the power source changed) makes timings meaningless and can cut a task
            # short; the suite is cheap next to a wrong decision, so it runs once more.
            self.emit("warn", f"{suite} was interrupted ({why}); running it again")
            if not self.ensure_loaded(t):
                return None
        t.quality[suite] = r
        self._qmemo[memo] = r
        tid = self._hist_trial(t)
        if tid is not None:
            self._hist("add_quality", tid, r)
        self.emit("quality", f"{suite}: {r.passed}/{r.total} in {r.seconds:.0f}s, {r.tokens} tokens",
                  suite=suite, passed=r.passed, total=r.total, config=t.cfg.describe())
        return r

    def _tune_thinking(self, inc: TrialRecord) -> TrialRecord:
        """Pick the thinking cap (or effort) on quality, then tokens. It is not speed-checked on purpose: the speed
        workloads run with thinking disabled, so on llama.cpp these settings cannot change the measured decode or
        prefill speed; they only change how long the model thinks, which the token count already captures.
        (Declined finding 1 of the 2026-09-30 strategy review, which only reproduces in the simulator.)"""
        if self.p.template_honours_effort:
            options = [("reasoning_effort", v) for v in ("low", "medium", "high", "xhigh")]
            what = "reasoning effort"
        else:
            options = [("reasoning_budget", v) for v in (1024, 2048, 4096, 8192, None)]
            what = "thinking cap"
        results = []
        for fld, val in options:
            cfg = inc.cfg.with_(**{fld: val})
            t = self.trial(cfg, "thinking")
            r = self._quality(t, "reasoning", self.quality_limit)
            if r is not None:
                results.append((r.passed, -r.tokens, t, val))
        if not results:
            return inc
        best_pass = max(r[0] for r in results)
        # Among the settings with the best score, the one that spends the fewest tokens. Tokens, not seconds: they
        # measure thinking length directly and a stall (sleep, a power change) cannot distort them.
        pick = max((r for r in results if r[0] == best_pass), key=lambda r: r[1])
        self.emit("decision", f"{what.capitalize()}: {pick[3]} ({best_pass} passed using {-pick[1]:,} tokens for "
                              f"the reasoning set; the leanest setting with the best score).")
        return pick[2]

    def phase_confirm(self, best: TrialRecord, base: TrialRecord) -> tuple[dict[str, float], dict[str, float]]:
        """Fresh loads of the winner and of the baseline, back to back, so the headline number compares two
        measurements from the same stretch of time (clocks and background load drift over a long run)."""
        self.emit("phase", "Confirm the winner against the baseline with fresh loads")
        workloads = [w for w in self.ref]
        out = []
        for t, name in ((best, "winner"), (base, "baseline")):
            if name == "baseline" and base.cfg.key() == best.cfg.key():
                out.append(out[0])
                break
            fresh = TrialRecord(cfg=t.cfg, phase="confirm")
            self.loaded_key = None                  # force a reload
            self.order.append(fresh)
            if self.measure(fresh, workloads, reps=self.o.reps):
                out.append(fresh.medians())
            else:
                self.emit("warn", f"the {name} failed to reload for confirmation; using its earlier numbers")
                out.append(t.medians())
        return out[0], out[1]

    # ------------------------------------------------------------------ main
    def run(self, topo: tuple[int, int] | None = None) -> TuneResult:
        t0 = time.time()
        status = "ok"
        best: TrialRecord | None = None
        base: TrialRecord | None = None
        confirmed: dict[str, float] = {}
        confirmed_base: dict[str, float] = {}
        self.emit("note", f"Mode: {self.o.mode}. Power state: {self.power_key}.")
        try:
            fitted = self.phase_fit()
            fitted = self.phase_validate(fitted)
            base = self.phase_baseline(fitted)
            inc = self.trial(fitted, "fit")
            workloads = [w for w in self.ref]
            for _ in range(3):
                if self.measure(inc, workloads) or inc.cfg.ngl < 999 or inc.cfg.ctx <= self.o.min_ctx:
                    break
                # the fitted config loaded but died under load: give it 10% less context and validate again
                smaller = self.phase_validate(inc.cfg.with_(ctx=round_ctx(int(inc.cfg.ctx * 0.9))))
                self.emit("decision", f"{inc.cfg.describe()} crashed under load; backing off to {smaller.ctx:,} tokens.")
                inc = self.trial(smaller, "fit")
            if inc is not base:
                v = self.challenge(base, inc, workloads, "fitted placement vs current settings")
                if v is stats.Verdict.WORSE and base.outcome == "measured":
                    self.emit("decision", "The fitted placement was slower than the current settings' placement; "
                                          "continuing from the current settings.")
                    inc = base
            # A previous run's winner for this exact file gets a head-to-head before searching from scratch.
            prev = self._hist("best_for", self.fingerprint) if self.fingerprint else None
            if prev:
                pcfg = prev[0].with_(ctx=min(prev[0].ctx, inc.cfg.ctx)) if prev[0].ngl >= 999 else prev[0]
                if pcfg.key() not in (inc.cfg.key(), base.cfg.key()) and self.fits(pcfg):
                    pt = self.trial(pcfg, "previous-best")
                    if self.challenge(inc, pt, workloads, "previous run's best") is stats.Verdict.BETTER:
                        self.emit("decision", f"The previous run's best ({pcfg.describe()}) is still the leader; "
                                              f"searching from it.")
                        inc = pt
            best = inc
            best = inc = self.phase_placement(inc)
            best = inc = self.phase_spec(inc)
            best = inc = self.phase_spec_params(inc)
            best = inc = self.phase_batch(inc)
            best = inc = self.phase_threads(inc, topo)
            best = inc = self.phase_quality(inc, base)
            confirmed, confirmed_base = self.phase_confirm(best, base)
        except _Budget:
            # Not "ok": whatever phase was cut short (a quality gate, say) did not finish, so this result must not be
            # recorded as the model's best.
            status = "budget"
            self.emit("note", f"Load budget ({self.o.max_loads}) reached; stopping with the best so far, which did "
                              f"not finish every phase.")
        except _RunFailed as e:
            status = "failed"
            self.emit("note", f"Could not continue: {e}")
        except Stopped:
            status = "stopped"
            self.emit("note", "Stopped on request.")
        except PowerChanged as e:
            status = "power_changed"
            self.emit("note", f"Stopped: {e}. Results are not comparable; run again in one power state.")
        best = best or base
        base_metrics = confirmed_base or (base.medians() if base else {})
        best_metrics = confirmed or (best.medians() if best else {})
        imp = stats.relative_score(best_metrics, base_metrics, self.o.weights) if best and base else 0.0
        if best is None:
            status = "failed" if status == "ok" else status
        return TuneResult(
            status=status, baseline=self.baseline, best=(best.cfg if best else self.baseline),
            baseline_metrics=base_metrics, best_metrics=best_metrics, improvement=imp,
            decisions=self.decisions, notes=self.notes, trials=self.order, loads=self.loads,
            seconds=time.time() - t0,
            quality={s: {"best": (best.quality[s].passed, best.quality[s].total) if best and s in best.quality else None,
                         "baseline": (base.quality[s].passed, base.quality[s].total) if base and s in base.quality else None}
                     for s in {*(best.quality if best else {}), *(base.quality if base else {})}},
        )


class _Budget(Exception):
    pass


class _RunFailed(Exception):
    """Nothing loadable was found (or loads kept failing): end the run with status "failed" and what it had."""


TRANSIENT = ("timeout", "refused", "verify", "ram")


def _q4(cfg: Config) -> bool:
    return any(c.startswith(("q4", "iq4")) for c in (cfg.cache_k, cfg.cache_v))


def interruption(r: QualityResult, key_before: str, key_after: str) -> str | None:
    """Why a quality run's timings can't be trusted, or None. A task far slower per token than the suite's median
    means the machine stalled mid-task (Modern Standby froze a reasoning task for 17 minutes on 2026-09-30)."""
    if key_before != key_after:
        return f"power state changed from {key_before} to {key_after}"
    rates = [d["tokens"] / d["seconds"] for d in r.details
             if d.get("tokens") and d.get("seconds") and d["seconds"] > 0]
    if len(rates) >= 3:
        mid = sorted(rates)[len(rates) // 2]
        slow = [x for x in rates if x < 0.25 * mid]
        if slow:
            return f"a task ran at {min(slow):.1f} tok/s against a typical {mid:.0f} tok/s (the machine stalled)"
    return None
