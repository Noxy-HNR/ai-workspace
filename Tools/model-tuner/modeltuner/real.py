"""The real Backend: llama-fit-params for projections, the dashboard for loads, llama-server for measurements."""

from __future__ import annotations

import importlib
import time
from pathlib import Path

from modeltuner import advisor, bench, hardware, paths, safety
from modeltuner.backend import LoadFailed, LoadResult, QualityResult, Sample
from modeltuner.dashboard import (Dashboard, DashboardError, classify_failure, llama_log_tail, llama_pid,
                                  process_alive)
from modeltuner.fitparams import FitParams, Projection
from modeltuner.knobs import Config, to_settings
from modeltuner.llama import LlamaClient, LlamaError
from modeltuner.profile import ModelProfile

VRAM_MARGIN_MIB = 1000      # headroom for peaks, the driver and other apps. 600 was not enough: OxCoder at 237K
                            # context left ~780 MiB free, a long prompt spilled VRAM into system RAM (WDDM shared
                            # memory) and available RAM fell to 0.13 GB before the watchdog stopped it.
SPILL_RAM_GB = 1.4          # soft floor during measurements (the hard watchdog is 0.8 GB)


class RealBackend:
    def __init__(self, profile: ModelProfile, *, dash: Dashboard | None = None, log=print,
                 run_dir: Path | None = None, use_advisor: bool = True):
        self.profile = profile
        self.model_file = profile.file
        self.dash = dash or Dashboard()
        self.client: LlamaClient = self.dash.llama
        self.fit = FitParams(Path(profile.path), profile.server_binary)
        self.log = log
        self._base_settings = dict(profile.current)      # serverBinary, loraFile, rope... carried into trials
        self._other_mib: int | None = None
        self._loaded: Config | None = None
        self._original_model: str | None = None
        self._original_running = False
        self.loads = 0
        self.run_dir = run_dir
        self.crashes = 0
        self._counts: dict[tuple[str, str], int] = {}   # (config key, workload) -> reps taken
        self.use_advisor = use_advisor and advisor.available()

    # ---- lifecycle ----
    def begin(self) -> None:
        """Remember what the user was running, then measure VRAM used by everything else with llama-server stopped."""
        self._original_model = self.client.loaded_model()
        self._original_running = self._original_model is not None
        old = llama_pid()
        self.dash.stop_llama()
        t0 = time.time()
        while old and process_alive(old) and time.time() - t0 < 120:
            time.sleep(1)       # never start anything while the old server is still exiting
        time.sleep(2)
        gpu = hardware.gpu()
        self._other_mib = gpu.used_mib if gpu else None
        self.log(f"VRAM used by other apps with llama-server stopped: {self._other_mib} MiB")

    def restore(self) -> None:
        """Put the user's model back with its saved settings (or leave llama-server stopped if it was stopped)."""
        try:
            if self._original_running and self._original_model:
                self.log(f"restoring {self._original_model} with its saved settings")
                old = llama_pid()
                config = self.dash.models_config()
                saved = config.get("models", {}).get(self._original_model, {})
                if config.get("activeModel") == self._original_model:
                    self.dash.load_saved(self._original_model)
                else:
                    # It was only trial-loaded (not the active model): a normal load would rewrite models.json
                    # and make it active, so bring it back the same way, without persisting anything.
                    self.dash.trial_load(self._original_model, dict(saved))
                from modeltuner.dashboard import wait_for_model
                wait_for_model(self.dash, self._original_model, int(saved.get("contextSize") or 0) or -1, old)
            else:
                self.dash.stop_llama()
        except DashboardError as e:
            self.log(f"restore problem: {e}")

    # ---- failure handling ----
    def _save_crash_log(self) -> str:
        """Keep llama-server's log now: the next load (or the restore) overwrites it."""
        self.crashes += 1
        tail = llama_log_tail(80)
        if self.run_dir:
            (self.run_dir / f"crash-{self.crashes}.log").write_text(tail, encoding="utf-8")
        return tail

    def _second_opinion(self, kind: str, msg: str, tail: str, cfg: Config, proj: Projection) -> tuple[str, str]:
        """Rules first; when they can only say 'crash', ask Jev what the log and numbers most likely mean."""
        if kind != "crash" or not self.use_advisor:
            return kind, msg
        gpu = hardware.gpu()
        d = advisor.diagnose_failure(tail, {
            "vram_total_mib": gpu.total_mib if gpu else None, "vram_other_apps_mib": self._other_mib,
            "projected_device_mib": proj.device_self_mib, "projected_host_mib": proj.host_self_mib,
            "available_ram_gb": round(hardware.memory().avail_gb, 2), "ctx": cfg.ctx, "gpu_layers": cfg.ngl,
            "cache_type": cfg.cache_k, "spec": cfg.spec, "ubatch": cfg.ubatch, "model_arch": self.profile.arch})
        if d is None:
            return kind, msg
        self.log(f"{d.describe()} (treated as '{d.kind}')")
        return (d.kind, f"{msg} [{d.describe()}]") if d.probability >= 0.6 else (kind, f"{msg} [{d.describe()}]")

    # ---- Backend protocol ----
    def project(self, cfg: Config) -> Projection:
        return self.fit.project(cfg)

    def vram_budget_mib(self) -> int:
        gpu = hardware.gpu()
        total = gpu.total_mib if gpu else 12227
        other = self._other_mib if self._other_mib is not None else (gpu.used_mib if gpu else 1500)
        return total - other - VRAM_MARGIN_MIB

    def load(self, cfg: Config) -> LoadResult:
        proj = self.project(cfg)
        try:
            safety.preflight(proj.host_self_mib)
        except safety.SafetyError as e:
            raise LoadFailed("ram", str(e)) from e
        settings = to_settings(cfg, self._base_settings)
        old = llama_pid()
        try:
            self.dash.trial_load(self.model_file, settings)
            from modeltuner.dashboard import wait_for_model
            pid, secs = wait_for_model(self.dash, self.model_file, cfg.ctx, old)
        except DashboardError as e:
            msg = str(e)
            kind = msg.split(":", 1)[0] if ":" in msg else "crash"
            if kind not in ("oom", "crash", "timeout", "verify"):
                kind = "refused"
            self._loaded = None
            if kind in ("oom", "crash"):
                tail = self._save_crash_log()
                kind, msg = self._second_opinion(kind, msg, tail, cfg, proj)
            raise LoadFailed(kind, msg) from e
        self.loads += 1
        self._loaded = cfg
        try:
            bench.warm_up(self.client)
        except Exception as e:  # noqa: BLE001 -- a slow first answer is not a load failure
            self.log(f"warm-up problem (ignored): {e}")
        gpu = hardware.gpu()
        return LoadResult(seconds=secs, vram_used_mib=gpu.used_mib if gpu else None, vram_other_mib=self._other_mib,
                          host_avail_gb=hardware.memory().avail_gb, pid=pid, n_ctx=cfg.ctx)

    def measure(self, workload: str, *, reps: int) -> list[Sample]:
        if self._loaded is None:
            raise RuntimeError("measure() before a successful load()")
        k = (self._loaded.key(), workload)
        start = self._counts.get(k, 0)
        guard = _SpillGuard(self.dash).start()
        try:
            with safety.LlamaLock():
                out = bench.measure(self.client, workload, reps, self.power_key, start=start)
        except LlamaError as e:
            if guard.fired:
                self._loaded = None
                self.log(f"available RAM fell to {guard.lowest:.2f} GB during '{workload}': VRAM is spilling into "
                         f"system RAM, so this config is too big; llama-server was stopped early")
                raise LoadFailed("spill", f"VRAM spilled into system RAM during {workload} "
                                          f"(available RAM {guard.lowest:.2f} GB)") from e
            if self.client.ready(timeout=5):
                raise                       # the server is fine; a genuine request problem
            # The server died mid-measurement (typically a VRAM spike). Keep its log before any reload overwrites
            # it, and report it as this config failing so the search can carry on without it.
            tail = self._save_crash_log()
            loaded, self._loaded = self._loaded, None
            kind, msg = self._second_opinion(classify_failure(tail), f"server died during {workload}: {e}", tail,
                                             loaded, self.project(loaded))
            self.log(f"llama-server died during '{workload}' ({kind}); log saved as crash-{self.crashes}.log")
            raise LoadFailed(kind, msg) from e
        finally:
            guard.stop()
        if guard.fired:     # the stop landed between requests
            self._loaded = None
            raise LoadFailed("spill", f"VRAM spilled into system RAM during {workload} "
                                      f"(available RAM {guard.lowest:.2f} GB)")
        self._counts[k] = start + len(out)
        return out

    def quality(self, suite: str, *, limit: int | None = None) -> QualityResult:
        try:
            mod = importlib.import_module(f"modeltuner.quality.{suite}")
        except ImportError as e:
            raise NotImplementedError(f"quality suite {suite!r} is not installed") from e
        with safety.LlamaLock():
            return mod.run(self.client, limit=limit, progress=lambda line: self.log(f"  [{suite}] {line}"))

    def power_key(self) -> str:
        return hardware.power().key()

    def should_stop(self) -> bool:
        return safety.stop_requested()


class _SpillGuard:
    """Watches available RAM during a measurement. When a config overcommits VRAM, Windows backs the overflow with
    system RAM and available RAM collapses within seconds; this stops llama-server at a soft floor, well before the
    hard watchdog (0.8 GB) has to, and lets the strategy treat the config as too big."""

    def __init__(self, dash: Dashboard, floor_gb: float = SPILL_RAM_GB, seconds: float = 2.0):
        import threading
        self.dash = dash
        self.floor_gb = floor_gb
        self.seconds = seconds
        self.fired = False
        self.lowest = float("inf")
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, name="spill-guard", daemon=True)

    def start(self) -> "_SpillGuard":
        self._t.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._t.join(timeout=2)

    def _loop(self) -> None:
        low_since = None
        while not self._stop.is_set():
            avail = hardware.memory().avail_gb
            self.lowest = min(self.lowest, avail)
            if avail < self.floor_gb:
                low_since = low_since or time.monotonic()
                if time.monotonic() - low_since >= self.seconds:
                    self.fired = True
                    try:
                        self.dash.stop_llama()
                    except DashboardError:
                        pass
                    return
            else:
                low_since = None
            self._stop.wait(0.5)
