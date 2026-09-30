"""Runs one tuning job end to end: safety checks, events and state files, the strategy, restore, report."""

from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

from modeltuner import hardware, paths, report, safety
from modeltuner.profile import build_profile
from modeltuner.real import RealBackend
from modeltuner.strategy import Tuner, TuneOptions


def new_run_id(model_file: str) -> str:
    stem = "".join(c for c in Path(model_file).stem if c.isalnum() or c in "-_.")[:32]
    return time.strftime("%Y%m%d-%H%M%S-") + stem


class RunState:
    """events.jsonl (append-only log) + state.json and current.json (latest status, read by the dashboard)."""

    def __init__(self, run_id: str, model_file: str, mode: str):
        self.run_id = run_id
        self.dir = paths.RUNS_DIR / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.events = self.dir / "events.jsonl"
        self.state = {"run_id": run_id, "model": model_file, "mode": mode, "status": "starting", "phase": "",
                      "message": "", "pid": os.getpid(), "started": time.time(), "updated": time.time(),
                      "loads": 0, "decisions": 0}

    def emit(self, kind: str, msg: str, **data) -> None:
        ev = {"t": time.time(), "kind": kind, "msg": msg, **{k: v for k, v in data.items() if _jsonable(v)}}
        with self.events.open("a", encoding="utf-8") as f:
            f.write(json.dumps(ev, default=str) + "\n")
        if kind == "phase":
            self.state["phase"] = msg
        if kind == "decision":
            self.state["decisions"] += 1
        if kind == "load":
            self.state["loads"] += 1
        self.state["message"] = msg
        self.save()
        print(f"[{time.strftime('%H:%M:%S')}] {kind:11} {msg}", flush=True)

    def save(self, **kw) -> None:
        self.state.update(kw)
        self.state["updated"] = time.time()
        text = json.dumps(self.state, indent=2, default=str)
        (self.dir / "state.json").write_text(text, encoding="utf-8")
        paths.CURRENT_RUN.write_text(text, encoding="utf-8")


def _jsonable(v) -> bool:
    try:
        json.dumps(v)
        return True
    except (TypeError, ValueError):
        return False


def run(model_file: str, opts: TuneOptions, run_id: str | None = None, use_advisor: bool = True,
        allow_battery: bool = False) -> dict:
    paths.ensure_state_dirs()
    safety.refuse_if_tripped()
    power = hardware.power()
    if power.on_ac is False and not allow_battery:
        # On battery Windows changes the power mode by itself, so measurements drift mid-run and the results would
        # not describe how the model runs when plugged in.
        raise safety.SafetyError("the laptop is on battery; plug it in (or pass --allow-battery to tune for battery "
                                 "use, knowing Windows may change the power mode mid-run)")
    safety.clear_stop()
    run_id = run_id or new_run_id(model_file)
    st = RunState(run_id, model_file, opts.mode)
    profile = build_profile(model_file)
    for line in profile.summary_lines():
        st.emit("profile", line)
    backend = RealBackend(profile, log=lambda m: st.emit("info", m), run_dir=st.dir, use_advisor=use_advisor)
    topo = hardware.cpu()
    meta = {"mode": opts.mode, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "power_key": hardware.power().key()}

    def on_trip(avail: float) -> None:
        st.save(status="watchdog", message=f"RAM watchdog tripped at {avail:.2f} GB available; llama-server stopped")
        try:
            backend.dash.stop_llama()
        except Exception:
            pass

    dog = safety.Watchdog(on_trip).start()
    result = None
    data: dict = {}
    hist, fp = _open_history(st, profile, run_id, opts, meta["power_key"])
    st.save(status="running")
    try:
        backend.begin()
        tuner = Tuner(backend, profile, profile.current_config(), opts, emit=st.emit, history=hist,
                      fingerprint=fp, run_id=run_id)
        result = tuner.run(topo=(topo.performance_cores or 8, topo.logical))
    except safety.SafetyError as e:
        st.emit("error", str(e))
        st.save(status="failed", message=str(e))
        raise
    except Exception as e:
        st.emit("error", f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-2000:])
        st.save(status="failed", message=str(e))
        raise
    finally:
        st.emit("phase", "Restoring your model and settings")
        backend.restore()
        dog.stop()
        if result is not None:
            meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
            data = report.write(st.dir, run_id, profile, result, meta)
            st.save(status=result.status, message=f"done: {result.improvement * 100:+.1f}% overall",
                    report=str(st.dir / "report.md"), improvement=result.improvement)
        if hist is not None:
            try:
                hist.finish_run(run_id, result.status if result else "failed", result.best if result else None,
                                {"improvement": result.improvement if result else None,
                                 "best_metrics": result.best_metrics if result else {}})
                if result is not None and result.status == "ok" and result.best_metrics:
                    hist.set_best(fp, profile.file, profile.family, result.best, result.best_metrics,
                                  meta["power_key"], run_id)
            except Exception as e:  # noqa: BLE001 -- history must never break a run
                st.emit("warn", f"history: {e}")
            hist.close()
    return data


def _open_history(st: RunState, profile, run_id: str, opts: TuneOptions, power_key: str):
    """The SQLite memory (priors, calibration, known failures). A problem here only costs the priors."""
    try:
        from modeltuner.history import History, fingerprint
        fp = fingerprint(Path(profile.path))
        hist = History()
        hist.start_run(run_id, profile.file, fp, profile.family, opts.mode, power_key, profile.current_config())
        return hist, fp
    except Exception as e:  # noqa: BLE001
        st.emit("warn", f"history unavailable ({type(e).__name__}: {e}); running without priors")
        return None, None
