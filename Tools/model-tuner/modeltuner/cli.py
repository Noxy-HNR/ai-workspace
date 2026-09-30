"""Command line: python -m modeltuner <command> ...

  profile [MODEL]                 what the tuner knows about each model file (no loading)
  plan MODEL [--ctx N]            decide placement, context and cache type on paper (no loading)
  run MODEL [--mode quick|medium|full] [--detach]
  status                          the live or last run
  stop                            ask the live run to stop (it restores your model first)
  report [RUN_ID]                 print a run's report (default: the latest)
  apply RUN_ID [--yes] [--switchboard]  write a run's settings to models.json (asks first), and optionally
                                  its measured speeds to Switchboard's models.toml
  runs                            list past runs
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from modeltuner import hardware, paths, safety


def _latest_run() -> Path | None:
    runs = sorted(paths.RUNS_DIR.glob("*/report.json"), key=lambda p: p.stat().st_mtime)
    return runs[-1].parent if runs else None


def cmd_profile(a) -> int:
    from modeltuner.profile import build_profile, list_models
    for m in ([a.model] if a.model else list_models()):
        p = build_profile(m)
        print("\n".join(p.summary_lines()))
        print(f"  current settings: {p.current_config().describe()}\n")
    return 0


def cmd_plan(a) -> int:
    """Fit phase only, with a budget estimated from what is running now (no loads, nothing stopped)."""
    from modeltuner.fitparams import FitParams
    from modeltuner.llama import LlamaClient
    from modeltuner.profile import build_profile, saved_settings
    from modeltuner.knobs import from_settings
    from modeltuner.real import VRAM_MARGIN_MIB
    from modeltuner.strategy import Tuner, TuneOptions

    p = build_profile(a.model)
    gpu = hardware.gpu()
    other = gpu.used_mib if gpu else 1500
    client = LlamaClient()
    running = client.loaded_model()
    if running:
        # subtract the running model's own projected use (at the context it is ACTUALLY serving, which a trial load
        # may have changed) to estimate what other apps hold. Windows keeps at least a few hundred MiB for itself.
        rp = build_profile(running)
        rs = saved_settings(running)
        live = (from_settings(rs) if rs else rp.current_config()).with_(ctx=client.n_ctx() or rp.current_config().ctx)
        try:
            proj = FitParams(Path(rp.path), rp.server_binary).project(live)
            other = max(400, other - proj.device_self_mib)
        except Exception as e:  # noqa: BLE001
            print(f"(could not project the running model {running}: {e})")

    class PlanBackend:
        model_file = p.file
        def __init__(self): self.fp = FitParams(Path(p.path), p.server_binary)
        def project(self, cfg): return self.fp.project(cfg)
        def vram_budget_mib(self): return (gpu.total_mib if gpu else 12227) - other - VRAM_MARGIN_MIB
        def power_key(self): return hardware.power().key()
        def should_stop(self): return False

    print("\n".join(p.summary_lines()))
    print(f"\nOther apps are using about {other} MiB of VRAM right now"
          f"{' (estimated: ' + running + ' is loaded)' if running else ''}.")
    t = Tuner(PlanBackend(), p, p.current_config(), TuneOptions(ctx_target=a.ctx, allow_q4=a.allow_q4),
              emit=lambda kind, msg, **d: print(f"  {kind:9} {msg}") if kind in ("decision", "note", "warn") else None)
    t0 = time.time()
    cfg = t.phase_fit()
    proj = PlanBackend().project(cfg)
    print(f"\nPlan: {cfg.describe()}")
    print(f"Projected VRAM: {proj.device_self_mib} MiB (model {proj.device_model_mib}, context "
          f"{proj.device_context_mib}, compute {proj.device_compute_mib}) + {t.overhead(cfg):.0f} MiB learned overhead; "
          f"host RAM {proj.host_self_mib} MiB. {PlanBackend().fp.calls if False else ''}")
    print(f"Current: {p.current_config().describe()}")
    print(f"({time.time() - t0:.0f}s of projections, nothing loaded)")
    return 0


def cmd_run(a) -> int:
    from modeltuner.strategy import TuneOptions
    if a.detach:
        if hardware.power().on_ac is False and not a.allow_battery:
            print("SAFETY: the laptop is on battery; plug it in or pass --allow-battery", file=sys.stderr)
            return 2
        args = [sys.executable, "-m", "modeltuner", "run", a.model, "--mode", a.mode]
        if a.ctx:
            args += ["--ctx", str(a.ctx)]
        if a.allow_q4:
            args += ["--allow-q4"]
        if a.no_advisor:
            args += ["--no-advisor"]
        if a.allow_battery:
            args += ["--allow-battery"]
        paths.ensure_state_dirs()
        log = (paths.STATE_DIR / "last-run.log").open("w", encoding="utf-8")
        flags = 0x00000008 | 0x00000200 | getattr(subprocess, "CREATE_NO_WINDOW", 0)  # DETACHED | NEW_GROUP
        try:
            p = subprocess.Popen(args, cwd=str(paths.PROJECT_ROOT), stdout=log, stderr=subprocess.STDOUT,
                                 creationflags=flags | 0x01000000)  # + BREAKAWAY_FROM_JOB
        except OSError:
            p = subprocess.Popen(args, cwd=str(paths.PROJECT_ROOT), stdout=log, stderr=subprocess.STDOUT,
                                 creationflags=flags)
        print(json.dumps({"started": True, "pid": p.pid, "log": str(paths.STATE_DIR / "last-run.log")}))
        return 0
    from modeltuner import runner
    opts = TuneOptions(mode=a.mode, ctx_target=a.ctx, allow_q4=a.allow_q4, reps=a.reps)
    data = runner.run(a.model, opts, use_advisor=not a.no_advisor, allow_battery=a.allow_battery)
    print(f"\nReport: {paths.RUNS_DIR / data['run_id'] / 'report.md'}")
    return 0


def cmd_status(a) -> int:
    if paths.CURRENT_RUN.exists():
        print(paths.CURRENT_RUN.read_text(encoding="utf-8"))
    else:
        print(json.dumps({"status": "idle"}))
    return 0


def cmd_stop(a) -> int:
    paths.ensure_state_dirs()
    paths.STOP_FILE.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    print("Stop requested. The run finishes its current step, restores your model, and writes its report.")
    return 0


def cmd_report(a) -> int:
    d = (paths.RUNS_DIR / a.run_id) if a.run_id else _latest_run()
    if not d or not (d / "report.md").exists():
        print("no report found")
        return 1
    print((d / "report.md").read_text(encoding="utf-8"))
    return 0


def cmd_apply(a) -> int:
    from modeltuner.dashboard import Dashboard
    d = paths.RUNS_DIR / a.run_id
    data = json.loads((d / "report.json").read_text(encoding="utf-8"))
    changes = data.get("changes") or {}
    if changes:
        print(f"Proposed changes for {data['model']}:")
        for k, (old, new) in changes.items():
            print(f"  {k}: {old!r} -> {new!r}")
        if not a.yes and input("Write these to models.json? [y/N] ").strip().lower() != "y":
            print("Not applied.")
            return 1
        resp = Dashboard().save_settings(data["model"], data["proposed_settings"])
        if resp.get("success") is False:
            print(f"The dashboard refused: {resp.get('message')}")
            return 1
        (d / "applied.json").write_text(json.dumps({"applied": time.strftime("%Y-%m-%d %H:%M:%S")}), encoding="utf-8")
        print("Applied to models.json. The new settings take effect the next time this model loads.")
    else:
        print("Nothing to change in models.json: the current settings were already the best.")
    if a.switchboard:
        from modeltuner import switchboard_sync
        try:
            fields = switchboard_sync.apply_report(data)
        except KeyError as e:
            print(f"Switchboard: {e}")
            return 1
        if fields:
            print("Switchboard models.toml measured speeds updated: "
                  + ", ".join(f"{k}={v}" for k, v in fields.items() if not k.endswith("_source")))
    return 0


def cmd_runs(a) -> int:
    for d in sorted(paths.RUNS_DIR.glob("*")):
        rj = d / "report.json"
        if rj.exists():
            r = json.loads(rj.read_text(encoding="utf-8"))
            print(f"{d.name:48} {r['status']:14} {r.get('improvement', 0) * 100:+6.1f}%  {r['mode']}")
        else:
            st = d / "state.json"
            s = json.loads(st.read_text(encoding="utf-8")) if st.exists() else {}
            print(f"{d.name:48} {s.get('status', '?'):14}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="modeltuner", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("profile"); s.add_argument("model", nargs="?"); s.set_defaults(fn=cmd_profile)
    s = sub.add_parser("plan"); s.add_argument("model"); s.add_argument("--ctx", type=int)
    s.add_argument("--allow-q4", action="store_true"); s.set_defaults(fn=cmd_plan)
    s = sub.add_parser("run"); s.add_argument("model"); s.add_argument("--mode", choices=["quick", "medium", "full"], default="quick")
    s.add_argument("--ctx", type=int); s.add_argument("--allow-q4", action="store_true"); s.add_argument("--reps", type=int, default=3)
    s.add_argument("--detach", action="store_true")
    s.add_argument("--no-advisor", action="store_true", help="never ask Jev about unexplained failures")
    s.add_argument("--allow-battery", action="store_true", help="tune even when the laptop is on battery")
    s.set_defaults(fn=cmd_run)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    sub.add_parser("stop").set_defaults(fn=cmd_stop)
    s = sub.add_parser("report"); s.add_argument("run_id", nargs="?"); s.set_defaults(fn=cmd_report)
    s = sub.add_parser("apply"); s.add_argument("run_id"); s.add_argument("--yes", action="store_true")
    s.add_argument("--switchboard", action="store_true", help="also write the measured speeds to Switchboard's models.toml")
    s.set_defaults(fn=cmd_apply)
    sub.add_parser("runs").set_defaults(fn=cmd_runs)
    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except safety.SafetyError as e:
        print(f"SAFETY: {e}", file=sys.stderr)
        if a.cmd == "run":
            # Tell the dashboard why nothing started (it reads current.json).
            paths.ensure_state_dirs()
            paths.CURRENT_RUN.write_text(json.dumps({"status": "refused", "model": getattr(a, "model", ""),
                                                     "mode": getattr(a, "mode", ""), "phase": "",
                                                     "message": str(e), "updated": time.time()}, indent=2),
                                         encoding="utf-8")
        return 2
