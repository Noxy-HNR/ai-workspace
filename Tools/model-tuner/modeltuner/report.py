"""Run reports: Markdown for people, JSON for the dashboard and for `apply`."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from modeltuner.knobs import Config, diff, to_settings
from modeltuner.profile import ModelProfile
from modeltuner.strategy import METRIC, TuneResult

WORKLOAD_LABEL = {"prose": "Decode, prose", "code": "Decode, code edit", "prefill": "Prompt processing"}


def _pct(new: float | None, old: float | None) -> str:
    if not new or not old:
        return "n/a"
    return f"{(new / old - 1) * 100:+.1f}%"


def proposed_settings(profile: ModelProfile, cfg: Config) -> dict[str, Any]:
    return to_settings(cfg, profile.current)


def settings_changes(profile: ModelProfile, cfg: Config) -> dict[str, tuple[Any, Any]]:
    new = proposed_settings(profile, cfg)
    return {k: (profile.current.get(k), v) for k, v in new.items() if profile.current.get(k) != v}


def to_json(run_id: str, profile: ModelProfile, res: TuneResult, meta: dict) -> dict:
    return {
        "run_id": run_id,
        "model": profile.file,
        "status": res.status,
        "mode": meta.get("mode"),
        "power_key": meta.get("power_key"),
        "started": meta.get("started"),
        "finished": meta.get("finished"),
        "seconds": round(res.seconds),
        "loads": res.loads,
        "improvement": res.improvement,
        "baseline": asdict(res.baseline),
        "best": asdict(res.best),
        "baseline_metrics": res.baseline_metrics,
        "best_metrics": res.best_metrics,
        "proposed_settings": proposed_settings(profile, res.best),
        "changes": {k: list(v) for k, v in settings_changes(profile, res.best).items()},
        "decisions": res.decisions,
        "notes": res.notes,
        "quality": res.quality,
        "trials": [
            {"phase": t.phase, "config": t.cfg.describe(), "outcome": t.outcome, "reason": t.reason[:300],
             "medians": t.medians(),
             "vram_self_mib": t.load.vram_self_mib if t.load else None,
             "projected_mib": t.projection.device_self_mib if t.projection else None,
             "load_s": round(t.load.seconds, 1) if t.load else None,
             "quality": {s: [q.passed, q.total] for s, q in t.quality.items()}}
            for t in res.trials
        ],
    }


def to_markdown(run_id: str, profile: ModelProfile, res: TuneResult, meta: dict) -> str:
    L: list[str] = []
    L.append(f"# Model Tuner report: {profile.file}")
    L.append("")
    L.append(f"Run `{run_id}`, mode **{meta.get('mode')}**, status **{res.status}**, {res.loads} loads, "
             f"{res.seconds / 60:.1f} minutes, power state `{meta.get('power_key')}`.")
    L.append("")
    L.append("## Result")
    L.append("")
    if res.best.key() == res.baseline.key():
        L.append("Your current settings are already the best found. Nothing to change.")
    else:
        L.append(f"Overall speed change versus your current settings: **{res.improvement * 100:+.1f}%** "
                 f"(weighted across the workloads below).")
    L.append("")
    L.append("| Workload | Current | Tuned | Change |")
    L.append("|---|---|---|---|")
    for w in ("prose", "code", "prefill"):
        a, b = res.baseline_metrics.get(w), res.best_metrics.get(w)
        if a or b:
            L.append(f"| {WORKLOAD_LABEL[w]} | {a or 0:.1f} tok/s | {b or 0:.1f} tok/s | {_pct(b, a)} |")
    if res.quality:
        L.append("")
        L.append("| Quality suite | Current | Tuned |")
        L.append("|---|---|---|")
        for s, q in sorted(res.quality.items()):
            fmt = lambda x: f"{x[0]}/{x[1]}" if x else "not run"  # noqa: E731
            L.append(f"| {s} | {fmt(q.get('baseline'))} | {fmt(q.get('best'))} |")
    L.append("")
    L.append(f"**Current:** {res.baseline.describe()}  ")
    L.append(f"**Tuned:** {res.best.describe()}")
    ch = settings_changes(profile, res.best)
    if ch:
        L.append("")
        L.append("## Proposed changes to models.json")
        L.append("")
        L.append("| Setting | Current | Proposed |")
        L.append("|---|---|---|")
        for k, (old, new) in ch.items():
            L.append(f"| {k} | `{old}` | `{new}` |")
        L.append("")
        L.append(f"Apply with: `python -m modeltuner apply {run_id}` (asks before writing), or the dashboard's "
                 f"Apply button.")
    L.append("")
    L.append("## What the tuner knew before loading anything")
    L.append("")
    for line in profile.summary_lines():
        L.append(f"- {line}")
    L.append("")
    L.append("## Decisions, in order")
    L.append("")
    for d in res.decisions:
        L.append(f"1. {d}")
    if res.notes:
        L.append("")
        L.append("## Notes")
        L.append("")
        for n in res.notes:
            L.append(f"- {n}")
    L.append("")
    L.append("## Every configuration tried")
    L.append("")
    L.append("| Phase | Configuration | Outcome | Prose | Code | Prefill | VRAM (proj. / actual MiB) |")
    L.append("|---|---|---|---|---|---|---|")
    for t in res.trials:
        m = t.medians()
        vr = f"{t.projection.device_self_mib if t.projection else '-'} / {t.load.vram_self_mib if t.load and t.load.vram_self_mib else '-'}"
        L.append(f"| {t.phase} | {t.cfg.describe()} | {t.outcome} | "
                 + " | ".join(f"{m[w]:.1f}" if w in m else "-" for w in ("prose", "code", "prefill"))
                 + f" | {vr} |")
    L.append("")
    return "\n".join(L)


def write(run_dir, run_id: str, profile: ModelProfile, res: TuneResult, meta: dict) -> dict:
    data = to_json(run_id, profile, res, meta)
    (run_dir / "report.json").write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    (run_dir / "report.md").write_text(to_markdown(run_id, profile, res, meta), encoding="utf-8")
    return data
