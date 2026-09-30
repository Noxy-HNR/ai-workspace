"""Writes measured speeds into Switchboard's config/models.toml ([model.measured] of one [[model]]).

Switchboard routes by these numbers until it has its own runtime samples, and every value carries a `*_source` note
saying where it came from. The file is hand-maintained and full of comments, so this edits only the lines it owns
(targeted line edits, no TOML round trip that would drop comments).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from modeltuner import paths

_SECTION = re.compile(r"^\[\[model\]\]\s*$")
_TABLE = re.compile(r"^\[")


def _toml_value(v: float | int | str) -> str:
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)     # TOML basic strings accept JSON escapes
    if isinstance(v, float):
        return f"{v:.1f}" if v >= 10 else f"{v:.2f}"
    return str(v)


def update_measured_text(text: str, model_file: str, fields: dict[str, float | int | str]) -> str:
    """Return `text` with `fields` set in the [model.measured] table of the [[model]] whose file is `model_file`."""
    lines = text.splitlines(keepends=True)
    starts = [i for i, l in enumerate(lines) if _SECTION.match(l.strip())] + [len(lines)]
    target = None
    for a, b in zip(starts, starts[1:]):
        if any(re.match(rf'^file\s*=\s*"{re.escape(model_file)}"\s*$', l.strip()) for l in lines[a:b]):
            target = (a, b)
            break
    if target is None:
        raise KeyError(f"{model_file} is not listed in models.toml")
    a, b = target
    meas = next((i for i in range(a, b) if lines[i].strip() == "[model.measured]"), None)
    if meas is None:
        end = b
        while end > a and not lines[end - 1].strip():
            end -= 1
        block = ["\n", "[model.measured]\n"] + [f"{k} = {_toml_value(v)}\n" for k, v in fields.items()]
        return "".join(lines[:end] + block + lines[end:])
    end = next((i for i in range(meas + 1, b) if _TABLE.match(lines[i].strip())), b)
    body_end = end
    while body_end > meas + 1 and not lines[body_end - 1].strip():
        body_end -= 1
    pending = dict(fields)
    for i in range(meas + 1, body_end):
        m = re.match(r"^(\s*)([A-Za-z0-9_]+)\s*=", lines[i])
        if m and m.group(2) in pending:
            lines[i] = f"{m.group(1)}{m.group(2)} = {_toml_value(pending.pop(m.group(2)))}\n"
    extra = [f"{k} = {_toml_value(v)}\n" for k, v in pending.items()]
    return "".join(lines[:body_end] + extra + lines[body_end:])


def fields_from_report(report: dict) -> dict[str, float | str]:
    """gen_tps / prompt_tps / load_s (+ sources) from a run's report.json."""
    run_id, when = report["run_id"], (report.get("finished") or report.get("started") or "")[:10]
    power = report.get("power_key", "")
    best = report.get("best_metrics") or {}
    src = f"Model Tuner run {run_id} ({when}, {power})"
    out: dict[str, float | str] = {}
    if best.get("prose"):
        out["gen_tps"] = round(best["prose"], 1)
        out["gen_tps_source"] = f"{src}, prose decode median, {report.get('mode')} mode"
    if best.get("prefill"):
        out["prompt_tps"] = round(best["prefill"], 1)
        out["prompt_tps_source"] = f"{src}, ~5k-token prompt, median"
    loads = [t["load_s"] for t in report.get("trials", []) if t.get("load_s") and t.get("phase") == "confirm"]
    if loads:
        out["load_s"] = round(sorted(loads)[len(loads) // 2], 1)
        out["load_s_source"] = f"{src}, fresh load of the tuned config, file cached"
    return out


def apply_report(report: dict, toml_path: Path = paths.SWITCHBOARD_MODELS_TOML) -> dict[str, float | str]:
    fields = fields_from_report(report)
    if not fields:
        return {}
    text = toml_path.read_text(encoding="utf-8")
    toml_path.write_text(update_measured_text(text, report["model"], fields), encoding="utf-8")
    return fields
