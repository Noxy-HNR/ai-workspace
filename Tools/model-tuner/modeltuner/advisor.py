"""Optional Jev (TypeSafe System One) advisor for failures the rules cannot classify.

When llama-server dies and its log matches no known pattern, the tuner asks Jev one typed question: what most likely
happened? The answer decides how the strategy reacts (back off memory, avoid a setting, or just mark the config
failed). Everything works without it; with it, unfamiliar failures get a calibrated guess instead of "crash".

Privacy: the state sent is numbers, setting names and a few filtered log lines (paths reduced to file names, no
prompts, no user content, capped length). The key never passes through the tuner: the call runs in Switchboard's
own environment and config loader (`_jev_helper.py`).
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from modeltuner import paths

SWITCHBOARD_PYTHON = paths.SWITCHBOARD_ROOT / ".venv" / "Scripts" / "python.exe"
HELPER = Path(__file__).with_name("_jev_helper.py")

FAILURE_CAUSES = {
    "out_of_vram": "The GPU ran out of memory: a CUDA allocation failed while loading the model, allocating the KV "
                   "cache or compute buffers, or during a large prompt.",
    "system_ram_exhausted": "System RAM ran out, for example because GPU memory spilled into shared system memory "
                            "or the host copy of the weights did not fit.",
    "unsupported_setting": "A flag or setting is invalid or unsupported by this llama-server build or this model "
                           "(for example a speculative-decoding type the model lacks, or a bad cache type).",
    "driver_or_runtime": "A GPU driver, CUDA runtime or kernel error unrelated to how much memory was used.",
    "model_file_problem": "The model file itself could not be read or is incompatible with this build.",
    "unknown": "The log does not show what went wrong.",
}
# How each cause maps onto the strategy's failure kinds.
CAUSE_TO_KIND = {"out_of_vram": "oom", "system_ram_exhausted": "spill", "unsupported_setting": "refused",
                 "driver_or_runtime": "crash", "model_file_problem": "refused", "unknown": "crash"}

_KEEP = re.compile(r"error|fail|abort|out of memory|cuda|alloc|invalid|unsupported|exception|assert|killed", re.I)
_PATH = re.compile(r"[A-Za-z]:\\[^\s'\"]+")


@dataclass(frozen=True)
class Diagnosis:
    cause: str
    probability: float
    confidence: float           # how concentrated the distribution is (TypeSafe's Choice confidence)
    kind: str
    probabilities: dict[str, float]

    @property
    def actionable(self) -> bool:
        """Only a probable AND concentrated answer changes how the tuner reacts; otherwise it is just logged."""
        return self.probability >= 0.6 and self.confidence >= 0.5

    def describe(self) -> str:
        return f"Jev: {self.probability:.0%} {self.cause.replace('_', ' ')} (confidence {self.confidence:.2f})"


def available() -> bool:
    return SWITCHBOARD_PYTHON.exists() and HELPER.exists()


def filter_log(tail: str, max_chars: int = 1500) -> list[str]:
    """Only error-looking lines, paths reduced to file names, capped."""
    out, total = [], 0
    for line in tail.splitlines():
        if not _KEEP.search(line):
            continue
        line = _PATH.sub(lambda m: Path(m.group(0)).name, line.strip())[:240]
        total += len(line)
        if total > max_chars:
            break
        out.append(line)
    return out[-12:]


def ask(state: dict, questions: dict, timeout_s: float = 25.0) -> dict:
    if not available():
        return {"error": "advisor unavailable (no Switchboard environment)"}
    try:
        p = subprocess.run([str(SWITCHBOARD_PYTHON), str(HELPER)], input=json.dumps({"state": state,
                           "questions": questions, "timeout_s": timeout_s * 0.6}), capture_output=True, text=True,
                           timeout=timeout_s, cwd=str(paths.SWITCHBOARD_ROOT), encoding="utf-8",
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"error": f"advisor call failed: {type(e).__name__}"}
    try:
        return json.loads((p.stdout or "").strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"error": "advisor returned no answer"}


def diagnose_failure(log_tail: str, facts: dict) -> Diagnosis | None:
    """One Choice question about why llama-server failed. None when the advisor is off or errors."""
    state = {"what_happened": "llama-server failed while loading or serving a model during automatic tuning",
             "facts": facts, "log_lines": filter_log(log_tail)}
    res = ask(state, {"cause": {
        "instructions": "llama-server (llama.cpp) stopped while loading or serving a model. Using the GPU and RAM "
                        "numbers and settings in `facts` and the error lines in `log_lines`, what most likely caused "
                        "the failure?",
        "criteria": FAILURE_CAUSES}})
    ans = (res.get("answers") or {}).get("cause")
    if not ans or ans.get("choice") not in FAILURE_CAUSES:
        return None
    probs = {k: float(v) for k, v in (ans.get("probabilities") or {}).items()}
    cause = ans["choice"]
    return Diagnosis(cause=cause, probability=probs.get(cause, 0.0), confidence=float(ans.get("confidence", 0.0)),
                     kind=CAUSE_TO_KIND[cause], probabilities=probs)
