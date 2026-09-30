"""Quality suite "coding": can the model write code that is actually right?

The task set and the hidden tests are carried over from `_legacy/codeeval/` unchanged (see
`coding_tasks_a.py` and `coding_tasks_b.py`). What is ported is the machinery around them: the model is
asked for a solution through `LlamaClient`, the last fenced ```python block of its answer is taken as the
program, that program is run in a subprocess against the task's hidden asserts, and the task passes only
when the asserts complete and print ALL PASSED.

The sandbox is the legacy one, kept as it was: an AST allow-list before anything runs (the tasks are pure
computation, so only computation-oriented stdlib modules may be imported), then `python -I -B prog.py` in a
temporary directory that is always deleted, with the legacy 30 s program timeout. Model code is never run
in-process.

Legacy reasons map onto the tuner's failure causes:

| legacy reason | cause         | meaning                                                        |
|---------------|---------------|----------------------------------------------------------------|
| `passed`      | --            | the hidden asserts ran and passed                              |
| `no_code`     | `no_code`     | the answer had no fenced python block                          |
| `truncated`   | `no_answer`   | the token budget ran out before any code block was closed      |
| `rejected`    | `wrong`       | the allow-list refused the program (the model wrote it)        |
| `assertion`   | `wrong`       | a hidden assert failed                                         |
| `error`       | `wrong`       | the program raised                                              |
| `timeout`     | `timeout`     | the program outlived the sandbox timeout                       |
| `request_error` | `server_error` | llama-server failed the request; `timeout` if it hung        |
| (harness)     | `sandbox_error` | the sandbox itself could not run the program                |

A truncated answer that still has a code block is run, exactly as before: it is usually incomplete, so it
fails on its own.

The suite talks to whatever llama-server is serving. It never loads a model, never starts or stops a
server, and `run` never raises: a server error or a timeout on one task is recorded as that task's cause
and the run carries on.
"""

from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from modeltuner.backend import QualityResult
from modeltuner.llama import ChatResult, LlamaClient, LlamaError
from modeltuner.quality import coding_tasks_a
from modeltuner.quality import coding_tasks_b  # noqa: F401  (registers part B into coding_tasks_a.TASKS)

__all__ = ["SUITE", "TASKS", "check_source", "extract_code", "run_solution", "run"]

SUITE = "coding"

TASKS = coding_tasks_a.TASKS

DEFAULT_MAX_TOKENS = 16384      # the legacy harness default
SANDBOX_TIMEOUT = 30.0          # the legacy per-program timeout
BASE_SEED = 1000                # + the task index, so every task differs and every run repeats

# The legacy sampling body, sent unchanged: answers must stay comparable with the hand-run results, so these
# belong to the suite, not to whatever config the tuner happens to be testing.
TEMPERATURE, TOP_P, TOP_K, MIN_P = 0.6, 0.95, 20, 0.0
REPEAT_PENALTY, PRESENCE_PENALTY = 1.0, 0.0

# ---------------------------------------------------------------- the sandbox


ALLOWED_IMPORTS = {
    "math", "collections", "heapq", "itertools", "functools", "re", "bisect", "typing", "dataclasses",
    "string", "decimal", "fractions", "datetime", "json", "enum", "operator", "copy", "abc", "textwrap",
    "statistics", "array", "numbers", "__future__", "contextlib", "types",
}
# Aimed at accidental damage (files, processes, dynamic code), not at style: idiomatic uses such as
# self.__class__.__name__ or getattr(obj, name, default) must stay legal.
BANNED_CALLS = {"open", "exec", "eval", "compile", "__import__", "input", "breakpoint"}
BANNED_ATTRS = {"__subclasses__", "__globals__", "__builtins__", "__code__", "__import__", "__loader__"}


def check_source(code: str) -> str | None:
    """Return None if acceptable, else a short reason. Nothing has run yet when this is called."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"syntax error: {e.msg} (line {e.lineno})"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    return f"disallowed import: {a.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                return f"disallowed import: {node.module}"
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in BANNED_CALLS:
            return f"disallowed call: {node.func.id}()"
        elif isinstance(node, ast.Attribute) and node.attr in BANNED_ATTRS:
            return f"disallowed attribute: {node.attr}"
    return None


def extract_code(content: str) -> str | None:
    """The last fenced python block of the answer, or None."""
    blocks = re.findall(r"```(?:python|py|Python)?[ \t]*\r?\n(.*?)```", content or "", re.S)
    return blocks[-1] if blocks else None


def _mkdtemp() -> Path:
    """Where one task's program lives. A seam for the tests, which watch the cleanup."""
    return Path(tempfile.mkdtemp(prefix="modeltuner-coding-"))


def _remove(root: Path) -> None:
    """Delete a sandbox. Never raises: a stray file must not turn a finished task into a failed run."""
    shutil.rmtree(root, ignore_errors=True)


def run_solution(code: str, tests: str, timeout: float | None = None) -> tuple[bool, str, str]:
    """Run `code` followed by `tests` in a subprocess. Returns (ok, reason, detail).

    ok only if the program ran to completion and printed ALL PASSED. `reason` is the legacy verdict:
    "passed", "rejected", "timeout", "assertion" or "error". The directory is always deleted, pass or fail.
    """
    timeout = SANDBOX_TIMEOUT if timeout is None else timeout
    bad = check_source(code)
    if bad:
        return False, "rejected", bad
    program = code + "\n\n" + tests + "\n\nprint('ALL PASSED')\n"
    root = _mkdtemp()
    t0 = time.monotonic()
    try:
        path = root / "prog.py"
        path.write_text(program, encoding="utf-8")
        env = {"PATH": os.environ.get("PATH", ""), "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
               "PYTHONHASHSEED": "0"}
        try:
            p = subprocess.run([sys.executable, "-I", "-B", str(path)], cwd=root, env=env, capture_output=True,
                               text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, "timeout", f">{timeout:.0f}s"
        took = time.monotonic() - t0
        if p.returncode == 0 and "ALL PASSED" in p.stdout:
            return True, "passed", f"{took:.1f}s"
        tail = (p.stderr or p.stdout or "").strip().splitlines()[-4:]
        kind = "assertion" if any("AssertionError" in x for x in tail) else "error"
        return False, kind, " | ".join(tail)[-400:]
    finally:
        _remove(root)


# ---------------------------------------------------------------- one task


def _cause_of_failed_request(e: Exception) -> str:
    """Classify a request that never came back: the server hung, or it broke."""
    msg = str(e).lower()
    if "timeout" in msg or "timed out" in msg:
        return "timeout"
    return "server_error"


def _ask(client: LlamaClient, prompt: str, *, max_tokens: int, seed: int, timeout: float) -> ChatResult:
    """One answer. The sampling parameters are the suite's, not the tuner's: they must not change between
    configs, or the quality numbers stop being comparable."""
    return client.chat([{"role": "user", "content": prompt}], max_tokens=max_tokens, temperature=TEMPERATURE,
                       seed=seed, timeout=timeout, top_p=TOP_P, top_k=TOP_K, min_p=MIN_P,
                       repeat_penalty=REPEAT_PENALTY, presence_penalty=PRESENCE_PENALTY)


def _judge(task: dict, content: str, finish_reason: str | None, tokens: int, max_tokens: int) -> tuple[str, str]:
    """Score an answer that came back: (cause, detail), with "passed" meaning the asserts passed."""
    code = extract_code(content)
    if code is None:
        if finish_reason == "length" or tokens >= max_tokens:
            return "no_answer", "the token budget ran out before a complete code block"
        return "no_code", " ".join((content or "").split())[:160] or "(empty answer)"
    ok, reason, detail = run_solution(code, task["tests"])
    if ok:
        return "passed", detail
    if reason == "timeout":
        return "timeout", detail
    # "rejected" is the model's own program, so it counts against the model like any other wrong answer.
    return "wrong", f"{reason}: {detail}"


def _run_task(client: LlamaClient, task: dict, *, seed: int, max_tokens: int, timeout: float) -> dict:
    """One task. Always returns a detail dict; the sandbox inside always removes its directory."""
    t0 = time.monotonic()
    r = _ask(client, task["prompt"], max_tokens=max_tokens, seed=seed, timeout=timeout)
    usage = r.usage or {}
    tokens = int(usage.get("completion_tokens") or 0)
    finish_reason = r.finish_reason
    cause, detail = _judge(task, r.content or "", finish_reason, tokens, max_tokens)
    seconds = round(time.monotonic() - t0, 1)
    out = dict(id=task["id"], passed=cause == "passed", tokens=tokens, seconds=seconds, finish_reason=finish_reason,
               category=task["category"], detail=detail)
    if cause != "passed":
        out["reason"] = detail
        out["cause"] = cause
    return out


def _blank(task: dict, cause: str, reason: str, seconds: float) -> dict:
    """A detail row for a task that never got as far as being judged."""
    return dict(id=task["id"], passed=False, tokens=0, seconds=seconds, finish_reason=None, category=task["category"],
                detail=reason, reason=reason, cause=cause)


def _line(detail: dict) -> str:
    verdict = "PASS" if detail["passed"] else "FAIL"
    cause = "" if detail["passed"] else detail["cause"]
    return (f"{detail['id']:28s} {verdict:4s} {cause:14s} {detail['tokens']:6d} tok {detail['seconds']:6.1f}s"
            f"  {'' if detail['passed'] else detail['reason'][:90]}")


def run(client: LlamaClient, *, limit: int | None = None, max_tokens: int | None = None,
        progress: Callable[[str], None] | None = None, timeout: float = 600.0) -> QualityResult:
    """Run the coding suite against whatever llama-server is serving.

    `limit` runs only the first N tasks, `max_tokens` caps one answer, `timeout` is one request's budget and
    `progress` gets one line per finished task. A server error or a timeout on one task is recorded as that
    task's cause and the run continues. Never raises.
    """
    max_tokens = max_tokens or DEFAULT_MAX_TOKENS
    todo = TASKS[:max(0, limit)] if limit is not None else TASKS
    details: list[dict] = []
    t0 = time.monotonic()
    for i, t in enumerate(todo):
        started = time.monotonic()
        try:
            detail = _run_task(client, t, seed=BASE_SEED + i, max_tokens=max_tokens, timeout=timeout)
        except LlamaError as e:
            cause = _cause_of_failed_request(e)
            reason = ("the request timed out" if cause == "timeout" else f"llama-server failed: {str(e)[:200]}")
            detail = _blank(t, cause, reason, round(time.monotonic() - started, 1))
        except Exception as e:  # noqa: BLE001 -- one broken task must not lose the rest of the suite
            detail = _blank(t, "sandbox_error", f"the harness failed on this task: {type(e).__name__}: {e}",
                            round(time.monotonic() - started, 1))
        details.append(detail)
        if progress:
            progress(_line(detail))
    return QualityResult(suite=SUITE, passed=sum(1 for d in details if d["passed"]), total=len(details),
                         tokens=sum(d["tokens"] for d in details), seconds=round(time.monotonic() - t0, 1),
                         details=details)
