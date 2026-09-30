"""Quality suite "tools": can the model drive a coding agent's tools?

A stand-in for an agent harness like opencode, not a reproduction of it: its own system prompt and a generic
nine-tool set shaped like a coding agent's. Each task runs the model's own multi-turn tool loop against a fresh
temporary copy of a small Python repo, then grades the final answer, the files it left behind and the tools it
called (see `tools_tasks.py`). Failure causes are scored separately, so a model that runs out of context is not
reported as a bad tool-caller.

The suite talks to whatever llama-server is serving, through `LlamaClient`. It never loads a model, never starts
or stops a server, and `run` never raises: a server error or a timeout on one task is recorded as that task's
cause and the run carries on.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from modeltuner.backend import QualityResult
from modeltuner.llama import ChatResult, LlamaClient, LlamaError
from modeltuner.quality import tools_tasks

__all__ = ["SUITE", "TASKS", "Sandbox", "check_args", "run", "run_tool"]

SUITE = "tools"

MAX_TURNS = 25            # a task that has not answered after this many model turns has failed
DEFAULT_MAX_TOKENS = 4096
BASE_SEED = 1000          # + the turn number, so every turn is reproducible and every task differs
LONG_MIN_CTX = 96_000     # the long tasks read a 620-function module first; below this they cannot run

TEMPERATURE, TOP_P, TOP_K, MIN_P = 0.6, 0.95, 20, 0.0

TASKS = tools_tasks.TASKS

SYSTEM = """You are a coding agent working inside a small Python repository. The current working directory is the \
repository root, and every path you pass to a tool must be relative to it (for example `shop/pricing.py`).

How to work:
- Use the tools to inspect the code before answering questions about it. Never guess file contents or paths.
- Prefer `grep` and `glob` to find things, then `read` the relevant part of a file.
- To change a file, use `edit` with an `old_string` copied exactly from the file (including indentation), or \
`write` to create a new file. Make the smallest change that does the job.
- Tests use the standard library: run them with the `bash` tool and the command `python -m unittest`. No other \
shell commands are available.
- If a tool returns an error, read it, adjust, and try again. Do not repeat a failing call unchanged.
- When the user asks for a plan or a todo list, use the `todowrite` tool.
- Do not ask the user questions. Do the task, then reply with a short final answer that states the result.
- If a question can be answered without looking at the code, answer it directly without tools.
"""

TOOLS = [
    {"type": "function", "function": {"name": "read", "description": "Read a text file. Returns numbered lines. Use offset/limit for large files.",
     "parameters": {"type": "object", "properties": {
         "path": {"type": "string", "description": "File path relative to the repository root"},
         "offset": {"type": "integer", "description": "1-based line to start from (default 1)"},
         "limit": {"type": "integer", "description": "Maximum number of lines to return (default 2000)"}},
         "required": ["path"]}}},
    {"type": "function", "function": {"name": "write", "description": "Create or overwrite a file with the given content.",
     "parameters": {"type": "object", "properties": {
         "path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "edit", "description": "Replace an exact string in a file. old_string must match exactly and be unique unless replace_all is true.",
     "parameters": {"type": "object", "properties": {
         "path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"},
         "replace_all": {"type": "boolean", "description": "Replace every occurrence (default false)"}},
         "required": ["path", "old_string", "new_string"]}}},
    {"type": "function", "function": {"name": "list", "description": "List the files and folders in a directory.",
     "parameters": {"type": "object", "properties": {
         "path": {"type": "string", "description": "Directory (default: repository root)"}}}}},
    {"type": "function", "function": {"name": "glob", "description": "Find files whose path matches a glob pattern such as `**/*.py`.",
     "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "grep", "description": "Search file contents with a regular expression. Returns path:line: text.",
     "parameters": {"type": "object", "properties": {
         "pattern": {"type": "string"}, "include": {"type": "string", "description": "Optional glob filter, e.g. `*.py`"}},
         "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "bash", "description": "Run a shell command in the repository root. Only `python -m unittest` (with optional arguments) is available.",
     "parameters": {"type": "object", "properties": {
         "command": {"type": "string"}, "description": {"type": "string", "description": "What the command does"}},
         "required": ["command"]}}},
    {"type": "function", "function": {"name": "todowrite", "description": "Replace the todo list for the current task.",
     "parameters": {"type": "object", "properties": {"todos": {"type": "array", "items": {"type": "object", "properties": {
         "content": {"type": "string"},
         "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
         "priority": {"type": "string", "enum": ["high", "medium", "low"]}},
         "required": ["content", "status", "priority"]}}}, "required": ["todos"]}}},
    {"type": "function", "function": {"name": "webfetch", "description": "Fetch a URL.",
     "parameters": {"type": "object", "properties": {
         "url": {"type": "string"}, "format": {"type": "string", "enum": ["text", "markdown", "html"]}}, "required": ["url"]}}},
]
SCHEMAS = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}
PYTYPES = {"string": str, "integer": int, "boolean": bool, "array": list, "object": dict}

MALFORMED = ("unknown_tool", "bad_json", "missing_args", "bad_type")


class ToolError(Exception):
    """A tool refused the call. The message goes back to the model, which may retry."""


class Sandbox:
    """The repository root the tools are confined to. Every path is checked, every write stays inside."""

    def __init__(self, root: Path):
        self.root = root.resolve()

    def path(self, p, must_exist=True):
        if not isinstance(p, str) or not p.strip():
            raise ToolError("path must be a non-empty string")
        q = p.strip().replace("\\", "/")
        if q.startswith("/") or re.match(r"^[A-Za-z]:", q):
            raise ToolError("absolute paths are not allowed; use a path relative to the repository root")
        full = (self.root / q).resolve()
        if full != self.root and self.root not in full.parents:
            raise ToolError("path is outside the repository")
        if must_exist and not full.exists():
            raise ToolError(f"file not found: {q}")
        return full

    def rel(self, p):
        return p.relative_to(self.root).as_posix()

    def files(self):
        return sorted(p for p in self.root.rglob("*") if p.is_file() and "__pycache__" not in p.parts)

    # --- tools
    def read(self, path, offset=1, limit=2000):
        f = self.path(path)
        if f.is_dir():
            raise ToolError(f"{path} is a directory; use list")
        lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(1, offset) - 1
        chunk = lines[start:start + max(1, limit)]
        body = "\n".join(f"{i + start + 1:6d}\t{line}" for i, line in enumerate(chunk))
        more = len(lines) - (start + len(chunk))
        return body + (f"\n\n(file has {more} more lines; use offset={start + len(chunk) + 1} to continue)" if more > 0 else "")

    def write(self, path, content):
        f = self.path(path, must_exist=False)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} characters to {self.rel(f)}"

    def edit(self, path, old_string, new_string, replace_all=False):
        f = self.path(path)
        text = f.read_text(encoding="utf-8", errors="replace")
        n = text.count(old_string) if old_string else 0
        if n == 0:
            raise ToolError("old_string not found in the file (it must match exactly, including whitespace)")
        if n > 1 and not replace_all:
            raise ToolError(f"old_string occurs {n} times; include more context or set replace_all")
        f.write_text(text.replace(old_string, new_string) if replace_all else text.replace(old_string, new_string, 1),
                     encoding="utf-8")
        return f"edited {self.rel(f)} ({n if replace_all else 1} replacement{'s' if replace_all and n > 1 else ''})"

    def list(self, path="."):
        d = self.path(path or ".")
        if not d.is_dir():
            raise ToolError(f"{path} is not a directory")
        items = sorted(d.iterdir(), key=lambda p: (p.is_file(), p.name))
        return "\n".join(p.name + ("/" if p.is_dir() else "") for p in items if p.name != "__pycache__") or "(empty)"

    def glob(self, pattern):
        pat = pattern.strip().lstrip("./")
        hits = [self.rel(p) for p in self.files()
                if fnmatch.fnmatch(self.rel(p), pat) or fnmatch.fnmatch(self.rel(p), pat.replace("**/", ""))]
        return "\n".join(hits[:200]) or "no files matched"

    def grep(self, pattern, include=None):
        try:
            rx = re.compile(pattern)
        except re.error as e:
            raise ToolError(f"invalid regular expression: {e}")
        out = []
        for p in self.files():
            r = self.rel(p)
            if include and not (fnmatch.fnmatch(r, include) or fnmatch.fnmatch(p.name, include)):
                continue
            for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if rx.search(line):
                    out.append(f"{r}:{i}: {line.strip()[:200]}")
                    if len(out) >= 200:
                        return "\n".join(out) + "\n(truncated at 200 matches)"
        return "\n".join(out) or "no matches"

    def bash(self, command, description=None):
        try:
            tokens = shlex.split(command, posix=True)
        except ValueError as e:
            raise ToolError(f"could not parse command: {e}")
        allowed = (len(tokens) >= 3 and tokens[0] in ("python", "python3", "py") and tokens[1:3] == ["-m", "unittest"]
                   and not any(ch in command for ch in "&|;<>`$"))
        if not allowed:
            raise ToolError("command not available in this sandbox. Only `python -m unittest [args]` can run; the working "
                            "directory is already the repository root. Use read/list/glob/grep for files.")
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        try:
            p = subprocess.run([sys.executable, "-m", "unittest", *tokens[3:]], cwd=self.root, capture_output=True,
                               text=True, timeout=60, env=env)
        except subprocess.TimeoutExpired:
            raise ToolError("command timed out after 60 s")
        return f"exit code {p.returncode}\n" + (p.stdout + p.stderr)[-6000:]

    def todowrite(self, todos):
        if not tools_tasks.todo_valid(todos):
            raise ToolError("todos must be a non-empty array of objects with content (string), status "
                            "(pending|in_progress|completed) and priority (high|medium|low)")
        return f"todo list updated ({len(todos)} items)"

    def webfetch(self, url, format=None):
        raise ToolError("network access is disabled in this sandbox")


def check_args(name: str, args: dict) -> tuple[str | None, str | None]:
    """Validate one call's arguments against the tool's own schema. Returns (kind, message) or (None, None)."""
    schema = SCHEMAS[name]
    for req in schema.get("required", []):
        if req not in args:
            return "missing_args", f"missing required argument '{req}'"
    for key, val in args.items():
        spec = schema["properties"].get(key)
        if spec is None:
            continue
        want = PYTYPES[spec["type"]]
        if not isinstance(val, want) or (want is int and isinstance(val, bool)):
            return "bad_type", f"argument '{key}' must be of type {spec['type']}"
        if "enum" in spec and val not in spec["enum"]:
            return "bad_type", f"argument '{key}' must be one of {spec['enum']}"
    return None, None


def run_tool(sb: Sandbox, call: dict, stats: dict, trace: list) -> str:
    """Run one tool call and return what the model sees next. Every refusal is text, never an exception."""
    fn = call.get("function", {})
    name, raw = fn.get("name"), fn.get("arguments", "{}")
    if name not in SCHEMAS:
        stats["unknown_tool"] += 1
        trace.append({"name": str(name), "ok": False, "args": {}})
        return f"Error: unknown tool '{name}'. Available tools: {', '.join(SCHEMAS)}"
    try:
        args = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
        if not isinstance(args, dict):
            raise ValueError("arguments are not an object")
    except Exception as e:
        stats["bad_json"] += 1
        trace.append({"name": name, "ok": False, "args": {}})
        return f"Error: arguments are not valid JSON ({e})"
    kind, msg = check_args(name, args)
    if kind:
        stats[kind] += 1
        trace.append({"name": name, "ok": False, "args": args})
        return f"Error: {msg}"
    try:
        out = getattr(sb, name)(**{k: v for k, v in args.items() if k in SCHEMAS[name]["properties"]})
        trace.append({"name": name, "ok": True, "args": args})
        return out
    except ToolError as e:
        stats["tool_error"] += 1
        trace.append({"name": name, "ok": False, "args": args})
        return f"Error: {e}"
    except Exception as e:  # an unexpected tool failure is still something the model can read and recover from
        stats["tool_error"] += 1
        trace.append({"name": name, "ok": False, "args": args})
        return f"Error: {type(e).__name__}: {e}"


# ---------------------------------------------------------------- one task

def _mkdtemp() -> Path:
    """Where one task's sandbox copy lives. A seam for the tests, which watch the cleanup."""
    return Path(tempfile.mkdtemp(prefix="modeltuner-tools-"))


def _remove(root: Path) -> None:
    """Delete a sandbox. Never raises: a stray file must not turn a finished task into a failed run."""
    shutil.rmtree(root, ignore_errors=True)


def _n_ctx(client: LlamaClient) -> int | None:
    """The context the server is serving, or None when it will not say."""
    try:
        return client.n_ctx()
    except Exception:
        return None


def _cause_of_failed_request(e: Exception) -> str:
    """Classify a request that never came back: the server said it was too long, it hung, or it broke."""
    msg = str(e)
    if re.search(r"context|exceed", msg, re.I):
        return "context_overflow"
    if isinstance(e, TimeoutError) or "timed out" in msg.lower() or "timeout" in msg.lower():
        return "timeout"
    return "server_error"


def _why(cause: str, detail: str, *, max_tokens: int, timeout: float) -> str:
    """One short line saying what went wrong. Failures caused by the harness say so themselves."""
    detail = " ".join(detail.split())[:200]
    return {
        "max_turns": f"no answer within {MAX_TURNS} turns",
        "timeout": f"the task ran past its {timeout:.0f} s budget",
        "context_overflow": f"the context was too small for this task: {detail}",
        "token_budget": f"the answer hit the {max_tokens}-token limit without answering",
        "server_error": f"llama-server failed: {detail}",
    }.get(cause, detail or cause)


def _opening_messages(sb: Sandbox, task: dict) -> list[dict]:
    """System prompt, the user's task, and for a long task the full-file read that fills the context."""
    msgs = [{"role": "system", "content": SYSTEM}]
    if task["long"]:
        pre = {"id": "call_prefill_1", "type": "function",
               "function": {"name": "read", "arguments": json.dumps({"path": "shop/legacy_billing.py", "limit": 20000})}}
        msgs += [{"role": "user", "content": tools_tasks.LONG_PRELUDE_USER},
                 {"role": "assistant", "content": "", "tool_calls": [pre]},
                 {"role": "tool", "tool_call_id": "call_prefill_1",
                  "content": sb.read("shop/legacy_billing.py", 1, 20000)},
                 {"role": "assistant", "content": "I've read shop/legacy_billing.py in full. What would you like me "
                                                 "to do with it?"}]
    msgs.append({"role": "user", "content": task["prompt"]})
    return msgs


def _ask(client: LlamaClient, msgs: list[dict], *, turns: int, max_tokens: int, timeout: float) -> ChatResult:
    """One turn. Sampling parameters are the harness's, not the tuner's: they must not change between configs."""
    return client.chat(msgs, max_tokens=max_tokens, temperature=TEMPERATURE, seed=BASE_SEED + turns, tools=TOOLS,
                       timeout=timeout, tool_choice="auto", parallel_tool_calls=True, top_p=TOP_P, top_k=TOP_K,
                       min_p=MIN_P)


def _run_task(client: LlamaClient, task: dict, *, max_tokens: int, timeout: float, n_ctx: int | None) -> dict:
    """One task in its own sandbox. Always returns a detail dict; always removes the sandbox.

    `timeout` is the task's wall-clock budget and the per-request budget: one request cannot usefully take
    longer than the whole task may.
    """
    root = _mkdtemp()
    t0 = time.monotonic()
    try:
        tools_tasks.make_sandbox(task, root)
        sb = Sandbox(root)
        stats = dict(unknown_tool=0, bad_json=0, missing_args=0, bad_type=0, tool_error=0, calls=0)
        trace: list[dict] = []
        msgs = _opening_messages(sb, task)
        turns, tokens, max_prompt, final, cause = 0, 0, 0, "", None
        while True:
            if turns >= MAX_TURNS:
                cause = "max_turns"
                break
            if time.monotonic() - t0 > timeout:
                cause = "timeout"
                break
            try:
                r = _ask(client, msgs, turns=turns, max_tokens=max_tokens, timeout=timeout)
            except LlamaError as e:
                cause, final = _cause_of_failed_request(e), str(e)[:300]
                break
            except Exception as e:  # noqa: BLE001 -- a client that is not a LlamaClient at all
                cause, final = _cause_of_failed_request(e), f"{type(e).__name__}: {e}"[:300]
                break
            turns += 1
            usage = r.usage or {}
            prompt_n, comp_n = int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
            tokens += comp_n
            max_prompt = max(max_prompt, prompt_n)
            calls = r.tool_calls or []
            if not calls:
                final = r.content or ""
                if r.finish_reason == "length":
                    # the answer was cut short: only the model's fault when the context still had room
                    cause = "token_budget"
                    if n_ctx is not None and prompt_n + comp_n >= n_ctx - 64:
                        cause = "context_overflow"
                else:
                    cause = "passed"
                break
            for i, c in enumerate(calls):
                c.setdefault("id", f"call_{turns}_{i}")
                c.setdefault("type", "function")
            msgs.append({"role": "assistant", "content": r.content or "", "tool_calls": calls})
            for c in calls:
                stats["calls"] += 1
                msgs.append({"role": "tool", "tool_call_id": c["id"], "content": run_tool(sb, c, stats, trace)})

        reason = ""
        if cause == "passed":
            try:
                ok, reason = task["grade"](root, final, trace)
            except Exception as e:  # noqa: BLE001 -- a grader that breaks is not a passing task
                ok, reason = False, f"grader failed: {type(e).__name__}: {e}"
            if not ok:
                cause = "wrong_result"
        else:
            ok = False
            reason = _why(cause, final, max_tokens=max_tokens, timeout=timeout)

        detail = dict(id=task["id"], passed=ok, cause=cause, turns=turns, tool_calls=stats["calls"], tokens=tokens,
                      max_prompt_tokens=max_prompt, seconds=round(time.monotonic() - t0, 1),
                      malformed=sum(stats[k] for k in MALFORMED), tool_errors=stats["tool_error"],
                      tools=[t["name"] for t in trace])
        if not ok:
            detail["reason"] = reason
        return detail
    finally:
        _remove(root)


def _line(detail: dict) -> str:
    return (f"{detail['id']:26s} {'PASS' if detail['passed'] else 'FAIL':4s} {detail['cause']:16s} "
            f"turns {detail['turns']:2d} calls {detail['tool_calls']:2d} malformed {detail['malformed']:d} "
            f"tool_err {detail['tool_errors']:d} ctx {detail['max_prompt_tokens']:6d} "
            f"{detail['tokens']:6d} tok {detail['seconds']:6.1f}s")


def _blank(task: dict, reason: str) -> dict:
    """A detail row for a task that never got as far as running."""
    return dict(id=task["id"], passed=False, cause="wrong_result", reason=reason, turns=0, tool_calls=0, tokens=0,
                max_prompt_tokens=0, seconds=0.0, malformed=0, tool_errors=0, tools=[])


def run(client: LlamaClient, *, limit: int | None = None, max_tokens: int | None = None,
        progress: Callable[[str], None] | None = None, timeout: float = 600.0) -> QualityResult:
    """Run the tool suite against whatever llama-server is serving.

    `limit` runs only the first N tasks, `max_tokens` caps one answer, `timeout` is one task's wall-clock budget
    and `progress` gets one line per finished task. The long tasks are skipped unless the server has at least
    96K of context, because failing them would score the context size, not the model. Never raises.
    """
    max_tokens = max_tokens or DEFAULT_MAX_TOKENS
    n_ctx = _n_ctx(client)
    todo = [t for t in TASKS if not t["long"] or n_ctx is None or n_ctx >= LONG_MIN_CTX]
    if limit is not None:
        todo = todo[:max(0, limit)]
    details: list[dict] = []
    t0 = time.monotonic()
    for t in todo:
        try:
            detail = _run_task(client, t, max_tokens=max_tokens, timeout=timeout, n_ctx=n_ctx)
        except Exception as e:  # noqa: BLE001 -- one broken task must not lose the rest of the suite
            detail = _blank(t, f"the harness failed on this task: {type(e).__name__}: {e}")
        details.append(detail)
        if progress:
            progress(_line(detail))
    return QualityResult(suite=SUITE, passed=sum(1 for d in details if d["passed"]), total=len(details),
                         tokens=sum(d["tokens"] for d in details), seconds=round(time.monotonic() - t0, 1),
                         details=details)
