"""Tests for the "tools" quality suite.

Nothing here talks to llama-server. A fake client with `LlamaClient.chat`'s signature replays scripted turns
(including `tool_calls`), so the multi-turn loop, the tool sandbox, the argument checks and the graders are
all exercised for real -- including the unittest runs that the bash tool and the hidden graders perform.
"""

from __future__ import annotations

import pytest

import importlib.util
import json
import socket
from pathlib import Path

from modeltuner.llama import ChatResult, LlamaError
from modeltuner.quality import tools
from modeltuner.quality import tools_tasks as tasks

ROOT = Path(__file__).resolve().parent.parent
IDS = [t["id"] for t in tasks.TASKS]

FIXED_TAX = "return round(amount_cents * rate)"
BROKEN_TAX = "return int(amount_cents * rate)"
LOYALTY_ANSWER = "It is apply_loyalty_discount in shop/pricing.py."
MUTABLE_ANSWER = "Lists are mutable, tuples are immutable."


# ---- the scripted client --------------------------------------------------

def result(*, content: str = "", tool_calls=None, finish_reason: str = "stop", completion_tokens: int = 12,
           prompt_tokens: int = 100) -> ChatResult:
    return ChatResult(content=content, reasoning="", tool_calls=tool_calls or [], finish_reason=finish_reason,
                      timings={}, usage={"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens},
                      wall_s=0.1, raw={})


def call(name: str, args: dict, id: str = "c1") -> dict:
    return {"id": id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def raw_call(name: str, arguments: str, id: str = "c1") -> dict:
    """A call whose `arguments` field is not decodable JSON."""
    return {"id": id, "type": "function", "function": {"name": name, "arguments": arguments}}


class FakeClient:
    """Same `chat(...)` signature as `LlamaClient`; every call pops the next scripted step.

    A step is a `ChatResult` (returned) or an exception instance (raised), so a mid-task server failure can be
    scripted exactly like a turn.
    """

    def __init__(self, script, n_ctx: int | None = None):
        self.script = list(script)
        self.n_ctx_value = n_ctx
        self.calls: list[dict] = []

    def n_ctx(self) -> int | None:
        return self.n_ctx_value

    def chat(self, messages, *, max_tokens=1024, temperature=0.0, seed=None, tools=None, thinking=None,
             timeout=None, **extra) -> ChatResult:
        self.calls.append({"messages": [dict(m) for m in messages], "max_tokens": max_tokens,
                           "temperature": temperature, "seed": seed, "tools": tools, "timeout": timeout,
                           "extra": extra})
        if not self.script:
            raise AssertionError("the fake client ran out of scripted turns")
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step

    def tool_result(self, index: int) -> str:
        """What the model saw after the tool call of scripted step `index` (the last message of the next call)."""
        return self.calls[index + 1]["messages"][-1]["content"]


def run_only(monkeypatch, script, ids, *, n_ctx=None, **kw):
    """Run just the named tasks, so a test can script the turns of one task and nothing else."""
    monkeypatch.setattr(tools, "TASKS", [t for t in tasks.TASKS if t["id"] in ids])
    client = FakeClient(script, n_ctx=n_ctx)
    return client, tools.run(client, **kw)


def tax_fix_turns(new_string: str) -> list:
    """read -> edit -> run the tests -> answer, the turns of a solve of T2_bugfix."""
    return [
        result(tool_calls=[call("read", {"path": "shop/tax.py"}, "c1")]),
        result(tool_calls=[call("edit", {"path": "shop/tax.py", "old_string": BROKEN_TAX,
                                         "new_string": new_string}, "c2")]),
        result(tool_calls=[call("bash", {"command": "python -m unittest"}, "c3")]),
        result(content="compute_tax truncated instead of rounding; it now rounds."),
    ]


# ---- a scripted solve -----------------------------------------------------

class TestScriptedSolve:
    def test_read_edit_test_then_answer_passes(self, monkeypatch):
        client, res = run_only(monkeypatch, [result(content=LOYALTY_ANSWER)] + tax_fix_turns(FIXED_TAX),
                               ids=["T1_locate", "T2_bugfix"])

        assert res.suite == "tools"
        assert (res.passed, res.total) == (2, 2)
        assert [d["id"] for d in res.details] == ["T1_locate", "T2_bugfix"]

        d = res.details[1]
        assert d["passed"] is True and d["cause"] == "passed"
        assert (d["turns"], d["tool_calls"], d["tools"]) == (4, 3, ["read", "edit", "bash"])
        assert "reason" not in d

        # the harness sent the tool output back, and the bash call really ran unittest in the sandbox
        assert "def compute_tax" in client.tool_result(1)
        assert client.tool_result(2) == "edited shop/tax.py (1 replacement)"
        assert client.tool_result(3).startswith("exit code 0")
        assert "OK" in client.tool_result(3)

    def test_the_harness_sends_the_agent_tools(self, monkeypatch):
        client, _ = run_only(monkeypatch, [result(content=MUTABLE_ANSWER)], ids=["T7_no_tool"])
        sent = client.calls[0]
        assert sent["tools"] is tools.TOOLS
        assert sent["extra"]["tool_choice"] == "auto" and sent["extra"]["parallel_tool_calls"] is True
        assert sent["temperature"] == tools.TEMPERATURE
        assert sent["messages"][0] == {"role": "system", "content": tools.SYSTEM}
        assert sent["messages"][-1] == {"role": "user", "content": tasks.TASKS[6]["prompt"]}

    def test_wrong_edit_fails_and_says_why(self, monkeypatch):
        _, res = run_only(monkeypatch, [result(content=LOYALTY_ANSWER),
                                         result(tool_calls=[call("read", {"path": "shop/tax.py"}, "c1")]),
                                         result(tool_calls=[call("edit", {"path": "shop/tax.py",
                                                                         "old_string": BROKEN_TAX,
                                                                         "new_string": f"{BROKEN_TAX} + 1"},
                                                                 "c2")]),
                                         result(content="Adjusted compute_tax.")],
                          ids=["T1_locate", "T2_bugfix"])

        assert (res.passed, res.total) == (1, 2)
        d = res.details[1]
        assert d["passed"] is False and d["cause"] == "wrong_result"
        assert "hidden suite" in d["reason"] and "FAILED" in d["reason"]
        assert d["tool_calls"] == 2 and d["tool_errors"] == 0 and d["tools"] == ["read", "edit"]
        assert "reason" not in res.details[0]

    def test_answering_without_using_a_tool_passes_t7(self, monkeypatch):
        _, res = run_only(monkeypatch, [result(content=MUTABLE_ANSWER)], ids=["T7_no_tool"])
        d = res.details[0]
        assert d["id"] == "T7_no_tool" and d["passed"] is True
        assert (d["turns"], d["tool_calls"], d["tools"]) == (1, 0, [])

    def test_a_long_task_starts_from_a_full_read_of_the_module(self, monkeypatch):
        client, res = run_only(monkeypatch, [result(content="The code is FIN-4471-QX.")], ids=["L1_recall_in_long_session"])
        opening = client.calls[0]["messages"]
        assert res.passed == 1
        assert [m["role"] for m in opening] == ["system", "user", "assistant", "tool", "assistant", "user"]
        assert opening[1]["content"] == tasks.LONG_PRELUDE_USER
        assert opening[3]["content"].startswith('     1\t"""Legacy billing rules')
        assert "LATE_FEE_GRACE_DAYS" in opening[3]["content"]
        assert "FIN-4471-QX" in opening[3]["content"]


# ---- malformed calls ------------------------------------------------------

class TestMalformedCalls:
    def test_bad_calls_are_readable_errors_not_failures(self, monkeypatch):
        client, res = run_only(monkeypatch, [
            result(tool_calls=[call("read", {})]),                                      # missing required argument
            result(tool_calls=[call("read", {"path": "shop/pricing.py", "offset": "2"})]),  # wrong type
            result(tool_calls=[call("read", {"path": 7})]),                             # wrong type
            result(tool_calls=[call("webfetch", {"url": "https://example.com", "format": "yaml"})]),  # outside enum
            result(tool_calls=[raw_call("read", "{not json")]),                         # not JSON
            result(tool_calls=[raw_call("frobnicate", "{}")]),                          # no such tool
            # a todo list with a bad status passes the schema (it is a list) and is caught by the tool itself
            result(tool_calls=[call("todowrite", {"todos": [{"content": "a", "status": "done",
                                                             "priority": "high"}]}, "c7")]),
            result(tool_calls=[call("read", {"path": "shop/pricing.py"}, "c8")]),
            result(content=LOYALTY_ANSWER),
        ], ids=["T1_locate"])

        assert res.passed == 1 and res.details[0]["turns"] == 9
        assert client.tool_result(0) == "Error: missing required argument 'path'"
        assert client.tool_result(1) == "Error: argument 'offset' must be of type integer"
        assert client.tool_result(2) == "Error: argument 'path' must be of type string"
        assert client.tool_result(3) == "Error: argument 'format' must be one of ['text', 'markdown', 'html']"
        assert client.tool_result(4).startswith("Error: arguments are not valid JSON")
        assert client.tool_result(5).startswith("Error: unknown tool 'frobnicate'")
        assert client.tool_result(6).startswith("Error: todos must be a non-empty array")
        assert "apply_loyalty_discount" in client.tool_result(7)

        d = res.details[0]
        assert d["malformed"] == 6 and d["tool_errors"] == 1
        assert d["tools"] == ["read", "read", "read", "webfetch", "read", "frobnicate", "todowrite", "read"]

    def test_a_tool_error_lets_the_model_retry(self, monkeypatch):
        client, res = run_only(monkeypatch, [
            result(tool_calls=[call("edit", {"path": "shop/tax.py", "old_string": "no such text",
                                             "new_string": "x"}, "c1")]),
            result(tool_calls=[call("read", {"path": "shop/pricing.py"}, "c2")]),
            result(content=LOYALTY_ANSWER),
        ], ids=["T1_locate"])
        assert "old_string not found" in client.tool_result(0)
        assert res.passed == 1
        assert res.details[0]["tool_errors"] == 1 and res.details[0]["tool_calls"] == 2

    def test_an_unexpected_tool_failure_is_still_a_message(self, monkeypatch):
        def boom(self, path, offset=1, limit=2000):
            raise ValueError("file is locked")

        monkeypatch.setattr(tools.Sandbox, "read", boom)
        client, res = run_only(monkeypatch, [
            result(tool_calls=[call("read", {"path": "shop/pricing.py"}, "c1")]),
            result(content=LOYALTY_ANSWER),
        ], ids=["T1_locate"])
        assert client.tool_result(0) == "Error: ValueError: file is locked"
        assert res.passed == 1
        assert res.details[0]["tool_errors"] == 1


# ---- the bash tool stays restricted ---------------------------------------

class TestBashRestriction:
    def test_only_python_m_unittest_runs(self, monkeypatch):
        client, res = run_only(monkeypatch, [
            result(tool_calls=[call("bash", {"command": "ls -la"}, "c1")]),
            result(tool_calls=[call("bash", {"command": "python -m pytest"}, "c2")]),
            result(tool_calls=[call("bash", {"command": "python -m unittest && echo done"}, "c3")]),
            result(tool_calls=[call("bash", {"command": "python -m unittest; del shop/tax.py"}, "c4")]),
            result(tool_calls=[call("bash", {"command": "python -m unittest tests.test_utils"}, "c5")]),
            result(content=LOYALTY_ANSWER),
        ], ids=["T1_locate"])

        for i in range(4):
            assert client.tool_result(i).startswith("Error: command not available in this sandbox")
            assert "Only `python -m unittest [args]` can run" in client.tool_result(i)
        assert client.tool_result(4).startswith("exit code 0")
        assert "OK" in client.tool_result(4)
        assert res.passed == 1
        assert res.details[0]["tool_errors"] == 4


# ---- sandboxes ------------------------------------------------------------

class TestSandboxLifecycle:
    def _watch(self, monkeypatch) -> list[Path]:
        seen: list[Path] = []
        real = tools._mkdtemp

        def spy() -> Path:
            p = real()
            seen.append(p)
            return p

        monkeypatch.setattr(tools, "_mkdtemp", spy)
        return seen

    def test_each_task_gets_a_fresh_copy_that_is_deleted_afterwards(self, monkeypatch):
        seen = self._watch(monkeypatch)
        client, res = run_only(monkeypatch, [
            result(tool_calls=[call("write", {"path": "shop/new.py", "content": "x = 1\n"}, "c1")]),
            result(content=LOYALTY_ANSWER),
            result(content=MUTABLE_ANSWER),
        ], ids=["T1_locate", "T7_no_tool"])

        assert res.passed == 2
        assert len(seen) == 2 and seen[0] != seen[1]
        assert all(not p.exists() for p in seen)
        assert "wrote 6 characters to shop/new.py" in client.tool_result(0)

    def test_the_copy_is_deleted_even_when_the_task_fails(self, monkeypatch):
        seen = self._watch(monkeypatch)
        client = FakeClient([LlamaError("POST /v1/chat/completions -> HTTP 500: boom")], n_ctx=32768)
        res = tools.run(client, limit=1)

        assert res.passed == 0 and len(seen) == 1
        assert not seen[0].exists()

    def test_tools_cannot_leave_the_repository(self, monkeypatch):
        client, res = run_only(monkeypatch, [
            result(tool_calls=[call("read", {"path": "../agent_tasks.py"}, "c1")]),
            result(tool_calls=[call("read", {"path": "C:/Windows/win.ini"}, "c2")]),
            result(content=LOYALTY_ANSWER),
        ], ids=["T1_locate"])
        assert "path is outside the repository" in client.tool_result(0)
        assert "absolute paths are not allowed" in client.tool_result(1)
        assert res.details[0]["tool_errors"] == 2


# ---- run() bookkeeping ----------------------------------------------------

class TestRunBookkeeping:
    def test_limit_runs_only_the_first_tasks(self):
        lines: list[str] = []
        client = FakeClient([result(content=LOYALTY_ANSWER), result(content="done")])
        res = tools.run(client, limit=2, progress=lines.append)

        assert [d["id"] for d in res.details] == IDS[:2]
        assert res.total == 2 and len(lines) == 2 and len(client.calls) == 2
        assert "T1_locate" in lines[0] and "PASS" in lines[0] and "calls  0" in lines[0]
        assert "T2_bugfix" in lines[1] and "FAIL" in lines[1] and "wrong_result" in lines[1]

    def test_limit_zero_runs_nothing(self):
        res = tools.run(FakeClient([]), limit=0)
        assert (res.passed, res.total, res.tokens, res.details) == (0, 0, 0, [])

    def test_tokens_are_completion_tokens_across_all_turns(self):
        client = FakeClient([result(content=LOYALTY_ANSWER, completion_tokens=40, prompt_tokens=1200),
                             result(content=MUTABLE_ANSWER, completion_tokens=7, prompt_tokens=1300)])
        res = tools.run(client, limit=7, max_tokens=4096)

        assert [d["tokens"] for d in res.details[:2]] == [40, 7]
        assert res.tokens == sum(d["tokens"] for d in res.details)
        assert res.details[1]["max_prompt_tokens"] == 1300
        assert client.calls[0]["max_tokens"] == 4096
        assert res.seconds >= 0.0 and all(d["seconds"] >= 0.0 for d in res.details)

    def test_max_tokens_defaults_to_the_harness_default(self):
        client = FakeClient([result(content=MUTABLE_ANSWER)])
        tools.run(client, limit=7)
        assert client.calls[0]["max_tokens"] == tools.DEFAULT_MAX_TOKENS

    def test_a_run_out_of_turns_is_reported_as_such(self, monkeypatch):
        _, res = run_only(monkeypatch,
                          [result(tool_calls=[call("read", {"path": "shop/pricing.py"}, f"c{i}")])
                           for i in range(tools.MAX_TURNS + 2)], ids=["T1_locate"])
        d = res.details[0]
        assert d["passed"] is False and d["cause"] == "max_turns"
        assert d["turns"] == tools.MAX_TURNS and d["reason"] == f"no answer within {tools.MAX_TURNS} turns"

    def test_an_answer_that_runs_into_the_token_limit_is_not_a_context_overflow(self, monkeypatch):
        client, res = run_only(monkeypatch, [result(content="a very long answer", finish_reason="length",
                                                    completion_tokens=4096, prompt_tokens=1200)],
                               ids=["T1_locate"], max_tokens=4096, n_ctx=32768)
        assert res.details[0]["cause"] == "token_budget"
        assert "4096-token limit" in res.details[0]["reason"]
        assert client.calls[0]["max_tokens"] == 4096

    def test_an_answer_that_runs_into_a_small_context_is_a_context_overflow(self, monkeypatch):
        _, res = run_only(monkeypatch, [result(content="a very long answer", finish_reason="length",
                                               completion_tokens=4000, prompt_tokens=8000)],
                          ids=["T1_locate"], n_ctx=4096)
        assert res.details[0]["cause"] == "context_overflow"
        assert "context was too small" in res.details[0]["reason"]


class TestFailuresKeepTheRunGoing:
    def test_server_errors_timeouts_and_passes_in_one_run(self):
        boom = LlamaError("POST /v1/chat/completions -> HTTP 500: boom")
        client = FakeClient([
            LlamaError("POST /v1/chat/completions -> HTTP 400: the request exceeds the available context size"),
            boom,
            RuntimeError("connection reset by peer"),
            LlamaError("POST /v1/chat/completions failed: <urlopen error timed out>"),
            socket.timeout("timed out"),
            # T6_log_to_config: grep the log, edit the config, answer
            result(tool_calls=[call("grep", {"pattern": "max_retries changed"}, "g1")]),
            result(tool_calls=[call("edit", {"path": "shop/config.py", "old_string": "DEFAULT_MAX_RETRIES = 3",
                                             "new_string": "DEFAULT_MAX_RETRIES = 7"}, "e1")]),
            result(content="DEFAULT_MAX_RETRIES is now 7."),
            # T7_no_tool: straight to the answer
            result(content=MUTABLE_ANSWER),
        ], n_ctx=32768)
        res = tools.run(client, limit=7)

        assert [d["cause"] for d in res.details] == ["context_overflow", "server_error", "server_error",
                                                     "timeout", "timeout", "passed", "passed"]
        assert (res.passed, res.total) == (2, 7)
        assert "context was too small" in res.details[0]["reason"]
        assert res.details[1]["turns"] == 0 and res.details[1]["tool_calls"] == 0
        assert res.details[4]["reason"] == "the task ran past its 600 s budget"
        assert res.details[5]["tool_calls"] == 2 and res.details[5]["tools"] == ["grep", "edit"]
        assert res.details[6]["tool_calls"] == 0

    def test_long_tasks_only_run_with_enough_context(self):
        boom = LlamaError("POST /v1/chat/completions -> HTTP 500: boom")

        small = FakeClient([boom] * len(tasks.TASKS), n_ctx=8192)
        assert [d["id"] for d in tools.run(small).details] == [t["id"] for t in tasks.TASKS if not t["long"]]

        big = FakeClient([boom] * len(tasks.TASKS), n_ctx=96_000)
        res = tools.run(big)
        assert [d["id"] for d in res.details] == IDS
        assert res.details[-1]["id"] == "L2_edit_in_long_session"

    def test_run_never_raises_even_on_a_hopeless_client(self):
        class Broken:
            def chat(self, *a, **kw):
                raise MemoryError("no client at all")

            def n_ctx(self):
                return 32768

        res = tools.run(Broken(), limit=3)
        assert res.total == 3 and res.passed == 0
        assert all(d["cause"] == "server_error" for d in res.details)
        assert all("MemoryError" in d["reason"] for d in res.details)


# ---- the tasks themselves -------------------------------------------------

def load_legacy_tasks() -> dict:
    """Import `_legacy/agent_tasks.py` as data, to check the port against the original."""
    path = ROOT / "_legacy" / "agent_tasks.py"
    if not path.exists():
        pytest.skip("_legacy/agent_tasks.py is not present (it is not in git)")
    spec = importlib.util.spec_from_file_location("legacy_agent_tasks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.__dict__


class TestTasksCarriedOver:
    def test_same_count_and_same_tasks_as_the_legacy_list(self):
        legacy = load_legacy_tasks()
        assert tools.TASKS is tasks.TASKS
        assert len(tasks.TASKS) == len(legacy["TASKS"]) == 11
        assert [(t["id"], t["prompt"], t["long"], sorted(t["extra_files"])) for t in tasks.TASKS] == \
               [(t["id"], t["prompt"], t["long"], sorted(t["extra_files"])) for t in legacy["TASKS"]]
        assert IDS == ["T1_locate", "T2_bugfix", "T3_rename", "T4_feature", "T5_todo_then_edit", "T6_log_to_config",
                       "T7_no_tool", "T8_recover_wrong_path", "T9_missing_test", "L1_recall_in_long_session",
                       "L2_edit_in_long_session"]

    def test_the_base_repo_is_byte_for_byte_the_legacy_one(self):
        legacy = load_legacy_tasks()
        assert tasks.BASE == legacy["BASE"]
        assert tasks.BASE_TESTS == legacy["BASE_TESTS"]
        assert tasks.LONG_PRELUDE_USER == legacy["LONG_PRELUDE_USER"]
        assert tasks.LEGACY_TEST == legacy["LEGACY_TEST"]

    def test_every_sandbox_is_a_copy_of_the_legacy_sandbox(self, tmp_path):
        legacy = load_legacy_tasks()
        for i, (new, old) in enumerate(zip(tasks.TASKS, legacy["TASKS"])):
            a, b = tasks.make_sandbox(new, tmp_path / f"new{i}"), legacy["make_sandbox"](old, tmp_path / f"old{i}")
            assert files_in(a) == files_in(b), new["id"]

    def test_every_task_is_graded_by_something(self):
        for t in tasks.TASKS:
            assert callable(t["grade"]) and callable(t["reference"]) and t["prompt"].strip()
            assert isinstance(t["long"], bool) and isinstance(t["extra_files"], dict)


def files_in(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}
