"""Tests for modeltuner.quality.coding.

Nothing here talks to llama-server. A fake client with `LlamaClient.chat`'s signature replays scripted
answers, and the solutions below are written from the task prompts, so a pass really means the hidden
asserts ran in the sandbox subprocess and passed.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from modeltuner.llama import ChatResult, LlamaError
from modeltuner.quality import coding

ROOT = Path(__file__).resolve().parent.parent
IDS = [t["id"] for t in coding.TASKS]
BY_ID = {t["id"]: t for t in coding.TASKS}

# -- solutions written from the task prompts --------------------------------------------------------

MERGE = """```python
def merge_intervals(intervals):
    merged = []
    for start, end in sorted(tuple(pair) for pair in intervals):
        if merged and start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged
```"""

# the same, but it forgets that touching integers merge too
MERGE_WRONG = """```python
def merge_intervals(intervals):
    merged = []
    for start, end in sorted(tuple(pair) for pair in intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged
```"""

TOP_K = """```python
def top_k_frequent(words, k):
    counts = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    if k <= 0:
        return []
    return [word for word, _ in ranked[:k]]
```"""

# ties broken by insertion order instead of alphabetically
TOP_K_WRONG = """```python
def top_k_frequent(words, k):
    counts = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: -item[1])
    if k <= 0:
        return []
    return [word for word, _ in ranked[:k]]
```"""

BALANCED = r"""```python
def is_balanced(s):
    closing = {")": "(", "]": "[", "}": "{"}
    stack = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "\\":
            i += 2
            continue
        if ch in "([{":
            stack.append(ch)
        elif ch in closing:
            if not stack or stack.pop() != closing[ch]:
                return False
        i += 1
    return not stack
```"""

DURATION = """```python
import re

def parse_duration(s):
    if not re.fullmatch(r"\\s*(\\d+\\s*[dhmsDHMS]\\s*)+", s):
        raise ValueError("bad duration")
    seconds = {"d": 86400, "h": 3600, "m": 60, "s": 1}
    seen = set()
    total = 0
    for number, unit in re.findall(r"(\\d+)\\s*([dhmsDHMS])", s):
        key = unit.lower()
        if key in seen:
            raise ValueError("repeated unit")
        seen.add(key)
        total += int(number) * seconds[key]
    return total
```"""

INFINITE_LOOP = "```python\nwhile True:\n    pass\n```"

# the first fenced block is a sketch, the last one is the answer: the last one wins
TWO_BLOCKS = "```python\ndef wrong_one():\n    return 1\n```\nand here is the real one\n" + MERGE

BANNED = "```python\nimport os\n\ndef show(p):\n    return open(p).read()\n```"

SYNTAX_ERROR = "```python\ndef merge_intervals(intervals)\n    return []\n```"


# ---- the scripted client ----------------------------------------------------


def result(*, content: str = "", finish_reason: str = "stop", completion_tokens: int = 100) -> ChatResult:
    return ChatResult(content=content, reasoning="", tool_calls=[], finish_reason=finish_reason, timings={},
                      usage={"completion_tokens": completion_tokens, "prompt_tokens": 200}, wall_s=0.1, raw={})


class FakeClient:
    """Same `chat(...)` signature as `LlamaClient`; every call pops the next scripted answer.

    A step is a `ChatResult` (returned) or an exception instance (raised), so a server failure is scripted
    exactly like an answer.
    """

    def __init__(self, script):
        self.script = list(script)
        self.calls: list[dict] = []

    def chat(self, messages, *, max_tokens=1024, temperature=0.0, seed=None, tools=None, thinking=None,
             timeout=None, **extra) -> ChatResult:
        self.calls.append({"messages": [dict(m) for m in messages], "max_tokens": max_tokens,
                           "temperature": temperature, "seed": seed, "thinking": thinking, "timeout": timeout,
                           "extra": extra})
        if not self.script:
            raise AssertionError("the fake client ran out of scripted answers")
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def run_only(monkeypatch, script, ids, **kw):
    """Run just the named tasks, in the order given, so a test can script the answers one-for-one."""
    monkeypatch.setattr(coding, "TASKS", [BY_ID[i] for i in ids])
    client = FakeClient(script)
    return client, coding.run(client, **kw)


def watch_sandboxes(monkeypatch) -> list[Path]:
    """Every directory the sandbox creates, so a test can prove they are gone."""
    seen: list[Path] = []
    real = coding._mkdtemp

    def spy() -> Path:
        p = real()
        seen.append(p)
        return p

    monkeypatch.setattr(coding, "_mkdtemp", spy)
    return seen


# ---- a scripted solve ------------------------------------------------------


class TestScriptedSolve:
    def test_a_correct_solution_passes(self, monkeypatch):
        _, res = run_only(monkeypatch, [result(content=MERGE, completion_tokens=180)], ids=["E2_merge_intervals"])

        assert res.suite == "coding"
        assert (res.passed, res.total) == (1, 1)
        d = res.details[0]
        assert d["id"] == "E2_merge_intervals" and d["passed"] is True
        assert d["tokens"] == 180 and d["finish_reason"] == "stop" and d["category"] == "easy"
        assert "cause" not in d and "reason" not in d
        assert res.rate == 1.0

    def test_several_tasks_in_one_run(self, monkeypatch):
        _, res = run_only(monkeypatch, [result(content=TOP_K), result(content=BALANCED), result(content=DURATION)],
                          ids=["E6_top_k_frequent", "E7_is_balanced", "E1_parse_duration"])

        assert (res.passed, res.total) == (3, 3)
        assert [d["id"] for d in res.details] == ["E6_top_k_frequent", "E7_is_balanced", "E1_parse_duration"]

    def test_the_harness_asks_for_the_suite_sampling_parameters(self, monkeypatch):
        client, _ = run_only(monkeypatch, [result(content=TOP_K)], ids=["E6_top_k_frequent"], max_tokens=8192)

        sent = client.calls[0]
        assert sent["messages"] == [{"role": "user", "content": BY_ID["E6_top_k_frequent"]["prompt"]}]
        assert BY_ID["E6_top_k_frequent"]["prompt"].endswith("```python code block and nothing after it. Do not "
                                                            "include example usage, tests or a __main__ block.")
        assert sent["temperature"] == coding.TEMPERATURE
        assert sent["extra"] == {"top_p": coding.TOP_P, "top_k": coding.TOP_K, "min_p": coding.MIN_P,
                                 "repeat_penalty": coding.REPEAT_PENALTY, "presence_penalty": coding.PRESENCE_PENALTY}
        assert sent["max_tokens"] == 8192
        assert sent["thinking"] is None

    def test_the_last_code_block_is_the_answer(self, monkeypatch):
        _, res = run_only(monkeypatch, [result(content=TWO_BLOCKS)], ids=["E2_merge_intervals"])
        assert res.passed == 1

    def test_the_reference_solutions_pass_and_the_known_wrong_ones_fail(self, monkeypatch):
        with_wrong = [t["id"] for t in coding.TASKS if t["wrong"]][:3]
        script = [result(content=f"```python\n{BY_ID[i]['reference']}```") for i in with_wrong]
        _, good = run_only(monkeypatch, script, ids=with_wrong)
        assert (good.passed, good.total) == (3, 3)

        script = [result(content=f"```python\n{BY_ID[i]['wrong'][0]}```") for i in with_wrong]
        _, bad = run_only(monkeypatch, script, ids=with_wrong)
        assert (bad.passed, bad.total) == (0, 3)
        assert all(d["cause"] == "wrong" for d in bad.details)


# ---- how an answer fails ---------------------------------------------------


class TestFailureCauses:
    def test_a_wrong_solution_fails_with_wrong(self, monkeypatch):
        _, res = run_only(monkeypatch, [result(content=MERGE_WRONG)], ids=["E2_merge_intervals"])

        d = res.details[0]
        assert d["passed"] is False and d["cause"] == "wrong"
        assert "assertion" in d["reason"] and "AssertionError" in d["detail"]
        assert res.passed == 0

    def test_a_solution_that_raises_fails_with_wrong(self, monkeypatch):
        _, res = run_only(monkeypatch, [result(content=TOP_K_WRONG)], ids=["E6_top_k_frequent"])
        assert res.details[0]["cause"] == "wrong"

    def test_an_answer_with_no_code_block_fails_with_no_code(self, monkeypatch):
        prose = "I would sort the intervals and merge the overlapping ones, then return the result."
        _, res = run_only(monkeypatch, [result(content=prose)], ids=["E2_merge_intervals"])

        d = res.details[0]
        assert d["passed"] is False and d["cause"] == "no_code"
        assert "I would sort" in d["reason"] and res.passed == 0

    def test_an_empty_answer_fails_with_no_code(self, monkeypatch):
        _, res = run_only(monkeypatch, [result(content="")], ids=["E2_merge_intervals"])
        assert res.details[0]["cause"] == "no_code" and res.details[0]["reason"] == "(empty answer)"

    def test_the_budget_running_out_before_any_code_is_no_answer(self, monkeypatch):
        _, res = run_only(monkeypatch, [result(content="Let me think about this first...", finish_reason="length",
                                                completion_tokens=coding.DEFAULT_MAX_TOKENS)],
                          ids=["E2_merge_intervals"])

        d = res.details[0]
        assert d["passed"] is False and d["cause"] == "no_answer"
        assert "token budget" in d["reason"] and d["finish_reason"] == "length"

    def test_the_budget_running_out_is_no_answer_even_when_the_server_says_stop(self, monkeypatch):
        _, res = run_only(monkeypatch, [result(content="still talking", completion_tokens=coding.DEFAULT_MAX_TOKENS)],
                          ids=["E2_merge_intervals"], max_tokens=coding.DEFAULT_MAX_TOKENS)
        assert res.details[0]["cause"] == "no_answer"

    def test_a_rejected_program_is_the_models_fault_not_the_harnesss(self, monkeypatch):
        _, res = run_only(monkeypatch, [result(content=BANNED), result(content=SYNTAX_ERROR)],
                          ids=["E2_merge_intervals", "E6_top_k_frequent"])

        assert [d["cause"] for d in res.details] == ["wrong", "wrong"]
        assert "disallowed import: os" in res.details[0]["reason"]
        assert "syntax error" in res.details[1]["reason"]

    def test_a_closed_but_truncated_block_is_still_run(self, monkeypatch):
        # legacy rule: only a MISSING block counts as truncated; a block that was closed is judged on its
        # merits, however unfinished it is
        _, res = run_only(monkeypatch, [result(content="```python\ndef merge_intervals(i):\n    return [[0, 0]]\n```",
                                                finish_reason="length", completion_tokens=200)],
                          ids=["E2_merge_intervals"])
        assert res.details[0]["cause"] == "wrong"


# ---- the sandbox -----------------------------------------------------------


class TestSandbox:
    def test_model_code_runs_in_a_subprocess_and_the_directory_is_always_deleted(self, monkeypatch):
        seen = watch_sandboxes(monkeypatch)
        real = subprocess.run
        argv: list[list[str]] = []

        def spy(cmd, **kw):
            argv.append([str(x) for x in cmd])
            return real(cmd, **kw)

        monkeypatch.setattr(coding.subprocess, "run", spy)
        script = [result(content=MERGE), result(content="no code here"), result(content=INFINITE_LOOP)]
        _, res = run_only(monkeypatch, script, ids=["E2_merge_intervals", "E6_top_k_frequent", "E7_is_balanced"])

        assert len(argv) == 2            # the answer with no code block never reached the sandbox
        assert argv[0][0] == sys.executable and argv[0][1:3] == ["-I", "-B"]
        assert Path(argv[0][3]).parent == seen[0] and argv[0][3].endswith("prog.py")
        assert len(seen) == 2 and all(not p.exists() for p in seen)
        assert [d["passed"] for d in res.details] == [True, False, False]

    def test_an_infinite_loop_is_killed_by_the_sandbox_timeout(self, monkeypatch):
        monkeypatch.setattr(coding, "SANDBOX_TIMEOUT", 2.0)
        seen = watch_sandboxes(monkeypatch)
        _, res = run_only(monkeypatch, [result(content=INFINITE_LOOP)], ids=["E7_is_balanced"])

        d = res.details[0]
        assert d["passed"] is False and d["cause"] == "timeout" and d["reason"] == ">2s"
        assert d["seconds"] >= 2.0
        assert len(seen) == 1 and not seen[0].exists()

    def test_the_directory_is_removed_even_when_the_program_was_rejected(self, monkeypatch):
        seen = watch_sandboxes(monkeypatch)
        coding.run_solution(BANNED.split("\n", 1)[1].split("```")[0], "", timeout=5)
        assert seen == []                # a rejected program is never written or run

        ok, reason, _ = coding.run_solution("x = 1", "\nassert x == 2", timeout=5)
        assert (ok, reason) == (False, "assertion")
        assert len(seen) == 1 and not seen[0].exists()

    def test_the_allow_list_rejects_escape_hatches_before_anything_runs(self, monkeypatch):
        seen = watch_sandboxes(monkeypatch)
        for code, why in [(BANNED.split("\n", 1)[1].split("```")[0], "disallowed import: os"),
                          ("eval('1')", "disallowed call: eval()"),
                          ("().__class__.__subclasses__", "disallowed attribute: __subclasses__"),
                          ("from os import path", "disallowed import: os")]:
            assert coding.check_source(code) == why, code
        assert coding.check_source("def f(:").startswith("syntax error: invalid syntax")
        assert seen == []

    def test_the_allow_list_keeps_ordinary_code(self):
        assert coding.check_source("import math\nimport collections\nfrom collections import deque") is None
        assert coding.check_source("x = self.__class__.__name__\ny = getattr(o, n, d)") is None

    def test_run_solution_reports_a_pass_with_its_runtime(self):
        ok, reason, detail = coding.run_solution("def add(a, b):\n    return a + b", "\nassert add(2, 3) == 5")
        assert (ok, reason) == (True, "passed") and detail.endswith("s")

    def test_the_timeout_is_the_legacy_thirty_seconds(self):
        assert coding.SANDBOX_TIMEOUT == 30.0
        assert coding.DEFAULT_MAX_TOKENS == 16384


# ---- run() bookkeeping -----------------------------------------------------


class TestRunBookkeeping:
    def test_limit_runs_only_the_first_tasks(self):
        lines: list[str] = []
        client = FakeClient([result(content=DURATION), result(content=MERGE), result(content="no code")])
        res = coding.run(client, limit=3, progress=lines.append)

        assert [d["id"] for d in res.details] == IDS[:3] == ["E1_parse_duration", "E2_merge_intervals",
                                                            "E3_roman_to_int"]
        assert res.total == 3 and len(lines) == 3 and len(client.calls) == 3
        assert "E1_parse_duration" in lines[0] and "PASS" in lines[0]
        assert "E2_merge_intervals" in lines[1] and "PASS" in lines[1]
        assert "E3_roman_to_int" in lines[2] and "FAIL" in lines[2] and "no_code" in lines[2]

    def test_limit_zero_runs_nothing(self):
        res = coding.run(FakeClient([]), limit=0)
        assert (res.passed, res.total, res.tokens, res.details) == (0, 0, 0, [])

    def test_token_accounting(self, monkeypatch):
        _, res = run_only(monkeypatch, [result(content=MERGE, completion_tokens=180),
                                        result(content=TOP_K, completion_tokens=95),
                                        result(content="no code", completion_tokens=12)],
                          ids=["E2_merge_intervals", "E6_top_k_frequent", "E7_is_balanced"])

        assert res.tokens == 287
        assert [d["tokens"] for d in res.details] == [180, 95, 12]
        assert res.tokens == sum(d["tokens"] for d in res.details)
        assert res.seconds >= 0.0 and all(d["seconds"] >= 0.0 for d in res.details)

    def test_a_task_that_never_answered_costs_no_tokens(self, monkeypatch):
        _, res = run_only(monkeypatch, [LlamaError("boom")], ids=["E2_merge_intervals"])
        assert res.tokens == 0 and res.details[0]["tokens"] == 0

    def test_max_tokens_defaults_to_the_harness_default(self):
        client = FakeClient([result(content=TOP_K)])
        coding.run(client, limit=6)
        assert client.calls[0]["max_tokens"] == coding.DEFAULT_MAX_TOKENS

    def test_every_task_gets_its_own_seed_and_the_timeout(self, monkeypatch):
        client, _ = run_only(monkeypatch, [result(content=MERGE), result(content=TOP_K), result(content=BALANCED)],
                             ids=["E2_merge_intervals", "E6_top_k_frequent", "E7_is_balanced"], timeout=123.0)

        assert [c["seed"] for c in client.calls] == [coding.BASE_SEED + i for i in range(3)]
        assert all(c["timeout"] == 123.0 for c in client.calls)
        assert len(set(c["seed"] for c in client.calls)) == 3


# ---- failures keep the run going -------------------------------------------


class TestFailuresKeepTheRunGoing:
    def test_a_server_error_mid_run_is_recorded_and_the_run_continues(self, monkeypatch):
        boom = LlamaError("POST /v1/chat/completions -> HTTP 500: boom")
        lines: list[str] = []
        ids = ["E6_top_k_frequent", "E7_is_balanced", "E1_parse_duration"]
        _, res = run_only(monkeypatch, [result(content=TOP_K), boom, result(content=DURATION)],
                          ids=ids, progress=lines.append)

        assert (res.passed, res.total) == (2, 3)
        assert [d["passed"] for d in res.details] == [True, False, True]
        assert res.details[1]["id"] == ids[1]
        assert res.details[1]["cause"] == "server_error"
        assert "llama-server failed" in res.details[1]["reason"] and "boom" in res.details[1]["reason"]
        assert res.details[1]["tokens"] == 0 and res.details[1]["finish_reason"] is None
        assert len(lines) == 3

    def test_a_hung_request_is_a_timeout(self, monkeypatch):
        _, res = run_only(monkeypatch, [LlamaError("POST /v1/chat/completions failed: <urlopen error timed out>")],
                          ids=["E2_merge_intervals"])
        assert res.details[0]["cause"] == "timeout"
        assert res.details[0]["reason"] == "the request timed out"

    def test_a_crash_inside_the_sandbox_is_a_sandbox_error(self, monkeypatch):
        def boom(*a, **kw):
            raise OSError("no process could be started")

        monkeypatch.setattr(coding.subprocess, "run", boom)
        seen = watch_sandboxes(monkeypatch)
        _, res = run_only(monkeypatch, [result(content=MERGE)], ids=["E2_merge_intervals"])

        d = res.details[0]
        assert d["passed"] is False and d["cause"] == "sandbox_error" and "OSError" in d["reason"]
        assert len(seen) == 1 and not seen[0].exists()      # still cleaned up

    def test_a_hopeless_client_never_raises(self, monkeypatch):
        class Broken:
            def chat(self, *a, **kw):
                raise MemoryError("no client at all")

        res = coding.run(Broken(), limit=3)
        assert res.total == 3 and res.passed == 0 and res.tokens == 0
        assert all(d["cause"] == "sandbox_error" for d in res.details)
        assert all("MemoryError" in d["reason"] for d in res.details)

    def test_a_broken_judge_does_not_lose_the_rest_of_the_run(self, monkeypatch):
        _, res = run_only(monkeypatch, [result(content=MERGE), result(content=TOP_K), result(content=BALANCED)],
                          ids=["E2_merge_intervals", "E6_top_k_frequent", "E7_is_balanced"])
        assert res.passed == 3                                   # the baseline for the next call

        def boom(*a, **kw):
            raise OSError("no sandbox left")

        monkeypatch.setattr(coding, "run_solution", boom)
        _, res = run_only(monkeypatch, [result(content=MERGE), result(content=TOP_K), result(content=BALANCED)],
                          ids=["E2_merge_intervals", "E6_top_k_frequent", "E7_is_balanced"])
        assert res.total == 3 and res.passed == 0
        assert all(d["cause"] == "sandbox_error" and "OSError" in d["reason"] for d in res.details)


# ---- the tasks themselves --------------------------------------------------


def load_legacy_tasks() -> dict:
    """Import `_legacy/codeeval/tasks_a.py` and `tasks_b.py` as data, to check the port against the original.

    Both are loaded by path; `tasks_b` finds `tasks_a` through the directory being on `sys.path`, which is how
    the legacy harness ran it too.
    """
    legacy = ROOT / "_legacy" / "codeeval"
    saved = {name: sys.modules.pop(name, None) for name in ("tasks_a", "tasks_b")}
    sys.path.insert(0, str(legacy))
    try:
        spec = importlib.util.spec_from_file_location("tasks_a", legacy / "tasks_a.py")
        a = importlib.util.module_from_spec(spec)
        sys.modules["tasks_a"] = a                       # so `tasks_b`'s `from tasks_a import ...` finds this one
        spec.loader.exec_module(a)
        spec = importlib.util.spec_from_file_location("tasks_b", legacy / "tasks_b.py")
        b = importlib.util.module_from_spec(spec)
        sys.modules["tasks_b"] = b
        spec.loader.exec_module(b)                        # registers part B into a.TASKS
    finally:
        sys.path.remove(str(legacy))
        for name, mod in saved.items():
            sys.modules.pop(name, None)
            if mod is not None:
                sys.modules[name] = mod
    return {**a.__dict__, "FIX": b.FIX}


class TestTasksCarriedOver:
    def test_the_whole_legacy_task_set_is_here_in_the_same_order(self):
        legacy = load_legacy_tasks()
        # 24 tasks x 2 samples is the 48 trials of the legacy runs; the suite itself is 24 tasks
        assert len(coding.TASKS) == 24 == len(legacy["TASKS"])
        assert IDS == [t["id"] for t in legacy["TASKS"]]
        assert coding.TASKS is coding.coding_tasks_a.TASKS

    def test_every_task_is_byte_for_byte_the_legacy_one(self):
        legacy = load_legacy_tasks()
        assert coding.TASKS == legacy["TASKS"]

    def test_the_shared_prompt_blocks_are_unchanged(self):
        legacy = load_legacy_tasks()
        assert coding.coding_tasks_a.COMMON == legacy["COMMON"]
        assert coding.coding_tasks_b.FIX == legacy["FIX"]

    def test_every_task_is_complete(self):
        for t in coding.TASKS:
            assert t["id"] and t["category"] in ("easy", "hard", "bugfix", "class")
            assert t["prompt"].strip() and t["tests"].strip() and t["reference"].strip()
            assert isinstance(t["wrong"], list)

    @pytest.mark.parametrize("task", coding.TASKS, ids=IDS)
    def test_the_reference_solution_passes_its_own_hidden_tests(self, task):
        ok, reason, detail = coding.run_solution(task["reference"], task["tests"], timeout=90)
        assert ok, f"{task['id']}: {reason}: {detail}"

    def test_a_known_wrong_solution_still_fails(self):
        wrong = [t for t in coding.TASKS if t["wrong"]]
        assert len(wrong) == 11
        for t in wrong:
            ok, _, _ = coding.run_solution(t["wrong"][0], t["tests"], timeout=30)
            assert not ok, t["id"]
