"""Tests for modeltuner.quality.reasoning"""

from __future__ import annotations

import pytest

from modeltuner.llama import ChatResult, LlamaClient, LlamaError
from modeltuner.quality.reasoning import SUITE, TASKS, run, DEFAULT_MAX_TOKENS


class FakeClient:
    """Fake LlamaClient that returns scripted ChatResults."""

    def __init__(self, responses: list[ChatResult | LlamaError]):
        self._responses = responses
        self._index = 0
        self.calls: list[dict] = []

    def chat(self, messages: list[dict], *, max_tokens: int = 1024, temperature: float = 0.0,
             seed: int | None = 1234, tools: list[dict] | None = None, thinking: bool | None = None,
             timeout: float | None = None, **extra) -> ChatResult:
        self.calls.append({
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
            "thinking": thinking,
            "timeout": timeout,
        })
        if self._index >= len(self._responses):
            raise RuntimeError("FakeClient: no more scripted responses")
        resp = self._responses[self._index]
        self._index += 1
        if isinstance(resp, LlamaError):
            raise resp
        return resp

    def __getattr__(self, name):
        # Delegate other methods to a minimal stub
        return lambda *a, **k: None


def _mk_result(content: str = "", reasoning: str = "", finish_reason: str = "stop",
               completion_tokens: int = 100, wall_s: float = 0.1) -> ChatResult:
    return ChatResult(
        content=content,
        reasoning=reasoning,
        tool_calls=[],
        finish_reason=finish_reason,
        timings={},
        usage={"completion_tokens": completion_tokens, "prompt_tokens": 10, "total_tokens": completion_tokens + 10},
        wall_s=wall_s,
        raw={},
    )


def test_tasks_count_matches_legacy():
    """TASKS has the same count as the legacy task list (8 tasks: F1,F2,F3,F5,F6,F7,F8,F9)."""
    assert len(TASKS) == 8


def test_suite_name():
    assert SUITE == "reasoning"


def test_correct_answers():
    """Correct answers for several tasks pass."""
    responses = [
        _mk_result(content="The answer is 28.", completion_tokens=50),  # F1
        _mk_result(content="1/6", completion_tokens=20),                # F2
        _mk_result(content="7 crossings", completion_tokens=30),        # F3
        _mk_result(content="360 miles", completion_tokens=25),          # F5
    ]
    client = FakeClient(responses)
    result = run(client, limit=4)

    assert result.suite == "reasoning"
    assert result.passed == 4
    assert result.total == 4
    assert result.tokens == 125
    assert all(d["passed"] for d in result.details)


def test_incorrect_answers():
    """Incorrect answers fail with cause='wrong'."""
    responses = [
        _mk_result(content="Sarah is 30 years old.", completion_tokens=50),  # F1 - wrong
        _mk_result(content="The probability is 1/3.", completion_tokens=30), # F2 - wrong
    ]
    client = FakeClient(responses)
    result = run(client, limit=2)

    assert result.passed == 0
    assert result.total == 2
    for d in result.details:
        assert d["passed"] is False
        assert d["cause"] == "wrong"
        assert "wrong" in d["reason"].lower()


def test_answer_in_reasoning_only():
    """Answer inside the thinking/reasoning text counts as correct (legacy rule)."""
    responses = [
        _mk_result(content="I'm not sure.", reasoning="The answer is 28.", completion_tokens=50),  # F1
        _mk_result(content="Hmm", reasoning="1/6 is the probability.", completion_tokens=30),       # F2
    ]
    client = FakeClient(responses)
    result = run(client, limit=2)

    assert result.passed == 2
    assert result.total == 2
    assert all(d["passed"] for d in result.details)


def test_limit_parameter():
    """limit runs only the first N tasks."""
    responses = [_mk_result(content="28", completion_tokens=10) for _ in range(9)]
    client = FakeClient(responses)

    result = run(client, limit=3)
    assert result.total == 3
    assert len(result.details) == 3
    assert len(client.calls) == 3

    result2 = run(client, limit=2)
    assert result2.total == 2
    assert len(client.calls) == 5  # 3 + 2


def test_token_accounting():
    """Aggregate tokens equals sum of per-task completion tokens."""
    responses = [
        _mk_result(content="28", completion_tokens=100),
        _mk_result(content="1/6", completion_tokens=50),
        _mk_result(content="7", completion_tokens=25),
    ]
    client = FakeClient(responses)
    result = run(client, limit=3)

    assert result.tokens == 175
    assert sum(d["tokens"] for d in result.details) == 175


def test_llama_error_recorded_run_continues():
    """A LlamaError on one task is recorded and the run continues."""
    responses = [
        _mk_result(content="28", completion_tokens=50),  # F1 - pass
        LlamaError("connection refused"),                # F2 - server_error
        _mk_result(content="7", completion_tokens=40),   # F3 - pass (wolf goat cabbage = 7)
    ]
    client = FakeClient(responses)
    result = run(client, limit=3)

    assert result.total == 3
    assert result.passed == 2
    assert result.details[0]["passed"] is True
    assert result.details[1]["passed"] is False
    assert result.details[1]["cause"] == "server_error"
    assert result.details[2]["passed"] is True


def test_timeout_cause():
    """Timeout error gets cause='timeout'."""
    responses = [
        LlamaError("Request timed out after 600 seconds"),
    ]
    client = FakeClient(responses)
    result = run(client, limit=1)

    assert result.total == 1
    assert result.passed == 0
    assert result.details[0]["cause"] == "timeout"


def test_no_answer_cause():
    """Finish reason 'length' or token budget exhausted gives cause='no_answer'."""
    responses = [
        _mk_result(content="Let me think...", finish_reason="length", completion_tokens=4000),  # budget exhausted
    ]
    client = FakeClient(responses)
    result = run(client, limit=1, max_tokens=4000)

    assert result.total == 1
    assert result.passed == 0
    assert result.details[0]["cause"] == "no_answer"


def test_max_tokens_override():
    """max_tokens parameter overrides the default token budget."""
    responses = [_mk_result(content="28", completion_tokens=50)]
    client = FakeClient(responses)
    run(client, limit=1, max_tokens=2000)

    assert client.calls[0]["max_tokens"] == 2000


def test_default_max_tokens_used():
    """When max_tokens not given, DEFAULT_MAX_TOKENS is used."""
    responses = [_mk_result(content="28", completion_tokens=50)]
    client = FakeClient(responses)
    run(client, limit=1)

    assert client.calls[0]["max_tokens"] == DEFAULT_MAX_TOKENS


def test_thinking_not_disabled():
    """thinking=None is passed (not disabled)."""
    responses = [_mk_result(content="28", completion_tokens=50)]
    client = FakeClient(responses)
    run(client, limit=1)

    assert client.calls[0]["thinking"] is None


def test_progress_callback():
    """progress callback receives one line per finished task."""
    lines = []
    def progress(line: str):
        lines.append(line)

    responses = [
        _mk_result(content="28", completion_tokens=50),
        _mk_result(content="1/6", completion_tokens=30),
    ]
    client = FakeClient(responses)
    run(client, limit=2, progress=progress)

    assert len(lines) == 2
    assert "F1_age_riddle_corrected" in lines[0]
    assert "F2_probability_plaintext" in lines[1]


def test_seconds_accumulated():
    """Total seconds equals sum of per-task wall times."""
    responses = [
        _mk_result(content="28", completion_tokens=50, wall_s=1.5),
        _mk_result(content="1/6", completion_tokens=30, wall_s=2.5),
    ]
    client = FakeClient(responses)
    result = run(client, limit=2)

    assert result.seconds == 4.0
    assert abs(sum(d["seconds"] for d in result.details) - 4.0) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])