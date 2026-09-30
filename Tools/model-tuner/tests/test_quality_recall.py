"""Tests for modeltuner.quality.recall"""

from __future__ import annotations

import pytest

from modeltuner.llama import ChatResult, LlamaClient, LlamaError
from modeltuner.quality.recall import (
    ANSWER_SLACK,
    DEFAULT_CONTEXT_TOKENS,
    DEFAULT_MAX_TOKENS,
    NEEDLE,
    QUESTION,
    RUNG_FRACTIONS,
    SUBJ,
    VERB,
    OBJ,
    TAIL,
    SUITE,
    build_prompt,
    run,
)


class FakeClient:
    """Fake LlamaClient: tokenize approximates tokens as len(text)//4, chat is scripted."""

    def __init__(self, responses: list[ChatResult | LlamaError], n_ctx: int = 32768):
        self._responses = list(responses)
        self._index = 0
        self._n_ctx = n_ctx
        self.calls: list[dict] = []
        self.tokenize_calls: list[str] = []

    def tokenize(self, text: str, timeout: float = 120) -> int:
        self.tokenize_calls.append(text)
        return len(text) // 4

    def n_ctx(self) -> int:
        return self._n_ctx

    def chat(self, messages: list[dict], *, max_tokens: int = 1024, temperature: float = 0.0,
             seed: int | None = 1234, tools: list[dict] | None = None, thinking: bool | None = None,
             timeout: float | None = None, **extra) -> ChatResult:
        self.calls.append({
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
            "timeout": timeout,
        })
        if self._index >= len(self._responses):
            raise RuntimeError("FakeClient: no more scripted responses")
        resp = self._responses[self._index]
        self._index += 1
        if isinstance(resp, LlamaError):
            raise resp
        return resp


def _mk_result(content: str = "", reasoning: str = "", finish_reason: str = "stop",
               completion_tokens: int = 100, wall_s: float = 0.1,
               prompt_per_second: float | None = None) -> ChatResult:
    timings = {"prompt_per_second": prompt_per_second} if prompt_per_second is not None else {}
    return ChatResult(
        content=content,
        reasoning=reasoning,
        tool_calls=[],
        finish_reason=finish_reason,
        timings=timings,
        usage={"completion_tokens": completion_tokens, "prompt_tokens": 10,
               "total_tokens": completion_tokens + 10},
        wall_s=wall_s,
        raw={},
    )


def _overflow_error() -> LlamaError:
    return LlamaError(
        "POST /v1/chat/completions -> HTTP 400: "
        '{"error":{"message":"exceeds maximum context length (117964 > 8192)","type":"invalid_request_error"}}'
    )


def test_suite_name():
    assert SUITE == "recall"


def test_needle_set_carried_over_unchanged():
    """NEEDLE, QUESTION and the haystack vocabulary are identical to _legacy/ctx_ladder.py."""
    assert NEEDLE == "Note for the archive team: the access code for the east archive is 7-4-1-9-EAGLE."
    assert QUESTION == ("What is the access code for the east archive mentioned in the notes above? "
                        "Reply with just the code.")
    assert SUBJ == ["The committee", "A regional office", "The night shift", "Our supplier", "The audit team",
                    "A visiting engineer", "The warehouse crew", "The finance group", "A new intern",
                    "The river authority", "The bakery on Elm Street"]
    assert VERB == ["reviewed", "postponed", "summarized", "questioned", "approved", "rewrote", "measured",
                    "archived", "discussed"]
    assert OBJ == ["the quarterly schedule", "a shipment of copper fittings", "the parking policy",
                   "an old maintenance log", "the draft budget", "two broken sensors", "the winter staffing plan",
                   "a list of vendor complaints"]
    assert TAIL == ["before lunch", "after a long debate", "without much comment", "for the third time this month",
                    "on a rainy Tuesday", "while the power was out", "at the request of the board",
                    "with some reluctance"]


def test_haystack_sizes_within_10pct_of_target():
    """Each rung's built prompt lands within 10% of its target (prompt_budget * fraction)."""
    n_ctx = 32768
    client = FakeClient([_mk_result(content="7-4-1-9-EAGLE") for _ in range(4)], n_ctx=n_ctx)
    result = run(client)

    prompt_budget = int(n_ctx * 0.9) - DEFAULT_MAX_TOKENS - ANSWER_SLACK
    assert result.total == len(RUNG_FRACTIONS)
    for d, frac in zip(result.details, RUNG_FRACTIONS):
        target = int(prompt_budget * frac)
        assert abs(d["context_tokens"] - target) / target < 0.10


def test_needles_at_legacy_depths():
    """The needle sits at the middle of each haystack (the legacy depth), in text and tokens."""
    text, got, needle_tokens = build_prompt(20000, lambda s: len(s) // 4)
    assert NEEDLE in text
    assert text.find(NEEDLE) / len(text) == pytest.approx(0.5, abs=0.1)
    assert needle_tokens / got == pytest.approx(0.5, abs=0.1)

    n_ctx = 32768
    client = FakeClient([_mk_result(content="7-4-1-9-EAGLE") for _ in range(4)], n_ctx=n_ctx)
    result = run(client)
    for d in result.details:
        assert d["depth"] == pytest.approx(0.5, abs=0.1)


def test_correct_answer_passes():
    """The needle in the content passes; the dashless form also passes (legacy scoring rule)."""
    responses = [
        _mk_result(content="7-4-1-9-EAGLE"),
        _mk_result(content="The code is 7419 EAGLE."),
    ]
    client = FakeClient(responses)
    result = run(client, limit=2)

    assert result.passed == 2
    assert result.total == 2
    assert all(d["passed"] for d in result.details)


def test_wrong_answer_fails():
    """A non-empty wrong answer fails with cause='wrong'."""
    client = FakeClient([_mk_result(content="I do not know the code.")])
    result = run(client, limit=1)

    assert result.passed == 0
    assert result.details[0]["passed"] is False
    assert result.details[0]["cause"] == "wrong"
    assert "wrong" in result.details[0]["reason"].lower()


def test_answer_in_reasoning_only_passes():
    """The needle inside the reasoning block counts (legacy rule: content + reasoning)."""
    client = FakeClient([_mk_result(content="Hmm.", reasoning="The code is 7-4-1-9-EAGLE.")])
    result = run(client, limit=1)

    assert result.passed == 1


def test_context_overflow_classified_and_run_continues():
    """A context-overflow HTTP error is classified and the run continues."""
    responses = [
        _overflow_error(),
        _mk_result(content="7-4-1-9-EAGLE"),
        _mk_result(content="7-4-1-9-EAGLE"),
        _mk_result(content="7-4-1-9-EAGLE"),
    ]
    client = FakeClient(responses)
    result = run(client)

    assert result.total == 4
    assert result.passed == 3
    assert result.details[0]["passed"] is False
    assert result.details[0]["cause"] == "context_overflow"
    assert all(d["passed"] for d in result.details[1:])


def test_timeout_classified():
    """A timeout error gets cause='timeout'."""
    client = FakeClient([LlamaError("Request timed out after 900 seconds")])
    result = run(client, limit=1)

    assert result.details[0]["cause"] == "timeout"


def test_no_answer_cause():
    """Hitting the token budget (finish_reason='length') gives cause='no_answer'."""
    client = FakeClient([_mk_result(content="Let me think...", finish_reason="length",
                                     completion_tokens=DEFAULT_MAX_TOKENS)])
    result = run(client, limit=1)

    assert result.details[0]["cause"] == "no_answer"


def test_limit_runs_only_first_n_rungs():
    """limit runs only the first N rungs."""
    client = FakeClient([_mk_result(content="7-4-1-9-EAGLE") for _ in range(4)])
    result = run(client, limit=2)

    assert result.total == 2
    assert len(result.details) == 2
    assert len(client.calls) == 2
    assert [d["id"] for d in result.details] == ["rung1", "rung2"]


def test_context_tokens_default_from_n_ctx():
    """Without context_tokens, the ladder scales to 90% of the server's n_ctx."""
    n_ctx = 100000
    client = FakeClient([_mk_result(content="7-4-1-9-EAGLE") for _ in range(4)], n_ctx=n_ctx)
    result = run(client)

    top = int(n_ctx * 0.9)
    prompt_budget = top - DEFAULT_MAX_TOKENS - ANSWER_SLACK
    assert result.details[-1]["context_tokens"] == pytest.approx(prompt_budget, rel=0.10)


def test_context_tokens_override():
    """An explicit context_tokens sets the top of the ladder."""
    client = FakeClient([_mk_result(content="7-4-1-9-EAGLE") for _ in range(4)])
    result = run(client, context_tokens=40000)

    prompt_budget = 40000 - DEFAULT_MAX_TOKENS - ANSWER_SLACK
    assert result.details[-1]["context_tokens"] == pytest.approx(prompt_budget, rel=0.10)


def test_progress_one_line_per_check():
    """progress receives one plain line per finished check."""
    lines = []
    client = FakeClient([_mk_result(content="7-4-1-9-EAGLE") for _ in range(4)])
    run(client, progress=lines.append)

    assert len(lines) == 4
    assert all(isinstance(line, str) for line in lines)


def test_prompt_tps_recorded_from_timings():
    """prompt_tps is taken from the chat timings."""
    client = FakeClient([_mk_result(content="7-4-1-9-EAGLE", prompt_per_second=1234.5)])
    result = run(client, limit=1)

    assert result.details[0]["prompt_tps"] == 1234.5


def test_run_never_raises_on_tokenize_error():
    """A tokenize failure is recorded as server_error and the run continues."""

    class BrokenTokenizeClient(FakeClient):
        def tokenize(self, text: str, timeout: float = 120) -> int:
            raise LlamaError("POST /tokenize -> HTTP 500: boom")

    client = BrokenTokenizeClient([_mk_result(content="7-4-1-9-EAGLE") for _ in range(4)])
    result = run(client)

    assert result.total == 4
    assert result.passed == 0
    assert all(d["cause"] == "server_error" for d in result.details)


def test_token_accounting_and_seconds():
    """Aggregate tokens and seconds are the sums of the per-rung values."""
    responses = [
        _mk_result(content="7-4-1-9-EAGLE", completion_tokens=50, wall_s=1.5),
        _mk_result(content="7-4-1-9-EAGLE", completion_tokens=30, wall_s=2.5),
    ]
    client = FakeClient(responses)
    result = run(client, limit=2)

    assert result.tokens == 80
    assert result.seconds == 4.0
    assert sum(d["tokens"] for d in result.details) == 80


def test_default_context_tokens_used_when_n_ctx_missing():
    """When n_ctx() returns None, the suite falls back to DEFAULT_CONTEXT_TOKENS."""

    class NoCtxClient(FakeClient):
        def n_ctx(self) -> int:
            return None

    client = NoCtxClient([_mk_result(content="7-4-1-9-EAGLE") for _ in range(4)])
    result = run(client)

    prompt_budget = DEFAULT_CONTEXT_TOKENS - DEFAULT_MAX_TOKENS - ANSWER_SLACK
    assert result.details[-1]["context_tokens"] == pytest.approx(prompt_budget, rel=0.10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
