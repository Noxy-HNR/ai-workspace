"""Long-context recall suite: a needle-in-a-haystack ladder at growing context sizes.

Carried over from _legacy/ctx_ladder.py with structure changes only — the needle, haystack
vocabulary, question and scoring rule are identical. Runs against whatever model llama-server
is serving through LlamaClient; the tuner handles loading, settings and watchdogs.
"""

from __future__ import annotations

import random
import time
from typing import Callable

from modeltuner.backend import QualityResult
from modeltuner.llama import ChatResult, LlamaClient, LlamaError

SUITE = "recall"

# Needle, question and haystack vocabulary — identical to _legacy/ctx_ladder.py
NEEDLE = "Note for the archive team: the access code for the east archive is 7-4-1-9-EAGLE."
QUESTION = "What is the access code for the east archive mentioned in the notes above? Reply with just the code."

SUBJ = ["The committee", "A regional office", "The night shift", "Our supplier", "The audit team", "A visiting engineer",
        "The warehouse crew", "The finance group", "A new intern", "The river authority", "The bakery on Elm Street"]
VERB = ["reviewed", "postponed", "summarized", "questioned", "approved", "rewrote", "measured", "archived", "discussed"]
OBJ = ["the quarterly schedule", "a shipment of copper fittings", "the parking policy", "an old maintenance log",
       "the draft budget", "two broken sensors", "the winter staffing plan", "a list of vendor complaints"]
TAIL = ["before lunch", "after a long debate", "without much comment", "for the third time this month",
        "on a rainy Tuesday", "while the power was out", "at the request of the board", "with some reluctance"]

# Ladder rungs as fractions of the top context size. The needle sits at the middle of each
# haystack — the legacy depth.
RUNG_FRACTIONS = (0.25, 0.5, 0.75, 1.0)

# Legacy chat settings for the long recall call
DEFAULT_MAX_TOKENS = 1500
TEMPERATURE = 0.3
SEED = 3

# Room kept free for the answer, on top of the token budget, so prompt + answer stays in context
ANSWER_SLACK = 300

# Fallback top size when the server does not report n_ctx
DEFAULT_CONTEXT_TOKENS = 32768

# Error-message fragments that mark a prompt longer than the server's context
_OVERFLOW_PATTERNS = ("context length", "context_length", "too long", "maximum context", "n_ctx", "context size")


def filler(rng: random.Random, n: int) -> str:
    """`n` haystack sentences — identical to _legacy/ctx_ladder.py."""
    return " ".join(f"{rng.choice(SUBJ)} {rng.choice(VERB)} {rng.choice(OBJ)} {rng.choice(TAIL)}; the tally was "
                    f"{rng.randint(10, 999)}." for _ in range(n))


def build_prompt(target_tokens: int, tokenize: Callable[[str], int]) -> tuple[str, int, int]:
    """Build one haystack prompt sized to ~`target_tokens`, needle at the middle.

    Text construction is identical to the legacy build_prompt: filler sentences with the
    needle at the half-way mark, then the question, re-measured up to 3 times to land near
    the target. `tokenize` is client.tokenize. Returns (text, got, needle_tokens): the prompt,
    its measured length, and the token position of the needle.
    """
    rng = random.Random(5)
    per = tokenize(filler(random.Random(9), 200)) / 200
    n = int(target_tokens / per)
    text = ""
    got = 0
    for _ in range(3):
        half = n // 2
        text = filler(rng, half) + "\n\n" + NEEDLE + "\n\n" + filler(rng, n - half) + "\n\n" + QUESTION
        got = tokenize(text)
        if abs(got - target_tokens) < 800:
            break
        n = int(n * target_tokens / got)
        rng = random.Random(5)
    needle_tokens = tokenize(text.split(NEEDLE, 1)[0])
    return text, got, needle_tokens


def _needle_ok(text: str) -> bool:
    """Legacy scoring rule: the code in content+reasoning, with or without dashes."""
    return ("7-4-1-9-EAGLE" in text) or ("7419" in text.replace("-", "") and "EAGLE" in text.upper())


def run(
    client: LlamaClient,
    *,
    limit: int | None = None,
    max_tokens: int | None = None,
    progress: Callable[[str], None] | None = None,
    timeout: float = 900.0,
    context_tokens: int | None = None,
) -> QualityResult:
    """Run the recall ladder against the currently loaded model.

    Args:
        client: LlamaClient connected to llama-server.
        limit: Run only the first N rungs (None = all).
        max_tokens: Override the per-rung answer token budget (default 1500).
        progress: Called with one plain line per finished check.
        timeout: Per-rung timeout in seconds.
        context_tokens: Top of the ladder: haystacks are built up to about this many tokens,
            measured with client.tokenize, leaving room for the question and the answer.
            Default: 90% of the server's n_ctx, so the tuner can test recall near the context
            size it is about to recommend. The ladder rungs scale to that top size.

    Returns:
        QualityResult with suite="recall", passed/total, aggregate tokens/seconds, and one
        detail dict per rung (id, context_tokens, depth, passed, tokens, seconds, prompt_tps,
        and reason/cause on failure). A server error or timeout on one rung is recorded and
        the run continues; run never raises.
    """
    token_budget = max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS
    if context_tokens is None:
        n = client.n_ctx()
        context_tokens = int(n * 0.9) if n else DEFAULT_CONTEXT_TOKENS

    # Leave room for the answer; the rungs scale the remaining prompt budget.
    prompt_budget = max(1, context_tokens - token_budget - ANSWER_SLACK)
    rungs = RUNG_FRACTIONS[:limit] if limit is not None else RUNG_FRACTIONS

    passed = 0
    total = 0
    total_tokens = 0
    total_seconds = 0.0
    details: list[dict] = []

    for i, frac in enumerate(rungs):
        t0 = time.perf_counter()
        rung_id = f"rung{i + 1}"
        target = int(prompt_budget * frac)
        got = 0
        depth = 0.0
        completion_tokens = 0
        prompt_tps = None
        finish_reason = None
        ok = False
        cause = ""
        reason = ""
        wall = 0.0
        try:
            prompt, got, needle_tokens = build_prompt(target, client.tokenize)
            depth = needle_tokens / got if got else 0.0
            result: ChatResult = client.chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=token_budget,
                temperature=TEMPERATURE,
                seed=SEED,
                timeout=timeout,
            )
            content = result.content or ""
            reasoning = result.reasoning or ""
            text = f"{content} {reasoning}"
            completion_tokens = (result.usage or {}).get("completion_tokens", 0)
            prompt_tps = (result.timings or {}).get("prompt_per_second")
            finish_reason = result.finish_reason
            wall = result.wall_s

            ok = _needle_ok(text)
            if ok:
                passed += 1
            elif finish_reason == "length" or completion_tokens >= token_budget or not text.strip():
                reason = "no answer within token budget"
                cause = "no_answer"
            else:
                reason = "wrong answer"
                cause = "wrong"

        except LlamaError as e:
            wall = time.perf_counter() - t0
            err_msg = str(e).lower()
            if "timeout" in err_msg or "timed out" in err_msg:
                cause = "timeout"
                reason = "request timed out"
            elif any(p in err_msg for p in _OVERFLOW_PATTERNS):
                cause = "context_overflow"
                reason = f"context overflow: {e}"
            else:
                cause = "server_error"
                reason = f"server error: {e}"

        except Exception as e:  # noqa: BLE001 -- run never raises
            wall = time.perf_counter() - t0
            cause = "server_error"
            reason = f"unexpected error: {e}"

        total += 1
        total_tokens += completion_tokens
        total_seconds += wall

        detail = {
            "id": rung_id,
            "context_tokens": got,
            "depth": round(depth, 2),
            "passed": ok,
            "tokens": completion_tokens,
            "seconds": round(wall, 1),
            "prompt_tps": prompt_tps,
            "finish_reason": finish_reason,
        }
        if not ok:
            detail["reason"] = reason
            detail["cause"] = cause
        details.append(detail)

        if progress:
            status = "PASS" if ok else "FAIL"
            line = f"  {rung_id:8s} {status:5s} ctx {got:6d} depth {depth:.2f} {completion_tokens:4d} tok {wall:6.1f}s"
            if not ok:
                line += f" cause={cause}"
            progress(line)

    return QualityResult(
        suite=SUITE,
        passed=passed,
        total=total,
        tokens=total_tokens,
        seconds=round(total_seconds, 1),
        details=details,
    )
