"""Reasoning quality suite: fixed task set, scored against exact expected answers.

Carried over from _legacy/effort_tasks.py and _legacy/effort_harness.py with structure
changes only — tasks, expected answers and scoring rules are identical.
"""

from __future__ import annotations

import time
from typing import Callable

from modeltuner.backend import QualityResult
from modeltuner.llama import LlamaClient, LlamaError, ChatResult


SUITE = "reasoning"


def _contains_any(*subs: str) -> Callable[[str], bool]:
    def check(text: str) -> bool:
        low = text.lower()
        return any(s.lower() in low for s in subs)
    return check


# Task definitions — identical to _legacy/effort_tasks.py
TASKS = [
    dict(
        id="F1_age_riddle_corrected",
        prompt=(
            "Sarah is twice as old as her brother Tom was 3 years ago. In 5 years, Tom will be 22. "
            "How old is Sarah now? State the final answer as a single number clearly."
        ),
        checker=_contains_any("28"),
        note="corrected from the earlier bugged checker; real answer is 28",
    ),
    dict(
        id="F2_probability_plaintext",
        prompt=(
            "You roll two fair six-sided dice. What is the probability that the sum is exactly 7? "
            "Give the answer as a plain decimal or plain fraction like 1/6 (not LaTeX), "
            "stated clearly as your final answer."
        ),
        checker=_contains_any("1/6", "0.1666", "0.167", "16.6", "16.7"),
        note="",
    ),
    dict(
        id="F3_wolf_goat_cabbage",
        prompt=(
            "A farmer needs to cross a river with a wolf, a goat, and a cabbage. The boat only fits "
            "the farmer plus one item. If left alone together unsupervised, the wolf eats the goat, "
            "and the goat eats the cabbage. How many total crossings (one-way trips) does the farmer "
            "need to make to get everything across safely? Give the final number."
        ),
        checker=_contains_any("7"),
        note="",
    ),
    dict(
        id="F5_train_meeting",
        prompt=(
            "Two trains start 450 miles apart and travel toward each other, one at 60 mph and the "
            "other at 90 mph. A fly starts at the first train and flies back and forth between the "
            "two trains at 120 mph until they meet. How many total miles does the fly travel? "
            "Give the final number."
        ),
        checker=_contains_any("360"),
        note="",
    ),
    dict(
        id="F6_sequence_pattern",
        prompt=(
            "What is the next number in this sequence, and briefly explain the rule: "
            "2, 3, 5, 9, 17, 33, ?"
        ),
        checker=_contains_any("65"),
        note="rule: each term = 2*previous - 1... check: 2*33-1=65",
    ),
    dict(
        id="F7_river_crossing_capacity",
        prompt=(
            "8 people need to cross a river using a boat that holds at most 3 people at a time, and "
            "at least 1 person must be in the boat to row it back. What is the minimum number of "
            "one-way boat trips needed to get everyone across? Give the final number."
        ),
        checker=_contains_any("7"),
        note="verified by simulation: 3 cycles of (3 over, 1 back) + 1 final trip = 7",
    ),
    dict(
        id="F8_conditional_probability",
        prompt=(
            "A factory has two machines. Machine A makes 70% of the products and has a 2% defect "
            "rate. Machine B makes 30% of the products and has a 5% defect rate. If you pick a "
            "random defective product, what is the probability it came from Machine B? Give a "
            "percentage to one decimal place as your final answer."
        ),
        checker=_contains_any("51.7", "51.8", "52%", "0.517", "0.518"),
        note="Bayes: (0.3*0.05)/(0.7*0.02+0.3*0.05) = 0.015/0.029 = 51.7%",
    ),
    dict(
        id="F9_multistep_word_problem",
        prompt=(
            "A store buys 200 shirts at $8 each. They sell 60% of them at a 50% markup over cost, "
            "then sell the rest at a 20% discount off the marked-up price. What is the store's "
            "total profit? Give the final dollar amount."
        ),
        checker=_contains_any("608"),
        note="verified by direct computation: cost 1600; 120 @ 12 = 1440; 80 @ 9.60 = 768; revenue 2208; profit 608",
    ),
]


# Default token budget per task (from legacy --max-tokens default)
DEFAULT_MAX_TOKENS = 4000


def _check_task(text: str, reasoning: str, checker: Callable[[str], bool]) -> bool:
    """Check if the answer is correct in either the content or the reasoning block.

    Legacy rule: an answer inside the thinking text only counts as correct.
    """
    return checker(text) or checker(reasoning)


def run(
    client: LlamaClient,
    *,
    limit: int | None = None,
    max_tokens: int | None = None,
    progress: Callable[[str], None] | None = None,
    timeout: float = 600.0,
) -> QualityResult:
    """Run the reasoning suite against the currently loaded model.

    Args:
        client: LlamaClient connected to llama-server.
        limit: Run only the first N tasks (None = all).
        max_tokens: Override the per-task token budget (default 4000).
        progress: Called with one plain line per finished task.
        timeout: Per-task timeout in seconds.

    Returns:
        QualityResult with suite="reasoning", passed/total, aggregate tokens/seconds,
        and per-task details (id, passed, tokens, seconds, finish_reason, reason, cause).
    """
    token_budget = max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS
    tasks = TASKS[:limit] if limit is not None else TASKS

    passed = 0
    total = 0
    total_tokens = 0
    total_seconds = 0.0
    details: list[dict] = []

    for task in tasks:
        t0 = time.perf_counter()
        task_id = task["id"]
        checker = task["checker"]
        prompt = task["prompt"]

        try:
            result: ChatResult = client.chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=token_budget,
                temperature=0.3,
                top_p=0.95,
                top_k=20,
                seed=1234,
                thinking=None,  # do NOT disable thinking; it's part of what we measure
                timeout=timeout,
            )

            content = result.content or ""
            reasoning = result.reasoning or ""
            finish_reason = result.finish_reason
            usage = result.usage or {}
            completion_tokens = usage.get("completion_tokens", 0)
            wall = result.wall_s

            ok = _check_task(content, reasoning, checker)

            if ok:
                passed += 1
                reason = ""
                cause = ""
            else:
                # Determine failure cause
                if finish_reason == "length" or (completion_tokens >= token_budget):
                    reason = "no answer within token budget"
                    cause = "no_answer"
                else:
                    reason = "wrong answer"
                    cause = "wrong"

            total += 1
            total_tokens += completion_tokens
            total_seconds += wall

            detail = {
                "id": task_id,
                "passed": ok,
                "tokens": completion_tokens,
                "seconds": round(wall, 1),
                "finish_reason": finish_reason,
            }
            if not ok:
                detail["reason"] = reason
                detail["cause"] = cause
            details.append(detail)

            if progress:
                status = "PASS" if ok else "FAIL"
                progress(f"  {task_id:26s} {status:5s} {completion_tokens:5d} tok  {wall:6.1f}s  finish={finish_reason}")

        except LlamaError as e:
            wall = time.perf_counter() - t0
            total += 1
            total_seconds += wall

            # Classify the error
            err_msg = str(e).lower()
            if "timeout" in err_msg or "timed out" in err_msg:
                cause = "timeout"
                reason = "request timed out"
            else:
                cause = "server_error"
                reason = f"server error: {e}"

            detail = {
                "id": task_id,
                "passed": False,
                "tokens": 0,
                "seconds": round(wall, 1),
                "finish_reason": None,
                "reason": reason,
                "cause": cause,
            }
            details.append(detail)

            if progress:
                progress(f"  {task_id:26s} FAIL   0 tok  {wall:6.1f}s  cause={cause}")

            # Continue to next task (never raise)

        except Exception as e:
            wall = time.perf_counter() - t0
            total += 1
            total_seconds += wall

            detail = {
                "id": task_id,
                "passed": False,
                "tokens": 0,
                "seconds": round(wall, 1),
                "finish_reason": None,
                "reason": f"unexpected error: {e}",
                "cause": "server_error",
            }
            details.append(detail)

            if progress:
                progress(f"  {task_id:26s} FAIL   0 tok  {wall:6.1f}s  cause=server_error")

    return QualityResult(
        suite=SUITE,
        passed=passed,
        total=total,
        tokens=total_tokens,
        seconds=round(total_seconds, 1),
        details=details,
    )