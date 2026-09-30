"""Build retry schedules for failed jobs."""

from __future__ import annotations

from typing import Any


def _require_int(name: str, value: Any, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def schedule_retries(
    jobs: list[dict[str, Any]],
    *,
    now: int,
    base_delay: int,
    max_delay: int,
) -> list[dict[str, Any]]:
    if not isinstance(jobs, list):
        raise ValueError("jobs must be a list")
    _require_int("now", now, 0)
    _require_int("base_delay", base_delay, 1)
    _require_int("max_delay", max_delay, base_delay)
    result: list[dict[str, Any]] = []
    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError("each job must be a dictionary")
        if job.get("status") != "failed" or job.get("retryable") is not True:
            continue
        attempt = job.get("attempt")
        maximum = job.get("max_attempts")
        _require_int("attempt", attempt, 1)
        _require_int("max_attempts", maximum, 1)
        if attempt > maximum:
            raise ValueError("attempt cannot exceed max_attempts")
        if attempt >= maximum:
            continue
        job_id = job.get("id")
        if not isinstance(job_id, str) or not job_id:
            raise ValueError("eligible job id must be a nonempty string")
        delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
        result.append({"id": job_id, "attempt": attempt + 1, "delay": delay, "due_at": now + delay})
    result.sort(key=lambda row: (row["due_at"], row["id"]))
    return result
