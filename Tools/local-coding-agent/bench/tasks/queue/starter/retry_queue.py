"""Build retry schedules for failed jobs."""

from __future__ import annotations

from typing import Any


def schedule_retries(
    jobs: list[dict[str, Any]],
    *,
    now: int,
    base_delay: int,
    max_delay: int,
) -> list[dict[str, Any]]:
    """Return sorted retry records for jobs that can be retried."""
    raise NotImplementedError
