"""Deliberately wrong implementation used to validate the grader."""

from __future__ import annotations


def schedule_retries(jobs, *, now, base_delay, max_delay):
    # Bug: schedules every failed job, including permanent failures and jobs
    # that have already used all attempts; also uses linear, uncapped delay.
    rows = []
    for job in jobs:
        if job.get("status") == "failed":
            delay = base_delay * job.get("attempt", 1)
            rows.append({"id": job.get("id"), "attempt": job.get("attempt", 1) + 1,
                         "delay": delay, "due_at": now + delay})
    return rows
