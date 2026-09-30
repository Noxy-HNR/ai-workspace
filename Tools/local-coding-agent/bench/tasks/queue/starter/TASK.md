# Task: implement deterministic retry scheduling

Implement `schedule_retries` in `retry_queue.py` for a small job runner.

For each job, schedule a retry only when its status is exactly `failed`, its
`retryable` field is true, and its current `attempt` is below `max_attempts`.
Attempt numbers start at 1. The next delay is exponential backoff:
`min(base_delay * 2 ** (attempt - 1), max_delay)`. Return one record per
eligible job with `id`, `attempt` (the next attempt number), `delay`, and
`due_at` (`now + delay`). Sort by `due_at`, then by string job ID.

Reject invalid global arguments and invalid eligible-job fields with
`ValueError`. Require `now >= 0`, `base_delay >= 1`, `max_delay >= base_delay`,
nonempty string IDs, and integer (not boolean) attempts/max attempts with
`1 <= attempt <= max_attempts`. Ignore jobs that are not eligible before
validating their other fields. Do not mutate input jobs. Add or update tests,
and run the available unittest tests.
