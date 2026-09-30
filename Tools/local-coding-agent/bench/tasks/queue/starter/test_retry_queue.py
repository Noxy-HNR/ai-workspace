import unittest

from retry_queue import schedule_retries


class RetryQueueVisibleTests(unittest.TestCase):
    def test_schedules_exponential_delays(self):
        jobs = [
            {"id": "b", "status": "failed", "retryable": True, "attempt": 2, "max_attempts": 4},
            {"id": "a", "status": "failed", "retryable": True, "attempt": 1, "max_attempts": 3},
        ]
        self.assertEqual(
            schedule_retries(jobs, now=100, base_delay=5, max_delay=20),
            [
                {"id": "a", "attempt": 2, "delay": 5, "due_at": 105},
                {"id": "b", "attempt": 3, "delay": 10, "due_at": 110},
            ],
        )

    def test_ignores_ineligible_jobs(self):
        jobs = [
            {"id": "done", "status": "done", "retryable": True},
            {"id": "permanent", "status": "failed", "retryable": False},
            {"id": "exhausted", "status": "failed", "retryable": True, "attempt": 2, "max_attempts": 2},
        ]
        self.assertEqual(schedule_retries(jobs, now=0, base_delay=1, max_delay=10), [])

    def test_caps_delay_and_sorts_ties_by_id(self):
        jobs = [
            {"id": "z", "status": "failed", "retryable": True, "attempt": 4, "max_attempts": 9},
            {"id": "a", "status": "failed", "retryable": True, "attempt": 4, "max_attempts": 9},
        ]
        rows = schedule_retries(jobs, now=7, base_delay=4, max_delay=9)
        self.assertEqual([r["id"] for r in rows], ["a", "z"])
        self.assertEqual([r["delay"] for r in rows], [9, 9])


if __name__ == "__main__":
    unittest.main()
