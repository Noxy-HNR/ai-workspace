import unittest

from retry_queue import schedule_retries


class RetryQueueHiddenTests(unittest.TestCase):
    def test_skips_nonretryable_and_exhausted(self):
        jobs = [
            {"id": "ok", "status": "failed", "retryable": True, "attempt": 1, "max_attempts": 2},
            {"id": "no", "status": "failed", "retryable": False, "attempt": 1, "max_attempts": 2},
            {"id": "done", "status": "failed", "retryable": True, "attempt": 3, "max_attempts": 3},
        ]
        rows = schedule_retries(jobs, now=10, base_delay=3, max_delay=8)
        self.assertEqual(rows, [{"id": "ok", "attempt": 2, "delay": 3, "due_at": 13}])

    def test_validation_rejects_booleans_as_integers(self):
        with self.assertRaises(ValueError):
            schedule_retries([], now=True, base_delay=1, max_delay=1)
        with self.assertRaises(ValueError):
            schedule_retries([{"id": "x", "status": "failed", "retryable": True,
                               "attempt": True, "max_attempts": 2}],
                             now=0, base_delay=1, max_delay=1)

    def test_validation_and_input_immutability(self):
        job = {"id": "x", "status": "failed", "retryable": True, "attempt": 2, "max_attempts": 4}
        before = job.copy()
        schedule_retries([job], now=20, base_delay=2, max_delay=20)
        self.assertEqual(job, before)
        with self.assertRaises(ValueError):
            schedule_retries([{"id": "", "status": "failed", "retryable": True,
                               "attempt": 1, "max_attempts": 2}],
                             now=0, base_delay=1, max_delay=1)
        with self.assertRaises(ValueError):
            schedule_retries([], now=0, base_delay=3, max_delay=2)

    def test_deterministic_order_for_identical_due_times(self):
        jobs = [
            {"id": "b", "status": "failed", "retryable": True, "attempt": 1, "max_attempts": 4},
            {"id": "a", "status": "failed", "retryable": True, "attempt": 1, "max_attempts": 4},
        ]
        rows = schedule_retries(jobs, now=0, base_delay=2, max_delay=10)
        self.assertEqual([r["id"] for r in rows], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
