import json
from pathlib import Path
import tempfile
import unittest

from logmerge.merge import merge_files


class LogMergeVisibleTests(unittest.TestCase):
    def test_deduplicates_and_sorts_in_utc(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "a.jsonl").write_text(
                '{"id":"b","timestamp":"2026-01-01T01:00:00+01:00","source":"z","n":1}\n'
                '{"id":"a","timestamp":"2025-12-31T23:30:00Z","source":"a"}\n', encoding="utf-8")
            (root / "b.jsonl").write_text(
                '{"id":"b","timestamp":"2026-01-01T00:00:00Z","source":"a","n":2}\n', encoding="utf-8")
            rows = merge_files([root / "a.jsonl", root / "b.jsonl"])
        self.assertEqual([r["id"] for r in rows], ["a", "b"])
        self.assertEqual(rows[1]["n"], 2)
        self.assertEqual(rows[1]["timestamp"], "2026-01-01T00:00:00Z")

    def test_skips_malformed_blank_and_invalid_records(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "events.jsonl"
            path.write_text('\nnot-json\n[]\n{"id":"x","timestamp":"no-zone","source":"s"}\n', encoding="utf-8")
            self.assertEqual(merge_files([path]), [])


if __name__ == "__main__":
    unittest.main()
