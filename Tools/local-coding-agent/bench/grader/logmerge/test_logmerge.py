import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from logmerge.merge import merge_files


class LogMergeHiddenTests(unittest.TestCase):
    def test_equal_instant_tie_uses_lexicographically_smallest_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "a").write_text('{"id":"same","timestamp":"2026-03-01T12:00:00+02:00","source":"z","v":1}\n', encoding="utf-8")
            (root / "b").write_text('{"id":"same","timestamp":"2026-03-01T10:00:00Z","source":"a","v":2}\n', encoding="utf-8")
            rows = merge_files([root / "a", root / "b"])
        self.assertEqual(rows, [{"id": "same", "timestamp": "2026-03-01T10:00:00Z", "source": "a", "v": 2}])

    def test_invalid_timezone_and_non_object_records_are_skipped(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "x"
            path.write_text('[1]\n{"id":"x","timestamp":"2026-01-01T00:00:00","source":"s"}\n'
                            '{"id":"y","timestamp":"2026-01-01T00:00:00+03:00","source":"s"}\n', encoding="utf-8")
            self.assertEqual([r["id"] for r in merge_files([path])], ["y"])

    def test_cli_writes_compact_sorted_output_and_excludes_output_as_input(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "in.jsonl"
            output = root / "nested" / "out.jsonl"
            source.write_text('{"id":"b","timestamp":"2026-01-01T00:00:02Z","source":"x"}\n'
                              '{"id":"a","timestamp":"2026-01-01T01:00:00+01:00","source":"x"}\n', encoding="utf-8")
            result = subprocess.run([sys.executable, "-B", "-m", "logmerge.cli", "--output", str(output), str(source)],
                                    capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([r["id"] for r in rows], ["a", "b"])
            self.assertNotIn(" ", output.read_text(encoding="utf-8").splitlines()[0])

    def test_cli_rejects_missing_input(self):
        result = subprocess.run([sys.executable, "-B", "-m", "logmerge.cli", "--output", "x.jsonl"],
                                capture_output=True, text=True, timeout=5)
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
