"""Context truncation: count window, token cap, and archive visibility."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot import llm as llm_mod


def msg(i, direction="in", text=None, in_context=True):
    return {"id": i, "direction": direction, "kind": "text",
            "text": text or f"message number {i} with some padding words",
            "in_context": in_context, "status": "done"}


class TruncationTest(unittest.TestCase):
    def hist(self, n=50):
        return [msg(i, "in" if i % 2 == 0 else "out") for i in range(1, n + 1)]

    def test_count_window_keeps_newest(self):
        ctx = llm_mod.build_context(self.hist(), "sys", 10, 6000, 12000)
        self.assertEqual(len(ctx["messages"]), 10)
        self.assertIn("message number 50", ctx["messages"][-1]["content"])
        self.assertEqual(len(ctx["excluded_ids"]), 40)
        # excluded ids are the oldest
        self.assertEqual(ctx["excluded_ids"], list(range(1, 41)))

    def test_token_cap_drops_oldest(self):
        big = [msg(i, "in", "x" * 2000) for i in range(1, 11)]
        ctx = llm_mod.build_context(big, "sys", 10, 1000, 120000)
        self.assertLessEqual(ctx["est_tokens"], 1000)
        self.assertGreater(ctx["dropped_for_tokens"], 0)
        # newest message always survives
        self.assertIn(big[-1]["id"], ctx["included_ids"])

    def test_oldest_kept_when_small(self):
        ctx = llm_mod.build_context(self.hist(5), "sys", 30, 6000, 12000)
        self.assertEqual(len(ctx["messages"]), 5)
        self.assertEqual(ctx["excluded_ids"], [])

    def test_long_user_message_trimmed_for_model_only(self):
        h = [msg(1, "in", "y" * 5000)]
        ctx = llm_mod.build_context(h, "sys", 30, 60000, 1000)
        self.assertLess(len(ctx["messages"][0]["content"]), 5000)
        self.assertIn("trimmed for context", ctx["messages"][0]["content"])
        self.assertEqual(h[0]["text"], "y" * 5000)  # stored copy untouched


class ChunkTest(unittest.TestCase):
    def test_short_passthrough(self):
        self.assertEqual(llm_mod.chunk_reply("hi"), ["hi"])

    def test_long_split_fits(self):
        parts = llm_mod.chunk_reply("a\n\n".join(f"para {i} " + "z" * 500 for i in range(30)))
        self.assertGreater(len(parts), 1)
        for p in parts:
            self.assertLessEqual(len(p), 4000)
        self.assertIn("para 0", parts[0])
        self.assertIn("para 29", parts[-1])


if __name__ == "__main__":
    unittest.main()
