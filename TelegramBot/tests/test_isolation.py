"""Isolation: one user's messages must never appear in another's context."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot import db as db_mod
from bot import llm as llm_mod


class IsolationTest(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.db = db_mod.Database(Path(tempfile.mkdtemp()) / "t.db")
        self.addCleanup(self.db.close)
        self.db.upsert_user(111, "Alice")
        self.db.upsert_user(222, "Bob")
        for i in range(5):
            self.db.add_message(111, "in", f"alice secret {i}")
            self.db.add_message(111, "out", f"reply {i}")
        for i in range(3):
            self.db.add_message(222, "in", f"bob msg {i}")

    def test_history_scoped(self):
        for m in self.db.history(222):
            self.assertEqual(m["user_id"], 222)
            self.assertNotIn("alice", m["text"].lower())

    def test_context_contains_only_owner(self):
        ctx = llm_mod.build_context(self.db.history(222), "sys", 30, 6000, 12000)
        blob = " ".join(m["content"] for m in ctx["messages"])
        self.assertNotIn("alice", blob.lower())
        self.assertIn("bob msg 2", blob)

    def test_owner_sees_own(self):
        ctx = llm_mod.build_context(self.db.history(111), "sys", 30, 6000, 12000)
        blob = " ".join(m["content"] for m in ctx["messages"])
        self.assertIn("alice secret 0", blob)


if __name__ == "__main__":
    unittest.main()
