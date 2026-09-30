"""Permissions, persistence, settings secrets, and LLM failure handling."""
import json
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot import db as db_mod
from bot import llm as llm_mod
from bot import settings as settings_mod


class PermissionsTest(unittest.TestCase):
    def setUp(self):
        self.db = db_mod.Database(Path(tempfile.mkdtemp()) / "t.db")
        self.addCleanup(self.db.close)

    def test_default_role_user(self):
        u = self.db.upsert_user(1, "A")
        self.assertEqual(u["role"], "user")
        self.assertEqual(u["blocked"], 0)

    def test_grant_revoke(self):
        self.db.upsert_user(1, "A")
        self.db.set_user_fields(1, role="admin")
        self.assertEqual(self.db.get_user(1)["role"], "admin")
        self.db.set_user_fields(1, role="user")
        self.assertEqual(self.db.get_user(1)["role"], "user")

    def test_block_unblock(self):
        self.db.upsert_user(1, "A")
        self.db.set_user_fields(1, blocked=1)
        self.assertEqual(self.db.get_user(1)["blocked"], 1)

    def test_disallowed_fields_ignored(self):
        self.db.upsert_user(1, "A")
        self.db.set_user_fields(1, role="admin", id=999, first_name="X")
        u = self.db.get_user(1)
        self.assertEqual(u["id"], 1)
        self.assertEqual(u["role"], "admin")


class PersistenceTest(unittest.TestCase):
    def test_survives_reopen(self):
        d = Path(tempfile.mkdtemp()) / "t.db"
        db1 = db_mod.Database(d)
        self.addCleanup(db1.close)
        db1.upsert_user(7, "Seven")
        db1.add_message(7, "in", "hello")
        db1.set_setting("llm_base_url", "http://x/v1")
        db2 = db_mod.Database(d)
        self.addCleanup(db2.close)
        self.assertEqual(db2.get_user(7)["first_name"], "Seven")
        self.assertEqual(len(db2.history(7)), 1)
        self.assertEqual(db2.get_setting("llm_base_url"), "http://x/v1")

    def test_reset_keeps_history_but_clears_context(self):
        db = db_mod.Database(Path(tempfile.mkdtemp()) / "t.db")
        self.addCleanup(db.close)
        db.upsert_user(9, "N")
        db.add_message(9, "in", "a")
        db.add_message(9, "out", "b")
        db.reset_context(9)
        hist = db.history(9)
        self.assertEqual(len(hist), 3)  # nothing deleted
        ctx = llm_mod.build_context(db.history(9), "sys", 30, 6000, 12000)
        self.assertEqual(ctx["messages"], [])

    def test_interrupted_marked(self):
        db = db_mod.Database(Path(tempfile.mkdtemp()) / "t.db")
        self.addCleanup(db.close)
        db.upsert_user(3, "C")
        mid = db.add_message(3, "in", "x")
        db.mark_processing(mid)
        self.assertEqual(db.mark_interrupted_on_boot(), 1)


class SettingsTest(unittest.TestCase):
    def test_secrets_masked(self):
        pub = settings_mod.public_settings(
            {"tg_bot_token": "123456:ABCDEF-secret-token", "llm_api_key": ""})
        self.assertEqual(pub["tg_bot_token"]["value"], "")
        self.assertTrue(pub["tg_bot_token"]["set"])
        self.assertNotIn("ABCDEF", pub["tg_bot_token"]["masked"])
        self.assertTrue(pub["tg_bot_token"]["masked"].endswith("oken"))

    def test_int_bounds(self):
        s = settings_mod.typed_settings({"ctx_history_limit": "99999",
                                         "llm_timeout": "1"})
        self.assertEqual(s["ctx_history_limit"], 500)
        self.assertEqual(s["llm_timeout"], 5)

    def test_bad_choice_falls_back(self):
        s = settings_mod.typed_settings({"tg_group_mode": "everything"})
        self.assertEqual(s["tg_group_mode"], "mention")


class LLMErrorsTest(unittest.TestCase):
    def test_http_error_raises_clean(self):
        with mock.patch("bot.llm.request_json") as m:
            m.side_effect = RuntimeError("HTTP 500; endpoint rejected the request")
            with self.assertRaises(RuntimeError) as c:
                llm_mod.chat_completion("http://x", "m", "", "sys", [], 0.7,
                                        10, 0.9, 5)
            self.assertIn("500", str(c.exception))

    def test_connection_error_raises_clean(self):
        with mock.patch("bot.llm.request_json",
                        side_effect=RuntimeError("HTTP request failed (ConnectionRefusedError)")):
            with self.assertRaises(RuntimeError):
                llm_mod.chat_completion("http://x", "m", "", "sys", [], 0.7,
                                        10, 0.9, 5)

    def test_bad_payload_raises_clean(self):
        with mock.patch("bot.llm.request_json", return_value={"nope": 1}):
            with self.assertRaises(RuntimeError):
                llm_mod.chat_completion("http://x", "m", "", "sys", [], 0.7,
                                        10, 0.9, 5)


if __name__ == "__main__":
    unittest.main()
