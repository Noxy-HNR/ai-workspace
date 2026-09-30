"""End-to-end: real Poller._generate against a fake OpenAI-compatible
LLM server and a stub Telegram client. Covers the happy path (reply
stored + delivered, long reply chunked) and the LLM-down path."""
import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot import db as db_mod
from bot import settings as settings_mod
from bot.poller import Poller


class FakeLLM(BaseHTTPRequestHandler):
    reply = "hello from fake model"

    def log_message(self, *args):
        pass

    def do_GET(self):
        data = json.dumps({"data": [{"id": "fake-model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        body = json.dumps({"choices": [{"message": {"content": self.reply}}]})
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class StubClient:
    configured = True

    def __init__(self):
        self.sent = []

    def send_text(self, chat_id, text):
        self.sent.append((chat_id, text))
        return len(self.sent)

    def send_chat_action(self, chat_id, action="typing"):
        pass


class E2ETest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeLLM)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever,
                         daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def make_poller(self, **overrides):
        db = db_mod.Database(Path(tempfile.mkdtemp()) / "t.db")
        self.addCleanup(db.close)
        for key, (default, *_r) in settings_mod.DEFS.items():
            db.set_setting(key, default)
        db.set_setting("llm_base_url", f"http://127.0.0.1:{self.port}")
        db.set_setting("tg_typing", "false")
        for k, v in overrides.items():
            db.set_setting(k, v)
        stub = StubClient()
        p = Poller(db, lambda: settings_mod.typed_settings(db.all_settings()),
                   lambda: stub)
        return db, p, stub

    def test_happy_path(self):
        db, p, stub = self.make_poller()
        db.upsert_user(100, "T")
        p._generate({"user_id": 100, "chat_id": 100, "chat_type": "private",
                     "text": "hi there", "tg_msg_id": 1})
        hist = db.history(100)
        self.assertEqual(len(hist), 2)
        self.assertEqual(hist[1]["direction"], "out")
        self.assertIn("fake model", hist[1]["text"])
        self.assertEqual(len(stub.sent), 1)
        self.assertEqual(stub.sent[0][0], 100)

    def test_long_reply_chunked(self):
        FakeLLM.reply = "para " + "z" * 9000
        try:
            db, p, stub = self.make_poller()
            db.upsert_user(101, "T")
            p._generate({"user_id": 101, "chat_id": 101,
                         "chat_type": "private", "text": "go",
                         "tg_msg_id": 1})
            self.assertGreater(len(stub.sent), 1)
            for _, part in stub.sent:
                self.assertLessEqual(len(part), 4000)
            # stored as ONE message (full text), delivered as chunks
            outs = [m for m in db.history(101) if m["direction"] == "out"]
            self.assertEqual(len(outs), 1)
            self.assertGreater(len(outs[0]["text"]), 9000)
        finally:
            FakeLLM.reply = "hello from fake model"

    def test_llm_down_sends_error_template(self):
        db, p, stub = self.make_poller(
            llm_base_url="http://127.0.0.1:1", llm_timeout="3")
        db.upsert_user(102, "T")
        p._generate({"user_id": 102, "chat_id": 102,
                     "chat_type": "private", "text": "hello?",
                     "tg_msg_id": 1})
        self.assertEqual(len(stub.sent), 1)
        self.assertIn("unavailable", stub.sent[0][1])
        outs = [m for m in db.history(102) if m["direction"] == "out"]
        self.assertEqual(outs[0]["kind"], "error")
        self.assertEqual(outs[0]["in_context"], 0)


    def test_rapid_messages_queue_in_order(self):
        import time as _t
        db, p, stub = self.make_poller()
        db.upsert_user(103, "T")
        p._enqueue(103, {"user_id": 103, "chat_id": 103,
                         "chat_type": "private", "text": "first",
                         "tg_msg_id": 1})
        p._enqueue(103, {"user_id": 103, "chat_id": 103,
                         "chat_type": "private", "text": "second",
                         "tg_msg_id": 2})
        deadline = _t.time() + 30
        while _t.time() < deadline:
            outs = [m for m in db.history(103)
                    if m["direction"] == "out"]
            if len(outs) >= 2:
                break
            _t.sleep(0.2)
        ins = [m["text"] for m in db.history(103)
               if m["direction"] == "in"]
        self.assertEqual(ins, ["first", "second"])
        outs = [m for m in db.history(103) if m["direction"] == "out"]
        self.assertEqual(len(outs), 2)


if __name__ == "__main__":
    unittest.main()
