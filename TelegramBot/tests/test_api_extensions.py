"""Dashboard regression coverage using disposable databases and local fixtures.

No real bot tokens, Telegram requests, or model requests are used here.
"""
import json
import sqlite3
import tempfile
import threading
import unittest
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import app
from bot.db import Database
from bot.poller import Poller
from bot import settings
from tests.test_e2e import StubClient


class TopicStubClient(StubClient):
    def __init__(self):
        super().__init__()
        self.topics = []

    def send_text(self, chat_id, text, thread_id=0, **kwargs):
        self.topics.append((chat_id, thread_id))
        return super().send_text(chat_id, text)


class DashboardExtensions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="relay-api-test-")
        self.addCleanup(self.tmp.cleanup)
        self.db = Database(Path(self.tmp.name) / "fixture.db")
        self.addCleanup(self.db.close)
        self.s = settings.typed_settings({"tg_typing": "false", "llm_model": "fixture"})
        self.client = TopicStubClient()
        self.poller = Poller(self.db, lambda: self.s, lambda: self.client)
        self.saved = app.DB, app.POLLER
        app.DB, app.POLLER = self.db, self.poller
        client_patch = mock.patch("app.make_client", return_value=self.client)
        client_patch.start()
        self.addCleanup(client_patch.stop)
        self.db.upsert_user(1, "Alex", chat_id=1)
        self.db.upsert_user(2, "Sam", chat_id=2)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        self.addCleanup(self.cleanup_server)

    def cleanup_server(self):
        self.server.shutdown()
        self.server.server_close()
        app.DB, app.POLLER = self.saved

    def request(self, path, body=None, headers=None):
        request_headers = {"X-Relay-CSRF": app.CSRF_TOKEN}
        request_headers.update(headers or {})
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, headers=request_headers)
        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def test_parallel_revocation_preserves_one_administrator(self):
        """Both HTTP requests must not act on the same stale admin count."""
        self.db.set_user_fields(1, role="admin")
        self.db.set_user_fields(2, role="admin")
        original_query = self.db.query
        original_query_one = self.db.query_one
        simultaneous_count = threading.Barrier(2)

        def coordinate(sql, result):
            if "FROM users WHERE role='admin'" in sql:
                count = (result["n"] if isinstance(result, sqlite3.Row)
                         and "n" in result.keys() else len(result))
                if count == 2:
                    try:
                        simultaneous_count.wait(timeout=0.15)
                    except threading.BrokenBarrierError:
                        pass
            return result

        def query(sql, params=()):
            return coordinate(sql, original_query(sql, params))

        def query_one(sql, params=()):
            return coordinate(sql, original_query_one(sql, params))

        with mock.patch.object(self.db, "query", side_effect=query), \
                mock.patch.object(self.db, "query_one", side_effect=query_one), \
                ThreadPoolExecutor(max_workers=2) as pool:
            replies = list(pool.map(lambda uid: self.request(
                f"/api/users/{uid}/revoke-admin", {}), (1, 2)))
        self.assertEqual(sorted(code for code, _ in replies), [200, 400])
        self.assertEqual(original_query_one(
            "SELECT COUNT(*) AS n FROM users WHERE role='admin'")["n"], 1)

    def test_parallel_deletion_preserves_one_administrator(self):
        self.db.set_user_fields(1, role="admin")
        self.db.set_user_fields(2, role="admin")
        original = self.db.query_one
        simultaneous_count = threading.Barrier(2)

        def query_one(sql, params=()):
            result = original(sql, params)
            if "FROM users WHERE role='admin'" in sql and result["n"] == 2:
                try:
                    simultaneous_count.wait(timeout=0.15)
                except threading.BrokenBarrierError:
                    pass
            return result

        with mock.patch.object(self.db, "query_one", side_effect=query_one), \
                ThreadPoolExecutor(max_workers=2) as pool:
            replies = list(pool.map(lambda uid: self.request(
                f"/api/users/{uid}/delete", {}), (1, 2)))
        self.assertEqual(sorted(code for code, _ in replies), [200, 400])
        self.assertEqual(original(
            "SELECT COUNT(*) AS n FROM users WHERE role='admin'")["n"], 1)

    def test_message_pagination_is_stable_and_scoped_to_user_chat_topic(self):
        target = [self.db.add_message(1, "in", f"Target {i}", chat_id=-100,
                   thread_id=42, chat_type="supergroup") for i in range(7)]
        self.db.add_message(1, "in", "Other topic", chat_id=-100, thread_id=73)
        self.db.add_message(1, "in", "Private", chat_id=1)
        self.db.add_message(2, "in", "Other user's same topic", chat_id=-100, thread_id=42)
        path = "/api/users/1/messages?chat_id=-100&thread_id=42&limit=2"
        code, page = self.request(path)
        self.assertEqual(code, 200)
        self.assertEqual([row["id"] for row in page["messages"]], target[-2:])
        self.assertTrue(page["has_more"])
        self.assertEqual(page["next_before"], target[-2])
        newest = self.db.add_message(1, "in", "Arrived between pages", chat_id=-100, thread_id=42)
        seen = [row["id"] for row in page["messages"]]
        while page["has_more"]:
            code, page = self.request(path + f"&before={page['next_before']}")
            self.assertEqual(code, 200)
            seen.extend(row["id"] for row in page["messages"])
        self.assertIsNone(page["next_before"])
        self.assertEqual(sorted(seen), target)
        self.assertEqual(len(seen), len(set(seen)))
        self.assertNotIn(newest, seen)

    def test_message_search_treats_sql_wildcards_as_literal_text(self):
        for text in ("Progress: 50%", "Progress: 500", "A_B", "ACB", "C:\\notes", "C:notes"):
            self.db.add_message(1, "in", text, chat_id=1)
        for query, expected in (("%", "Progress: 50%"), ("_", "A_B"), ("\\", "C:\\notes")):
            with self.subTest(query=query):
                code, body = self.request("/api/users/1/messages?" + urllib.parse.urlencode({"search":query}))
                self.assertEqual(code, 200)
                self.assertEqual([row["text"] for row in body["messages"]], [expected])

    def test_chats_keep_topic_metadata_and_users_separate(self):
        for uid, cid, thread in ((1, 1, 0), (1, -100, 42), (1, -100, 73), (2, -100, 42)):
            self.db.add_message(uid, "in", f"Fixture {uid}/{thread}", chat_id=cid, thread_id=thread)
        code, _ = self.request("/api/users/1/chat-meta", {"chat_id":-100,"thread_id":42,"label":"  Chemistry  ","bookmarked":True})
        self.assertEqual(code, 200)
        code, body = self.request("/api/users/1/chats")
        self.assertEqual(code, 200)
        chats = {(row["chat_id"], row["thread_id"]):row for row in body["chats"]}
        self.assertEqual(set(chats), {(1, 0), (-100, 42), (-100, 73)})
        self.assertEqual(body["chats"][0]["label"], "Chemistry")
        self.assertEqual(chats[(-100, 42)]["bookmarked"], 1)
        self.assertEqual(chats[(-100, 42)]["messages"], 1)
        self.assertEqual(chats[(-100, 73)]["label"], "")
        other = self.request("/api/users/2/chats")[1]["chats"][0]
        self.assertEqual(other["label"], "")
        self.assertEqual(other["bookmarked"], 0)

    def test_chat_metadata_requires_real_conversation_and_valid_body(self):
        self.db.add_message(1, "in", "private", chat_id=1)
        for body, expected in (
            ({"chat_id":-999,"label":"Missing"},404),
            ({"chat_id":1,"thread_id":7,"label":"Missing topic"},404),
            ({"chat_id":1,"label":"x" * 101},400),
            ({"chat_id":1,"label":None},400),
            ({"chat_id":1,"bookmarked":"yes"},400),
            ({"chat_id":1,"bookmarked":0.0},400),
            ({"chat_id":1,"bookmarked":1.0},400),
            ({"chat_id":True,"label":"Bad ID"},400),
            ({"chat_id":1,"thread_id":-1},400),
        ):
            with self.subTest(body=body):
                self.assertEqual(self.request("/api/users/1/chat-meta",body)[0], expected)
        self.assertEqual(self.db.query_one("SELECT COUNT(*) AS n FROM chat_metadata")["n"], 0)

    def test_chat_prompt_retires_only_selected_topic_and_cancels_its_queue(self):
        for thread in (42, 73):
            self.db.add_message(1, "in", f"Context {thread}", chat_id=-100, thread_id=thread)
            self.db.create_job(dict(user_id=1,chat_id=-100,thread_id=thread,
                chat_type="supergroup",text=f"Queued {thread}",update_id=thread))
        self.db.set_chat_prompt(1, -100, "Other instructions", 73)
        code, _ = self.request("/api/users/1/chat-prompt", {"chat_id":-100,"thread_id":42,"prompt":"Explain chemistry briefly."})
        self.assertEqual(code, 200)
        self.assertEqual(self.poller.context(1,-100,self.s,thread_id=42)["system"], "Explain chemistry briefly.")
        self.assertEqual(self.poller.context(1,-100,self.s,thread_id=42)["messages"], [])
        other = self.poller.context(1,-100,self.s,thread_id=73)
        self.assertEqual(other["system"], "Other instructions")
        self.assertIn("Context 73", [row["content"] for row in other["messages"]])
        jobs = {row["thread_id"]:row["status"] for row in self.db.query("SELECT * FROM jobs")}
        self.assertEqual(jobs, {42:"cancelled",73:"queued"})

    def test_empty_chat_prompt_returns_to_default_without_affecting_other_topic(self):
        self.db.add_message(1,"in","topic",chat_id=-100,thread_id=42)
        self.db.set_chat_prompt(1,-100,"Custom",42)
        self.db.set_chat_prompt(1,-100,"Other",73)
        self.assertEqual(self.request("/api/users/1/chat-prompt",{"chat_id":-100,"thread_id":42,"prompt":""})[0],200)
        self.assertEqual(self.db.chat_prompt(1,-100,42),"")
        self.assertEqual(self.db.chat_prompt(1,-100,73),"Other")
        self.assertEqual(self.poller.context(1,-100,self.s,thread_id=42)["system"],self.s["gen_system_prompt"])

    def test_jobs_filter_rows_and_report_state_counts(self):
        for uid, status in ((1,"queued"),(2,"queued"),(1,"failed"),(1,"interrupted")):
            job = self.db.create_job(dict(user_id=uid,chat_id=uid,chat_type="private",text=status))
            self.db.set_job_status(job["id"],status)
        code, body = self.request("/api/jobs?status=queued&user_id=1&limit=1")
        self.assertEqual(code,200)
        self.assertEqual(len(body["jobs"]),1)
        self.assertEqual(body["jobs"][0]["user_id"],1)
        self.assertEqual(body["jobs"][0]["status"],"queued")
        self.assertEqual(body["jobs"][0]["first_name"],"Alex")
        self.assertEqual(body["counts"],{"queued":2,"failed":1,"interrupted":1})
        self.assertEqual(self.request("/api/jobs?status=unknown")[0],400)

    def test_job_and_message_page_limits_are_bounded(self):
        with self.db._lock,self.db._conn:
            self.db._conn.executemany("INSERT INTO messages(user_id,chat_id,direction,text) VALUES(1,1,'in',?)",
                                      [(str(i),) for i in range(550)])
            self.db._conn.executemany("INSERT INTO jobs(user_id,chat_id,chat_type,text,created_at,updated_at) VALUES(1,1,'private',?,'','')",
                                      [(str(i),) for i in range(550)])
        self.assertEqual(len(self.request("/api/jobs?limit=999999")[1]["jobs"]),500)
        page = self.request("/api/users/1/messages?limit=999999")[1]
        self.assertEqual(len(page["messages"]),500)
        self.assertTrue(page["has_more"])
        for path in ("/api/jobs?limit=invalid","/api/users/1/messages?before=0","/api/users/1/messages?thread_id=-1"):
            self.assertEqual(self.request(path)[0],400)

    def test_export_explicitly_reports_truncation_and_keeps_topic_scope(self):
        with self.db._lock,self.db._conn:
            self.db._conn.executemany("INSERT INTO messages(user_id,chat_id,thread_id,direction,text) VALUES(1,-100,42,'in',?)",
                                      [(str(i),) for i in range(10001)])
        self.db.add_message(1,"in","Private secret",chat_id=1)
        self.db.add_message(1,"in","Other topic",chat_id=-100,thread_id=73)
        code, body = self.request("/api/users/1/export?chat_id=-100&thread_id=42")
        self.assertEqual(code,200)
        self.assertEqual(body["total"],10001)
        self.assertEqual(body["exported"],10000)
        self.assertTrue(body["truncated"])
        self.assertEqual(body["thread_id"],42)
        self.assertTrue(body["exported_at"].endswith("Z"))
        self.assertEqual(body["messages"][0]["text"],"1")
        self.assertTrue(all(row["chat_id"]==-100 and row["thread_id"]==42 for row in body["messages"]))

    def test_topic_send_uses_selected_destination_and_records_correct_scope(self):
        self.db.add_message(1,"in","Question",chat_id=-100,thread_id=42,chat_type="supergroup")
        self.db.add_message(1,"in","Other topic",chat_id=-100,thread_id=73,chat_type="supergroup")
        code, _ = self.request("/api/users/1/send",{"chat_id":-100,"thread_id":42,"text":"Reviewed example"})
        self.assertEqual(code,200)
        self.assertEqual(self.client.topics,[(-100,42)])
        stored = self.db.query_one("SELECT * FROM messages ORDER BY id DESC LIMIT 1")
        self.assertEqual((stored["chat_id"],stored["thread_id"],stored["chat_type"]),(-100,42,"supergroup"))
        self.assertEqual(self.request("/api/users/1/send",{"chat_id":-100,"thread_id":999,"text":"Missing"})[0],400)
        self.assertEqual(len(self.client.sent),1)

    def test_manual_send_to_blocked_user_never_calls_transport(self):
        self.db.add_message(1,"in","Private question",chat_id=1)
        self.db.set_user_fields(1,blocked=1)
        code, _ = self.request("/api/users/1/send",{"chat_id":1,"text":"Should stay local"})
        self.assertEqual(code,400)
        self.assertEqual(self.client.sent,[])
        self.assertEqual(self.db.user_stats(1)["outbound"],0)

    def test_presets_validate_create_update_and_unknown_ids(self):
        code, body = self.request("/api/presets",{"name":"  Study partner  ","prompt":"  Ask one question at a time.  "})
        self.assertEqual(code,200)
        preset = body["preset"]
        self.assertEqual(preset["name"],"Study partner")
        self.assertEqual(preset["prompt"],"Ask one question at a time.")
        code, body = self.request("/api/presets",{"id":preset["id"],"name":"Revised","prompt":"Keep answers brief."})
        self.assertEqual(code,200)
        self.assertEqual(body["preset"]["id"],preset["id"])
        self.assertEqual(self.request("/api/presets")[1]["presets"][0]["name"],"Revised")
        self.assertEqual(self.request("/api/presets",{"id":999,"name":"Absent","prompt":"Instructions"})[0],404)
        for invalid in ({"name":"","prompt":"Valid"},{"name":"Valid","prompt":" "},{"name":"x"*101,"prompt":"Valid"},{"name":"Valid","prompt":"x"*4001}):
            self.assertEqual(self.request("/api/presets",invalid)[0],400)
        self.assertEqual(len(self.request("/api/presets")[1]["presets"]),1)

    def test_extension_mutations_require_csrf_and_same_origin(self):
        self.db.add_message(1,"in","private",chat_id=1)
        for path, body in (("/api/users/1/chat-meta",{"chat_id":1,"label":"Unauthorized"}),
                           ("/api/presets",{"name":"Unauthorized","prompt":"Fixture"})):
            self.assertEqual(self.request(path,body,{"X-Relay-CSRF":""})[0],403)
            self.assertEqual(self.request(path,body,{"Origin":"https://attacker.invalid"})[0],403)
        self.assertEqual(self.db.query_one("SELECT COUNT(*) AS n FROM chat_metadata")["n"],0)
        self.assertEqual(self.db.query_one("SELECT COUNT(*) AS n FROM prompt_presets")["n"],0)
        self.assertEqual(self.request("/api/jobs",headers={"Host":"attacker.invalid"})[0],403)

    def test_unknown_nested_user_routes_do_not_fall_back_to_profile(self):
        self.assertEqual(self.request("/api/users/1/unknown")[0],404)
        self.assertEqual(self.request("/api/users/1/messages/extra")[0],404)
        self.assertEqual(self.request("/api/users/1/block/extra",{})[0],404)
        self.assertEqual(self.db.get_user(1)["blocked"],0)

    def test_events_filter_literal_search_and_bound_large_pages(self):
        for i in range(1050):
            self.db.log("info","fixture",f"Event {i}")
        self.db.log("warning","transport","Rate limit at 50%")
        self.db.log("warning","transport","Rate limit at 500")
        code, body = self.request("/api/events?limit=999999")
        self.assertEqual(code,200)
        self.assertEqual(len(body["events"]),1000)
        self.assertIn("transport",body["sources"])
        code, body = self.request("/api/events?"+urllib.parse.urlencode({"level":"warning","source":"transport","search":"%"}))
        self.assertEqual(code,200)
        self.assertEqual([row["message"] for row in body["events"]],["Rate limit at 50%"])

    def test_profile_validation_is_atomic_and_good_edit_saves_all_fields(self):
        self.db.set_user_fields(1,notes="Original note",context_limit_override=12,
                                system_prompt_override="Original instructions")
        for invalid in (
            {"notes":"Should not save","context_limit_override":"invalid"},
            {"notes":"Should not save","context_limit_override":True},
            {"notes":"Should not save","system_prompt_override":[]},
            {"notes":"x"*2001,"context_limit_override":20},
        ):
            with self.subTest(invalid=invalid):
                self.assertEqual(self.request("/api/users/1/profile",invalid)[0],400)
                user = self.db.get_user(1)
                self.assertEqual(user["notes"],"Original note")
                self.assertEqual(user["context_limit_override"],12)
                self.assertEqual(user["system_prompt_override"],"Original instructions")
        code, body = self.request("/api/users/1/profile",{"notes":"Revised note","context_limit_override":25,"system_prompt_override":"Revised instructions"})
        self.assertEqual(code,200)
        self.assertEqual(body["user"]["notes"],"Revised note")
        self.assertEqual(body["user"]["context_limit_override"],25)
        self.assertEqual(body["user"]["system_prompt_override"],"Revised instructions")
        self.assertEqual(self.request("/api/users/1/profile",{"context_limit_override":None})[0],200)
        self.assertIsNone(self.db.get_user(1)["context_limit_override"])

    def test_explicit_secret_clear_is_atomic_and_responses_stay_masked(self):
        self.db.set_setting("tg_bot_token","fixture-bot-token")
        self.db.set_setting("llm_api_key","fixture-api-key")
        for invalid in ("tg_bot_token",["tg_bot_token","gen_temperature"],[None]):
            with self.subTest(invalid=invalid):
                code, _ = self.request("/api/settings",{"settings":{"tg_welcome":"Should not save"},"clear_secrets":invalid})
                self.assertEqual(code,400)
                self.assertEqual(self.db.get_setting("tg_bot_token"),"fixture-bot-token")
                self.assertEqual(self.db.get_setting("tg_welcome"),"")
        code, _ = self.request("/api/settings",{"settings":{"gen_temperature":"nan"},"clear_secrets":["tg_bot_token"]})
        self.assertEqual(code,400)
        self.assertEqual(self.db.get_setting("tg_bot_token"),"fixture-bot-token")
        code, body = self.request("/api/settings",{"settings":{"tg_welcome":"Saved welcome"},"clear_secrets":["tg_bot_token"]})
        self.assertEqual(code,200)
        self.assertEqual(self.db.get_setting("tg_bot_token"),"")
        self.assertEqual(self.db.get_setting("llm_api_key"),"fixture-api-key")
        self.assertFalse(body["settings"]["tg_bot_token"]["set"])
        self.assertTrue(body["settings"]["llm_api_key"]["set"])
        self.assertNotIn("fixture-api-key",json.dumps(body))
        code, _ = self.request("/api/settings",{"settings":{"llm_api_key":""}})
        self.assertEqual(code,200)
        self.assertEqual(self.db.get_setting("llm_api_key"),"fixture-api-key")

    def test_delete_cannot_be_followed_by_orphaned_manual_send_history(self):
        self.db.add_message(1,"in","Private question",chat_id=1)
        sending = threading.Event()
        release = threading.Event()
        original = self.client.send_text

        def delayed_send(chat_id,text,**kwargs):
            sending.set()
            if not release.wait(timeout=2):
                raise RuntimeError("Fixture send was not released")
            return original(chat_id,text,**kwargs)

        with mock.patch.object(self.client,"send_text",side_effect=delayed_send), \
                ThreadPoolExecutor(max_workers=2) as pool:
            sent = pool.submit(self.request,"/api/users/1/send",{"chat_id":1,"text":"Manual response"})
            self.assertTrue(sending.wait(timeout=1))
            deleted = pool.submit(self.request,"/api/users/1/delete",{})
            try:
                # Old code deletes while delivery is still blocked; fixed code
                # holds the same user lock until delivery/history are recorded.
                try:
                    deleted.result(timeout=0.15)
                except TimeoutError:
                    pass
            finally:
                release.set()
            self.assertEqual(sent.result(timeout=3)[0],200)
            self.assertEqual(deleted.result(timeout=3)[0],200)
        self.assertIsNone(self.db.get_user(1))
        self.assertEqual(self.db.query_one("SELECT COUNT(*) AS n FROM messages WHERE user_id=1")["n"],0)

    def test_block_signals_active_delivery_before_waiting_for_user_lock(self):
        job = self.db.create_job(dict(user_id=1,chat_id=1,chat_type="private",text="Queued fixture"))
        cancellation = threading.Event()
        self.poller._cancel_signals[job["id"]] = (1,1,0,cancellation)
        with ThreadPoolExecutor(max_workers=1) as pool:
            with self.poller.user_lock(1):
                future = pool.submit(self.request,"/api/users/1/block",{})
                signalled_before_release = cancellation.wait(timeout=0.5)
            self.assertEqual(future.result(timeout=3)[0],200)
        self.assertTrue(signalled_before_release)
        self.assertEqual(self.db.get_user(1)["blocked"],1)
        self.assertEqual(self.db.job_status(job["id"]),"cancelled")

    def test_chat_preferences_cannot_recreate_rows_after_concurrent_delete(self):
        for action, fields in (("chat-meta",{"label":"Concurrent label","bookmarked":True}),
                               ("chat-prompt",{"prompt":"Concurrent instructions"})):
            with self.subTest(action=action):
                self.db.upsert_user(1,"Alex",chat_id=1)
                self.db.add_message(1,"in","Private fixture",chat_id=1)
                existence_checked = threading.Event()
                release = threading.Event()
                original_exists = self.db.chat_exists

                def delayed_exists(uid,cid,thread=0):
                    result = original_exists(uid,cid,thread)
                    existence_checked.set()
                    if not release.wait(timeout=2):
                        raise RuntimeError("Fixture preference check was not released")
                    return result

                with mock.patch.object(self.db,"chat_exists",side_effect=delayed_exists), \
                        ThreadPoolExecutor(max_workers=2) as pool:
                    preference = pool.submit(self.request,f"/api/users/1/{action}",{"chat_id":1,**fields})
                    self.assertTrue(existence_checked.wait(timeout=1))
                    deletion = pool.submit(self.request,"/api/users/1/delete",{})
                    try:
                        try:
                            deletion.result(timeout=0.15)
                        except TimeoutError:
                            pass
                    finally:
                        release.set()
                    self.assertIn(preference.result(timeout=3)[0],(200,404))
                    self.assertEqual(deletion.result(timeout=3)[0],200)
                self.assertIsNone(self.db.get_user(1))
                for table in ("messages","chat_metadata","chat_prompts"):
                    self.assertEqual(self.db.query_one(f"SELECT COUNT(*) AS n FROM {table} WHERE user_id=1")["n"],0)


class LegacyMigration(unittest.TestCase):
    def test_future_schema_is_refused_without_changes(self):
        with tempfile.TemporaryDirectory(prefix="relay-future-schema-") as data:
            path = Path(data) / "future.db"
            conn = sqlite3.connect(path)
            conn.executescript("CREATE TABLE future_marker(value TEXT); INSERT INTO future_marker VALUES('preserve'); PRAGMA user_version=999;")
            conn.close()
            with self.assertRaisesRegex(RuntimeError,"newer"):
                Database(path)
            conn = sqlite3.connect(path)
            try:
                self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0],999)
                self.assertEqual(conn.execute("SELECT value FROM future_marker").fetchone()[0],"preserve")
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'").fetchone()[0],1)
            finally:
                conn.close()

    def test_original_database_migrates_without_losing_chat_data(self):
        with tempfile.TemporaryDirectory(prefix="relay-migration-") as data:
            path = Path(data) / "legacy.db"
            conn = sqlite3.connect(path)
            conn.executescript("""
                CREATE TABLE messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL, chat_id INTEGER DEFAULT 0,
                    chat_type TEXT DEFAULT 'private', direction TEXT NOT NULL,
                    kind TEXT DEFAULT 'text', text TEXT DEFAULT '',
                    in_context INTEGER DEFAULT 1, status TEXT DEFAULT 'done',
                    telegram_message_id INTEGER, created_at TEXT DEFAULT ''
                );
                CREATE TABLE jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    update_id INTEGER UNIQUE, user_id INTEGER NOT NULL,
                    chat_id INTEGER NOT NULL, chat_type TEXT NOT NULL,
                    text TEXT NOT NULL, tg_msg_id INTEGER, inbound_id INTEGER,
                    status TEXT NOT NULL DEFAULT 'queued', created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE chat_prompts (
                    user_id INTEGER NOT NULL, chat_id INTEGER NOT NULL,
                    prompt TEXT NOT NULL DEFAULT '', PRIMARY KEY(user_id, chat_id)
                );
                INSERT INTO messages(user_id,chat_id,direction,text,created_at)
                    VALUES(1,-100,'in','Original conversation','2026-09-28T12:00:00Z');
                INSERT INTO jobs(update_id,user_id,chat_id,chat_type,text,inbound_id,created_at,updated_at)
                    VALUES(50,1,-100,'group','Queued before upgrade',1,
                           '2026-09-28T12:00:00Z','2026-09-28T12:00:00Z');
                INSERT INTO chat_prompts VALUES(1,-100,'Original chat instructions');
            """)
            conn.close()
            for _ in range(2):
                db = Database(path)
                try:
                    self.assertEqual(db.chat_prompt(1, -100, 0), "Original chat instructions")
                    self.assertEqual(db.chat_prompt(1, -100, 42), "")
                    row = db.query_one("SELECT * FROM messages WHERE id=1")
                    self.assertEqual(row["thread_id"], 0)
                    self.assertEqual(row["text"], "Original conversation")
                    job = db.query_one("SELECT * FROM jobs WHERE id=1")
                    self.assertEqual(job["thread_id"], 0)
                    self.assertEqual(job["status"], "queued")
                    self.assertEqual(job["update_id"], 50)
                finally:
                    db.close()


class StartupCleanup(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="relay-startup-test-")
        self.addCleanup(self.tmp.cleanup)
        self.data = Path(self.tmp.name) / "data"
        self.saved = app.DB,app.POLLER
        app.DB,app.POLLER = None,None
        self.addCleanup(self.restore_globals)

    def restore_globals(self):
        app.DB,app.POLLER = self.saved

    def assert_lock_released(self):
        handle = app.lock_data_directory(self.data)
        handle.close()

    def test_database_constructor_failure_releases_server_and_instance_lock(self):
        with mock.patch("app.DashboardServer") as server, \
                mock.patch("app.db_mod.Database",side_effect=RuntimeError("Fixture DB initialization failure")):
            with self.assertRaisesRegex(RuntimeError,"Fixture DB"):
                app.run(0,self.data)
            server.return_value.server_close.assert_called_once()
        self.assert_lock_released()

    def test_poller_constructor_failure_closes_database_server_and_lock(self):
        databases = []

        def create_database(path):
            database = Database(path)
            databases.append(database)
            return database

        with mock.patch("app.DashboardServer") as server, \
                mock.patch("app.db_mod.Database",side_effect=create_database), \
                mock.patch("app.Poller",side_effect=RuntimeError("Fixture poller initialization failure")):
            with self.assertRaisesRegex(RuntimeError,"Fixture poller"):
                app.run(0,self.data)
            server.return_value.server_close.assert_called_once()
        self.assert_lock_released()
        self.assertEqual(len(databases),1)
        with self.assertRaises(sqlite3.ProgrammingError):
            databases[0].query_one("SELECT COUNT(*) AS n FROM users")

    def test_server_request_drain_waits_until_blocked_handler_finishes(self):
        entered = threading.Event()
        release = threading.Event()

        class BlockingHandler(BaseHTTPRequestHandler):
            def log_message(self,*args):
                pass

            def do_GET(self):
                entered.set()
                release.wait(timeout=2)
                self.send_response(200)
                self.send_header("Content-Length","2")
                self.end_headers()
                self.wfile.write(b"ok")

        server = app.DashboardServer(("127.0.0.1",0),BlockingHandler)
        threading.Thread(target=server.serve_forever,daemon=True).start()

        def fetch():
            with urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/",timeout=3) as response:
                return response.read()

        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(fetch)
                self.assertTrue(entered.wait(timeout=1))
                self.assertFalse(server.wait_for_requests(timeout=0.02))
                release.set()
                self.assertEqual(future.result(timeout=3),b"ok")
                self.assertTrue(server.wait_for_requests(timeout=0.5))
        finally:
            release.set()
            server.shutdown()
            server.server_close()


class PreviewSafety(unittest.TestCase):
    def test_rich_demo_supports_retry_without_starting_external_workers(self):
        from tests.preview_dashboard import PreviewClient, PreviewPoller, seed_demo

        with tempfile.TemporaryDirectory(prefix="relay-preview-test-") as data:
            db = Database(Path(data) / "demo.db")
            try:
                seed_demo(db)
                self.assertEqual(db.counts()["users"], 6)
                self.assertGreater(db.counts()["messages"], 180)
                chats = {(row["chat_id"],row["thread_id"]):row for row in db.list_chats(101)}
                self.assertIn((-100777,42),chats)
                self.assertIn((-100777,73),chats)
                self.assertEqual(chats[(-100777,42)]["label"],"Chemistry • review")
                self.assertTrue(db.message_page(101,101,limit=50)["has_more"])
                self.assertGreater(len(db.daily_stats()),5)
                self.assertEqual(db.list_jobs()["counts"],{"queued":2,"failed":1,"interrupted":1})
                client = PreviewClient()
                s = settings.typed_settings({"tg_typing":"false"})
                poller = PreviewPoller(db,lambda:s,lambda:client)
                with mock.patch("bot.llm.chat_completion") as model, \
                        mock.patch("bot.http_json.request_json") as transport, \
                        mock.patch("threading.Thread") as worker:
                    poller.start()
                    self.assertTrue(poller.retry(202,202))
                    poller.stop()
                    model.assert_not_called()
                    transport.assert_not_called()
                    worker.assert_not_called()
                self.assertEqual(db.list_jobs()["counts"]["queued"],3)
                client.send_text(-100777,"Simulated manual reply",thread_id=42)
                self.assertEqual(client.sent,[(-100777,"Simulated manual reply")])
            finally:
                db.close()


if __name__ == "__main__":
    unittest.main()
