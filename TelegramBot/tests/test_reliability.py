"""Regression coverage using temporary storage and simulated services only."""
import io
import json
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock
import app
from bot.db import Database
from bot.poller import Poller
from bot import settings, llm, http_json
from bot.telegram_api import TelegramClient, TelegramError
from tests.test_e2e import StubClient


class BotRegression(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Database(Path(self.tmp.name) / "test.db")
        self.addCleanup(self.db.close)
        self.client = StubClient()
        self.s = settings.typed_settings({"tg_typing": "false", "llm_model": "fixture"})
        self.p = Poller(self.db, lambda: self.s, lambda: self.client)
        self.db.upsert_user(1, "Alice", chat_id=1)

    def item(self, text="hello", **extra):
        return dict(user_id=1, chat_id=1, chat_type="private", text=text, tg_msg_id=1, **extra)

    def update(self, text, cid=1, uid=1, **extra):
        return {"update_id": 10, "message": {"message_id": 5, "from": {"id": uid},
                "chat": {"id": cid, "type": "private" if cid > 0 else "group"}, "text": text, **extra}}

    def test_recent_history_not_oldest_page(self):
        for i in range(6): self.db.add_message(1,"in",str(i))
        self.assertEqual([x["text"] for x in self.db.history(1,limit=2)], ["4","5"])

    def test_private_and_groups_are_isolated(self):
        self.db.add_message(1,"in","private secret",chat_id=1)
        self.db.add_message(1,"in","group question",chat_id=-9)
        group = self.p.context(1,-9)
        self.assertEqual([x["content"] for x in group["messages"]],["group question"])

    def test_queued_message_is_durable_and_deduplicated(self):
        job = self.db.create_job(self.item(update_id=5))
        self.assertEqual(self.db.job_status(job["id"]),"queued")
        self.assertIsNone(self.db.create_job(self.item(update_id=5)))
        self.assertEqual(len(self.db.history(1)),1)

    def test_restart_recovers_queued_but_not_uncertain_delivery(self):
        queued = self.db.create_job(self.item("queued"))
        running = self.db.create_job(self.item("running"))
        self.db.set_job_status(running["id"],"delivering")
        self.assertEqual([x["id"] for x in self.db.recover_jobs()],[queued["id"]])
        self.assertEqual(self.db.job_status(running["id"]),"interrupted")

    def test_cancel_during_generation_discards_reply(self):
        def generate(*args):
            self.p.cancel(1,1)
            return "stale reply"
        with mock.patch("bot.llm.chat_completion",side_effect=generate):
            self.p._generate(self.item())
        self.assertEqual(self.client.sent,[])
        self.assertEqual(self.p._llm_active,0)

    def test_reset_during_generation_discards_reply(self):
        def generate(*args):
            self.p.reset(1,1)
            return "stale reply"
        with mock.patch("bot.llm.chat_completion",side_effect=generate):
            self.p._generate(self.item())
        self.assertFalse(any(x["in_context"] for x in self.db.history(1)))
        self.assertEqual(self.client.sent,[])

    def test_failed_delivery_not_added_to_context(self):
        with mock.patch("bot.llm.chat_completion",return_value="unsent answer"), mock.patch.object(self.client,"send_text",side_effect=TelegramError("blocked",code=403)):
            self.p._generate(self.item())
        self.assertFalse(any(x["text"] == "unsent answer" for x in self.db.history(1)))
        self.assertEqual(self.p.context(1,1)["messages"],[])

    def test_stopped_wait_does_not_call_model_or_release_other_slot(self):
        self.p._stop.set()
        self.p._llm_active = 1
        with mock.patch("bot.llm.chat_completion") as call:
            self.p._generate(self.item())
        call.assert_not_called()
        self.assertEqual(self.p._llm_active,1)

    def test_successful_retry_does_not_repeat_old_failure(self):
        job = self.db.create_job(self.item())
        self.db.set_job_status(job["id"],"failed")
        completed = self.db.create_job(self.item())
        self.db.set_job_status(completed["id"],"done")
        self.assertFalse(self.p.retry(1,1))

    def test_duplicate_data_directory_instance_is_refused(self):
        lock = app.lock_data_directory(Path(self.tmp.name))
        try:
            with self.assertRaisesRegex(RuntimeError,"Another"):
                app.lock_data_directory(Path(self.tmp.name))
        finally:
            lock.close()
        new_lock = app.lock_data_directory(Path(self.tmp.name))
        new_lock.close()

    def test_group_cannot_claim_administrator(self):
        self.p._maybe_claim_admin(self.db.get_user(1), self.s,"group")
        self.assertEqual(self.db.get_user(1)["role"],"user")

    def test_explicit_owner_and_fresh_role_for_first_stats(self):
        self.s["bot_admin_user_id"] = "2"
        self.p._handle_update(self.client,self.s,self.update("/stats"))
        self.assertEqual(self.db.get_user(1)["role"],"user")
        self.p._handle_update(self.client,self.s,self.update("/stats",uid=2,cid=2))
        self.assertIn("Users:",self.client.sent[-1][1])

    def test_foreign_bot_reply_does_not_address_this_bot(self):
        self.p._username, self.p._bot_id = "relay",99
        self.assertFalse(self.p._is_addressed({"reply_to_message":{"from":{"id":88,"is_bot":True}}},"hi"))
        self.assertTrue(self.p._is_addressed({"reply_to_message":{"from":{"id":99,"is_bot":True}}},"hi"))
        self.assertFalse(self.p._is_addressed({},"@relay_other"))
        self.assertTrue(self.p._is_addressed({},"😀 @relay hello"))

    def test_commands_for_other_bot_ignored(self):
        self.p._username = "relay"
        self.p._handle_update(self.client,self.s,self.update("/reset@elsewhere"))
        self.assertEqual(self.client.sent,[])

    def test_chat_prompt_and_reset_are_scoped(self):
        self.db.add_message(1,"in","group",chat_id=-1)
        self.db.add_message(1,"in","private",chat_id=1)
        self.p._handle_command(self.client,self.s,self.db.get_user(1),1,"private","/prompt Be brief",1)
        self.assertEqual(self.p.context(1,1)["system"],"Be brief")
        self.assertEqual(self.p.context(1,-1)["messages"][0]["content"],"group")

    def test_blank_fallback_and_system_prompt_stay_blank(self):
        typed = settings.typed_settings({"tg_unsupported":"", "gen_system_prompt":""})
        self.assertEqual(typed["tg_unsupported"],"")
        self.assertEqual(typed["gen_system_prompt"],"")

    def test_settings_validation_rejects_invalid_and_remote(self):
        for values in ({"gen_temperature":"nan"},{"gen_top_p":"2"},{"ctx_history_limit":"0"},{"tg_enabled":"perhaps"},{"llm_base_url":"https://example.com/v1"}):
            with self.subTest(values=values),self.assertRaises(ValueError):
                settings.validate_settings(values,{})
        settings.validate_settings({"llm_base_url":"https://example.com/v1","llm_allow_remote":"true"},{})

    def test_newest_user_preserved_with_trailing_assistant(self):
        h = [dict(id=1,direction="in",kind="text",status="done",in_context=1,text="latest user"),
             dict(id=2,direction="out",kind="text",status="done",in_context=1,text="a"*5000)]
        ctx = llm.build_context(h,"system",30,100,12000)
        self.assertIn(1,ctx["included_ids"])
        self.assertLessEqual(ctx["est_tokens"],100)

    def test_single_oversized_message_fits_cap(self):
        self.db.add_message(1,"in","x"*10000,chat_id=1)
        ctx = llm.build_context(self.db.history(1),"s"*2000,30,256,12000)
        self.assertLessEqual(ctx["est_tokens"],256)

    def test_http_429_parses_retry_after_and_redacts(self):
        token = "123:fixture-secret"
        body = {"ok":False,"description":token,"parameters":{"retry_after":4}}
        with mock.patch("bot.telegram_api.request_json",return_value=(429,body)):
            with self.assertRaises(TelegramError) as ctx:
                TelegramClient(token).call("sendMessage")
        self.assertEqual(ctx.exception.retry_after,4)
        self.assertNotIn(token,str(ctx.exception))


class DashboardRegression(unittest.TestCase):
    def setUp(self):
        BotRegression.setUp(self)
        self.saved = app.DB,app.POLLER
        app.DB,app.POLLER = self.db,self.p
        app.PROBE_CACHE.clear()
        self.server = ThreadingHTTPServer(("127.0.0.1",0),app.Handler)
        threading.Thread(target=self.server.serve_forever,daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        self.addCleanup(self.cleanup_server)

    def cleanup_server(self):
        self.server.shutdown(); self.server.server_close()
        app.DB,app.POLLER = self.saved

    def request(self,path,body=None,headers=None,raw=None):
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        req = urllib.request.Request(self.base+path,data=data,headers=headers or {})
        try:
            with urllib.request.urlopen(req,timeout=3) as response:
                return response.status,json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code,json.loads(error.read())

    def test_csrf_required_and_cross_origin_refused(self):
        self.assertEqual(self.request("/api/settings",{"tg_enabled":"false"})[0],403)
        self.assertEqual(self.request("/api/settings",{}, {"X-Relay-CSRF":app.CSRF_TOKEN,"Origin":"https://attacker.invalid"})[0],403)

    def test_settings_write_atomic_and_bad_json_rejected(self):
        headers = {"X-Relay-CSRF":app.CSRF_TOKEN}
        self.assertEqual(self.request("/api/settings",{"tg_welcome":"changed","gen_temperature":"nan"},headers)[0],400)
        self.assertEqual(self.db.get_setting("tg_welcome"),"")
        self.assertEqual(self.request("/api/settings",headers=headers,raw=b"bad")[0],400)
        self.assertEqual(self.request("/api/settings",{"tg_welcome":"hello"},headers)[0],200)

    def test_last_admin_cannot_be_deleted(self):
        self.db.set_user_fields(1,role="admin")
        self.assertEqual(self.request("/api/users/1/delete",{}, {"X-Relay-CSRF":app.CSRF_TOKEN})[0],400)

    def test_dashboard_reset_route_and_export(self):
        self.db.add_message(1,"in","private",chat_id=1)
        self.db.add_message(1,"in","group",chat_id=-1)
        self.assertEqual(self.request("/api/users/1/reset-context",{"chat_id":1},{"X-Relay-CSRF":app.CSRF_TOKEN})[0],200)
        code,body = self.request("/api/users/1/export?chat_id=1")
        self.assertEqual(code,200)
        self.assertFalse(any(x["text"] == "group" for x in body["messages"]))

    def test_probe_result_persists_between_refreshes(self):
        with mock.patch("app.check_telegram",return_value={"connected":True}),mock.patch("app.check_llm",return_value={"connected":True}) as probe:
            self.assertTrue(self.request("/api/status")[1]["llm"]["connected"])
            self.assertTrue(self.request("/api/status")[1]["llm"]["connected"])
            self.assertEqual(probe.call_count,1)


class TransportRegression(unittest.TestCase):
    def test_sustained_low_ram_stops_through_dashboard_only(self):
        class Slow(BaseHTTPRequestHandler):
            def log_message(self,*a): pass
            def do_GET(self):
                time.sleep(1)
                try:
                    self.send_response(200); self.end_headers(); self.wfile.write(b'{}')
                except OSError: pass
        server = ThreadingHTTPServer(("127.0.0.1",0),Slow)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever,daemon=True).start()
        real_clock = time.monotonic
        start = real_clock()
        def accelerated(): return start + (real_clock()-start)*20
        try:
            with mock.patch("bot.http_json.available_ram",side_effect=[2*1024**3]+[500*1024**2]*30),mock.patch("bot.http_json.time.monotonic",side_effect=accelerated),mock.patch("bot.http_json.emergency_stop_model") as stop:
                with self.assertRaisesRegex(RuntimeError,"watchdog"):
                    http_json.request_json(f"http://127.0.0.1:{server.server_port}",timeout=100,guard_model=True)
                stop.assert_called_once()
        finally:
            server.shutdown(); server.server_close()

    def test_redirect_does_not_forward_payload(self):
        class Redirect(BaseHTTPRequestHandler):
            def log_message(self,*a): pass
            def do_GET(self):
                self.send_response(302)
                self.send_header("Location","https://example.invalid/should-not-be-contacted")
                self.end_headers()
        server = ThreadingHTTPServer(("127.0.0.1",0),Redirect)
        threading.Thread(target=server.serve_forever,daemon=True).start()
        try:
            with self.assertRaisesRegex(RuntimeError,"302"):
                http_json.request_json(f"http://127.0.0.1:{server.server_port}")
        finally:
            server.shutdown(); server.server_close()

    def test_low_ram_prevents_model_request(self):
        with mock.patch("bot.http_json.available_ram",return_value=500*1024**2),mock.patch("http.client.HTTPConnection") as connection:
            with self.assertRaisesRegex(RuntimeError,"RAM"):
                http_json.request_json("http://127.0.0.1:10000/v1/chat/completions",{},guard_model=True)
            connection.assert_not_called()

    def test_trickled_headers_have_total_deadline(self):
        class Slow(BaseHTTPRequestHandler):
            def log_message(self,*a): pass
            def do_GET(self):
                try:
                    for _ in range(20):
                        self.wfile.write(b" "); self.wfile.flush(); time.sleep(.05)
                except OSError: pass
        server = ThreadingHTTPServer(("127.0.0.1",0),Slow)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever,daemon=True).start()
        try:
            start = time.monotonic()
            with self.assertRaises(RuntimeError):
                http_json.request_json(f"http://127.0.0.1:{server.server_port}",timeout=.25)
            self.assertLess(time.monotonic()-start,.8)
        finally:
            server.shutdown(); server.server_close()


if __name__ == "__main__": unittest.main()
