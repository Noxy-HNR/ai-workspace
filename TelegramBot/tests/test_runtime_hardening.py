"""Runtime regressions: temp databases, mocked APIs and disposable HTTP servers."""
import json
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from bot import http_json, llm, settings
from bot.db import Database
from bot.poller import Poller
from bot.telegram_api import TelegramClient, TelegramError
from tests.test_e2e import StubClient


class RuntimeHardening(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db = Database(Path(self.temp.name) / "fixture.db")
        self.addCleanup(self.db.close)
        self.db.upsert_user(1, "Fixture", chat_id=1)
        self.s = settings.typed_settings({"tg_typing": "false", "llm_model": "fixture"})
        self.client = StubClient()
        self.poller = Poller(self.db, lambda: self.s, lambda: self.client)

    def job(self, text="hello", thread_id=0):
        return self.db.create_job(dict(user_id=1,chat_id=-1,chat_type="supergroup",
                                       thread_id=thread_id,text=text,tg_msg_id=11))

    def test_forum_context_and_instructions_stay_separate(self):
        self.db.add_message(1,"in","topic one secret",chat_id=-1,thread_id=10)
        self.db.add_message(1,"in","topic two question",chat_id=-1,thread_id=20)
        self.db.add_message(1,"in","general topic",chat_id=-1)
        self.db.set_chat_prompt(1,-1,"Topic one instructions",thread_id=10)
        self.assertEqual(self.poller.context(1,-1,thread_id=10)["system"],"Topic one instructions")
        self.assertEqual([m["content"] for m in self.poller.context(1,-1,thread_id=20)["messages"]],["topic two question"])
        self.assertEqual([m["content"] for m in self.poller.context(1,-1)["messages"]],["general topic"])

    def test_topic_reset_and_cancel_leave_other_topic_jobs_alone(self):
        first, other = self.job(thread_id=10), self.job(thread_id=20)
        self.poller.reset(1,-1,thread_id=10)
        self.assertEqual(self.db.job_status(first["id"]),"cancelled")
        self.assertEqual(self.db.job_status(other["id"]),"queued")

    def test_queued_turn_preserves_previous_answer_in_causal_order(self):
        first = self.job("first question",10)
        second = self.job("second question",10)
        third = self.job("future question",10)
        self.db.set_job_status(first["id"],"done")
        self.db.set_job_status(second["id"],"processing")
        self.db.set_job_status(third["id"],"done")  # Explicitly proves the cap.
        self.db.add_message(1,"out","first answer",chat_id=-1,thread_id=10,job_id=first["id"])
        ctx = self.poller.context(1,-1,thread_id=10,before_id=second["inbound_id"])
        self.assertEqual([(m["role"],m["content"]) for m in ctx["messages"]],
                         [("user","first question"),("assistant","first answer"),("user","second question")])

    def test_generation_delivers_and_stores_the_correct_topic(self):
        client = TelegramClient("123:fixture-only")
        self.poller.make_client = lambda: client
        with mock.patch("bot.telegram_api.request_json",return_value=(200,{"ok":True,"result":{"message_id":71}})) as transport, mock.patch("bot.llm.chat_completion",return_value="**Hello**") as model:
            self.poller._generate(self.job("ask",10))
        params = transport.call_args.args[1]
        self.assertEqual(params["message_thread_id"],10)
        self.assertEqual(params["reply_parameters"]["message_id"],11)
        self.assertEqual(params["parse_mode"],"HTML")
        self.assertEqual(params["text"],"<b>Hello</b>")
        answer = [m for m in self.db.history(1) if m["direction"] == "out"][0]
        self.assertEqual(answer["thread_id"],10)
        self.assertIsNotNone(answer["job_id"])
        self.assertEqual(self.poller._llm_active,0)
        self.assertEqual(self.poller._cancel_signals,{})

    def test_topic_is_propagated_from_telegram_update(self):
        self.s["tg_group_mode"] = "all"
        update = dict(update_id=7,message=dict(message_id=3,from_=None))
        update["message"] = {"message_id":3,"from":{"id":1},"chat":{"id":-1,"type":"supergroup"},"text":"hello","message_thread_id":77}
        with mock.patch.object(self.poller,"_enqueue") as enqueue:
            self.poller._handle_update(self.client,self.s,update)
        self.assertEqual(enqueue.call_args.args[1]["thread_id"],77)

    def test_identity_change_discards_stale_recovery_and_context(self):
        job = self.job("old bot pending",10)
        self.db.add_message(1,"in","old bot context",chat_id=-1,thread_id=10)
        self.db.set_chat_prompt(1,-1,"explicit prompt",thread_id=10)
        self.db.set_meta("tg_bot_id","100")
        self.poller._recovery = [job]
        self.poller._pending = {1:[job]}
        identity_client = mock.Mock(_token="123:new-fixture",get_me=mock.Mock(return_value={"id":200,"username":"new_fixture"}))
        self.poller._ensure_username(identity_client)
        self.assertEqual(self.db.job_status(job["id"]),"cancelled")
        self.assertEqual(self.poller._recovery,[])
        self.assertEqual(self.poller._pending,{})
        self.assertEqual(self.poller.context(1,-1,thread_id=10)["messages"],[])
        self.assertEqual(self.db.chat_prompt(1,-1,thread_id=10),"explicit prompt")
        self.assertGreater(len(self.db.history(1)),0)

    def test_invalid_identity_never_retires_existing_context(self):
        self.db.set_meta("tg_bot_id","100")
        self.db.add_message(1,"in","retained context",chat_id=-1)
        identity_client = mock.Mock(_token="fixture",get_me=mock.Mock(return_value={"id":None}))
        with self.assertRaisesRegex(TelegramError,"invalid bot identity"):
            self.poller._ensure_username(identity_client)
        self.assertEqual(self.poller.context(1,-1)["messages"][0]["content"],"retained context")

    def test_polling_honors_telegram_rate_limit_delay(self):
        client = mock.Mock(configured=True)
        client.get_updates.side_effect = TelegramError("rate limited",code=429,retry_after=17)
        self.poller.make_client = lambda: client
        delays = []
        def wait(delay):
            delays.append(delay)
            self.poller._stop.set()
            return True
        with mock.patch.object(self.poller,"_ensure_username"),mock.patch.object(self.poller._stop,"wait",side_effect=wait):
            self.poller._run()
        self.assertEqual(delays,[17])

    def test_queue_notice_does_not_hold_global_worker_lock(self):
        self.poller._busy.add(1)
        self.s["bot_queue_limit"] = 1
        self.poller._pending[1] = [self.job("queued")]
        entered, release, snap_done = threading.Event(),threading.Event(),threading.Event()
        def send(*args,**kwargs):
            entered.set()
            release.wait(2)
            return 1
        with mock.patch.object(self.poller,"_safe_send",side_effect=send):
            worker = threading.Thread(target=self.poller._enqueue,args=(1,dict(user_id=1,chat_id=-1,chat_type="group",text="overflow")))
            worker.start()
            self.assertTrue(entered.wait(1))
            reader = threading.Thread(target=lambda:(self.poller.snapshot(),snap_done.set()))
            reader.start()
            try:
                self.assertTrue(snap_done.wait(.5),"A slow queue notice blocked global status/cancellation")
            finally:
                release.set()
                worker.join(2)
                reader.join(2)

    def test_cancellation_interrupts_delivery_wait_without_deadlock(self):
        client = TelegramClient("123:fixture-only")
        self.poller.make_client = lambda: client
        rate_limited = threading.Event()
        def request(*args,**kwargs):
            rate_limited.set()
            return (429,{"ok":False,"error_code":429,"description":"Too Many Requests","parameters":{"retry_after":20}})
        job = self.job()
        with mock.patch("bot.telegram_api.request_json",side_effect=request),mock.patch("bot.llm.chat_completion",return_value="answer"):
            worker = threading.Thread(target=self.poller._generate,args=(job,))
            worker.start()
            self.assertTrue(rate_limited.wait(1))
            start = time.monotonic()
            self.poller.cancel(1,-1,0)
            worker.join(1)
            self.assertLess(time.monotonic()-start,1)
            self.assertFalse(worker.is_alive())
        self.assertEqual(self.db.job_status(job["id"]),"cancelled")

    def test_cancel_retains_model_slot_until_bounded_request_finishes(self):
        started, release = threading.Event(),threading.Event()
        def generate(*args):
            started.set()
            release.wait(2)
            return "discarded answer"
        job = self.job()
        with mock.patch("bot.llm.chat_completion",side_effect=generate) as model:
            worker = threading.Thread(target=self.poller._generate,args=(job,))
            worker.start()
            self.assertTrue(started.wait(1))
            try:
                self.poller.cancel(1,-1,0)
                self.assertEqual(self.poller._llm_active,1)
                self.assertTrue(worker.is_alive())
                self.assertIs(model.call_args.args[-1],self.poller._shutdown_event)
                self.assertFalse(self.poller._shutdown_event.is_set(),"Job cancellation must not detach the guarded model HTTP request")
            finally:
                release.set()
                worker.join(2)
        self.assertEqual(self.poller._llm_active,0)
        self.assertEqual(self.client.sent,[])

    def test_process_shutdown_detaches_model_request_and_joins_workers(self):
        started = threading.Event()
        def generate(*args):
            started.set()
            args[-1].wait(2)
            raise RuntimeError("shutdown transport cancelled")
        with mock.patch("bot.llm.chat_completion",side_effect=generate):
            self.poller._enqueue(1,dict(user_id=1,chat_id=-1,chat_type="group",text="pending"))
            self.assertTrue(started.wait(1))
            self.assertTrue(self.poller.shutdown(timeout=1))
        self.assertEqual(self.poller._workers,set())
        self.assertEqual(self.poller._busy,set())
        self.assertEqual(self.poller._llm_active,0)
        self.assertEqual(self.client.sent,[])

    def test_uncooperative_worker_keeps_database_teardown_blocked(self):
        started, release = threading.Event(),threading.Event()
        def generate(*args):
            started.set()
            release.wait(2)
            return "discarded"
        with mock.patch("bot.llm.chat_completion",side_effect=generate):
            self.poller._enqueue(1,dict(user_id=1,chat_id=-1,chat_type="group",text="pending"))
            self.assertTrue(started.wait(1))
            try:
                self.assertFalse(self.poller.shutdown(timeout=.05))
                self.assertTrue(self.db.get_user(1))
            finally:
                release.set()
                for worker in list(self.poller._workers):
                    worker.join(2)
        self.assertTrue(self.poller.shutdown(timeout=.1))

    def test_html_rejection_retries_plain_once(self):
        client = TelegramClient("123:fixture-only")
        with mock.patch.object(client,"send_text",side_effect=[TelegramError("can't parse entities",code=400),8]) as send:
            self.assertEqual(self.poller._safe_send(client,1,"**hello**"),8)
        self.assertEqual(send.call_count,2)
        self.assertEqual(send.call_args.args[1],"**hello**")
        self.assertNotIn("parse_mode",send.call_args.kwargs)

    def test_nonformatting_400_never_resends(self):
        client = TelegramClient("123:fixture-only")
        with mock.patch.object(client,"send_text",side_effect=TelegramError("chat not found",code=400)) as send:
            with self.assertRaises(TelegramError):
                self.poller._safe_send(client,1,"**hello**")
        self.assertEqual(send.call_count,1)

    def test_prompt_accepts_newlines_and_stays_in_topic(self):
        self.poller._handle_command(self.client,self.s,self.db.get_user(1),-1,"supergroup","/prompt\nKeep this concise",1,10)
        self.assertEqual(self.db.chat_prompt(1,-1,10),"Keep this concise")
        self.assertEqual(self.db.chat_prompt(1,-1,20),"")

    def test_job_rechecks_current_settings_after_waiting_for_model(self):
        self.poller.get_settings = lambda: dict(self.s)
        self.poller._llm_active = 1
        def acquire(*args):
            self.s["tg_enabled"] = False
            return True
        job = self.job()
        with mock.patch.object(self.poller,"_wait_llm_slot",side_effect=acquire), mock.patch("bot.llm.chat_completion") as model:
            self.poller._generate(job)
        model.assert_not_called()
        self.assertEqual(self.db.job_status(job["id"]),"cancelled")
        self.assertEqual(self.poller._llm_active,0)


class TelegramTransport(unittest.TestCase):
    def test_http_error_redacts_token_without_exception_chain(self):
        token = "123:fixture-secret"
        with mock.patch("bot.telegram_api.request_json",return_value=(429,{"ok":False,"description":token,"parameters":{"retry_after":4}})):
            with self.assertRaises(TelegramError) as raised:
                TelegramClient(token).call("sendMessage")
        self.assertEqual(raised.exception.retry_after,4)
        self.assertEqual(raised.exception.code,429)
        self.assertNotIn(token,str(raised.exception))
        self.assertIsNone(raised.exception.__cause__)

    def test_malformed_retry_after_is_safe(self):
        with mock.patch("bot.telegram_api.request_json",return_value=(429,{"ok":False,"parameters":{"retry_after":"bad"}})):
            with self.assertRaises(TelegramError) as raised:
                TelegramClient("123:fixture").call("sendMessage")
        self.assertIsNone(raised.exception.retry_after)

    def test_uncertain_send_failure_never_retries(self):
        client = TelegramClient("123:fixture")
        with mock.patch.object(client,"call",side_effect=TelegramError("connection lost")) as send:
            with self.assertRaises(TelegramError):
                client.send_text(1,"hello")
        self.assertEqual(send.call_count,1)

    def test_topic_alias_and_reply_parameters(self):
        client = TelegramClient("123:fixture")
        with mock.patch.object(client,"call",return_value={"message_id":7}) as send:
            self.assertEqual(client.send_text(-1,"hello",thread_id=10,reply_to_message_id=9),7)
        self.assertEqual(send.call_args.args[1]["message_thread_id"],10)
        self.assertTrue(send.call_args.args[1]["reply_parameters"]["allow_sending_without_reply"])

    def test_total_delivery_deadline_prevents_another_chunk(self):
        client = TelegramClient("123:fixture")
        with mock.patch.object(client,"call") as send:
            with self.assertRaisesRegex(TelegramError,"time limit"):
                client.send_text(1,"hello",delivery_deadline=time.monotonic()-1)
        send.assert_not_called()


class TextRendering(unittest.TestCase):
    def test_emoji_chunks_preserve_text_and_fit_utf16(self):
        original = " 😀 " * 2500 + "\n\n  preserved end  "
        chunks = llm.chunk_reply(original)
        self.assertEqual("".join(chunks),original)
        self.assertTrue(all(len(chunk.encode("utf-16-le"))//2 <= 4000 for chunk in chunks))

    def test_safe_markdown_does_not_allow_raw_html_or_script_links(self):
        rendered = llm.telegram_html('<script>steal()</script> **bold** `a < b` [safe](https://example.invalid) [bad](javascript:alert(1))')
        self.assertNotIn("<script>",rendered)
        self.assertIn("&lt;script&gt;",rendered)
        self.assertIn("<b>bold</b>",rendered)
        self.assertIn("<code>a &lt; b</code>",rendered)
        self.assertIn('<a href="https://example.invalid">safe</a>',rendered)
        self.assertNotIn('href="javascript:',rendered)

    def test_code_never_formats_nested_markdown(self):
        self.assertEqual(llm.telegram_html("```python\n**literal** <x>```"),"<pre>**literal** &lt;x&gt;</pre>")

    def test_history_limit_one_still_preserves_latest_user(self):
        history = [dict(id=1,direction="in",kind="text",status="done",in_context=1,text="user question"),dict(id=2,direction="out",kind="text",status="done",in_context=1,text="answer")]
        self.assertEqual(llm.build_context(history,"sys",1,100,100)["messages"],[{"role":"user","content":"user question"}])

    def test_nontext_model_content_rejected(self):
        with mock.patch("bot.llm.request_json",return_value={"choices":[{"message":{"content":[{"text":"bad"}]}}]}):
            with self.assertRaisesRegex(RuntimeError,"invalid chat"):
                llm.chat_completion("http://fixture","model","","",[],.7,10,.9,3)

    def test_oversized_system_prompt_does_not_needlessly_drop_small_turns(self):
        history = [dict(id=i,direction="in",kind="text",status="done",in_context=1,text="small message") for i in range(1,5)]
        ctx = llm.build_context(history,"s"*2000,30,100,12000)
        self.assertTrue(ctx["system_truncated"])
        self.assertEqual(ctx["included_ids"],[1,2,3,4])
        self.assertEqual(ctx["dropped_for_tokens"],0)
        self.assertLessEqual(ctx["est_tokens"],100)

    def test_context_reports_partial_message_truncation_and_empty_system_cost(self):
        original = "x"*1000
        history = [dict(id=1,direction="in",kind="text",status="done",in_context=1,text=original)]
        ctx = llm.build_context(history,"",30,50,500)
        self.assertEqual(ctx["content_trimmed_ids"],[1])
        self.assertEqual(ctx["trimmed_for_chars"],1)
        self.assertEqual(ctx["trimmed_for_tokens"],1)
        self.assertEqual(ctx["token_budget"],50)
        self.assertEqual(ctx["est_tokens"],50)
        self.assertEqual(history[0]["text"],original)
        self.assertEqual(llm.estimate_tokens(""),0)


class HTTPDeadlines(unittest.TestCase):
    def test_connection_setup_cannot_outlive_request_deadline(self):
        class SlowConnection:
            def __init__(self,*args,**kwargs): self.sock = None
            def connect(self): time.sleep(.4)
            def close(self): pass
            def request(self,*args,**kwargs): raise AssertionError("Expired connection sent a payload")
        with mock.patch("http.client.HTTPConnection",SlowConnection):
            start = time.monotonic()
            with self.assertRaisesRegex(RuntimeError,"time limit"):
                http_json.request_json("http://fixture.invalid",{},timeout=.1)
            self.assertLess(time.monotonic()-start,.3)

    def test_cancellation_closes_trickled_body(self):
        entered = threading.Event()
        class Trickle(BaseHTTPRequestHandler):
            def log_message(self,*args): pass
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                entered.set()
                try:
                    for _ in range(30):
                        self.wfile.write(b" ")
                        self.wfile.flush()
                        time.sleep(.05)
                except OSError: pass
        server = ThreadingHTTPServer(("127.0.0.1",0),Trickle)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever,daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        cancelled = threading.Event()
        def cancel():
            entered.wait(1)
            cancelled.set()
        threading.Thread(target=cancel,daemon=True).start()
        with self.assertRaisesRegex(RuntimeError,"cancelled"):
            http_json.request_json(f"http://127.0.0.1:{server.server_port}",timeout=3,cancel_event=cancelled)


if __name__ == "__main__":
    unittest.main()
