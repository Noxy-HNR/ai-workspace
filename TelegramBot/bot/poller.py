"""Durable Telegram queue, isolated chats and bounded model concurrency."""
from __future__ import annotations
import re
import threading
import time
import urllib.parse
from . import llm as llm_mod
from .settings import validate_endpoint
from .telegram_api import TelegramError

COMMAND_HELP = "\n\n/start — welcome\n/help — commands\n/id — your ID\n/reset — fresh context here\n/cancel — discard pending replies here\n/retry — last failed reply here\n/status — queue and model\n/context — context usage\n/prompt <text> — instructions here\n/prompt reset — default instructions\n/privacy — storage and model routing\n/stats — admin statistics\n\nPrivate chats, groups and forum topics keep separate context."


class Poller:
    def __init__(self, db, get_settings, make_client):
        self.db, self.get_settings, self.make_client = db, get_settings, make_client
        self._thread = None
        self._stop = threading.Event()
        self._shutdown_event = threading.Event()
        self._lock = threading.RLock()
        self._busy, self._pending = set(), {}
        self._user_locks = {}
        self._cancel_signals = {}
        self._workers = set()
        self._worker_users = {}
        self._llm_active = 0
        self._username = self._bot_id = self._identity_token = None
        self._recovered = False
        self._models_cache = (0.0, None, [])
        self.status = dict(running=False, last_poll="", last_error="", updates_handled=0, started_at="")

    def start(self):
        if self._shutdown_event.is_set():
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="tg-poller")
        self._thread.start()

    def stop(self, timeout=1):
        self._stop.set()
        with self._lock:
            users = list(self._busy)
        for uid in users:
            self.cancel(uid)
        if self._thread:
            self._thread.join(timeout=timeout)

    def shutdown(self, timeout=3):
        """Permanently stop transports and join threads before DB teardown.

        Unlike a chat cancellation, process teardown cannot enqueue another
        model request, so detaching its bounded HTTP read is safe here.
        Returns whether callers can safely close the shared database.
        """
        deadline = time.monotonic() + max(0,timeout)
        self._shutdown_event.set()
        self._stop.set()
        with self._lock:
            for _,_,_,signal in self._cancel_signals.values():
                signal.set()
        self.stop(timeout=max(0,deadline-time.monotonic()))
        with self._lock:
            workers = list(self._workers)
        for worker in workers:
            if worker is not threading.current_thread():
                worker.join(timeout=max(0,deadline-time.monotonic()))
        with self._lock:
            return not self._workers and self._llm_active == 0 and not (self._thread and self._thread.is_alive())

    def snapshot(self):
        with self._lock:
            return {**self.status, "stopping": self._stop.is_set() and bool(self._thread and self._thread.is_alive()),
                    "busy_users": sorted(self._busy), "llm_active": self._llm_active,
                    "pending": {str(k): sum(self.db.job_status(x["id"]) == "queued" for x in v) for k,v in self._pending.items()},
                    "bot_username": self._username}

    def _run(self):
        self.status.update(running=True, started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        try:
            if not self._recovered:
                self._recovery = self.db.recover_jobs()
                self._recovered = True
            while not self._stop.is_set():
                try:
                    s, client = self.get_settings(), self.make_client()
                    if not s["tg_enabled"] or not client.configured:
                        self.status["last_error"] = "Polling paused." if client.configured else "Bot token is not configured."
                        self._stop.wait(2)
                        continue
                    self._ensure_username(client)
                    remaining = []
                    for job in getattr(self, "_recovery", []):
                        if self.db.job_status(job["id"]) != "queued":
                            continue
                        if not self._enqueue(job["user_id"], job):
                            remaining.append(job)
                    self._recovery = remaining
                    offset = int(self.db.get_meta("tg_offset", "0") or "0")
                    updates = (client.get_updates(offset=offset, timeout_secs=5,cancel_event=self._shutdown_event)
                               if getattr(client,"supports_reply_options",False) else
                               client.get_updates(offset=offset, timeout_secs=5))
                    self.status.update(last_poll=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), last_error="")
                    for upd in updates:
                        if self._stop.is_set():
                            break
                        try:
                            self._handle_update(client, s, upd)
                        except TelegramError as e:
                            if e.code not in (400, 403):
                                raise
                            self.db.log("warning", "telegram", f"Permanent delivery failure: {e}")
                        self.db.set_meta("tg_offset", str(upd["update_id"] + 1))
                        self.status["updates_handled"] += 1
                except Exception as e:
                    self.status["last_error"] = str(e)[:200]
                    self.db.log("warning", "poller", str(e)[:500])
                    delay = e.retry_after if isinstance(e, TelegramError) and e.code == 429 and e.retry_after else 5
                    self._stop.wait(delay)
        finally:
            self.status["running"] = False

    def _ensure_username(self, client):
        token = getattr(client, "_token", None)
        if self._username and token == self._identity_token:
            return
        me = (client.get_me(cancel_event=self._shutdown_event)
              if getattr(client,"supports_reply_options",False) else client.get_me()) or {}
        if not isinstance(me,dict) or isinstance(me.get("id"),bool) or not isinstance(me.get("id"),int) or me["id"] <= 0 or not isinstance(me.get("username"),str) or not me["username"]:
            raise TelegramError("Telegram getMe returned an invalid bot identity.")
        previous_id = self.db.get_meta("tg_bot_id", "")
        if previous_id and previous_id != str(me.get("id")):
            for row in self.db.query("SELECT DISTINCT user_id FROM jobs WHERE status IN ('queued','processing','delivering')"):
                self.cancel(row["user_id"])
            self.db.execute("UPDATE jobs SET update_id=NULL")
            self.db.execute("UPDATE messages SET in_context=0")
            self._recovery = []
            with self._lock:
                self._pending.clear()
            self.db.log("warning", "telegram", "Bot identity changed; pending replies cancelled and previous context retired. Stored history and explicit instructions retained.")
            self.db.set_meta("tg_offset", "0")
        self._identity_token = token
        self._username, self._bot_id = me.get("username", ""), me.get("id")
        self.db.set_meta("tg_username", self._username)
        self.db.set_meta("tg_bot_id", str(self._bot_id))

    def _is_addressed(self, msg, text):
        if self._username and re.search(r"@" + re.escape(self._username) + r"(?![\w])", text, re.I):
            return True
        sender = (msg.get("reply_to_message") or {}).get("from") or {}
        return bool(self._bot_id and sender.get("id") == self._bot_id)

    def _maybe_claim_admin(self, user, s=None, chat_type="private"):
        s = s or self.get_settings()
        if chat_type != "private" or user.get("blocked"):
            return
        owner = s.get("bot_admin_user_id", "")
        if owner and str(user["id"]) != owner:
            return
        with self.db._lock:
            count = self.db.query_one("SELECT COUNT(*) AS n FROM users WHERE role='admin'")["n"]
            if not count:
                self.db.set_user_fields(user["id"], role="admin")
                self.db.log("info", "admin", f"Private-chat user {user['id']} claimed administrator.")

    def _allowed(self, user, s):
        return (user.get("role") == "admin" or s.get("bot_access_mode", "all") == "all" or
                (s.get("bot_access_mode") == "allowlist" and str(user["id"]) in
                 {x.strip() for x in s.get("bot_allowed_users", "").split(",")}))

    def _handle_update(self, client, s, upd):
        msg = upd.get("message")  # Edits never generate a second reply.
        if not msg:
            return
        frm, chat = msg.get("from") or {}, msg.get("chat") or {}
        uid, cid = frm.get("id"), chat.get("id")
        if not uid or not cid or frm.get("is_bot"):
            return
        ctype, text = chat.get("type", "private"), msg.get("text") or ""
        thread_id = int(msg.get("message_thread_id") or 0)
        if not text and not any(msg.get(k) for k in ("photo","sticker","voice","document","video","audio","animation","video_note")):
            return
        user = self.db.upsert_user(uid, frm.get("first_name", ""), frm.get("last_name", ""), frm.get("username", ""), cid if ctype == "private" else 0)
        self._maybe_claim_admin(user, s, ctype)
        user = self.db.get_user(uid)
        addressed = ctype == "private" or self._is_addressed(msg, text) or s["tg_group_mode"] == "all"
        if text.startswith("/"):
            target = text.split()[0].partition("@")[2]
            if target and target.lower() != (self._username or "").lower():
                addressed = False
        if user["blocked"] or not self._allowed(user, s) or not addressed:
            self.db.add_message(uid, "in", text or "[attachment]", chat_id=cid, chat_type=ctype, thread_id=thread_id, kind="ignored", in_context=False, telegram_message_id=msg.get("message_id"))
            if addressed and not user["blocked"] and not self._allowed(user, s):
                self._reply_stored(client, uid, cid, ctype, "Access is restricted. Ask the owner to allow your Telegram ID.", thread_id, msg.get("message_id"))
            return
        if not text:
            self.db.add_message(uid, "in", "[non-text attachment]", chat_id=cid, chat_type=ctype, thread_id=thread_id, kind="unsupported", in_context=False, telegram_message_id=msg.get("message_id"))
            self._reply_stored(client, uid, cid, ctype, s["tg_unsupported"], thread_id, msg.get("message_id"))
        elif text.startswith("/"):
            self._handle_command(client, s, user, cid, ctype, text, msg.get("message_id"), thread_id)
        else:
            self._enqueue(uid, dict(user_id=uid, chat_id=cid, chat_type=ctype, thread_id=thread_id, text=text, tg_msg_id=msg.get("message_id"), update_id=upd.get("update_id")))

    def context(self, uid, cid, s=None, thread_id=0, before_id=None):
        s = s or self.get_settings()
        user = self.db.get_user(uid) or {}
        system = self.db.chat_prompt(uid, cid, thread_id=thread_id) or user.get("system_prompt_override") or s["gen_system_prompt"]
        history = self.db.history(uid, eligible_only=True, chat_id=cid, thread_id=thread_id, before_id=before_id)
        # Queued input can be stored before the previous answer finishes.
        # Associate answers with their originating turn rather than insert time.
        history.sort(key=lambda message: (message.get("turn_id",message["id"]),
                                         0 if message["direction"] == "in" else 1,message["id"]))
        return llm_mod.build_context(history, system,
                                    user.get("context_limit_override") or s["ctx_history_limit"], s["ctx_max_tokens"], s["ctx_max_user_chars"])

    def _signal_cancel(self, uid, cid=None, thread_id=None):
        # Signal before waiting for the user's delivery lock. This interrupts
        # rate-limit waits and in-flight HTTP reads instead of blocking reset.
        with self._lock:
            for user_id, chat_id, topic_id, signal in self._cancel_signals.values():
                if user_id == uid and (cid is None or cid == chat_id) and (thread_id is None or thread_id == topic_id):
                    signal.set()

    def cancel(self, uid, cid=None, thread_id=None):
        self._signal_cancel(uid, cid, thread_id)
        with self.user_lock(uid):
            return self.db.cancel_jobs(uid, cid, thread_id)

    def reset(self, uid, cid=None, thread_id=None):
        self._signal_cancel(uid, cid, thread_id)
        with self.user_lock(uid):
            self.db.reset_context(uid, cid, thread_id)

    def user_lock(self, uid):
        with self._lock:
            return self._user_locks.setdefault(uid, threading.RLock())

    def retry(self, uid, cid, thread_id=0):
        with self._lock:
            if uid in self._busy:
                return False
            row = self.db.query_one("SELECT * FROM jobs WHERE user_id=? AND chat_id=? AND thread_id=? ORDER BY id DESC LIMIT 1", (uid,cid,thread_id))
            if not row or row["status"] not in ("failed", "interrupted"):
                return False
            item = dict(row)
            item.pop("id")
            item.pop("update_id", None)
        return self._enqueue(uid, item)

    def _handle_command(self, client, s, user, cid, ctype, text, tg_msg_id, thread_id=0):
        parts = text.split(maxsplit=1)
        cmd, arg, uid = parts[0].split("@")[0].lower(), parts[1].strip() if len(parts) > 1 else "", user["id"]
        self.db.add_message(uid, "in", text, chat_id=cid, chat_type=ctype, thread_id=thread_id, kind="command", in_context=False, telegram_message_id=tg_msg_id)
        if cmd == "/start":
            reply = s["tg_welcome"] + ("\n\nYou are an administrator." if user["role"] == "admin" else "")
        elif cmd == "/help":
            reply = s["tg_help"] + COMMAND_HELP
        elif cmd == "/id":
            reply = f"Your Telegram user ID: {uid}"
        elif cmd == "/reset":
            self.reset(uid, cid, thread_id)
            reply = "Fresh conversation started here. History retained; pending replies cancelled."
        elif cmd == "/cancel":
            reply = f"Cancelled {self.cancel(uid,cid,thread_id)} pending reply(s). Any remaining model output will be discarded."
        elif cmd == "/retry":
            reply = "Retry queued." if self.retry(uid,cid,thread_id) else "No failed reply to retry here, or a reply is still running."
        elif cmd == "/prompt":
            if not arg:
                reply = "Instructions here:\n" + self.context(uid,cid,s,thread_id)["system"]
            elif len(arg) > 4000:
                reply = "Use at most 4000 characters for instructions."
            else:
                self._signal_cancel(uid,cid,thread_id)
                with self.user_lock(uid):
                    self.db.reset_context(uid,cid,thread_id)
                    self.db.set_chat_prompt(uid,cid,"" if arg.lower() == "reset" else arg,thread_id=thread_id)
                reply = "Instructions updated here; fresh context started."
        elif cmd == "/context":
            ctx = self.context(uid,cid,s,thread_id)
            reply = f"Context here: {len(ctx['included_ids'])} messages, ~{ctx['est_tokens']} / {s['ctx_max_tokens']} estimated tokens."
        elif cmd == "/status":
            row = self.db.query_one("SELECT COUNT(*) AS n FROM jobs WHERE user_id=? AND chat_id=? AND thread_id=? AND status IN ('queued','processing','delivering')", (uid,cid,thread_id))
            reply = f"Pending replies here: {row['n']}\nModel: {s['llm_model'] or 'automatic (currently loaded model)'}"
        elif cmd == "/privacy":
            hostname = urllib.parse.urlsplit(s["llm_base_url"]).hostname or ""
            destination = "this computer" if hostname.lower() in ("localhost","127.0.0.1","::1") else "a remote server selected by the owner"
            reply = (f"Model processing: {destination}.\n\n"
                     "The bot stores conversation history locally. Its owner can review and export that history. "
                     "Your model context stays separate from other users, chats and forum topics.\n\n"
                     "/reset starts fresh model context while retaining history. /cancel discards pending replies. "
                     "Ask the owner to delete stored history.")
        elif cmd == "/stats" and user["role"] == "admin":
            c = self.db.counts()
            reply = f"Users: {c['users']} · Messages: {c['messages']} ({c['messages_today']} today)"
        else:
            reply = "Unknown command. Try /help."
        self._reply_stored(client,uid,cid,ctype,reply,thread_id,tg_msg_id)

    def _reply_stored(self, client, uid, cid, ctype, text, thread_id=0, reply_to_message_id=None):
        if text:
            mid = self._safe_send(client,cid,text,thread_id,reply_to_message_id)
            self.db.add_message(uid,"out",text,chat_id=cid,chat_type=ctype,thread_id=thread_id,in_context=False,telegram_message_id=mid)

    def _enqueue(self, uid, item):
        notice, accepted = "", False
        with self._lock:
            if self._stop.is_set():
                return False
            if uid in self._pending:
                self._pending[uid] = [x for x in self._pending[uid] if self.db.job_status(x["id"]) == "queued"]
            if uid in self._busy and len(self._pending.get(uid,[])) >= self.get_settings()["bot_queue_limit"]:
                if "id" not in item:
                    notice = "Your queue is full. Wait or use /cancel."
            elif uid not in self._busy and len(self._busy) >= 64:
                if "id" not in item:
                    notice = "The bot is busy. Please try again shortly."
            else:
                job = item if "id" in item else self.db.create_job(item)
                if job is None or self.db.job_status(job["id"]) != "queued":
                    return False
                accepted = True
                if uid in self._busy:
                    self._pending.setdefault(uid,[]).append(job)
                    if self.get_settings()["bot_queue_notice"] and len(self._pending[uid]) == 1:
                        notice = self.get_settings()["bot_queue_notice_text"]
                else:
                    self._busy.add(uid)
                    worker = threading.Thread(target=self._drain,args=(uid,job),daemon=True,name=f"tg-user-{uid}")
                    self._workers.add(worker)
                    self._worker_users[uid] = worker
                    worker.start()
        # Never hold the global worker lock during network I/O or 429 waits.
        if notice:
            try:
                self._safe_send(self.make_client(),item["chat_id"],notice,item.get("thread_id",0),item.get("tg_msg_id"))
            except TelegramError:
                pass
        return accepted

    def _drain(self, uid, item):
        current = threading.current_thread()
        try:
            while item:
                try:
                    self._generate(item)
                except Exception as e:
                    if self.db.job_status(item["id"]) != "cancelled":
                        self.db.set_job_status(item["id"],"failed")
                    self.db.log("error","worker",f"Worker failed for {uid}: {e}")
                with self._lock:
                    queue = self._pending.get(uid,[])
                    if queue:
                        item = queue.pop(0)
                    else:
                        self._pending.pop(uid,None)
                        self._busy.discard(uid)
                        self._worker_users.pop(uid,None)
                        item = None
        finally:
            with self._lock:
                self._workers.discard(current)
                if self._worker_users.get(uid) is current:
                    self._pending.pop(uid,None)
                    self._busy.discard(uid)
                    self._worker_users.pop(uid,None)

    def _wait_llm_slot(self, s, jid=None):
        while not self._stop.is_set():
            with self._lock:
                if jid and self.db.job_status(jid) != "queued":
                    return False
                if self._llm_active < self.get_settings()["bot_max_parallel_llm"]:
                    self._llm_active += 1
                    return True
            self._stop.wait(.1)
        return False

    def _generate(self, item):
        if "id" not in item:
            item = self.db.create_job(item)
        if item is None:
            return
        jid, uid, cid = item["id"],item["user_id"],item["chat_id"]
        thread_id = int(item.get("thread_id") or 0)
        s,client = self.get_settings(),self.make_client()
        acquired,typing_stop = False,threading.Event()
        cancelled = threading.Event()
        with self._lock:
            self._cancel_signals[jid] = (uid,cid,thread_id,cancelled)
        try:
            acquired = self._wait_llm_slot(s,jid)
            if not acquired:
                return
            # A job may wait behind a long generation. Use the owner's current
            # endpoint, token and access settings when this job actually starts.
            s,client = self.get_settings(),self.make_client()
            with self.user_lock(uid):
                user = self.db.get_user(uid)
                if self._stop.is_set() or cancelled.is_set() or not s["tg_enabled"] or self.db.job_status(jid) != "queued" or not user or user["blocked"] or not self._allowed(user,s):
                    self.db.set_job_status(jid,"cancelled")
                    return
                self.db.set_job_status(jid,"processing")
            if s["tg_typing"]:
                threading.Thread(target=self._typing_loop,args=(client,cid,typing_stop,thread_id),daemon=True).start()
            validate_endpoint(s["llm_base_url"],s.get("llm_allow_remote",False))
            ctx = self.context(uid,cid,s,thread_id,before_id=item.get("inbound_id"))
            model = s["llm_model"] or self._default_model(s)
            # Keep the model slot and RAM watchdog until bounded completion.
            # Some servers continue inference after an HTTP disconnect; early
            # slot release could overlap requests on this memory-limited PC.
            reply = llm_mod.chat_completion(s["llm_base_url"],model,s["llm_api_key"],ctx["system"],ctx["messages"],s["gen_temperature"],s["gen_max_tokens"],s["gen_top_p"],s["llm_timeout"],self._shutdown_event)
            reply = (reply or "").strip()
            if not reply:
                raise RuntimeError("Model returned an empty reply.")
            with self.user_lock(uid):
                user = self.db.get_user(uid)
                if self._stop.is_set() or cancelled.is_set() or not self.get_settings()["tg_enabled"] or self.db.job_status(jid) != "processing" or not user or user["blocked"] or not self._allowed(user,self.get_settings()):
                    self.db.set_job_status(jid,"cancelled")
                    return
                self.db.set_job_status(jid,"delivering")
                mid = self._safe_send(client,cid,reply,thread_id,item.get("tg_msg_id"),cancelled)
                if cancelled.is_set():
                    self.db.set_job_status(jid,"cancelled")
                    return
                self.db.add_message(uid,"out",reply,chat_id=cid,chat_type=item["chat_type"],thread_id=thread_id,telegram_message_id=mid,job_id=jid)
                self.db.set_job_status(jid,"done")
        except Exception as e:
            with self.user_lock(uid):
                if self._stop.is_set() or cancelled.is_set() or self.db.job_status(jid) == "cancelled":
                    self.db.set_job_status(jid,"cancelled")
                    return
                self.db.set_job_status(jid,"failed")
                self.db.log("error","reply",f"Reply to {uid} failed: {e}"[:500])
                self.db.add_message(uid,"out","[Reply failed; see Activity]",chat_id=cid,chat_type=item["chat_type"],thread_id=thread_id,kind="error",in_context=False,status="failed")
            try:
                self._safe_send(client,cid,s["tg_error"] + "\nUse /retry to try this message again.",thread_id,item.get("tg_msg_id"),cancelled)
            except TelegramError:
                pass
        finally:
            typing_stop.set()
            if acquired:
                with self._lock:
                    self._llm_active -= 1
            with self._lock:
                self._cancel_signals.pop(jid,None)

    def _typing_loop(self, client, cid, stop, thread_id=0):
        while not stop.is_set() and not self._stop.is_set():
            if getattr(client,"supports_reply_options",False):
                client.send_chat_action(cid,"typing",message_thread_id=thread_id)
            else:
                client.send_chat_action(cid,"typing")
            stop.wait(4)

    def _default_model(self, s):
        key = (s["llm_base_url"],s["llm_api_key"])
        now,cached_key,models = self._models_cache
        if cached_key != key or time.monotonic()-now > 10:
            models = llm_mod.list_models(s["llm_base_url"],s["llm_api_key"],timeout=5)
            self._models_cache = (time.monotonic(),key,models)
        if not models:
            raise RuntimeError("No loaded models reported by endpoint.")
        return models[0]

    def _safe_send(self, client, cid, text, thread_id=0, reply_to_message_id=None, cancel_event=None):
        first = 0
        delivery_deadline = time.monotonic() + 120
        for part in llm_mod.chunk_reply(text):
            if cancel_event is not None and cancel_event.is_set():
                raise TelegramError("Telegram delivery cancelled.")
            if not part:
                continue
            if getattr(client,"supports_reply_options",False):
                s = self.get_settings()
                options = dict(message_thread_id=thread_id,cancel_event=cancel_event,
                               delivery_deadline=delivery_deadline,
                               reply_to_message_id=(first or reply_to_message_id) if s.get("tg_reply_to_message",True) else None)
                formatted = s.get("tg_reply_format","html") == "html"
                try:
                    mid = client.send_text(cid,llm_mod.telegram_html(part) if formatted else part,
                                           parse_mode="HTML" if formatted else None,**options)
                except TelegramError as error:
                    # Retry plain text only after a definite formatting rejection;
                    # blocked chats and uncertain network failures never resend.
                    if formatted and error.code == 400 and any(reason in str(error).lower() for reason in ("parse entities","unsupported start tag","can't find end tag")):
                        mid = client.send_text(cid,part,**options)
                    else:
                        raise
            else:
                mid = client.send_text(cid,part)
            first = first or mid
        return first
