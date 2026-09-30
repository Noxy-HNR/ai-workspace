"""Telegram bot dashboard server.

Serves the web UI (static/) and a JSON API, owns the Telegram poller.
Loopback-only by default. Stdlib only: no pip dependencies to break.

Run:  py app.py [--port 9095] [--data ./data]
"""
from __future__ import annotations

import json
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import secrets

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bot import db as db_mod
from bot import llm as llm_mod
from bot import settings as settings_mod
from bot.poller import Poller
from bot.telegram_api import TelegramClient, TelegramError

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
STARTED_AT = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

DB: db_mod.Database | None = None
POLLER: Poller | None = None
PROBE_CACHE = {}
PREVIEW = False
PROBE_LOCK = threading.Lock()
CSRF_TOKEN = secrets.token_urlsafe(32)


def integer(value,name="value",minimum=-(2**63),maximum=2**63-1):
    if isinstance(value,bool) or not isinstance(value,(str,int)):
        raise ValueError(f"{name} must be an integer")
    try:
        result = int(value)
    except ValueError:
        raise ValueError(f"{name} must be an integer") from None
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return result


def lock_data_directory(data_dir):
    """Hold a process-owned lock so another port cannot poll the same bot DB."""
    data_dir.mkdir(parents=True, exist_ok=True)
    handle = (data_dir / "instance.lock").open("a+b")
    try:
        handle.seek(0)
        if handle.read(1) == b"":
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if sys.platform == "win32":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError("Another Relay Station instance is using this data directory") from None
    return handle


def get_raw_settings() -> dict:
    raw = DB.all_settings()
    for key, (default, *_r) in settings_mod.DEFS.items():
        raw.setdefault(key, default)
    return raw


def get_settings() -> dict:
    return settings_mod.typed_settings(get_raw_settings())


def make_client() -> TelegramClient:
    raw = get_raw_settings()
    return TelegramClient(raw.get("tg_bot_token", ""))


# ---------------------------------------------------------------- API helpers

def check_llm(s: dict) -> dict:
    out = {"connected": False, "models": [], "error": "",
           "base_url": s.get("llm_base_url", ""),
           "model": s.get("llm_model") or "(auto)"}
    try:
        settings_mod.validate_endpoint(s["llm_base_url"], s.get("llm_allow_remote", False))
        models = llm_mod.list_models(s.get("llm_base_url"),
                                     s.get("llm_api_key", ""), timeout=8)
        out["models"] = models
        out["connected"] = bool(models) and (not s.get("llm_model") or s["llm_model"] in models)
        if not out["connected"]:
            out["error"] = "No loaded model, or the configured model is unavailable."
    except Exception as e:
        out["error"] = str(e)[:300]
    return out


def check_telegram() -> dict:
    client = make_client()
    out = {"configured": client.configured, "connected": False,
           "bot_username": DB.get_meta("tg_username", ""), "error": ""}
    if not client.configured:
        out["error"] = "Bot token is not configured."
        return out
    try:
        me = client.get_me()
        out["connected"] = True
        out["bot_username"] = (me or {}).get("username", "")
        out["bot_name"] = (me or {}).get("first_name", "")
        if out["bot_username"]:
            DB.set_meta("tg_username", out["bot_username"])
    except TelegramError as e:
        out["error"] = str(e)[:300]
    return out


class Handler(BaseHTTPRequestHandler):
    server_version = "TgDashboard/2.0"
    timeout = 10

    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def log_message(self, *args):
        pass

    # -- plumbing -------------------------------------------------------
    def _read_json(self):
        try:
            length = int(self.headers.get("Content-Length", 0) or 0)
        except ValueError:
            raise ValueError("Invalid Content-Length") from None
        if length < 0 or length > 131072:
            raise ValueError("Request body must be at most 128 KiB")
        if not length:
            return {}
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeError):
            raise ValueError("Invalid JSON body") from None
        if not isinstance(body, dict):
            raise ValueError("JSON body must be an object")
        return body

    def _trusted_request(self, mutate=False):
        allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        host = self.headers.get("Host", "")
        origin = self.headers.get("Origin")
        if host not in allowed or (origin and origin not in {"http://" + h for h in allowed}):
            self._reject_request("Only same-origin loopback requests are allowed")
            return False
        if mutate and not secrets.compare_digest(self.headers.get("X-Relay-CSRF", ""), CSRF_TOKEN):
            self._reject_request("Refresh the dashboard and try again")
            return False
        return True

    def _reject_request(self, message):
        # Drain a bounded body before closing, avoiding a TCP reset that can
        # hide the useful 403 response on Windows.
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if 0 < length <= 131072:
                self.rfile.read(length)
        except (ValueError, OSError):
            pass
        self._send({"error": message}, 403)

    def _headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        self.send_header("Referrer-Policy", "no-referrer")

    def _send(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self._headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _serve_static(self, path: str):
        rel = path[1:] if path.startswith("/") else path
        if rel in ("", "/"):
            rel = "index.html"
        target = (STATIC / rel.lstrip("/")).resolve()
        if not target.is_relative_to(STATIC.resolve()) or not target.is_file():
            return self._send({"error": "File not found"}, 404)
        ctype = "text/html; charset=utf-8"
        if target.suffix == ".css":
            ctype = "text/css; charset=utf-8"
        elif target.suffix == ".js":
            ctype = "application/javascript; charset=utf-8"
        data = target.read_bytes()
        self.send_response(200)
        self._headers()
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    # -- routing -----------------------------------------------------------
    def do_GET(self):
        if not self._trusted_request():
            return
        parsed = urllib.parse.urlparse(self.path)
        path, qs = parsed.path, urllib.parse.parse_qs(parsed.query)
        try:
            if path == "/api/session":
                self._send({"csrf": CSRF_TOKEN})
            elif path == "/api/health":
                self._send({"service": "relay-station", "version": "2.0"})
            elif path == "/api/status":
                self._api_status()
            elif path == "/api/users":
                self._api_users(qs)
            elif path.startswith("/api/users/"):
                self._api_user_get(path, qs)
            elif path == "/api/settings":
                self._send({"settings": settings_mod.public_settings(
                    get_raw_settings())})
            elif path == "/api/events":
                self._send({"events": DB.events(qs.get("level", [""])[0],
                                                integer(qs.get("limit", ["300"])[0],"limit"),
                                                qs.get("search",[""])[0][:500],qs.get("source",[""])[0]),
                            "sources":[r["source"] for r in DB.query("SELECT DISTINCT source FROM events ORDER BY source")]})
            elif path == "/api/jobs":
                status = qs.get("status",[""])[0]
                if status not in ("","queued","processing","delivering","done","failed","cancelled","interrupted"):
                    raise ValueError("Unknown job status")
                self._send(DB.list_jobs(status,integer(qs["user_id"][0],"user_id",1) if "user_id" in qs else None,
                                        integer(qs.get("limit",["100"])[0],"limit")))
            elif path == "/api/presets":
                self._send({"presets":[dict(r) for r in DB.query("SELECT * FROM prompt_presets ORDER BY name COLLATE NOCASE,id")]})
            elif path == "/api/stats":
                self._api_stats()
            elif path == "/api/poller":
                self._send({"poller": POLLER.snapshot()})
            elif path.startswith("/api/"):
                self._send({"error": "unknown route"}, 404)
            else:
                self._serve_static(path)
        except ValueError as e:
            self._send({"error": str(e)[:300]}, status=400)
        except Exception:
            self._send({"error": "Server error; check Activity."}, status=500)

    def do_POST(self):
        if not self._trusted_request(mutate=True):
            return
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        try:
            body = self._read_json()
            if path == "/api/settings":
                self._api_settings_save(body)
            elif path == "/api/test/telegram":
                self._send({"result": check_telegram()})
            elif path == "/api/test/llm":
                self._send({"result": check_llm(get_settings())})
            elif path == "/api/poller/start":
                POLLER.start()
                self._send({"ok": True, "poller": POLLER.snapshot()})
            elif path == "/api/poller/stop":
                POLLER.stop()
                self._send({"ok": True, "poller": POLLER.snapshot()})
            elif path == "/api/shutdown":
                self._send({"ok": True, "shutting_down": True})
                threading.Thread(target=self.server.shutdown,daemon=True,
                                 name="relay-shutdown").start()
            elif path == "/api/presets":
                name,prompt = body.get("name"),body.get("prompt")
                if not isinstance(name,str) or not name.strip() or len(name)>100:
                    raise ValueError("Preset name must contain 1 to 100 characters")
                if not isinstance(prompt,str) or not prompt.strip() or len(prompt)>4000:
                    raise ValueError("Preset instructions must contain 1 to 4000 characters")
                preset = DB.save_preset(name.strip(),prompt.strip(),integer(body["id"],"preset id",1) if "id" in body else None)
                if preset is None:
                    return self._send({"error":"Preset not found"},404)
                self._send({"ok":True,"preset":preset})
            elif path.startswith("/api/users/"):
                self._api_user_post(path, body)
            else:
                self._send({"error": "unknown route"}, status=404)
        except ValueError as e:
            self._send({"error": str(e)[:300]}, status=400)
        except Exception:
            self._send({"error": "Server error; check Activity."}, status=500)

    # -- endpoints ----------------------------------------------------------
    def _api_status(self):
        s = get_settings()
        with PROBE_LOCK:
            if time.monotonic() - PROBE_CACHE.get("at", -100) > 20:
                PROBE_CACHE.update(at=time.monotonic(), telegram=check_telegram(), llm=check_llm(s))
            tg, llm_state = PROBE_CACHE["telegram"], PROBE_CACHE["llm"]
        admin_row = DB.query_one(
            "SELECT COUNT(*) AS n FROM users WHERE role='admin'")
        counts = DB.counts()
        self._send({
            "preview": PREVIEW,
            "telegram": tg, "llm": llm_state,
            "poller": POLLER.snapshot(),
            "jobs": DB.list_jobs(limit=1)["counts"],
            "admin_claimed": bool(admin_row and admin_row["n"]),
            "counts": counts, "uptime_since": STARTED_AT,
            "now": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })

    def _api_users(self, qs):
        users = DB.list_users(qs.get("search", [""])[0],
                              qs.get("role", [""])[0],
                              qs.get("blocked", [""])[0])
        rows = DB.query("SELECT user_id,COUNT(*) AS n,SUM(direction='in') AS nin,SUM(direction='out') AS nout,SUM(LENGTH(text)) AS chars,MAX(created_at) AS last FROM messages GROUP BY user_id")
        stats = {r["user_id"]:{"messages":r["n"],"inbound":r["nin"],"outbound":r["nout"],"chars":r["chars"],"last_message":r["last"]} for r in rows}
        out = []
        for u in users:
            st = stats.get(u["id"],{"messages":0,"inbound":0,"outbound":0,"chars":0,"last_message":None})
            out.append({**u, "stats": st,
                        "context_limit_override": u["context_limit_override"]})
        self._send({"users": out})

    def _api_user_get(self, path, qs):
        parts = path.split("/")
        # /api/users/<id>[/messages|context]
        try:
            uid = integer(parts[3],"user id",1)
        except (IndexError, ValueError):
            return self._send({"error": "bad user id"}, status=400)
        user = DB.get_user(uid)
        if not user:
            return self._send({"error": "user not found"}, status=404)
        sub = parts[4] if len(parts) > 4 else ""
        if len(parts)>5:
            return self._send({"error":"Unknown user route"},404)
        chat_id = integer(qs.get("chat_id", [user.get("chat_id") or uid])[0],"chat_id")
        thread_id = integer(qs.get("thread_id",["0"])[0],"thread_id",0)
        if sub == "messages":
            limit = integer(qs.get("limit", ["100"])[0],"limit")
            before = integer(qs["before"][0],"before",1) if "before" in qs else None
            self._send({"user": user,
                        **DB.message_page(uid,chat_id,thread_id,limit,before,qs.get("search",[""])[0][:500]),
                        "chats": DB.list_chats(uid)})
        elif sub == "chats":
            self._send({"chats":DB.list_chats(uid)})
        elif sub == "context":
            s = get_settings()
            limit = (user.get("context_limit_override")
                     or s.get("ctx_history_limit", 30))
            ctx = POLLER.context(uid, chat_id, s,thread_id=thread_id)
            self._send({"user": user, "context": ctx,
                        "limits": {"history_limit": int(limit),
                                   "max_tokens": int(s.get("ctx_max_tokens", 6000))}})
        elif sub == "export":
            total = DB.query_one("SELECT COUNT(*) AS n FROM messages WHERE user_id=? AND chat_id=? AND thread_id=?",(uid,chat_id,thread_id))["n"]
            messages = DB.history(uid,chat_id=chat_id,thread_id=thread_id)
            self._send({"user": user,"chat_id":chat_id,"thread_id":thread_id,"messages":messages,
                        "total":total,"truncated":total>len(messages),"exported":len(messages),"exported_at":db_mod.utcnow()})
        elif not sub:
            self._send({"user": user, "stats": DB.user_stats(uid)})
        else:
            self._send({"error":"Unknown user route"},404)

    def _api_user_post(self, path, body):
        parts = path.split("/")
        try:
            uid = integer(parts[3],"user id",1)
        except (IndexError, ValueError):
            return self._send({"error": "bad user id"}, status=400)
        action = parts[4] if len(parts) > 4 else ""
        if len(parts)!=5:
            return self._send({"error":"Unknown user action"},404)
        user = DB.get_user(uid)
        if not user and action != "delete":
            return self._send({"error": "user not found"}, status=404)

        if action == "block":
            POLLER._signal_cancel(uid)
            with POLLER.user_lock(uid):
                DB.cancel_jobs(uid)
                DB.set_user_fields(uid, blocked=1)
            DB.log("info", "admin", f"User {uid} blocked.")
        elif action == "unblock":
            DB.set_user_fields(uid, blocked=0)
            DB.log("info", "admin", f"User {uid} unblocked.")
        elif action == "grant-admin":
            DB.set_user_fields(uid, role="admin")
            DB.log("info", "admin", f"User {uid} granted admin.")
        elif action == "revoke-admin":
            with DB._lock:
                admins = DB.query("SELECT id FROM users WHERE role='admin'")
                user = DB.get_user(uid)
                if len(admins) <= 1 and user and user.get("role") == "admin":
                    return self._send({"error": "Cannot revoke the last admin."},status=400)
                DB.set_user_fields(uid, role="user")
            DB.log("info", "admin", f"User {uid} admin revoked.")
        elif action == "reset-context":
            POLLER.reset(uid,integer(body["chat_id"],"chat_id") if "chat_id" in body else None,
                         thread_id=integer(body.get("thread_id",0),"thread_id",0) if "chat_id" in body else None)
            DB.log("info", "admin", f"User {uid} context reset.")
        elif action == "cancel":
            POLLER.cancel(uid,integer(body["chat_id"],"chat_id") if "chat_id" in body else None,
                          thread_id=integer(body.get("thread_id",0),"thread_id",0) if "chat_id" in body else None)
        elif action == "retry":
            if not POLLER.retry(uid,integer(body.get("chat_id") or user.get("chat_id") or uid,"chat_id"),
                                thread_id=integer(body.get("thread_id",0),"thread_id",0)):
                return self._send({"error": "No failed reply to retry, or user is busy"}, 409)
        elif action in ("chat-meta","chat-prompt"):
            chat_id = integer(body.get("chat_id"),"chat_id")
            thread_id = integer(body.get("thread_id",0),"thread_id",0)
            if action == "chat-meta":
                fields = {}
                if "label" in body:
                    if not isinstance(body["label"],str) or len(body["label"])>100:
                        raise ValueError("Conversation label must be at most 100 characters")
                    fields["label"] = body["label"].strip()
                if "bookmarked" in body:
                    if not isinstance(body["bookmarked"],(bool,int)) or body["bookmarked"] not in (True,False,0,1):
                        raise ValueError("Bookmarked must be true or false")
                    fields["bookmarked"] = int(bool(body["bookmarked"]))
                with POLLER.user_lock(uid):
                    if not DB.get_user(uid) or not DB.chat_exists(uid,chat_id,thread_id):
                        return self._send({"error":"Conversation not found"},404)
                    DB.set_chat_metadata(uid,chat_id,thread_id,**fields)
            else:
                prompt = body.get("prompt")
                if not isinstance(prompt,str) or len(prompt)>4000:
                    raise ValueError("Chat instructions must be at most 4000 characters")
                POLLER._signal_cancel(uid,chat_id,thread_id)
                with POLLER.user_lock(uid):
                    if not DB.get_user(uid) or not DB.chat_exists(uid,chat_id,thread_id):
                        return self._send({"error":"Conversation not found"},404)
                    DB.reset_context(uid,chat_id,thread_id)
                    DB.set_chat_prompt(uid,chat_id,prompt,thread_id)
            DB.log("info","admin",f"User {uid} conversation preferences updated.")
        elif action in ("note","overrides","profile"):
            fields: dict = {}
            if action in ("note","profile") and "notes" in body:
                if not isinstance(body["notes"],str) or len(body["notes"])>2000:
                    raise ValueError("Notes must be at most 2000 characters")
                fields["notes"] = body["notes"]
            if "context_limit_override" in body:
                v = body["context_limit_override"]
                if isinstance(v,bool):
                    raise ValueError("Context limit must be an integer or blank")
                fields["context_limit_override"] = (
                    None if v in (None, "", 0, "0") else integer(v,"context limit",1,500))
            if "system_prompt_override" in body:
                if not isinstance(body["system_prompt_override"],str) or len(body["system_prompt_override"])>4000:
                    raise ValueError("Instructions must be at most 4000 characters")
                fields["system_prompt_override"] = body["system_prompt_override"]
            DB.set_user_fields(uid, **fields)
            DB.log("info", "admin", f"User {uid} profile updated.")
        elif action == "send":
            text = body.get("text", "")
            if not isinstance(text,str) or not text.strip() or len(text)>4000:
                return self._send({"error": "Message must contain 1 to 4000 characters"},status=400)
            chat_id = integer(body.get("chat_id") or user.get("chat_id") or uid,"chat_id")
            thread_id = integer(body.get("thread_id",0),"thread_id",0)
            with POLLER.user_lock(uid):
                fresh = DB.get_user(uid)
                if not fresh or fresh["blocked"]:
                    return self._send({"error":"User is blocked or no longer exists"},400)
                if not DB.chat_exists(uid,chat_id,thread_id) or chat_id==0:
                    return self._send({"error": "Choose an existing conversation; private users must start the bot first"},400)
                chat_type = DB.query_one("SELECT chat_type FROM messages WHERE user_id=? AND chat_id=? AND thread_id=? ORDER BY id DESC LIMIT 1",(uid,chat_id,thread_id))["chat_type"]
                mid = 0
                try:
                    client = make_client()
                    deadline = time.monotonic() + 25
                    for part in llm_mod.chunk_reply(text):
                        if getattr(client,"supports_reply_options",False):
                            mid = client.send_text(chat_id,part,max_retries=0,thread_id=thread_id,
                                                   delivery_deadline=deadline) or mid
                        else:
                            mid = (client.send_text(chat_id,part,thread_id=thread_id) if thread_id else client.send_text(chat_id,part)) or mid
                except TelegramError as e:
                    DB.log("warning", "admin",f"Admin message to {uid} failed: {e}")
                    return self._send({"error":f"Telegram send failed: {e}"},502)
                DB.add_message(uid,"out",text,chat_id=chat_id,thread_id=thread_id,chat_type=chat_type,
                               kind="text",in_context=True,telegram_message_id=mid)
            DB.log("info", "admin", f"Admin sent a message to user {uid}.")
        elif action == "delete":
            POLLER._signal_cancel(uid)
            with POLLER.user_lock(uid), DB._lock, DB._conn:
                user = DB.get_user(uid)
                if user and user["role"] == "admin" and DB.query_one("SELECT COUNT(*) AS n FROM users WHERE role='admin'")["n"] <= 1:
                    return self._send({"error": "Cannot delete the last administrator"}, 400)
                DB.cancel_jobs(uid)
                for table in ("jobs", "chat_prompts","chat_metadata", "messages"):
                    DB._conn.execute(f"DELETE FROM {table} WHERE user_id=?", (uid,))
                DB._conn.execute("DELETE FROM users WHERE id=?", (uid,))
            DB.log("warning", "admin", f"User {uid} and history deleted.")
            return self._send({"ok": True, "deleted": True})
        else:
            return self._send({"error": "unknown action"}, status=404)
        self._send({"ok": True, "user": DB.get_user(uid)})

    def _api_settings_save(self, body):
        raw = get_raw_settings()
        incoming = body.get("settings", body)
        values = settings_mod.validate_settings(incoming, raw)
        clear_secrets = body.get("clear_secrets",[])
        if not isinstance(clear_secrets,list) or any(not isinstance(key,str) or key not in settings_mod.SECRET_KEYS for key in clear_secrets):
            raise ValueError("clear_secrets must contain only bot token or API key setting names")
        for key in clear_secrets:
            values[key] = ""
        DB.save_settings(values)
        updated = list(values)
        with PROBE_LOCK:
            PROBE_CACHE.clear()
        DB.log("info", "admin", f"Settings updated: {', '.join(updated)}")
        self._send({"ok": True, "updated": updated,
                    "settings": settings_mod.public_settings(
                        get_raw_settings())})

    def _api_stats(self):
        top = DB.query(
            "SELECT user_id, COUNT(*) AS n FROM messages GROUP BY user_id"
            " ORDER BY n DESC LIMIT 10")
        self._send({"daily": DB.daily_stats(30),
                    "top_users": [dict(r) for r in top],
                    "counts": DB.counts(),
                    "jobs":DB.list_jobs(limit=1)["counts"],
                    "delivery": [dict(r) for r in DB.query("SELECT SUBSTR(updated_at,1,10) AS day,SUM(status='done') AS replies,SUM(status='failed') AS failures FROM jobs WHERE updated_at>=DATE('now','-30 days') GROUP BY day ORDER BY day")],
                    "conversation_count": DB.query_one("SELECT COUNT(*) AS n FROM (SELECT user_id,chat_id,thread_id FROM messages GROUP BY user_id,chat_id,thread_id)")["n"]})


class DashboardServer(ThreadingHTTPServer):
    """Track daemon handlers so teardown never closes their database early."""
    daemon_threads = True

    def __init__(self,*args,**kwargs):
        self._request_condition = threading.Condition()
        self._active_requests = 0
        super().__init__(*args,**kwargs)

    def process_request(self,request,client_address):
        # Count before spawning; otherwise shutdown can miss a thread that
        # was accepted but has not begun running its handler yet.
        with self._request_condition:
            self._active_requests += 1
        try:
            super().process_request(request,client_address)
        except BaseException:
            with self._request_condition:
                self._active_requests -= 1
                self._request_condition.notify_all()
            raise

    def process_request_thread(self,request,client_address):
        try:
            super().process_request_thread(request,client_address)
        finally:
            with self._request_condition:
                self._active_requests -= 1
                self._request_condition.notify_all()

    def wait_for_requests(self,timeout=3):
        deadline = time.monotonic() + max(0,timeout)
        with self._request_condition:
            while self._active_requests:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._request_condition.wait(remaining)
            return True


def run(port: int, data_dir: Path):
    global DB, POLLER
    server = DashboardServer(("127.0.0.1", port), Handler)
    instance_lock = None
    database = None
    poller = None
    try:
        instance_lock = lock_data_directory(data_dir)
        database = DB = db_mod.Database(data_dir / "bot.db")
        # Seed defaults for any missing keys.
        for key, (default, *_r) in settings_mod.DEFS.items():
            if DB.get_setting(key, None) is None:
                DB.set_setting(key, default)
        DB.log("info", "server", f"Dashboard starting on 127.0.0.1:{port}.")
        poller = POLLER = Poller(DB, get_settings, make_client)
        POLLER.start()
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        quiescent = poller is None
        try:
            if poller is not None:
                quiescent = poller.shutdown(timeout=3)
        finally:
            try:
                server.server_close()
            finally:
                try:
                    quiescent = server.wait_for_requests(timeout=3) and quiescent
                    if database is not None:
                        database.log("info", "server", "Dashboard stopped.")
                        if quiescent:
                            database.close()
                finally:
                    if instance_lock is not None:
                        instance_lock.close()
                    if quiescent:
                        DB = POLLER = None


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9095)
    ap.add_argument("--data", type=str, default=str(ROOT / "data"))
    args = ap.parse_args()
    run(args.port, Path(args.data))
