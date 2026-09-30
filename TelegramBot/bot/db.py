"""SQLite persistence layer for the Telegram bot dashboard.

All conversation reads are scoped by user_id at the SQL level so one user's
messages can never leak into another user's context. There is no code path
that loads messages without a user filter except explicit admin exports.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    first_name TEXT DEFAULT '',
    last_name TEXT DEFAULT '',
    username TEXT DEFAULT '',
    role TEXT DEFAULT 'user',
    blocked INTEGER DEFAULT 0,
    notes TEXT DEFAULT '',
    context_limit_override INTEGER,
    system_prompt_override TEXT DEFAULT '',
    first_seen TEXT DEFAULT '',
    last_seen TEXT DEFAULT '',
    chat_id INTEGER DEFAULT 0,
    is_bot INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    chat_id INTEGER DEFAULT 0,
    thread_id INTEGER NOT NULL DEFAULT 0,
    chat_type TEXT DEFAULT 'private',
    job_id INTEGER,
    direction TEXT NOT NULL,
    kind TEXT DEFAULT 'text',
    text TEXT DEFAULT '',
    in_context INTEGER DEFAULT 1,
    status TEXT DEFAULT 'done',
    telegram_message_id INTEGER,
    created_at TEXT DEFAULT '',
    FOREIGN KEY(user_id) REFERENCES users(id)
);
CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(user_id, id);
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT DEFAULT '',
    updated_at TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    level TEXT DEFAULT 'info',
    source TEXT DEFAULT '',
    message TEXT DEFAULT '',
    created_at TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    update_id INTEGER UNIQUE,
    user_id INTEGER NOT NULL,
    chat_id INTEGER NOT NULL,
    thread_id INTEGER NOT NULL DEFAULT 0,
    chat_type TEXT NOT NULL,
    text TEXT NOT NULL,
    tg_msg_id INTEGER,
    inbound_id INTEGER,
    status TEXT NOT NULL DEFAULT 'queued',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, id);
CREATE TABLE IF NOT EXISTS chat_prompts (
    user_id INTEGER NOT NULL,
    chat_id INTEGER NOT NULL,
    thread_id INTEGER NOT NULL DEFAULT 0,
    prompt TEXT NOT NULL DEFAULT '',
    PRIMARY KEY(user_id, chat_id, thread_id)
);
CREATE TABLE IF NOT EXISTS chat_metadata (
    user_id INTEGER NOT NULL, chat_id INTEGER NOT NULL,
    thread_id INTEGER NOT NULL DEFAULT 0,
    label TEXT NOT NULL DEFAULT '', bookmarked INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(user_id,chat_id,thread_id)
);
CREATE TABLE IF NOT EXISTS prompt_presets (
    id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, prompt TEXT NOT NULL,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
"""


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._log_count = 0
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        try:
            with self._lock:
                version = self._conn.execute("PRAGMA user_version").fetchone()[0]
                if version>2:
                    raise RuntimeError("This database was created by a newer Relay Station version")
                self._conn.executescript(SCHEMA)
                self._conn.commit()
                self._migrate()
        except Exception:
            self._conn.close()
            raise

    def _migrate(self):
        """Transactional additive upgrade; existing history remains in topic 0."""
        with self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            for table in ("messages", "jobs"):
                columns = {r["name"] for r in self._conn.execute(f"PRAGMA table_info({table})")}
                if "thread_id" not in columns:
                    self._conn.execute(f"ALTER TABLE {table} ADD COLUMN thread_id INTEGER NOT NULL DEFAULT 0")
                if table=="messages" and "job_id" not in columns:
                    self._conn.execute("ALTER TABLE messages ADD COLUMN job_id INTEGER")
            columns = {r["name"] for r in self._conn.execute("PRAGMA table_info(chat_prompts)")}
            if "thread_id" not in columns:
                self._conn.execute("ALTER TABLE chat_prompts RENAME TO chat_prompts_v1")
                self._conn.execute("CREATE TABLE chat_prompts (user_id INTEGER NOT NULL,chat_id INTEGER NOT NULL,thread_id INTEGER NOT NULL DEFAULT 0,prompt TEXT NOT NULL DEFAULT '',PRIMARY KEY(user_id,chat_id,thread_id))")
                self._conn.execute("INSERT INTO chat_prompts SELECT user_id,chat_id,0,prompt FROM chat_prompts_v1")
                self._conn.execute("DROP TABLE chat_prompts_v1")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_scope ON messages(user_id,chat_id,thread_id,id)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_user_scope ON jobs(user_id,chat_id,thread_id,id)")
            self._conn.execute("PRAGMA user_version=2")

    # -- low level -----------------------------------------------------
    def execute(self, sql: str, params: tuple = ()):
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    def query(self, sql: str, params: tuple = ()):
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def query_one(self, sql: str, params: tuple = ()):
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    def close(self):
        with self._lock:
            self._conn.close()

    def save_settings(self, values: dict):
        with self._lock, self._conn:
            self._conn.executemany(
                "INSERT INTO settings VALUES (?,?,?) ON CONFLICT(key)"
                " DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                [(k, v, utcnow()) for k, v in values.items()])

    def create_job(self, item: dict):
        """Persist inbound text and queue entry atomically, before acknowledgement."""
        with self._lock, self._conn:
            if item.get("update_id") is not None:
                existing = self.query_one("SELECT * FROM jobs WHERE update_id=?",
                                          (item["update_id"],))
                if existing:
                    return None
            mid = self._conn.execute(
                "INSERT INTO messages (user_id,chat_id,thread_id,chat_type,direction,text,"
                "in_context,status,telegram_message_id,created_at)"
                " VALUES (?,?,?,?,'in',?,0,'queued',?,?)",
                (item["user_id"], item["chat_id"], int(item.get("thread_id",0)), item["chat_type"],
                 item["text"], item.get("tg_msg_id"), utcnow())).lastrowid
            jid = self._conn.execute(
                "INSERT INTO jobs (update_id,user_id,chat_id,thread_id,chat_type,text,"
                "tg_msg_id,inbound_id,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (item.get("update_id"), item["user_id"], item["chat_id"], int(item.get("thread_id",0)),
                 item["chat_type"], item["text"], item.get("tg_msg_id"),
                 mid, utcnow(), utcnow())).lastrowid
            return dict(self.query_one("SELECT * FROM jobs WHERE id=?", (jid,)))

    def job_status(self, job_id):
        row = self.query_one("SELECT status FROM jobs WHERE id=?", (job_id,))
        return row["status"] if row else "cancelled"

    def set_job_status(self, job_id, status):
        with self._lock, self._conn:
            self._conn.execute("UPDATE jobs SET status=?,updated_at=? WHERE id=?",
                               (status, utcnow(), job_id))
            self._conn.execute(
                "UPDATE messages SET status=?,in_context=? WHERE id="
                " (SELECT inbound_id FROM jobs WHERE id=?)",
                ("done" if status in ("processing", "done") else status,
                 int(status in ("processing", "done")), job_id))

    def cancel_jobs(self, user_id, chat_id=None,thread_id=None):
        where = "user_id=? AND status IN ('queued','processing','delivering')"
        args = (user_id,)
        if chat_id is not None:
            where += " AND chat_id=?"
            args += (chat_id,)
        if thread_id is not None:
            where += " AND thread_id=?"
            args += (int(thread_id),)
        with self._lock:
            rows = self.query("SELECT id FROM jobs WHERE " + where, args)
            for row in rows:
                self.set_job_status(row["id"], "cancelled")
            return len(rows)

    def recover_jobs(self):
        with self._lock:
            for row in self.query("SELECT id FROM jobs WHERE status IN ('processing','delivering')"):
                self.set_job_status(row["id"], "interrupted")
            return [dict(r) for r in self.query("SELECT * FROM jobs WHERE status='queued' ORDER BY id")]

    def chat_prompt(self, user_id, chat_id,thread_id=0):
        row = self.query_one("SELECT prompt FROM chat_prompts WHERE user_id=? AND chat_id=? AND thread_id=?",
                             (user_id, chat_id,int(thread_id)))
        return row["prompt"] if row else ""

    def set_chat_prompt(self, user_id, chat_id, prompt,thread_id=0):
        self.execute("INSERT INTO chat_prompts(user_id,chat_id,thread_id,prompt) VALUES (?,?,?,?) ON CONFLICT(user_id,chat_id,thread_id)"
                     " DO UPDATE SET prompt=excluded.prompt", (user_id, chat_id,int(thread_id),prompt))

    def chat_exists(self,user_id,chat_id,thread_id=0):
        return self.query_one("SELECT 1 FROM messages WHERE user_id=? AND chat_id=? AND thread_id=? LIMIT 1", (user_id,chat_id,thread_id)) is not None

    def list_chats(self,user_id):
        rows = self.query(
            "SELECT m.chat_id,m.thread_id,MAX(m.chat_type) AS chat_type,COUNT(*) AS messages,"
            "MAX(m.created_at) AS last_message,MAX(m.id) AS latest_id,"
            "COALESCE(meta.label,'') AS label,COALESCE(meta.bookmarked,0) AS bookmarked,"
            "COALESCE(p.prompt,'') AS prompt,"
            "(SELECT SUBSTR(text,1,160) FROM messages x WHERE x.user_id=m.user_id AND x.chat_id=m.chat_id AND x.thread_id=m.thread_id ORDER BY id DESC LIMIT 1) AS last_text "
            "FROM messages m LEFT JOIN chat_metadata meta ON meta.user_id=m.user_id AND meta.chat_id=m.chat_id AND meta.thread_id=m.thread_id "
            "LEFT JOIN chat_prompts p ON p.user_id=m.user_id AND p.chat_id=m.chat_id AND p.thread_id=m.thread_id "
            "WHERE m.user_id=? GROUP BY m.chat_id,m.thread_id ORDER BY bookmarked DESC,latest_id DESC", (user_id,))
        return [dict(r) for r in rows]

    def set_chat_metadata(self,user_id,chat_id,thread_id=0,**fields):
        with self._lock,self._conn:
            self._conn.execute("INSERT OR IGNORE INTO chat_metadata(user_id,chat_id,thread_id) VALUES (?,?,?)", (user_id,chat_id,thread_id))
            for key in ("label","bookmarked"):
                if key in fields:
                    self._conn.execute(f"UPDATE chat_metadata SET {key}=? WHERE user_id=? AND chat_id=? AND thread_id=?", (fields[key],user_id,chat_id,thread_id))

    def message_page(self,user_id,chat_id,thread_id=0,limit=100,before=None,search=""):
        limit = max(1,min(500,int(limit)))
        where,args = "user_id=? AND chat_id=? AND thread_id=?", [user_id,chat_id,thread_id]
        if before is not None:
            where += " AND id<?"
            args.append(int(before))
        if search:
            where += " AND text LIKE ? ESCAPE '\\'"
            args.append("%"+search.replace("\\","\\\\").replace("%","\\%").replace("_","\\_")+"%")
        rows = self.query("SELECT * FROM messages WHERE "+where+" ORDER BY id DESC LIMIT ?", (*args,limit+1))
        has_more = len(rows)>limit
        messages = [dict(r) for r in reversed(rows[:limit])]
        return {"messages":messages,"has_more":has_more,"next_before":messages[0]["id"] if messages and has_more else None}

    def list_jobs(self,status="",user_id=None,limit=100):
        where,args = "1=1",[]
        if status:
            where += " AND j.status=?"
            args.append(status)
        if user_id is not None:
            where += " AND j.user_id=?"
            args.append(user_id)
        rows = self.query("SELECT j.*,u.first_name,u.last_name,u.username FROM jobs j LEFT JOIN users u ON u.id=j.user_id WHERE "+where+" ORDER BY j.id DESC LIMIT ?", (*args,max(1,min(500,int(limit)))))
        counts = {r["status"]:r["n"] for r in self.query("SELECT status,COUNT(*) AS n FROM jobs GROUP BY status")}
        return {"jobs":[dict(r) for r in rows],"counts":counts}

    def save_preset(self,name,prompt,preset_id=None):
        if preset_id is None:
            preset_id = self.execute("INSERT INTO prompt_presets(name,prompt,created_at,updated_at) VALUES (?,?,?,?)", (name,prompt,utcnow(),utcnow())).lastrowid
        else:
            self.execute("UPDATE prompt_presets SET name=?,prompt=?,updated_at=? WHERE id=?", (name,prompt,utcnow(),preset_id))
        row = self.query_one("SELECT * FROM prompt_presets WHERE id=?", (preset_id,))
        return dict(row) if row else None

    # -- users ----------------------------------------------------------
    def upsert_user(self, user_id: int, first_name="", last_name="",
                    username="", chat_id=0, is_bot=False) -> dict:
        now = utcnow()
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO users (id, first_name, last_name, username,"
                    " role, blocked, notes, first_seen, last_seen, chat_id,"
                    " is_bot) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (user_id, first_name, last_name, username, "user", 0,
                     "", now, now, chat_id, 1 if is_bot else 0))
            else:
                self._conn.execute(
                    "UPDATE users SET first_name=?, last_name=?, username=?,"
                    " last_seen=?, chat_id=CASE WHEN ?!=0 THEN ? ELSE chat_id"
                    " END WHERE id=?",
                    (first_name, last_name, username, now, chat_id, chat_id,
                     user_id))
            self._conn.commit()
            fresh = self._conn.execute(
                "SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
            return dict(fresh)

    def get_user(self, user_id: int):
        row = self.query_one("SELECT * FROM users WHERE id=?", (user_id,))
        return dict(row) if row else None

    def list_users(self, search="", role="", blocked=""):
        sql = "SELECT * FROM users WHERE 1=1"
        params: list = []
        if search:
            sql += (" AND (CAST(id AS TEXT) LIKE ? OR username LIKE ?"
                    " OR first_name LIKE ? OR last_name LIKE ?)")
            like = f"%{search}%"
            params += [like, like, like, like]
        if role in ("user", "admin"):
            sql += " AND role=?"
            params.append(role)
        if blocked == "1":
            sql += " AND blocked=1"
        elif blocked == "0":
            sql += " AND blocked=0"
        sql += " ORDER BY last_seen DESC"
        return [dict(r) for r in self.query(sql, tuple(params))]

    def set_user_fields(self, user_id: int, **fields) -> dict | None:
        allowed = {"role", "blocked", "notes", "context_limit_override",
                   "system_prompt_override"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if updates:
            cols = ", ".join(f"{k}=?" for k in updates)
            self.execute(f"UPDATE users SET {cols} WHERE id=?",
                         (*updates.values(), user_id))
        return self.get_user(user_id)

    # -- messages --------------------------------------------------------
    def add_message(self, user_id: int, direction: str, text: str,
                    chat_id=0, chat_type="private", kind="text",
                    in_context=True, status="done",
                    telegram_message_id=None,thread_id=0,job_id=None) -> int:
        cur = self.execute(
            "INSERT INTO messages (user_id, chat_id, thread_id,job_id,chat_type, direction,"
            " kind, text, in_context, status, telegram_message_id,"
            " created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (user_id, chat_id,int(thread_id),job_id,chat_type,direction,kind,text,
             1 if in_context else 0, status, telegram_message_id, utcnow()))
        return cur.lastrowid

    def history(self, user_id: int, limit=10000, eligible_only=False, chat_id=None,thread_id=None,before_id=None):
        """Full stored history for ONE user, oldest first. Never unscoped."""
        sql = ("SELECT messages.*,COALESCE((SELECT inbound_id FROM jobs WHERE jobs.id=messages.job_id),messages.id) AS turn_id FROM messages WHERE user_id=?")
        params = [user_id]
        if chat_id is not None:
            sql += " AND chat_id=?"
            params.append(chat_id)
        if thread_id is not None:
            sql += " AND thread_id=?"
            params.append(thread_id)
        if before_id is not None:
            sql += " AND (direction!='in' OR id<=?)"
            params.append(before_id)
        if eligible_only:
            sql += " AND in_context=1 AND status='done'"
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(max(1, min(10000, int(limit))))
        return [dict(r) for r in reversed(self.query(sql, tuple(params)))]

    def user_stats(self, user_id: int) -> dict:
        row = self.query_one(
            "SELECT COUNT(*) AS n, SUM(CASE WHEN direction='in' THEN 1 ELSE 0"
            " END) AS nin, SUM(CASE WHEN direction='out' THEN 1 ELSE 0 END)"
            " AS nout, SUM(LENGTH(text)) AS chars, MAX(created_at) AS last"
            " FROM messages WHERE user_id=?", (user_id,))
        d = dict(row) if row else {}
        return {"messages": d.get("n") or 0, "inbound": d.get("nin") or 0,
                "outbound": d.get("nout") or 0,
                "chars": d.get("chars") or 0, "last_message": d.get("last")}

    def reset_context(self, user_id: int, chat_id=None,thread_id=None):
        """Retire current context but keep every message viewable."""
        self.cancel_jobs(user_id, chat_id,thread_id)
        where,args = "user_id=? AND status='done' AND kind IN ('text','command')",[user_id]
        if chat_id is not None:
            where += " AND chat_id=?"
            args.append(chat_id)
        if thread_id is not None:
            where += " AND thread_id=?"
            args.append(thread_id)
        self.execute(
            "UPDATE messages SET in_context=0 WHERE "+where,tuple(args))
        self.add_message(user_id, "out", "Context reset by administrator.",
                         chat_id=chat_id or 0,thread_id=thread_id or 0,kind="system", in_context=False)

    def mark_processing(self, msg_id: int):
        self.execute("UPDATE messages SET status='processing' WHERE id=?",
                     (msg_id,))

    def mark_interrupted_on_boot(self) -> int:
        cur = self.execute(
            "UPDATE messages SET status='interrupted' WHERE status="
            "'processing'")
        return cur.rowcount

    # -- settings / meta / events -----------------------------------------
    def get_setting(self, key: str, default="") -> str:
        row = self.query_one("SELECT value FROM settings WHERE key=?", (key,))
        return row["value"] if row else default

    def set_setting(self, key: str, value: str):
        self.execute(
            "INSERT INTO settings (key, value, updated_at) VALUES (?,?,?)"
            " ON CONFLICT(key) DO UPDATE SET value=excluded.value,"
            " updated_at=excluded.updated_at",
            (key, value, utcnow()))

    def all_settings(self) -> dict:
        return {r["key"]: r["value"] for r in self.query("SELECT * FROM settings")}

    def get_meta(self, key: str, default="") -> str:
        row = self.query_one("SELECT value FROM meta WHERE key=?", (key,))
        return row["value"] if row else default

    def set_meta(self, key: str, value: str):
        self.execute(
            "INSERT INTO meta (key, value) VALUES (?,?) ON CONFLICT(key)"
            " DO UPDATE SET value=excluded.value", (key, value))

    def log(self, level: str, source: str, message: str):
        self.execute(
            "INSERT INTO events (level, source, message, created_at)"
            " VALUES (?,?,?,?)", (level, source, message[:2000], utcnow()))
        with self._lock:
            self._log_count += 1
            if self._log_count % 50 == 0:
                self.execute("DELETE FROM events WHERE id NOT IN (SELECT id FROM events ORDER BY id DESC LIMIT 5000)")

    def events(self, level="", limit=300,search="",source=""):
        where,args = "1=1",[]
        if level:
            where += " AND level=?"
            args.append(level)
        if source:
            where += " AND source=?"
            args.append(source)
        if search:
            where += " AND message LIKE ? ESCAPE '\\'"
            args.append("%"+search.replace("\\","\\\\").replace("%","\\%").replace("_","\\_")+"%")
        rows = self.query("SELECT * FROM events WHERE "+where+" ORDER BY id DESC LIMIT ?", (*args,max(1,min(1000,int(limit)))))
        return [dict(r) for r in rows]

    def counts(self) -> dict:
        u = self.query_one("SELECT COUNT(*) AS n FROM users")
        m = self.query_one("SELECT COUNT(*) AS n FROM messages")
        today = time.strftime("%Y-%m-%d", time.gmtime())
        t = self.query_one(
            "SELECT COUNT(*) AS n FROM messages WHERE created_at LIKE ?",
            (today + "%",))
        return {"users": u["n"], "messages": m["n"],
                "messages_today": t["n"]}

    def daily_stats(self, days=30):
        rows = self.query(
            "SELECT SUBSTR(created_at,1,10) AS day, COUNT(*) AS n FROM"
            " messages WHERE created_at >= DATE('now', ?) GROUP BY day"
            " ORDER BY day ASC", (f"-{days} days",))
        from datetime import datetime,timedelta,timezone
        found = {r["day"]:r["n"] for r in rows}
        today = datetime.now(timezone.utc).date()
        return [{"day":(today-timedelta(days=offset)).isoformat(),"n":found.get((today-timedelta(days=offset)).isoformat(),0)} for offset in range(max(1,min(366,int(days)))-1,-1,-1)]
