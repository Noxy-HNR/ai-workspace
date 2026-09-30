"""Dashboard preview with disposable demo data and no external API calls."""
import argparse
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from http.server import ThreadingHTTPServer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import app
from bot.db import Database
from bot.poller import Poller
from bot import settings
from tests.test_e2e import StubClient


class PreviewClient(StubClient):
    def send_text(self, chat_id, text, thread_id=0, **kwargs):
        return super().send_text(chat_id, text)


class PreviewPoller(Poller):
    """Interactive dashboard fixtures that never create model/polling workers."""

    def start(self):
        self._stop.clear()
        self.status.update(running=True, last_error="Demo polling is simulated.")

    def stop(self, timeout=1):
        self._stop.set()
        self.status.update(running=False, last_error="Demo polling is paused.")

    def _enqueue(self, uid, item):
        # Retry remains useful in the demo, but only persists another fixture.
        return bool(item if "id" in item else self.db.create_job(item))


def seed_demo(db):
    """Create fictional, varied conversations without touching the live store."""
    now = datetime.now(timezone.utc).replace(microsecond=0)

    def at(days=0, minutes=0):
        return (now - timedelta(days=days, minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")

    def message(uid, direction, text, cid=None, *, thread=0, days=0,
                minutes=0, context=True, kind="text", status="done"):
        cid = uid if cid is None else cid
        mid = db.add_message(uid, direction, text, chat_id=cid,
                             chat_type="private" if cid > 0 else "supergroup",
                             thread_id=thread, in_context=context, kind=kind,
                             status=status)
        db.execute("UPDATE messages SET created_at=? WHERE id=?", (at(days, minutes), mid))
        return mid

    accounts = [
        (101, "Alex", "Morgan", "demo_alex", 101),
        (202, "Sam", "Rivera", "demo_sam", 202),
        (303, "Jamie", "Chen", "demo_jamie", 303),
        (404, "Taylor", "Brooks", "demo_taylor", 404),
        (505, "Robin", "", "", 505),
        (606, "Casey", "Patel", "demo_casey", 0),
    ]
    for uid, first, last, username, cid in accounts:
        db.upsert_user(uid, first, last, username, chat_id=cid)
        db.execute("UPDATE users SET first_seen=?,last_seen=? WHERE id=?",
                   (at(24), at(minutes=(uid // 101 - 1) * 17), uid))
    db.set_user_fields(101, role="admin", notes="Demo owner. All conversations are fictional and disposable.")
    db.set_user_fields(202, notes="Prefers short explanations and worked examples.")
    db.set_user_fields(303, context_limit_override=12, notes="Uses the bot for Python practice.")
    db.set_user_fields(404, blocked=1, notes="Blocked sample account for testing user filters.")

    study_topics = [
        ("cell membranes", "Sketch a phospholipid bilayer, then explain why small nonpolar molecules cross it more easily than ions."),
        ("molarity", "Molarity = moles of solute / liters of solution. Convert volume to liters before dividing."),
        ("classical conditioning", "Identify the unconditioned stimulus and response first. Then find the cue that becomes the conditioned stimulus."),
        ("enzyme activity", "Compare reaction rate while changing one factor at a time. Explain how temperature and pH affect protein structure."),
        ("Lewis structures", "Count valence electrons, connect atoms, complete outer octets, and check formal charges."),
        ("memory retrieval", "Use practice questions without looking at the notes, then check your answer and repeat missed concepts."),
    ]
    # Enough rows to exercise older-message pagination and a useful activity chart.
    for index in range(106):
        topic, explanation = study_topics[index % len(study_topics)]
        age = 14 - index // 8
        in_context = index >= 96
        message(101, "in", f"Study session {index + 1}: can we review {topic}?",
                days=age, minutes=22, context=in_context)
        message(101, "out", explanation + "\n\nTry explaining it in your own words before checking your notes.",
                days=age, minutes=21, context=in_context)
    message(101, "in", "I have 45 minutes before class. Help me review cellular respiration.", minutes=12)
    message(101, "out", "Here's a focused 45-minute plan:\n\n1. 10 min — glycolysis inputs and outputs\n2. 15 min — citric acid cycle and electron carriers\n3. 15 min — electron transport and chemiosmosis\n4. 5 min — answer three questions from memory\n\nStart with this: where does each stage occur?", minutes=11)
    message(101, "in", "Glycolysis is in the cytosol. The cycle is in the mitochondrial matrix.", minutes=8)
    message(101, "out", "Correct. The electron transport chain is in the inner mitochondrial membrane. What gradient powers ATP synthase?", minutes=7)
    db.set_chat_metadata(101, 101, label="Study desk", bookmarked=1)
    db.set_chat_prompt(101, 101, "Act as a patient study partner. Ask one question at a time and help me check my understanding.")

    message(101, "in", "@relay_demo Help us review chapter 4.", -900, days=2)
    message(101, "out", "Let's build a review checklist: vocabulary, diagrams, and one worked example for each key idea.", -900, days=2)
    message(101, "in", "@relay_demo Can you explain active transport to our group?", -900, minutes=39)
    message(101, "out", "Active transport moves substances against their concentration gradient using energy. A sodium–potassium pump is one example.", -900, minutes=38)
    db.set_chat_metadata(101, -900, label="Biology study group")

    forum = -100777
    message(101, "in", "Let's keep chemistry questions in this topic.", forum, thread=42, days=1)
    message(101, "out", "This topic has its own instructions and conversation context. Send a problem and we'll work through it.", forum, thread=42, days=1)
    message(101, "in", "What does a negative formal charge tell us?", forum, thread=42, minutes=28)
    message(101, "out", "It means the atom is assigned more valence electrons in the Lewis structure than the neutral free atom has.", forum, thread=42, minutes=27)
    db.set_chat_metadata(101, forum, 42, label="Chemistry • review", bookmarked=1)
    db.set_chat_prompt(101, forum, "Use short chemistry explanations with units, electron counts, and one worked example.", 42)
    message(101, "in", "Can you quiz me on reinforcement schedules?", forum, thread=73, minutes=24)
    message(101, "out", "Sure. Which schedule gives reinforcement after an unpredictable number of responses?", forum, thread=73, minutes=23)
    db.set_chat_metadata(101, forum, 73, label="Psychology • flashcards")
    db.set_chat_prompt(101, forum, "Give one psychology practice question at a time. Wait for an answer before explaining.", 73)

    for age in range(11, 0, -1):
        message(202, "in", f"Can you help me structure my revision notes for day {12 - age}?", days=age, context=False)
        message(202, "out", "Use three sections: key terms, a diagram, and two retrieval questions. Keep examples beside the term they explain.", days=age, context=False)
    message(202, "in", "Summarize passive versus active transport in two sentences.", minutes=31)
    message(202, "out", "Passive transport follows a concentration or electrochemical gradient without direct energy input. Active transport uses energy to move substances against that gradient.", minutes=30)
    message(202, "in", "/reset", minutes=20, context=False, kind="command")
    message(202, "out", "Fresh conversation started in this chat. History retained; pending replies cancelled.", minutes=20, context=False, kind="system")
    message(202, "in", "Create a checklist for my lab notebook.", minutes=18)
    message(202, "out", "Before lab:\n• Date, title, and goal\n• Materials and method\n• Data table with units\n\nAfter lab:\n• Observations and calculations\n• Sources of uncertainty\n• Conclusion tied to the goal", minutes=17)
    db.set_chat_metadata(202, 202, label="Lab preparation")

    message(303, "in", "Why does this Python loop skip entries?\n\n```python\nfor item in items:\n    if item < 0:\n        items.remove(item)\n```", days=1)
    message(303, "out", "You're changing the list while iterating over it. Build a new list instead:\n\n```python\nitems = [item for item in items if item >= 0]\n```\n\nThe comprehension visits each original item exactly once.", days=1)
    message(303, "in", "How can I test that with an empty list and repeated negatives?", minutes=44)
    message(303, "out", "Try these cases:\n\n```python\ncases = [[], [-1, -1, 2], [0, 1, 2], [-3, -2]]\nfor values in cases:\n    print([item for item in values if item >= 0])\n```\n\nExpected outputs: [], [2], [0, 1, 2], [].", minutes=43)
    db.set_chat_metadata(303, 303, label="Python practice", bookmarked=1)

    message(404, "in", "[Demo ignored message from blocked user]", minutes=60, context=False, kind="ignored")
    message(505, "in", "[non-text attachment]", days=2, context=False, kind="unsupported")
    message(505, "out", "I can read text messages. Paste the question here and I'll help.", days=2, context=False)
    message(505, "in", "Hello! Can we work through a practice problem?", minutes=65)
    message(505, "out", "Of course. Send the problem and tell me which step is confusing.", minutes=64)
    message(606, "in", "@relay_demo What's the difference between a hypothesis and a prediction?", -900, minutes=76)
    message(606, "out", "A hypothesis is a proposed explanation. A prediction describes an observation you expect if that explanation is correct.", -900, minutes=75)
    db.set_chat_metadata(606, -900, label="Biology study group")

    for index, (uid, cid, thread, text, status) in enumerate([
        (202, 202, 0, "Turn my lab checklist into a brief preparation plan.", "failed"),
        (303, 303, 0, "Explain the difference between a shallow and deep copy.", "interrupted"),
        (101, forum, 42, "Give me one formal-charge practice problem.", "queued"),
        (505, 505, 0, "Can you help me practice balancing equations?", "queued"),
    ], 1):
        job = db.create_job(dict(user_id=uid, chat_id=cid, thread_id=thread,
                                chat_type="private" if cid > 0 else "supergroup",
                                text=text, update_id=10000 + index, tg_msg_id=index))
        db.set_job_status(job["id"], status)
        stamp = at(minutes=16 - index * 3)
        db.execute("UPDATE jobs SET created_at=?,updated_at=? WHERE id=?", (stamp, stamp, job["id"]))
        db.execute("UPDATE messages SET created_at=? WHERE id=?", (stamp, job["inbound_id"]))
        if status == "failed":
            message(uid, "out", "[Reply failed; see Activity]", cid, thread=thread,
                    minutes=13, context=False, kind="error", status="failed")

    db.save_preset("Study partner", "Ask one focused question at a time. Help me explain ideas in my own words and correct misunderstandings gently.")
    db.save_preset("Clear explanations", "Answer concisely, use a concrete example, and define unfamiliar terms.")
    db.save_preset("Python practice", "Explain Python code step by step. Suggest small examples and cover useful edge cases.")
    event_rows = [
        ("info", "preview", "Demo loaded. Every conversation is fictional; Telegram and model requests are simulated."),
        ("info", "queue", "Queued a chemistry practice question in forum topic 42."),
        ("warning", "reply", "Demo delivery interrupted. Use Retry in the queue to inspect the workflow."),
        ("error", "reply", "Demo model request failed. No real service was contacted."),
        ("info", "admin", "Sam's conversation context reset; stored history retained."),
        ("info", "admin", "Study partner instructions saved for Alex's private chat."),
        ("info", "server", "Disposable dashboard preview started on loopback."),
    ]
    for index, (level, source, text) in enumerate(reversed(event_rows)):
        db.log(level, source, text)
        db.execute("UPDATE events SET created_at=? WHERE id=(SELECT MAX(id) FROM events)", (at(minutes=12 - index),))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=9096)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="relay-preview-") as data:
        app.DB = Database(Path(data)/"demo.db")
        for key,(default,*_) in settings.DEFS.items():
            app.DB.set_setting(key,default)
        app.PREVIEW = True
        stub = PreviewClient()
        app.make_client = lambda: stub
        app.check_telegram = lambda: dict(configured=True, connected=True,bot_username="relay_demo",error="")
        app.check_llm = lambda s: dict(connected=True,models=["demo-model"],base_url="Demo endpoint (simulated)",model="demo-model",error="")
        app.POLLER = PreviewPoller(app.DB,app.get_settings,lambda:stub)
        app.POLLER.status.update(last_error="Preview mode. Polling and generation are simulated.")
        seed_demo(app.DB)
        server = ThreadingHTTPServer(("127.0.0.1",args.port),app.Handler)
        print(f"PREVIEW http://127.0.0.1:{server.server_port}",flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
            app.DB.close()


if __name__ == "__main__": main()
