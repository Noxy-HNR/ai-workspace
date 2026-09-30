# Relay Station 2 — Telegram bot console

Local web dashboard + bot backend that connects Telegram users to an
existing OpenAI-compatible LLM endpoint (e.g. llama-server on `:10000`).
Chat only: no tools, no shell, no filesystem, no computer control.

- Dashboard: `http://127.0.0.1:9095` (loopback only, no login — same trust
  model as the existing `:9090` dashboard)
- Stack: Python 3 stdlib only (`http.server`, `sqlite3`, `urllib`). No pip
  packages, nothing to break on update.
- Tests: `py -m unittest discover -s tests -t .` (stdlib only; results in
  [verification/validation.md](verification/validation.md))

## Start / stop

```powershell
cd C:\AI\TelegramBot
.\start-telegram-bot.ps1        # starts hidden, prints the URL
.\start-telegram-bot.ps1 -Foreground # visible terminal for troubleshooting
.\start-telegram-bot.ps1 -PythonPath "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe"
```

The launcher detects an already-running dashboard instead of launching a
duplicate. It checks `py` and `python` on PATH, then standard per-user Python
installs under `%LOCALAPPDATA%\Programs\Python`; `-PythonPath` selects another
executable. Startup errors are written under `logs/`. A process-owned data
directory lock prevents two updated instances from sharing a database.
Use Stop polling in the dashboard to pause replies; this cancels pending
work. Start polling resumes once the previous long poll has exited.
To upgrade a running older version, stop only its Python process and run
the launcher again. Direct run (visible console + logs to stdout):

```powershell
py -u app.py --port 9095
```

## Desktop shortcut and main dashboard

The desktop shortcut **Relay Station - Telegram Bot** starts or reuses this bot
and opens `http://127.0.0.1:9095`. It uses a hidden launcher, so no terminal stays
open. Reinstall it with `powershell -File .\install-desktop-shortcut.ps1`.

The existing `http://127.0.0.1:9090` dashboard includes a **Relay Station /
Telegram Bot** service card with Start, Stop, Restart, and Open console controls.
It reports the bot's process and console readiness independently of llama-server.
An older running backend is recognized and marked; Restart loads the updated bot.
Starting the bot does not load or switch a model.

The launcher records its PID, exact command, and process creation time under
`run/`. Stop refuses an unrelated process or a reused PID. Updated instances
support a protected graceful shutdown endpoint; the dashboard permits a forced
stop only after those ownership checks, for older or unresponsive instances.
Direct controls are `start-telegram-bot.ps1 -OpenBrowser` and
`stop-telegram-bot.ps1 -Force`. Concurrent shortcut launches share a named mutex.

## First-run setup (all in the dashboard, no source edits)

1. Open Settings → Telegram, paste the @BotFather token, Save.
2. Check Overview: Telegram should read Online, model Up/Down honestly.
3. Optionally set your Telegram ID as Administrator Telegram ID in
   Settings → Behavior. Otherwise the first private-chat messager claims
   admin. Groups cannot claim ownership. Existing admins stay admins.
4. Tune Settings → LLM (`http://127.0.0.1:10000/v1` default, blank model =
   whatever llama-server currently serves), generation, and context.

## Using the console

- **Conversations:** select a person and chat/topic, add a local label or
  bookmark, search stored text, and load earlier pages. Copy messages, inspect
  effective context, and export the selected conversation as JSON or Markdown.
- **Chat instructions:** use a prompt for this chat/topic or the person/global
  default. Reusable prompt starters fill the editor; save explicitly to apply.
  Saving instructions starts fresh context and cancels pending replies there.
- **People:** filter access/roles, keep owner-only notes, and save person defaults
  together. Invalid profile edits leave every field unchanged.
- **Reply queue:** inspect job states, filter by person/status, cancel pending
  work, or retry eligible failed/interrupted replies in their original scope.
- **Activity and Insights:** inspect filtered events, daily activity, delivery
  outcomes, and active people. Light/dark themes, mobile navigation, a `Ctrl+K`
  command menu, and preserved conversation drafts make the console easier to use.

## Architecture decisions (why)

- **One conversation per user, chat, and forum topic**, stored in SQLite
  (`data/bot.db`). A user's private history cannot enter a group reply;
  different groups and topics stay separate. The dashboard has a chat/topic
  selector. Existing records migrate to topic 0. Old history remains stored; records
  without a chat ID are separately viewable and are not reused in replies.
- **Stored history vs. effective context are separate.** The dashboard's
  Conversation tab shows every stored message with an *in context* /
  *archived* badge, plus a context inspector (system prompt + the exact
  messages that will be sent + token estimate + drop reasons).
- **Context window = last N messages + token cap** (`ctx_history_limit`
  default 30, `ctx_max_tokens` default 6000 estimated as chars/4, oldest
  dropped first, newest user message never dropped). Per-user overrides
  for both the count and the system prompt.
- **Durable generation queue.** Inbound text and the queue entry are saved
  in one SQLite transaction before acknowledging the update. Duplicate
  update IDs do not create another job. Queued jobs recover on restart.
  Jobs interrupted while generating or delivering require `/retry` because
  Telegram delivery cannot be made exactly-once across a crash. A retry
  after partial delivery can repeat an already-sent chunk.
- **Per-user FIFO + global LLM cap.** A second message while busy waits in
  that user's queue; `bot_max_parallel_llm` defaults to one request at a
  time. Saved configuration is preserved. Answers are associated with their
  originating input even when later messages arrive early. Each user may
  queue eight additional messages by default; up to 64 user workers exist.
- **Replies chunked at about 4000 Telegram UTF-16 units**, preserving Unicode;
  stored as one message, delivered as parts. HTML mode converts supported
  Markdown and escapes raw model HTML; plain text is configurable. A definite
  formatting rejection falls back to plain text. Replies can attach to the
  triggering message and preserve its forum topic. Typing shows while generating.
- **Groups:** `mention` mode (default) replies only when addressed
  (@mention or reply-to-bot); anything else is stored but excluded from
  context. `all` answers everything. Replies to other bots and commands
  addressed to other bots do not trigger this bot. Edits do not regenerate
  replies. Attachment captions remain unsupported.
- **Blocked users are silently ignored** (stored as `ignored`, excluded
  from context, no reply). Last-admin revocation is refused.
- **Telegram commands:** `/start`, `/help`, `/id`, `/reset`, `/cancel`,
  `/retry`, `/status`, `/context`, `/prompt <text>`, `/prompt reset`, and
  `/privacy`, and admin `/stats`. Prompt/reset/cancel apply to the current
  chat/topic. `/privacy` explains storage, owner access, and model routing.
  `/cancel` discards pending output and interrupts delivery waits. An active
  model request retains its generation slot and RAM watchdog until bounded
  completion, preventing a replacement request from overlapping inference that
  a server may continue after client disconnect. Already-sent messages remain
  in Telegram. Process shutdown uses separate bounded worker/transport cleanup.
- **Secrets:** token/API key live in the local SQLite db, the API only
  ever emits masked placeholders, and the Telegram client redacts the
  token from every error string. `data/` is git-ignored.
- **Settings and access:** invalid edits reject the entire settings save.
  Access can be open, admins-only, or a Telegram ID allowlist. Remote model
  endpoints require explicit opt-in; HTTP redirects are refused. Empty
  secret fields keep existing secrets; the explicit Clear saved value control
  removes a selected secret on the next save. Invalid edits also reject secret
  clearing. Empty reply templates stay empty.
- **Model safety:** requests have a total wall-clock deadline, finite reply
  token limits and a 4 MiB response cap. Calls to the local `:10000` endpoint
  require at least 1 GiB available Windows RAM (`GlobalMemoryStatusEx`).
  While generating, sustained availability below 0.8 GiB for three seconds
  closes the request and asks the existing `:9090` dashboard to stop
  `llama-server`. No server is launched or model switched by this bot.
- **Dashboard:** same-origin CSRF checks and Host validation protect writes.
  Status probes cache their results for 20 seconds. Conversation export
  downloads the selected chat/topic as JSON or Markdown (up to 10,000 recent
  records), explicitly reporting omitted older history. Message pages are capped
  at 500 records (150 initially in the UI); job pages at 500 and activity responses
  at 1000. About 5000 activity events are retained. Manual messages and chat
  instructions allow 4000 characters; labels allow 100. Navigation works on
  narrow screens and errors appear in a visible notice.

## Data locations

- `data/bot.db` — users, message history, settings, jobs, chat preferences,
  prompt starters, and events
- `logs/` — launcher stdout/stderr; activity events are in SQLite
- `static/` — dashboard UI (`index.html`, `style.css`, `app.js`)

## Manual configuration needed

- Telegram bot token (Settings → Telegram). Nothing works until this is set.
- LLM endpoint if different from `http://127.0.0.1:10000/v1`.
- That's it — everything else has working defaults.

## Known limitations

- No streaming: replies are generated whole, then chunk-sent. (Typing
  indicator covers the wait.)
- Text only: photos/voice/stickers get a configurable fallback reply.
- Forum topics are isolated. Arbitrary branching within one chat/topic is not
  supported. Chat/topic prompts override per-user/global ones. Context token
  counts are character-based estimates rather than tokenizer measurements.
- Model changes on the llama side are picked up automatically when the
  model field is blank (first entry of `/v1/models`, cached for 10 seconds); a pinned name goes
  stale if you unload that model — the bot then replies with the error
  template and logs the failure.
- Dashboard has no login; it binds 127.0.0.1 only. Don't port-forward it.

## Offline dashboard preview

```powershell
py tests\preview_dashboard.py --port 9096
```

This uses disposable sample conversations, simulated Telegram/model checks,
and a temporary database. Six fictional users, long history, forum topics,
prompts, presets, and queue states exercise the UI. Polling controls, retries,
and manual sends are simulated; no real Telegram/model calls or generation
workers start. If `py` is unavailable, use your Python executable with the same
arguments, for example `& "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe"`.
Close
the terminal with Ctrl+C when finished. The preview banner distinguishes it
from the running bot.

## Verification

The tests use temporary SQLite databases, loopback fake HTTP servers, and
stub Telegram clients. They cover FIFO ordering, persistence, restart
recovery, cancellation/reset races, failed delivery, user/chat isolation,
group/topic targeting, admin ownership, formatting, Unicode, validation, CSRF,
search/pagination, explicit export truncation, profile/secret atomicity,
concurrent deletion/administrator guards, schema migration, safe previews,
startup/shutdown cleanup, cached status, total deadlines, and the pre-request
RAM gate. The watchdog's emergency
stop is tested with mocks; no real server is stopped by the test suite.
Live Telegram delivery and an actual low-RAM shutdown require separate
operational validation.
