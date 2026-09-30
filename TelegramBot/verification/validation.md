# Relay Station verification — September 29, 2026

## Desktop and main-dashboard integration

- Installed `C:\Users\braxt\OneDrive\Desktop\Relay Station - Telegram Bot.lnk`
  with a dedicated icon and hidden start/reuse-and-open launcher. Verified its
  target, arguments, working directory, icon, and existence through Windows COM.
- The main dashboard now reports `telegram-bot` on port 9095 and provides Start,
  Stop, Restart, and Open console controls. Its previously stopped server was
  started independently; the live bot and model PIDs remained 32732 and 8456.
- Live `/api/start` for `telegram-bot` returned the already-running result, with
  no duplicate process. The existing legacy backend is identified explicitly.
- Five shutdown endpoint tests pass: response-before-shutdown ordering, CSRF,
  loopback Host/Origin protection, invalid methods/bodies, and no model/Telegram
  activity. Process-ownership fixtures passed 30 assertions under both PowerShell
  5.1 and 7.6, including reused PID, changed command, malformed record, and refusal
  of unrelated processes. No live Stop/Restart was used in verification.
- Dashboard inline JavaScript syntax and isolated card/control fixtures passed.
  PowerShell syntax passed for the dashboard and all lifecycle/shortcut scripts.
- Consolidated Python test run after the integration: **119 tests passed in
  34.894 seconds**.
- Disposable Windows PowerShell 5.1 lifecycle fixture: **14 checks passed in
  8.724 seconds**, covering cold launch, exact process ownership, reuse of the
  same PID, graceful stop without `-Force`, process exit, and released port.
  Copied scripts matched the current source; all fixture processes were removed.
- Main-dashboard screenshot: `main-dashboard-telegram-bot.png`.
- Integration backup: `C:\AI\Backups\TelegramBot-management-20260929-234542`.

## Phase 2 — expanded features

- Final consolidated test run: **114 tests passed in 34.526 seconds**, including
  the final context-budget fixes and dashboard-handler shutdown regression.
- The API extension file covers profile/secret atomicity, literal search, stable
  pagination, chat/topic metadata and prompts, presets, queue filtering, export
  truncation, same-origin protections, migration, future-schema refusal, concurrent
  administrator/deletion races, cancellation signalling, and startup cleanup.
- Runtime regressions use simulated Telegram/model services for topic isolation,
  originating-message replies, HTML/plain formatting, Unicode chunks, delivery
  failures, request deadlines, model-slot retention during cancellation, and
  worker shutdown. RAM gates and emergency stops remain mocked.
- The demo now has six fictional people, more than 200 messages in Alex's private
  conversation, separate forum topics, archived pages, labels/bookmarks, instructions,
  presets, events, and queued/failed/interrupted jobs. Preview safety checks prove
  start/retry controls create no workers and call neither model nor transport.
- The launcher supports `-PythonPath` and standard per-user Python installs when
  `py`/`python` are absent from PATH. Startup failures release owned resources.
- Browser validation used only disposable preview data. Verified 216-message
  pagination, literal message search, topic-specific prompts, per-topic drafts,
  queue retry and conversation navigation, and rejected/accepted settings saves.
- Final prompt-starter creation verified in a fresh browser tab; the reusable
  starter was saved to the disposable database and appeared selected in the UI.
- Final desktop and 390-pixel mobile layouts inspected. Document width was 375
  pixels, with navigation visible and horizontally scrollable inside its own row.
- Final screenshots: `dashboard-final-desktop.jpg`, `dashboard-final-mobile.jpg`.
- Final JavaScript syntax and PowerShell launcher syntax checks passed. HTML IDs
  are unique, with no missing literal JavaScript ID references. Conversation,
  starter, manual-send and profile input limits match server validation.
- Source backups: `C:\AI\Backups\TelegramBot-20260929-151055` and
  `C:\AI\Backups\TelegramBot-phase2-20260929-154147`.
- The live bot was not restarted for this phase. No live Telegram delivery or real
  model-generation request was used to validate these changes. No live database,
  credentials, `.env` file, auth store, or private user log was inspected.

Current limits: generation remains non-streaming and attachments/captions are
unsupported. User/chat/forum-topic context is isolated. Cancellation discards
output and interrupts delivery waits, but keeps an active model's slot and RAM
watchdog until completion/deadline to avoid overlapping server inference. Stored
history survives `/reset`; `/privacy` explains owner access and model routing.
Exactly-once delivery across crashes is not guaranteed; explicit retries may
repeat previously delivered chunks. JSON/Markdown exports include at most 10,000
recent records and disclose truncation. The console remains loopback-only without
a login, and cloud endpoints are never enabled automatically.

## Phase 1 — historical validation record

- `py -m unittest discover -s tests -t .`: 55 tests passed.
- `py -m compileall -q app.py bot tests`: passed.
- `node --check static/app.js`: passed.
- PowerShell parser check of `start-telegram-bot.ps1`: passed.
- Launcher against the existing service: detected it and did not start a duplicate.
- Browser: private/group context selector, invalid settings rejection, successful
  settings save, conversation rendering and 390-pixel mobile navigation verified
  using `tests/preview_dashboard.py` and disposable demo data.
- Mobile DOM measurement: 390-pixel viewport; document width 375; navigation visible.
- Screenshots: `dashboard-preview.jpg` and `dashboard-mobile.jpg`.
- Live bot on port 9095 was left running; source changes require its restart.
- No live database contents, credentials, .env files or auth stores were inspected.
- No Telegram messages or real model generation requests were sent during testing.
- RAM watchdog emergency shutdown was validated with mocked memory and a mocked
  stop function; it did not stop the real model server.
- Source backup: `C:\AI\Backups\TelegramBot-20260929-151055`.

First-pass limitations at that time: non-streaming cancellation discarded pending output but
does not stop an already-running model generation. Delivery cannot be exactly-once
across a crash; interrupted jobs need explicit retry, and partial retries can repeat
chunks. Attachments and forum-topic isolation were not supported. Existing saved
model configuration is preserved. No cloud endpoint is enabled automatically.
