# AI Workspace

A local-first AI workstation for a Windows PC with a 12 GB GPU and 15 GB of RAM. It runs
open-weight language models on the machine itself (llama.cpp), and wraps them in tools for
tuning, serving, chatting, coding, and on-device vision and search. The design centres on
two things: getting the most out of limited hardware, and not crashing it. A careless model
load once exhausted RAM and forced a hard reset, so resource guards are built into the
tooling rather than left to the operator.

> The stack is written for Windows and PowerShell. Most Python components are standard
> library only and have their own test suites.

## Components

| Component | What it does | Stack | Details |
|---|---|---|---|
| **Local AI dashboard** | Web console (port 9090) for loading models, editing per-model settings, and managing services | PowerShell server, single-page HTML | [Start](#start) below |
| **Model Tuner** | Finds the fastest safe llama-server settings for each model: plans placement and context from the GGUF header, then measures only what matters, and never trades answer quality for speed | Python 3.13+, stdlib, pytest suite | [Tools/model-tuner](Tools/model-tuner/README.md), [design](Tools/model-tuner/DESIGN.md) |
| **Relay Station (Telegram bot)** | Connects Telegram users to the local OpenAI-compatible endpoint, with a loopback web console, per-chat context limits, and secret handling | Python stdlib (`http.server`, `sqlite3`), 119 unit tests | [TelegramBot](TelegramBot/README.md) |
| **Local coding agent** | Installs an OpenCode profile into a repository that points at the local model. It refuses to switch models or overwrite configs, and enforces a session time limit and a RAM watchdog | PowerShell, OpenCode | [Tools/local-coding-agent](Tools/local-coding-agent/README.md) |
| **On-device inference service** | Shared loopback service that runs text embeddings on the Intel iGPU and object detection on the NPU, chosen by measured benchmarks | OpenVINO, Python | [Tools/npu-services](Tools/npu-services/README.md) |
| **Website JSONL scraper** | Polite crawler for building text datasets: honours `robots.txt`, fails closed, and deduplicates by content hash | Python | [Tools/website-jsonl-scraper](Tools/website-jsonl-scraper/README.md) |
| **Desktop project launchers** | Desktop shortcuts with custom icons that start and reuse local web apps, verifying readiness first | PowerShell | [Tools/project-launchers](Tools/project-launchers/README.md) |
| **cptr safety profile** | Audit, apply, and roll back a safety profile for the cptr service | PowerShell, tests | `Tools/cptr-safety-profile.ps1` |
| **Chemistry tools** | Lewis-structure app for the TI-84 Evo calculator suite | Python for TI-84 Evo | `Chemistry-Suite-Transfer/LEWIS.py` |

## Design principles

- **Local first.** Models run on-device. Cloud endpoints are used only when the operator
  configures them. Secrets stay in local files and are never printed back to the UI or logs.
- **Hardware-aware.** Model placement, context size, and batch settings are derived from each
  model's file and the machine's free memory. Long jobs use request timeouts and a RAM watchdog
  that stops the server before the system runs out of memory.
- **One server at a time.** Models are switched through the dashboard, which stops the old
  server before starting the new one. Nothing else launches a second llama-server.
- **Measure, don't guess.** The Model Tuner and the NPU device choice both rest on benchmark
  results, and each one records the evidence it used.
- **Refuse rather than repair silently.** Installers and launchers refuse to overwrite existing
  configuration, switch models, or invent model IDs without an explicit flag.

## Repository layout

```
dashboard-server.ps1, dashboard.html   Local AI dashboard (port 9090)
start.ps1, stop.ps1, update.ps1        Start, stop, and update the stack
setup.ps1, model-*.ps1                 Setup and model management
llama-proxy.py                         Logging proxy (port 10001) in front of llama-server
Config/                                Model settings (models.json), remote backend example
Tools/
  model-tuner/                         Settings search and benchmark (Python)
  local-coding-agent/                  OpenCode profile for local models
  npu-services/                        iGPU and NPU inference service
  website-jsonl-scraper/               Polite crawler for text datasets
  project-launchers/                   Desktop shortcuts and icons
  scripts/, cptr-safety-profile.ps1    Utilities
TelegramBot/                           Relay Station Telegram bot (port 9095)
Chemistry-Suite-Transfer/              TI-84 Evo chemistry program
```

## Start

```powershell
cd C:\AI
.\start.ps1
```

This starts llama-server (with the active model from `Config\models.json`), the dashboard
server, and cptr, then opens the dashboard at `http://127.0.0.1:9090`.

To run only the dashboard server:

```powershell
cd C:\AI
powershell -ExecutionPolicy Bypass -File .\dashboard-server.ps1
```

Local service addresses:

| Service | Address |
|---|---|
| llama-server (OpenAI-compatible API) | `http://127.0.0.1:10000` |
| Request-logging proxy | `http://127.0.0.1:10001` |
| Dashboard | `http://127.0.0.1:9090` |
| Relay Station (Telegram) console | `http://127.0.0.1:9095` |
| cptr | `http://127.0.0.1:8000` |
| On-device inference | `http://127.0.0.1:8788` |

Running llama-server natively on Windows, instead of in Docker Desktop or WSL2, avoids
several GB of idle RAM overhead and starts noticeably faster.

## Models

Available models are scanned from `C:\AI\models\*.gguf`. Per-model startup parameters
(context size, GPU layers, batch size, and so on) are stored in `Config\models.json` and
edited from the dashboard's Models tab. Loading a model there stops any running llama-server
process and starts a new one with that model's saved settings.

The dashboard does not modify your `.env`, model files, or cptr installation.

## Remote model backend (HP server)

The dashboard can also use the llama-server on the HP home server (see `hp/llm/` in
homelab-hp). Its Services tab shows both backends with their health and model, and the
choice of **This PC** or **HP server** applies to Chat and Benchmark. The choice is remembered
in the browser.

Set the address and API key with environment variables, then restart the dashboard server:

```powershell
setx AI_REMOTE_LLM_URL "http://192.168.1.241:10000"   # or the HP's Tailscale name
setx AI_REMOTE_LLM_API_KEY "<the key from the HP's llama-server settings>"
```

Or copy `Config\remote-backend.example.json` to `Config\remote-backend.json` (gitignored)
and fill in `url`, plus either `apiKey` or `apiKeyEnv`, the name of a variable holding the key.
Environment variables win over the file.

The browser never sees the key. The dashboard checks the HP's `/health` and `/v1/models`
itself and relays chat requests through `POST /api/remote/v1/chat/completions`, off its main
request loop, so a slow CPU reply doesn't stall the rest of the dashboard. When the HP is
offline, still loading, or rejects the key, the panel says so and chat replies with the reason
instead of hanging; nothing switches backends on its own.

## Testing

| Suite | Command |
|---|---|
| Relay Station | `py -m unittest discover -s tests -t .` (in `TelegramBot/`) |
| Model Tuner | `uv run --with pytest pytest -q` (in `Tools/model-tuner/`) |
| Model Tuner, from a scratch venv | `python -m pytest -q` after `pip install pytest` |
| Remote model backend | `powershell -NoProfile -ExecutionPolicy Bypass -File Tools\tests\Test-RemoteBackend.ps1` |

The Relay Station suite passes (119 tests, stdlib only). Test coverage for the remaining
components is listed in each component's README.

## Known gap

The Auto-Optimizer tab still assumes Docker (it benchmarks configs by spinning up temporary
containers) and currently fails with a clear message rather than running. It needs a native
rewrite that launches temporary llama-server.exe processes on an alternate port instead.
Manual tuning via the Models tab settings panel works normally in the meantime. The Model
Tuner covers the same job natively.
