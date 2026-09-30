# Local coding-agent environment: status and findings

Updated 2026-09-28. This is a setup and benchmark-readiness report; no live agent score is claimed.

## Verified environment

- Windows-native OpenCode v2.0.18, Codex CLI 0.158.0-alpha.2.1, and Claude Code 2.1.281 are installed. The GPU is an RTX 5070 Ti Laptop with 12 GB VRAM; system RAM is 15.42 GiB.
- The dashboard is reachable on port 9090, but no `llama-server` was running on port 10000 during the last check. Available physical RAM was about 2.27 GiB. A separate local-model workstream was active, so no model was loaded or switched.
- `C:\AI\Config\models.json` and other root files had concurrent changes. This work did not edit those files.

## Selected workflow

Use a repository-scoped OpenCode v2 profile with the dashboard's existing server. The installer refuses to proceed until the selected model is already served and its live `/v1/models` ID matches. The profile limits provider selection to that local endpoint, caps context at 32K and output at 4K, keeps shell actions approval-gated, and denies destructive Git operations. `Start-LocalCode.ps1` checks AC, available physical RAM, process/listener count, and live model identity; it does not switch models.

The native Windows llama.cpp path is the first target because it is already installed and avoids adding another runtime. The profile and launcher are not live-verified yet. In particular, the dashboard stop path needs a confirmed-process-exit fix before model switching; the existing path may proceed after its wait expires. Keep `--load-mode none --cache-ram 0` on model configurations. Current model configs and live request timeout behavior need a coordinated review before server experiments.

## Candidate models

The configured roster includes Qwen3.8-27B IQ3_XXS and IQ2_S MTP, Qwen3.5-9B Q6_K_XL-MTP, OxCoder-9B i1-Q6_K, Ternary-Bonsai-2-27B PTQ1_0, and MedGemma-27B Q4_K_M. For coding trials, prioritize an already-supported 9B candidate that fits GPU memory, then compare the two Qwen3.8 quantizations only if measured RAM headroom allows. OxCoder is a coding-focused hypothesis worth testing. Bonsai requires its Prism-specific backend; it should not be treated as a stock llama.cpp candidate. MedGemma is not a coding-first choice and its current config lacks the standard RAM-safe load flags. Model-card benchmark claims are not local evidence.

Historical local scratch evaluation found Huihui at 42/48 tasks and Ornith at 35/48; those runs do not use this new harness and are not current head-to-head results. A prior run also documented a RAM crash involving mmap/cache and overlapping server startup, which is why current server comparison remains paused.

## Benchmark readiness

Two synthetic, dependency-free Python repo tasks are prepared: retry scheduling and timezone-aware JSONL log merging. `bench.py selfcheck` passes for both reference implementations and rejects both deliberately wrong implementations. Fresh starter copies correctly fail the hidden grader, as expected before implementation. The first self-check found and fixed a malformed UTC timestamp formatter in the logmerge reference.

The fixtures support scoring local OpenCode, Claude Code, and Codex on equivalent clean checkouts, but no agent runs have been made. The comparison is authorized only against these synthetic fixtures. Record task pass/fail, elapsed time, final diff, test output, CLI/model/backend metadata, and—on local runs—llama.cpp prompt/decode timings and RAM/VRAM headroom. Treat this as a whole-agent-stack comparison, since the CLI harnesses differ.

## Current blockers and next safe step

Live model trials wait until the other local-model workstream releases the server and the server stop path has a confirmed-exit safeguard. Before any load, check AC and `GlobalMemoryStatusEx` available physical RAM against the chosen model's host-RAM footprint plus about 1 GiB. Run one server only, retain a RAM watchdog, and use hard request/session limits. Then validate tool calling and one smoke task before beginning repeated trials.
