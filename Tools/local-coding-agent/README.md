# Local coding-agent setup

This profile uses OpenCode on native Windows and the existing dashboard-managed
llama.cpp server. It does not start or switch llama-server. It does not use
Docker or WSL. The profile is installed inside a repository so it does not
change the user's global OpenCode configuration.

## Install for a repository

First load the intended model through the C:\AI dashboard and make sure it is
serving on port 10000. Then install the profile using the exact GGUF filename:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File C:\AI\Tools\local-coding-agent\Install-LocalCodeProfile.ps1 `
  -Repository C:\path\to\your\repo `
  -ModelFile 'Qwen3.5-9B-UD-Q6_K_XL-mtp.gguf'
```

The installer checks that one configured llama-server owns port 10000, that its
binary matches the model configuration, and that `/v1/models` returns one real
model ID. It refuses to switch models or invent an ID. If the repository already
has an OpenCode config, it refuses to overwrite it unless `-Replace` is passed;
replacement first creates a timestamped backup.

Start the session:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File C:\AI\Tools\local-coding-agent\Start-LocalCode.ps1 `
  -Repository C:\path\to\your\repo
```

The launcher rechecks AC power, available physical RAM, the configured model,
the live model ID, and the server process. It applies a 90-minute session limit
(configurable up to 240 minutes) and a watchdog that asks the dashboard to stop
llama-server after available physical RAM remains below 0.8 GiB for about six
seconds. If the dashboard is unavailable, it stops the OpenCode client and
reports that the server stop request failed; do not start another server until
the existing process is confirmed exited.

## Agent behavior

- The project profile permits only its `local-llama` provider. The default model
  is fixed to the live model ID captured at install time; no cloud fallback is
  configured.
- OpenCode uses its repository-aware read, glob, grep, and edit tools. The
  reusable reviewer subagent is read-only.
- The profile caps the advertised working context at 32K tokens, output at 4K,
  keeps recent context during automatic compaction, and enables snapshots.
- The build agent allows up to 24 steps per turn, then must summarize. The user
  can continue or resume a session after checking its status and the Git diff.
- File edits in the active repository are available to the agent. Shell
  commands require approval except read-only `git status`, `git diff`, `git log`,
  and `git show`. Push, reset, clean, and rebase commands are denied.
- Use OpenCode's session picker or `opencode --continue` to resume the last
  session in that repository. Re-read the current task and Git diff after a
  restart or compaction.

## Benchmarks

The isolated synthetic repository tasks and scoring instructions are in
[`bench/README.md`](bench/README.md). The grader is separate from each prepared
repo; its references and tests are not exposed to the agent. `selfcheck` confirms
that reference patches pass and deliberately wrong patches fail.

## Rollback

If this profile was installed without replacing another OpenCode config, remove
only the profile files created under that repository's `.opencode` directory.
If `-Replace` was used, restore the timestamped `.bak` config first. The
installer never changes llama.cpp models or the dashboard config. Benchmark
checkouts and logs remain isolated beneath `bench/runs`.

## Current limitations

The live server was stopped during this setup, and another local-LLM workstream
was active. Model loading, OpenCode tool execution, and live coding-agent scores
are therefore not yet verified. The dashboard stop route also needs a safety
fix before model switching: current code can proceed after its 30-second
wait even if a llama-server process has not exited. Keep model experiments
paused until that race is fixed and the other workstream releases the server.

The installed llama.cpp build advertises a one-hour HTTP read/write timeout.
The local launcher adds a hard session limit and output budget; a shorter
per-request server timeout still needs coordinated configuration in
`Config/models.json` after its concurrent edits are settled.

## Sources checked

- [OpenCode v2 local and compatible model configuration](https://opencode.ai/v2/docs/models)
- [OpenCode v2 permissions](https://opencode.ai/v2/docs/permissions)
- [OpenCode v2 compaction and configuration](https://opencode.ai/v2/docs/config)
- [OpenCode v2 agents and step limits](https://opencode.ai/v2/docs/agents)
- [llama.cpp function calling](https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md)
- [OpenAI Docs: Codex CLI](https://learn.chatgpt.com/docs/codex/cli)
