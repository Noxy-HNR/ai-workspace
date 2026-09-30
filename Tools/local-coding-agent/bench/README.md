# Local coding-agent comparison fixtures

Two dependency-free, synthetic Python repositories test multi-file changes. No
private repository content, network, credentials, model, or server is needed to
prepare and score them. Python 3.11+ is sufficient.

## Commands

From `C:\AI\Tools\local-coding-agent\bench`:

```powershell
python -B .\bench.py selfcheck
python -B .\bench.py prepare queue .\runs\queue-agent-a
python -B .\bench.py grade queue .\runs\queue-agent-a
python -B .\bench.py prepare logmerge .\runs\logmerge-agent-a
python -B .\bench.py grade logmerge .\runs\logmerge-agent-a
```

`prepare` refuses an existing output path. Give every agent and repeat a fresh
repository. Read the prepared repo's `TASK.md` and pass its text as the agent
prompt. The starter repo includes visible tests; the scorer holds additional
authoritative tests outside the prepared repo. `selfcheck` builds the reference
and a deliberately wrong variant of each task, then verifies that the scorer
passes the reference and rejects the wrong variant.

## Comparison protocol

1. Pin CLI, model, runtime, quantization, context, sampling settings, AC state,
   power mode, and fixture revision. Run only when the local server is available
   for this work; the dashboard is the only model switcher. Check available
   physical RAM before a model load, keep `--load-mode none --cache-ram 0`, and
   run one server with a RAM watchdog and hard request/agent timeouts.
2. Give each candidate a clean prepared repo and exactly the same `TASK.md`.
   Allow repository read/search, edits, local tests, and read-only Git inspection.
   Turn off web, extra MCP servers, subagents, and provider-specific helpers.
   Never allow the agent to inspect `bench/grader`, `bench/reference`, or
   `bench/negative`. Keep one attempt per task and an equal wall-time budget.
3. Save each agent's trace, final diff, exit status, wall time, test output, and
   model/backend metadata. Grade only the final checkout with `bench.py grade`.
   Do not let an agent edit its own grading tests. Re-run from a fresh repo for
   additional samples rather than continuing a previous attempt.
4. Report pass/fail per task and failure type (timeout, tool error, crash,
   test failure, or no patch). Report time to a passing patch and total wall
   time. For local inference separately report llama.cpp `timings` prompt and
   predicted tokens/second, first-token latency, peak resident RAM, minimum
   available physical RAM, and post-generation free VRAM. Do not call total
   completion tokens divided by whole request time the decode rate.

The scoring target is **both tasks passing**. Task-level scores are binary;
test counts are diagnostic, not percentages to compare across tasks. The
reference/wrong self-check catches inverted or vacuous graders. A reasonable
pilot is three independent attempts per task and candidate; show all six rows,
median wall time, and pass count with no implied population claim.

For the agent-product comparison, use the installed OpenCode, Claude Code, and
Codex CLIs in their noninteractive JSON event modes, with the same prepared
repos and as close to the tool list above as each CLI permits. Give the local
model its own OpenCode run against the dashboard-managed llama-server. CLI
harnesses and built-in prompts still differ, so this measures the **whole agent
stack**. A separate direct-API coding suite is needed to isolate model ability.
Cloud comparison is limited to these synthetic fixtures and requires the user's
authorization for authenticated service use. No claim about cloud performance
should be made from an unrun or non-comparable row.

## Rollback

All benchmark outputs are isolated under a path selected by `prepare`; remove
an unwanted prepared repo only after checking that it contains no work to keep.
The benchmark itself changes no model or agent configuration.
