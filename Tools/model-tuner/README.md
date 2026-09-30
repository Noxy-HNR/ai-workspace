# Model Tuner

Finds the fastest safe llama-server settings for each model in `C:\AI\models`, the way a careful person would:
read the model file, plan placement and context on paper, test only what matters for that model, push in whichever
direction wins, and never trade answer quality for speed. Design and rationale: [DESIGN.md](DESIGN.md).

## Use it

From the dashboard: open the **Model Tuner** tab, pick a model and a depth, press **Plan on paper** (loads nothing)
or **Start tuning**. The tab shows every decision live. When the run ends your model is reloaded with its saved
settings, and nothing is written to `models.json` until you press **Apply**.

From a terminal (in this folder):

```bash
python -m modeltuner profile                       # what the tuner reads from every model file
python -m modeltuner plan Qwen3.5-9B-UD-Q6_K_XL-mtp.gguf   # placement, context, cache type; nothing loaded
python -m modeltuner run OxCoder-9B.i1-Q6_K.gguf --mode quick
python -m modeltuner status
python -m modeltuner stop                          # finishes the current step, restores your model
python -m modeltuner report                        # latest report
python -m modeltuner apply <run-id>                # asks before writing models.json
python -m modeltuner apply <run-id> --switchboard  # ...and writes the measured speeds to Switchboard's models.toml
python -m modeltuner runs                          # past runs
```

`C:\AI\model-optimize.ps1 -Model <file> [-Mode quick|medium|full] [-Plan] [-Detach]` does the same from PowerShell.

Modes:

| Mode | What it does | Typical time |
|---|---|---|
| quick | fit, placement, speculative decoding, batch sizes, threads (only when layers run on the CPU) | 15-25 min |
| medium | quick + thinking cap or reasoning effort + reasoning and tool-calling quality gates | 45-90 min |
| full | medium + coding and long-context recall gates at the chosen context | 2-4 h |

## How it decides

1. **Profile** (no loading): architecture, hybrid attention layers, sliding window, MoE, MTP head, training context,
   whether the chat template supports tools and honours reasoning effort.
2. **Fit on paper** with `llama-fit-params` (about 2 s per projection, nothing allocated): all layers on the GPU if
   possible, then the largest context by binary search, f16 cache only when it already reaches the training context.
   Models with an MTP head get its draft memory reserved up front. Split models pick context first, then the most
   GPU layers.
3. **Validate** with one real load and calibrate: the gap between projection and reality is measured (MTP's draft
   context is invisible to `llama-fit-params`, about 700 MiB on the 9B) and the context is trimmed if the load went
   over budget.
4. **Baseline**: your current settings, measured in the same power state.
5. **Tune what matters**, noise-aware (3 repetitions, medians, a bar of max(3%, 2x the spread)):
   speculative decoding (MTP only with an MTP head), draft length and n-gram match length by direction search,
   ubatch for prompt speed (never at the expense of context), threads only when layers run on the CPU.
6. **Quality gates** (medium/full): a faster config that answers worse is rejected.
7. **Confirm** the winner on a fresh load, restore your model, write the report.

Measurement details that matter on this laptop:
- Every repetition uses a different prompt (same sequence for every config): llama-server's n-gram speculator
  remembers earlier answers across requests, and identical prompts inflated speeds up to 5x.
- A warm-up request runs after every load: Windows pages an idle model's VRAM out.
- Samples record the power source and Windows power mode; samples from different power states are never compared.

## Memory and second opinions

- **History** (`state/tuner.db`, SQLite): every trial, sample, quality score and calibration is kept. A later run
  never reloads a config that ran out of memory before, starts its fit from the calibrated projection error and MTP
  overhead of the model's family, gives the previous run's winner a head-to-head, and searches each knob toward the
  value that won for a sibling model first.
- **Jev advisor** (optional): when llama-server dies and the log matches no known pattern, the tuner asks Jev (through
  Switchboard's environment, so the key never passes through the tuner) what most likely happened, and reacts to the
  answer (back off memory, avoid the setting, or just mark it failed). Only numbers, setting names and filtered log
  lines are sent. `--no-advisor` turns it off.

## Safety

The rules in `C:\AI\AGENTS.md` are built in: loads only through the dashboard (trial loads use
`persist: false`, so `models.json` is untouched), a RAM preflight before every load, a RAM watchdog that stops
llama-server and refuses further runs (`state/WATCHDOG_TRIPPED`), verified loads (new PID, model file, n_ctx),
hard timeouts on every request, and `--load-mode none --cache-ram 0` always. A spill guard stops any config whose
measurement drives available RAM under 1.4 GB (VRAM overflowing into shared system memory), well before the watchdog.

## Who built what

Written with help from other models through OpenCode, each piece reviewed and integrated here:
GGUF reader and reasoning suite (Nemotron 3 Ultra), hardware probes, tool-calling suite, coding suite and history
(Space Bunny), simulator (Muse Spark 1.3), recall suite (LongCat 2.5), statistics (OxCoder-9B, running locally).

## Layout

`modeltuner/` package (standard library only), `tests/` (pytest; `uv run --with pytest pytest -q`),
`state/` (runs, reports, flags; not in git), `devtools/` (build tooling), `_legacy/` (the hand-run harnesses the
quality suites were ported from; not in git).
