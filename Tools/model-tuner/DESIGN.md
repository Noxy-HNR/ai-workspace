# Model Tuner: design

Model Tuner finds the best llama-server settings for each GGUF model on this PC. It does the job the way a
careful person would: it reads the model file and the hardware first, predicts what fits before loading
anything, tests only the settings that matter for that model, pushes in whichever direction improves results,
and stops searching directions that do not. A faster setting never wins if it makes answers worse.

Read `C:\AI\AGENTS.md` first. Its RAM rules are hard constraints for every part of this project.

## Machine

- Windows 11 laptop, 15.4 GB RAM, RTX 5070 Ti Laptop GPU (12 GB), Intel Core Ultra 7 255HX (8 P-cores + 12
  E-cores, 20 threads, no hyper-threading).
- Power source and Windows power mode change speed a lot. Low GPU power while idle is intentional: never report it
  as a fault and never suggest power or MSI changes. Power state is recorded only to keep measurements comparable.
- Windows demotes an idle model's VRAM to system memory. The first request after a long idle can take 10-20 s just
  to page it back. Every measurement must be preceded by a warm-up request whose timing is discarded.

## Stack it drives

- llama-server on port 10000, managed by the dashboard on port 9090 (`C:\AI\dashboard-server.ps1`). The tuner never
  starts llama-server itself. It loads through `POST /api/models/load`, which stops the old server, waits for it to
  exit, then starts the new one.
- Trial loads pass `{"filename", "settings", "persist": false}` so nothing is written to `Config/models.json`.
- Per-model settings: contextSize, gpuLayers, flashAttn, parallel, jinja, batchSize, ubatchSize, threads,
  cacheTypeK, cacheTypeV, reasoningEffort, extraArgs, and optionally serverBinary, loraFile, loraScale.
- `llama-fit-params.exe` (in both `Tools/llama-native/bin` and `Tools/llama-prism`) projects device and host memory
  for a configuration in about 2 seconds without allocating anything. It is the tuner's primary fit oracle.

## Package layout (`C:\AI\Tools\model-tuner`)

Standard library only at runtime (Python 3.13). pytest for tests.

| Module | Job |
|---|---|
| `paths.py` | Every path, port and URL in one place. |
| `gguf.py` | Full GGUF reader: metadata (including arrays), tensor table, per-layer byte sizes. |
| `hardware.py` | RAM (GlobalMemoryStatusEx), GPU (nvidia-smi), CPU topology, power source and power mode. |
| `stats.py` | Robust statistics and the noise-aware "is B really better than A" decision. |
| `profile.py` | `ModelProfile`: what the tuner knows about a model (architecture, hybrid attention, MTP head, MoE, tool and reasoning support in the template, training context). |
| `knobs.py` | `Config`: typed settings. Round-trips dashboard settings, parses and rebuilds extraArgs. |
| `fitparams.py` | Runs llama-fit-params and parses its memory breakdown into a `Projection`. |
| `dashboard.py` | Dashboard client: trial load (verified), stop, status, props, config read/write. |
| `safety.py` | RAM preflight, RAM watchdog thread, trip flag, power-change detection, cooperative stop file. |
| `bench.py` | Speed workloads (prose, copy-heavy code edit, long prefill) and measurement from llama-server `timings`. |
| `backend.py` | `Backend` protocol and the shared result types. |
| `real.py` | The real backend: fit-params projections, verified trial loads, measurements with a RAM spill guard, crash logs, restore. |
| `sim.py` | Simulated backend with a plausible performance and memory model, for testing the strategy offline. |
| `strategy.py` | The brain: priors, phase order, pruning rules, direction search, context search, finalist selection. |
| `quality/` | Quality gates: reasoning set, agent tool tasks, coding tasks, long-context recall. |
| `history.py` | SQLite history: runs, trials, samples, calibrations, best configs. Priors for new models. |
| `advisor.py` | Optional Jev (TypeSafe) second opinion on failures the rules can't classify. Runs in Switchboard's environment so the key never passes through the tuner; sends numbers, setting names and filtered log lines only. |
| `switchboard_sync.py` | `apply --switchboard`: writes measured speeds into Switchboard's models.toml with line edits that keep its comments. |
| `report.py` | Markdown and JSON run reports. |
| `apply.py` | Applies an approved result to models.json (via the dashboard) and Switchboard's models.toml. |
| `runner.py` | Orchestrates a run: events log, state file, resume, cleanup and restore. |
| `cli.py` | `profile`, `plan`, `run`, `status`, `stop`, `apply`, `report`, `history`. |

## Core types (shared contract)

```python
@dataclass(frozen=True)
class Config:                       # knobs.py
    ctx: int                        # contextSize
    ngl: int                        # gpuLayers (999 = all)
    threads: int
    batch: int
    ubatch: int
    cache_k: str                    # f16 | q8_0 | q4_0 ...
    cache_v: str
    flash_attn: bool
    spec: str                       # "none" | "ngram-mod" | "draft-mtp" | "draft-mtp,ngram-mod"
    draft_n_max: int | None         # --spec-draft-n-max
    ngram_n_match: int | None       # --spec-ngram-mod-n-match
    reasoning_budget: int | None    # --reasoning-budget (None = unlimited)
    reasoning_effort: str           # "default" | low | medium | high | xhigh
    threads_batch: int | None       # -tb (None = same as threads)
    n_cpu_moe: int | None           # --n-cpu-moe (MoE only)
    n_cpu_ffn: int | None           # -ot: FFN weights of blocks 0..N-1 on the CPU, everything else on the GPU (dense only)
    passthrough: tuple[str, ...]    # other extraArgs kept verbatim (e.g. --load-mode none --cache-ram 0 are enforced)

@dataclass
class Projection:                   # fitparams.py
    device_model_mib: int; device_context_mib: int; device_compute_mib: int
    host_model_mib: int; host_context_mib: int; host_compute_mib: int
    device_total_mib: int; device_free_mib: int
    fitted_ngl: int | None; fitted_ctx: int | None
    @property device_self_mib -> int     # model + context + compute
    @property host_self_mib -> int

@dataclass
class Sample:                       # bench.py
    workload: str                   # "prose" | "code" | "prefill"
    gen_tps: float | None; prompt_tps: float | None
    draft_n: int | None; draft_accepted: int | None
    prompt_n: int; predicted_n: int; wall_s: float
    power: PowerState               # captured at measurement time
```

## How the tuner decides (strategy.py)

1. **Profile.** Read the GGUF: layer count, which layers use attention (hybrid qwen35 models use attention on only
   every Nth layer, so their context memory is tiny), KV heads and head sizes, sliding-window layers (gemma3), MoE
   experts, an MTP/NextN head, training context, the chat template (tool calls? does it honour reasoning effort?).
   Probe hardware and power.
2. **Start from what's known.** History for this exact file, then the best settings of other models with the same
   architecture, then the model's current models.json settings.
3. **Fit without loading.** Use fit-params projections to decide placement and context:
   - If the weights plus a useful context fit in VRAM (budget = total VRAM - VRAM used by other apps - margin), keep
     every layer on the GPU and binary-search the largest context that fits (projections cost ~2 s each). Cap it at
     the training context. Round down to a multiple of 4096 (8192 above 64K).
   - If not, choose the context target first (the current setting, or a floor), then let fit-params pick the most
     GPU layers that fit at that context.
   - Cache type: f16 when f16 already reaches the training context; otherwise q8_0. q4_0 only when the user allows it,
     and always quality-gated.
4. **Validate and calibrate.** One verified load. Compare actual VRAM use with the projection and store the error per
   architecture and binary. A failed load (out of memory, crash) means "does not fit": back off by the calibrated
   error and bisect between the last good and first bad values. Never retry the same failing config.
5. **Tune speed, in order of expected impact, skipping what cannot matter.**
   - Placement (dense models split across GPU and CPU only): every layer on the GPU with the FFN weights of the
     fewest blocks that fit moved to the CPU (`-ot`), challenged against the layer split. Both leave about the same
     weight bytes in system RAM, but the per-tensor split keeps attention and the whole KV cache on the GPU, which
     matters more as context grows. A rule written by hand in extraArgs is kept verbatim and never reinterpreted.
   - Speculative decoding: candidates depend on the model (draft-mtp only with an MTP head). Then its parameters by
     direction search (draft-n-max, n-gram match length).
   - Batch and ubatch: prompt speed versus compute-buffer VRAM. Never shrink the context target to win prompt speed.
   - Threads (and threads-batch, n-cpu-moe): only when layers, FFN weights or experts run on the CPU. With everything on the GPU,
     threads barely matter, so the tuner keeps the current value and says why.
6. **Search rules.**
   - Direction search on numeric knobs: step one way; keep going while it improves; once a direction is worse, never
     test further along it; stop when gains level off.
   - Noise: at least 3 repetitions (after a discarded warm-up), compare medians, and only accept a change larger than
     both ~3% and the run-to-run spread. Stop early when the leader is clearly ahead.
   - Speed objective: a weighted mix of decode speed on prose and on copy-heavy code (spec decoding helps a lot on
     the latter), plus prompt speed and load time as secondary terms.
   - Power: every sample records power source and mode. Samples from different power states are never compared. If
     power changes mid-run, the affected samples are discarded and re-measured; on battery the run refuses to start
     unless allowed.
7. **Quality stages** (the dashboard's modes):
   - quick: fit and speed only.
   - medium: quick + reasoning set + tool tasks on the top 2-3 configs, and the thinking cap or effort search.
   - full: medium + coding tasks + long-context recall near the chosen context.
   - Gate: a config whose quality drops beyond noise versus the baseline is rejected however fast it is.
8. **Finish.** Re-measure the winner against the baseline in the same power state, restore the user's original model
   and settings, write the report, and propose the change. Nothing is applied until the user approves it.

## Safety (non-negotiable)

- One llama-server at a time, only through the dashboard. Wait for exits; never stack servers.
- Before every load, available RAM (ullAvailPhys) must cover the projected host footprint plus 1 GB.
- Always keep `--load-mode none --cache-ram 0`.
- Every HTTP call has a hard timeout.
- Watchdog thread: if available RAM stays under 0.8 GB for 3 s, stop llama-server, write `state/WATCHDOG_TRIPPED`,
  abort. The tuner refuses to start while that flag exists.
- Verify every load: new PID, expected model file and n_ctx from /props.
- The user's models.json is never changed by a trial, and the original model is reloaded when the run ends,
  including on errors and Ctrl+C.
- Never close the user's other apps. If RAM is short, pause and say what is needed.

## State (`state/`, gitignored)

`tuner.db` (history), `runs/<run-id>/events.jsonl`, `runs/<run-id>/state.json`, `runs/<run-id>/report.md|json`,
`current.json` (the live run, read by the dashboard), `STOP` (cooperative stop request), `WATCHDOG_TRIPPED`.

## Lessons from real runs (2026-09-30)

- **Vary prompts per repetition.** llama-server's n-gram speculator keeps a cache across requests, so repeating one
  prompt let it replay the previous answer and inflated decode speed up to 5x. Repetition i uses prompt variant i,
  the same sequence for every config, and the warm-up prompt is unrelated to every workload.
- **llama-fit-params cannot see MTP.** The MTP draft context cost 700-1200 MiB on the 9B. The fit reserves the
  calibrated overhead up front for models with an MTP head, and a load that lands over budget trims the context.
- **VRAM overflow shows up as RAM exhaustion.** On WDDM a nearly full GPU spills into shared system memory; OxCoder at
  237K context drove available RAM to 0.13 GB and the watchdog tripped. The VRAM margin is now 1000 MiB and a spill
  guard stops a config at 1.4 GB available RAM, long before the 0.8 GB watchdog.
- **The dashboard overwrites llama-server's log on every start**, so the backend copies it into the run folder the
  moment something fails (`crash-N.log`).
- **Unset knobs mean the server's defaults** (draft-n-max 3, n-gram match 24); the direction search starts there.
