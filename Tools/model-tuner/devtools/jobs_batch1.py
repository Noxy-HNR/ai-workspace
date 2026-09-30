"""Creates the first batch of bot-queue jobs (state/botqueue/jobs/<name>/). Idempotent: existing jobs are left alone.

Everything the bots produce stays in their job folders until it is reviewed and merged by hand.
Run: python devtools/jobs_batch1.py, then start the queue: python devtools/botqueue.py run
"""

import json
import pathlib
import shutil

PROJECT = pathlib.Path(__file__).resolve().parent.parent
JOBS = PROJECT / "state" / "botqueue" / "jobs"
AI = pathlib.Path(r"C:\AI")

RULES = r"""
## Ground rules
- Work ONLY inside the current directory. Never modify anything under C:\AI or elsewhere on this machine.
- Never load, stop or start models, llama-server or the dashboard, and never call their HTTP APIs. Tests must use fakes.
- Python 3.13, standard library only (pytest for tests). Match the existing modules' style.
- Run tests with `uv run --with pytest pytest -q`. The existing tests must keep passing.
- Finish with a short summary of what you did and anything you were unsure about.
"""

REVIEW = """
## What to deliver
Write `REVIEW.md`: findings ranked most severe first. For each one: file and line, what goes wrong, a concrete
scenario (inputs or state -> wrong behaviour), and a suggested fix. Only real defects or real risks; no style notes.
Where a finding can be shown by a test, add that test under `proposed_tests/` (not under `tests/`), and say in
REVIEW.md whether it currently fails. Do not change the modules themselves.
"""

CHECK_REVIEW = 'python -c "import pathlib,sys; p=pathlib.Path(\'REVIEW.md\'); sys.exit(0 if p.exists() and p.stat().st_size>400 else 1)"'


def job(name, models, task, *, check=None, timeout_s=3600, extra_files=(), local_model=None, local_settings=None):
    d = JOBS / name
    if (d / "job.json").exists():
        print(f"exists: {name}")
        return
    d.mkdir(parents=True, exist_ok=True)
    spec = {"models": models, "after": [], "timeout_s": timeout_s}
    if check:
        spec["check"] = check
    if local_model:
        spec["local_model"] = local_model
        spec["local_settings"] = local_settings or {}
    (d / "job.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    (d / "TASK.md").write_text(task.strip() + "\n" + RULES, encoding="utf-8")
    for f in extra_files:
        shutil.copy2(f, d / pathlib.Path(f).name)
    print(f"created: {name}")


job("01-review-strategy", ["opencode/nemotron-3-ultra-free", "opencode/space-bunny-free",
                           "opencode/muse-spark-1.3-contributor-free"], """
# Task: review `modeltuner/strategy.py` for logic bugs

Read DESIGN.md, then `strategy.py` closely, with `backend.py`, `knobs.py`, `stats.py` and `sim.py` for context. Look
for decisions that could be wrong on real hardware: search loops that can stop early or never stop, comparisons made
across different sample sets, configs that are never measured but get chosen, VRAM checks that can be bypassed, the
load budget, stop/power handling, the history integration (`_hist*`, priors, known failures) and the quality gate.
The simulator (`modeltuner/sim.py`) can reproduce many scenarios in a test.
""" + REVIEW, check=CHECK_REVIEW)

job("02-review-real-backend", ["opencode/space-bunny-free", "opencode/nemotron-3-ultra-free",
                               "opencode/longcat-2.5-preview-free"], """
# Task: review the real backend for robustness bugs

Review `modeltuner/real.py`, `dashboard.py`, `safety.py`, `bench.py` and `runner.py`: the code that drives a real
llama-server on Windows through a dashboard. Look for: hangs without a timeout, races between the watchdog or spill
guard threads and the main thread, a crashed server mistaken for a slow one (or the reverse), restore paths that can
leave the user's model unloaded or models.json changed, log handling that can lose the crash log, measurements that
could compare different prompts, and anything that could load a second llama-server.
""" + REVIEW, check=CHECK_REVIEW)

job("03-tests-real-backend", ["opencode/muse-spark-1.3-contributor-free", "opencode/space-bunny-free",
                              "opencode/nemotron-3-ultra-free"], """
# Task: integration tests for `RealBackend` against a fake dashboard and a fake llama-server

Write `tests/test_real_backend.py`. Start two small `http.server` threads on port 0: a fake dashboard
(`/api/models/load` with persist:false returns {"success": true, "trial": true}; `/api/stop`) and a fake llama-server
(`/props`, `/v1/models`, `/v1/chat/completions` returning realistic `timings`). Build `RealBackend` with
`dash=Dashboard(base_url=..., llama=LlamaClient(base_url=...))`, and monkeypatch what touches the real machine
(`modeltuner.real.llama_pid`, `process_alive`, `hardware.gpu`, `hardware.memory`, `hardware.power`, `FitParams.project`,
`llama_log_tail`). Cover: a successful trial load and warm-up, a load that never comes up (timeout -> LoadFailed),
the server dying mid-measurement (LoadFailed with the crash log saved to run_dir), the spill guard firing when
available RAM drops, the prompt-variant sequence continuing across calls for the same config and restarting for a new
config, and restore() using a trial load (not a saved load) when the original model was not the active one.
""", timeout_s=4500)

job("04-tests-cli-report", ["opencode/longcat-2.5-preview-free", "opencode/mimo-v2.6-flash-free",
                            "opencode/space-bunny-free"], """
# Task: tests for `modeltuner/report.py` and the non-hardware CLI commands

Write `tests/test_report.py`: run the strategy against `sim.SimBackend` (see `tests/test_strategy_sim.py` for how) to get a
real `TuneResult`, then check `report.to_json` (keys, proposed settings keep serverBinary/loraFile from the profile's
current settings, changes list only differing keys) and `report.to_markdown` (result table, proposed-changes table,
every decision listed, the trials table). Write `tests/test_cli.py` for `status`, `stop`, `runs`, `report` and `apply`
(with `MODELTUNER_STATE_DIR` pointed at a temp dir before importing modeltuner.paths, a fake report.json, and
`Dashboard.save_settings` monkeypatched): apply without --yes asks and does nothing on "n", apply --yes calls
save_settings with the proposed settings and writes applied.json, apply --switchboard edits a temp models.toml.
""", timeout_s=4500)

job("05-local-tests-knobs", ["llama-local/qwen35-active", "opencode/space-bunny-free"], """
# Task: thorough tests for `modeltuner/knobs.py`

Write `tests/test_knobs.py`. Cover: `from_settings` on each real model's settings (copy them from the table below),
`to_settings(from_settings(x), x)` preserving every key the tuner does not own (serverBinary, loraFile, ropeFreqBase...),
`parse_extra_args` for every typed flag and its aliases (`--spec-draft-n-max` / `--draft-max`, `-tb` /
`--threads-batch`, `-ncmoe` / `--n-cpu-moe`), `--flag=value` form, `--reasoning-budget -1` meaning unlimited, unknown
flags kept verbatim and in order, the enforced `--load-mode none --cache-ram 0` appearing exactly once after a round
trip even if the input had them twice or not at all, owned flags such as `-c` in extraArgs raising ValueError, a flag
with a missing or non-numeric value raising ValueError, `Config.key()` stability, `describe()` wording, `diff()`.

Real settings (models.json):
- 9B: contextSize 139264, gpuLayers 999, batchSize 1024, ubatchSize 512, threads 8, cacheTypeK/V q8_0, reasoningEffort
  default, extraArgs "--spec-type draft-mtp,ngram-mod --spec-draft-n-max 2 --load-mode none --cache-ram 0 --reasoning-budget 4096"
- Bonsai: contextSize 147456, extraArgs "--spec-type ngram-mod --spec-ngram-mod-n-match 8 --load-mode none --cache-ram 0",
  serverBinary "Tools/llama-prism/llama-server.exe", reasoningEffort medium
- MedGemma: contextSize 40960, gpuLayers 37, threads 20, extraArgs "--spec-type ngram-mod --spec-ngram-mod-n-match 8"
""", timeout_s=5400)

job("06-review-dashboard-tuner", ["opencode/nemotron-3-ultra-free", "opencode/space-bunny-free"], """
# Task: security and robustness review of the dashboard's Model Tuner integration

`dashboard-server.ps1` and `dashboard.html` in this folder are copies of the local dashboard (a PowerShell
HttpListener on 127.0.0.1:9090 and its single-page UI). Review only the Model Tuner parts: `Invoke-TunerAction`,
`Get-TunerStatus`, `Start-TunerProcess`, `Test-TunerRunning`, `Start-ModelTrial`, the `/api/optimizer` routes and the
`/api/models/load` persist:false branch, and the JavaScript `tuner*` functions. Look for: argument injection into
`Start-Process` or `& $TunerPython`, path traversal through `model` or `run_id`, overrides in `Start-ModelTrial` that
could drop the RAM-safety flags or load something unexpected, XSS in the tab (every value rendered from status JSON),
races between two start requests, and states where the UI shows the wrong thing.
""" + REVIEW, check=CHECK_REVIEW, extra_files=[AI / "dashboard-server.ps1", AI / "dashboard.html"])
