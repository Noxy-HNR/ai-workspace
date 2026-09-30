# Review: the real backend (`real.py`, `dashboard.py`, `safety.py`, `bench.py`, `runner.py`)

Findings, most severe first. Every one is a defect or a concrete risk, not a style note. Each has a test under
`proposed_tests/`; the status column says whether it fails against the code as it stands (23 of the 29 proposed
tests fail, and every failure below is a defect being demonstrated, not a broken test).

Run them with `uv run --with pytest pytest -q proposed_tests`. No test loads a model, starts a process, opens a
socket to llama-server or the dashboard, or reads anything under `C:\AI`: the machine is faked throughout
(`proposed_tests/fakes.py`).

---

## 1. The RAM watchdog kills the process from its own thread, so nothing that has to happen afterwards happens

**`safety.py:83-93`** (`Watchdog._trip`), used by **`runner.py:84-91`** and **`runner.py:109-128`**

`_trip` ends with `os._exit(3)` in a `finally`, on the watchdog thread. `os._exit` cannot be caught or deferred,
so it skips every `finally` in `runner.run`: `backend.restore()` (line 111), `dog.stop()` (112),
`report.write()` (115) and `hist.finish_run()`/`hist.set_best()` (118-125) are all skipped.

What goes wrong:

* The user's model is not reloaded. DESIGN.md lists "the original model is reloaded when the run ends, including
  on errors and Ctrl+C" as a non-negotiable, and the run is unattended, so the user comes back to a stopped
  server with no explanation anywhere except the trip flag.
* The run's report is never written and its history row stays `running` for ever, so the trial history and the
  calibration the next run depends on are silently incomplete.
* `on_trip` runs *before* the exit and calls `backend.dash.stop_llama()`, which has a 120 s HTTP timeout
  (`dashboard.py:67`). If the dashboard is slow, llama-server keeps running for up to two minutes after the
  watchdog has decided RAM is critically low.
* Nothing anywhere records the trip in the run's own event log. `on_trip` only calls `st.save(...)`, and the
  run then continues to overwrite that status.

Scenario: available RAM sits at 0.6 GB for 3 s during `phase_batch`. The watchdog writes `WATCHDOG_TRIPPED`,
asks the dashboard to stop llama-server, and exits with code 3. The process is gone before the report and the
restore. The next `modeltuner run` refuses to start ("refusing to run: ... WATCHDOG_TRIPPED exists"), and if the
user deletes the flag as instructed, `tuner.db` still has a run that never finished and no record of the
settings that were loaded when the machine ran out of memory.

The trip is also invisible in the other direction: made survivable (which is what a fix would do), the run
carries straight on and reports `status: "ok"` and "done: +12.3% overall" after the watchdog stopped
llama-server underneath it.

Fix: the watchdog should signal, not kill. Give it a `threading.Event` (or a `Tripped` exception injected into
the main loop) that `Watchdog._loop` sets, and let `runner.run` notice it: stop the strategy, restore, write the
report with `status="watchdog"`, finish the history row, then exit non-zero from the main thread. Keep
`os._exit` only as a last resort after the main thread has had its chance (or drop it and let the CLI return 3).
At minimum, wrap the trip in an `emit("watchdog", ...)` so the event log and the report carry it, and put the
report/history writes before `os._exit`.

Tests: `proposed_tests/test_runner_watchdog.py::test_a_watchdog_trip_is_not_invisible_to_the_run` (**fails**).
`::test_the_trip_flag_is_written_before_the_process_dies` (**passes**) pins the part that does work.

---

## 2. A restore that raises eats the report, the history and the run's real error

**`real.py:65`** and **`real.py:77`**, in **`runner.py:110-111`**

`restore()` is best-effort by design: it catches `DashboardError` and logs "restore problem". But
`self.dash.models_config()` (`dashboard.py:69-70`) reads `C:\AI\Config\models.json` straight off disk, outside
the `_call` wrapper, so it can raise `OSError` or `json.JSONDecodeError` - neither is a `DashboardError`, both
escape `restore()`, and both escape `runner.run`'s `finally` at line 111.

What goes wrong: the exception from the `finally` replaces whatever the run was doing. `dog.stop()`, the report
and the history finish are skipped, and the traceback the user gets is about a JSON file they never mentioned
rather than about their run. A run that had already finished tuning now reports a crash.

Scenario: the dashboard is rewriting `models.json` (an `apply`, or the dashboard's own UI) while the tuner
reloads the user's model. The tuner reads a half-written file. The run that measured 20 configs throws away its
report and blames models.json.

Fix: catch `Exception` in `restore()` (not just `DashboardError`) and always report; and in `runner.run`, put
the restore in its own `try/except` so the report, the history finish and `dog.stop()` run whatever the restore
did. Also write `models.json` reads through a retry or read it from the dashboard (`/api/config`) rather than
from disk.

Tests: `proposed_tests/test_runner_watchdog.py::test_a_failing_restore_still_writes_the_report` (**fails**);
`proposed_tests/test_real_restore.py::test_restore_survives_an_unreadable_models_json` (**fails**).
`::test_restore_survives_a_dashboard_that_refuses` (**passes**) shows the `DashboardError` half already works.

---

## 3. A `/props` hiccup makes the run forget the user's model, and the run then stops it

**`real.py:47-48`** with **`real.py:59-78`** and **`llama.py:59-63`**

`begin()` records what the user was running with `self.client.loaded_model()`. That helper swallows every error
(`except (LlamaError, KeyError, TypeError): return None`), and `props()` has a 10 s timeout. A busy or briefly
unreachable server - the user is mid-conversation, or the machine is paging - and `_original_running` becomes
`False`. `restore()` then takes the `else` branch at line 75 and calls `dash.stop_llama()`.

What goes wrong: at the end of a completely successful run the user finds their model stopped and unloaded. The
run's own log says "done: +12.3% overall" with nothing about it.

Scenario: the user asks llama-server a long question at 09:00 while the tuner starts. `/props` does not answer
within 10 s. The tuner believes llama-server was not running, stops it at the end of the run, and the user's
in-flight session dies. `llama_pid()` (already imported at `real.py:11`) would have answered correctly.

Fix: treat "no model" as a fact only when it is corroborated. If `llama_pid()` returns a PID, something is
listening: remember that something was running and restore it, even if the model name could not be read. At
minimum, retry `loaded_model()` a few times over ~30 s before concluding that nothing was loaded, and treat
"could not read" as an error rather than as "no".

Tests: `proposed_tests/test_real_restore.py::test_a_props_hiccup_does_not_leave_the_users_model_unloaded`
(**fails**), `::test_begin_uses_the_listening_port_when_props_fails` (**fails**).

---

## 4. `begin()` gives up waiting for the old server to exit and then measures VRAM through it

**`real.py:49-57`**

```python
old = llama_pid()
self.dash.stop_llama()
t0 = time.time()
while old and process_alive(old) and time.time() - t0 < 120:
    time.sleep(1)       # never start anything while the old server is still exiting
time.sleep(2)
gpu = hardware.gpu()
self._other_mib = gpu.used_mib if gpu else None
```

What goes wrong: when the 120 s expires the loop simply ends and the run continues. `self._other_mib` is then
read while the user's model is still resident, so the "VRAM used by other apps" figure in the log and the report
is wrong, and `vram_budget_mib()` (line 108-112) - which decides the context and the layer split for every load
in the run - is computed from it. The log line at line 57 then states a number that is not what it says it is.
AGENTS.md is explicit: "If a llama-server won't exit, wait. Starting another one while it's stuck is what caused
the crash." Nothing here waits; the code only waits for 120 s and then proceeds.

The tuner does not launch llama-server itself, so the second server is the dashboard's to start, and the
dashboard is documented to wait for the old one. That is the mitigation, not a defence: the tuner is the component
that decided the old server was gone, told nobody, and moved on with a VRAM budget derived from a false reading.

Scenario: the user closes a 30B model that is stuck in kernel I/O. `begin()` waits 120 s, gives up, records
`_other_mib` with the old model's 9 GB still on the card, and then loads a 9B with a budget shrunk by 9 GB. The
fitted context comes out far too small and the run's conclusions are about a configuration the tuner never had
any reason to choose.

Fix: if `old` is still alive when the wait expires, raise (`safety.SafetyError`, or `LoadFailed("timeout")`) and
let the run end with the user's model untouched. Wait without a ceiling, or with a ceiling that reports, and
never take the VRAM reading until the old process is gone.

Tests: `proposed_tests/test_real_restore.py::test_begin_refuses_to_continue_while_the_old_server_is_alive`
(**fails**).

---

## 5. A load that is merely slow is failed using the previous server's log, and the verdict is permanent

**`dashboard.py:150-154`**

```python
elif seen_pid is None and time.time() - t0 > 20 and pid is None:
    tail = llama_log_tail()
    if any(m.lower() in tail.lower() for m in OOM_MARKERS) or "error" in tail.lower().split("\n")[-1]:
        raise DashboardError(f"{classify_failure(tail)}: llama-server failed to start")
```

The dashboard overwrites `llama.stderr.log` on every llama-server start (DESIGN.md, "Lessons from real runs"), so
for the whole window in which the new server has not yet bound the port, the log on disk is the *previous*
server's. That window is exactly the slow-load window: a 30B model takes 40-90 s to allocate, and this branch
starts looking at the log at 20 s.

What goes wrong: two markers are read out of a log that has nothing to do with the load being verified:
`OOM_MARKERS` is searched across the whole tail, and `"error"` in the last line catches llama-server's own
shutdown lines, which contain "error" on many builds. A good load of a big model is reported as `oom` or
`crash`. `real.load` (line 126-135) turns that into `LoadFailed("oom"/"crash")`, saves a crash log full of the
*previous* run's output, and the strategy records it: `strategy.ensure_loaded` (line 199-203) then treats
`oom`/`spill`/`crash`/`refused` as a `known_failure` for this file and **never loads that config again**, in this
run or any later one. One slow load on a busy machine permanently removes a good configuration from the search
space, and the history that does it survives the run.

Scenario: the previous trial was a 237K-context OxCoder that OOMed. The next trial is a 16K-context 9B that
needs 40 s to load. At t=21 s nothing is listening, the log still holds `cudaMalloc failed`, and the tuner
records a 16K 9B as an OOM. The run carries on with a config it has already decided never to try again.

Fix: only read the log once the new server has demonstrably started (a new PID has been seen at least once) or
once its log has been rotated. Better: capture the log file's mtime/size at the moment of the load call and
ignore it until it changes, or have `load()` copy the log before it calls `trial_load` and diff afterwards.

Tests: `proposed_tests/test_dashboard_wait.py::test_a_slow_load_is_not_called_a_crash_because_of_the_previous_servers_log`
(**fails**), `::test_a_stale_error_line_does_not_fail_a_load_that_is_still_starting` (**fails**).

---

## 6. `_SpillGuard.stop()` neither disarms the guard nor waits for it, so it can stop a server it should not

**`real.py:196-235`**, driven from **`real.py:151`** and **`real.py:172-173`**

The loop checks the stop event once, at the top (line 221), and then does three blocking things: read RAM
(`hardware.memory()`), decide, and `stop_llama()` (an HTTP call with a 120 s timeout). `stop()` (215-217) sets
the event and joins for 2 s.

What goes wrong, two ways:

* The guard is inside `hardware.memory()` when the measurement ends. It never sees the event, the low condition
  still holds, and it stops llama-server - which by then is the *next* config's server, or the model the user's
  restore has just reloaded. The samples it just returned were accepted as a success, and the next measurement
  fails against a server that was killed for a reason that has nothing to do with it.
* The guard had already fired and its `stop_llama()` is still in flight (the dashboard waits for the server to
  exit, so this takes seconds). `stop()` returns after 2 s with the call still running, and it lands on whatever
  server is up by then. The main thread cannot see it: `guard.fired` is checked at line 174 *after* `stop()`
  returns, and a trip that arrives just after that check is lost entirely.

The second point also breaks the guard's own contract in `measure`: "if the guard fired, this config spilled" is
only true for a window shorter than the join.

Scenario: VRAM is spilling, available RAM crosses 1.4 GB during the last repetition of `prose`, the guard fires
and calls the dashboard. The last `chat` happens to complete before the server stops, so `bench.measure`
returns, `guard.stop()` returns after 2 s, and `measure()` hands back three good samples. A moment later the
in-flight `stop_llama` lands and the freshly loaded next config dies; the tuner records a crash for a config
that was never at fault, and `_save_crash_log` copies a log with nothing useful in it. If the last action of the
run is the restore, it is the user's model that gets stopped.

Fix: make `stop()` authoritative. Set a `disarmed` flag that `_loop` re-checks immediately before acting (after
every blocking call), and have `stop()` join without a short timeout, or join with a timeout longer than the
dashboard's own (120 s), so the main thread cannot proceed while a stop is in flight. Have `stop()` return whether
the guard fired so `measure()` can decide what to do with it rather than sampling `fired` once.

Tests: `proposed_tests/test_real_spill_guard.py::test_stop_disarms_the_guard` (**fails**),
`::test_stop_waits_for_a_stop_that_is_already_under_way` (**fails** - `stop()` returns after 2.0 s with the call
still running).

---

## 7. A server that has exited is reported as a slow one, and the log that explains why is thrown away

**`dashboard.py:94-100`**, **`dashboard.py:147-155`**, **`real.py:129-135`**

`process_alive` answers `True` whenever `tasklist` fails (`except (OSError, subprocess.TimeoutExpired): return True`,
line 98-99). That is the right answer for a waiting loop, but it is the *only* thing that distinguishes
"crashed" from "still loading" once a PID has been seen: line 147 needs `pid is None and not process_alive(seen_pid)`,
so a dead server whose PID cannot be queried is never noticed. The loop then runs to the 360 s ceiling and
raises `DashboardError("timeout: ...")`.

What goes wrong, twice over:

* The failure is misclassified. `real.load` line 129 keeps "timeout" as a kind, so nothing tries to read the log.
  A server that died of an OOM at t=30 s is reported as a timeout, six minutes later, with no reason.
* The log is lost. `_save_crash_log` is only called `if kind in ("oom", "crash")` (line 132). The `timeout` and
  `verify` paths skip it, and the next `trial_load` makes the dashboard truncate `llama.stderr.log`. The one
  copy of "why" is gone. That is the exact failure mode DESIGN.md says the `crash-N.log` copy exists to prevent,
  on the two paths that are most likely to be how a real failure shows up on a machine that is short of memory.

Scenario: available RAM is low, the dashboard starts the server, the server's first allocation fails. Nothing is
listening, `tasklist` is slow or blocked, `process_alive` says yes, and 360 s later the tuner records "timeout"
and moves on to the next config - which overwrites the log that said `failed to allocate 8000 MiB`.

Fix: at the 360 s ceiling, read the log and classify from it (the log is the best evidence available at that
point). Call `_save_crash_log()` for `timeout` and `verify` too, not only for `oom`/`crash`: a timeout is exactly
when the log is about to be lost. Independently, make the "did it exit?" test not depend solely on `tasklist` -
the port no longer being held plus a stale log is enough to call it a crash.

Tests: `proposed_tests/test_dashboard_wait.py::test_a_server_that_exited_is_reported_from_its_log_not_as_a_timeout`
(**fails**), `::test_process_alive_says_yes_when_tasklist_fails` (**passes** - the trade-off, documented);
`proposed_tests/test_real_measure.py::test_a_load_that_times_out_still_saves_the_log` (**fails**);
`::test_a_server_that_died_mid_measurement_keeps_its_log` (**passes**) is the control for the path that does work.

---

## 8. A slow dashboard is recorded as "refused", which the strategy never retries

**`real.py:126-130`**

```python
msg = str(e)
kind = msg.split(":", 1)[0] if ":" in msg else "crash"
if kind not in ("oom", "crash", "timeout", "verify"):
    kind = "refused"
```

The kind is parsed out of a human-readable message. `Dashboard._call` (line 39) formats every failure as
`f"dashboard {method} {path} failed: {e}"`, so a `urllib` timeout on `POST /api/models/load` (the call has a
180 s timeout, `dashboard.py:51`, and the dashboard stops the old server and waits for it to exit *before* it
replies) arrives as `"dashboard POST /api/models/load failed: <urlopen error timed out>"`. Its prefix is
`"dashboard POST /api/models/load failed"`, which is not in the whitelist, so it becomes `"refused"`.

What goes wrong: "refused" is one of the four outcomes `strategy.ensure_loaded` stores as a `known_failure` for
this model file (line 200) and never loads again. A dashboard that was slow for three minutes - because the old
server was slow to exit, or because the machine was thrashing - permanently marks a perfectly good
configuration as refused. Nothing was refused; the tuner just ran out of patience.

Scenario: the previous trial's llama-server takes 200 s to exit. The dashboard's load call is still waiting when
the tuner's 180 s fires. The config is recorded as refused, the run moves on, and the next run of the same model
skips it from history without trying.

Fix: do not parse English to classify failures. Give `DashboardError` a `kind` attribute set where the error is
raised (the HTTP layer knows whether it timed out, was refused, or returned an error body), and have `load()`
use that. Failing that, at least map `"timed out"`/`"timeout"` in the message to `timeout` before falling back to
`refused`.

Tests: `proposed_tests/test_real_measure.py::test_a_slow_dashboard_is_not_recorded_as_a_refusal` (**fails**).

---

## 9. One slow request ends the whole run

**`real.py:155-163`**

```python
except LlamaError as e:
    if guard.fired: ...
    if self.client.ready(timeout=5):
        raise                       # the server is fine; a genuine request problem
```

`ready()` asks `/v1/models`, which llama-server answers while it is generating, so it says nothing about the
generation path. When `bench.measure`'s 300 s per-request timeout (`bench.py:169,176`) fires on a server that is
merely slow - a 27B with layers on the CPU answering a 400-token prose prompt, or a machine that was paging -
the raw `LlamaError` is re-raised.

What goes wrong: `strategy.measure` catches only `LoadFailed` (`strategy.py:253`). A `LlamaError` propagates out
of `Tuner.run`, out of `runner.run`, and the whole run is marked failed. One slow prompt on one config throws
away every trial, the report and the restore path that was working. The inverse is equally unhelpful: a server
that is wedged (accepting connections, answering `/v1/models`, never finishing a generation) is reported as
"the server is fine", and each repetition burns the full 300 s.

Also, nothing happens to the request that timed out. llama-server keeps generating it, so the next measurement
of the same config queues behind the abandoned one and its samples are contaminated.

Fix: give `measure()` a third outcome for "the request timed out on a live server" - a `LoadFailed("timeout")`,
or better a dedicated exception the strategy treats as "re-measure this once, then give up on this config" -
instead of letting `LlamaError` escape. Judge liveness by something that reflects generation (a short chat
completion, or the server's slot state), not by `/v1/models`.

Tests: `proposed_tests/test_real_measure.py::test_a_timed_out_request_on_a_live_server_is_a_timeout_not_a_crash`
(**fails**).

---

## 10. The prompt sequence is keyed by config, so a re-measured config is compared on different prompts

**`real.py:149-150`**, **`real.py:178`**, **`bench.py:123-138`**, with **`strategy.py:687-696`** and
**`strategy.py:318`**

`_counts` maps `(config key, workload) -> repetitions taken` and is never reset - not by `load()`, not by a
reload. `measure()` therefore continues the variant sequence instead of restarting it.

What goes wrong, two ways:

* **Different prompts in a comparison.** `strategy.phase_confirm` (line 690) builds a *new* `TrialRecord` for the
  config that has already been measured, forces a reload (line 691) and measures it again. That round gets
  variants 3,4,5; the baseline it is confirmed against was measured on 0,1,2. `run()` then computes
  `imp = stats.relative_score(confirmed, base.medians(), weights)` (lines 741, 756-757) - a weighted geometric
  mean of per-workload median ratios. The number the user is asked to approve therefore divides the winner's
  median, measured on three other texts, by the baseline's median. For `prefill` the four passages are different
  documents; for `prose` they are different topics. The design's rule - "Repetition i uses prompt variant i, the
  same sequence for every config" - is broken at the one place the result is read.
* **The same prompt twice for one config.** `_counts` is only advanced on success (line 178), so a round that
  dies on its last repetition leaves the counter where it was and the retry replays prompts 0 and 1 for that
  same config, immediately after the n-gram speculator has seen the answers. Separately, `_variant` cycles with
  `i % len(pool)` and the pools are shorter than the repetition counts a run asks for: `prefill` has 4 passages
  (`bench.py:112-120`) while `strategy.challenge` asks for `reps + 2` (line 318) = 5, so a single config's fifth
  sample is a repeat of its first. That is the 5x inflation DESIGN.md records from 2026-09-30, still reachable.

`strategy._measure_power_safe` (line 276-290) makes it worse: after a power-change discard it calls
`backend.measure` again, which continues the sequence, so the samples that survive are the *second* set of
prompts while every other config in the run was measured on the first.

Fix: the variant sequence must belong to the *comparison*, not to the config. Key `_counts` on the workload and
the round, or reset it whenever a config is (re)loaded, so that every config in a comparison is asked the same
texts in the same order; advance it per repetition as they are taken, not per call, so an interrupted round
cannot replay a prompt. Make the pools at least as long as the largest `reps` a run can ask for, or have
`_variant` refuse to hand out a text this config has already seen.

Tests: `proposed_tests/test_real_measure.py::test_a_config_measured_twice_is_measured_on_the_same_prompts`
(**fails**), `::test_an_interrupted_measurement_does_not_replay_the_same_prompts` (**fails**);
`proposed_tests/test_bench_prompts.py::test_a_run_never_measures_the_same_prompt_twice_for_one_config`
(**fails**), `::test_a_second_round_of_measurements_does_not_repeat_a_prompt` (**fails**),
`::test_every_workload_offers_a_distinct_prompt_per_repetition` (**fails** - pools are prose 8, code 8,
prefill 4).
`::test_the_warm_up_prompt_is_not_one_of_the_workload_prompts` (**passes**) - that part is right.

---

## 11. Context verification is fatal on the first observation, and `restore()` asks for a context of -1

**`dashboard.py:141-146`**, called from **`real.py:74`** and **`real.py:125`**

```python
if n_ctx == want_ctx and dash.llama.ready():
    return pid, time.time() - t0
if n_ctx and n_ctx != want_ctx:
    raise DashboardError(f"verify: server reports n_ctx {n_ctx}, expected {want_ctx}")
```

What goes wrong:

* The mismatch is fatal on the very first poll, with no grace period. A new server that is still applying
  `-c` (or whose `/props` still describes the previous generation) fails the load. `verify` is in the
  never-retry list at `strategy.ensure_loaded` line 193, and is a `known_failure` in history, so a transient
  value costs that config permanently.
* `restore()` computes `want_ctx` as `int(saved.get("contextSize") or 0) or -1` (line 74). For a model whose
  `models.json` entry has no `contextSize` that is `-1`, which no server can ever report, so the first poll
  raises `verify: server reports n_ctx 4096, expected -1`. A restore that worked perfectly is logged as
  "restore problem", which is exactly the noise that hides a restore that really did fail. The same call also
  cannot distinguish "the model is back" from "the model is back with the wrong context".

Scenario: the user's `models.json` entry for a model that was added by hand carries only `gpuLayers`. The run
finishes, the user's model is reloaded correctly, and the log ends with `restore problem: verify: server reports
n_ctx 4096, expected -1`. The user learns to ignore restore problems.

Fix: let `want_ctx` be `None` to mean "do not check the context" and pass `None` from `restore()` when
`models.json` has no `contextSize` (or fall back to `self.dash.llama.n_ctx()` read back before the load). For
the load path, require the mismatch to persist for a few seconds - and match on the value the dashboard was
asked for, allowing the server's documented rounding - before calling it a `verify` failure. Verification should
be a conclusion, not the first observation.

Tests: `proposed_tests/test_dashboard_wait.py::test_a_context_that_is_not_applied_yet_is_not_a_permanent_verification_failure`
(**fails**); `proposed_tests/test_real_restore.py::test_restore_verifies_the_context_it_actually_asked_for`
(**fails**).

---

## 12. A server that dies during a quality suite is scored as a quality regression

**`real.py:181-187`**

`quality()` returns whatever the suite returns. Every suite records a `LlamaError` as a failed task rather than
raising (`quality/reasoning.py:214-242` and the same pattern in `coding.py`, `recall.py`, `tools.py`), so a
server that crashed half way through the reasoning set comes back as `0/8`, with `"cause": "server_error"` in
the details.

What goes wrong: `strategy.phase_quality` compares that 0/8 with the baseline's score and, when the drop exceeds
the allowance, concludes "Quality gate: suite dropped by N tasks; reverting cache type and thinking settings to
the baseline's" - a confident, wrong decision caused by a crash, not by quality. Nothing in `quality()` checks
that the server survived, and `_save_crash_log` is never called, so the log is overwritten by the next load
either. `strategy.interruption` cannot catch it either: it needs `tokens` and `seconds`, and a failed task has
`tokens == 0`.

Fix: `quality()` should look at the reasons in `QualityResult.details`. If most failures share a
`cause` of `server_error`/`timeout`, treat it as a load failure: save the crash log, clear `self._loaded` and
raise `LoadFailed` exactly as `measure()` does. A suite that scores 0/8 with no server errors is still a real
result and should be kept.

Tests: `proposed_tests/test_real_measure.py::test_a_server_that_died_during_a_quality_suite_is_not_scored`
(**fails**).

---

## 13. Dashboard calls have a socket timeout, not a deadline

**`dashboard.py:31-39`**

`urllib.request.urlopen(req, timeout=timeout)` applies the timeout to each socket operation, and `r.read()` is a
loop of those. A peer that answers the headers and then dribbles the body keeps the tuner blocked for as long as
it likes: with a 120 s timeout on `stop_llama` a response of one byte per second takes 120 s per byte to fail.
AGENTS.md requires "hard request timeouts" on every call, and DESIGN.md repeats it; this is not one.

What goes wrong: a wedged or trickling dashboard hangs the run at a point where the watchdog and the spill guard
are the only things still running. The spill guard will stop llama-server for a RAM reason that has nothing to do
with what is actually wrong, and the run dies for a misleading reason.

Fix: enforce a total deadline around the call (read with a deadline, or run the request in a worker thread and
bound the wait), and log the elapsed time when a call is cut short so the run's log says which call stalled.

Tests: `proposed_tests/test_dashboard_wait.py::test_dashboard_calls_have_a_total_deadline` (**fails**: a 1 s
timeout took 5.8 s against a server that trickles a byte every 300 ms).

---

## Also worth knowing, not counted as findings

* `real.measure` calls `self.project(loaded)` (line 169) and `load` calls `self.project(cfg)` (line 115) while
  the machine is at its worst - `FitParams.project` spawns `llama-fit-params.exe` with a 90 s timeout
  (`fitparams.py:133`). Spawning a process when available RAM is under a gigabyte is the last thing that machine
  needs, and a fit-params timeout inside crash handling replaces the real diagnosis with `FitError`.
* `real.restore` re-reads `models.json` from disk rather than asking the dashboard (line 65), so the tuner's idea
  of "the user's saved settings" can be a different version of the file from the one the dashboard will use for
  the restore.
* `dashboard.models_config` (line 69-70) and `llama_log_tail` (line 103-108) read files the dashboard owns. Both
  are correct-looking and both are outside the `_call` wrapper's error handling; finding 2 is what that costs.

---

## Proposed test index

| # | Test | File | Now |
|---|---|---|---|
| 1 | `test_a_watchdog_trip_is_not_invisible_to_the_run` | `test_runner_watchdog.py` | fails |
| 2 | `test_the_trip_flag_is_written_before_the_process_dies` | `test_runner_watchdog.py` | passes |
| 3 | `test_a_failing_restore_still_writes_the_report` | `test_runner_watchdog.py` | fails |
| 4 | `test_restore_survives_an_unreadable_models_json` | `test_real_restore.py` | fails |
| 5 | `test_restore_survives_a_dashboard_that_refuses` | `test_real_restore.py` | passes |
| 6 | `test_a_props_hiccup_does_not_leave_the_users_model_unloaded` | `test_real_restore.py` | fails |
| 7 | `test_begin_uses_the_listening_port_when_props_fails` | `test_real_restore.py` | fails |
| 8 | `test_begin_refuses_to_continue_while_the_old_server_is_alive` | `test_real_restore.py` | fails |
| 9 | `test_restore_verifies_the_context_it_actually_asked_for` | `test_real_restore.py` | fails |
| 10 | `test_a_trial_loaded_model_is_restored_with_its_own_settings` | `test_real_restore.py` | passes |
| 11 | `test_a_slow_load_is_not_called_a_crash_because_of_the_previous_servers_log` | `test_dashboard_wait.py` | fails |
| 12 | `test_a_stale_error_line_does_not_fail_a_load_that_is_still_starting` | `test_dashboard_wait.py` | fails |
| 13 | `test_a_server_that_exited_is_reported_from_its_log_not_as_a_timeout` | `test_dashboard_wait.py` | fails |
| 14 | `test_process_alive_says_yes_when_tasklist_fails` | `test_dashboard_wait.py` | passes |
| 15 | `test_a_context_that_is_not_applied_yet_is_not_a_permanent_verification_failure` | `test_dashboard_wait.py` | fails |
| 16 | `test_dashboard_calls_have_a_total_deadline` | `test_dashboard_wait.py` | fails |
| 17 | `test_stop_disarms_the_guard` | `test_real_spill_guard.py` | fails |
| 18 | `test_stop_waits_for_a_stop_that_is_already_under_way` | `test_real_spill_guard.py` | fails |
| 19 | `test_a_config_measured_twice_is_measured_on_the_same_prompts` | `test_real_measure.py` | fails |
| 20 | `test_an_interrupted_measurement_does_not_replay_the_same_prompts` | `test_real_measure.py` | fails |
| 21 | `test_a_timed_out_request_on_a_live_server_is_a_timeout_not_a_crash` | `test_real_measure.py` | fails |
| 22 | `test_a_slow_dashboard_is_not_recorded_as_a_refusal` | `test_real_measure.py` | fails |
| 23 | `test_a_load_that_times_out_still_saves_the_log` | `test_real_measure.py` | fails |
| 24 | `test_a_server_that_died_mid_measurement_keeps_its_log` | `test_real_measure.py` | passes |
| 25 | `test_a_server_that_died_during_a_quality_suite_is_not_scored` | `test_real_measure.py` | fails |
| 26 | `test_a_run_never_measures_the_same_prompt_twice_for_one_config` | `test_bench_prompts.py` | fails |
| 27 | `test_a_second_round_of_measurements_does_not_repeat_a_prompt` | `test_bench_prompts.py` | fails |
| 28 | `test_every_workload_offers_a_distinct_prompt_per_repetition` | `test_bench_prompts.py` | fails |
| 29 | `test_the_warm_up_prompt_is_not_one_of_the_workload_prompts` | `test_bench_prompts.py` | passes |

The six that pass are the controls: they pin behaviour that is already correct (the warm-up prompt, the crash-log
copy on the `measure` crash path, the `DashboardError` half of `restore`, the trial-load settings, the trip flag)
so a fix for the findings above cannot quietly break them.

`proposed_tests` is outside `pyproject.toml`'s `testpaths`, so `pytest -q` does not collect it; the reviewed
modules were not modified.
