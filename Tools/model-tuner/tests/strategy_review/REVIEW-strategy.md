# Review: `modeltuner/strategy.py`

Findings ranked most severe first. Each one names the file and line, what goes wrong, a concrete scenario, a
suggested fix, and (where one exists) the proposed test under `proposed_tests/` and whether it fails today.

**Test status at the time of writing:** `uv run --with pytest pytest -q tests/` → 352 passed, 3 skipped, 3 failed.
The three failures are pre-existing and unrelated (`tests/test_quality_coding.py::TestTasksCarriedOver` needs a
`_legacy/codeeval/` directory that is not part of this snapshot). `proposed_tests/` contains 18 tests: 16 fail and
demonstrate the findings below, 2 pass as controls. Nothing under `modeltuner/` or `tests/` was modified.

---

## 1. The thinking-length search replaces the incumbent with a config whose speed was never measured

**`modeltuner/strategy.py:663-685`** (`_tune_thinking`), called from **`strategy.py:605`**.

`_tune_thinking` builds one candidate per option, runs only the quality suite on each, and returns
`pick[2]`:

```python
results.append((r.passed, -r.tokens, t, val))
...
pick = max((r for r in results if r[0] == best_pass), key=lambda r: r[1])
return pick[2]
```

Nothing in that path calls `challenge()`, so `stats.compare()` never sees the pick. `phase_quality` then does
`inc = self._tune_thinking(inc)` and `run()` does `best = inc`, so the unchecked config becomes the winner and
the number in the report.

The speed objective of the whole tuner is the weighted mix of prose/code decode and prefill; the thinking knobs
are on that path like any other, and `reasoning_effort` / `--reasoning-budget` change how much work a request
does per token.

**Scenario.** A model whose template honours `reasoning_effort`. `low` gives shorter answers that are harder to
decode, so prose decode drops from ~91 tok/s to ~31 tok/s, while the reasoning suite passes the same number of
tasks for every setting (so tokens are the only tiebreak and `low` wins). The run reports
`improvement = -38.5%` and returns the slow config as `best` anyway.

**Fix.** Keep the quality search, but only accept a thinking setting that the incumbent does not lose to on
speed: run `challenge()` between the incumbent and each pick (or between the incumbent and the chosen pick), and
fall back to the incumbent when the pick is not clearly better. Also record the speed comparison in the decision
message, so the report shows why the setting was taken.

**Test.** `proposed_tests/test_f1_thinking_never_speed_checked.py` — **fails today** (`-38.5% > 0` is false).

---

## 2. The quality gate can announce a revert, skip it silently, and still return the failing config

**`modeltuner/strategy.py:622-630`** (`phase_quality`).

```python
if drop > allowed:
    self.emit("decision", f"Quality gate: {suite} dropped by {drop} tasks; reverting cache type and "
                          f"thinking settings to the baseline's.")
    cfg = inc.cfg.with_(cache_k=base.cfg.cache_k, cache_v=base.cfg.cache_v, ...)
    if cfg.key() != inc.cfg.key() and self.fits(cfg):
        inc = self.trial(cfg, "quality")
        self.measure(inc, [w for w in self.ref])
```

The decision is emitted before the revert is attempted, and the revert is guarded by two conditions that fail in
three ordinary situations:

* **the regression is not caused by the cache type or the thinking settings** — speculation is the usual culprit
  (`cfg.key() == inc.cfg.key()` when the winner already uses the baseline's cache and effort);
* **the baseline's cache type does not fit at the chosen context** — which is exactly what happens after
  `phase_fit` chose q4_0 because only q4_0 reached the requested context, so `self.fits(cfg)` is false;
* the reverted config is never re-gated, so even a successful revert is unverified.

DESIGN.md §7: *"A config whose quality drops beyond noise versus the baseline is rejected however fast it is."*
Today it is not rejected; it is announced and kept.

**Scenario.** Speculation is a large speed win and costs six tool tasks. The gate fires, prints
*"Quality gate: tools dropped by 5 tasks; reverting cache type and thinking settings to the baseline's."*, then
finds that the baseline's `f16` cache does not fit at the chosen 73,728-token context and skips the revert. The
run finishes with `status="ok"`, `improvement=+75.8%`, and `best` still speculating and still failing the tool
suite (2/8 against the baseline's 7/8).

**Fix.** Separate "detected" from "reverted": try the revert, and when it cannot be applied (or does not restore
quality) fall back to the baseline config for that suite and say so, e.g. *"…; reverting … was not possible at
this context, so the candidate is rejected and the baseline's cache/thinking are kept."* If the only remaining
option is to keep the config, mark the run as degraded instead of returning it as a clean winner.

**Test.** `proposed_tests/test_f2_quality_gate_revert_skipped.py` — both tests **fail today**.

---

## 3. The gate compares a full suite against a truncated one, because `_quality` ignores `limit`

**`modeltuner/strategy.py:633-636`** (`_quality`) together with **`strategy.py:604-620`** (`phase_quality`).

```python
def _quality(self, t, suite, limit):
    if suite in t.quality:
        return t.quality[suite]          # `limit` is not part of the key
```

`phase_quality` runs `_tune_thinking` first, which calls `self._quality(t, "reasoning", None)` — the whole
reasoning set. The picked thinking config therefore carries a full-size result, and when the gate loop later
asks for `"reasoning"` with `limit=8` it gets that cached full-size result back, while the baseline is measured
at 8 tasks. `phase_quality` then falls into its "totals differ" branch:

```python
drop = round((r_old.rate - r_new.rate) * r_old.total)     # r_old.total == 8
allowed = max(1, round(0.1 * r_old.total))                # == 1
```

One baseline task is 12.5 rate points, so `drop` can only take the values 0, ±1, ±2 … and the gate only fires
above 2, i.e. above a 25-point rate drop. A real 5-point KV-quantisation regression becomes
`round(0.075 * 8) = 1`, which is not `> 1`, so the gate does nothing. The emitted line shows the mismatch
outright: `quality reasoning: 43/50 vs baseline 6/8`.

It also spends five full reasoning suites in medium mode, whose stated budget is 8 tasks per suite.

**Scenario.** 96K requested, so `phase_fit` picks q4_0 (the sim and real llama.cpp both lose a few points of
accuracy at q4_0 KV). The winner scores 0.80, the baseline 0.875 — a 7.5-point regression that DESIGN says must
be gated. `round((0.875 - 0.80) * 8) = 1`, `allowed = 1`, `1 > 1` is false. The q4_0 config ships.

**Fix.** Make the memo key `(suite, limit)`, or store the limit alongside the result and refuse to reuse a
result measured at a different size. In medium mode, bound the gate by pass *rate* with an explicit tolerance
rather than by a task count taken from the smaller of the two suites.

**Test.** `proposed_tests/test_f3_quality_suite_size_mismatch.py` — `…_same_size` and
`…_whole_suites_for_the_thinking_search` **fail today**; `…_five_point_cache_regression` **fails today** with the
rate-based assertion (the earlier count-based form passed for the wrong reason, which is itself the bug).

---

## 4. Any load failure that is not oom/crash/spill destroys the whole run

**`modeltuner/strategy.py:445-446`** (`phase_validate`).

```python
if t.outcome not in ("oom", "crash", "spill"):
    raise RuntimeError(f"could not load {cfg.describe()}: {t.reason}")
```

`run()` catches only `_Budget`, `Stopped` and `PowerChanged`, so this `RuntimeError` escapes `run()` entirely.
The other four outcomes the backend can produce say nothing about whether the configuration fits:

| outcome | raised by | transient? |
|---|---|---|
| `timeout` | `real.py` → `dashboard.wait_for_model` | yes — a slow first load, a busy dashboard |
| `refused` | `real.py` maps any unrecognised `DashboardError` here | yes — wrong port, dashboard restarting |
| `verify` | `/props` disagreed (new PID ok, `n_ctx` read too early) | yes |
| `ram` | `safety.preflight` refused because another app is using RAM | yes |

The cost is the entire run: `runner.run` marks it failed, writes no report, and the `TuneResult` never reaches
the history or `report.write`, so every measurement already taken is lost.

**Scenario.** `wait_for_model` times out once on the very first load (a cold dashboard on a busy machine). The
run raises `RuntimeError: could not load ctx 73728, all layers on GPU, q8_0 cache, …: wait_for_model timed out
after 120s` after **one** load, and nothing is reported.

**Fix.** Only `oom`/`crash`/`spill`/`ram` mean "this configuration does not work here" and should trigger the
back-off loop. `timeout`, `refused` and `verify` should be retried a couple of times, and if they persist, raise
a typed exception that `run()` turns into `status="failed"` with a `TuneResult` — so the runner still restores the
user's model and writes what it had.

**Test.** `proposed_tests/test_f4_load_failure_aborts_run.py` — the four parameterised cases **fail today**;
`test_a_candidate_that_fails_to_load_is_only_that_candidate` passes (the equivalent failure *inside* a search is
already handled locally, which is the contrast).

---

## 5. The n-gram search hardcodes a workload the baseline may not have, and crashes the run

**`modeltuner/strategy.py:549`** (`phase_spec_params`).

```python
inc = self.search(inc, "ngram match", [4, 6, 8, 12, 16, 24], "ngram_n_match", ["code"], "spec-params",
                  default=24)
```

Every other phase filters its workload list against what the baseline actually produced — `phase_spec:480` and
`phase_threads:597` use `[w for w in ("prose", "code") if w in self.ref]`, and `phase_batch:554` returns early
when `"prefill" not in self.ref`. Here `["code"]` is unconditional. When the baseline has no usable
`gen_tps` for the code workload, `TrialRecord.medians()` (`strategy.py:68-74`) drops `"code"` from `self.ref`,
`composite()` returns `[]` (`strategy.py:296-297`) and `stats.compare([], [...])` raises
`ValueError: compare() requires non-empty input for a` — uncaught, losing the whole run.

llama-server reports no `predicted_per_second` when a request produced no timings (`bench.py:150`), and a
partial code set is a normal outcome of a slow or interrupted response.

**Scenario.** Two of the three code requests in the baseline come back without timings. The run dies with
`ValueError: compare() requires non-empty input for a` raised from `stats.py:111`, after the fit, the baseline
and half the spec phase have already been measured.

**Fix.** Build the workload list the same way the other phases do
(`[w for w in ("prose", "code") if w in self.ref]`) and skip the phase when it is empty. As a backstop,
`challenge()` should treat an empty composite as "cannot decide" instead of calling `stats.compare`.

**Test.** `proposed_tests/test_f5_code_workload_hardcoded.py` — **fails today**.

---

## 6. The baseline is loaded with no VRAM check at all, and can end up as the winner

**`modeltuner/strategy.py:462-463`** (`phase_baseline`) via `ensure_loaded` (`strategy.py:189-223`).

`ensure_loaded` — the single funnel for every load in the tuner — never calls `fits()`. The only VRAM gate is
`phase_validate`, and the baseline never goes through it. A configuration whose projection is already over
`vram_budget_mib()` is therefore loaded deliberately. `RealBackend.load` preflights *host* RAM
(`safety.preflight(proj.host_self_mib)`) and never device RAM, so nothing downstream stops it either. Under
WDDM the overflow is backed by system RAM — the failure DESIGN records for OxCoder, where available RAM fell to
0.13 GB and the watchdog tripped. `_SpillGuard` only runs during `measure`, after the damage.

**Scenario.** Other apps hold 1.5 GB of VRAM; the budget is 10,215 MiB. The user's saved config projects to
10,471 MiB. The tuner loads it anyway (`loading: ctx 16384, all layers on GPU, q8_0 cache, …`), and because the
in-budget fitted config (29 layers on the GPU) loses the speed comparison, `result.best` **is** the over-budget
config — the tuner proposes a setting it knows does not fit.

**Fix.** Before loading the baseline, project it and refuse it if `pr.device_self_mib + self.overhead(cfg) >
self.budget`; say why in the note that already exists ("the current settings did not load right now"), and fall
back to the fitted config. If it does load over budget, run the same trim `phase_validate` uses instead of
accepting it.

**Test.** `proposed_tests/test_f6_baseline_ignores_vram_budget.py` — **fails today** (the winner is 256 MiB over
the budget).

---

## 7. `DEFAULT_MTP_OVERHEAD_MIB` is less than half the overhead this machine actually shows

**`modeltuner/strategy.py:33`.**

```python
DEFAULT_MTP_OVERHEAD_MIB = 450          # until calibration has measured it for this family
```

DESIGN.md, "Lessons from real runs": *"llama-fit-params cannot see MTP. The MTP draft context cost 700-1200 MiB
on the 9B."* Until a family has been calibrated, `fits()` (`strategy.py:153-155`) under-reserves by 250-750 MiB.
Two consequences:

* `phase_fit` spends that slack on context, and `phase_validate` then trims it away again — a whole 4096/8192
  grid step per attempt (`strategy.py:438-440`), so a 4 MiB overflow still costs 4096 tokens of context;
* `phase_spec:487` calls `self.fits(cfg)` before loading a `draft-mtp` candidate, and that check says yes, so a
  configuration that cannot fit is loaded on purpose.

Calibration repairs this from the second run onwards, which is exactly why it is easy to miss.

**Scenario.** A 9B-shaped model on a quiet machine with a real 900 MiB MTP overhead. `phase_fit` returns a
config the `draft-mtp` variant of which `fits()` approves; the load really needs 10,651 MiB against a
10,215 MiB budget. On hardware that is a WDDM spill, not a rounding error.

**Fix.** Raise the default to the low end of the measured range (700 MiB), and treat an uncalibrated family as
"reserve 700 MiB, expect the first `phase_validate` to confirm". If `phase_fit` sees no calibration for the
family, prefer giving up some context up front over discovering the overflow with a load.

**Test.** `proposed_tests/test_f8_mtp_overhead_default.py` — both tests **fail today**.

---

## 8. A stale failure blacklists a configuration for ever

**`modeltuner/strategy.py:198-203`** (`ensure_loaded`) with **`modeltuner/history.py:330-343`** (`known_failure`).

```python
prior = self._hist("known_failure", self.fingerprint, t.cfg)
if prior in ("oom", "spill", "crash", "refused"):
    t.outcome, t.reason = prior, "failed in an earlier run; not loading it again"
```

`known_failure` selects only rows whose outcome is a failure (`outcome IN ('oom','crash','spill','refused','ram',
'verify','timeout')`), newest first. A later *success* for the same config is invisible to it. Its own docstring
is explicit: *"whether a stale failure is still believed is the caller's call, not the database's"* — and the
caller believes it unconditionally and permanently. Nothing ties the failure to the conditions that caused it,
and `real.py:128-130` maps every unrecognised dashboard error to `refused`, so a single busy moment records a
permanent "refused".

**Scenario.** Run 1 finds that 73,728 tokens with MTP does not fit because a browser held 3 GB of VRAM. Run 2,
on a quiet machine, finds `known_failure` still returns `oom` for that exact config — even though history also
contains a *later* run in which it loaded and measured successfully — and never loads it. Every future run does
the same, so the tuner can never go back above the context it once proved it could use.

**Fix.** Treat a failure as stale when a later trial for the same file+config succeeded, and age failures out
(for example, ignore anything older than the last successful run of that model, or re-check `ram`/`refused`
outcomes once). For `oom` specifically, re-check against the current budget before trusting it.

**Test.** `proposed_tests/test_f7_stale_known_failure.py` — **fails today**.

---

## 9. Hitting the load budget reports success, so an unquality-gated winner is persisted as the model's best

**`modeltuner/strategy.py:742-747`** (`run`), consumed by **`modeltuner/runner.py:123-125`**.

```python
except _Budget:
    status = "ok"
    self.emit("note", f"Load budget ({self.o.max_loads}) reached; stopping the search with the best so far.")
```

`_Budget` can be raised from `ensure_loaded` anywhere, including inside the quality phase. `runner.py` then
writes the winner into history as this model's best whenever `status == "ok"`, and `best_for` makes the next run
start from it. Two problems:

* a medium/full run cut short after one of five thinking suites reports `status="ok"` with no quality gate
  having run, and that ungated config becomes the model's recorded best;
* `best` is only assigned at `strategy.py:740` (`best = inc`, after `phase_quality`), so a budget hit in any
  earlier phase discards the incumbent and falls back to
  `max((t for t in self.order if t.outcome == "measured"), key=...)`, which is not the same thing — with
  `max_loads=1` that fallback finds nothing and the run reports `status="failed"` after having emitted
  "Load budget reached; stopping the search with the best so far".

**Scenario.** `mode="medium", max_loads=13`: the run reaches *"Thinking length and quality gates"*, runs one
quality suite, then the budget stops it. `status="ok"`, no `quality … vs baseline …` line was ever emitted, and
`runner.py` would store that config as the model's best.

**Fix.** Give the budget path its own status (`"budget"`), or set a flag when the quality phase was entered and
did not finish and refuse to report `"ok"` in that case. Assign `best = inc` as soon as the incumbent is known
rather than at the end, so the fallback is a formality.

**Test.** `proposed_tests/test_f9_budget_reports_ok.py` — **fails today** (`status="ok"`, no gate comparison).

---

## 10. The final confirmation never re-measures the baseline it is compared against

**`modeltuner/strategy.py:687-696`** (`phase_confirm`).

```python
def phase_confirm(self, best: TrialRecord, base: TrialRecord) -> dict[str, float]:
    ...
    fresh = TrialRecord(cfg=best.cfg, phase="confirm")
```

`base` is a parameter and is never used. DESIGN.md §8: *"Re-measure the winner against the baseline in the same
power state."* Only the winner is re-measured, so `TuneResult.improvement` — the one number the user is shown
before approving anything — is a difference between a measurement taken at the very start of the run and one
taken at the end, tens of minutes and dozens of loads apart, after the machine has warmed up.

**Scenario.** A 45-load run takes 25 minutes. The baseline is measured in the first minute at cold clocks; the
winner is confirmed in the last. The report's headline percentage mixes the two sample sets.

**Fix.** Load and re-measure the baseline in `phase_confirm` too (it is one extra load out of 45), and compute
`improvement` from that pair. If the re-measured baseline differs from the original by more than the noise bar,
say so rather than silently using whichever number is available.

**Test.** `proposed_tests/test_f10_confirm_ignores_baseline.py` — **fails today** (the baseline is loaded exactly
once in the whole run).

---

## 11. `challenge()` promotes a candidate when the incumbent cannot be measured, with no evidence and no message

**`modeltuner/strategy.py:313-315`**.

```python
if not self.measure(cand, workloads):
    ...
    return stats.Verdict.WORSE
if not self.measure(inc, workloads):
    return stats.Verdict.BETTER
```

`BETTER` is returned on the strength of zero measurements, and unlike the `WORSE` branch above it emits no
event, so the report silently changes leader. `challenge` is documented as the noise-aware decision; it should
be able to say "I cannot tell". Reachable when the incumbent's trial carries a sticky failure outcome from an
earlier phase (`measure` returns `False` through `ensure_loaded`), e.g. after a `spill` during a previous
workload.

**Fix.** Return a third state (`NEED_MORE` or a new "cannot decide") and have callers keep the incumbent, or at
minimum emit the same decision line the `WORSE` branch emits.

---

## 12. `phase_validate` trims whole grid steps for a few MiB of overflow, and never re-checks the trim

**`modeltuner/strategy.py:435-443`**.

```python
per_tok = (t.projection.device_context_mib / cfg.ctx) if t.projection and cfg.ctx else 0.02
cut = int((used - self.budget) / max(per_tok, 1e-3) * 1.1)
new = cfg.with_(ctx=round_ctx(max(self.o.min_ctx, cfg.ctx - cut)))
```

`round_ctx` floors to a multiple of 4096 (8192 above 64K), so a 4 MiB overflow — 0.04% of the budget — is
answered by discarding 4096 tokens of context, and a 750 MiB overflow on a 9B model walks the context down to
`min_ctx` in one step. `device_context_mib` is also the *device share* of the context, which for a hybrid model
or a split model is a small fraction of the real per-token cost, so `per_tok` under-estimates and `cut` explodes.
The trimmed config is then loaded without a `fits()` check and without a calibration update.

**Fix.** Compute the step from the projection's own context cost over the range actually being given up
(`device_context_mib` at `lo` minus at `hi`), keep the 1.1 factor, and prefer one step over several. After the
trim, run `fits()` and, if the trimmed config is still over budget because the overhead was underestimated,
raise `self.mtp_overhead` from the measurement instead of only trimming context.

---

## 13. Smaller items

* **`strategy.py:515-516`** — the note *"a model of the same family did best at N; searching that way first"*
  is emitted whenever `prior != cur`, even when `prior not in values`, in which case the direction is **not**
  changed. The report then claims a search order the code did not use. Move the `emit` inside the condition that
  actually flips `directions`.
* **`strategy.py:382`** — `choice = (choice[0], want)` uses the user's `--ctx` unrounded, while every other path
  goes through `round_ctx` (`strategy.py:405`, `413`). One path can return a context that is not on the 4096
  grid, which then differs from the value `phase_validate` would compute for the same request.
* **`strategy.py:535-538`** — `search`'s closing message compares against `cur`, which is the *default* when the
  field was `None`, so `draft-n-max: 3 -> 2` is reported for a config that actually went from unset to 2.
* **`_quality` / `measure` never call `check_stop()`** — a 50-task suite (and `_tune_thinking` runs five of them)
  cannot be interrupted, so a STOP file can take minutes to take effect in medium/full mode. `safety.stop_requested()`
  is checked only between loads and measurements.
* **`strategy.py:728`** — `pcfg = prev[0].with_(ctx=min(prev[0].ctx, inc.cfg.ctx))` silently shrinks the previous
  run's winner when it was GPU-resident, without recording or explaining the change; the previous best is then
  challenged in a form the previous run never measured.
* **`phase_confirm` creates a `TrialRecord` that is not in `self.trials`**, so `_hist_trial` inserts a second
  history `trials` row for a `config_key` that already has one, under phase `"confirm"`. `known_failure` orders
  by `id DESC`, so a confirm row can shadow a real failure row from the same run.

---

## What I was unsure about

* The `_quality` cache ignoring `limit` (finding 3) may be deliberate: the code comment on the "totals differ"
  branch suggests an author who expected them to differ. Even so, in medium mode the difference is between a
  full suite and a truncated one, and the resulting quantisation is what breaks the gate, so I would fix the
  quantisation either way.
* Finding 1 depends on the reading that the speed objective applies to thinking settings too. If the intent is
  that thinking length is chosen purely on quality and tokens, then the defect is narrower: the tuner should at
  least not *report* a win it did not measure, and `phase_quality` should challenge the pick.
* I did not run anything against the real dashboard, llama-server or `llama-fit-params`; every scenario above is
  reproduced with `sim.SimBackend` or a small subclass of it, so the numbers in the write-ups are simulated.
* The three `tests/test_quality_coding.py::TestTasksCarriedOver` failures are pre-existing in this snapshot
  (they read a `_legacy/codeeval/` directory that is not here); I left them alone.
