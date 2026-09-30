"""Proposed tests: the final confirmation never re-measures the baseline it is compared against.

Finding: modeltuner/strategy.py `phase_confirm(self, best, base)` takes `base` and never uses it:
        fresh = TrialRecord(cfg=best.cfg, phase="confirm")
        self.loaded_key = None                  # force a reload
        self.order.append(fresh)
        if not self.measure(fresh, workloads, reps=self.o.reps): ...
        return fresh.medians()
DESIGN.md step 8: "Re-measure the winner against the baseline in the same power state."  Only the winner is
re-measured, so `TuneResult.improvement` (and the headline number in the report) compares a fresh measurement
taken at the end of the run with a baseline measured at the very start - possibly tens of minutes, dozens of
loads and a different thermal state apart, after the machine has warmed up.  The one number the user acts on is
therefore a difference between two different sample sets.

STATUS: currently FAILS.
"""

from modeltuner.knobs import Config

from tests.strategy_review.helpers import SimBackend, SimGpu, SimModel, loads_of, tuner


def test_the_baseline_must_be_re_measured_at_confirmation_time():
    m = SimModel()
    b = SimBackend(m, SimGpu(), seed=3)
    base = Config(ctx=32768, spec="none", ubatch=512, cache_k="f16", cache_v="f16")
    t, ev = tuner(b, m, base)
    r = t.run(topo=(8, 20))

    confirm = [tr for tr in r.trials if tr.phase == "confirm"]
    assert confirm, "the confirm phase did not run"
    baseline_cfg = r.baseline.describe()
    # the confirm phase must load the baseline again, in the same session as the winner it is compared to
    reloaded_baseline = [msg for msg in loads_of(ev) if baseline_cfg in msg]
    assert len(reloaded_baseline) >= 2, (
        f"the baseline was loaded {len(reloaded_baseline)} time(s) in the whole run: it was measured once at the "
        f"start and the winner was confirmed against it at the end, from a different measurement set"
    )
