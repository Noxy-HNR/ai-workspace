"""Proposed tests: the n-gram search hardcodes a workload that may not exist, and crashes the run.

Finding: modeltuner/strategy.py `phase_spec_params`
        inc = self.search(inc, "ngram match", [4, 6, 8, 12, 16, 24], "ngram_n_match", ["code"], ...)
Every other phase filters its workload list against what the baseline actually produced
(`phase_spec`, `phase_threads`: `[w for w in ("prose", "code") if w in self.ref]`; `phase_batch` returns early
when `"prefill" not in self.ref`).  Here `["code"]` is unconditional, so when the baseline has no usable
`gen_tps` for the code workload `TrialRecord.medians()` omits "code" from `self.ref`, `composite()` returns `[]`
and `stats.compare()` raises `ValueError: compare() requires non-empty input for a` - an uncaught crash that
loses the whole run, including the measurements already taken.

llama-server reports no `predicted_per_second` when a request produced no timings, so an empty or partial code
set is possible on real hardware.

STATUS: currently FAILS.
"""

import pytest

from modeltuner.backend import Sample

from tests.strategy_review.helpers import Config, SimBackend, SimGpu, SimModel, tuner


class NoCodeTimings(SimBackend):
    """The code workload comes back without decode timings."""

    def measure(self, workload, *, reps):
        out = super().measure(workload, reps=reps)
        if workload == "code":
            return [Sample(**{**s.__dict__, "gen_tps": None}) for s in out]
        return out


def test_a_baseline_without_code_timings_must_not_crash_the_run():
    m = SimModel()
    b = NoCodeTimings(m, SimGpu(), seed=3)
    base = Config(ctx=32768, spec="none", ubatch=512, cache_k="f16", cache_v="f16")
    t, ev = tuner(b, m, base)
    try:
        r = t.run(topo=(8, 20))
    except ValueError as e:
        pytest.fail(f"run() crashed with {e}: phase_spec_params searched the 'code' workload although the "
                    f"baseline produced no code timings, so composite() returned [] and stats.compare() raised")
    assert r.status == "ok", r.status
    assert r.best_metrics, "the run should still report what it measured"
