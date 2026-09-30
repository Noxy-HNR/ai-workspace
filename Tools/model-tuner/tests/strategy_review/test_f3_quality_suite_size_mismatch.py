"""Proposed tests: the quality gate compares suites of different sizes.

Finding: modeltuner/strategy.py `Tuner._quality` memoises on `suite` alone and ignores `limit`:
        if suite in t.quality: return t.quality[suite]
`_tune_thinking` runs first with `limit=None` (the whole reasoning set), so when `phase_quality` later asks for
the same suite with `limit=8` it gets the cached full-size result back, while the baseline is measured at 8
tasks.  `phase_quality` then falls into its "totals differ" branch and compares rates on the *baseline's* 8-task
scale: `drop = round((r_old.rate - r_new.rate) * r_old.total)` with `allowed = max(1, round(0.1 * r_old.total))`.
One baseline task is 12.5 rate points, so the gate only fires in 12.5-point steps and only above ~25 points -
it cannot see a 5-point KV-quantisation regression at all, which is exactly what DESIGN says q4_0 must be gated
for.  It also runs five full reasoning suites in a mode whose gate budget is 8 tasks each.

STATUS: currently FAILS.
"""

import re

from modeltuner.backend import QualityResult

from tests.strategy_review.helpers import (Config, SimBackend, SimGpu, SimModel, quality_compares, tuner)


class Q4CostsQuality(SimBackend):
    """q4_0 KV loses five points of accuracy - a real and small regression."""

    def quality(self, suite, *, limit=None):
        total = limit if limit else 50
        cfg = self._loaded
        rate = 0.80 if "q4_0" in (cfg.cache_k, cfg.cache_v) else 0.85
        passed = min(total, max(0, round(rate * total)))
        return QualityResult(suite=suite, passed=passed, total=total, tokens=total * 150, seconds=total * 0.4)


def _gate_counts():
    m = SimModel()
    b = Q4CostsQuality(m, SimGpu(), seed=7)
    # a slow baseline so the fit (which wants q4_0 to reach 96K) actually wins, and a f16 baseline whose cache
    # cannot hold 96K - so the revert is both unnecessary and impossible
    base = Config(ctx=32768, ngl=10, spec="none", ubatch=512, cache_k="f16", cache_v="f16")
    t, ev = tuner(b, m, base, mode="medium", allow_q4=True, ctx_target=98304)
    return t.run(topo=(8, 20)), ev


def test_gate_must_compare_two_suites_of_the_same_size():
    r, ev = _gate_counts()
    lines = quality_compares(ev)
    assert lines, "the quality gate never compared anything"
    sizes = []
    for line in lines:
        m2 = re.search(r": (\d+)/(\d+) vs baseline (\d+)/(\d+)", line)
        assert m2, line
        sizes.append((int(m2.group(2)), int(m2.group(4))))
    for new_total, old_total in sizes:
        assert new_total == old_total, (
            f"the gate compared a {new_total}-task result against a {old_total}-task baseline: "
            f"{[line for line in lines]}  (limit is ignored by _quality's cache)"
        )


def test_gate_must_see_a_real_five_point_cache_regression():
    r, ev = _gate_counts()
    q4 = [tr for tr in r.trials if "q4_0" in (tr.cfg.cache_k, tr.cfg.cache_v) and tr.quality]
    base_q = [tr for tr in r.trials if tr.phase == "baseline" and tr.quality]
    assert q4 and base_q, "test setup: the fit should have chosen q4_0 and both should have a reasoning score"
    new, old = q4[-1].quality["reasoning"], base_q[0].quality["reasoning"]
    assert r.best.cache_k != "q4_0" or new.rate >= old.rate - 0.1, (
        f"q4_0 scored {new.passed}/{new.total} ({new.rate:.1%}) against the baseline's {old.passed}/{old.total} "
        f"({old.rate:.1%}) - a {100 * (old.rate - new.rate):.1f} point regression - and still shipped as "
        f"{r.best.describe()}.  `drop = round((r_old.rate - r_new.rate) * r_old.total)` on an 8-task baseline "
        f"turns that into {round((old.rate - new.rate) * old.total)} task(s), which is not more than "
        f"`allowed = max(1, round(0.1 * {old.total}))`, so the gate does nothing"
    )


def test_medium_mode_should_not_run_whole_suites_for_the_thinking_search():
    m = SimModel()
    b = Q4CostsQuality(m, SimGpu(), seed=11)
    base = Config(ctx=32768, spec="none", ubatch=512, cache_k="f16", cache_v="f16")
    t, ev = tuner(b, m, base, mode="medium")
    t.run(topo=(8, 20))
    from tests.strategy_review.helpers import compares
    full = [c for c in compares(ev) if c.startswith("quality reasoning") and re.search(r": \d+/(\d+) vs", c)
            and int(re.search(r": \d+/(\d+) vs", c).group(1)) > 8]
    assert not full, (
        f"medium mode uses limit=8 for the gate but these comparisons used the full suite: {full}"
    )
