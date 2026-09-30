"""Proposed tests: the quality gate's revert can be a no-op, so a config that failed the gate still wins.

Finding: modeltuner/strategy.py `phase_quality` (the `if drop > allowed:` branch).
The gate emits "reverting cache type and thinking settings to the baseline's" *before* it checks whether the
revert is possible:

    cfg = inc.cfg.with_(cache_k=base.cfg.cache_k, cache_v=base.cfg.cache_v, ...)
    if cfg.key() != inc.cfg.key() and self.fits(cfg):
        inc = self.trial(cfg, "quality")
        self.measure(inc, [w for w in self.ref])

Three ways that guard can fail, all of which leave the failing config as the winner:
  a. the regression is not caused by the cache type or the thinking settings (here: speculation);
  b. the baseline's cache type does not fit at the chosen context;
  c. the winner already carries the baseline's cache and thinking settings (`cfg.key() == inc.cfg.key()`).

DESIGN.md: "A config whose quality drops beyond noise versus the baseline is rejected however fast it is."

STATUS: currently FAILS.
"""

from modeltuner.backend import QualityResult

from tests.strategy_review.helpers import (Config, SimBackend, SimGpu, SimModel, decisions, quality_compares, tuner)


class SpecHurtsQuality(SimBackend):
    """Speculation is a large speed win and a real quality loss on the tool tasks."""

    def quality(self, suite, *, limit=None):
        total = limit if limit else 50
        cfg = self._loaded
        drop = 6 if cfg.spec != "none" else 1
        return QualityResult(suite=suite, passed=total - drop, total=total, tokens=total * 150, seconds=total * 0.4)


def _run():
    m = SimModel()
    b = SpecHurtsQuality(m, SimGpu(), seed=3)
    base = Config(ctx=32768, spec="none", ubatch=512, cache_k="f16", cache_v="f16")
    t, ev = tuner(b, m, base, mode="medium")
    return t.run(topo=(8, 20)), ev


def test_gate_that_fires_must_change_the_winner():
    r, ev = _run()
    fired = [d for d in decisions(ev) if "Quality gate" in d]
    assert fired, "expected the quality gate to fire on a 5/8 tool regression"
    # The gate fired, so the winner must not be the config that failed it: for every suite that has both scores,
    # the winner is within the gate's tolerance of the baseline.
    scored = {s: q for s, q in r.quality.items() if q.get("best") and q.get("baseline")}
    assert scored, "no suite has both a winner and a baseline score"
    for suite, q in scored.items():
        best, baseline = q["best"], q["baseline"]
        allowed = max(1, round(0.1 * baseline[1]))
        assert best[0] >= baseline[0] - allowed, (
            f"the gate reported {fired[0]!r} but the winner still scores {best} on {suite} against {baseline}"
        )


def test_a_revert_that_was_announced_must_be_applied():
    r, ev = _run()
    announced = [d for d in decisions(ev) if "reverting cache type" in d]
    if not announced:
        return
    reverted = [tr for tr in r.trials if tr.phase == "quality"]
    assert reverted, (
        f"{announced[0]!r} was emitted but no quality-phase trial exists: the revert was skipped silently "
        f"and the failing config ({r.best.describe()}) was returned as the winner"
    )
