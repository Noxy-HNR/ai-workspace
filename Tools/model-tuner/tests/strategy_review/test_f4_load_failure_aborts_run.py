"""Proposed tests: one non-OOM load failure kills the whole run.

Finding: modeltuner/strategy.py `phase_validate`
        if t.outcome not in ("oom", "crash", "spill"):
            raise RuntimeError(f"could not load {cfg.describe()}: {t.reason}")
`run()` only catches `_Budget`, `Stopped` and `PowerChanged`, so this RuntimeError escapes `run()` entirely.  The
backend uses five other outcomes that say nothing about whether the config fits:
  * "timeout"  - `real.py` / `dashboard.wait_for_model` timing out;
  * "refused"  - `real.py` maps any unrecognised DashboardError to "refused" (dashboard busy, wrong port, ...);
  * "verify"   - the server came up but `/props` disagreed (a slow n_ctx read);
  * "ram"      - `safety.preflight` refused because another app was using RAM.
All of them are transient, and the cost is the entire run: the runner marks it failed, writes no report, and the
TuneResult never reaches the history or the report.

STATUS: currently FAILS.
"""

import pytest

from modeltuner.backend import LoadFailed

from tests.strategy_review.helpers import Config, SimBackend, SimGpu, SimModel, tuner


def _flaky(kind, fail_on=1):
    class Flaky(SimBackend):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.tries = 0

        def load(self, cfg):
            self.tries += 1
            if self.tries == fail_on:
                raise LoadFailed(kind, f"synthetic {kind}")
            return super().load(cfg)

    return Flaky


@pytest.mark.parametrize("kind", ["timeout", "refused", "verify", "ram"])
def test_a_transient_load_failure_must_not_destroy_the_run(kind):
    m = SimModel()
    b = _flaky(kind)(m, SimGpu(), seed=3)
    base = Config(ctx=32768, spec="none", ubatch=512, cache_k="f16", cache_v="f16")
    t, ev = tuner(b, m, base)
    try:
        r = t.run(topo=(8, 20))
    except RuntimeError as e:                       # noqa: BLE001 - this is the bug being demonstrated
        pytest.fail(f"run() let a {kind} load failure escape as {type(e).__name__}: {e}\n"
                    f"a transient dashboard or RAM problem costs every measurement in the run")
    assert r.status in ("ok", "stopped", "power_changed"), r.status
    assert r.baseline_metrics, "the baseline was measured before the failure and should still be reported"


def test_a_candidate_that_fails_to_load_is_only_that_candidate():
    """The same failure later in the run (a search candidate, not the validated fit) must stay local."""
    m = SimModel()
    b = _flaky("timeout", fail_on=4)(m, SimGpu(), seed=3)
    base = Config(ctx=32768, spec="none", ubatch=512, cache_k="f16", cache_v="f16")
    t, ev = tuner(b, m, base)
    try:
        r = t.run(topo=(8, 20))
    except RuntimeError as e:                       # noqa: BLE001
        pytest.fail(f"run() let a candidate's timeout escape: {e}")
    assert r.status == "ok", r.status
