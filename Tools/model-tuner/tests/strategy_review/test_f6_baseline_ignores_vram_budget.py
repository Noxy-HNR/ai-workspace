"""Proposed tests: the baseline is loaded without any VRAM check.

Finding: modeltuner/strategy.py `phase_baseline` -> `measure()` -> `ensure_loaded()`.  `ensure_loaded` never calls
`fits()`; the only VRAM gate in the tuner is `phase_validate`, and the baseline never goes through it.  So a
configuration whose projection is already over `vram_budget_mib()` is loaded on purpose - which on WDDM means the
overflow is backed by system RAM.  That is the exact failure DESIGN records for OxCoder (available RAM fell to
0.13 GB and the watchdog tripped), and `RealBackend.load` only preflights *host* RAM, never device RAM.

Here the over-budget config is not even rejected: the fitted, in-budget config loses the speed comparison, and the
over-budget baseline becomes `result.best`.

STATUS: currently FAILS.
"""

from modeltuner.knobs import Config

from tests.strategy_review.helpers import SimBackend, SimGpu, SimModel, loads_of, tuner


def test_no_config_over_the_vram_budget_should_ever_be_loaded():
    m = SimModel(weights_mib=9000)
    b = SimBackend(m, SimGpu(), seed=3)
    base = Config(ctx=16384, ngl=999, spec="none", ubatch=512, cache_k="q8_0", cache_v="q8_0")
    budget = b.vram_budget_mib()
    over = b.project(base).device_self_mib - budget
    assert over > 0, f"test setup: the baseline should be over budget (it is {over} MiB over)"

    t, ev = tuner(b, m, base)
    r = t.run(topo=(8, 20))

    for msg in loads_of(ev):
        for tr in t.trials.values():
            if tr.cfg.describe() in msg:
                pr = b.project(tr.cfg)
                assert pr.device_self_mib <= budget, (
                    f"loaded a config projected {pr.device_self_mib} MiB against a {budget} MiB budget: {msg}"
                )
    assert r.best.ctx > 0
    best_pr = b.project(r.best).device_self_mib
    assert best_pr <= budget, (
        f"the winner {r.best.describe()} is projected at {best_pr} MiB, {best_pr - budget} MiB over the "
        f"{budget} MiB VRAM budget"
    )
