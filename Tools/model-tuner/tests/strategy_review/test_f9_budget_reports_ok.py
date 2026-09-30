"""Proposed tests: the load budget silently bypasses the quality gate and still reports "ok".

Finding: modeltuner/strategy.py `run()`
        except _Budget:
            status = "ok"
            self.emit("note", f"Load budget ({self.o.max_loads}) reached; stopping the search with the best so far.")
`_Budget` can be raised from `ensure_loaded` anywhere, including inside the quality phase.  The run then reports
status "ok", which is exactly what runner.py keys on before persisting the winner:
        if result is not None and result.status == "ok" and result.best_metrics:
            hist.set_best(fp, profile.file, profile.family, result.best, ...)
So a medium/full run that ran out of budget after one of five quality suites stores an unquality-gated config in
history as this model's best, and the next run starts from it through `best_for`.  "ok" also overwrites the
"stopped" distinction: a run cut short by the budget is indistinguishable from a finished one.

STATUS: currently FAILS.
"""

from modeltuner.knobs import Config

from tests.strategy_review.helpers import SimBackend, SimGpu, SimModel, decisions, quality_compares, tuner


def test_a_budget_truncated_medium_run_must_not_report_success():
    m = SimModel()
    b = SimBackend(m, SimGpu(), seed=3)
    base = Config(ctx=32768, spec="none", ubatch=512, cache_k="f16", cache_v="f16")
    t, ev = tuner(b, m, base, mode="medium", max_loads=13)
    r = t.run(topo=(8, 20))

    truncated = [d for d in decisions(ev)] + [msg for kind, msg in ev if kind == "note"]
    assert any("Load budget" in msg for msg in truncated), "the budget was not the thing that stopped the run"
    gated = quality_compares(ev)
    suites = {line.split()[1].rstrip(":") for line in gated}
    # runner.py persists any status=="ok" result as this model's best, so a run cut short before every quality
    # gate finished must not claim success.
    assert r.status != "ok" or suites >= {"reasoning", "tools"}, (
        f"status={r.status!r} with only {sorted(suites)} gated; the ungated {r.best.describe()} would be stored"
    )
