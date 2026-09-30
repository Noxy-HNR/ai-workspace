"""Two simulated runs sharing one history database: the second must use what the first learned."""

from modeltuner.history import History
from modeltuner.knobs import Config
from modeltuner.sim import SimBackend, SimGpu, SimModel
from modeltuner.strategy import TuneOptions, Tuner

from tests.test_strategy_sim import _profile


def _run(hist, run_id, seed):
    model = SimModel()
    b = SimBackend(model, SimGpu(), seed=seed)
    events = []
    hist.start_run(run_id, model.name, "fp-sim", "qwen35-hybrid", "quick", b.power_key(), Config(ctx=32768))
    t = Tuner(b, _profile(model), Config(ctx=32768, spec="none", ubatch=512), TuneOptions(),
              emit=lambda k, m, **d: events.append((k, m)), history=hist, fingerprint="fp-sim", run_id=run_id)
    r = t.run(topo=(8, 20))
    hist.finish_run(run_id, r.status, r.best, {"improvement": r.improvement})
    hist.set_best("fp-sim", model.name, "qwen35-hybrid", r.best, r.best_metrics, b.power_key(), run_id)
    return b, r, events


def test_second_run_uses_calibration_best_and_records_everything(tmp_path):
    hist = History(tmp_path / "t.db")
    b1, r1, ev1 = _run(hist, "run-1", seed=1)
    assert r1.status == "ok"
    assert hist.calibration_error("qwen35-hybrid", "", True) is not None
    assert hist.best_for("fp-sim") is not None
    runs = hist.runs()
    assert runs and runs[0]["id"] in ("run-1",)

    b2, r2, ev2 = _run(hist, "run-2", seed=2)
    notes = [m for k, m in ev2 if k == "note"]
    assert any("Calibration from earlier runs" in n for n in notes)
    assert r2.status == "ok"
    # the calibrated MTP overhead means the second fit does not need the over-budget trims the first one did
    trims1 = sum("over the" in m for k, m in ev1 if k == "decision")
    trims2 = sum("over the" in m for k, m in ev2 if k == "decision")
    assert trims2 <= trims1
    hist.close()


def test_known_failures_are_not_loaded_again(tmp_path):
    # run 1 without history, to learn which spec-phase configs the tuner tries on this simulated model
    model = SimModel()
    first = Tuner(SimBackend(model, SimGpu(), seed=5), _profile(model), Config(ctx=32768, spec="none", ubatch=512),
                  TuneOptions()).run(topo=(8, 20))
    tried = [t.cfg for t in first.trials if t.phase == "spec"]
    assert tried
    bad = tried[0]
    # record that config as having run out of memory in an earlier run
    hist = History(tmp_path / "t.db")
    hist.start_run("r0", model.name, "fp-sim", "qwen35-hybrid", "quick", "k", Config(ctx=32768))
    hist.set_trial_outcome(hist.add_trial("r0", "fp-sim", bad, "spec", None), "oom", "boom")
    events = []
    t = Tuner(SimBackend(model, SimGpu(), seed=5), _profile(model), Config(ctx=32768, spec="none", ubatch=512),
              TuneOptions(), emit=lambda k, m, **d: events.append((k, m)), history=hist, fingerprint="fp-sim",
              run_id="r1")
    t.run(topo=(8, 20))
    assert any("failed (oom) in an earlier run" in m for k, m in events if k == "decision")
    assert all(bad.describe() not in m for k, m in events if k == "load")
    hist.close()
