"""Proposed tests: a stale failure blacklists a config for ever.

Finding: modeltuner/strategy.py `ensure_loaded`
        prior = self._hist("known_failure", self.fingerprint, t.cfg)
        if prior in ("oom", "spill", "crash", "refused"):
            t.outcome, t.reason = prior, "failed in an earlier run; not loading it again"
and `History.known_failure` (modeltuner/history.py) queries only failure rows:
        WHERE model_fingerprint=? AND config_key=? AND outcome IN (...) ORDER BY id DESC LIMIT 1
Its own docstring says "whether a stale failure is still believed is the caller's call, not the database's" - and
the caller believes it unconditionally, for ever.  Two consequences:
  * a config that loaded and measured fine in a *later* run is still reported as failed, because the later
    success is invisible to the query;
  * nothing ties the failure to the conditions that caused it, so an OOM caused by another app holding the GPU
    that day permanently removes that context from every future run.
The tuner can then never rediscover a config the machine can actually run, and the report offers a smaller
context than the hardware allows.

STATUS: currently FAILS.
"""

from modeltuner.history import History
from modeltuner.knobs import Config
from modeltuner.sim import SimBackend, SimGpu, SimModel

from tests.strategy_review.helpers import decisions, profile, tuner


def test_a_config_that_worked_later_must_not_stay_blacklisted(tmp_path):
    m = SimModel()
    base = Config(ctx=32768, spec="none", ubatch=512, cache_k="f16", cache_v="f16")

    # pass 1: learn which configs the spec phase tries
    warm = Tuner0(SimBackend(m, SimGpu(), seed=3), m, base)
    first = warm.run()
    spec_cfgs = [tr.cfg for tr in first.trials if tr.phase == "spec"]
    assert spec_cfgs

    # an older run OOMed on one of them; a newer run measured it successfully
    hist = History(tmp_path / "t.db")
    target = spec_cfgs[1]
    hist.start_run("old", m.name, "fp", "qwen35-hybrid", "quick", "k", base)
    hist.set_trial_outcome(hist.add_trial("old", "fp", target, "spec", None), "oom", "another app had the GPU")
    hist.start_run("new", m.name, "fp", "qwen35-hybrid", "quick", "k", base)
    hist.set_trial_outcome(hist.add_trial("new", "fp", target, "spec", None), "measured", "")

    b = SimBackend(m, SimGpu(), seed=3)
    t, ev = tuner(b, m, base, prof=profile(m))
    t.history, t.fingerprint, t.run_id = hist, "fp", "again"
    r = t.run(topo=(8, 20))

    blocked = [d for d in decisions(ev) if "in an earlier run" in d]
    assert not blocked, (
        f"{len(blocked)} config(s) were refused on the strength of a stale failure, including "
        f"{target.describe()} - which a later run loaded and measured successfully "
        f"(improvement this run {r.improvement:+.1%})"
    )
    hist.close()


def Tuner0(backend, m, base):
    from modeltuner.strategy import TuneOptions, Tuner
    return Tuner(backend, profile(m), base, TuneOptions())
