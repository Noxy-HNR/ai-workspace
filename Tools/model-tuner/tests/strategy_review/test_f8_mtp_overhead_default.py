"""Proposed tests: the default MTP reserve is less than half of what DESIGN measured.

Finding: modeltuner/strategy.py:33
        DEFAULT_MTP_OVERHEAD_MIB = 450          # until calibration has measured it for this family
DESIGN.md, "Lessons from real runs": "llama-fit-params cannot see MTP. The MTP draft context cost 700-1200 MiB
on the 9B."  So on the first run for a family `fits()` under-reserves by 250-750 MiB.  Two consequences:
  * `phase_fit` fills VRAM with context against the small reserve, and `phase_validate` then has to throw that
    context away - a whole 4096/8192 grid step at a time - even though the run "succeeded";
  * `phase_spec` calls `self.fits(cfg)` before loading a draft-mtp config, and that check says yes, so a config
    that cannot fit is loaded on purpose.

Calibration fixes this from the second run onwards, which is why it is easy to miss.

STATUS: currently FAILS.
"""

from modeltuner import strategy
from modeltuner.knobs import Config

from tests.strategy_review.helpers import SimBackend, SimGpu, SimModel, profile


def _nine_billion():
    return SimModel(name="sim-9b.gguf", n_layer=36, weights_mib=8300, kv_mib_per_1k_ctx_f16=26.0,
                    compute_mib_per_ubatch_256=560, has_mtp=True, train_ctx=262144)


def test_the_default_reserve_should_match_the_measured_overhead():
    assert strategy.DEFAULT_MTP_OVERHEAD_MIB >= 700, (
        f"DEFAULT_MTP_OVERHEAD_MIB is {strategy.DEFAULT_MTP_OVERHEAD_MIB} MiB; DESIGN measured 700-1200 MiB for "
        f"the MTP draft context, so the first run for a family under-reserves and phase_validate has to trim "
        f"the context back"
    )


def test_the_mtp_config_the_tuner_claims_fits_must_actually_fit():
    m = _nine_billion()
    b = SimBackend(m, SimGpu(), seed=3, mtp_overhead_mib=900)   # DESIGN's measured range, mid-point
    base = Config(ctx=32768, spec="none", ubatch=512, cache_k="q8_0", cache_v="q8_0")
    t = strategy.Tuner(b, profile(m), base, strategy.TuneOptions())
    fitted = t.phase_fit()
    t.budget = t.b.vram_budget_mib()
    mtp_cfg = fitted.with_(spec="draft-mtp", draft_n_max=2)     # exactly what phase_spec will try
    assert t.fits(mtp_cfg), "test setup: phase_spec should want to try the MTP config"
    load = b.load(mtp_cfg)                                      # ... and it must survive the load
    assert load.vram_self_mib <= t.budget, (
        f"fits() said yes, but the MTP config really needs {load.vram_self_mib} MiB against a {t.budget} MiB "
        f"budget - the reserve ({strategy.DEFAULT_MTP_OVERHEAD_MIB} MiB) is smaller than the real draft context"
    )
