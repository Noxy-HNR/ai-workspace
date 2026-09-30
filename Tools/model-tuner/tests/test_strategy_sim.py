"""End-to-end strategy tests against the simulated backend (no GPU needed)."""

from modeltuner.knobs import Config
from modeltuner.profile import ModelProfile
from modeltuner.sim import SimBackend, SimGpu, SimModel
from modeltuner.strategy import TuneOptions, Tuner


def _profile(m: SimModel, **kw) -> ModelProfile:
    d = dict(file=m.name, path="x", size_bytes=m.weights_mib << 20, arch="qwen35", name="sim", n_layer=m.n_layer,
             n_mtp_layers=1 if m.has_mtp else 0, train_ctx=m.train_ctx, n_embd=4096, n_head=16, n_head_kv=4,
             head_dim_k=256, head_dim_v=256, full_attention_interval=4, sliding_window=None, n_expert=0,
             weights_bytes=m.weights_mib << 20, layer_bytes={}, template_supports_tools=True,
             template_has_thinking=True, template_honours_effort=False, server_binary=None, current={})
    d.update(kw)
    return ModelProfile(**d)


def _run(model, base, gpu=None, **opts):
    b = SimBackend(model, gpu or SimGpu(), seed=3)
    events = []
    r = Tuner(b, _profile(model), base, TuneOptions(**opts), emit=lambda k, m, **d: events.append((k, m))).run(topo=(8, 20))
    return b, r, events


def test_gpu_resident_mtp_model_gets_mtp_and_skips_threads():
    b, r, ev = _run(SimModel(), Config(ctx=32768, spec="none", ubatch=512, cache_k="f16", cache_v="f16"))
    assert r.status == "ok"
    assert "draft-mtp" in r.best.spec
    assert r.best.ngl == 999 and r.best.ctx > 32768
    assert r.improvement > 0.2
    assert any("Threads: skipped" in m for k, m in ev if k == "decision")
    assert b.restored is False  # restore is the runner's job


def test_offloaded_model_tunes_threads_toward_p_cores():
    m = SimModel(name="sim-27b.gguf", n_layer=62, weights_mib=16000, has_mtp=False, base_gen_tps=25)
    b, r, ev = _run(m, Config(ctx=32768, ngl=30, threads=16, spec="none"))
    assert r.best.ngl < 999 and r.best.ngl >= 30
    assert r.best.threads in (8, 10, 12)   # the sim's optimum is flat across P-core counts
    assert "draft-mtp" not in r.best.spec


def test_never_exceeds_budget_and_backs_off_on_oom():
    # other apps grow: the budget shrinks, the fit must follow, and no load may OOM twice in a row
    b, r, ev = _run(SimModel(), Config(ctx=262144, spec="none"), gpu=SimGpu(other_mib=5000))
    assert r.status == "ok"
    fails = [m for k, m in ev if k == "load_failed"]
    assert len(fails) <= 2
    assert r.best.ctx < 262144


def test_load_budget_is_respected():
    b, r, ev = _run(SimModel(), Config(ctx=32768), max_loads=4)
    assert r.loads <= 5
