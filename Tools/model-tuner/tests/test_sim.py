"""Tests for modeltuner.sim.SimBackend.

Each test pins one behaviour the strategy relies on: determinism, OOMs, the
hidden MTP overhead, thread sensitivity only when offloaded, per-workload
speculation winners, monotonic projections, and the power-key flip.
"""

from __future__ import annotations

import pytest

from modeltuner.backend import LoadFailed
from modeltuner.knobs import Config
from modeltuner.sim import SimBackend, SimGpu, SimModel


def base_cfg(**overrides) -> Config:
    cfg = Config(ctx=32768, ngl=999, threads=8, batch=1024, ubatch=256)
    return cfg.with_(**overrides) if overrides else cfg


def mean_tps(samples) -> float:
    return sum(s.gen_tps for s in samples) / len(samples)


class TestDeterminism:
    def test_same_seed_replays(self):
        def run(seed: int):
            sim = SimBackend(seed=seed)
            sim.load(base_cfg())
            prose = sim.measure("prose", reps=3)
            code = sim.measure("code", reps=2)
            quality = sim.quality("reasoning", limit=20)
            return prose, code, quality

        first = run(7)
        second = run(7)
        for a, b in zip(first[0], second[0]):
            assert a.gen_tps == b.gen_tps
        for a, b in zip(first[1], second[1]):
            assert a.gen_tps == b.gen_tps
        assert first[2] == second[2]

    def test_different_seeds_differ(self):
        a = SimBackend(seed=1)
        b = SimBackend(seed=2)
        a.load(base_cfg())
        b.load(base_cfg())
        ma = a.measure("prose", reps=3)
        mb = b.measure("prose", reps=3)
        assert [s.gen_tps for s in ma] != [s.gen_tps for s in mb]

    def test_projection_has_no_noise(self):
        sim = SimBackend(seed=3)
        cfg = base_cfg()
        assert sim.project(cfg) == sim.project(cfg)


class TestMemory:
    def test_small_ctx_loads(self):
        sim = SimBackend(seed=0)
        result = sim.load(base_cfg(ctx=32768))
        assert result.vram_used_mib is not None
        assert result.vram_other_mib == 1500
        assert result.n_ctx == 32768

    def test_huge_ctx_ooms(self):
        sim = SimBackend(seed=0)
        with pytest.raises(LoadFailed) as exc:
            sim.load(base_cfg(ctx=262144))
        assert exc.value.kind == "oom"

    def test_mtp_crash_without_head(self):
        sim = SimBackend(model=SimModel(has_mtp=False), seed=0)
        with pytest.raises(LoadFailed) as exc:
            sim.load(base_cfg(spec="draft-mtp", draft_n_max=2))
        assert exc.value.kind == "crash"

    def test_batch_below_ubatch_refused(self):
        sim = SimBackend(seed=0)
        with pytest.raises(LoadFailed) as exc:
            sim.load(base_cfg(batch=128, ubatch=256))
        assert exc.value.kind == "refused"

    def test_measure_needs_load(self):
        sim = SimBackend(seed=0)
        with pytest.raises(RuntimeError):
            sim.measure("prose", reps=1)

    def test_mtp_overhead_breaks_near_limit_fit(self):
        # Default margin (512) is wider than the MTP overhead (350), so use a
        # tight GPU where the budget sits within one overhead of the OOM line.
        sim = SimBackend(gpu=SimGpu(total_mib=12227, other_mib=1500, margin_mib=100), seed=0)
        budget = sim.vram_budget_mib()
        found = None
        for ctx in range(60000, 200000, 2048):
            proj = sim.project(base_cfg(ctx=ctx))
            if proj.device_self_mib <= budget:
                with_mtp = base_cfg(ctx=ctx, spec="draft-mtp", draft_n_max=2)
                plain = base_cfg(ctx=ctx)
                try:
                    sim.load(plain)
                    plain_ok = True
                except LoadFailed:
                    plain_ok = False
                try:
                    sim.load(with_mtp)
                    mtp_ok = True
                except LoadFailed as e:
                    assert e.kind == "oom"
                    mtp_ok = False
                if plain_ok and not mtp_ok:
                    found = ctx
                    break
        assert found is not None, "no ctx where MTP overhead alone tips the load over"

    def test_projections_monotonic(self):
        sim = SimBackend(seed=0)
        ctxs = [16384, 32768, 65536, 131072]
        totals = [sim.project(base_cfg(ctx=c)).device_self_mib for c in ctxs]
        assert totals == sorted(totals)
        assert len(set(totals)) == len(totals)
        ngls = [0, 8, 16, 24, 32]
        devices = [sim.project(base_cfg(ctx=32768, ngl=n)).device_self_mib for n in ngls]
        assert devices == sorted(devices)
        assert devices[0] < devices[-1]
        # Host gets the rest: offloading shrinks device and grows host.
        full = sim.project(base_cfg(ctx=32768, ngl=999))
        part = sim.project(base_cfg(ctx=32768, ngl=16))
        assert part.device_self_mib < full.device_self_mib
        assert part.host_self_mib > full.host_self_mib

    def test_projection_hides_mtp_overhead(self):
        sim = SimBackend(seed=0)
        plain = sim.project(base_cfg(ctx=32768))
        mtp = sim.project(base_cfg(ctx=32768, spec="draft-mtp", draft_n_max=2))
        assert mtp.device_self_mib == plain.device_self_mib

    def test_vram_budget(self):
        sim = SimBackend(seed=0)
        assert sim.vram_budget_mib() == 12227 - 1500 - 512


class TestSpeed:
    def test_threads_barely_matter_on_gpu(self):
        sim = SimBackend(seed=0, noise=0.0)
        sim.load(base_cfg(threads=4))
        slow = mean_tps(sim.measure("prose", reps=3))
        sim.load(base_cfg(threads=16))
        fast = mean_tps(sim.measure("prose", reps=3))
        assert abs(fast - slow) / slow < 0.01

    def test_threads_matter_offloaded(self):
        sim = SimBackend(seed=0, noise=0.0)
        sim.load(base_cfg(ngl=0, threads=8))
        best = mean_tps(sim.measure("prose", reps=3))
        sim.load(base_cfg(ngl=0, threads=4))
        narrow = mean_tps(sim.measure("prose", reps=3))
        sim.load(base_cfg(ngl=0, threads=16))
        wide = mean_tps(sim.measure("prose", reps=3))
        assert narrow < 0.8 * best
        assert wide < 0.9 * best

    def test_offload_is_much_slower(self):
        sim = SimBackend(seed=0, noise=0.0)
        sim.load(base_cfg(ngl=999))
        gpu = mean_tps(sim.measure("prose", reps=2))
        sim.load(base_cfg(ngl=0))
        cpu = mean_tps(sim.measure("prose", reps=2))
        assert cpu < 0.25 * gpu

    def test_code_rewards_ngram(self):
        sim = SimBackend(seed=0, noise=0.0)
        sim.load(base_cfg(spec="none"))
        plain = mean_tps(sim.measure("code", reps=2))
        sim.load(base_cfg(spec="ngram-mod", ngram_n_match=10))
        ngram = mean_tps(sim.measure("code", reps=2))
        assert ngram > 2.0 * plain

    def test_prose_rewards_mtp(self):
        sim = SimBackend(seed=0, noise=0.0)
        sim.load(base_cfg(spec="none"))
        plain = mean_tps(sim.measure("prose", reps=2))
        sim.load(base_cfg(spec="ngram-mod", ngram_n_match=10))
        ngram = mean_tps(sim.measure("prose", reps=2))
        sim.load(base_cfg(spec="draft-mtp", draft_n_max=2))
        mtp = mean_tps(sim.measure("prose", reps=2))
        assert mtp > 1.3 * plain
        assert mtp > ngram

    def test_mtp_n_max_tuning(self):
        sim = SimBackend(seed=0, noise=0.0)
        speeds = {}
        for n in (1, 2, 3, 4):
            sim.load(base_cfg(spec="draft-mtp", draft_n_max=n))
            speeds[n] = mean_tps(sim.measure("prose", reps=2))
        assert speeds[2] >= speeds[3] > 0
        assert speeds[2] > speeds[1]
        assert speeds[2] > speeds[4]

    def test_ngram_match_tuning(self):
        sim = SimBackend(seed=0, noise=0.0)
        speeds = {}
        for m in (4, 10, 24):
            sim.load(base_cfg(spec="ngram-mod", ngram_n_match=m))
            speeds[m] = mean_tps(sim.measure("code", reps=2))
        assert speeds[10] > speeds[4]
        assert speeds[10] > speeds[24]

    def test_acceptance_consistent(self):
        sim = SimBackend(seed=0, noise=0.0)
        sim.load(base_cfg(spec="draft-mtp", draft_n_max=2))
        [sample] = sim.measure("prose", reps=1)
        assert sample.draft_n
        assert sample.draft_accepted is not None
        assert 0 < sample.draft_accepted < sample.draft_n

    def test_prefill_grows_to_512_then_flattens(self):
        sim = SimBackend(seed=0, noise=0.0)
        prompts = {}
        for ub in (128, 256, 512, 1024):
            sim.load(base_cfg(ubatch=ub, batch=2048))
            [sample] = sim.measure("prefill", reps=1)
            prompts[ub] = sample.prompt_tps
        assert prompts[128] < prompts[256] < prompts[512]
        assert prompts[512] == pytest.approx(prompts[1024])

    def test_bigger_ctx_costs_decode(self):
        sim = SimBackend(seed=0, noise=0.0)
        sim.load(base_cfg(ctx=32768))
        small = mean_tps(sim.measure("prose", reps=2))
        sim.load(base_cfg(ctx=131072))
        big = mean_tps(sim.measure("prose", reps=2))
        assert big < small
        assert big > 0.9 * small  # ~1% per doubling, not a cliff

    def test_q4_cache_costs_decode(self):
        sim = SimBackend(seed=0, noise=0.0)
        sim.load(base_cfg(cache_k="q8_0", cache_v="q8_0"))
        ref = mean_tps(sim.measure("prose", reps=2))
        sim.load(base_cfg(cache_k="q4_0", cache_v="q4_0"))
        q4 = mean_tps(sim.measure("prose", reps=2))
        assert q4 == pytest.approx(ref * 0.97, rel=0.005)


class TestQualityPowerLifecycle:
    def test_q4_lowers_quality(self):
        good = SimBackend(seed=0)
        good.load(base_cfg(cache_k="q8_0", cache_v="q8_0"))
        bad = SimBackend(seed=0)
        bad.load(base_cfg(cache_k="q4_0", cache_v="q4_0"))
        assert bad.quality("tools", limit=200).rate < good.quality("tools", limit=200).rate

    def test_thinking_cap_lowers_reasoning(self):
        sim = SimBackend(seed=0)
        sim.load(base_cfg(reasoning_budget=None))
        full = sim.quality("reasoning", limit=200).rate
        sim.load(base_cfg(reasoning_budget=256))
        capped = sim.quality("reasoning", limit=200).rate
        assert capped < full

    def test_power_key_flips(self):
        sim = SimBackend(seed=0, power_changes_at=2)
        first = sim.power_key()
        assert first == "ac|best_performance|saver-off"
        sim.load(base_cfg())
        sim.measure("prose", reps=1)
        assert sim.power_key() == first
        sim.measure("prose", reps=1)
        assert sim.power_key() != first

    def test_power_stable_without_schedule(self):
        sim = SimBackend(seed=0)
        sim.load(base_cfg())
        key = sim.power_key()
        sim.measure("prose", reps=1)
        assert sim.power_key() == key

    def test_counters_log_and_lifecycle(self):
        sim = SimBackend(seed=0)
        assert sim.loads == sim.measures == sim.projections == 0
        assert sim.restored is False
        assert sim.should_stop() is False
        sim.project(base_cfg())
        sim.load(base_cfg())
        sim.measure("prose", reps=2)
        assert (sim.projections, sim.loads, sim.measures) == (1, 1, 1)
        assert sim.log
        sim.request_stop()
        assert sim.should_stop() is True
        sim.restore()
        assert sim.restored is True
