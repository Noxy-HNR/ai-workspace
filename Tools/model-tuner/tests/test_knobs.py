"""Tests for modeltuner.knobs module."""

from __future__ import annotations

import pytest

from modeltuner import knobs


# ============================================================================
# Real model settings (from models.json)
# ============================================================================

SETTINGS_9B = {
    "contextSize": 139264,
    "gpuLayers": 999,
    "threads": 8,
    "batchSize": 1024,
    "ubatchSize": 512,
    "cacheTypeK": "q8_0",
    "cacheTypeV": "q8_0",
    "reasoningEffort": "default",
    "extraArgs": "--spec-type draft-mtp,ngram-mod --spec-draft-n-max 2 --load-mode none --cache-ram 0 --reasoning-budget 4096",
}

SETTINGS_BONSAI = {
    "contextSize": 147456,
    "threads": 8,
    "batchSize": 2048,
    "ubatchSize": 512,
    "cacheTypeK": "f16",
    "cacheTypeV": "f16",
    "reasoningEffort": "medium",
    "extraArgs": "--spec-type ngram-mod --spec-ngram-mod-n-match 8 --load-mode none --cache-ram 0",
    "serverBinary": "Tools/llama-prism/llama-server.exe",
}

SETTINGS_MEDGEMMA = {
    "contextSize": 40960,
    "gpuLayers": 37,
    "threads": 20,
    "batchSize": 2048,
    "ubatchSize": 512,
    "cacheTypeK": "f16",
    "cacheTypeV": "f16",
    "reasoningEffort": "default",
    "extraArgs": "--spec-type ngram-mod --spec-ngram-mod-n-match 8",
}


# ============================================================================
# Tests for from_settings with real models
# ============================================================================

class TestFromSettings:
    """Test from_settings on each real model's settings."""

    def test_from_settings_9b(self):
        cfg = knobs.from_settings(SETTINGS_9B)
        assert cfg.ctx == 139264
        assert cfg.ngl == 999
        assert cfg.threads == 8
        assert cfg.batch == 1024
        assert cfg.ubatch == 512
        assert cfg.cache_k == "q8_0"
        assert cfg.cache_v == "q8_0"
        assert cfg.reasoning_effort == "default"
        assert cfg.spec == "draft-mtp,ngram-mod"
        assert cfg.draft_n_max == 2
        assert cfg.reasoning_budget == 4096
        assert cfg.threads_batch is None
        assert cfg.n_cpu_moe is None

    def test_from_settings_bonsai(self):
        cfg = knobs.from_settings(SETTINGS_BONSAI)
        assert cfg.ctx == 147456
        assert cfg.ngl == 999
        assert cfg.threads == 8
        assert cfg.batch == 2048
        assert cfg.ubatch == 512
        assert cfg.cache_k == "f16"
        assert cfg.cache_v == "f16"
        assert cfg.reasoning_effort == "medium"
        assert cfg.spec == "ngram-mod"
        assert cfg.ngram_n_match == 8
        # Enforced args are parsed from extraArgs, so passthrough is empty after parsing
        # They get re-added by to_settings
        assert cfg.passthrough == ()

    def test_from_settings_medgemma(self):
        cfg = knobs.from_settings(SETTINGS_MEDGEMMA)
        assert cfg.ctx == 40960
        assert cfg.ngl == 37
        assert cfg.threads == 20
        assert cfg.batch == 2048
        assert cfg.ubatch == 512
        assert cfg.cache_k == "f16"
        assert cfg.cache_v == "f16"
        assert cfg.spec == "ngram-mod"
        assert cfg.ngram_n_match == 8


# ============================================================================
# Tests for to_settings round-trip preserving unknown keys
# ============================================================================

class TestToSettingsRoundTrip:
    """Test to_settings(from_settings(x), x) preserves every key the tuner does not own."""

    def test_9b_round_trip(self):
        base = dict(SETTINGS_9B)
        base["ropeFreqBase"] = 1000
        base["ropeFreqScale"] = 500
        cfg = knobs.from_settings(base)
        result = knobs.to_settings(cfg, base)
        for key in base:
            assert key in result
        assert result["contextSize"] == 139264
        assert result["extraArgs"] == "--spec-type draft-mtp,ngram-mod --spec-draft-n-max 2 --load-mode none --cache-ram 0 --reasoning-budget 4096"
        assert result["ropeFreqBase"] == 1000
        assert result["ropeFreqScale"] == 500

    def test_bonsai_round_trip(self):
        base = dict(SETTINGS_BONSAI)
        base["loraFile"] = "models/bonsai-7b-q4_0.gguf"
        base["ropeFreqBase"] = 500
        cfg = knobs.from_settings(base)
        result = knobs.to_settings(cfg, base)
        assert result["serverBinary"] == "Tools/llama-prism/llama-server.exe"
        assert result["loraFile"] == "models/bonsai-7b-q4_0.gguf"
        assert result["ropeFreqBase"] == 500

    def test_medgemma_round_trip(self):
        base = dict(SETTINGS_MEDGEMMA)
        base["loraScale"] = 1.5
        cfg = knobs.from_settings(base)
        result = knobs.to_settings(cfg, base)
        assert result["loraScale"] == 1.5


# ============================================================================
# Tests for parse_extra_args
# ============================================================================

class TestParseExtraArgs:
    """Test parse_extra_args for every typed flag and its aliases."""

    def test_spec_type_flag(self):
        typed, rest = knobs.parse_extra_args("--spec-type ngram-mod")
        assert typed["spec"] == "ngram-mod"
        assert rest == ()

    def test_spec_draft_n_max_flag(self):
        typed, rest = knobs.parse_extra_args("--spec-draft-n-max 5")
        assert typed["draft_n_max"] == 5
        assert rest == ()

    def test_draft_max_alias(self):
        typed, rest = knobs.parse_extra_args("--draft-max 10")
        assert typed["draft_n_max"] == 10
        assert rest == ()

    def test_spec_ngram_mod_n_match_flag(self):
        typed, rest = knobs.parse_extra_args("--spec-ngram-mod-n-match 16")
        assert typed["ngram_n_match"] == 16
        assert rest == ()

    def test_reasoning_budget_flag(self):
        typed, rest = knobs.parse_extra_args("--reasoning-budget 2048")
        assert typed["reasoning_budget"] == 2048
        assert rest == ()

    def test_threads_batch_alias_tb(self):
        typed, rest = knobs.parse_extra_args("-tb 32")
        assert typed["threads_batch"] == 32
        assert rest == ()

    def test_threads_batch_flag_threads_batch(self):
        typed, rest = knobs.parse_extra_args("--threads-batch 64")
        assert typed["threads_batch"] == 64
        assert rest == ()

    def test_n_cpu_moe_alias_ncmpoe(self):
        typed, rest = knobs.parse_extra_args("-ncmoe 4")
        assert typed["n_cpu_moe"] == 4
        assert rest == ()

    def test_n_cpu_moe_flag_n_cpu_moe(self):
        typed, rest = knobs.parse_extra_args("--n-cpu-moe 8")
        assert typed["n_cpu_moe"] == 8
        assert rest == ()

    def test_flag_value_form(self):
        typed, rest = knobs.parse_extra_args("--spec-type=ngram-mod")
        assert typed["spec"] == "ngram-mod"
        assert rest == ()

    def test_reasoning_budget_negative_one_unlimited(self):
        typed, rest = knobs.parse_extra_args("--reasoning-budget -1")
        assert typed["reasoning_budget"] is None
        assert rest == ()

    def test_multiple_flags_in_order(self):
        args = "--spec-type ngram-mod --spec-draft-n-max 3 --threads-batch 16 --reasoning-budget 4096"
        typed, rest = knobs.parse_extra_args(args)
        assert typed["spec"] == "ngram-mod"
        assert typed["draft_n_max"] == 3
        assert typed["threads_batch"] == 16
        assert typed["reasoning_budget"] == 4096
        assert rest == ()

    def test_missing_value_raises(self):
        with pytest.raises(ValueError, match="has no value"):
            knobs.parse_extra_args("--spec-type")

    def test_non_numeric_value_raises(self):
        with pytest.raises(ValueError, match="bad value"):
            knobs.parse_extra_args("--spec-draft-n-max abc")

    def test_qualified_value_raises(self):
        with pytest.raises(ValueError, match="bad value"):
            knobs.parse_extra_args("--spec-draft-n-max=abc")


# ============================================================================
# Tests for enforced flags round-trip
# ============================================================================

class TestEnforcedFlags:
    """Test that --load-mode none --cache-ram 0 appears exactly once after round trip."""

    def test_round_trip_preserves_enforced(self):
        cfg = knobs.from_settings(SETTINGS_9B)
        result = knobs.to_settings(cfg, SETTINGS_9B)
        extra_args = result["extraArgs"]
        assert extra_args.count("--load-mode") == 1
        assert extra_args.count("--cache-ram") == 1
        assert "--load-mode none" in extra_args
        assert "--cache-ram 0" in extra_args

    def test_round_trip_with_no_enforced_in_input(self):
        settings = dict(SETTINGS_9B)
        settings["extraArgs"] = "--spec-type draft-mtp,ngram-mod --spec-draft-n-max 2"
        cfg = knobs.from_settings(settings)
        result = knobs.to_settings(cfg, settings)
        extra_args = result["extraArgs"]
        assert extra_args.count("--load-mode") == 1
        assert extra_args.count("--cache-ram") == 1

    def test_round_trip_with_duplicate_enforced_in_input(self):
        settings = dict(SETTINGS_9B)
        settings["extraArgs"] = "--spec-type draft-mtp,ngram-mod --spec-draft-n-max 2 --load-mode none --cache-ram 0 --load-mode none --cache-ram 0"
        cfg = knobs.from_settings(settings)
        result = knobs.to_settings(cfg, settings)
        extra_args = result["extraArgs"]
        assert extra_args.count("--load-mode") == 1
        assert extra_args.count("--cache-ram") == 1

    def test_enforced_args_appear_at_correct_position(self):
        cfg = knobs.from_settings(SETTINGS_9B)
        result = knobs.to_settings(cfg, SETTINGS_9B)
        extra_args = result["extraArgs"]
        assert "--spec-type draft-mtp,ngram-mod" in extra_args
        assert "--spec-draft-n-max 2" in extra_args
        assert "--load-mode none" in extra_args
        assert "--cache-ram 0" in extra_args
        assert "--reasoning-budget 4096" in extra_args


# ============================================================================
# Tests for ValueError on owned flags in extraArgs
# ============================================================================

class TestOwnedFlagsValueError:
    """Test that owned flags in extraArgs raise ValueError."""

    def test_ctx_size_owns_extraArgs(self):
        with pytest.raises(ValueError, match="belongs in the model's own settings"):
            knobs.parse_extra_args("-c 32768")

    def test_n_gpu_layers_owns_extraArgs(self):
        with pytest.raises(ValueError, match="belongs in the model's own settings"):
            knobs.parse_extra_args("-ngl 64")

    def test_gpu_layers_owns_extraArgs(self):
        with pytest.raises(ValueError, match="belongs in the model's own settings"):
            knobs.parse_extra_args("--gpu-layers 64")

    def test_threads_owns_extraArgs(self):
        with pytest.raises(ValueError, match="belongs in the model's own settings"):
            knobs.parse_extra_args("-t 16")

    def test_batch_size_owns_extraArgs(self):
        with pytest.raises(ValueError, match="belongs in the model's own settings"):
            knobs.parse_extra_args("-b 4096")

    def test_ubatch_size_owns_extraArgs(self):
        with pytest.raises(ValueError, match="belongs in the model's own settings"):
            knobs.parse_extra_args("-ub 1024")

    def test_cache_type_k_owns_extraArgs(self):
        with pytest.raises(ValueError, match="belongs in the model's own settings"):
            knobs.parse_extra_args("-ctk f16")

    def test_cache_type_v_owns_extraArgs(self):
        with pytest.raises(ValueError, match="belongs in the model's own settings"):
            knobs.parse_extra_args("-ctv q8_0")

    def test_flash_attn_owns_extraArgs(self):
        with pytest.raises(ValueError, match="belongs in the model's own settings"):
            knobs.parse_extra_args("-fa off")

    def test_reasoning_effort_owns_extraArgs(self):
        with pytest.raises(ValueError, match="belongs in the model's own settings"):
            knobs.parse_extra_args("--reasoning-effort high")


# ============================================================================
# Tests for Config.key() stability
# ============================================================================

class TestConfigKey:
    """Test Config.key() produces stable, deterministic output."""

    def test_key_is_deterministic(self):
        cfg1 = knobs.Config(ctx=4096, ngl=999, threads=8)
        cfg2 = knobs.Config(ctx=4096, ngl=999, threads=8)
        assert cfg1.key() == cfg2.key()

    def test_key_includes_passthrough(self):
        cfg1 = knobs.Config(ctx=4096, passthrough=("extra", "flag"))
        cfg2 = knobs.Config(ctx=4096, passthrough=())
        assert cfg1.key() != cfg2.key()

    def test_key_order_is_consistent(self):
        cfg1 = knobs.Config(ctx=4096, spec="ngram-mod", ngram_n_match=8)
        cfg2 = knobs.Config(spec="ngram-mod", ctx=4096, ngram_n_match=8)
        assert cfg1.key() == cfg2.key()

    def test_key_uses_space_for_passthrough(self):
        cfg = knobs.Config(ctx=4096, passthrough=("foo bar", "baz"))
        assert "foo bar" in cfg.key()


# ============================================================================
# Tests for describe() wording
# ============================================================================

class TestConfigDescribe:
    """Test Config.describe() produces correct human-readable output."""

    def test_describe_basic(self):
        cfg = knobs.Config(ctx=4096, ngl=999)
        desc = cfg.describe()
        assert "ctx 4096" in desc
        assert "all layers on GPU" in desc

    def test_describe_partial_gpu(self):
        cfg = knobs.Config(ctx=4096, ngl=37)
        desc = cfg.describe()
        assert "37 layers on GPU" in desc

    def test_describe_different_cache(self):
        cfg = knobs.Config(ctx=4096, cache_k="f16", cache_v="q8_0")
        desc = cfg.describe()
        assert "K f16/V q8_0" in desc

    def test_describe_spec_with_mtp(self):
        cfg = knobs.Config(ctx=4096, spec="draft-mtp,ngram-mod", draft_n_max=2, ngram_n_match=8)
        desc = cfg.describe()
        assert "MTP+ngram" in desc
        assert "n_max 2" in desc
        assert "match 8" in desc

    def test_describe_spec_ngram_only(self):
        cfg = knobs.Config(ctx=4096, spec="ngram-mod", ngram_n_match=8)
        desc = cfg.describe()
        assert "ngram (match 8)" in desc

    def test_describe_batch(self):
        cfg = knobs.Config(ctx=4096, batch=1024, ubatch=512)
        desc = cfg.describe()
        assert "batch 1024/512" in desc

    def test_describe_threads(self):
        cfg = knobs.Config(ctx=4096, threads=16, threads_batch=32)
        desc = cfg.describe()
        assert "16 threads" in desc
        assert "+32 batch" in desc

    def test_describe_threads_without_threads_batch(self):
        cfg = knobs.Config(ctx=4096, threads=16)
        desc = cfg.describe()
        assert "16 threads" in desc
        assert "+32" not in desc

    def test_describe_reasoning_budget(self):
        cfg = knobs.Config(ctx=4096, reasoning_budget=4096)
        desc = cfg.describe()
        assert "thinking cap 4096" in desc

    def test_describe_reasoning_effort(self):
        cfg = knobs.Config(ctx=4096, reasoning_effort="high")
        desc = cfg.describe()
        assert "effort high" in desc

    def test_describe_cpu_moe(self):
        cfg = knobs.Config(ctx=4096, n_cpu_moe=4)
        desc = cfg.describe()
        assert "4 MoE layers on CPU" in desc

    def test_describe_no_spec(self):
        cfg = knobs.Config(ctx=4096, spec="none")
        desc = cfg.describe()
        assert "no speculation" in desc


# ============================================================================
# Tests for diff()
# ============================================================================

class TestConfigDiff:
    """Test Config.diff() returns correct differing fields."""

    def test_no_diff(self):
        a = knobs.Config(ctx=4096, ngl=999)
        b = knobs.Config(ctx=4096, ngl=999)
        assert knobs.diff(a, b) == {}

    def test_single_field_diff(self):
        a = knobs.Config(ctx=4096, ngl=999)
        b = knobs.Config(ctx=8192, ngl=999)
        d = knobs.diff(a, b)
        assert len(d) == 1
        assert "ctx" in d
        assert d["ctx"] == (4096, 8192)

    def test_multiple_field_diff(self):
        a = knobs.Config(ctx=4096, ngl=37, threads=8)
        b = knobs.Config(ctx=8192, ngl=999, threads=16)
        d = knobs.diff(a, b)
        assert len(d) == 3
        assert "ctx" in d
        assert "ngl" in d
        assert "threads" in d
        assert d["ctx"] == (4096, 8192)
        assert d["ngl"] == (37, 999)
        assert d["threads"] == (8, 16)

    def test_diff_ignores_passthrough_for_comparison(self):
        a = knobs.Config(ctx=4096, passthrough=("a", "b"))
        b = knobs.Config(ctx=4096, passthrough=("c", "d"))
        # diff() compares passthrough as a single string, so different passthroughs are considered different
        d = knobs.diff(a, b)
        assert d == {"passthrough": (("a", "b"), ("c", "d"))}

    def test_diff_with_none_values(self):
        a = knobs.Config(ctx=4096, draft_n_max=None, threads_batch=None)
        b = knobs.Config(ctx=4096, draft_n_max=2, threads_batch=16)
        d = knobs.diff(a, b)
        assert len(d) == 2
        assert d["draft_n_max"] == (None, 2)
        assert d["threads_batch"] == (None, 16)


# ============================================================================
# Per-tensor placement (n_cpu_ffn -> -ot)
# ============================================================================

class TestFfnOverride:
    def test_round_trip_through_extra_args(self):
        cfg = knobs.from_settings(dict(SETTINGS_MEDGEMMA, gpuLayers=999))
        cfg = cfg.with_(n_cpu_ffn=30)
        back = knobs.from_settings(knobs.to_settings(cfg))
        assert back == cfg
        assert back.passthrough == ()
        assert "FFN of 30 layers on CPU" in back.describe()

    def test_value_selects_exactly_the_first_n_blocks(self):
        import re
        pattern, target = knobs.ffn_override(12).split("=")
        assert target == "CPU"
        assert re.search(pattern, "blk.11.ffn_down.weight")
        assert re.search(pattern, "blk.0.ffn_gate.weight")
        assert not re.search(pattern, "blk.12.ffn_up.weight")
        assert not re.search(pattern, "blk.1.attn_q.weight")

    def test_hand_written_rules_stay_in_passthrough(self):
        rule = r"blk\.(3|4)\.ffn_(up|gate|down)\.weight=CPU"
        cfg = knobs.from_settings({"contextSize": 8192, "extraArgs": f"-ot {rule}"})
        assert cfg.n_cpu_ffn is None
        assert cfg.passthrough == ("-ot", rule)

    def test_fit_params_gets_the_rule(self):
        args = knobs.fit_params_args(knobs.Config(ctx=8192, n_cpu_ffn=4))
        assert args[args.index("-ot") + 1] == knobs.ffn_override(4)

    def test_key_of_configs_without_it_is_unchanged(self):
        assert "n_cpu_ffn" not in knobs.Config(ctx=8192).key()
        assert "n_cpu_ffn=5" in knobs.Config(ctx=8192, n_cpu_ffn=5).key()
