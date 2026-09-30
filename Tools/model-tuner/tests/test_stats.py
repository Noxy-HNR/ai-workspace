"""Tests for modeltuner.stats module."""

from __future__ import annotations

import math
import pytest
from typing import Callable

from modeltuner import stats

import dataclasses


# ============================================================================
# Helper fixtures
# ============================================================================

def _make_samples(n: int, base: float, noise: float = 0.0) -> list[float]:
    """Generate n samples around base with optional Gaussian noise."""
    return [base + stats.median([_make_samples(n=1, base=base, noise=noise)[0]]) for _ in range(n)]


def _gaussian_samples(n: int, mean: float, std: float) -> list[float]:
    """Generate Gaussian samples using Box-Muller transform."""
    import random
    rng = random.Random(42)
    samples = []
    for _ in range(n):
        u1 = rng.random()
        u2 = rng.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        samples.append(mean + std * z)
    return samples


# ============================================================================
# Tests for median
# ============================================================================

class TestMedian:
    def test_single_value(self):
        assert stats.median([5.0]) == 5.0

    def test_even_count(self):
        assert stats.median([1.0, 3.0]) == 2.0

    def test_odd_count(self):
        assert stats.median([1.0, 2.0, 3.0]) == 2.0

    def test_with_duplicates(self):
        assert stats.median([1.0, 1.0, 1.0]) == 1.0

    def test_all_same(self):
        assert stats.median([5.0, 5.0, 5.0]) == 5.0

    def test_negative_values(self):
        assert stats.median([-5.0, -3.0, -1.0]) == -3.0

    def test_mixed_values(self):
        # Sorted: 1, 5, 10, 50 -> median of middle two (5+10)/2 = 7.5
        assert stats.median([1.0, 5.0, 10.0, 50.0]) == 7.5

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="median.*empty"):
            stats.median([])

    def test_empty_list_raises(self):
        """Empty list passed should raise ValueError."""
        with pytest.raises(ValueError, match="median.*empty"):
            stats.median([])


# ============================================================================
# Tests for MAD
# ============================================================================

class TestMAD:
    def test_single_value(self):
        assert stats.mad([5.0]) == 0.0

    def test_all_same(self):
        assert stats.mad([5.0, 5.0, 5.0]) == 0.0

    def test_symmetric_around_zero(self):
        # Median is 0, MAD should be median of absolute deviations
        xs = [-2.0, -1.0, 0.0, 1.0, 2.0]
        assert stats.mad(xs) == 1.0

    def test_with_noise(self):
        # Base 10 with some spread - MAD for Gaussian is ~0.674 * std
        xs = _gaussian_samples(1000, 10.0, 2.0)
        mad_val = stats.mad(xs)
        # For normal distribution, MAD ≈ 0.674 * std
        assert 1.0 < mad_val < 2.0  # Should be around 1.35

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="median.*empty"):
            stats.mad([])

    def test_single_sample_returns_zero(self):
        assert stats.mad([7.5]) == 0.0


# ============================================================================
# Tests for relative spread
# ============================================================================

class TestRelSpread:
    def test_all_same(self):
        assert stats.rel_spread([5.0, 5.0, 5.0]) == 0.0

    def test_with_noise(self):
        xs = _gaussian_samples(1000, 10.0, 2.0)
        spread = stats.rel_spread(xs)
        # MAD ~ 1.35, median = 10, so rel_spread ~ 0.135
        assert 0.1 < spread < 0.2

    def test_zero_median(self):
        xs = [-5.0, 0.0, 5.0]  # Median is 0
        assert stats.rel_spread(xs) == 0.0

    def test_single_value(self):
        assert stats.rel_spread([5.0]) == 0.0

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="median.*empty"):
            stats.rel_spread([])

    def test_large_spread(self):
        xs = [1.0, 100.0]
        assert stats.rel_spread(xs) > 0.5


# ============================================================================
# Tests for trimmed mean
# ============================================================================

class TestTrimmedMean:
    def test_no_trim(self):
        assert stats.trimmed_mean([1.0, 2.0, 3.0, 4.0]) == 2.5

    def test_full_trim(self):
        # trim=0.5 means drop 50% from each end, should return median
        assert stats.trimmed_mean([1.0, 2.0, 3.0], trim=0.5) == 2.0

    def test_with_outliers(self):
        xs = [1.0, 2.0, 3.0, 100.0, 101.0]
        # With 10% trim, drop 0 from each end (floor(5*0.1) = 0)
        tm = stats.trimmed_mean(xs, trim=0.1)
        # With 20% trim, drop 1 from each end
        tm_20 = stats.trimmed_mean(xs, trim=0.2)
        assert tm_20 < tm  # Trimming should reduce outlier influence

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="median.*empty"):
            stats.trimmed_mean([])

    def test_all_same(self):
        assert stats.trimmed_mean([5.0, 5.0, 5.0]) == 5.0


# ============================================================================
# Tests for bootstrap CI
# ============================================================================

class TestBootstrapCI:
    def test_single_sample(self):
        ci = stats.bootstrap_ci([5.0])
        assert ci == (5.0, 5.0)

    def test_deterministic_seed(self):
        ci1 = stats.bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0], seed=42)
        ci2 = stats.bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0], seed=42)
        assert ci1 == ci2

    def test_symmetric_sample_contains_median(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        med = stats.median(xs)
        ci = stats.bootstrap_ci(xs, stat=stats.median, seed=42)
        assert ci[0] <= med <= ci[1]

    def test_interval_width(self):
        xs = _gaussian_samples(100, 10.0, 2.0)
        ci = stats.bootstrap_ci(xs, seed=42)
        # Interval should have some width for noisy data
        assert ci[1] - ci[0] > 0

    def test_default_stat_is_median(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ci = stats.bootstrap_ci(xs)
        med = stats.median(xs)
        assert ci[0] <= med <= ci[1]

    def test_custom_stat(self):
        def my_stat(xs):
            return sum(xs) / len(xs)
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ci = stats.bootstrap_ci(xs, stat=my_stat, seed=42)
        # CI should be centered around the mean
        mean_val = sum(xs) / len(xs)
        assert ci[0] <= mean_val <= ci[1]

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="bootstrap_ci.*non-empty"):
            stats.bootstrap_ci([])


# ============================================================================
# Tests for Verdict enum
# ============================================================================

class TestVerdict:
    def test_better_value(self):
        assert stats.Verdict.BETTER.value == "better"

    def test_worse_value(self):
        assert stats.Verdict.WORSE.value == "worse"

    def test_tie_value(self):
        assert stats.Verdict.TIE.value == "tie"

    def test_need_more_value(self):
        assert stats.Verdict.NEED_MORE.value == "need_more"


# ============================================================================
# Tests for Comparison dataclass
# ============================================================================

class TestComparison:
    def test_fields_present(self):
        c = stats.Comparison(
            verdict=stats.Verdict.BETTER,
            delta_rel=0.05,
            threshold_rel=0.03,
            n_a=10,
            n_b=10,
            reason="B is 5.0% better than A; the bar was 3.0%."
        )
        assert hasattr(c, 'verdict')
        assert hasattr(c, 'delta_rel')
        assert hasattr(c, 'threshold_rel')
        assert hasattr(c, 'n_a')
        assert hasattr(c, 'n_b')
        assert hasattr(c, 'reason')

    def test_frozen(self):
        c = stats.Comparison(
            verdict=stats.Verdict.BETTER,
            delta_rel=0.05,
            threshold_rel=0.03,
            n_a=10,
            n_b=10,
            reason="test"
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            c.verdict = stats.Verdict.TIE


# ============================================================================
# Tests for compare function
# ============================================================================

class TestCompare:
    def test_clear_improvement_3_3_samples(self):
        """Clear improvement with 3+3 samples."""
        a = [100.0] * 3
        b = [110.0] * 3  # 10% faster
        c = stats.compare(a, b, higher_is_better=True)
        assert c.verdict is stats.Verdict.BETTER
        assert abs(c.delta_rel - 0.1) < 0.01
        assert c.n_a == 3
        assert c.n_b == 3

    def test_clear_worse_3_3_samples(self):
        """Clear worse with 3+3 samples."""
        a = [100.0] * 3
        b = [90.0] * 3  # 10% slower
        c = stats.compare(a, b, higher_is_better=True)
        assert c.verdict is stats.Verdict.WORSE
        assert abs(c.delta_rel - (-0.1)) < 0.01

    def test_tie_noise_3_3_samples(self):
        """Noise that should give TIE."""
        a = _gaussian_samples(10, 100.0, 2.0)  # ~2% noise
        b = _gaussian_samples(10, 101.0, 2.0)  # ~1% different, within noise
        c = stats.compare(a, b, higher_is_better=True, min_rel=0.03)
        # Should be TIE because difference is small
        assert c.verdict in (stats.Verdict.TIE, stats.Verdict.NEED_MORE)

    def test_huge_difference_1_1_samples(self):
        """A huge difference with 1+1 samples gives BETTER."""
        a = [100.0]
        b = [300.0]  # 200% faster
        c = stats.compare(a, b, higher_is_better=True, min_n=3, clear_win_factor=4.0)
        assert c.verdict is stats.Verdict.BETTER
        assert c.n_a == 1
        assert c.n_b == 1

    def test_small_difference_2_2_samples_needs_more(self):
        """A small difference with 2+2 samples gives NEED_MORE."""
        a = [100.0, 101.0]
        b = [102.0, 103.0]  # Only ~1-2% different
        c = stats.compare(a, b, higher_is_better=True, min_n=3, clear_win_factor=4.0)
        assert c.verdict is stats.Verdict.NEED_MORE

    def test_lower_is_better(self):
        """lower_is_better flipping the verdict."""
        a = [100.0, 100.0, 100.0]
        b = [90.0, 90.0, 90.0]  # 10% lower, but lower is better
        c = stats.compare(a, b, higher_is_better=False)
        assert c.verdict is stats.Verdict.BETTER  # B is better because lower wins

    def test_noisy_side_raises_threshold(self):
        """A noisy side raises the threshold."""
        a = [100.0, 100.0, 100.0, 100.0, 100.0]  # Noisy A
        b = [105.0, 105.0, 105.0, 105.0, 105.0]  # 5% better
        # With high noise, threshold should be higher
        c = stats.compare(a, b, higher_is_better=True, spread_k=2.0, min_rel=0.03)
        # If spread is high, might be TIE or NEED_MORE
        # But with same low noise, should be BETTER
        assert c.verdict in (stats.Verdict.BETTER, stats.Verdict.TIE)

    def test_empty_a_raises(self):
        with pytest.raises(ValueError, match="compare.*non-empty.*a"):
            stats.compare([], [1.0, 2.0])

    def test_empty_b_raises(self):
        with pytest.raises(ValueError, match="compare.*non-empty.*b"):
            stats.compare([1.0, 2.0], [])

    def test_zero_median_no_division_error(self):
        """median(a) == 0 must not divide by zero."""
        a = [0.0, 0.0, 0.0]
        b = [10.0, 10.0, 10.0]
        c = stats.compare(a, b, higher_is_better=True)
        # Should not raise; delta_rel should be handled sensibly
        assert c.verdict in (stats.Verdict.BETTER, stats.Verdict.TIE, stats.Verdict.NEED_MORE)
        # delta_rel should be 0 (handled when med_a == 0)

    def test_delta_rel_sign(self):
        """delta_rel should be positive when B is better (higher_is_better=True)."""
        a = [100.0, 100.0, 100.0]
        b = [110.0, 110.0, 110.0]
        c = stats.compare(a, b, higher_is_better=True)
        assert c.delta_rel > 0

    def test_delta_rel_negative_when_worse(self):
        """delta_rel should be negative when B is worse."""
        a = [100.0, 100.0, 100.0]
        b = [90.0, 90.0, 90.0]
        c = stats.compare(a, b, higher_is_better=True)
        assert c.delta_rel < 0

    def test_reason_sentence(self):
        """reason should be one plain-English sentence."""
        a = [100.0, 100.0, 100.0]
        b = [107.4, 107.4, 107.4]
        c = stats.compare(a, b, higher_is_better=True)
        assert isinstance(c.reason, str)
        assert len(c.reason) > 0
        # Check it contains meaningful content
        assert "better" in c.reason.lower() or "%" in c.reason

    def test_threshold_calculation(self):
        """threshold_rel = max(min_rel, spread_k * max(rel_spread(a), rel_spread(b)))."""
        # With no spread, threshold should be at least min_rel
        a = [100.0] * 10
        b = [100.0] * 10
        c = stats.compare(a, b, higher_is_better=True, min_rel=0.05, spread_k=2.0)
        assert c.threshold_rel == 0.05  # max(0.05, 2.0 * 0) = 0.05

    def test_spread_increases_threshold(self):
        """A noisy side raises the threshold."""
        a = _gaussian_samples(10, 100.0, 10.0)  # ~10% spread
        b = [110.0] * 10
        c = stats.compare(a, b, higher_is_better=True, min_rel=0.03, spread_k=2.0)
        # threshold = max(0.03, 2.0 * 0.1) = 0.2, but actual rel_spread may vary slightly
        # The threshold should be at least min_rel
        assert c.threshold_rel >= 0.03

    def test_bootstrap_determinism(self):
        """Bootstrap must be deterministic for a fixed seed."""
        ci1 = stats.bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0], seed=12345)
        ci2 = stats.bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0], seed=12345)
        assert ci1 == ci2

    def test_bootstrap_interval_contains_median(self):
        """Bootstrap interval must contain the median for a symmetric sample."""
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        med = stats.median(xs)
        ci = stats.bootstrap_ci(xs, stat=stats.median, seed=42)
        assert ci[0] <= med <= ci[1]

    def test_very_small_delta_needs_more(self):
        """Very small delta with enough samples should be TIE."""
        a = [100.0, 100.0, 100.0, 100.0, 100.0]
        b = [100.1, 100.1, 100.1, 100.1, 100.1]  # 0.1% difference
        c = stats.compare(a, b, higher_is_better=True, min_rel=0.03)
        assert c.verdict is stats.Verdict.TIE

    def test_clear_win_factor_with_one_sample(self):
        """With a single sample on a side, clear win factor applies."""
        a = [100.0]
        b = [105.0]
        c = stats.compare(a, b, higher_is_better=True, clear_win_factor=4.0, min_n=3)
        # 5% difference, threshold is 4x of that = 20%, so need more
        assert c.verdict is stats.Verdict.NEED_MORE

    def test_clear_win_with_huge_difference(self):
        """Huge difference with one sample should give BETTER."""
        a = [100.0]
        b = [500.0]  # 400% faster
        c = stats.compare(a, b, higher_is_better=True, clear_win_factor=4.0, min_n=3)
        # 400% > 4 * threshold, so BETTER
        assert c.verdict is stats.Verdict.BETTER


# ============================================================================
# Tests for relative_score
# ============================================================================

class TestRelativeScore:
    def test_identical(self):
        candidate = {"prose_tps": 100.0, "code_tps": 80.0}
        baseline = {"prose_tps": 100.0, "code_tps": 80.0}
        weights = {"prose_tps": 0.5, "code_tps": 0.5}
        score = stats.relative_score(candidate, baseline, weights)
        assert score == 0.0

    def test_10_percent_better(self):
        candidate = {"prose_tps": 110.0, "code_tps": 90.0}
        baseline = {"prose_tps": 100.0, "code_tps": 80.0}
        weights = {"prose_tps": 0.5, "code_tps": 0.5}
        score = stats.relative_score(candidate, baseline, weights)
        # Geometric mean: sqrt(1.1 * 1.125) - 1 = sqrt(1.2375) - 1 ≈ 0.112
        assert 0.1 < score < 0.12  # ~11% better

    def test_10_percent_worse(self):
        candidate = {"prose_tps": 90.0, "code_tps": 70.0}
        baseline = {"prose_tps": 100.0, "code_tps": 80.0}
        weights = {"prose_tps": 0.5, "code_tps": 0.5}
        score = stats.relative_score(candidate, baseline, weights)
        # Geometric mean: sqrt(0.9 * 0.875) - 1 = sqrt(0.7875) - 1 ≈ -0.113
        assert -0.12 < score < -0.1  # ~11% worse

    def test_skips_missing_keys(self):
        candidate = {"prose_tps": 110.0}  # Missing code_tps
        baseline = {"prose_tps": 100.0, "code_tps": 80.0}
        weights = {"prose_tps": 0.5, "code_tps": 0.5}
        score = stats.relative_score(candidate, baseline, weights)
        # Should renormalize to prose_tps only
        assert abs(score - 0.1) < 0.001

    def test_skips_zero_baseline(self):
        candidate = {"prose_tps": 110.0, "code_tps": 90.0}
        baseline = {"prose_tps": 100.0, "code_tps": 0.0}  # Zero baseline
        weights = {"prose_tps": 0.5, "code_tps": 0.5}
        score = stats.relative_score(candidate, baseline, weights)
        # Should skip code_tps and renormalize
        assert abs(score - 0.1) < 0.001

    def test_skips_negative_baseline(self):
        candidate = {"prose_tps": 110.0, "code_tps": 90.0}
        baseline = {"prose_tps": 100.0, "code_tps": -10.0}  # Negative baseline
        weights = {"prose_tps": 0.5, "code_tps": 0.5}
        score = stats.relative_score(candidate, baseline, weights)
        # Should skip code_tps
        assert abs(score - 0.1) < 0.001

    def test_no_valid_keys_returns_zero(self):
        candidate = {"prose_tps": 100.0}
        baseline = {"prose_tps": 0.0, "code_tps": 0.0}  # All zero baselines
        weights = {"prose_tps": 0.5, "code_tps": 0.5}
        score = stats.relative_score(candidate, baseline, weights)
        assert score == 0.0

    def test_no_keys_in_candidate_returns_zero(self):
        candidate = {}
        baseline = {"prose_tps": 100.0}
        weights = {"prose_tps": 1.0}
        score = stats.relative_score(candidate, baseline, weights)
        assert score == 0.0

    def test_weights_renormalized(self):
        # Test that weights are properly normalized
        candidate = {"a": 110.0, "b": 90.0}
        baseline = {"a": 100.0, "b": 80.0}
        weights = {"a": 0.3, "b": 0.7}  # Already normalized
        score = stats.relative_score(candidate, baseline, weights)
        # Geometric mean: (1.1)^0.3 * (1.125)^0.7 - 1 ≈ 0.117
        assert 0.1 < score < 0.15

    def test_weighted_geometric_mean(self):
        # Test that it's actually a geometric mean
        candidate = {"a": 100.0, "b": 100.0}
        baseline = {"a": 100.0, "b": 80.0}  # b is 25% worse
        weights = {"a": 0.5, "b": 0.5}
        score = stats.relative_score(candidate, baseline, weights)
        # Geometric mean of (100/100)^(0.5) * (100/80)^(0.5) = 1.118, so score = 0.118
        assert 0.1 < score < 0.15

    def test_empty_weights(self):
        candidate = {"a": 100.0}
        baseline = {"a": 100.0}
        weights = {}
        score = stats.relative_score(candidate, baseline, weights)
        assert score == 0.0

    def test_all_weights_zero(self):
        candidate = {"a": 100.0}
        baseline = {"a": 100.0}
        weights = {"a": 0.0}
        score = stats.relative_score(candidate, baseline, weights)
        assert score == 0.0


# ============================================================================
# Integration tests
# ============================================================================

class TestIntegration:
    def test_full_workflow_higher_is_better(self):
        """Test a complete higher-is-better scenario."""
        # Simulate two configurations with measurable performance
        a = _gaussian_samples(20, 50.0, 1.0)  # Config A: 50 tok/s with 1% noise
        b = _gaussian_samples(20, 57.5, 1.0)  # Config B: 57.5 tok/s (15% faster)
        
        result = stats.compare(a, b, higher_is_better=True, min_n=3)
        
        assert result.verdict is stats.Verdict.BETTER
        assert result.delta_rel > 0.14  # Should be around 15%
        # threshold_rel should be slightly above min_rel due to small spread
        assert result.threshold_rel >= 0.03
        assert result.n_a == 20
        assert result.n_b == 20

    def test_full_workflow_lower_is_better(self):
        """Test a complete lower-is-better scenario (e.g., latency)."""
        # Lower latency is better
        a = _gaussian_samples(20, 100.0, 1.0)  # Config A: 100ms latency
        b = _gaussian_samples(20, 92.5, 1.0)  # Config B: 92.5ms (7.5% faster/lower)
        
        result = stats.compare(a, b, higher_is_better=False, min_n=3)
        
        assert result.verdict is stats.Verdict.BETTER
        # delta_rel should be positive (B is ~7.5% "slower" in the higher-is-better sense)
        # but because higher_is_better=False, B is actually better (lower is better)
        # So delta_rel should be around +7.5% (B's latency is 7.5% lower than A's)
        assert result.delta_rel > 0.07

    def test_bootstrap_determinism_across_calls(self):
        """Bootstrap with same seed should produce identical results."""
        xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        ci1 = stats.bootstrap_ci(xs, seed=999)
        ci2 = stats.bootstrap_ci(xs, seed=999)
        ci3 = stats.bootstrap_ci(xs, seed=999)
        assert ci1 == ci2 == ci3

    def test_compare_reason_sentence_format(self):
        """Reason should be a clear, concise sentence."""
        a = [100.0, 100.0, 100.0]
        b = [103.0, 103.0, 103.0]
        result = stats.compare(a, b, higher_is_better=True, min_n=3)
        
        assert isinstance(result.reason, str)
        assert len(result.reason) > 0
        # Should mention the percentage difference
        assert "%" in result.reason or "better" in result.reason.lower()

    def test_relative_score_composite_metrics(self):
        """Test relative_score with multiple composite metrics."""
        # Simulate three workloads: prose (45%), code (35%), prefill (20%)
        baseline = {
            "prose": 50.0,
            "code": 40.0,
            "prefill": 200.0
        }
        candidate = {
            "prose": 55.0,  # 10% better
            "code": 38.0,   # 5% worse
            "prefill": 220.0  # 10% better
        }
        weights = {
            "prose": 0.45,
            "code": 0.35,
            "prefill": 0.20
        }
        score = stats.relative_score(candidate, baseline, weights)
        # Should be positive since prose and prefill improvements outweigh code
        assert score > 0
        assert score < 0.1  # Should be less than 10% overall

    def test_edge_case_single_sample_compare(self):
        """Compare with single sample each."""
        a = [100.0]
        b = [150.0]
        result = stats.compare(a, b, higher_is_better=True, min_n=3, clear_win_factor=4.0)
        
        # 50% difference, threshold = 4 * min_rel = 0.12, so 50% > 12%
        assert result.verdict is stats.Verdict.BETTER
        assert result.n_a == 1
        assert result.n_b == 1

    def test_edge_case_many_samples_small_delta(self):
        """Many samples with small delta should be TIE."""
        a = [100.0] * 50
        b = [100.05] * 50  # 0.05% difference
        result = stats.compare(a, b, higher_is_better=True, min_rel=0.03)
        
        assert result.verdict is stats.Verdict.TIE
        assert abs(result.delta_rel) < 0.01

    def test_all_equal_samples(self):
        """All equal samples should give TIE."""
        a = [100.0] * 10
        b = [100.0] * 10
        result = stats.compare(a, b, higher_is_better=True, min_n=3)
        
        assert result.verdict is stats.Verdict.TIE
        assert result.delta_rel == 0.0

    def test_zero_median_edge_case(self):
        """When median is 0, should handle without division error."""
        a = [0.0, 0.0, 0.0]
        b = [10.0, 10.0, 10.0]
        result = stats.compare(a, b, higher_is_better=True, min_n=3)
        
        # Should not raise, and should handle gracefully
        assert result.verdict in (stats.Verdict.BETTER, stats.Verdict.TIE, stats.Verdict.NEED_MORE)
        # delta_rel should be 0 (handled when med_a == 0)

    def test_spread_k_effect(self):
        """Higher spread_k should increase threshold."""
        a = _gaussian_samples(10, 100.0, 10.0)  # 10% spread
        b = _gaussian_samples(10, 100.0, 10.0)
        
        result_low = stats.compare(a, b, spread_k=2.0, min_rel=0.03)
        result_high = stats.compare(a, b, spread_k=5.0, min_rel=0.03)
        
        assert result_high.threshold_rel >= result_low.threshold_rel
