"""Robust statistics and the noise-aware "is B really better than A" decision.

Written by OxCoder-9B running locally (llama.cpp on this PC) through OpenCode, from the spec in DESIGN.md;
reviewed against the strategy tests and an independent implementation.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Mapping, Sequence


def median(xs: Sequence[float]) -> float:
    """Median of a list of numbers. Raises ValueError on empty input."""
    if not xs:
        raise ValueError("median() arg is an empty sequence")
    sorted_xs = sorted(xs)
    n = len(sorted_xs)
    mid = n // 2
    if n % 2 == 1:
        return sorted_xs[mid]
    else:
        return (sorted_xs[mid - 1] + sorted_xs[mid]) / 2


def mad(xs: Sequence[float]) -> float:
    """Median absolute deviation from the median (unscaled)."""
    med = median(xs)
    deviations = [abs(x - med) for x in xs]
    return median(deviations)


def rel_spread(xs: Sequence[float]) -> float:
    """mad(xs) / |median(xs)|, 0.0 if the median is 0."""
    med = median(xs)
    if med == 0:
        return 0.0
    return mad(xs) / abs(med)


def trimmed_mean(xs: Sequence[float], trim: float = 0.1) -> float:
    """Trimmed mean: drop floor(n*trim) from each end, then average."""
    n = len(xs)
    drop = int(n * trim)
    if drop * 2 >= n:
        return median(xs)
    sorted_xs = sorted(xs)
    return sum(sorted_xs[drop:n - drop]) / (n - 2 * drop)


def bootstrap_ci(xs: Sequence[float], stat: Callable[[Sequence[float]], float] = median,
                 n: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap confidence interval.
    
    Deterministic for a fixed seed. A single sample returns (x, x).
    """
    if not xs:
        raise ValueError("bootstrap_ci() requires non-empty input")
    
    x = stat(xs)
    if len(xs) == 1:
        return (x, x)
    
    rng = random.Random(seed)
    boot_vals = []
    for _ in range(n):
        sample = [rng.random() * len(xs) for _ in range(len(xs))]
        boot_vals.append(stat([xs[int(i)] for i in sample]))
    
    boot_vals.sort()
    lower = int((alpha / 2) * n)
    upper = int((1 - alpha / 2) * n) - 1
    return (boot_vals[lower], boot_vals[upper])


class Verdict(Enum):
    BETTER = "better"
    WORSE = "worse"
    TIE = "tie"
    NEED_MORE = "need_more"


@dataclass(frozen=True)
class Comparison:
    verdict: Verdict
    delta_rel: float
    threshold_rel: float
    n_a: int
    n_b: int
    reason: str


def compare(a: Sequence[float], b: Sequence[float], *, higher_is_better: bool = True,
            min_rel: float = 0.03, spread_k: float = 2.0, min_n: int = 3,
            clear_win_factor: float = 4.0) -> Comparison:
    """Compare two sequences to decide if B is really better than A.
    
    Rules:
    - threshold_rel = max(min_rel, spread_k * max(rel_spread(a), rel_spread(b))).
    - If either side has fewer than min_n samples: return BETTER/WORSE only when |delta_rel| > 
      clear_win_factor * threshold_rel, otherwise NEED_MORE.
    - With enough samples on both sides: BETTER if delta_rel > threshold_rel, WORSE if delta_rel < 
      -threshold_rel, else TIE.
    - median(a) == 0 must not divide by zero: treat delta_rel as +inf/-inf/0 sensibly.
    - Empty input on either side raises ValueError.
    """
    if not a:
        raise ValueError("compare() requires non-empty input for a")
    if not b:
        raise ValueError("compare() requires non-empty input for b")
    
    med_a = median(a)
    med_b = median(b)
    
    # Compute delta_rel = (median(b) - median(a)) / |median(a)|
    if med_a == 0:
        # If A's median is 0, the delta is undefined; we treat it as 0 to avoid division by zero
        delta_rel = 0.0
    else:
        delta_rel = (med_b - med_a) / abs(med_a)
    
    # Flip sign if lower is better
    if not higher_is_better:
        delta_rel = -delta_rel
    
    spread_a = rel_spread(a)
    spread_b = rel_spread(b)
    max_spread = max(spread_a, spread_b)
    threshold_rel = max(min_rel, spread_k * max_spread)
    
    n_a = len(a)
    n_b = len(b)
    
    # Handle insufficient samples
    if n_a < min_n or n_b < min_n:
        clear_threshold = clear_win_factor * threshold_rel
        if abs(delta_rel) > clear_threshold:
            # Determine BETTER vs WORSE based on direction
            if delta_rel > 0:
                verdict = Verdict.BETTER
            else:
                verdict = Verdict.WORSE
            reason_parts = []
            if med_a == 0:
                reason_parts.append("median of A is 0")
            else:
                if higher_is_better:
                    sign = "+" if delta_rel > 0 else ""
                    reason_parts.append(f"B is {sign}{abs(delta_rel * 100):.1f}% different from A")
                else:
                    sign = "+" if delta_rel > 0 else ""
                    reason_parts.append(f"B is {sign}{abs(delta_rel * 100):.1f}% different from A")
                reason_parts.append(f"(clear win factor threshold: {clear_threshold * 100:.1f}%)")
            reason = "; ".join(reason_parts)
        else:
            verdict = Verdict.NEED_MORE
            reason = f"insufficient samples ({n_a} vs {n_b}); need more to decide"
        return Comparison(
            verdict=verdict,
            delta_rel=delta_rel,
            threshold_rel=threshold_rel,
            n_a=n_a,
            n_b=n_b,
            reason=reason
        )
    
    # Enough samples on both sides
    if delta_rel > threshold_rel:
        verdict = Verdict.BETTER
        if med_a == 0:
            reason = "B beats A; median of A is 0"
        elif higher_is_better:
            reason = f"B is {delta_rel * 100:.1f}% better than A; the bar was {threshold_rel * 100:.1f}%."
        else:
            reason = f"B is {delta_rel * 100:.1f}% slower than A; the bar was {threshold_rel * 100:.1f}%."
    elif delta_rel < -threshold_rel:
        verdict = Verdict.WORSE
        if med_a == 0:
            reason = "B loses to A; median of A is 0"
        elif higher_is_better:
            reason = f"B is {abs(delta_rel) * 100:.1f}% worse than A; the bar was {threshold_rel * 100:.1f}%."
        else:
            reason = f"B is {abs(delta_rel) * 100:.1f}% faster than A; the bar was {threshold_rel * 100:.1f}%."
    else:
        verdict = Verdict.TIE
        if med_a == 0:
            reason = "B and A are comparable; median of A is 0"
        elif higher_is_better:
            reason = f"B is {delta_rel * 100:.1f}% different from A; the bar was {threshold_rel * 100:.1f}%."
        else:
            reason = f"B is {delta_rel * 100:.1f}% different from A; the bar was {threshold_rel * 100:.1f}%."
    
    return Comparison(
        verdict=verdict,
        delta_rel=delta_rel,
        threshold_rel=threshold_rel,
        n_a=n_a,
        n_b=n_b,
        reason=reason
    )


def relative_score(candidate: Mapping[str, float], baseline: Mapping[str, float],
                   weights: Mapping[str, float]) -> float:
    """Weighted geometric mean of candidate/baseline ratios, minus 1.
    
    0.0 = same as baseline, 0.1 = 10% better overall.
    Keys missing from either side, or with a zero/negative baseline, are skipped and 
    the remaining weights renormalised; if nothing is left, return 0.0.
    """
    # Filter to valid keys and renormalize weights
    valid_weights = {}
    total_weight = 0.0
    for key, weight in weights.items():
        if key in candidate and key in baseline and baseline[key] > 0:
            valid_weights[key] = weight
            total_weight += weight
    
    if total_weight == 0:
        return 0.0
    
    # Renormalize weights
    renorm_weights = {k: w / total_weight for k, w in valid_weights.items()}
    
    # Compute weighted geometric mean of ratios
    log_sum = 0.0
    for key, w in renorm_weights.items():
        ratio = candidate[key] / baseline[key]
        log_sum += w * math.log(ratio)
    
    # Geometric mean minus 1
    return math.exp(log_sum) - 1
