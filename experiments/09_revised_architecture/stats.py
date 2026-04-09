"""Reusable statistical utilities for experiment analysis.

Pure-Python implementations (no scipy dependency) of:
- Welch's t-test
- Bootstrap confidence intervals
- Cohen's d effect size
- Standard error of the mean

These match scipy.stats.ttest_ind(equal_var=False) to within floating-point
precision for the t-statistic and use the Welch-Satterthwaite degrees-of-
freedom approximation for the p-value.
"""
from __future__ import annotations

import math
import random


def sem(values: list[float]) -> float:
    """Standard error of the mean: std / sqrt(n).

    Returns 0.0 when n < 2 (undefined, but avoids crashing callers).
    """
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    return math.sqrt(var / n)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _var(values: list[float]) -> float:
    """Sample variance (Bessel-corrected, ddof=1)."""
    n = len(values)
    if n < 2:
        return 0.0
    m = _mean(values)
    return sum((x - m) ** 2 for x in values) / (n - 1)


def _std(values: list[float]) -> float:
    return math.sqrt(_var(values))


# -- Welch's t-test --


def _regularised_incomplete_beta(a: float, b: float, x: float) -> float:
    """Compute I_x(a, b) via the continued-fraction expansion (Lentz).

    Used to convert (t, df) -> two-tailed p-value without scipy.
    Accuracy is ~1e-12 for typical df values in small-sample experiments.
    """
    if x < 0.0 or x > 1.0:
        raise ValueError(f"x must be in [0, 1], got {x}")
    if x == 0.0 or x == 1.0:
        return x

    # Use the symmetry relation when x > (a+1)/(a+b+2) for convergence
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _regularised_incomplete_beta(b, a, 1.0 - x)

    # Log of the front factor: x^a * (1-x)^b / (a * B(a,b))
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1.0 - x) - lbeta) / a

    # Lentz continued fraction
    TINY = 1e-30
    MAX_ITER = 200

    f = 1.0 + TINY
    c = f
    d = 0.0

    for m in range(0, MAX_ITER):
        # Two sub-iterations per m -- even and odd numerators
        for j in range(2):
            if m == 0 and j == 0:
                numerator = 1.0  # a_0 = 1
            elif j == 0:
                # even numerator: d_{2m}
                numerator = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
            else:
                # odd numerator: d_{2m+1}
                mm = m + 1 if m == 0 else m
                if m == 0:
                    mm = 0
                    numerator = -(a + mm) * (a + b + mm) * x / ((a + 2 * mm) * (a + 2 * mm + 1))
                else:
                    numerator = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))

            d = 1.0 + numerator * d
            if abs(d) < TINY:
                d = TINY
            d = 1.0 / d

            c = 1.0 + numerator / c
            if abs(c) < TINY:
                c = TINY

            delta = c * d
            f *= delta

            if abs(delta - 1.0) < 1e-14:
                return front * (f - 1.0)

    return front * (f - 1.0)


def _t_cdf(t_val: float, df: float) -> float:
    """CDF of the Student-t distribution at *t_val* with *df* degrees of freedom."""
    x = df / (df + t_val * t_val)
    ib = _regularised_incomplete_beta(df / 2.0, 0.5, x)
    cdf = 0.5 * ib
    if t_val > 0:
        cdf = 1.0 - cdf
    return cdf


def welch_ttest(a: list[float], b: list[float]) -> tuple[float, float]:
    """Welch's t-test (unequal variances).

    Returns (t_statistic, two_tailed_p_value).
    Requires len(a) >= 2 and len(b) >= 2.
    """
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return (0.0, 1.0)

    ma, mb = _mean(a), _mean(b)
    va, vb = _var(a), _var(b)

    denom = math.sqrt(va / na + vb / nb)
    if denom == 0:
        return (0.0, 1.0)

    t_stat = (ma - mb) / denom

    # Welch-Satterthwaite degrees of freedom
    num = (va / na + vb / nb) ** 2
    den = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    if den == 0:
        return (t_stat, 1.0)
    df = num / den

    # Two-tailed p-value
    p = 2.0 * (1.0 - _t_cdf(abs(t_stat), df))
    p = max(0.0, min(1.0, p))
    return (t_stat, p)


# -- Bootstrap CI --


def bootstrap_ci(
    values: list[float],
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for the mean.

    Returns (lower, upper) bounds of the *ci*-level interval.
    Falls back to (mean, mean) for n < 2.
    """
    n = len(values)
    if n < 2:
        m = values[0] if values else 0.0
        return (m, m)

    rng = random.Random(seed)
    means = sorted(
        sum(rng.choices(values, k=n)) / n for _ in range(n_boot)
    )
    alpha = (1.0 - ci) / 2.0
    lo_idx = int(math.floor(alpha * n_boot))
    hi_idx = int(math.ceil((1.0 - alpha) * n_boot)) - 1
    lo_idx = max(0, min(lo_idx, n_boot - 1))
    hi_idx = max(0, min(hi_idx, n_boot - 1))
    return (means[lo_idx], means[hi_idx])


# -- Cohen's d --


def cohens_d(a: list[float], b: list[float]) -> float:
    """Cohen's d effect size (pooled std denominator).

    Returns 0.0 when either sample has fewer than 2 values.
    """
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    ma, mb = _mean(a), _mean(b)
    va, vb = _var(a), _var(b)
    pooled = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return (ma - mb) / pooled
