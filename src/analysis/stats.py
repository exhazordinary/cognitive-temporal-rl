"""Statistics helpers for comparing experiments across seeds.

Small-sample RL comparisons (5-10 seeds) need more than mean +/- std:
- percentile bootstrap CIs make no normality assumption
- Welch's t-test does not assume equal variances between configs
- IQM (interquartile mean) is robust to outlier seeds (rliable's
  recommended aggregate, computed here with scipy alone)
- Hedges' g gives a small-sample-corrected effect size
"""

from typing import Callable, NamedTuple, Optional, Sequence

import numpy as np
from scipy import stats as sps


class BootstrapCI(NamedTuple):
    estimate: float
    low: float
    high: float


class WelchResult(NamedTuple):
    statistic: float
    p_value: float


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 10_000,
    ci: float = 0.95,
    statistic: Callable = np.mean,
    rng: Optional[np.random.Generator] = None,
) -> BootstrapCI:
    """Percentile bootstrap confidence interval for a statistic."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("bootstrap_ci needs at least one value")
    rng = rng or np.random.default_rng()

    samples = rng.choice(values, size=(n_boot, values.size), replace=True)
    boot_stats = np.apply_along_axis(statistic, 1, samples)

    alpha = (1.0 - ci) / 2.0
    low, high = np.quantile(boot_stats, [alpha, 1.0 - alpha])
    return BootstrapCI(float(statistic(values)), float(low), float(high))


def bootstrap_diff_ci(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = 10_000,
    ci: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> BootstrapCI:
    """Bootstrap CI for the difference of means (a - b), independent samples."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    rng = rng or np.random.default_rng()

    boot_a = rng.choice(a, size=(n_boot, a.size), replace=True).mean(axis=1)
    boot_b = rng.choice(b, size=(n_boot, b.size), replace=True).mean(axis=1)
    diffs = boot_a - boot_b

    alpha = (1.0 - ci) / 2.0
    low, high = np.quantile(diffs, [alpha, 1.0 - alpha])
    return BootstrapCI(float(a.mean() - b.mean()), float(low), float(high))


def welch_t_test(a: Sequence[float], b: Sequence[float]) -> WelchResult:
    """Welch's t-test (unequal variances) between two seed-level samples."""
    result = sps.ttest_ind(np.asarray(a, dtype=float), np.asarray(b, dtype=float),
                           equal_var=False)
    return WelchResult(float(result.statistic), float(result.pvalue))


def iqm(values: Sequence[float]) -> float:
    """Interquartile mean: mean of the middle 50% of values."""
    return float(sps.trim_mean(np.asarray(values, dtype=float), 0.25))


def hedges_g(a: Sequence[float], b: Sequence[float]) -> float:
    """Hedges' g effect size (small-sample-corrected Cohen's d) for a vs b."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n_a, n_b = a.size, b.size
    dof = n_a + n_b - 2
    if dof <= 0:
        return float("nan")

    pooled_var = ((n_a - 1) * a.var(ddof=1) + (n_b - 1) * b.var(ddof=1)) / dof
    if pooled_var == 0:
        return float("nan")

    d = (a.mean() - b.mean()) / np.sqrt(pooled_var)
    correction = 1.0 - 3.0 / (4.0 * dof - 1.0)
    return float(d * correction)
