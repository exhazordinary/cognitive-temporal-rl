"""Tests for the statistical analysis helpers."""

import numpy as np
import pytest

from src.analysis.stats import bootstrap_ci, bootstrap_diff_ci, hedges_g, iqm, welch_t_test


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestBootstrapCI:
    def test_ci_brackets_true_mean(self, rng):
        values = rng.normal(10.0, 2.0, size=50)

        ci = bootstrap_ci(values, rng=rng)

        assert ci.low <= 10.0 <= ci.high
        assert ci.low <= ci.estimate <= ci.high

    def test_ci_narrows_with_more_data(self, rng):
        small = bootstrap_ci(rng.normal(0, 1, size=5), rng=rng)
        large = bootstrap_ci(rng.normal(0, 1, size=500), rng=rng)

        assert (large.high - large.low) < (small.high - small.low)

    def test_empty_input_rejected(self, rng):
        with pytest.raises(ValueError):
            bootstrap_ci([], rng=rng)


class TestBootstrapDiffCI:
    def test_clear_difference_excludes_zero(self, rng):
        a = rng.normal(10.0, 1.0, size=20)
        b = rng.normal(0.0, 1.0, size=20)

        diff = bootstrap_diff_ci(a, b, rng=rng)

        assert diff.estimate == pytest.approx(a.mean() - b.mean())
        assert diff.low > 0  # CI of the difference excludes zero

    def test_identical_distributions_include_zero(self, rng):
        a = rng.normal(0.0, 1.0, size=30)
        b = rng.normal(0.0, 1.0, size=30)

        diff = bootstrap_diff_ci(a, b, rng=rng)

        assert diff.low < 0 < diff.high


class TestWelch:
    def test_shifted_samples_significant(self, rng):
        a = rng.normal(5.0, 1.0, size=15)
        b = rng.normal(0.0, 1.0, size=15)

        result = welch_t_test(a, b)

        assert result.p_value < 0.01
        assert result.statistic > 0

    def test_identical_samples_not_significant(self, rng):
        a = rng.normal(0.0, 1.0, size=15)
        b = rng.normal(0.0, 1.0, size=15)

        result = welch_t_test(a, b)

        assert result.p_value > 0.05


class TestIQM:
    def test_ignores_outliers(self):
        values = [1.0, 2.0, 3.0, 4.0, 1000.0]

        assert iqm(values) < np.mean(values)
        assert iqm(values) == pytest.approx(3.0)

    def test_symmetric_data(self):
        values = list(range(1, 101))

        assert iqm(values) == pytest.approx(np.mean(values), rel=0.01)


class TestHedgesG:
    def test_positive_for_larger_first_sample(self, rng):
        a = rng.normal(5.0, 1.0, size=10)
        b = rng.normal(0.0, 1.0, size=10)

        assert hedges_g(a, b) > 1.0

    def test_near_zero_for_identical(self, rng):
        a = rng.normal(0.0, 1.0, size=50)
        b = rng.normal(0.0, 1.0, size=50)

        assert abs(hedges_g(a, b)) < 0.5

    def test_nan_for_degenerate(self):
        assert np.isnan(hedges_g([1.0], [2.0]))
