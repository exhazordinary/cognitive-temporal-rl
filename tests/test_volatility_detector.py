"""Tests for the volatility-vs-noise detector (Gershman 2020 inspired)."""

import numpy as np
import pytest

from src.surprise import VolatilityDetector


def feed(detector, values):
    outputs = [detector.step(float(v)) for v in values]
    return outputs


class TestStationaryNoise:
    """A stationary noisy PE stream should be treated as noise, not change."""

    def test_no_boost_under_stationary_noise(self):
        rng = np.random.default_rng(0)
        detector = VolatilityDetector()

        feed(detector, rng.normal(0.0, 1.0, size=300))

        assert detector.get_rollout_recommendation() <= 1.0

    def test_few_change_points_under_stationary_noise(self):
        rng = np.random.default_rng(1)
        detector = VolatilityDetector()

        feed(detector, rng.normal(0.0, 1.0, size=300))

        # Allow a couple of spurious detections, but the stream is stationary
        assert len(detector.change_points) <= 5


class TestChangeDetection:
    """A shift in the PE mean (environment changed) should be detected."""

    def test_mean_shift_detected(self):
        rng = np.random.default_rng(2)
        detector = VolatilityDetector(window_short=20, window_long=100)

        feed(detector, rng.normal(0.0, 1.0, size=150))
        outputs = feed(detector, rng.normal(5.0, 1.0, size=30))

        assert any(o.change_detected for o in outputs)
        # Detection should happen within window_short of the shift
        first_detection = next(i for i, o in enumerate(outputs) if o.change_detected)
        assert first_detection < detector.window_short

    def test_recommendation_boosts_lr_after_change(self):
        rng = np.random.default_rng(3)
        detector = VolatilityDetector(window_short=20, window_long=100)

        feed(detector, rng.normal(0.0, 1.0, size=150))
        feed(detector, rng.normal(5.0, 1.0, size=30))

        assert detector.get_rollout_recommendation() > 1.0


class TestBounds:
    def test_multiplier_within_bounds(self):
        rng = np.random.default_rng(4)
        detector = VolatilityDetector(min_multiplier=0.5, max_multiplier=2.0)

        values = np.concatenate([
            rng.normal(0.0, 1.0, size=100),
            rng.normal(50.0, 10.0, size=100),  # extreme shift
        ])
        outputs = feed(detector, values)

        for out in outputs:
            assert 0.5 <= out.lr_multiplier <= 2.0

    def test_warmup_returns_neutral(self):
        detector = VolatilityDetector(window_short=20)

        out = detector.step(1.0)

        assert out.lr_multiplier == 1.0
        assert not out.change_detected
        assert detector.get_rollout_recommendation() >= 0.0

    def test_hard_reset(self):
        rng = np.random.default_rng(5)
        detector = VolatilityDetector()
        feed(detector, rng.normal(0.0, 1.0, size=100))

        detector.hard_reset()

        assert detector.step_count == 0
        assert len(detector.pe_history) == 0
        assert detector.get_rollout_recommendation() == 1.0
