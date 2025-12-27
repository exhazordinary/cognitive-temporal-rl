"""Volatility vs. Noise detection for intelligent LR modulation.

Based on Gershman (2020) "Unpredictability vs. volatility and the control of learning":
- Volatility (environment changed) → should INCREASE learning rate
- Unpredictability (inherent noise) → should DECREASE learning rate

This module attempts to disentangle these two sources of prediction error.
"""

import numpy as np
from typing import Optional, NamedTuple, List
from collections import deque


class VolatilityOutput(NamedTuple):
    """Output from volatility detection."""
    volatility_estimate: float      # Estimated rate of change (0-1)
    noise_estimate: float           # Estimated stochasticity (0-1)
    lr_multiplier: float            # Recommended LR multiplier
    change_detected: bool           # Did we detect a change point?


class VolatilityDetector:
    """Distinguishes volatility (change) from noise (stochasticity).

    Key insight from computational neuroscience:
    - High variance of prediction errors = noise → decrease LR
    - Sudden shift in mean prediction error = volatility → increase LR

    Uses change-point detection combined with variance tracking.
    """

    def __init__(
        self,
        window_short: int = 20,      # Recent window for mean
        window_long: int = 100,      # Baseline window
        change_threshold: float = 2.0,  # Std devs for change detection
        noise_sensitivity: float = 0.5,  # How much noise affects LR
        volatility_boost: float = 0.5,   # How much volatility boosts LR
        min_multiplier: float = 0.5,
        max_multiplier: float = 2.0,
    ):
        """Initialize volatility detector.

        Args:
            window_short: Recent PE window for detecting changes
            window_long: Longer baseline for comparison
            change_threshold: How many stds to count as change
            noise_sensitivity: How strongly noise decreases LR
            volatility_boost: How strongly volatility increases LR
            min_multiplier: Minimum LR multiplier
            max_multiplier: Maximum LR multiplier
        """
        self.window_short = window_short
        self.window_long = window_long
        self.change_threshold = change_threshold
        self.noise_sensitivity = noise_sensitivity
        self.volatility_boost = volatility_boost
        self.min_multiplier = min_multiplier
        self.max_multiplier = max_multiplier

        # Prediction error history
        self.pe_history: deque = deque(maxlen=window_long)

        # Running statistics
        self.baseline_mean: Optional[float] = None
        self.baseline_std: Optional[float] = None
        self.ema_momentum = 0.95

        # Tracking
        self.volatility_history: List[float] = []
        self.noise_history: List[float] = []
        self.change_points: List[int] = []
        self.step_count = 0

    def step(self, prediction_error: float) -> VolatilityOutput:
        """Process a new prediction error and compute volatility/noise estimates.

        Args:
            prediction_error: Raw prediction error (can be negative)

        Returns:
            VolatilityOutput with estimates and LR recommendation
        """
        self.step_count += 1
        self.pe_history.append(prediction_error)

        # Need enough history
        if len(self.pe_history) < self.window_short:
            return VolatilityOutput(
                volatility_estimate=0.0,
                noise_estimate=0.0,
                lr_multiplier=1.0,
                change_detected=False,
            )

        pe_array = np.array(self.pe_history)

        # Compute recent vs baseline statistics
        recent = pe_array[-self.window_short:]
        recent_mean = np.mean(recent)
        recent_std = np.std(recent) + 1e-8

        if len(pe_array) >= self.window_long:
            baseline = pe_array[:-self.window_short]
            baseline_mean = np.mean(baseline)
            baseline_std = np.std(baseline) + 1e-8
        else:
            baseline_mean = np.mean(pe_array)
            baseline_std = np.std(pe_array) + 1e-8

        # Update running baseline with EMA
        if self.baseline_mean is None:
            self.baseline_mean = baseline_mean
            self.baseline_std = baseline_std
        else:
            self.baseline_mean = self.ema_momentum * self.baseline_mean + (1 - self.ema_momentum) * recent_mean
            self.baseline_std = self.ema_momentum * self.baseline_std + (1 - self.ema_momentum) * recent_std

        # === VOLATILITY DETECTION ===
        # Large shift in mean PE suggests environment changed
        mean_shift = abs(recent_mean - baseline_mean) / (baseline_std + 1e-8)
        change_detected = mean_shift > self.change_threshold

        # Normalize to 0-1 range
        volatility_estimate = np.clip(mean_shift / (2 * self.change_threshold), 0, 1)

        if change_detected:
            self.change_points.append(self.step_count)

        # === NOISE ESTIMATION ===
        # High variance of PE = inherent noise/unpredictability
        # Normalized by baseline variance
        noise_ratio = recent_std / (self.baseline_std + 1e-8)
        noise_estimate = np.clip((noise_ratio - 0.5) / 1.5, 0, 1)  # 0.5-2.0 → 0-1

        # === LR MULTIPLIER ===
        # Volatility → increase LR (adapt to change)
        # Noise → decrease LR (don't trust noisy signals)
        if change_detected:
            # Volatility dominates - increase LR
            multiplier = 1.0 + self.volatility_boost * volatility_estimate
        else:
            # Noise dominates - decrease LR
            multiplier = 1.0 - self.noise_sensitivity * noise_estimate

        multiplier = np.clip(multiplier, self.min_multiplier, self.max_multiplier)

        # Track history
        self.volatility_history.append(volatility_estimate)
        self.noise_history.append(noise_estimate)

        return VolatilityOutput(
            volatility_estimate=volatility_estimate,
            noise_estimate=noise_estimate,
            lr_multiplier=multiplier,
            change_detected=change_detected,
        )

    def get_rollout_recommendation(self) -> float:
        """Get LR multiplier recommendation for entire rollout.

        Call this at the end of a rollout, before PPO update.

        Returns:
            Recommended LR multiplier
        """
        if not self.volatility_history:
            return 1.0

        recent_n = min(50, len(self.volatility_history))

        # Check if any change points in recent history
        recent_changes = sum(1 for cp in self.change_points if cp > self.step_count - recent_n)

        if recent_changes > 0:
            # Volatility detected - boost LR
            avg_volatility = np.mean(self.volatility_history[-recent_n:])
            return 1.0 + self.volatility_boost * avg_volatility
        else:
            # Just noise - reduce LR
            avg_noise = np.mean(self.noise_history[-recent_n:])
            return 1.0 - self.noise_sensitivity * avg_noise

    def reset(self):
        """Reset for new episode (keep baseline estimates)."""
        pass  # Keep learning across episodes

    def hard_reset(self):
        """Full reset for new experiment."""
        self.pe_history.clear()
        self.baseline_mean = None
        self.baseline_std = None
        self.volatility_history.clear()
        self.noise_history.clear()
        self.change_points.clear()
        self.step_count = 0

    def get_stats(self) -> dict:
        """Get current detector statistics."""
        return {
            "step_count": self.step_count,
            "n_change_points": len(self.change_points),
            "recent_change_points": self.change_points[-5:] if self.change_points else [],
            "mean_volatility": np.mean(self.volatility_history[-100:]) if self.volatility_history else 0,
            "mean_noise": np.mean(self.noise_history[-100:]) if self.noise_history else 0,
            "baseline_mean": self.baseline_mean,
            "baseline_std": self.baseline_std,
        }
