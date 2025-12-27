"""Pearce-Hall inspired learning rate modulation.

Based on Pearce & Hall (1980) and SurNoR (2021):
- Associability (effective learning rate) is modulated by surprise
- α_t = γ|PE| + (1-γ)α_{t-1}
- High surprise -> increased learning rate (or decreased, configurable)

Key difference from SalienceLR:
- Applies modulation per-rollout, not per-step
- Uses proper Pearce-Hall smoothing
- Designed for PPO update phase, not experience collection
"""

import torch
from typing import Optional


class PearceHallLR:
    """Learning rate modulator using Pearce-Hall associability.

    The key insight from Pearce-Hall: learning rate should adapt based
    on recent prediction errors, but SMOOTHLY - not per-step jitter.
    """

    def __init__(
        self,
        base_lr: float = 3e-4,
        min_multiplier: float = 0.5,
        max_multiplier: float = 2.0,
        invert: bool = False,
    ):
        """Initialize Pearce-Hall LR modulator.

        Args:
            base_lr: Base learning rate
            min_multiplier: Minimum LR multiplier (0.5 = can halve LR)
            max_multiplier: Maximum LR multiplier (2.0 = can double LR)
            invert: If True, high surprise -> LOWER LR (stabilization mode)
                   If False, high surprise -> HIGHER LR (Pearce-Hall mode)
        """
        self.base_lr = base_lr
        self.min_multiplier = min_multiplier
        self.max_multiplier = max_multiplier
        self.invert = invert

        # Tracking
        self.current_multiplier = 1.0
        self.lr_history: list[float] = []
        self.alpha_history: list[float] = []

    def compute_lr(self, alpha: float) -> float:
        """Compute modulated learning rate from associability.

        Args:
            alpha: Pearce-Hall associability (from SurpriseModule)
                   Expected range roughly [0, 2] with 1.0 being neutral

        Returns:
            Modulated learning rate
        """
        # Alpha > 1 means above-average surprise recently
        # Alpha < 1 means below-average surprise recently

        if self.invert:
            # High alpha (surprise) -> lower LR (stabilize during chaos)
            # Map alpha inversely: high alpha -> low multiplier
            multiplier = 2.0 - alpha  # alpha=1.5 -> mult=0.5, alpha=0.5 -> mult=1.5
        else:
            # High alpha (surprise) -> higher LR (Pearce-Hall standard)
            # alpha directly becomes multiplier
            multiplier = alpha

        # Clamp to valid range
        multiplier = max(self.min_multiplier, min(self.max_multiplier, multiplier))
        self.current_multiplier = multiplier

        lr = self.base_lr * multiplier
        self.lr_history.append(lr)
        self.alpha_history.append(alpha)

        return lr

    def apply_to_optimizer(
        self,
        optimizer: torch.optim.Optimizer,
        alpha: float,
    ) -> float:
        """Apply modulated LR to optimizer.

        Call this ONCE per PPO update, not per step!

        Args:
            optimizer: The optimizer to modify
            alpha: Current associability from SurpriseModule

        Returns:
            The new learning rate
        """
        new_lr = self.compute_lr(alpha)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

    def reset(self):
        """Reset tracking (keep for new experiment)."""
        self.lr_history.clear()
        self.alpha_history.clear()
        self.current_multiplier = 1.0

    def get_stats(self) -> dict:
        """Get modulator statistics."""
        return {
            "base_lr": self.base_lr,
            "current_multiplier": self.current_multiplier,
            "current_lr": self.base_lr * self.current_multiplier,
            "invert": self.invert,
            "mean_lr": (
                sum(self.lr_history) / len(self.lr_history)
                if self.lr_history else self.base_lr
            ),
            "n_updates": len(self.lr_history),
        }
