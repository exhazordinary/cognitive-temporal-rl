"""Learning rate modulation based on salience."""

import torch
from typing import Optional


class SalienceLR:
    """Modulates learning rate based on entropy-derived salience.

    High salience moments get higher learning rates, allowing the
    agent to encode novel experiences more strongly.
    """

    def __init__(
        self,
        base_lr: float = 3e-4,
        gamma: float = 0.5,         # Salience influence strength
        min_multiplier: float = 0.5,
        max_multiplier: float = 2.0,
        baseline_momentum: float = 0.99,
        invert_direction: bool = False,
    ):
        """Initialize the salience-based LR modulator.

        Args:
            base_lr: Base learning rate
            gamma: How much salience affects LR (higher = more effect)
            min_multiplier: Minimum LR multiplier
            max_multiplier: Maximum LR multiplier
            baseline_momentum: EMA momentum for baseline salience
            invert_direction: If True, high salience DECREASES LR (consolidate during chaos)
        """
        self.base_lr = base_lr
        self.gamma = gamma
        self.min_multiplier = min_multiplier
        self.max_multiplier = max_multiplier
        self.baseline_momentum = baseline_momentum
        self.invert_direction = invert_direction

        self.baseline_salience: Optional[float] = None
        self.current_multiplier: float = 1.0
        self.lr_history: list[float] = []

    def get_lr(self, salience_score: float) -> float:
        """Get modulated learning rate based on current salience.

        Args:
            salience_score: Current salience from entropy clock

        Returns:
            Modulated learning rate
        """
        # Update baseline
        if self.baseline_salience is None:
            self.baseline_salience = salience_score
        else:
            self.baseline_salience = (
                self.baseline_momentum * self.baseline_salience
                + (1 - self.baseline_momentum) * salience_score
            )

        # Compute multiplier based on deviation from baseline
        deviation = salience_score - self.baseline_salience
        if self.invert_direction:
            deviation = -deviation  # High salience -> lower LR
        multiplier = 1.0 + self.gamma * deviation

        # Clamp to valid range
        multiplier = max(self.min_multiplier, min(self.max_multiplier, multiplier))
        self.current_multiplier = multiplier

        lr = self.base_lr * multiplier
        self.lr_history.append(lr)

        return lr

    def get_lr_multiplier(self, salience_score: float) -> float:
        """Get just the multiplier (for use with external optimizers).

        Args:
            salience_score: Current salience from entropy clock

        Returns:
            Learning rate multiplier (1.0 = no change)
        """
        self.get_lr(salience_score)  # Updates internal state
        return self.current_multiplier

    def update_optimizer_lr(
        self,
        optimizer: torch.optim.Optimizer,
        salience_score: float,
    ) -> float:
        """Directly update an optimizer's learning rate.

        Args:
            optimizer: PyTorch optimizer to update
            salience_score: Current salience from entropy clock

        Returns:
            New learning rate
        """
        new_lr = self.get_lr(salience_score)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

    def reset(self) -> None:
        """Reset for new experiment (keeps baseline for continuity)."""
        self.lr_history.clear()
        self.current_multiplier = 1.0

    def hard_reset(self) -> None:
        """Full reset including baseline."""
        self.reset()
        self.baseline_salience = None

    def get_stats(self) -> dict:
        """Get current modulator statistics."""
        return {
            "base_lr": self.base_lr,
            "current_multiplier": self.current_multiplier,
            "current_lr": self.base_lr * self.current_multiplier,
            "baseline_salience": self.baseline_salience,
            "mean_lr": sum(self.lr_history) / len(self.lr_history) if self.lr_history else self.base_lr,
        }
