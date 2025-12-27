"""Surprise module combining prediction error with Pearce-Hall smoothing."""

import torch
import numpy as np
from typing import Optional, NamedTuple, List
from .forward_model import ForwardModel


class SurpriseOutput(NamedTuple):
    """Output from surprise computation."""
    surprise: float              # Normalized prediction error
    smoothed_surprise: float     # Pearce-Hall smoothed value (associability)
    raw_error: float             # Raw MSE


class SurpriseModule:
    """Computes surprise via prediction error with Pearce-Hall smoothing.

    Key differences from EntropyClockModule:
    1. Uses prediction error (like ICM/RND) not state entropy
    2. Applies Pearce-Hall smoothing: α_t = γ|PE| + (1-γ)α_{t-1}
    3. Designed for per-rollout aggregation, not per-step LR modulation

    Based on SurNoR (2021) and Pearce-Hall (1980) theories.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        # Pearce-Hall parameters
        gamma: float = 0.3,           # How quickly associability updates
        initial_alpha: float = 1.0,   # Starting associability
        # Forward model parameters
        hidden_dim: int = 64,
        model_lr: float = 1e-3,
        # Training
        train_every: int = 100,       # Train forward model every N steps
        batch_size: int = 64,
    ):
        """Initialize surprise module.

        Args:
            state_dim: Observation dimension
            action_dim: Action space size
            gamma: Pearce-Hall update rate (higher = faster adaptation)
            initial_alpha: Starting associability value
            hidden_dim: Forward model hidden size
            model_lr: Forward model learning rate
            train_every: Steps between forward model updates
            batch_size: Batch size for forward model training
        """
        self.gamma = gamma
        self.initial_alpha = initial_alpha
        self.train_every = train_every
        self.batch_size = batch_size

        # Forward dynamics model
        self.forward_model = ForwardModel(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            learning_rate=model_lr,
        )

        # Pearce-Hall associability (smoothed learning rate multiplier)
        self.alpha = initial_alpha

        # Transition buffer for training forward model
        self.transition_buffer: List[tuple] = []
        self.max_buffer_size = 10000

        # History for analysis
        self.surprise_history: List[float] = []
        self.alpha_history: List[float] = []
        self.step_count = 0

        # Per-rollout accumulator
        self.rollout_surprises: List[float] = []

    def step(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
    ) -> SurpriseOutput:
        """Process a transition and compute surprise.

        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state

        Returns:
            SurpriseOutput with surprise and smoothed associability
        """
        self.step_count += 1

        # Store transition for training
        self.transition_buffer.append((
            state.detach().clone(),
            action.detach().clone() if isinstance(action, torch.Tensor) else torch.tensor(action),
            next_state.detach().clone(),
        ))
        if len(self.transition_buffer) > self.max_buffer_size:
            self.transition_buffer.pop(0)

        # Compute surprise (prediction error)
        surprise, raw_error = self.forward_model.compute_surprise(
            state,
            action if isinstance(action, torch.Tensor) else torch.tensor(action),
            next_state,
        )

        # Pearce-Hall update: α = γ|PE| + (1-γ)α
        # Using absolute surprise (unsigned prediction error)
        abs_surprise = abs(surprise)
        self.alpha = self.gamma * abs_surprise + (1 - self.gamma) * self.alpha

        # Track for rollout aggregation
        self.rollout_surprises.append(surprise)

        # Record history
        self.surprise_history.append(surprise)
        self.alpha_history.append(self.alpha)

        # Periodically train forward model
        if self.step_count % self.train_every == 0 and len(self.transition_buffer) >= self.batch_size:
            self._train_forward_model()

        return SurpriseOutput(
            surprise=surprise,
            smoothed_surprise=self.alpha,
            raw_error=raw_error.item() if isinstance(raw_error, torch.Tensor) else raw_error,
        )

    def get_rollout_stats(self) -> dict:
        """Get aggregated statistics for the current rollout.

        Call this at the end of a rollout, before PPO update.
        This is the RIGHT time to modulate learning rate.

        Returns:
            Dictionary with mean_surprise, max_surprise, alpha, etc.
        """
        if not self.rollout_surprises:
            return {
                "mean_surprise": 0.0,
                "max_surprise": 0.0,
                "alpha": self.alpha,
                "n_steps": 0,
            }

        return {
            "mean_surprise": float(np.mean(self.rollout_surprises)),
            "max_surprise": float(np.max(self.rollout_surprises)),
            "std_surprise": float(np.std(self.rollout_surprises)),
            "alpha": self.alpha,
            "n_steps": len(self.rollout_surprises),
        }

    def clear_rollout(self):
        """Clear rollout accumulator. Call after PPO update."""
        self.rollout_surprises.clear()

    def _train_forward_model(self):
        """Train forward model on buffered transitions."""
        if len(self.transition_buffer) < self.batch_size:
            return

        # Sample random batch
        indices = np.random.choice(len(self.transition_buffer), self.batch_size, replace=False)
        batch = [self.transition_buffer[i] for i in indices]

        states = torch.stack([t[0] for t in batch])
        actions = torch.stack([t[1] for t in batch])
        next_states = torch.stack([t[2] for t in batch])

        self.forward_model.train_step(states, actions, next_states)

    def reset_episode(self):
        """Reset for new episode (keep alpha for continuity)."""
        # Don't clear alpha - it should persist across episodes
        # Only clear per-episode tracking if any
        pass

    def hard_reset(self):
        """Full reset for new experiment."""
        self.alpha = self.initial_alpha
        self.surprise_history.clear()
        self.alpha_history.clear()
        self.rollout_surprises.clear()
        self.transition_buffer.clear()
        self.step_count = 0
        self.forward_model.reset_statistics()

    def get_stats(self) -> dict:
        """Get current module statistics."""
        return {
            "alpha": self.alpha,
            "step_count": self.step_count,
            "buffer_size": len(self.transition_buffer),
            "mean_recent_surprise": (
                float(np.mean(self.surprise_history[-100:]))
                if self.surprise_history else 0.0
            ),
        }
