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


class BatchSurpriseOutput(NamedTuple):
    """Output from batched surprise computation (one row per parallel env)."""
    surprises: np.ndarray        # Normalized prediction errors (n,)
    alpha: float                 # Pearce-Hall associability after this vec-step
    raw_errors: np.ndarray       # Raw MSEs (n,)


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
        """Process a single transition and compute surprise.

        Thin wrapper over step_batch with batch size 1.

        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state

        Returns:
            SurpriseOutput with surprise and smoothed associability
        """
        state = torch.as_tensor(state, dtype=torch.float32)
        next_state = torch.as_tensor(next_state, dtype=torch.float32)
        out = self.step_batch(
            state.unsqueeze(0),
            torch.as_tensor(action).reshape(1),
            next_state.unsqueeze(0),
        )
        return SurpriseOutput(
            surprise=float(out.surprises[0]),
            smoothed_surprise=out.alpha,
            raw_error=float(out.raw_errors[0]),
        )

    def step_batch(
        self,
        states,
        actions,
        next_states,
    ) -> BatchSurpriseOutput:
        """Process one vectorized-env step (n parallel transitions).

        Pearce-Hall smoothing is updated ONCE per vec-step using the mean
        absolute prediction error across envs, so alpha stays a single
        scalar signal regardless of n_envs. Note this means gamma is
        denominated in vec-steps: with n_envs parallel envs, alpha receives
        n_envs times fewer updates per env-step than a single-env run.

        Args:
            states: Pre-step observations (n, state_dim), array or tensor
            actions: Actions taken (n,)
            next_states: Resulting observations (n, state_dim)

        Returns:
            BatchSurpriseOutput with per-env surprises and updated alpha
        """
        states = torch.as_tensor(np.asarray(states), dtype=torch.float32)
        actions = torch.as_tensor(np.asarray(actions)).long().flatten()
        next_states = torch.as_tensor(np.asarray(next_states), dtype=torch.float32)
        n = states.shape[0]

        self.step_count += n

        # Store transitions for forward model training
        for i in range(n):
            self.transition_buffer.append((states[i], actions[i], next_states[i]))
        if len(self.transition_buffer) > self.max_buffer_size:
            del self.transition_buffer[:len(self.transition_buffer) - self.max_buffer_size]

        # Compute surprise (prediction error) in one forward pass
        surprises, raw_errors = self.forward_model.compute_surprise_batch(
            states, actions, next_states,
        )

        # Pearce-Hall update: α = γ·mean|PE| + (1-γ)α, once per vec-step
        pe = float(np.mean(np.abs(surprises)))
        self.alpha = self.gamma * pe + (1 - self.gamma) * self.alpha

        # Track for rollout aggregation
        self.rollout_surprises.extend(float(s) for s in surprises)

        # Record history (one alpha entry per vec-step)
        self.surprise_history.extend(float(s) for s in surprises)
        self.alpha_history.append(self.alpha)

        # Periodically train forward model (train_every is in env-steps;
        # trigger whenever a multiple of train_every was crossed)
        if self.step_count % self.train_every < n and len(self.transition_buffer) >= self.batch_size:
            self._train_forward_model()

        return BatchSurpriseOutput(
            surprises=surprises,
            alpha=self.alpha,
            raw_errors=raw_errors,
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
