"""Forward dynamics model for computing prediction error (surprise)."""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class ForwardModel(nn.Module):
    """Predicts next state given current state and action.

    The prediction error serves as a measure of surprise/novelty.
    Based on ICM (Pathak et al. 2017) but simplified for LR modulation.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        learning_rate: float = 1e-3,
    ):
        """Initialize forward model.

        Args:
            state_dim: Dimension of state observations
            action_dim: Dimension of action space (discrete = num actions)
            hidden_dim: Hidden layer size
            learning_rate: Learning rate for forward model training
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Simple MLP: (state, action_onehot) -> next_state
        input_dim = state_dim + action_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # Running statistics for normalization
        self.obs_mean = torch.zeros(state_dim)
        self.obs_var = torch.ones(state_dim)
        self.error_mean = 0.0
        self.error_var = 1.0
        self.update_count = 0

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Predict next state.

        Args:
            state: Current state (batch_size, state_dim)
            action: Action taken (batch_size,) for discrete

        Returns:
            Predicted next state (batch_size, state_dim)
        """
        # One-hot encode discrete actions
        if action.dim() == 1:
            action_onehot = torch.zeros(action.shape[0], self.action_dim)
            action_onehot.scatter_(1, action.unsqueeze(1).long(), 1.0)
        else:
            action_onehot = action

        x = torch.cat([state, action_onehot], dim=-1)
        return self.network(x)

    def compute_surprise(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        normalize: bool = True,
    ) -> Tuple[float, torch.Tensor]:
        """Compute surprise (prediction error) for a transition.

        Args:
            state: Current state
            action: Action taken
            next_state: Actual next state
            normalize: Whether to normalize by running statistics

        Returns:
            Tuple of (normalized_surprise, raw_error_tensor)
        """
        with torch.no_grad():
            predicted = self.forward(state.unsqueeze(0), action.unsqueeze(0))
            error = (predicted - next_state.unsqueeze(0)).pow(2).mean()

            if normalize and self.update_count > 100:
                # Normalize by running statistics (like RND)
                normalized = (error.item() - self.error_mean) / (np.sqrt(self.error_var) + 1e-8)
                # Clip to reasonable range
                normalized = np.clip(normalized, -5.0, 5.0)
                return normalized, error

            return error.item(), error

    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
    ) -> float:
        """Train the forward model on a batch of transitions.

        Args:
            states: Batch of states (batch_size, state_dim)
            actions: Batch of actions (batch_size,)
            next_states: Batch of next states (batch_size, state_dim)

        Returns:
            Training loss
        """
        predicted = self.forward(states, actions)
        loss = nn.functional.mse_loss(predicted, next_states)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update running error statistics
        with torch.no_grad():
            batch_error = (predicted - next_states).pow(2).mean(dim=-1)
            batch_mean = batch_error.mean().item()
            batch_var = batch_error.var().item() if len(batch_error) > 1 else 0.0

            # Welford's online algorithm
            self.update_count += 1
            delta = batch_mean - self.error_mean
            self.error_mean += delta / self.update_count
            self.error_var += (batch_var - self.error_var) / self.update_count

        return loss.item()

    def reset_statistics(self):
        """Reset running statistics."""
        self.error_mean = 0.0
        self.error_var = 1.0
        self.update_count = 0
