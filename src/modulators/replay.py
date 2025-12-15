"""Salience-based prioritized experience replay."""

import torch
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class Transition:
    """A single transition in the replay buffer."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    td_error: float = 0.0
    salience: float = 0.0


class SalienceReplay:
    """Prioritized replay buffer using salience scores.

    Combines TD-error priority with entropy-based salience for
    experience prioritization.
    """

    def __init__(
        self,
        capacity: int = 100000,
        alpha: float = 0.5,      # Balance between TD-error and salience
        beta: float = 0.6,       # Priority exponent
        beta_increment: float = 0.001,
        epsilon: float = 1e-6,
    ):
        """Initialize the salience replay buffer.

        Args:
            capacity: Maximum buffer size
            alpha: Weight for TD-error vs salience (0=all salience, 1=all TD)
            beta: Initial importance sampling exponent
            beta_increment: Per-sample increase in beta toward 1.0
            epsilon: Small constant for numerical stability
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon

        # Storage
        self.buffer: list[Transition] = []
        self.priorities: np.ndarray = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        td_error: float = 0.0,
        salience: float = 0.0,
    ) -> None:
        """Add a transition to the buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode done flag
            td_error: TD error for this transition
            salience: Salience score from entropy clock
        """
        transition = Transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            td_error=td_error,
            salience=salience,
        )

        # Compute priority
        priority = self._compute_priority(td_error, salience)

        if self.size < self.capacity:
            self.buffer.append(transition)
            self.size += 1
        else:
            self.buffer[self.position] = transition

        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity

    def _compute_priority(self, td_error: float, salience: float) -> float:
        """Compute combined priority from TD-error and salience.

        Args:
            td_error: Temporal difference error
            salience: Entropy-based salience score

        Returns:
            Combined priority score
        """
        td_priority = abs(td_error) + self.epsilon
        salience_priority = salience + self.epsilon
        combined = self.alpha * td_priority + (1 - self.alpha) * salience_priority
        return combined

    def sample(
        self,
        batch_size: int,
    ) -> Tuple[list[Transition], np.ndarray, np.ndarray]:
        """Sample a batch of transitions based on priority.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (transitions, indices, importance sampling weights)
        """
        if self.size < batch_size:
            batch_size = self.size

        # Compute sampling probabilities
        priorities = self.priorities[:self.size]
        probs = priorities ** self.beta
        probs = probs / probs.sum()

        # Sample indices
        indices = np.random.choice(self.size, size=batch_size, p=probs, replace=False)

        # Compute importance sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize

        # Get transitions
        transitions = [self.buffer[i] for i in indices]

        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        return transitions, indices, weights

    def update_priorities(
        self,
        indices: np.ndarray,
        td_errors: np.ndarray,
        saliences: Optional[np.ndarray] = None,
    ) -> None:
        """Update priorities for sampled transitions.

        Args:
            indices: Indices of transitions to update
            td_errors: New TD errors
            saliences: New salience scores (optional, uses stored if None)
        """
        for i, idx in enumerate(indices):
            transition = self.buffer[idx]
            transition.td_error = td_errors[i]

            if saliences is not None:
                transition.salience = saliences[i]

            self.priorities[idx] = self._compute_priority(
                transition.td_error,
                transition.salience,
            )

    def __len__(self) -> int:
        return self.size
