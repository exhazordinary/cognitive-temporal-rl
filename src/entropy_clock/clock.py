"""Entropy-based internal clock module for temporal cognition."""

import torch
from typing import Optional, Tuple, NamedTuple
from .buffers import StateBuffer


class ClockOutput(NamedTuple):
    """Output from the entropy clock at each timestep."""
    entropy: float                    # Current entropy estimate
    delta_entropy: float              # Change from previous timestep
    salience_score: float             # Normalized absolute delta
    internal_time_delta: float        # Subjective time elapsed
    raw_entropy_tensor: torch.Tensor  # For debugging/visualization


class EntropyClockModule:
    """Entropy-based internal clock that measures 'cognitive time'.

    Computes entropy over a rolling window of states and produces
    salience signals based on entropy changes. High entropy changes
    (either increases or decreases) indicate salient moments.

    The internal time delta scales with salience - more salient moments
    contribute more to 'experienced time'.
    """

    def __init__(
        self,
        window_size: int = 50,
        min_samples: int = 10,
        baseline_momentum: float = 0.99,
        epsilon: float = 1e-6,
        salience_scale: float = 1.0,
    ):
        """Initialize the entropy clock.

        Args:
            window_size: Number of states to keep in rolling window
            min_samples: Minimum samples before computing entropy
            baseline_momentum: EMA momentum for baseline entropy (0-1)
            epsilon: Small constant for numerical stability
            salience_scale: Multiplier for salience scores
        """
        self.window_size = window_size
        self.min_samples = min_samples
        self.baseline_momentum = baseline_momentum
        self.epsilon = epsilon
        self.salience_scale = salience_scale

        # State
        self.buffer = StateBuffer(window_size)
        self.prev_entropy: Optional[float] = None
        self.baseline_entropy: Optional[float] = None
        self.baseline_delta: Optional[float] = None  # Running avg of |delta|

        # History for analysis
        self.entropy_history: list[float] = []
        self.salience_history: list[float] = []
        self.internal_time: float = 0.0  # Cumulative internal time

    def _compute_covariance_entropy(self, states: torch.Tensor) -> torch.Tensor:
        """Compute entropy using Gaussian approximation.

        For a multivariate Gaussian, entropy = 0.5 * log(det(2*pi*e*Σ))
        We simplify to 0.5 * log(det(Σ)) since the constant doesn't affect deltas.

        Args:
            states: Tensor of shape (n_samples, state_dim)

        Returns:
            Scalar tensor with entropy estimate
        """
        # Center the data
        states_centered = states - states.mean(dim=0, keepdim=True)

        # Compute covariance matrix
        n = states.shape[0]
        cov = (states_centered.T @ states_centered) / (n - 1)

        # Add regularization for numerical stability
        cov = cov + self.epsilon * torch.eye(cov.shape[0])

        # Entropy = 0.5 * log(det(Σ))
        # Using logdet for numerical stability
        sign, logdet = torch.linalg.slogdet(cov)

        # If determinant is negative (shouldn't happen with regularization),
        # fall back to sum of log eigenvalues
        if sign <= 0:
            eigenvalues = torch.linalg.eigvalsh(cov)
            eigenvalues = torch.clamp(eigenvalues, min=self.epsilon)
            logdet = torch.sum(torch.log(eigenvalues))

        return 0.5 * logdet

    def step(self, state: torch.Tensor) -> ClockOutput:
        """Process a new state and return clock outputs.

        Args:
            state: Current state vector

        Returns:
            ClockOutput with entropy metrics and salience score
        """
        # Add state to buffer
        self.buffer.push(state)

        # Check if we have enough samples
        if not self.buffer.is_ready(self.min_samples):
            return ClockOutput(
                entropy=0.0,
                delta_entropy=0.0,
                salience_score=0.0,
                internal_time_delta=0.0,
                raw_entropy_tensor=torch.tensor(0.0),
            )

        # Compute current entropy
        states = self.buffer.get_states()
        entropy_tensor = self._compute_covariance_entropy(states)
        entropy = entropy_tensor.item()

        # Update history
        self.entropy_history.append(entropy)

        # Compute delta entropy
        if self.prev_entropy is None:
            delta_entropy = 0.0
        else:
            delta_entropy = entropy - self.prev_entropy

        # Update baselines with exponential moving average
        if self.baseline_entropy is None:
            self.baseline_entropy = entropy
            self.baseline_delta = abs(delta_entropy) + self.epsilon
        else:
            self.baseline_entropy = (
                self.baseline_momentum * self.baseline_entropy
                + (1 - self.baseline_momentum) * entropy
            )
            self.baseline_delta = (
                self.baseline_momentum * self.baseline_delta
                + (1 - self.baseline_momentum) * (abs(delta_entropy) + self.epsilon)
            )

        # Compute salience as normalized absolute delta
        # Both increases AND decreases in entropy are salient
        salience_score = (
            self.salience_scale * abs(delta_entropy) / self.baseline_delta
        )

        # Internal time: base tick + salience contribution
        # More salient moments = more "internal time" passes
        internal_time_delta = 1.0 + salience_score
        self.internal_time += internal_time_delta

        # Update state
        self.prev_entropy = entropy
        self.salience_history.append(salience_score)

        return ClockOutput(
            entropy=entropy,
            delta_entropy=delta_entropy,
            salience_score=salience_score,
            internal_time_delta=internal_time_delta,
            raw_entropy_tensor=entropy_tensor,
        )

    def reset(self) -> None:
        """Reset the clock for a new episode."""
        self.buffer.reset()
        self.prev_entropy = None
        # Keep baseline estimates across episodes for stability
        self.entropy_history.clear()
        self.salience_history.clear()
        self.internal_time = 0.0

    def hard_reset(self) -> None:
        """Full reset including baselines (for new experiment)."""
        self.reset()
        self.baseline_entropy = None
        self.baseline_delta = None

    def get_stats(self) -> dict:
        """Get current clock statistics."""
        return {
            "internal_time": self.internal_time,
            "baseline_entropy": self.baseline_entropy,
            "baseline_delta": self.baseline_delta,
            "buffer_size": len(self.buffer),
            "entropy_history_len": len(self.entropy_history),
            "mean_salience": (
                sum(self.salience_history) / len(self.salience_history)
                if self.salience_history else 0.0
            ),
        }
