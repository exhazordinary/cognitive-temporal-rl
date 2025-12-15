"""Rolling state buffer for entropy computation."""

import torch
from typing import Optional


class StateBuffer:
    """Maintains a rolling window of recent states for entropy estimation.

    Attributes:
        window_size: Maximum number of states to store
        state_dim: Dimensionality of state vectors
        buffer: Tensor storing recent states
        count: Number of states currently in buffer
    """

    def __init__(self, window_size: int = 50, state_dim: Optional[int] = None):
        """Initialize the state buffer.

        Args:
            window_size: Maximum number of states to keep
            state_dim: Dimension of state vectors (can be set on first push)
        """
        self.window_size = window_size
        self.state_dim = state_dim
        self.buffer: Optional[torch.Tensor] = None
        self.count = 0
        self._index = 0  # Current write position (circular buffer)

    def push(self, state: torch.Tensor) -> None:
        """Add a state to the buffer.

        Args:
            state: State vector of shape (state_dim,) or (1, state_dim)
        """
        # Flatten if needed
        state = state.flatten()

        # Initialize buffer on first push
        if self.buffer is None:
            self.state_dim = state.shape[0]
            self.buffer = torch.zeros(self.window_size, self.state_dim)

        # Add state to circular buffer
        self.buffer[self._index] = state
        self._index = (self._index + 1) % self.window_size
        self.count = min(self.count + 1, self.window_size)

    def get_states(self) -> torch.Tensor:
        """Get all states currently in buffer.

        Returns:
            Tensor of shape (count, state_dim) with states in chronological order
        """
        if self.buffer is None or self.count == 0:
            raise ValueError("Buffer is empty")

        if self.count < self.window_size:
            # Buffer not yet full, return from start
            return self.buffer[:self.count].clone()
        else:
            # Buffer full, reorder to chronological
            return torch.cat([
                self.buffer[self._index:],
                self.buffer[:self._index]
            ], dim=0)

    def is_ready(self, min_samples: int = 10) -> bool:
        """Check if buffer has enough samples for entropy estimation.

        Args:
            min_samples: Minimum samples needed

        Returns:
            True if buffer has at least min_samples states
        """
        return self.count >= min_samples

    def reset(self) -> None:
        """Clear the buffer."""
        if self.buffer is not None:
            self.buffer.zero_()
        self.count = 0
        self._index = 0

    def __len__(self) -> int:
        return self.count
