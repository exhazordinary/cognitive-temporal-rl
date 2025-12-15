"""Exploration modulation based on internal entropy state."""

import torch
import numpy as np
from typing import Optional


class SalienceExploration:
    """Modulates exploration based on internal entropy state.

    Counterintuitive but principled approach:
    - High internal entropy → already in novel territory → exploit more
    - Low internal entropy → too confident/stuck → explore more

    This prevents compounding chaos and encourages escaping local patterns.
    """

    def __init__(
        self,
        base_temperature: float = 1.0,
        lambda_scale: float = 0.3,     # How much entropy affects exploration
        min_temperature: float = 0.1,
        max_temperature: float = 3.0,
        history_window: int = 100,
    ):
        """Initialize the exploration modulator.

        Args:
            base_temperature: Base policy temperature
            lambda_scale: Strength of entropy influence on temperature
            min_temperature: Minimum allowed temperature
            max_temperature: Maximum allowed temperature
            history_window: Window for computing entropy statistics
        """
        self.base_temperature = base_temperature
        self.lambda_scale = lambda_scale
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.history_window = history_window

        self.entropy_history: list[float] = []
        self.current_temperature: float = base_temperature
        self.temperature_history: list[float] = []

    def get_temperature(self, entropy: float) -> float:
        """Get modulated temperature based on current entropy state.

        Args:
            entropy: Current entropy from the entropy clock

        Returns:
            Modulated temperature for policy sampling
        """
        self.entropy_history.append(entropy)

        # Need enough history to compute statistics
        if len(self.entropy_history) < 10:
            self.current_temperature = self.base_temperature
            self.temperature_history.append(self.current_temperature)
            return self.current_temperature

        # Use recent window for statistics
        recent = self.entropy_history[-self.history_window:]
        mean_entropy = np.mean(recent)
        std_entropy = np.std(recent) + 1e-8

        # Normalize current entropy
        normalized = (entropy - mean_entropy) / std_entropy

        # High normalized entropy → lower temperature (exploit)
        # Low normalized entropy → higher temperature (explore)
        # Using tanh for smooth bounded scaling
        temperature = self.base_temperature * (1 - self.lambda_scale * np.tanh(normalized))

        # Clamp to valid range
        temperature = max(self.min_temperature, min(self.max_temperature, temperature))
        self.current_temperature = temperature
        self.temperature_history.append(temperature)

        return temperature

    def apply_temperature(
        self,
        logits: torch.Tensor,
        entropy: float,
    ) -> torch.Tensor:
        """Apply temperature scaling to policy logits.

        Args:
            logits: Raw policy logits from the network
            entropy: Current entropy from entropy clock

        Returns:
            Temperature-scaled logits
        """
        temperature = self.get_temperature(entropy)
        return logits / temperature

    def sample_action(
        self,
        logits: torch.Tensor,
        entropy: float,
    ) -> tuple[int, torch.Tensor]:
        """Sample an action with temperature-modulated exploration.

        Args:
            logits: Raw policy logits
            entropy: Current entropy from entropy clock

        Returns:
            Tuple of (sampled action, action probabilities)
        """
        scaled_logits = self.apply_temperature(logits, entropy)
        probs = torch.softmax(scaled_logits, dim=-1)
        action = torch.multinomial(probs, 1).item()
        return action, probs

    def get_epsilon(self, entropy: float) -> float:
        """Get epsilon for epsilon-greedy exploration.

        Alternative interface for algorithms using epsilon-greedy.

        Args:
            entropy: Current entropy from entropy clock

        Returns:
            Epsilon value (higher = more exploration)
        """
        temperature = self.get_temperature(entropy)
        # Map temperature to epsilon: higher temp → higher epsilon
        # Using sigmoid-like mapping
        epsilon = 0.5 * (1 + np.tanh((temperature - self.base_temperature) / self.base_temperature))
        return np.clip(epsilon, 0.01, 0.5)

    def reset(self) -> None:
        """Reset for new episode."""
        self.entropy_history.clear()
        self.temperature_history.clear()
        self.current_temperature = self.base_temperature

    def get_stats(self) -> dict:
        """Get current modulator statistics."""
        return {
            "base_temperature": self.base_temperature,
            "current_temperature": self.current_temperature,
            "mean_temperature": (
                np.mean(self.temperature_history)
                if self.temperature_history else self.base_temperature
            ),
            "entropy_history_len": len(self.entropy_history),
        }
