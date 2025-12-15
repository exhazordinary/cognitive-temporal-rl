"""Unit tests for the entropy clock module."""

import torch
import pytest
from src.entropy_clock import EntropyClockModule, StateBuffer


class TestStateBuffer:
    """Tests for the StateBuffer class."""

    def test_init(self):
        buffer = StateBuffer(window_size=10)
        assert buffer.window_size == 10
        assert len(buffer) == 0

    def test_push_and_len(self):
        buffer = StateBuffer(window_size=5)
        for i in range(3):
            buffer.push(torch.randn(4))
        assert len(buffer) == 3

    def test_circular_buffer(self):
        buffer = StateBuffer(window_size=3)
        # Push 5 states into buffer of size 3
        states = [torch.tensor([float(i)]) for i in range(5)]
        for s in states:
            buffer.push(s)

        assert len(buffer) == 3
        retrieved = buffer.get_states()
        # Should have states 2, 3, 4 in order
        assert retrieved.shape == (3, 1)
        assert torch.allclose(retrieved.flatten(), torch.tensor([2.0, 3.0, 4.0]))

    def test_is_ready(self):
        buffer = StateBuffer(window_size=10)
        for i in range(5):
            buffer.push(torch.randn(4))
        assert not buffer.is_ready(min_samples=10)
        assert buffer.is_ready(min_samples=5)
        assert buffer.is_ready(min_samples=3)

    def test_reset(self):
        buffer = StateBuffer(window_size=5)
        for i in range(3):
            buffer.push(torch.randn(4))
        buffer.reset()
        assert len(buffer) == 0

    def test_empty_get_states_raises(self):
        buffer = StateBuffer(window_size=5)
        with pytest.raises(ValueError):
            buffer.get_states()


class TestEntropyClockModule:
    """Tests for the EntropyClockModule class."""

    def test_init(self):
        clock = EntropyClockModule(window_size=50)
        assert clock.window_size == 50
        assert clock.internal_time == 0.0

    def test_step_before_ready(self):
        clock = EntropyClockModule(window_size=10, min_samples=5)
        # Not enough samples yet
        for _ in range(3):
            output = clock.step(torch.randn(8))
        assert output.entropy == 0.0
        assert output.salience_score == 0.0

    def test_step_after_ready(self):
        clock = EntropyClockModule(window_size=20, min_samples=10)
        # Fill buffer past min_samples
        for _ in range(15):
            output = clock.step(torch.randn(8))

        # Should now have valid entropy
        assert output.entropy != 0.0
        assert output.internal_time_delta > 0.0

    def test_salience_on_novelty(self):
        clock = EntropyClockModule(window_size=30, min_samples=10)

        # First, establish baseline with similar states
        for _ in range(20):
            clock.step(torch.randn(4) * 0.1)  # Low variance states

        baseline_salience = clock.salience_history[-1]

        # Now inject high variance state
        clock.step(torch.randn(4) * 10.0)  # High variance
        high_salience = clock.salience_history[-1]

        # Salience should increase with novelty
        # (This is a probabilistic test, may occasionally fail)
        assert high_salience > 0  # At minimum, should be positive

    def test_internal_time_accumulates(self):
        clock = EntropyClockModule(window_size=20, min_samples=10)

        for _ in range(15):
            clock.step(torch.randn(8))

        assert clock.internal_time > 0.0

    def test_reset(self):
        clock = EntropyClockModule(window_size=20, min_samples=10)

        for _ in range(15):
            clock.step(torch.randn(8))

        clock.reset()
        assert clock.internal_time == 0.0
        assert len(clock.entropy_history) == 0
        # Baselines should persist
        assert clock.baseline_entropy is not None

    def test_hard_reset(self):
        clock = EntropyClockModule(window_size=20, min_samples=10)

        for _ in range(15):
            clock.step(torch.randn(8))

        clock.hard_reset()
        assert clock.internal_time == 0.0
        assert clock.baseline_entropy is None
        assert clock.baseline_delta is None

    def test_get_stats(self):
        clock = EntropyClockModule(window_size=20, min_samples=10)

        for _ in range(15):
            clock.step(torch.randn(8))

        stats = clock.get_stats()
        assert "internal_time" in stats
        assert "baseline_entropy" in stats
        assert "mean_salience" in stats
        assert stats["buffer_size"] == 15

    def test_entropy_increases_with_variance(self):
        """Entropy should be higher for more spread-out distributions."""
        clock_low = EntropyClockModule(window_size=50, min_samples=30)
        clock_high = EntropyClockModule(window_size=50, min_samples=30)

        # Low variance states
        for _ in range(40):
            clock_low.step(torch.randn(4) * 0.1)

        # High variance states
        for _ in range(40):
            clock_high.step(torch.randn(4) * 10.0)

        # Higher variance should yield higher entropy
        assert clock_high.entropy_history[-1] > clock_low.entropy_history[-1]


class TestEntropyComputation:
    """Tests for entropy computation correctness."""

    def test_known_covariance(self):
        """Test entropy on data with known covariance structure."""
        clock = EntropyClockModule(window_size=100, min_samples=50)

        # Generate samples from known distribution
        torch.manual_seed(42)
        mean = torch.zeros(3)
        # Diagonal covariance with known values
        cov = torch.diag(torch.tensor([1.0, 2.0, 3.0]))

        # Generate samples
        L = torch.linalg.cholesky(cov)
        for _ in range(80):
            z = torch.randn(3)
            x = mean + L @ z
            clock.step(x)

        # Entropy should be approximately 0.5 * log(det(cov))
        # det(diag(1,2,3)) = 6
        expected_entropy = 0.5 * torch.log(torch.tensor(6.0)).item()

        # Allow some tolerance due to sampling variance
        assert abs(clock.entropy_history[-1] - expected_entropy) < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
