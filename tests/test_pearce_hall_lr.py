"""Tests for Pearce-Hall learning rate modulation."""

import pytest
import torch
from src.modulators.pearce_hall_lr import PearceHallLR


class TestPearceHallLR:
    """Tests for Pearce-Hall LR modulator."""

    def test_init(self):
        """Test initialization with default values."""
        lr = PearceHallLR()
        assert lr.base_lr == 3e-4
        assert lr.min_multiplier == 0.5
        assert lr.max_multiplier == 2.0
        assert lr.invert is False

    def test_neutral_alpha(self):
        """Test that alpha=1.0 gives base LR."""
        lr = PearceHallLR(base_lr=1e-3)
        result = lr.compute_lr(alpha=1.0)
        assert result == 1e-3  # No modulation

    def test_high_alpha_increases_lr(self):
        """Test Pearce-Hall mode: high alpha -> high LR."""
        lr = PearceHallLR(base_lr=1e-3, invert=False)

        # Alpha = 1.5 should increase LR
        result = lr.compute_lr(alpha=1.5)
        assert result == 1.5e-3

    def test_low_alpha_decreases_lr(self):
        """Test Pearce-Hall mode: low alpha -> low LR."""
        lr = PearceHallLR(base_lr=1e-3, invert=False)

        # Alpha = 0.5 should decrease LR
        result = lr.compute_lr(alpha=0.5)
        assert result == 0.5e-3

    def test_invert_mode(self):
        """Test stabilization mode: high alpha -> low LR."""
        lr = PearceHallLR(base_lr=1e-3, invert=True)

        # Alpha = 1.5 should DECREASE LR when inverted
        result = lr.compute_lr(alpha=1.5)
        assert result < 1e-3

        # Alpha = 0.5 should INCREASE LR when inverted
        result = lr.compute_lr(alpha=0.5)
        assert result > 1e-3

    def test_clamping_max(self):
        """Test that LR is clamped to max."""
        lr = PearceHallLR(base_lr=1e-3, max_multiplier=2.0)

        # Very high alpha should be clamped
        result = lr.compute_lr(alpha=10.0)
        assert result == 2e-3  # Clamped to max_multiplier * base_lr

    def test_clamping_min(self):
        """Test that LR is clamped to min."""
        lr = PearceHallLR(base_lr=1e-3, min_multiplier=0.5)

        # Very low alpha should be clamped
        result = lr.compute_lr(alpha=0.1)
        assert result == 0.5e-3  # Clamped to min_multiplier * base_lr

    def test_apply_to_optimizer(self):
        """Test applying modulated LR to optimizer."""
        lr = PearceHallLR(base_lr=1e-3)

        # Create a simple optimizer
        params = [torch.nn.Parameter(torch.randn(10))]
        optimizer = torch.optim.Adam(params, lr=1e-3)

        # Apply modulation
        new_lr = lr.apply_to_optimizer(optimizer, alpha=1.5)

        assert new_lr == 1.5e-3
        assert optimizer.param_groups[0]['lr'] == 1.5e-3

    def test_history_tracking(self):
        """Test that LR history is tracked."""
        lr = PearceHallLR()

        lr.compute_lr(alpha=1.0)
        lr.compute_lr(alpha=1.5)
        lr.compute_lr(alpha=0.8)

        assert len(lr.lr_history) == 3
        assert len(lr.alpha_history) == 3

    def test_reset(self):
        """Test reset clears history."""
        lr = PearceHallLR()

        lr.compute_lr(alpha=1.0)
        lr.compute_lr(alpha=1.5)

        lr.reset()

        assert len(lr.lr_history) == 0
        assert lr.current_multiplier == 1.0

    def test_stats(self):
        """Test statistics computation."""
        lr = PearceHallLR(base_lr=1e-3)

        lr.compute_lr(alpha=1.0)
        lr.compute_lr(alpha=1.5)

        stats = lr.get_stats()

        assert stats['base_lr'] == 1e-3
        assert stats['n_updates'] == 2
        assert 'mean_lr' in stats
        assert 'current_multiplier' in stats
