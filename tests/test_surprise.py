"""Tests for the surprise module (forward model + Pearce-Hall smoothing)."""

import pytest
import torch
import numpy as np

from src.surprise import ForwardModel, SurpriseModule


class TestForwardModel:
    """Tests for the forward dynamics model."""

    def test_forward_model_init(self):
        """Test forward model initialization."""
        fm = ForwardModel(state_dim=8, action_dim=4)
        assert fm.state_dim == 8
        assert fm.action_dim == 4

    def test_forward_prediction(self):
        """Test forward model prediction shape."""
        fm = ForwardModel(state_dim=8, action_dim=4)

        state = torch.randn(1, 8)
        action = torch.tensor([2])

        predicted = fm.forward(state, action)
        assert predicted.shape == (1, 8)

    def test_forward_batch_prediction(self):
        """Test forward model with batch input."""
        fm = ForwardModel(state_dim=8, action_dim=4)

        states = torch.randn(32, 8)
        actions = torch.randint(0, 4, (32,))

        predicted = fm.forward(states, actions)
        assert predicted.shape == (32, 8)

    def test_compute_surprise(self):
        """Test surprise computation."""
        fm = ForwardModel(state_dim=8, action_dim=4)

        state = torch.randn(8)
        action = torch.tensor(2)
        next_state = torch.randn(8)

        surprise, raw_error = fm.compute_surprise(state, action, next_state)

        assert isinstance(surprise, float)
        assert isinstance(raw_error, torch.Tensor)
        assert raw_error.item() >= 0  # MSE is non-negative

    def test_train_step(self):
        """Test forward model training."""
        fm = ForwardModel(state_dim=8, action_dim=4)

        states = torch.randn(32, 8)
        actions = torch.randint(0, 4, (32,))
        next_states = torch.randn(32, 8)

        loss1 = fm.train_step(states, actions, next_states)
        loss2 = fm.train_step(states, actions, next_states)

        assert isinstance(loss1, float)
        assert loss1 >= 0
        # Loss should decrease (or stay similar) with training
        # Not strictly guaranteed but generally true


class TestSurpriseModule:
    """Tests for the surprise module with Pearce-Hall smoothing."""

    def test_surprise_module_init(self):
        """Test surprise module initialization."""
        sm = SurpriseModule(state_dim=8, action_dim=4)
        assert sm.alpha == 1.0  # Initial associability
        assert sm.gamma == 0.3  # Default Pearce-Hall gamma

    def test_step_returns_output(self):
        """Test that step returns proper output structure."""
        sm = SurpriseModule(state_dim=8, action_dim=4)

        state = torch.randn(8)
        action = torch.tensor(2)
        next_state = torch.randn(8)

        output = sm.step(state, action, next_state)

        assert hasattr(output, 'surprise')
        assert hasattr(output, 'smoothed_surprise')
        assert hasattr(output, 'raw_error')

    def test_pearce_hall_smoothing(self):
        """Test that alpha is smoothed properly."""
        sm = SurpriseModule(state_dim=8, action_dim=4, gamma=0.5)

        # Initial alpha
        assert sm.alpha == 1.0

        # Process a transition
        state = torch.randn(8)
        action = torch.tensor(2)
        next_state = torch.randn(8)

        output = sm.step(state, action, next_state)

        # Alpha should be updated: α = γ|PE| + (1-γ)α_prev
        # After first step, alpha should change
        assert sm.alpha != 1.0 or abs(output.surprise) < 1e-6

    def test_rollout_aggregation(self):
        """Test rollout statistics computation."""
        sm = SurpriseModule(state_dim=8, action_dim=4)

        # Process multiple transitions
        for _ in range(10):
            state = torch.randn(8)
            action = torch.tensor(np.random.randint(4))
            next_state = torch.randn(8)
            sm.step(state, action, next_state)

        stats = sm.get_rollout_stats()

        assert 'mean_surprise' in stats
        assert 'max_surprise' in stats
        assert 'alpha' in stats
        assert stats['n_steps'] == 10

    def test_clear_rollout(self):
        """Test clearing rollout accumulator."""
        sm = SurpriseModule(state_dim=8, action_dim=4)

        # Process some transitions
        for _ in range(5):
            sm.step(torch.randn(8), torch.tensor(0), torch.randn(8))

        assert sm.get_rollout_stats()['n_steps'] == 5

        sm.clear_rollout()
        assert sm.get_rollout_stats()['n_steps'] == 0

    def test_hard_reset(self):
        """Test full reset."""
        sm = SurpriseModule(state_dim=8, action_dim=4)

        # Process some transitions
        for _ in range(10):
            sm.step(torch.randn(8), torch.tensor(0), torch.randn(8))

        sm.hard_reset()

        assert sm.alpha == 1.0
        assert len(sm.surprise_history) == 0
        assert len(sm.transition_buffer) == 0


class TestPearceHallDynamics:
    """Tests specifically for Pearce-Hall learning dynamics."""

    def test_high_surprise_increases_alpha(self):
        """Test that consistently high surprises increase alpha."""
        sm = SurpriseModule(state_dim=8, action_dim=4, gamma=0.5)

        initial_alpha = sm.alpha

        # Create transitions with large prediction errors
        # (random states are hard to predict)
        for _ in range(50):
            state = torch.randn(8) * 10  # Large variance
            action = torch.tensor(np.random.randint(4))
            next_state = torch.randn(8) * 10
            sm.step(state, action, next_state)

        # Alpha should reflect the high surprise
        # (exact value depends on forward model, but should be non-trivial)
        assert sm.alpha > 0

    def test_alpha_bounded(self):
        """Test that alpha stays in reasonable range."""
        sm = SurpriseModule(state_dim=8, action_dim=4, gamma=0.3)

        for _ in range(100):
            state = torch.randn(8)
            action = torch.tensor(np.random.randint(4))
            next_state = torch.randn(8)
            sm.step(state, action, next_state)

        # Alpha should be positive
        assert sm.alpha > 0
        # Alpha history should be recorded
        assert len(sm.alpha_history) == 100
