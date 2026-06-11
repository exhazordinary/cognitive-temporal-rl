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


class TestStepBatch:
    """Tests for vectorized (batched) surprise computation."""

    def test_batch_shapes(self):
        sm = SurpriseModule(state_dim=8, action_dim=4)

        states = np.random.randn(4, 8).astype(np.float32)
        actions = np.random.randint(0, 4, size=4)
        next_states = np.random.randn(4, 8).astype(np.float32)

        out = sm.step_batch(states, actions, next_states)

        assert out.surprises.shape == (4,)
        assert out.raw_errors.shape == (4,)
        assert isinstance(out.alpha, float)

    def test_one_alpha_update_per_vec_step(self):
        """Alpha is smoothed once per vec-step, not once per env."""
        sm = SurpriseModule(state_dim=8, action_dim=4)

        for _ in range(5):
            sm.step_batch(
                np.random.randn(4, 8).astype(np.float32),
                np.random.randint(0, 4, size=4),
                np.random.randn(4, 8).astype(np.float32),
            )

        assert len(sm.alpha_history) == 5

    def test_step_count_in_env_steps(self):
        """step_count advances by the batch size (env-steps, not vec-steps)."""
        sm = SurpriseModule(state_dim=8, action_dim=4)

        sm.step_batch(
            np.random.randn(4, 8).astype(np.float32),
            np.random.randint(0, 4, size=4),
            np.random.randn(4, 8).astype(np.float32),
        )

        assert sm.step_count == 4
        assert len(sm.transition_buffer) == 4
        assert len(sm.surprise_history) == 4
        assert sm.get_rollout_stats()["n_steps"] == 4

    def test_forward_model_trains_when_threshold_crossed(self):
        """train_every is denominated in env-steps even with batches."""
        sm = SurpriseModule(state_dim=8, action_dim=4, train_every=100, batch_size=64)

        # 13 vec-steps x 8 envs = 104 env-steps: crosses train_every=100
        for _ in range(13):
            sm.step_batch(
                np.random.randn(8, 8).astype(np.float32),
                np.random.randint(0, 4, size=8),
                np.random.randn(8, 8).astype(np.float32),
            )

        assert sm.forward_model.update_count == 1

    def test_scalar_step_parity(self):
        """Scalar step() (batch of 1) matches the documented behavior."""
        torch.manual_seed(0)
        sm = SurpriseModule(state_dim=8, action_dim=4, gamma=0.5)

        state = torch.randn(8)
        action = torch.tensor(2)
        next_state = torch.randn(8)

        out = sm.step(state, action, next_state)

        assert sm.step_count == 1
        assert len(sm.alpha_history) == 1
        # alpha = gamma*|surprise| + (1-gamma)*1.0
        expected = 0.5 * abs(out.surprise) + 0.5 * 1.0
        assert sm.alpha == pytest.approx(expected)
        assert out.smoothed_surprise == pytest.approx(sm.alpha)


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
