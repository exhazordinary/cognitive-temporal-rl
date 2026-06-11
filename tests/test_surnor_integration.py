"""Integration tests for SurNoRPPO.

The key regression test here guards against the silent-no-op bug where the
callback wrote the modulated LR directly into the optimizer, only for SB3's
``PPO.train()`` to overwrite it from the (constant) lr_schedule before any
gradient step.
"""

import pytest

from src.agents.surnor_ppo import SurNoRPPO


@pytest.fixture(scope="module")
def trained_agent():
    """Short training run on CartPole (cheap, no box2d needed)."""
    agent = SurNoRPPO(
        env_name="CartPole-v1",
        n_steps=256,
        batch_size=64,
        use_lr_modulation=True,
        invert_lr=False,
        pearce_hall_gamma=0.3,
        seed=0,
        device="cpu",
        verbose=0,
    )
    agent.train(total_timesteps=1024)
    yield agent
    agent.close()


def test_modulated_lr_reaches_optimizer(trained_agent):
    """The LR used by the optimizer must be the callback's modulated LR,
    not the constant base LR re-applied by SB3's schedule."""
    callback = trained_agent.callback
    assert len(callback.update_lrs) > 0, "no LR modulations recorded"

    optimizer_lr = trained_agent.model.policy.optimizer.param_groups[0]["lr"]
    assert optimizer_lr == pytest.approx(callback.update_lrs[-1])


def test_modulation_actually_changes_lr(trained_agent):
    """Sanity: with modulation enabled, at least one update should deviate
    from the base LR (otherwise the regression test above is vacuous)."""
    base_lr = trained_agent.lr_modulator.base_lr
    assert any(lr != pytest.approx(base_lr) for lr in trained_agent.callback.update_lrs)


def test_episode_stats_collected(trained_agent):
    assert len(trained_agent.callback.episode_rewards) > 0
