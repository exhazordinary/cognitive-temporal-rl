"""Integration tests for SurNoRPPO.

The key regression test here guards against the silent-no-op bug where the
callback wrote the modulated LR directly into the optimizer, only for SB3's
``PPO.train()`` to overwrite it from the (constant) lr_schedule before any
gradient step.
"""

import numpy as np
import pytest

from src.agents.surnor_ppo import SurNoRPPO


@pytest.fixture(scope="module")
def trained_agent():
    """Short training run on CartPole (cheap, no box2d needed)."""
    agent = SurNoRPPO(
        env_name="CartPole-v1",
        n_envs=2,
        n_steps_total=256,
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


def test_surprise_counted_in_env_steps(trained_agent):
    """The surprise module sees every env transition (n_envs per vec-step)."""
    assert trained_agent.surprise_module.step_count == 1024


def test_last_obs_is_pre_step_observation():
    """Guard the semi-private SB3 API the callback relies on:
    inside _on_step, model._last_obs must still hold the PRE-step
    observations (differing from new_obs on at least one step)."""
    from stable_baselines3.common.callbacks import BaseCallback

    class ProbeCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.saw_difference = False

        def _on_step(self) -> bool:
            if not np.array_equal(self.model._last_obs, self.locals["new_obs"]):
                self.saw_difference = True
            return True

    agent = SurNoRPPO(
        env_name="CartPole-v1",
        n_envs=2,
        n_steps_total=128,
        seed=1,
        device="cpu",
    )
    probe = ProbeCallback()
    agent.model.learn(total_timesteps=128, callback=probe)
    agent.close()

    assert probe.saw_difference
