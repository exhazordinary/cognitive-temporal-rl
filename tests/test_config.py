"""Tests for the experiment config registry."""

import pytest

from src.experiments.surnor_config import (
    ENV_DEFAULTS,
    SURNOR_EXPERIMENTS,
    build_experiments,
    get_experiment,
)


def test_build_experiments_applies_env_defaults():
    experiments = build_experiments("CartPole-v1")

    assert set(experiments) == set(SURNOR_EXPERIMENTS)
    for config in experiments.values():
        assert config.env_name == "CartPole-v1"
        assert config.total_timesteps == ENV_DEFAULTS["CartPole-v1"]["total_timesteps"]
        assert config.ent_coef == ENV_DEFAULTS["CartPole-v1"]["ent_coef"]


def test_build_experiments_preserves_experiment_overrides():
    experiments = build_experiments("Acrobot-v1")

    assert experiments["baseline"].lr_mode == "none"
    assert experiments["surnor_stabilize"].lr_mode == "stabilize"
    assert experiments["surnor_adaptive"].lr_mode == "adaptive"
    assert experiments["surnor_adaptive_sensitive"].vol_change_threshold == 1.5


def test_build_experiments_does_not_mutate_base_registry():
    build_experiments("CartPole-v1")

    assert SURNOR_EXPERIMENTS["baseline"].env_name == "LunarLander-v3"


def test_unknown_env_rejected():
    with pytest.raises(ValueError, match="Unknown environment"):
        build_experiments("Pendulum-v1")


def test_get_experiment_with_env():
    config = get_experiment("surnor_stabilize", "Acrobot-v1")

    assert config.env_name == "Acrobot-v1"
    assert config.lr_mode == "stabilize"
