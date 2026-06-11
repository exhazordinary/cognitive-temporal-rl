"""Experiment configurations for SurNoR-based experiments.

New experiments based on research synthesis:
1. Prediction error (surprise) instead of state entropy
2. Pearce-Hall smoothing
3. LR modulation at update time
4. Both directions tested (high surprise -> high/low LR)
"""

from dataclasses import dataclass, field, replace
from typing import Optional, List


@dataclass
class SurNoRConfig:
    """Configuration for SurNoR experiments."""
    # Environment
    env_name: str = "LunarLander-v3"
    total_timesteps: int = 200000
    # Use None for random seeds to avoid deterministic trajectory issue
    seeds: List[Optional[int]] = field(default_factory=lambda: [None] * 10)

    # Vectorization
    n_envs: int = 8
    vec_env_type: str = "dummy"  # "dummy" or "subproc"

    # PPO params
    learning_rate: float = 3e-4
    n_steps_total: int = 2048  # rollout size in env-steps, split across envs
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    ent_coef: float = 0.01

    # Surprise module params
    pearce_hall_gamma: float = 0.3  # How quickly alpha adapts
    forward_model_lr: float = 1e-3
    forward_hidden_dim: int = 64

    # LR modulation mode:
    #   none        - no modulation (baseline; forward model still trains)
    #   pearce_hall - high surprise -> higher LR
    #   stabilize   - high surprise -> lower LR
    #   adaptive    - volatility detector picks the direction per rollout
    lr_mode: str = "pearce_hall"
    lr_min_multiplier: float = 0.5
    lr_max_multiplier: float = 2.0

    # Volatility detector (adaptive mode only)
    vol_window_short: int = 20
    vol_window_long: int = 100
    vol_change_threshold: float = 2.0
    vol_noise_sensitivity: float = 0.5
    vol_volatility_boost: float = 0.5
    vol_rollout_window: int = 50

    # Intrinsic reward
    intrinsic_reward_scale: float = 0.0  # 0 = disabled

    # Experiment info
    experiment_name: str = "surnor_baseline"
    verbose: int = 0


# Predefined experiment configurations
SURNOR_EXPERIMENTS = {
    # Baseline: No modulation (just forward model training)
    "baseline": SurNoRConfig(
        experiment_name="baseline",
        lr_mode="none",
        intrinsic_reward_scale=0.0,
    ),

    # ===== PEARCE-HALL DIRECTION =====
    # High surprise -> Higher LR (learn more from surprising events)

    "surnor_pearce_hall": SurNoRConfig(
        experiment_name="surnor_pearce_hall",
        lr_mode="pearce_hall",
        pearce_hall_gamma=0.3,
    ),

    "surnor_ph_gamma_0.1": SurNoRConfig(
        experiment_name="surnor_ph_gamma_0.1",
        lr_mode="pearce_hall",
        pearce_hall_gamma=0.1,  # Slower adaptation
    ),

    "surnor_ph_gamma_0.5": SurNoRConfig(
        experiment_name="surnor_ph_gamma_0.5",
        lr_mode="pearce_hall",
        pearce_hall_gamma=0.5,  # Faster adaptation
    ),

    # ===== STABILIZATION DIRECTION =====
    # High surprise -> Lower LR (stabilize during chaos)

    "surnor_stabilize": SurNoRConfig(
        experiment_name="surnor_stabilize",
        lr_mode="stabilize",
        pearce_hall_gamma=0.3,
    ),

    "surnor_stab_gamma_0.1": SurNoRConfig(
        experiment_name="surnor_stab_gamma_0.1",
        lr_mode="stabilize",
        pearce_hall_gamma=0.1,
    ),

    # ===== ADAPTIVE: VOLATILITY VS NOISE =====
    # Detector picks the direction: change points -> boost LR,
    # noise-dominated -> reduce LR (Gershman 2020)

    "surnor_adaptive": SurNoRConfig(
        experiment_name="surnor_adaptive",
        lr_mode="adaptive",
    ),

    "surnor_adaptive_sensitive": SurNoRConfig(
        experiment_name="surnor_adaptive_sensitive",
        lr_mode="adaptive",
        vol_change_threshold=1.5,  # Easier to trigger change points
        vol_volatility_boost=1.0,  # Stronger LR boost on volatility
    ),

    # ===== HYBRID: LR + INTRINSIC REWARD =====
    # Combine both mechanisms

    "surnor_hybrid": SurNoRConfig(
        experiment_name="surnor_hybrid",
        lr_mode="pearce_hall",
        intrinsic_reward_scale=0.01,  # Small intrinsic bonus
    ),

    "surnor_intrinsic_only": SurNoRConfig(
        experiment_name="surnor_intrinsic_only",
        lr_mode="none",
        intrinsic_reward_scale=0.01,
    ),

    # ===== MULTIPLIER RANGE EXPERIMENTS =====

    "surnor_narrow_range": SurNoRConfig(
        experiment_name="surnor_narrow_range",
        lr_mode="pearce_hall",
        lr_min_multiplier=0.8,
        lr_max_multiplier=1.2,  # Only ±20%
    ),

    "surnor_wide_range": SurNoRConfig(
        experiment_name="surnor_wide_range",
        lr_mode="pearce_hall",
        lr_min_multiplier=0.25,
        lr_max_multiplier=4.0,  # ±4x
    ),
}


# Per-environment defaults applied on top of every experiment config.
# All envs must have discrete action spaces (forward model one-hot encodes).
ENV_DEFAULTS = {
    # Main benchmark
    "LunarLander-v3": {"total_timesteps": 200_000, "ent_coef": 0.01},
    # Solves fast with a reward ceiling of 500 - sanity/smoke env more than
    # a discriminative benchmark
    "CartPole-v1": {"total_timesteps": 100_000, "ent_coef": 0.0},
    # PPO reliably reaches ~-80 to -100; good discriminative env
    "Acrobot-v1": {"total_timesteps": 150_000, "ent_coef": 0.01},
    # EXPLORATORY: sparse reward, PPO often floors at -200 with default
    # hyperparameters. Interesting for the surprise hypothesis but don't
    # let it drive conclusions.
    "MountainCar-v0": {"total_timesteps": 300_000, "ent_coef": 0.01},
}


def build_experiments(env_id: str = "LunarLander-v3") -> dict:
    """Build the experiment registry for a given environment.

    Applies ENV_DEFAULTS for the env on top of every experiment config.
    """
    if env_id not in ENV_DEFAULTS:
        raise ValueError(
            f"Unknown environment: {env_id}. Available: {list(ENV_DEFAULTS)}. "
            f"Add an entry to ENV_DEFAULTS to use a new (discrete-action) env."
        )
    overrides = {"env_name": env_id, **ENV_DEFAULTS[env_id]}
    return {
        name: replace(config, **overrides)
        for name, config in SURNOR_EXPERIMENTS.items()
    }


def get_experiment_names() -> List[str]:
    """Get list of all experiment names."""
    return list(SURNOR_EXPERIMENTS.keys())


def get_experiment(name: str, env_id: str = "LunarLander-v3") -> SurNoRConfig:
    """Get experiment config by name for a given environment."""
    experiments = build_experiments(env_id)
    if name not in experiments:
        raise ValueError(f"Unknown experiment: {name}. Available: {get_experiment_names()}")
    return experiments[name]
