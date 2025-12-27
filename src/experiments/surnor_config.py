"""Experiment configurations for SurNoR-based experiments.

New experiments based on research synthesis:
1. Prediction error (surprise) instead of state entropy
2. Pearce-Hall smoothing
3. LR modulation at update time
4. Both directions tested (high surprise -> high/low LR)
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class SurNoRConfig:
    """Configuration for SurNoR experiments."""
    # Environment
    env_name: str = "LunarLander-v3"
    total_timesteps: int = 200000
    # Use None for random seeds to avoid deterministic trajectory issue
    seeds: List[Optional[int]] = field(default_factory=lambda: [None] * 10)

    # PPO params
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    ent_coef: float = 0.01

    # Surprise module params
    pearce_hall_gamma: float = 0.3  # How quickly alpha adapts
    forward_model_lr: float = 1e-3
    forward_hidden_dim: int = 64

    # LR modulation
    use_lr_modulation: bool = True
    lr_min_multiplier: float = 0.5
    lr_max_multiplier: float = 2.0
    invert_lr: bool = False  # False = Pearce-Hall, True = Stabilization

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
        use_lr_modulation=False,
        intrinsic_reward_scale=0.0,
    ),

    # ===== PEARCE-HALL DIRECTION =====
    # High surprise -> Higher LR (learn more from surprising events)

    "surnor_pearce_hall": SurNoRConfig(
        experiment_name="surnor_pearce_hall",
        use_lr_modulation=True,
        invert_lr=False,
        pearce_hall_gamma=0.3,
    ),

    "surnor_ph_gamma_0.1": SurNoRConfig(
        experiment_name="surnor_ph_gamma_0.1",
        use_lr_modulation=True,
        invert_lr=False,
        pearce_hall_gamma=0.1,  # Slower adaptation
    ),

    "surnor_ph_gamma_0.5": SurNoRConfig(
        experiment_name="surnor_ph_gamma_0.5",
        use_lr_modulation=True,
        invert_lr=False,
        pearce_hall_gamma=0.5,  # Faster adaptation
    ),

    # ===== STABILIZATION DIRECTION =====
    # High surprise -> Lower LR (stabilize during chaos)

    "surnor_stabilize": SurNoRConfig(
        experiment_name="surnor_stabilize",
        use_lr_modulation=True,
        invert_lr=True,
        pearce_hall_gamma=0.3,
    ),

    "surnor_stab_gamma_0.1": SurNoRConfig(
        experiment_name="surnor_stab_gamma_0.1",
        use_lr_modulation=True,
        invert_lr=True,
        pearce_hall_gamma=0.1,
    ),

    # ===== HYBRID: LR + INTRINSIC REWARD =====
    # Combine both mechanisms

    "surnor_hybrid": SurNoRConfig(
        experiment_name="surnor_hybrid",
        use_lr_modulation=True,
        invert_lr=False,
        intrinsic_reward_scale=0.01,  # Small intrinsic bonus
    ),

    "surnor_intrinsic_only": SurNoRConfig(
        experiment_name="surnor_intrinsic_only",
        use_lr_modulation=False,
        intrinsic_reward_scale=0.01,
    ),

    # ===== MULTIPLIER RANGE EXPERIMENTS =====

    "surnor_narrow_range": SurNoRConfig(
        experiment_name="surnor_narrow_range",
        use_lr_modulation=True,
        invert_lr=False,
        lr_min_multiplier=0.8,
        lr_max_multiplier=1.2,  # Only ±20%
    ),

    "surnor_wide_range": SurNoRConfig(
        experiment_name="surnor_wide_range",
        use_lr_modulation=True,
        invert_lr=False,
        lr_min_multiplier=0.25,
        lr_max_multiplier=4.0,  # ±4x
    ),
}


def get_experiment_names() -> List[str]:
    """Get list of all experiment names."""
    return list(SURNOR_EXPERIMENTS.keys())


def get_experiment(name: str) -> SurNoRConfig:
    """Get experiment config by name."""
    if name not in SURNOR_EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {name}. Available: {get_experiment_names()}")
    return SURNOR_EXPERIMENTS[name]
