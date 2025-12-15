"""Experiment configurations for ablation studies."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EntropyClockConfig:
    """Configuration for the entropy clock module."""
    window_size: int = 50
    min_samples: int = 10
    baseline_momentum: float = 0.99
    salience_scale: float = 1.0


@dataclass
class PPOConfig:
    """Configuration for PPO training."""
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01


@dataclass
class ModulatorConfig:
    """Configuration for salience modulators."""
    # LR modulator
    lr_gamma: float = 0.5
    lr_min_multiplier: float = 0.5
    lr_max_multiplier: float = 2.0

    # Exploration modulator
    exploration_lambda: float = 0.3
    exploration_min_temp: float = 0.1
    exploration_max_temp: float = 3.0

    # Replay modulator
    replay_alpha: float = 0.5  # TD vs salience balance
    replay_beta: float = 0.6


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    # Environment
    env_name: str = "LunarLander-v3"
    total_timesteps: int = 500000
    n_seeds: int = 5
    seeds: list[int] = field(default_factory=lambda: [42, 123, 456, 789, 1011])

    # Components
    ppo: PPOConfig = field(default_factory=PPOConfig)
    entropy_clock: EntropyClockConfig = field(default_factory=EntropyClockConfig)
    modulators: ModulatorConfig = field(default_factory=ModulatorConfig)

    # Ablation toggles
    use_salience_replay: bool = False
    use_salience_lr: bool = False
    use_salience_exploration: bool = False

    # Logging
    log_dir: str = "results"
    experiment_name: str = "baseline"
    verbose: int = 1


# Predefined experiment configurations for ablation study
EXPERIMENTS = {
    "baseline": ExperimentConfig(
        experiment_name="baseline",
        use_salience_replay=False,
        use_salience_lr=False,
        use_salience_exploration=False,
    ),
    "salience_replay": ExperimentConfig(
        experiment_name="salience_replay",
        use_salience_replay=True,
        use_salience_lr=False,
        use_salience_exploration=False,
    ),
    "salience_lr": ExperimentConfig(
        experiment_name="salience_lr",
        use_salience_replay=False,
        use_salience_lr=True,
        use_salience_exploration=False,
    ),
    "salience_exploration": ExperimentConfig(
        experiment_name="salience_exploration",
        use_salience_replay=False,
        use_salience_lr=False,
        use_salience_exploration=True,
    ),
    "all_modulators": ExperimentConfig(
        experiment_name="all_modulators",
        use_salience_replay=True,
        use_salience_lr=True,
        use_salience_exploration=True,
    ),
}
