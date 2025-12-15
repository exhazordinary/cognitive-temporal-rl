"""Vanilla PPO baseline using stable-baselines3."""

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
import numpy as np
from typing import Optional


class EpisodeLoggerCallback(BaseCallback):
    """Callback for logging episode statistics."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []

    def _on_step(self) -> bool:
        # Check for episode completion
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
                if self.verbose > 0:
                    print(f"Episode {len(self.episode_rewards)}: "
                          f"reward={info['episode']['r']:.2f}, "
                          f"length={info['episode']['l']}")
        return True


def create_baseline_ppo(
    env_name: str = "LunarLander-v3",
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    seed: Optional[int] = None,
    device: str = "auto",
) -> tuple[PPO, gym.Env]:
    """Create a vanilla PPO agent for LunarLander.

    Args:
        env_name: Gymnasium environment name
        learning_rate: Learning rate
        n_steps: Steps per update
        batch_size: Minibatch size
        n_epochs: Epochs per update
        gamma: Discount factor
        gae_lambda: GAE lambda
        clip_range: PPO clip range
        ent_coef: Entropy coefficient
        seed: Random seed
        device: Device to use

    Returns:
        Tuple of (PPO model, environment)
    """
    env = gym.make(env_name)
    if seed is not None:
        env.reset(seed=seed)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        seed=seed,
        device=device,
        verbose=0,
    )

    return model, env


def train_baseline(
    total_timesteps: int = 500000,
    seed: Optional[int] = None,
    verbose: int = 1,
) -> tuple[PPO, list[float]]:
    """Train a baseline PPO agent.

    Args:
        total_timesteps: Total training steps
        seed: Random seed
        verbose: Verbosity level

    Returns:
        Tuple of (trained model, episode rewards)
    """
    model, env = create_baseline_ppo(seed=seed)
    callback = EpisodeLoggerCallback(verbose=verbose)

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True,
    )

    env.close()
    return model, callback.episode_rewards


if __name__ == "__main__":
    model, rewards = train_baseline(total_timesteps=100000, seed=42)
    print(f"\nTraining complete!")
    print(f"Episodes: {len(rewards)}")
    print(f"Mean reward (last 100): {np.mean(rewards[-100:]):.2f}")
