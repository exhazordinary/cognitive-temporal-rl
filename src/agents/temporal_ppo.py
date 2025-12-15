"""PPO agent with entropy-clock integration and salience modulators."""

import torch
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from typing import Optional, Dict, Any

from ..entropy_clock import EntropyClockModule
from ..modulators import SalienceReplay, SalienceLR, SalienceExploration


class TemporalCallback(BaseCallback):
    """Callback that integrates entropy clock with PPO training."""

    def __init__(
        self,
        entropy_clock: EntropyClockModule,
        salience_lr: Optional[SalienceLR] = None,
        salience_exploration: Optional[SalienceExploration] = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.entropy_clock = entropy_clock
        self.salience_lr = salience_lr
        self.salience_exploration = salience_exploration

        # Logging
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.episode_saliences: list[float] = []
        self.episode_internal_times: list[float] = []

        # Current episode tracking
        self._current_saliences: list[float] = []

    def _on_step(self) -> bool:
        # Get current observation
        obs = self.locals.get("new_obs")
        if obs is not None:
            # Process through entropy clock
            state_tensor = torch.from_numpy(obs[0]).float()
            clock_output = self.entropy_clock.step(state_tensor)

            # Modulate learning rate if enabled
            if self.salience_lr is not None:
                self.salience_lr.update_optimizer_lr(
                    self.model.policy.optimizer,
                    clock_output.salience_score,
                )

            self._current_saliences.append(clock_output.salience_score)

        # Check for episode completion
        for idx, done in enumerate(self.locals.get("dones", [])):
            if done:
                infos = self.locals.get("infos", [])
                if idx < len(infos) and "episode" in infos[idx]:
                    ep_info = infos[idx]["episode"]
                    self.episode_rewards.append(ep_info["r"])
                    self.episode_lengths.append(ep_info["l"])

                    # Log salience statistics for episode
                    if self._current_saliences:
                        self.episode_saliences.append(np.mean(self._current_saliences))
                    self.episode_internal_times.append(self.entropy_clock.internal_time)

                    if self.verbose > 0:
                        print(f"Episode {len(self.episode_rewards)}: "
                              f"reward={ep_info['r']:.2f}, "
                              f"mean_salience={self.episode_saliences[-1]:.3f}, "
                              f"internal_time={self.entropy_clock.internal_time:.1f}")

                # Reset for next episode
                self.entropy_clock.reset()
                self._current_saliences.clear()

        return True

    def _on_training_end(self) -> None:
        if self.verbose > 0:
            print(f"\nTraining complete!")
            print(f"Episodes: {len(self.episode_rewards)}")
            if self.episode_rewards:
                print(f"Mean reward (last 100): {np.mean(self.episode_rewards[-100:]):.2f}")
                print(f"Mean salience: {np.mean(self.episode_saliences):.3f}")


class TemporalPPO:
    """PPO wrapper with entropy-clock temporal cognition.

    Integrates the entropy clock module with PPO training,
    optionally enabling salience-based modulation of:
    - Learning rate
    - Exploration (temperature)
    - Experience replay (for off-policy variants)
    """

    def __init__(
        self,
        env_name: str = "LunarLander-v3",
        # PPO params
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        # Entropy clock params
        window_size: int = 50,
        min_samples: int = 10,
        # Modulator toggles
        use_salience_lr: bool = False,
        use_salience_exploration: bool = False,
        # Other
        seed: Optional[int] = None,
        device: str = "auto",
        verbose: int = 0,
    ):
        """Initialize TemporalPPO.

        Args:
            env_name: Gymnasium environment name
            learning_rate: Base learning rate
            n_steps: Steps per update
            batch_size: Minibatch size
            n_epochs: Epochs per update
            gamma: Discount factor
            window_size: Entropy clock window size
            min_samples: Minimum samples for entropy computation
            use_salience_lr: Enable LR modulation
            use_salience_exploration: Enable exploration modulation
            seed: Random seed
            device: Device to use
            verbose: Verbosity level
        """
        self.env_name = env_name
        self.seed = seed
        self.verbose = verbose

        # Create environment
        self.env = gym.make(env_name)
        if seed is not None:
            self.env.reset(seed=seed)

        # Create entropy clock
        self.entropy_clock = EntropyClockModule(
            window_size=window_size,
            min_samples=min_samples,
        )

        # Create modulators
        self.salience_lr = SalienceLR(base_lr=learning_rate) if use_salience_lr else None
        self.salience_exploration = SalienceExploration() if use_salience_exploration else None

        # Create PPO model
        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            seed=seed,
            device=device,
            verbose=0,
        )

        # Create callback
        self.callback = TemporalCallback(
            entropy_clock=self.entropy_clock,
            salience_lr=self.salience_lr,
            salience_exploration=self.salience_exploration,
            verbose=verbose,
        )

    def train(self, total_timesteps: int = 500000) -> Dict[str, Any]:
        """Train the agent.

        Args:
            total_timesteps: Total training steps

        Returns:
            Dictionary with training results
        """
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=self.callback,
            progress_bar=True,
        )

        return {
            "episode_rewards": self.callback.episode_rewards,
            "episode_lengths": self.callback.episode_lengths,
            "episode_saliences": self.callback.episode_saliences,
            "episode_internal_times": self.callback.episode_internal_times,
            "entropy_clock_stats": self.entropy_clock.get_stats(),
            "salience_lr_stats": self.salience_lr.get_stats() if self.salience_lr else None,
            "salience_exploration_stats": (
                self.salience_exploration.get_stats() if self.salience_exploration else None
            ),
        }

    def save(self, path: str) -> None:
        """Save the model."""
        self.model.save(path)

    def load(self, path: str) -> None:
        """Load a saved model."""
        self.model = PPO.load(path, env=self.env)

    def close(self) -> None:
        """Clean up resources."""
        self.env.close()


if __name__ == "__main__":
    import os
    from pathlib import Path

    # Create models directory
    models_dir = Path(__file__).parent.parent.parent / "models"
    models_dir.mkdir(exist_ok=True)

    # Train agent
    agent = TemporalPPO(
        use_salience_lr=True,
        seed=42,
        verbose=1,
    )
    results = agent.train(total_timesteps=50000)
    print(f"\nFinal mean reward: {np.mean(results['episode_rewards'][-50:]):.2f}")

    # Save trained model
    model_path = models_dir / "temporal_ppo_lunarlander"
    agent.save(str(model_path))
    print(f"\nModel saved to: {model_path}")

    agent.close()

    print("\nTo watch the trained agent:")
    print(f"  uv run python scripts/watch.py --model {model_path}")
