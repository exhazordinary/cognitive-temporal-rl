"""SurNoR-inspired PPO: Surprise-modulated learning rate with proper timing.

Based on:
- SurNoR (2021): Separates novelty (exploration) from surprise (learning rate)
- Pearce-Hall (1980): Associability modulated by prediction error
- RND (2018): Prediction error normalization techniques

Key fixes from original temporal_ppo.py:
1. LR modulation happens at PPO UPDATE time, not during rollout
2. Uses prediction error (forward model) instead of state entropy
3. Pearce-Hall smoothing prevents per-step jitter
4. Optional intrinsic reward bonus for exploration
"""

import torch
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from typing import Optional, Dict, Any, List

from ..surprise import SurpriseModule
from ..modulators.pearce_hall_lr import PearceHallLR


class SurNoRCallback(BaseCallback):
    """Callback integrating surprise-based LR modulation with PPO.

    Critical difference from TemporalCallback:
    - Collects surprise during rollout
    - Applies LR modulation ONCE before PPO update (not per-step)
    """

    def __init__(
        self,
        surprise_module: SurpriseModule,
        lr_modulator: Optional[PearceHallLR] = None,
        intrinsic_reward_scale: float = 0.0,  # 0 = disabled
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.surprise_module = surprise_module
        self.lr_modulator = lr_modulator
        self.intrinsic_reward_scale = intrinsic_reward_scale

        # Episode tracking
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_surprises: List[float] = []

        # Per-update tracking
        self.update_alphas: List[float] = []
        self.update_lrs: List[float] = []

        # Step tracking
        self._prev_obs: Optional[np.ndarray] = None
        self._prev_action: Optional[int] = None

    def _on_training_start(self) -> None:
        """Called before training starts."""
        self._prev_obs = None
        self._prev_action = None

    def _on_step(self) -> bool:
        """Called after each environment step during rollout.

        We collect surprise here but DO NOT modulate LR yet.
        """
        # Get transition data
        new_obs = self.locals.get("new_obs")
        actions = self.locals.get("actions")
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")

        if new_obs is not None and self._prev_obs is not None:
            # Compute surprise for this transition
            state = torch.from_numpy(self._prev_obs[0]).float()
            action = torch.tensor(self._prev_action)
            next_state = torch.from_numpy(new_obs[0]).float()

            output = self.surprise_module.step(state, action, next_state)

            # Optional: add intrinsic reward bonus
            if self.intrinsic_reward_scale > 0:
                intrinsic = output.surprise * self.intrinsic_reward_scale
                # Note: modifying rewards in-place for SB3
                # This affects the rewards used in GAE computation
                self.locals["rewards"][0] += intrinsic

        # Store for next transition
        if new_obs is not None:
            self._prev_obs = new_obs.copy()
        if actions is not None:
            self._prev_action = actions[0]

        # Handle episode completion
        if dones is not None:
            for idx, done in enumerate(dones):
                if done:
                    infos = self.locals.get("infos", [])
                    if idx < len(infos) and "episode" in infos[idx]:
                        ep_info = infos[idx]["episode"]
                        self.episode_rewards.append(ep_info["r"])
                        self.episode_lengths.append(ep_info["l"])

                        # Mean surprise for this episode
                        stats = self.surprise_module.get_rollout_stats()
                        self.episode_surprises.append(stats["mean_surprise"])

                    self.surprise_module.reset_episode()
                    self._prev_obs = None
                    self._prev_action = None

        return True

    def _on_rollout_end(self) -> None:
        """Called after rollout collection, BEFORE PPO update.

        THIS is the right time to modulate learning rate!
        """
        if self.lr_modulator is None:
            return

        # Get rollout statistics
        stats = self.surprise_module.get_rollout_stats()
        alpha = stats["alpha"]

        # Apply LR modulation to optimizer
        new_lr = self.lr_modulator.apply_to_optimizer(
            self.model.policy.optimizer,
            alpha,
        )

        # Track
        self.update_alphas.append(alpha)
        self.update_lrs.append(new_lr)

        if self.verbose > 0:
            print(f"  [SurNoR] Rollout alpha={alpha:.3f}, LR={new_lr:.6f}")

        # Clear rollout accumulator
        self.surprise_module.clear_rollout()

    def _on_training_end(self) -> None:
        if self.verbose > 0:
            print(f"\nSurNoR Training Complete!")
            print(f"  Episodes: {len(self.episode_rewards)}")
            if self.episode_rewards:
                print(f"  Mean reward (last 100): {np.mean(self.episode_rewards[-100:]):.2f}")
            if self.update_alphas:
                print(f"  Mean alpha: {np.mean(self.update_alphas):.3f}")
                print(f"  Alpha range: [{min(self.update_alphas):.3f}, {max(self.update_alphas):.3f}]")


class SurNoRPPO:
    """PPO with SurNoR-inspired surprise-based learning rate modulation.

    Key features:
    1. Forward model computes prediction error (surprise)
    2. Pearce-Hall smoothing creates associability signal
    3. LR modulated at update time (not per-step)
    4. Optional intrinsic reward bonus
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
        ent_coef: float = 0.01,
        # Surprise module params
        pearce_hall_gamma: float = 0.3,
        forward_model_lr: float = 1e-3,
        forward_hidden_dim: int = 64,
        # LR modulation params
        use_lr_modulation: bool = True,
        lr_min_multiplier: float = 0.5,
        lr_max_multiplier: float = 2.0,
        invert_lr: bool = False,
        # Intrinsic reward (optional, set > 0 to enable)
        intrinsic_reward_scale: float = 0.0,
        # Other
        seed: Optional[int] = None,
        device: str = "auto",
        verbose: int = 0,
    ):
        self.env_name = env_name
        self.seed = seed
        self.verbose = verbose

        # Create environment
        self.env = gym.make(env_name)
        if seed is not None:
            self.env.reset(seed=seed)

        # Get dimensions
        state_dim = self.env.observation_space.shape[0]
        if hasattr(self.env.action_space, 'n'):
            action_dim = self.env.action_space.n
        else:
            action_dim = self.env.action_space.shape[0]

        # Create surprise module
        self.surprise_module = SurpriseModule(
            state_dim=state_dim,
            action_dim=action_dim,
            gamma=pearce_hall_gamma,
            hidden_dim=forward_hidden_dim,
            model_lr=forward_model_lr,
        )

        # Create LR modulator
        self.lr_modulator = PearceHallLR(
            base_lr=learning_rate,
            min_multiplier=lr_min_multiplier,
            max_multiplier=lr_max_multiplier,
            invert=invert_lr,
        ) if use_lr_modulation else None

        # Create PPO model
        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            ent_coef=ent_coef,
            seed=seed,
            device=device,
            verbose=0,
        )

        # Create callback
        self.callback = SurNoRCallback(
            surprise_module=self.surprise_module,
            lr_modulator=self.lr_modulator,
            intrinsic_reward_scale=intrinsic_reward_scale,
            verbose=verbose,
        )

    def train(self, total_timesteps: int = 500000) -> Dict[str, Any]:
        """Train the agent."""
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=self.callback,
            progress_bar=True,
        )

        return {
            "episode_rewards": self.callback.episode_rewards,
            "episode_lengths": self.callback.episode_lengths,
            "episode_surprises": self.callback.episode_surprises,
            "update_alphas": self.callback.update_alphas,
            "update_lrs": self.callback.update_lrs,
            "surprise_stats": self.surprise_module.get_stats(),
            "lr_stats": self.lr_modulator.get_stats() if self.lr_modulator else None,
        }

    def save(self, path: str) -> None:
        self.model.save(path)

    def load(self, path: str) -> None:
        self.model = PPO.load(path, env=self.env)

    def close(self) -> None:
        self.env.close()


if __name__ == "__main__":
    # Quick test
    print("Testing SurNoR PPO...")

    agent = SurNoRPPO(
        use_lr_modulation=True,
        invert_lr=False,  # Pearce-Hall: high surprise -> high LR
        pearce_hall_gamma=0.3,
        seed=42,
        verbose=1,
    )

    results = agent.train(total_timesteps=50000)

    print(f"\nFinal mean reward: {np.mean(results['episode_rewards'][-50:]):.2f}")
    print(f"Mean alpha: {np.mean(results['update_alphas']):.3f}")
    print(f"LR range: [{min(results['update_lrs']):.6f}, {max(results['update_lrs']):.6f}]")

    agent.close()
