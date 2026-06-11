"""SurNoR-inspired PPO: Surprise-modulated learning rate with proper timing.

Based on:
- SurNoR (2021): Separates novelty (exploration) from surprise (learning rate)
- Pearce-Hall (1980): Associability modulated by prediction error
- RND (2018): Prediction error normalization techniques

Key fixes from the original approach (see legacy/temporal_ppo.py):
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
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from typing import Optional, Dict, Any, List

from ..surprise import SurpriseModule, VolatilityDetector
from ..modulators.pearce_hall_lr import PearceHallLR

LR_MODES = ("none", "pearce_hall", "stabilize", "adaptive")


class MutableLRSchedule:
    """Picklable LR schedule whose value the callback mutates per update.

    SB3's ``PPO.train()`` calls ``_update_learning_rate()`` as its first
    action, overwriting whatever the callback wrote into the optimizer with
    the schedule's value. Writing the modulated LR into this schedule (and
    letting SB3 apply it) is therefore the only path that actually reaches
    the gradient steps.
    """

    def __init__(self, base_lr: float):
        self.current_lr = base_lr

    def __call__(self, progress_remaining: float) -> float:
        return self.current_lr


class SurNoRCallback(BaseCallback):
    """Callback integrating surprise-based LR modulation with PPO.

    Critical difference from TemporalCallback:
    - Collects surprise during rollout
    - Applies LR modulation ONCE before PPO update (not per-step)
    """

    def __init__(
        self,
        surprise_module: SurpriseModule,
        lr_mode: str = "pearce_hall",
        lr_modulator: Optional[PearceHallLR] = None,
        volatility_detector: Optional[VolatilityDetector] = None,
        lr_schedule: Optional[MutableLRSchedule] = None,
        base_lr: float = 3e-4,
        lr_min_multiplier: float = 0.5,
        lr_max_multiplier: float = 2.0,
        intrinsic_reward_scale: float = 0.0,  # 0 = disabled
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.surprise_module = surprise_module
        self.lr_mode = lr_mode
        self.lr_modulator = lr_modulator
        self.volatility_detector = volatility_detector
        self.lr_schedule = lr_schedule
        self.base_lr = base_lr
        self.lr_min_multiplier = lr_min_multiplier
        self.lr_max_multiplier = lr_max_multiplier
        self.intrinsic_reward_scale = intrinsic_reward_scale

        # Episode tracking
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_surprises: List[float] = []

        # Per-update tracking
        self.update_alphas: List[float] = []
        self.update_lrs: List[float] = []
        self.update_volatility: List[float] = []
        self.update_noise: List[float] = []
        self.update_change_points: List[int] = []

    def _on_step(self) -> bool:
        """Called after each vectorized environment step during rollout.

        We collect surprise here but DO NOT modulate LR yet.

        At this point in SB3's collect_rollouts, ``self.model._last_obs``
        still holds the PRE-step observations (it is reassigned to new_obs
        only after the callback returns), so the correctly paired batched
        transition (s_t, a_t, s_{t+1}) is available without tracking state.
        """
        prev_obs = self.model._last_obs            # (n_envs, obs_dim)
        actions = self.locals["actions"]           # (n_envs,)
        new_obs = self.locals["new_obs"]
        dones = self.locals["dones"]
        infos = self.locals["infos"]

        # On done steps, SB3 VecEnvs auto-reset: new_obs is the NEXT
        # episode's reset obs. The true terminal obs lives in infos.
        next_states = np.array(new_obs, copy=True)
        for idx, done in enumerate(dones):
            if done:
                terminal_obs = infos[idx].get("terminal_observation")
                if terminal_obs is not None:
                    next_states[idx] = terminal_obs

        output = self.surprise_module.step_batch(prev_obs, actions, next_states)

        # Feed the volatility detector one signed PE per vec-step
        if self.volatility_detector is not None:
            self.volatility_detector.step(float(np.mean(output.surprises)))

        # Optional: add intrinsic reward bonus (in-place so the modified
        # rewards reach the rollout buffer / GAE computation)
        if self.intrinsic_reward_scale > 0:
            self.locals["rewards"] += self.intrinsic_reward_scale * output.surprises

        # Handle episode completion
        for idx, done in enumerate(dones):
            if done:
                if "episode" in infos[idx]:
                    ep_info = infos[idx]["episode"]
                    self.episode_rewards.append(ep_info["r"])
                    self.episode_lengths.append(ep_info["l"])

                    # Mean surprise for this episode
                    stats = self.surprise_module.get_rollout_stats()
                    self.episode_surprises.append(stats["mean_surprise"])

                self.surprise_module.reset_episode()

        return True

    def _on_rollout_end(self) -> None:
        """Called after rollout collection, BEFORE PPO update.

        THIS is the right time to modulate learning rate!
        """
        # Get rollout statistics
        stats = self.surprise_module.get_rollout_stats()
        alpha = stats["alpha"]
        self.update_alphas.append(alpha)

        if self.lr_mode == "adaptive":
            # Volatility detected -> boost LR; noise-dominated -> reduce LR
            multiplier = self.volatility_detector.get_rollout_recommendation()
            multiplier = float(np.clip(multiplier, self.lr_min_multiplier, self.lr_max_multiplier))
            new_lr = self.base_lr * multiplier

            det_stats = self.volatility_detector.get_stats()
            self.update_volatility.append(float(det_stats["mean_volatility"]))
            self.update_noise.append(float(det_stats["mean_noise"]))
            self.update_change_points.append(int(det_stats["n_change_points"]))
        elif self.lr_mode in ("pearce_hall", "stabilize"):
            new_lr = self.lr_modulator.compute_lr(alpha)
        else:
            # "none": leave the schedule at its base value
            self.surprise_module.clear_rollout()
            return

        # Write the modulated LR into the schedule; SB3's
        # _update_learning_rate() applies it at the start of train().
        if self.lr_schedule is not None:
            self.lr_schedule.current_lr = new_lr
        self.update_lrs.append(new_lr)

        if self.verbose > 0:
            print(f"  [SurNoR] Rollout alpha={alpha:.3f}, LR={new_lr:.6f} ({self.lr_mode})")

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
        # Vectorization
        n_envs: int = 8,
        vec_env_type: str = "dummy",  # "dummy" or "subproc"
        # PPO params
        learning_rate: float = 3e-4,
        n_steps_total: int = 2048,  # rollout size in env-steps, split across envs
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        ent_coef: float = 0.01,
        # Surprise module params
        pearce_hall_gamma: float = 0.3,
        forward_model_lr: float = 1e-3,
        forward_hidden_dim: int = 64,
        # LR modulation params
        lr_mode: str = "pearce_hall",  # none | pearce_hall | stabilize | adaptive
        lr_min_multiplier: float = 0.5,
        lr_max_multiplier: float = 2.0,
        # Volatility detector params (adaptive mode only)
        vol_window_short: int = 20,
        vol_window_long: int = 100,
        vol_change_threshold: float = 2.0,
        vol_noise_sensitivity: float = 0.5,
        vol_volatility_boost: float = 0.5,
        vol_rollout_window: int = 50,
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
        self.n_envs = n_envs

        # SB3's n_steps is PER ENV; keep the rollout size (and therefore the
        # number of PPO updates and LR modulations) constant in env-steps.
        self.n_steps_per_env = max(64, n_steps_total // n_envs)
        if verbose > 0 and self.n_steps_per_env * n_envs != n_steps_total:
            print(f"  [SurNoR] n_steps_total={n_steps_total} adjusted to "
                  f"{self.n_steps_per_env * n_envs} ({self.n_steps_per_env} x {n_envs} envs)")

        # Create vectorized environment. DummyVecEnv is the default: for
        # cheap envs the speedup comes from batching policy forward passes,
        # and SubprocVecEnv IPC overhead can exceed the env step cost.
        vec_env_cls = SubprocVecEnv if vec_env_type == "subproc" else DummyVecEnv
        self.env = make_vec_env(env_name, n_envs=n_envs, seed=seed, vec_env_cls=vec_env_cls)

        # Get dimensions (VecEnv exposes the single-env spaces)
        if not isinstance(self.env.action_space, gym.spaces.Discrete):
            raise ValueError(
                f"SurNoRPPO requires a discrete action space (the forward model "
                f"one-hot encodes actions); {env_name} has {self.env.action_space}."
            )
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n

        # Create surprise module
        self.surprise_module = SurpriseModule(
            state_dim=state_dim,
            action_dim=action_dim,
            gamma=pearce_hall_gamma,
            hidden_dim=forward_hidden_dim,
            model_lr=forward_model_lr,
        )

        if lr_mode not in LR_MODES:
            raise ValueError(f"Unknown lr_mode: {lr_mode!r}. Valid modes: {LR_MODES}")
        self.lr_mode = lr_mode

        # Create LR modulator (Pearce-Hall directions only)
        self.lr_modulator = PearceHallLR(
            base_lr=learning_rate,
            min_multiplier=lr_min_multiplier,
            max_multiplier=lr_max_multiplier,
            invert=(lr_mode == "stabilize"),
        ) if lr_mode in ("pearce_hall", "stabilize") else None

        # Create volatility detector (adaptive mode only)
        self.volatility_detector = VolatilityDetector(
            window_short=vol_window_short,
            window_long=vol_window_long,
            change_threshold=vol_change_threshold,
            noise_sensitivity=vol_noise_sensitivity,
            volatility_boost=vol_volatility_boost,
            min_multiplier=lr_min_multiplier,
            max_multiplier=lr_max_multiplier,
            rollout_window=vol_rollout_window,
        ) if lr_mode == "adaptive" else None

        # Mutable schedule: the callback writes the modulated LR here and
        # SB3 applies it at the start of each train() call.
        self.lr_schedule = MutableLRSchedule(learning_rate)

        # Create PPO model
        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=self.lr_schedule,
            n_steps=self.n_steps_per_env,
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
            lr_mode=lr_mode,
            lr_modulator=self.lr_modulator,
            volatility_detector=self.volatility_detector,
            lr_schedule=self.lr_schedule,
            base_lr=learning_rate,
            lr_min_multiplier=lr_min_multiplier,
            lr_max_multiplier=lr_max_multiplier,
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
            "update_volatility": self.callback.update_volatility,
            "update_noise": self.callback.update_noise,
            "update_change_points": self.callback.update_change_points,
            "surprise_stats": self.surprise_module.get_stats(),
            "lr_stats": self.lr_modulator.get_stats() if self.lr_modulator else None,
            "volatility_stats": (
                self.volatility_detector.get_stats() if self.volatility_detector else None
            ),
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
        lr_mode="pearce_hall",  # high surprise -> high LR
        pearce_hall_gamma=0.3,
        seed=42,
        verbose=1,
    )

    results = agent.train(total_timesteps=50000)

    print(f"\nFinal mean reward: {np.mean(results['episode_rewards'][-50:]):.2f}")
    print(f"Mean alpha: {np.mean(results['update_alphas']):.3f}")
    print(f"LR range: [{min(results['update_lrs']):.6f}, {max(results['update_lrs']):.6f}]")

    agent.close()
