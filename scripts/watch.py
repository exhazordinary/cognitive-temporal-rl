"""Watch a trained or untrained agent play LunarLander."""

import argparse
import time
import gymnasium as gym
import numpy as np
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.entropy_clock import EntropyClockModule
from src.agents.base_ppo import create_baseline_ppo
import torch


def watch_agent(
    model_path: str = None,
    episodes: int = 5,
    show_salience: bool = True,
    delay: float = 0.02,
):
    """Watch agent play LunarLander with optional salience display.

    Args:
        model_path: Path to saved model (None = untrained random agent)
        episodes: Number of episodes to watch
        show_salience: Whether to print salience scores
        delay: Delay between frames (seconds)
    """
    # Create environment with rendering
    env = gym.make("LunarLander-v3", render_mode="human")

    # Load or create agent
    if model_path and Path(model_path).exists():
        from stable_baselines3 import PPO
        model = PPO.load(model_path)
        print(f"Loaded model from {model_path}")
    else:
        model, _ = create_baseline_ppo()
        print("Using untrained agent (random-ish policy)")

    # Create entropy clock for salience tracking
    entropy_clock = EntropyClockModule(window_size=50, min_samples=10)

    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        step = 0
        max_salience = 0

        print(f"\n{'='*50}")
        print(f"Episode {ep + 1}/{episodes}")
        print(f"{'='*50}")

        while True:
            # Get action from model
            action, _ = model.predict(obs, deterministic=False)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1

            # Track salience
            if show_salience:
                state_tensor = torch.from_numpy(obs).float()
                clock_out = entropy_clock.step(state_tensor)

                if clock_out.salience_score > max_salience:
                    max_salience = clock_out.salience_score

                # Print high salience moments
                if clock_out.salience_score > 1.5:
                    bar = "!" * min(int(clock_out.salience_score * 5), 30)
                    print(f"  Step {step:3d}: SALIENT {bar} ({clock_out.salience_score:.2f})")

            # Small delay for visibility
            time.sleep(delay)

            if terminated or truncated:
                break

        print(f"\nResult: reward={total_reward:.1f}, steps={step}, max_salience={max_salience:.2f}")
        entropy_clock.reset()

    env.close()
    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(description="Watch LunarLander agent")
    parser.add_argument("--model", type=str, default=None, help="Path to saved model")
    parser.add_argument("--episodes", type=int, default=3, help="Episodes to watch")
    parser.add_argument("--no-salience", action="store_true", help="Hide salience output")
    parser.add_argument("--delay", type=float, default=0.02, help="Frame delay (seconds)")

    args = parser.parse_args()

    watch_agent(
        model_path=args.model,
        episodes=args.episodes,
        show_salience=not args.no_salience,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
