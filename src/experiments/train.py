"""Training script for running experiments."""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
import numpy as np

from .config import EXPERIMENTS, ExperimentConfig
from ..agents.temporal_ppo import TemporalPPO
from ..agents.base_ppo import train_baseline


def run_experiment(config: ExperimentConfig, seed: int) -> dict:
    """Run a single experiment with given config and seed.

    Args:
        config: Experiment configuration
        seed: Random seed

    Returns:
        Dictionary with results
    """
    print(f"\n{'='*60}")
    print(f"Running: {config.experiment_name} (seed={seed})")
    print(f"{'='*60}")

    # Check if this is baseline (no modulators)
    is_baseline = not any([
        config.use_salience_replay,
        config.use_salience_lr,
        config.use_salience_exploration,
    ])

    if is_baseline:
        # Use vanilla PPO
        model, rewards = train_baseline(
            total_timesteps=config.total_timesteps,
            seed=seed,
            verbose=config.verbose,
        )
        results = {
            "episode_rewards": rewards,
            "episode_lengths": [],
            "config": "baseline",
        }
    else:
        # Use TemporalPPO
        agent = TemporalPPO(
            env_name=config.env_name,
            learning_rate=config.ppo.learning_rate,
            n_steps=config.ppo.n_steps,
            batch_size=config.ppo.batch_size,
            n_epochs=config.ppo.n_epochs,
            gamma=config.ppo.gamma,
            window_size=config.entropy_clock.window_size,
            min_samples=config.entropy_clock.min_samples,
            use_salience_lr=config.use_salience_lr,
            use_salience_exploration=config.use_salience_exploration,
            seed=seed,
            verbose=config.verbose,
        )
        results = agent.train(total_timesteps=config.total_timesteps)
        agent.close()

    results["seed"] = seed
    results["experiment_name"] = config.experiment_name

    return results


def run_ablation_study(
    experiment_names: list[str] = None,
    total_timesteps: int = None,
    n_seeds: int = None,
    output_dir: str = "results",
) -> dict:
    """Run full ablation study across experiments and seeds.

    Args:
        experiment_names: List of experiment names to run (default: all)
        total_timesteps: Override timesteps (optional)
        n_seeds: Override number of seeds (optional)
        output_dir: Directory to save results

    Returns:
        Dictionary with all results
    """
    if experiment_names is None:
        experiment_names = list(EXPERIMENTS.keys())

    all_results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for exp_name in experiment_names:
        config = EXPERIMENTS[exp_name]

        # Apply overrides
        if total_timesteps is not None:
            config.total_timesteps = total_timesteps
        if n_seeds is not None:
            config.n_seeds = n_seeds
            config.seeds = config.seeds[:n_seeds]

        exp_results = []
        for seed in config.seeds[:config.n_seeds]:
            result = run_experiment(config, seed)
            exp_results.append(result)

        all_results[exp_name] = {
            "runs": exp_results,
            "summary": compute_summary(exp_results),
        }

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results_file = output_path / f"ablation_{timestamp}.json"

    # Convert numpy arrays to lists for JSON serialization
    serializable_results = _make_serializable(all_results)
    with open(results_file, "w") as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    return all_results


def compute_summary(results: list[dict]) -> dict:
    """Compute summary statistics across seeds.

    Args:
        results: List of result dictionaries

    Returns:
        Summary statistics
    """
    all_rewards = [r["episode_rewards"] for r in results]

    # Get final performance (last 100 episodes)
    final_rewards = [np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
                     for rewards in all_rewards]

    return {
        "mean_final_reward": float(np.mean(final_rewards)),
        "std_final_reward": float(np.std(final_rewards)),
        "n_runs": len(results),
    }


def _make_serializable(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    else:
        return obj


def main():
    parser = argparse.ArgumentParser(description="Run cognitive temporal RL experiments")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        choices=list(EXPERIMENTS.keys()),
        help="Experiments to run (default: all)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Total timesteps per run",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=None,
        help="Number of seeds to run",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory",
    )

    args = parser.parse_args()

    run_ablation_study(
        experiment_names=args.experiments,
        total_timesteps=args.timesteps,
        n_seeds=args.seeds,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
