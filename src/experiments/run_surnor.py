"""Training script for SurNoR experiments.

Usage:
    # Run single experiment with 10 seeds
    uv run python -m src.experiments.run_surnor --experiments surnor_pearce_hall

    # Run multiple experiments
    uv run python -m src.experiments.run_surnor --experiments baseline surnor_pearce_hall surnor_stabilize

    # Run all experiments
    uv run python -m src.experiments.run_surnor --all

    # Quick test (fewer timesteps/seeds)
    uv run python -m src.experiments.run_surnor --experiments baseline surnor_pearce_hall --timesteps 50000 --seeds 3
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
import numpy as np

from .surnor_config import SURNOR_EXPERIMENTS, SurNoRConfig, get_experiment_names
from ..agents.surnor_ppo import SurNoRPPO


def run_single_experiment(config: SurNoRConfig, seed: int) -> dict:
    """Run a single experiment with given config and seed."""
    print(f"\n{'='*60}")
    print(f"Running: {config.experiment_name} (seed={seed})")
    print(f"{'='*60}")

    agent = SurNoRPPO(
        env_name=config.env_name,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        ent_coef=config.ent_coef,
        pearce_hall_gamma=config.pearce_hall_gamma,
        forward_model_lr=config.forward_model_lr,
        forward_hidden_dim=config.forward_hidden_dim,
        use_lr_modulation=config.use_lr_modulation,
        lr_min_multiplier=config.lr_min_multiplier,
        lr_max_multiplier=config.lr_max_multiplier,
        invert_lr=config.invert_lr,
        intrinsic_reward_scale=config.intrinsic_reward_scale,
        seed=seed,
        verbose=config.verbose,
    )

    results = agent.train(total_timesteps=config.total_timesteps)
    agent.close()

    # Add metadata
    results["seed"] = seed
    results["experiment_name"] = config.experiment_name
    results["config"] = asdict(config)

    return results


def compute_summary(results_list: list[dict]) -> dict:
    """Compute summary statistics across seeds."""
    all_rewards = [r["episode_rewards"] for r in results_list]

    # Final performance (last 100 episodes)
    final_rewards = [
        np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
        for rewards in all_rewards
    ]

    # Alpha statistics (if LR modulation was used)
    all_alphas = [r.get("update_alphas", []) for r in results_list]
    mean_alphas = [np.mean(a) if a else 0.0 for a in all_alphas]

    return {
        "mean_final_reward": float(np.mean(final_rewards)),
        "std_final_reward": float(np.std(final_rewards)),
        "min_final_reward": float(np.min(final_rewards)),
        "max_final_reward": float(np.max(final_rewards)),
        "mean_alpha": float(np.mean(mean_alphas)) if any(mean_alphas) else None,
        "n_runs": len(results_list),
    }


def run_experiments(
    experiment_names: list[str],
    n_seeds: int = 10,
    total_timesteps: int = None,
    output_dir: str = "results",
) -> dict:
    """Run multiple experiments across seeds."""
    all_results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for exp_name in experiment_names:
        config = SURNOR_EXPERIMENTS[exp_name]

        # Apply overrides
        if total_timesteps is not None:
            config.total_timesteps = total_timesteps

        seeds = config.seeds[:n_seeds]

        exp_results = []
        for seed in seeds:
            result = run_single_experiment(config, seed)
            exp_results.append(result)

            # Print progress
            final_reward = np.mean(result["episode_rewards"][-100:]) if len(result["episode_rewards"]) >= 100 else np.mean(result["episode_rewards"])
            print(f"  Seed {seed}: Final reward = {final_reward:.2f}")

        summary = compute_summary(exp_results)
        print(f"\n{exp_name} Summary:")
        print(f"  Mean: {summary['mean_final_reward']:.2f} ± {summary['std_final_reward']:.2f}")

        all_results[exp_name] = {
            "runs": exp_results,
            "summary": summary,
        }

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results_file = output_path / f"surnor_{timestamp}.json"

    # Convert to JSON-serializable format
    serializable = _make_serializable(all_results)
    with open(results_file, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {results_file}")
    print(f"{'='*60}")

    # Print comparison table
    print("\nExperiment Comparison:")
    print("-" * 50)
    print(f"{'Experiment':<30} {'Mean ± Std':>18}")
    print("-" * 50)
    for name, data in all_results.items():
        s = data["summary"]
        print(f"{name:<30} {s['mean_final_reward']:>7.2f} ± {s['std_final_reward']:<7.2f}")
    print("-" * 50)

    return all_results


def _make_serializable(obj):
    """Convert numpy types to Python types for JSON."""
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
    parser = argparse.ArgumentParser(description="Run SurNoR experiments")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        choices=get_experiment_names(),
        help="Experiments to run",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all experiments",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Total timesteps per run (default: from config)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=10,
        help="Number of seeds to run (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory",
    )

    args = parser.parse_args()

    if args.all:
        experiments = get_experiment_names()
    elif args.experiments:
        experiments = args.experiments
    else:
        # Default: run key comparison experiments
        experiments = ["baseline", "surnor_pearce_hall", "surnor_stabilize"]

    run_experiments(
        experiment_names=experiments,
        n_seeds=args.seeds,
        total_timesteps=args.timesteps,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
