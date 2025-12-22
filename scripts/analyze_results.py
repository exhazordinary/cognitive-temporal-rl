#!/usr/bin/env python
"""Analyze experiment results and generate diagnostic plots."""

import argparse
import json
from pathlib import Path
import numpy as np

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization.diagnostics import (
    plot_lr_trajectory,
    plot_salience_trajectory,
    plot_learning_curves,
    plot_divergence_analysis,
    compute_divergence_point,
)


def load_results(results_path: str) -> dict:
    """Load results from JSON file."""
    with open(results_path) as f:
        return json.load(f)


def print_summary(results: dict) -> None:
    """Print summary table of results."""
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"{'Experiment':<30} {'Mean Reward':>15} {'Std':>10} {'Runs':>8}")
    print("-" * 70)

    for exp_name, exp_data in results.items():
        if "summary" in exp_data:
            summary = exp_data["summary"]
            print(
                f"{exp_name:<30} "
                f"{summary.get('mean_final_reward', 0):>15.2f} "
                f"{summary.get('std_final_reward', 0):>10.2f} "
                f"{summary.get('n_runs', 0):>8}"
            )

    print("=" * 70)


def analyze_single_run(run_data: dict, output_dir: Path, prefix: str) -> None:
    """Generate diagnostic plots for a single run."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if diagnostics are available
    diagnostics = run_data.get("diagnostics", {})

    # LR trajectory
    if diagnostics.get("step_lr_history") and diagnostics.get("step_salience_history"):
        plot_lr_trajectory(
            lr_history=diagnostics["step_lr_history"],
            salience_history=diagnostics["step_salience_history"],
            save_path=str(output_dir / f"{prefix}_lr_trajectory.png"),
            title=f"LR Trajectory - {prefix}",
        )

    # Salience analysis
    if diagnostics.get("step_salience_history"):
        plot_salience_trajectory(
            salience_history=diagnostics["step_salience_history"],
            deviation_history=diagnostics.get("step_deviation_history"),
            save_path=str(output_dir / f"{prefix}_salience.png"),
            title=f"Salience Analysis - {prefix}",
        )


def compare_experiments(results: dict, output_dir: Path) -> None:
    """Generate comparison plots across experiments."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect episode rewards from first run of each experiment
    experiment_rewards = {}
    for exp_name, exp_data in results.items():
        if "runs" in exp_data and len(exp_data["runs"]) > 0:
            first_run = exp_data["runs"][0]
            if "episode_rewards" in first_run:
                experiment_rewards[exp_name] = {
                    "episode_rewards": first_run["episode_rewards"]
                }

    if len(experiment_rewards) >= 2:
        # Learning curves comparison
        plot_learning_curves(
            results=experiment_rewards,
            save_path=str(output_dir / "learning_curves_comparison.png"),
            title="Learning Curves Comparison",
        )

        # Divergence analysis (if baseline exists)
        if "baseline" in experiment_rewards:
            baseline_rewards = experiment_rewards["baseline"]["episode_rewards"]
            for exp_name, exp_data in experiment_rewards.items():
                if exp_name != "baseline":
                    treatment_rewards = exp_data["episode_rewards"]
                    plot_divergence_analysis(
                        baseline_rewards=baseline_rewards,
                        treatment_rewards=treatment_rewards,
                        treatment_name=exp_name,
                        save_path=str(output_dir / f"divergence_{exp_name}.png"),
                    )

                    # Print divergence point
                    div_point = compute_divergence_point(baseline_rewards, treatment_rewards)
                    if div_point:
                        print(f"{exp_name}: Divergence from baseline at episode {div_point}")


def main():
    parser = argparse.ArgumentParser(description="Analyze cognitive temporal RL experiment results")
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to results JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/plots",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Generate detailed per-run diagnostics",
    )
    parser.add_argument(
        "--run-index",
        type=int,
        default=0,
        help="Which run to analyze in detail (default: 0)",
    )

    args = parser.parse_args()

    # Load results
    results = load_results(args.results)
    output_dir = Path(args.output)

    # Print summary
    print_summary(results)

    # Generate comparison plots
    print("\nGenerating comparison plots...")
    compare_experiments(results, output_dir)

    # Generate detailed diagnostics if requested
    if args.detailed:
        print("\nGenerating detailed diagnostics...")
        for exp_name, exp_data in results.items():
            if "runs" in exp_data and len(exp_data["runs"]) > args.run_index:
                run_data = exp_data["runs"][args.run_index]
                prefix = f"{exp_name}_run{args.run_index}"
                analyze_single_run(run_data, output_dir / "detailed", prefix)

    print(f"\nPlots saved to: {output_dir}")


if __name__ == "__main__":
    main()
