#!/usr/bin/env python
"""Analyze experiment results and generate diagnostic plots."""

import argparse
import json
from pathlib import Path
import numpy as np

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.stats import bootstrap_ci, bootstrap_diff_ci, hedges_g, iqm, welch_t_test
from src.visualization.diagnostics import (
    plot_lr_trajectory,
    plot_salience_trajectory,
    plot_learning_curves,
    plot_divergence_analysis,
    compute_divergence_point,
)


def load_results(results_paths: list[str]) -> dict:
    """Load and merge results from one or more JSON files, grouped by env.

    Args:
        results_paths: List of paths to result JSON files

    Returns:
        Nested dict: {env_name: {experiment_name: exp_data}}.
        Grouping by env ensures runs from different environments are
        never pooled into one comparison.
    """
    merged = {}
    for path in results_paths:
        with open(path) as f:
            data = json.load(f)
        for exp_name, exp_data in data.items():
            runs = exp_data.get("runs", [])
            env_name = "unknown"
            if runs:
                env_name = runs[0].get("config", {}).get("env_name", "unknown")
            # Merge experiments, newer files overwrite older ones
            merged.setdefault(env_name, {})[exp_name] = exp_data
    return merged


def final_rewards_per_seed(exp_data: dict) -> np.ndarray:
    """Seed-level final performance: mean of last 100 episode rewards per run."""
    finals = []
    for run in exp_data.get("runs", []):
        rewards = run.get("episode_rewards", [])
        if rewards:
            finals.append(float(np.mean(rewards[-100:])))
    return np.asarray(finals)


def print_summary(results: dict, env_name: str, rng: np.random.Generator) -> None:
    """Print summary table with bootstrap CIs and IQM per experiment."""
    print("\n" + "=" * 92)
    print(f"EXPERIMENT SUMMARY - {env_name}")
    print("=" * 92)
    print(f"{'Experiment':<30} {'Mean ± Std':>18} {'95% CI':>22} {'IQM':>10} {'Runs':>6}")
    print("-" * 92)

    for exp_name, exp_data in results.items():
        finals = final_rewards_per_seed(exp_data)
        if finals.size == 0:
            continue
        ci = bootstrap_ci(finals, rng=rng)
        print(
            f"{exp_name:<30} "
            f"{np.mean(finals):>9.2f} ± {np.std(finals):<6.2f} "
            f"[{ci.low:>8.2f}, {ci.high:>8.2f}] "
            f"{iqm(finals):>10.2f} "
            f"{finals.size:>6}"
        )
        if finals.size < 5:
            print(f"{'':<30} CAUTION: only {finals.size} run(s) - statistics unreliable")

    print("=" * 92)


def print_baseline_comparison(results: dict, env_name: str, rng: np.random.Generator) -> None:
    """Print each experiment vs baseline: diff of means, CI, Welch test, effect size."""
    if "baseline" not in results:
        return
    baseline = final_rewards_per_seed(results["baseline"])
    if baseline.size < 2:
        print("\n(vs-baseline comparison skipped: baseline has < 2 runs)")
        return

    print(f"\nVS BASELINE - {env_name}")
    print("-" * 92)
    print(f"{'Experiment':<30} {'ΔMean':>9} {'95% CI of Δ':>22} {'Welch t':>9} {'p':>8} {'g':>7}")
    print("-" * 92)

    for exp_name, exp_data in results.items():
        if exp_name == "baseline":
            continue
        treatment = final_rewards_per_seed(exp_data)
        if treatment.size < 2:
            print(f"{exp_name:<30} skipped (< 2 runs)")
            continue
        diff = bootstrap_diff_ci(treatment, baseline, rng=rng)
        welch = welch_t_test(treatment, baseline)
        g = hedges_g(treatment, baseline)
        print(
            f"{exp_name:<30} "
            f"{diff.estimate:>9.2f} "
            f"[{diff.low:>8.2f}, {diff.high:>8.2f}] "
            f"{welch.statistic:>9.2f} "
            f"{welch.p_value:>8.4f} "
            f"{g:>7.2f}"
        )

    print("-" * 92)


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
        nargs="+",
        required=True,
        help="Path(s) to results JSON file(s)",
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

    # Load results grouped by environment (never pool across envs)
    results_by_env = load_results(args.results)
    output_dir = Path(args.output)
    rng = np.random.default_rng(0)  # fixed seed: reproducible CIs

    for env_name, results in results_by_env.items():
        print_summary(results, env_name, rng)
        print_baseline_comparison(results, env_name, rng)

        # Generate comparison plots (per env subdir when multiple envs)
        env_output = output_dir / env_name if len(results_by_env) > 1 else output_dir
        print("\nGenerating comparison plots...")
        compare_experiments(results, env_output)

        # Generate detailed diagnostics if requested
        if args.detailed:
            print("\nGenerating detailed diagnostics...")
            for exp_name, exp_data in results.items():
                if "runs" in exp_data and len(exp_data["runs"]) > args.run_index:
                    run_data = exp_data["runs"][args.run_index]
                    prefix = f"{exp_name}_run{args.run_index}"
                    analyze_single_run(run_data, env_output / "detailed", prefix)

    print(f"\nPlots saved to: {output_dir}")


if __name__ == "__main__":
    main()
