"""Diagnostic visualization tools for analyzing experiment results."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional


def smooth(data: list[float], window: int = 100) -> np.ndarray:
    """Apply rolling average smoothing."""
    if len(data) < window:
        return np.array(data)
    return np.convolve(data, np.ones(window) / window, mode='valid')


def plot_lr_trajectory(
    lr_history: list[float],
    salience_history: list[float],
    base_lr: float = 3e-4,
    save_path: Optional[str] = None,
    title: str = "Learning Rate Trajectory",
) -> plt.Figure:
    """Plot LR and salience over training steps.

    Args:
        lr_history: Learning rate at each step
        salience_history: Salience score at each step
        base_lr: Base learning rate for reference line
        save_path: Optional path to save the figure
        title: Plot title

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    steps = np.arange(len(lr_history))

    # Panel 1: Learning rate over time
    axes[0].plot(steps, lr_history, alpha=0.3, color='blue', label='LR (raw)')
    if len(lr_history) >= 100:
        smoothed_lr = smooth(lr_history, 100)
        axes[0].plot(
            np.arange(len(smoothed_lr)) + 50,
            smoothed_lr,
            color='blue',
            linewidth=2,
            label='LR (smoothed)',
        )
    axes[0].axhline(y=base_lr, color='red', linestyle='--', label=f'Base LR ({base_lr})')
    axes[0].set_ylabel("Learning Rate")
    axes[0].set_title(title)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Salience over time
    axes[1].plot(steps[:len(salience_history)], salience_history, alpha=0.3, color='orange', label='Salience (raw)')
    if len(salience_history) >= 100:
        smoothed_sal = smooth(salience_history, 100)
        axes[1].plot(
            np.arange(len(smoothed_sal)) + 50,
            smoothed_sal,
            color='orange',
            linewidth=2,
            label='Salience (smoothed)',
        )
    axes[1].axhline(y=1.0, color='red', linestyle='--', label='Baseline (1.0)')
    axes[1].set_ylabel("Salience Score")
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    # Panel 3: LR multiplier distribution over time
    if len(lr_history) > 0:
        multipliers = np.array(lr_history) / base_lr
        axes[2].plot(steps, multipliers, alpha=0.3, color='green', label='Multiplier (raw)')
        if len(multipliers) >= 100:
            smoothed_mult = smooth(list(multipliers), 100)
            axes[2].plot(
                np.arange(len(smoothed_mult)) + 50,
                smoothed_mult,
                color='green',
                linewidth=2,
                label='Multiplier (smoothed)',
            )
        axes[2].axhline(y=1.0, color='red', linestyle='--', label='No modulation (1.0)')
        axes[2].set_ylabel("LR Multiplier")
        axes[2].set_xlabel("Training Step")
        axes[2].legend(loc='upper right')
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_salience_trajectory(
    salience_history: list[float],
    deviation_history: Optional[list[float]] = None,
    save_path: Optional[str] = None,
    title: str = "Salience Analysis",
) -> plt.Figure:
    """Plot detailed salience analysis.

    Args:
        salience_history: Salience scores over time
        deviation_history: Optional deviation from baseline
        save_path: Optional path to save figure
        title: Plot title

    Returns:
        matplotlib Figure
    """
    n_panels = 3 if deviation_history else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 4 * n_panels))

    steps = np.arange(len(salience_history))

    # Panel 1: Raw salience
    axes[0].plot(steps, salience_history, alpha=0.5, color='orange')
    if len(salience_history) >= 100:
        smoothed = smooth(salience_history, 100)
        axes[0].plot(np.arange(len(smoothed)) + 50, smoothed, color='darkorange', linewidth=2)
    axes[0].set_ylabel("Salience")
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Salience distribution over time (histogram)
    n_bins = min(50, len(salience_history) // 1000 + 1)
    chunk_size = len(salience_history) // n_bins if n_bins > 0 else len(salience_history)
    if chunk_size > 0:
        chunk_means = [np.mean(salience_history[i:i + chunk_size]) for i in range(0, len(salience_history), chunk_size)]
        chunk_stds = [np.std(salience_history[i:i + chunk_size]) for i in range(0, len(salience_history), chunk_size)]
        chunk_x = np.arange(len(chunk_means)) * chunk_size + chunk_size // 2
        axes[1].errorbar(chunk_x, chunk_means, yerr=chunk_stds, fmt='o-', capsize=3, alpha=0.7)
        axes[1].set_ylabel("Salience (mean ± std)")
        axes[1].set_xlabel("Training Step")
        axes[1].grid(True, alpha=0.3)

    # Panel 3: Deviation from baseline (if provided)
    if deviation_history:
        axes[2].plot(steps[:len(deviation_history)], deviation_history, alpha=0.5, color='purple')
        if len(deviation_history) >= 100:
            smoothed = smooth(deviation_history, 100)
            axes[2].plot(np.arange(len(smoothed)) + 50, smoothed, color='darkviolet', linewidth=2)
        axes[2].axhline(y=0, color='red', linestyle='--', label='Zero deviation')
        axes[2].set_ylabel("Deviation from Baseline")
        axes[2].set_xlabel("Training Step")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_learning_curves(
    results: dict[str, dict],
    metric: str = "episode_rewards",
    window: int = 50,
    save_path: Optional[str] = None,
    title: str = "Learning Curves Comparison",
) -> plt.Figure:
    """Plot learning curves for multiple experiments.

    Args:
        results: Dict mapping experiment names to their results
        metric: Which metric to plot (episode_rewards, episode_saliences, etc.)
        window: Smoothing window
        save_path: Optional path to save figure
        title: Plot title

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for (name, data), color in zip(results.items(), colors):
        if metric not in data:
            continue

        values = data[metric]
        if len(values) == 0:
            continue

        # Plot raw data with low alpha
        episodes = np.arange(len(values))
        ax.plot(episodes, values, alpha=0.2, color=color)

        # Plot smoothed
        if len(values) >= window:
            smoothed = smooth(values, window)
            ax.plot(
                np.arange(len(smoothed)) + window // 2,
                smoothed,
                color=color,
                linewidth=2,
                label=f"{name} (final: {np.mean(values[-50:]):.1f})",
            )
        else:
            ax.plot(episodes, values, color=color, linewidth=2, label=name)

    ax.set_xlabel("Episode")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_divergence_analysis(
    baseline_rewards: list[float],
    treatment_rewards: list[float],
    treatment_name: str = "Treatment",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Analyze when and how treatment diverges from baseline.

    Args:
        baseline_rewards: Episode rewards for baseline
        treatment_rewards: Episode rewards for treatment
        treatment_name: Name of the treatment condition
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    min_len = min(len(baseline_rewards), len(treatment_rewards))
    baseline = np.array(baseline_rewards[:min_len])
    treatment = np.array(treatment_rewards[:min_len])
    episodes = np.arange(min_len)

    # Panel 1: Both curves with difference shading
    window = 50
    if min_len >= window:
        baseline_smooth = smooth(list(baseline), window)
        treatment_smooth = smooth(list(treatment), window)
        x = np.arange(len(baseline_smooth)) + window // 2

        axes[0].plot(x, baseline_smooth, color='blue', linewidth=2, label='Baseline')
        axes[0].plot(x, treatment_smooth, color='orange', linewidth=2, label=treatment_name)
        axes[0].fill_between(
            x,
            baseline_smooth,
            treatment_smooth,
            where=baseline_smooth > treatment_smooth,
            alpha=0.3,
            color='blue',
            label='Baseline winning',
        )
        axes[0].fill_between(
            x,
            baseline_smooth,
            treatment_smooth,
            where=treatment_smooth > baseline_smooth,
            alpha=0.3,
            color='orange',
            label=f'{treatment_name} winning',
        )
    else:
        axes[0].plot(episodes, baseline, color='blue', alpha=0.7, label='Baseline')
        axes[0].plot(episodes, treatment, color='orange', alpha=0.7, label=treatment_name)

    axes[0].set_ylabel("Episode Reward")
    axes[0].set_title("Learning Curves with Divergence")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Cumulative difference
    diff = treatment - baseline
    cumulative_diff = np.cumsum(diff)

    axes[1].plot(episodes, cumulative_diff, color='purple', linewidth=2)
    axes[1].axhline(y=0, color='red', linestyle='--', label='Break-even')
    axes[1].fill_between(episodes, 0, cumulative_diff, where=cumulative_diff > 0, alpha=0.3, color='green')
    axes[1].fill_between(episodes, 0, cumulative_diff, where=cumulative_diff < 0, alpha=0.3, color='red')

    # Find divergence point
    divergence_point = compute_divergence_point(baseline_rewards, treatment_rewards)
    if divergence_point is not None:
        axes[1].axvline(x=divergence_point, color='black', linestyle=':', label=f'Divergence @ ep {divergence_point}')

    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Cumulative Advantage")
    axes[1].set_title(f"Cumulative Difference ({treatment_name} - Baseline)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def compute_divergence_point(
    baseline_rewards: list[float],
    treatment_rewards: list[float],
    window: int = 50,
    threshold: float = 10.0,
) -> Optional[int]:
    """Find the episode where treatment consistently diverges from baseline.

    Args:
        baseline_rewards: Episode rewards for baseline
        treatment_rewards: Episode rewards for treatment
        window: Window for computing rolling difference
        threshold: Minimum absolute difference to count as divergence

    Returns:
        Episode number where divergence starts, or None if no clear divergence
    """
    min_len = min(len(baseline_rewards), len(treatment_rewards))
    if min_len < window:
        return None

    baseline = np.array(baseline_rewards[:min_len])
    treatment = np.array(treatment_rewards[:min_len])

    # Compute rolling mean difference
    diff = treatment - baseline
    rolling_diff = np.convolve(diff, np.ones(window) / window, mode='valid')

    # Find first point where rolling difference exceeds threshold consistently
    consecutive = 0
    required_consecutive = window // 2

    for i, d in enumerate(rolling_diff):
        if abs(d) > threshold:
            consecutive += 1
            if consecutive >= required_consecutive:
                return i - required_consecutive + window // 2
        else:
            consecutive = 0

    return None
