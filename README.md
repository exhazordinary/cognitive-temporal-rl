# Cognitive Temporal RL

Entropy-based internal clock for modeling cognitive temporal experience in RL agents.

## Overview

This project implements an entropy-based "internal clock" that creates a cognitive analog of temporal experience in reinforcement learning agents. The system measures entropy changes in internal state distributions to produce salience signals, which modulate training dynamics.

**Key idea:** High entropy changes (novelty/chaos OR sudden stability) = salient moments = more "internal time" passes.

## Architecture

```
┌─────────────────────────────────────────────────┐
│           Entropy-Clock Module                  │
│  (Standalone, reusable across RL/LLM/etc.)      │
├─────────────────────────────────────────────────┤
│           Salience Modulators                   │
│  (Replay / Learning Rate / Exploration)         │
├─────────────────────────────────────────────────┤
│           Base RL Agent (PPO)                   │
│  (LunarLander environment)                      │
└─────────────────────────────────────────────────┘
```

## Quick Start

### Using uv (recommended)

```bash
# Clone and enter project
cd cognitive-temporal-rl

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
uv pip install -e ".[dev]"

# Run tests
uv run pytest tests/ -v

# Train baseline
uv run python -m src.agents.base_ppo

# Train with entropy clock (salience LR enabled)
uv run python -m src.agents.temporal_ppo

# Run full ablation study
uv run python -m src.experiments.train --experiments baseline salience_lr --timesteps 100000 --seeds 3
```

### Using pip

```bash
cd cognitive-temporal-rl
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v
```

## Experiments

The ablation study tests these conditions:

| Experiment | Salience Replay | Salience LR | Salience Exploration |
|------------|-----------------|-------------|---------------------|
| baseline | ❌ | ❌ | ❌ |
| salience_replay | ✅ | ❌ | ❌ |
| salience_lr | ❌ | ✅ | ❌ |
| salience_exploration | ❌ | ❌ | ✅ |
| all_modulators | ✅ | ✅ | ✅ |

Run specific experiments:

```bash
# Quick test (short runs)
uv run python -m src.experiments.train --experiments baseline salience_lr --timesteps 50000 --seeds 2

# Full ablation (takes longer)
uv run python -m src.experiments.train --timesteps 500000 --seeds 5
```

Results are saved to `results/` as JSON files.

## Project Structure

```
cognitive-temporal-rl/
├── src/
│   ├── entropy_clock/      # Core entropy computation
│   │   ├── clock.py        # EntropyClockModule
│   │   └── buffers.py      # StateBuffer
│   ├── modulators/         # Salience-based modulators
│   │   ├── replay.py       # Prioritized replay
│   │   ├── learning_rate.py
│   │   └── exploration.py
│   ├── agents/
│   │   ├── base_ppo.py     # Vanilla PPO baseline
│   │   └── temporal_ppo.py # PPO + entropy clock
│   └── experiments/
│       ├── config.py       # Experiment configs
│       └── train.py        # Training script
├── tests/
├── results/
├── docs/plans/
└── pyproject.toml
```

## Key Concepts

### Entropy Clock

Measures Shannon entropy over a rolling window of states using Gaussian approximation:

```
H(S) ≈ 0.5 * log(det(Σ))
```

High |ΔH| = salient moment.

### Salience Modulators

1. **Replay**: High-salience transitions get sampled more during training
2. **Learning Rate**: Higher LR during novel moments, lower during routine
3. **Exploration**: High internal entropy → exploit; Low → explore

## Future Work

- LLM integration (entropy over hidden states)
- Complex environments (Atari, MuJoCo)
- Meta-monitoring layer
