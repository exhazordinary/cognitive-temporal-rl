# Cognitive Temporal RL

**Research Question:** Does giving an RL agent a "sense of time" improve learning?

## The Problem

Humans don't experience time uniformly:
- **Novel/surprising moments** → time "slows down" (you encode more, remember better)
- **Routine moments** → time "speeds up" (blurs together, less attention)

Standard RL agents treat every timestep equally. What if they didn't?

## The Hypothesis

Use **entropy changes** in the agent's state representation as a proxy for "how surprising is this moment":

```
High |ΔEntropy| = "Something novel happened"  = SALIENT MOMENT
Low  |ΔEntropy| = "Business as usual"         = ROUTINE MOMENT
```

Then modulate training based on salience:
- **Learn more** from surprising moments
- **Learn less** from routine moments

Expected benefits: faster learning, better sample efficiency, more robust policies.

## Architecture

```
State observation
       ↓
┌─────────────────────────────────────┐
│      Entropy Clock Module           │
│  Computes entropy over state window │
│  Outputs: salience_score, Δentropy  │
└─────────────────────────────────────┘
       ↓
┌─────────────────────────────────────┐
│      Salience Modulators            │
│  • Learning Rate modulation         │
│  • Exploration/exploitation balance │
│  • Experience replay prioritization │
└─────────────────────────────────────┘
       ↓
┌─────────────────────────────────────┐
│      PPO Agent (LunarLander)        │
│  Training with modulated parameters │
└─────────────────────────────────────┘
```

## Current Findings

### Experiment: Baseline vs Salience LR (5 seeds, 200k timesteps)

| Condition | Mean Final Reward | Std Dev |
|-----------|-------------------|---------|
| **Baseline PPO** | **151.27** | 51.77 |
| Salience LR | 70.35 | 77.27 |

**Result:** Baseline wins. The current salience LR implementation hurts performance.

### Interpretation

The hypothesis as currently implemented doesn't work. Possible reasons:
1. LR modulation too aggressive (γ=0.5 might be too strong)
2. Per-step modulation too noisy (should modulate per-episode?)
3. Direction might be wrong (high salience should *decrease* LR to stabilize?)
4. The hypothesis itself might not apply to PPO

### Next Steps
- [ ] Tune hyperparameters (try γ=0.1, 0.2, 0.3)
- [ ] Test other modulators (exploration, replay)
- [ ] Try inverting the hypothesis
- [ ] Analyze when/why it diverges

## Quick Start

```bash
# Setup
cd cognitive-temporal-rl
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Run tests
uv run pytest tests/ -v

# Train and watch agent
uv run python -m src.agents.temporal_ppo
uv run python scripts/watch.py --model models/temporal_ppo_lunarlander

# Run ablation study
uv run python -m src.experiments.train --experiments baseline salience_lr --timesteps 200000 --seeds 5
```

## Project Structure

```
cognitive-temporal-rl/
├── src/
│   ├── entropy_clock/         # Core: entropy computation
│   │   ├── clock.py           # EntropyClockModule - the main idea
│   │   └── buffers.py         # Rolling state buffer
│   ├── modulators/            # Experiments: how salience affects training
│   │   ├── learning_rate.py   # Modulate LR by salience
│   │   ├── exploration.py     # Modulate exploration by entropy
│   │   └── replay.py          # Prioritize replay by salience
│   ├── agents/
│   │   ├── base_ppo.py        # Vanilla PPO baseline
│   │   └── temporal_ppo.py    # PPO + entropy clock
│   └── experiments/
│       ├── config.py          # Experiment configurations
│       └── train.py           # Ablation runner
├── scripts/
│   └── watch.py               # Visualize agent playing
├── tests/
├── results/                   # Experiment outputs (JSON)
└── docs/plans/
```

## Key Code Locations

| Concept | File | Function/Class |
|---------|------|----------------|
| Entropy calculation | `entropy_clock/clock.py` | `_compute_covariance_entropy()` |
| Salience scoring | `entropy_clock/clock.py` | `EntropyClockModule.step()` |
| LR modulation | `modulators/learning_rate.py` | `SalienceLR.get_lr()` |
| Integration point | `agents/temporal_ppo.py` | `TemporalCallback._on_step()` |

## Background

This project explores whether cognitive-inspired temporal mechanisms can improve RL training. The core idea:

- Humans experience "time dilation" during novel events
- This can be modeled computationally using entropy as a proxy
- An "internal clock" that speeds up during routine and slows during novelty might help agents focus on what matters

The goal is to test whether this mechanism has practical benefits for RL, not to make claims about AI consciousness.

## References

- PPO: [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- Environment: [LunarLander-v3](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
