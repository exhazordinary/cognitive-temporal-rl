# Cognitive Temporal RL

**Research Question:** Does giving an RL agent a "sense of time" improve learning?

## The Problem

Humans don't experience time uniformly:
- **Novel/surprising moments** → time "slows down" (you encode more, remember better)
- **Routine moments** → time "speeds up" (blurs together, less attention)

Standard RL agents treat every timestep equally. What if they didn't?

## The Hypothesis

Use **surprise (prediction error)** as a proxy for "how unexpected is this moment":

```
High Prediction Error = "Something unexpected"  = SURPRISING
Low  Prediction Error = "As expected"           = ROUTINE
```

Then modulate learning rate based on surprise using Pearce-Hall dynamics:
- **Pearce-Hall mode:** High surprise → higher LR (learn more from surprises)
- **Stabilization mode:** High surprise → lower LR (consolidate during chaos)

## Theoretical Foundation

Based on:
1. **SurNoR (2021):** Separates novelty (exploration) from surprise (learning rate modulation)
2. **Pearce-Hall (1980):** Associability α updated by prediction error: `α = γ|PE| + (1-γ)α`
3. **RND/ICM:** Prediction error as novelty signal (we use for LR, not just rewards)

## Architecture (v2 - SurNoR-inspired)

```
State observation
       ↓
┌─────────────────────────────────────┐
│      Forward Model                  │
│  Predicts next state from (s, a)    │
│  Prediction error = Surprise        │
└─────────────────────────────────────┘
       ↓
┌─────────────────────────────────────┐
│      Surprise Module                │
│  • Pearce-Hall smoothing            │
│  • Computes associability α         │
│  • Aggregates over rollout          │
└─────────────────────────────────────┘
       ↓ (at PPO update time!)
┌─────────────────────────────────────┐
│      Pearce-Hall LR Modulator       │
│  • Modulates LR based on α          │
│  • Applied once per update          │
└─────────────────────────────────────┘
       ↓
┌─────────────────────────────────────┐
│      PPO Agent (LunarLander)        │
│  Training with modulated LR         │
└─────────────────────────────────────┘
```

**Key Fix:** LR modulation happens at PPO UPDATE time (after rollout), not per-step!

## Current Status

### Original Approach (Entropy-based) - FAILED

| Condition | Mean Final Reward | Std Dev |
|-----------|-------------------|---------|
| **Baseline PPO** | **151.27** | 51.77 |
| Salience LR | 70.35 | 77.27 |

**Why it failed:** See `docs/RESEARCH_SYNTHESIS.md` for detailed analysis.

### New Approach (SurNoR-inspired) - PRELIMINARY RESULTS

| Experiment | Mean Final Reward | Std Dev |
|------------|-------------------|---------|
| Baseline PPO | -7.93 | 49.25 |
| Pearce-Hall (high surprise → high LR) | -3.47 | 47.64 |
| **Stabilization (high surprise → low LR)** | **2.59** | **45.19** |

*5 runs × 100k timesteps, random seeds*

**Key Finding:** The **stabilization approach** (inverting Pearce-Hall) performs best. When the environment is surprising/chaotic, decreasing the learning rate helps consolidate learning rather than amplifying noise.

This contradicts the Pearce-Hall prediction but aligns with the "uncertainty-weighted learning" literature (Gershman 2022).

**Note:** High variance suggests more runs needed for statistical significance.

## Key Files

| Component | File | Description |
|-----------|------|-------------|
| Forward Model | `src/surprise/forward_model.py` | Predicts next state, computes PE |
| Surprise Module | `src/surprise/surprise_module.py` | Pearce-Hall smoothing, rollout aggregation |
| Pearce-Hall LR | `src/modulators/pearce_hall_lr.py` | LR modulation at update time |
| SurNoR PPO | `src/agents/surnor_ppo.py` | Fixed PPO with proper timing |
| Experiments | `src/experiments/run_surnor.py` | New experiment runner |
| Research | `docs/RESEARCH_SYNTHESIS.md` | Full literature review |

## Quick Start

```bash
# Setup
cd cognitive-temporal-rl
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Run new SurNoR experiments
uv run python -m src.experiments.run_surnor --experiments baseline surnor_pearce_hall surnor_stabilize

# Quick test (fewer timesteps)
uv run python -m src.experiments.run_surnor --experiments baseline surnor_pearce_hall --timesteps 50000 --seeds 3

# Run tests
uv run pytest tests/ -v
```

## Project Structure

```
cognitive-temporal-rl/
├── src/
│   ├── surprise/                 # NEW: Prediction error based surprise
│   │   ├── forward_model.py      # Predicts next state
│   │   └── surprise_module.py    # Pearce-Hall smoothing
│   ├── entropy_clock/            # LEGACY: Entropy-based approach
│   │   ├── clock.py              # EntropyClockModule
│   │   └── buffers.py            # Rolling state buffer
│   ├── modulators/
│   │   ├── pearce_hall_lr.py     # NEW: Fixed LR modulation
│   │   ├── learning_rate.py      # LEGACY: Per-step LR (broken)
│   │   └── exploration.py        # Exploration modulation
│   ├── agents/
│   │   ├── surnor_ppo.py         # NEW: Fixed PPO with proper timing
│   │   ├── temporal_ppo.py       # LEGACY: Original (has timing bug)
│   │   └── base_ppo.py           # Vanilla PPO baseline
│   └── experiments/
│       ├── run_surnor.py         # NEW: SurNoR experiment runner
│       ├── surnor_config.py      # NEW: SurNoR configurations
│       ├── train.py              # LEGACY: Original runner
│       └── config.py             # LEGACY: Original configs
├── docs/
│   ├── RESEARCH_SYNTHESIS.md     # Literature review & analysis
│   └── FINDINGS.md               # Original experiment findings
├── results/                      # Experiment outputs (JSON)
└── tests/
```

## References

- **SurNoR:** [Novelty is not surprise (2021)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009070)
- **Pearce-Hall:** [Prediction errors, attention and associative learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC4862921/)
- **RND:** [Exploration by Random Network Distillation (2018)](https://arxiv.org/abs/1810.12894)
- **ICM:** [Curiosity-driven Exploration (2017)](https://arxiv.org/abs/1705.05363)
- **PPO:** [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)

---

## Legacy Quick Start (Original Approach)

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
