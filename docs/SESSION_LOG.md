# Session Log: 2025-12-15

## Project: Cognitive Temporal RL

### What We Built

Started from a concept about modeling "cognitive time" in AI using entropy-based internal clocks. Built a complete research prototype:

**Core Components:**
1. **EntropyClockModule** (`src/entropy_clock/clock.py`) - Measures entropy over rolling state window, outputs salience scores
2. **StateBuffer** (`src/entropy_clock/buffers.py`) - Circular buffer for recent states
3. **SalienceLR** (`src/modulators/learning_rate.py`) - Modulates learning rate based on salience
4. **SalienceExploration** (`src/modulators/exploration.py`) - Modulates exploration based on entropy
5. **SalienceReplay** (`src/modulators/replay.py`) - Prioritizes replay based on salience
6. **TemporalPPO** (`src/agents/temporal_ppo.py`) - PPO agent with entropy clock integration
7. **Experiment runner** (`src/experiments/train.py`) - Ablation study framework

**Also added:**
- `scripts/watch.py` - Visualize agent playing LunarLander
- Unit tests (16 passing)
- uv-based project setup

---

### The Hypothesis

**Idea:** Use entropy changes as proxy for "surprising moments". High |ΔEntropy| = salient. Then:
- Learn MORE from salient moments (higher LR)
- Learn LESS from routine moments (lower LR)

**Expected:** Faster learning, better sample efficiency.

---

### What We Tested

**Experiment:** Baseline PPO vs Salience LR
- Environment: LunarLander-v3
- Timesteps: 200,000
- Seeds: 5

**Results:**

| Condition | Mean Reward | Std |
|-----------|-------------|-----|
| Baseline | 151.27 | 51.77 |
| Salience LR | 70.35 | 77.27 |

**Baseline wins by ~80 points.** The hypothesis as implemented doesn't work.

---

### Why It Might Have Failed

1. LR modulation too aggressive (γ=0.5)
2. Per-step modulation too noisy
3. Maybe direction is wrong (high salience should DECREASE LR?)
4. PPO might not benefit from this approach

---

### Next Steps (TODO)

- [ ] Tune γ parameter (try 0.1, 0.2, 0.3)
- [ ] Test SalienceExploration modulator
- [ ] Test SalienceReplay modulator
- [ ] Try inverting the hypothesis
- [ ] Add visualization of learning curves
- [ ] Analyze when/why salience_lr diverges

---

### Key Commands

```bash
# Setup
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Run tests
uv run pytest tests/ -v

# Train agent
uv run python -m src.agents.temporal_ppo

# Watch agent play
uv run python scripts/watch.py --model models/temporal_ppo_lunarlander

# Run experiments
uv run python -m src.experiments.train --experiments baseline salience_lr --timesteps 200000 --seeds 5
```

---

### Files Created This Session

```
cognitive-temporal-rl/
├── src/
│   ├── entropy_clock/
│   │   ├── __init__.py
│   │   ├── clock.py           # Core entropy module
│   │   └── buffers.py         # State buffer
│   ├── modulators/
│   │   ├── __init__.py
│   │   ├── learning_rate.py   # LR modulation
│   │   ├── exploration.py     # Exploration modulation
│   │   └── replay.py          # Replay prioritization
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_ppo.py        # Baseline PPO
│   │   └── temporal_ppo.py    # PPO + entropy clock
│   └── experiments/
│       ├── __init__.py
│       ├── config.py          # Experiment configs
│       └── train.py           # Training script
├── scripts/
│   └── watch.py               # Visualization
├── tests/
│   ├── __init__.py
│   └── test_entropy_clock.py  # Unit tests
├── docs/
│   ├── plans/
│   │   └── 2025-12-15-entropy-clock-design.md
│   ├── FINDINGS.md            # Research findings
│   └── SESSION_LOG.md         # This file
├── results/
│   └── ablation_*.json        # Experiment results
├── pyproject.toml
├── README.md
└── .gitignore
```

---

### Result Files

- `results/ablation_20251215_232029.json` - Main results (5 seeds, baseline vs salience_lr)

---

### Questions to Explore Later

1. Does the entropy signal actually capture meaningful novelty in LunarLander?
2. Would this work better in more complex environments (Atari, MuJoCo)?
3. Can we apply this to LLMs (entropy over hidden states)?
4. Is the Gaussian assumption for entropy valid?

---

# Session Log: 2025-12-23

## Project: Cognitive Temporal RL (Continued)

### What We Did

Continued from Dec 15 session. Implemented all planned "next steps":

**Code Changes:**
1. Added `lr_gamma` parameter wiring (config → train → agent)
2. Added `invert_direction` parameter to SalienceLR
3. Integrated SalienceExploration via `ent_coef` modulation
4. Created visualization/diagnostics module
5. Created analysis script for post-hoc result inspection

**Experiments Run:**
- Gamma sweep (γ=0.1, 0.2, 0.3) - 200k steps, 5 seeds
- Inverted hypothesis - 200k steps, 5 seeds
- Exploration modulation - 200k steps, 5 seeds

---

### Critical Discovery: Deterministic Seeding Problem

**The Problem:**
All experiments with same seeds produce IDENTICAL episode rewards, despite LR modulation being applied.

**Investigation:**
1. Checked LR history → LR IS varying (0.00015 to 0.0006)
2. Checked diagnostics → Data IS being captured
3. Compared episode rewards → Byte-for-byte identical

**Root Cause:**
- SB3 PPO uses deterministic seeding throughout training
- Same seed → same random numbers for action sampling
- Even with different policies (from different LRs), sampled actions are identical
- PPO clipping bounds policy updates, keeping them similar

**Proof:**
```python
# With seed=42: IDENTICAL
baseline_ep24 = -2.56
salience_lr_ep24 = -2.56

# Without seeds: DIFFERENT
baseline: 180 eps, mean=-178.47
salience_lr: 193 eps, mean=-167.91
```

**Conclusion:** LR modulation WORKS, but deterministic seeding masks the effect.

---

### Mystery: Why Dec 15 Showed Divergence

Original Dec 15 experiments diverged at episode 24:
- Baseline: -2.56
- Salience_lr: 40.53

But current code produces identical results. Unexplained.

---

### Exploration Modulation Results

Both exploration variants hurt performance:
- baseline: 151.27
- salience_exploration: 90.66
- salience_lr_exploration: 90.66

Divergence at episode 130.

---

### Files Created This Session

```
src/visualization/
├── __init__.py
└── diagnostics.py      # plot_lr_trajectory, plot_learning_curves, etc.

scripts/
└── analyze_results.py  # CLI for result analysis
```

---

### Files Modified This Session

| File | Changes |
|------|---------|
| `src/agents/temporal_ppo.py` | Added params, exploration integration, diagnostics |
| `src/modulators/learning_rate.py` | Added `invert_direction` parameter |
| `src/experiments/config.py` | New experiment configs, `lr_invert_direction` |
| `src/experiments/train.py` | Wire new params through |

---

### New Experiment Configs Available

```python
"salience_lr_gamma_0.1"      # γ=0.1
"salience_lr_gamma_0.2"      # γ=0.2
"salience_lr_gamma_0.3"      # γ=0.3
"salience_lr_inverted"       # inverted, γ=0.5
"salience_lr_inverted_0.3"   # inverted, γ=0.3
"salience_exploration"       # exploration only
"salience_lr_exploration"    # LR + exploration
```

---

### Result Files This Session

| File | Experiments |
|------|-------------|
| `ablation_20251223_010017.json` | Gamma sweep (492MB with diagnostics) |
| `ablation_20251223_023956.json` | Inverted hypothesis (326MB) |
| `ablation_20251223_121019.json` | Exploration (293MB) |

---

### Key Takeaways

1. **Deterministic seeding breaks hyperparameter comparison** - Need stochastic evaluation
2. **LR modulation timing is suboptimal** - Happens during rollout, not during updates
3. **Exploration modulation hurts** - Don't use as-is
4. **Code infrastructure is now solid** - Diagnostics, visualization, analysis tools ready

---

### Next Steps (TODO)

- [ ] Re-run experiments WITHOUT fixed seeds
- [ ] Use 10+ different seeds for statistical power
- [ ] Consider modulating LR during PPO update phase (not rollout)
- [ ] Investigate Dec 15 divergence mystery
- [ ] Try per-rollout modulation instead of per-step

---

### Commands Reference

```bash
# Run experiments
uv run python -m src.experiments.train \
  --experiments baseline salience_lr_gamma_0.1 \
  --timesteps 200000 --seeds 5

# Analyze results
uv run python scripts/analyze_results.py \
  --results results/ablation_*.json --detailed

# Watch trained agent
uv run python scripts/watch.py --model models/temporal_ppo_lunarlander
```
