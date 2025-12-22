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
