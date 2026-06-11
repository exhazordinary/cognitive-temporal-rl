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

## ⚠️ Important: prior results are unvalidated (June 2026)

A code review found that **the LR modulation never actually reached PPO's
gradient updates**: SB3's `PPO.train()` re-applies the (constant) LR schedule
as its first action, silently overwriting the value the callback had written
into the optimizer. Two further bugs were found in the transition pairing fed
to the forward model (action off by one step, and surprise computed across
episode auto-reset boundaries). All three are fixed, with regression tests.

**Consequence:** every number in the tables below — including the headline
+14.8% stabilization result — predates these fixes and must be re-run before
being cited. See `docs/FINDINGS.md` ("June 2026") for details. Training is
now vectorized, so the re-run is considerably cheaper.

## Current Status

### Original Approach (Entropy-based) - FAILED

| Condition | Mean Final Reward | Std Dev |
|-----------|-------------------|---------|
| **Baseline PPO** | **151.27** | 51.77 |
| Salience LR | 70.35 | 77.27 |

**Why it failed:** See `docs/RESEARCH_SYNTHESIS.md` for detailed analysis.

### New Approach (SurNoR-inspired) - PRE-BUGFIX RESULTS (re-run required)

| Experiment | Mean Final Reward | Std Dev | vs Baseline |
|------------|-------------------|---------|-------------|
| Baseline PPO | 118.50 | 70.76 | - |
| Stabilization γ=0.3 | 100.64 | 87.30 | -15.1% |
| Pearce-Hall γ=0.1 | 125.16 | 60.75 | +5.6% |
| **Stabilization γ=0.1** | **135.99** | **54.59** | **+14.8%** |

*10 runs × 200k timesteps, random seeds*

**Key Findings:**

1. **Stabilization with slow adaptation wins** - The stabilization approach with γ=0.1 achieves:
   - Highest mean reward (135.99)
   - Lowest variance (54.59) - more consistent learning
   - 14.8% improvement over baseline

2. **Gamma parameter is critical** - Slow adaptation (γ=0.1) vastly outperforms fast adaptation (γ=0.3):
   - Fast γ=0.3: Hurt performance (100.64 mean, high 87.30 variance)
   - Slow γ=0.1: Best performance (135.99 mean, low 54.59 variance)

3. **Stabilization beats Pearce-Hall** - At matched γ=0.1:
   - Stabilization: 135.99 mean
   - Pearce-Hall: 125.16 mean

**Theoretical Interpretation:** See `docs/THEORETICAL_FOUNDATION.md` for full analysis. In short:
- High prediction error in LunarLander is mostly **noise** (stochastic physics), not **volatility** (environment change)
- Gershman (2020): "Learning should slow down as unpredictability increases"
- Adam optimizer implements this principle via signal-to-noise ratio scaling
- Slow γ allows the system to average over noise rather than react to every fluctuation

## LR Modulation Modes

Set per experiment via `lr_mode` in `src/experiments/surnor_config.py`:

| Mode | Behavior |
|------|----------|
| `none` | No modulation (baseline; forward model still trains) |
| `pearce_hall` | High surprise → higher LR |
| `stabilize` | High surprise → lower LR |
| `adaptive` | VolatilityDetector picks the direction per rollout: detected change points boost LR, noise-dominated rollouts reduce it (Gershman 2020) |

## Key Files

| Component | File | Description |
|-----------|------|-------------|
| Forward Model | `src/surprise/forward_model.py` | Predicts next state, computes PE (batched) |
| Surprise Module | `src/surprise/surprise_module.py` | Pearce-Hall smoothing, rollout aggregation |
| Volatility Detector | `src/surprise/volatility_detector.py` | Distinguishes volatility from noise (`lr_mode="adaptive"`) |
| Pearce-Hall LR | `src/modulators/pearce_hall_lr.py` | LR modulation at update time |
| SurNoR PPO | `src/agents/surnor_ppo.py` | Vectorized PPO with surprise-modulated LR |
| Experiments | `src/experiments/run_surnor.py` | Experiment runner (`--env`, `--n-envs`) |
| Statistics | `src/analysis/stats.py` | Bootstrap CIs, Welch's t-test, IQM, Hedges' g |
| Theory | `docs/THEORETICAL_FOUNDATION.md` | Why stabilization works |
| Research | `docs/RESEARCH_SYNTHESIS.md` | Full literature review |
| Future Work | `docs/FUTURE_WORK.md` | Next research directions |

## Quick Start

```bash
# Setup (needs swig on the system for box2d: apt install swig)
cd cognitive-temporal-rl
uv venv && uv sync --extra dev

# Smoke test: verify everything works end to end (~2 min on CPU)
uv run python -m src.experiments.run_surnor \
  --env CartPole-v1 --experiments baseline surnor_stabilize surnor_adaptive \
  --n-envs 8 --timesteps 10000 --seeds 1
uv run python scripts/analyze_results.py --results results/surnor_CartPole-v1_*.json

# Full LunarLander comparison (10 seeds x 200k steps each)
uv run python -m src.experiments.run_surnor \
  --experiments baseline surnor_pearce_hall surnor_stabilize surnor_adaptive

# Other environments
uv run python -m src.experiments.run_surnor --env Acrobot-v1 --experiments baseline surnor_stabilize

# Run tests
uv run pytest tests/ -v
```

Training uses 8 parallel environments by default (`--n-envs` to change).
The rollout size (`n_steps_total=2048`) is defined in env-steps and split
across envs, so the number of PPO updates is independent of `n_envs`. On a
4-core CPU, 50k LunarLander steps: 84s at 1 env → 55s at 8 → 40s at 16.

Available environments (`--env`): `LunarLander-v3` (main benchmark),
`Acrobot-v1` (discriminative), `CartPole-v1` (smoke/sanity, reward ceiling),
`MountainCar-v0` (exploratory — sparse reward, PPO often floors at −200).

The analysis script reports mean ± std, 95% bootstrap CIs, IQM, and a
vs-baseline table with Welch's t-test and Hedges' g, grouped per environment.

## Project Structure

```
cognitive-temporal-rl/
├── src/
│   ├── surprise/                 # Prediction-error based surprise
│   │   ├── forward_model.py      # Predicts next state (batched)
│   │   ├── surprise_module.py    # Pearce-Hall smoothing
│   │   └── volatility_detector.py # Volatility vs noise (adaptive mode)
│   ├── modulators/
│   │   └── pearce_hall_lr.py     # LR modulation at update time
│   ├── agents/
│   │   ├── surnor_ppo.py         # Vectorized SurNoR PPO
│   │   └── base_ppo.py           # Vanilla PPO baseline
│   ├── analysis/
│   │   └── stats.py              # Bootstrap CI, Welch, IQM, Hedges' g
│   └── experiments/
│       ├── run_surnor.py         # Experiment runner
│       └── surnor_config.py      # Configs + per-env defaults
├── legacy/                       # Deprecated entropy-clock approach (see legacy/README.md)
├── docs/
│   ├── RESEARCH_SYNTHESIS.md     # Literature review & analysis
│   └── FINDINGS.md               # Experiment findings & bug history
├── scripts/
│   └── analyze_results.py        # Statistics + plots
├── results/                      # Experiment outputs (JSON)
└── tests/
```

The original entropy-based approach (failed; see `docs/FINDINGS.md`) is
preserved under `legacy/` for reference.

## References

- **SurNoR:** [Novelty is not surprise (2021)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009070)
- **Pearce-Hall:** [Prediction errors, attention and associative learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC4862921/)
- **Gershman:** [Unpredictability vs. volatility and the control of learning (2020)](https://www.biorxiv.org/content/10.1101/2020.10.05.327007v2)
- **RND:** [Exploration by Random Network Distillation (2018)](https://arxiv.org/abs/1810.12894)
- **ICM:** [Curiosity-driven Exploration (2017)](https://arxiv.org/abs/1705.05363)
- **PPO:** [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
