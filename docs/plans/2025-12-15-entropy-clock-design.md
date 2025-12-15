# Cognitive Temporal RL - Design Document

## Overview

An entropy-based internal clock module that creates a cognitive analog of temporal experience in RL agents. The system measures entropy changes in internal state distributions to produce salience signals, which then modulate training dynamics.

**Goal:** Test whether entropy-driven temporal salience improves RL agent performance and produces interpretable "cognitive time" dynamics.

**Inspiration:** Reddit discussion on modeling temporal cognition in AI without invoking consciousness - using entropy as a proxy for subjective time dilation.

---

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

**Design Principles:**
- Entropy-clock is environment-agnostic (takes any state vector)
- Modulators are pluggable and independently toggleable
- Clean separation enables future LLM extension

---

## Entropy-Clock Module

### Inputs
- `state_vector`: Current state embedding (shape: `[batch, state_dim]`)
- `window_size`: How many past states to consider (default: 50)

### Internal State
- `state_buffer`: Rolling window of recent states
- `entropy_history`: Tracked entropy values over time
- `baseline_entropy`: Running average for normalization

### Entropy Computation

Covariance-based (Gaussian approximation) for fast iteration:

```python
# Entropy ≈ 0.5 * log(det(Σ)) for Gaussian approximation
cov = torch.cov(state_buffer.T)
entropy = 0.5 * torch.logdet(cov + ε * I)
```

Alternative: KNN estimator for complex distributions (future enhancement).

### Outputs
- `entropy`: Current entropy estimate
- `delta_entropy`: Change from previous timestep
- `salience_score`: Normalized `|delta_entropy|` against baseline
- `internal_time_delta`: `f(delta_entropy)` - higher salience = more "internal time"

**Key insight:** Salience is based on *absolute* entropy change. Both sudden increases (novelty/chaos) AND sudden decreases (unexpected stability) are salient.

---

## Salience Modulators

### Modulator A: Prioritized Replay by Salience

```python
class SalienceReplay:
    def compute_priority(self, transition, salience_score):
        td_priority = abs(transition.td_error) + ε
        combined = α * td_priority + (1 - α) * salience_score
        return combined ** β
```

- High-salience transitions get replayed more often
- Hypothesis: Agent learns critical moments faster

### Modulator B: Learning Rate Modulation

```python
class SalienceLR:
    def get_lr_multiplier(self, salience_score):
        multiplier = 1.0 + γ * (salience_score - baseline)
        return clamp(multiplier, min=0.5, max=2.0)
```

- Novel moments → learn faster, routine → learn slower
- Hypothesis: More efficient credit assignment

### Modulator C: Exploration Modulation

```python
class SalienceExploration:
    def get_temperature(self, entropy, entropy_history):
        normalized = (entropy - mean(entropy_history)) / std(entropy_history)
        return base_temp * (1 - λ * tanh(normalized))
```

- High internal entropy → exploit more (already in novel territory)
- Low internal entropy → explore more (escape local patterns)
- Hypothesis: Prevents compounding chaos

---

## Experimental Design

### Conditions
1. Vanilla PPO (baseline)
2. PPO + standard PER (TD-error based)
3. PPO + Salience Replay only
4. PPO + Salience LR only
5. PPO + Salience Exploration only
6. PPO + All three combined

### Primary Metrics
- **Sample efficiency**: Episodes to reach reward threshold (200+ avg)
- **Final performance**: Converged reward after N steps
- **Stability**: Variance across seeds (5-10 runs per condition)

### Secondary Metrics
- **Salience distribution**: When do high-salience moments occur?
- **Internal time vs wall time**: Correlation with meaningful events
- **Learning dynamics**: Different loss curves across conditions

### Visualization Outputs
- Real-time salience heatmap over episodes
- "Internal clock" timeline showing subjective time dilation
- Correlation plots: salience vs reward, salience vs TD-error

### Statistical Validation
- Paired t-tests or Mann-Whitney U across seeds
- Report effect sizes, not just p-values

---

## Project Structure

```
cognitive-temporal-rl/
├── src/
│   ├── entropy_clock/
│   │   ├── __init__.py
│   │   ├── clock.py           # Core entropy computation
│   │   └── buffers.py         # State rolling window
│   ├── modulators/
│   │   ├── __init__.py
│   │   ├── replay.py          # Salience-based replay
│   │   ├── learning_rate.py   # LR modulation
│   │   └── exploration.py     # Temperature modulation
│   ├── agents/
│   │   ├── base_ppo.py        # Vanilla PPO baseline
│   │   └── temporal_ppo.py    # PPO with entropy-clock hooks
│   └── experiments/
│       ├── config.py          # Hyperparameter configs
│       ├── train.py           # Training loop
│       └── evaluate.py        # Metrics & visualization
├── scripts/
│   ├── run_ablations.sh       # Run all experimental conditions
│   └── visualize.py           # Generate plots/demos
├── tests/
│   └── test_entropy_clock.py  # Unit tests for core module
├── docs/
│   └── plans/                 # Design documents
├── results/                   # Experiment outputs
├── requirements.txt
└── README.md
```

### Dependencies
- `torch` - Core ML
- `gymnasium` - LunarLander env
- `stable-baselines3` - PPO baseline
- `wandb` or `tensorboard` - Experiment tracking
- `matplotlib` / `plotly` - Visualization

---

## Implementation Phases

### Phase 1: Foundation
- Set up project structure, dependencies
- Implement `EntropyClockModule` with covariance-based entropy
- Unit tests: verify entropy calculations on synthetic data
- Vanilla PPO baseline running on LunarLander

### Phase 2: Integration & First Experiment
- Hook entropy-clock into PPO
- Implement Modulator A (Salience Replay)
- Run baseline vs salience-replay comparison
- Basic logging: salience scores, entropy history

### Phase 3: Remaining Modulators
- Implement Modulator B (Learning Rate)
- Implement Modulator C (Exploration)
- Config system to toggle modulators on/off

### Phase 4: Full Ablation Study
- Run all 6 conditions
- 5-10 seeds per condition
- Collect all metrics

### Phase 5: Visualization & Demo
- Real-time salience visualization
- "Internal clock" timeline visualization
- Interactive demo (Streamlit/Gradio)

### Phase 6: Write-up & Polish
- Document findings
- Clean up code for open-source release
- Draft blog post or paper

---

## Risks & Open Questions

### Technical Risks
1. **Noisy entropy signal** - Mitigation: smoothing window, hidden layer activations
2. **Gaussian assumption** - Mitigation: swap to KNN if needed
3. **Hyperparameter sensitivity** - Mitigation: thorough sweeps

### Research Risks
4. **Null result possible** - Still publishable as negative result
5. **Reinventing PER** - Need careful analysis of differences

### Open Questions
- Episode-scoped vs global entropy window?
- How to handle episode boundaries?
- For LLM extension: which layer's hidden states?

---

## Future Extensions

- **LLM integration**: Feed transformer hidden states into entropy-clock
- **Complex environments**: Test on Atari, MuJoCo
- **Meta-monitoring**: Track internal clock statistics for self-optimization
