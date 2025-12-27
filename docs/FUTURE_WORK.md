# Future Work

Ideas for continuing this research, prioritized by potential impact.

---

## High Priority

### 1. Test the VolatilityDetector

We implemented `src/surprise/volatility_detector.py` but haven't run experiments with it yet.

**The idea:** Dynamically switch between stabilization (when noise dominates) and Pearce-Hall (when true volatility is detected).

```python
# Current: static approach
if noise_detected:
    lr_multiplier = decrease  # stabilization

# Proposed: adaptive approach
if volatility_detected:
    lr_multiplier = increase  # Pearce-Hall (environment changed)
else:
    lr_multiplier = decrease  # stabilization (just noise)
```

**Why it matters:** Could get the best of both worlds - stable learning in noise, fast adaptation to real change.

**To run:**
```bash
# Add volatility_detector experiment to surnor_config.py
uv run python -m src.experiments.run_surnor --experiments baseline volatility_adaptive --timesteps 200000 --seeds 10
```

---

### 2. Test in Volatile Environments

LunarLander is noise-dominated (stochastic physics, no rule changes). Stabilization wins here.

**Question:** Does Pearce-Hall win in environments with actual change points?

**Candidate environments:**
- Non-stationary bandits (reward contingencies flip)
- Reversal learning tasks
- Meta-World with task switching
- ProcGen (procedurally-generated levels)

**Hypothesis:**
- Stable environments → Stabilization wins
- Volatile environments → Pearce-Hall wins
- Mixed → VolatilityDetector wins

---

## Medium Priority

### 3. Add Intrinsic Reward Bonuses (Full SurNoR)

The SurNoR paper separates:
- **Novelty** → exploration bonuses (intrinsic reward)
- **Surprise** → LR modulation

We only implemented LR modulation. Adding novelty-based intrinsic rewards could help exploration in sparse-reward environments.

```python
# Current
reward = extrinsic_reward

# Full SurNoR
novelty_bonus = compute_novelty(state)  # e.g., RND-style
reward = extrinsic_reward + beta * novelty_bonus
```

---

### 4. Statistical Significance Testing

Current results (10 seeds) show:
- Stabilization γ=0.1: 135.99 ± 54.59
- Baseline: 118.50 ± 70.76

**To do:**
- Welch's t-test or Mann-Whitney U
- Bootstrap confidence intervals
- Effect size (Cohen's d)

```python
from scipy import stats
t_stat, p_value = stats.ttest_ind(stabilization_results, baseline_results)
```

---

## Longer-Term

### 5. Harder Environments

Test on more challenging domains where surprise-based adaptation might matter more:

| Environment | Why interesting |
|-------------|-----------------|
| MuJoCo (HalfCheetah, Ant) | High-dimensional continuous control |
| Atari | Visual inputs, longer horizons |
| MiniGrid | Sparse rewards, exploration-heavy |
| Meta-World | Multi-task, requires adaptation |

---

### 6. Meta-Learn the Gamma Parameter

Instead of hand-tuning γ=0.1, learn it online:

**Options:**
1. **Population-based training** - Evolve γ across runs
2. **Learned meta-controller** - Small network predicts optimal γ from context
3. **Bayesian optimization** - Tune γ as hyperparameter

---

### 7. Combine with Other Adaptive Methods

Test interactions with:
- Learning rate schedulers (cosine annealing, warmup)
- Adaptive optimizers (Adam already does SNR-based scaling)
- Curriculum learning

**Question:** Does surprise-based LR modulation provide orthogonal benefits?

---

## Completed Work

| Task | Status | Result |
|------|--------|--------|
| Implement forward model surprise | Done | `src/surprise/forward_model.py` |
| Implement Pearce-Hall smoothing | Done | `src/surprise/surprise_module.py` |
| Fix LR modulation timing | Done | Moved to `_on_rollout_end()` |
| Test stabilization vs Pearce-Hall | Done | Stabilization +14.8% |
| Test gamma variations | Done | γ=0.1 optimal |
| Implement VolatilityDetector | Done | `src/surprise/volatility_detector.py` |
| Document theoretical foundation | Done | `docs/THEORETICAL_FOUNDATION.md` |

---

## Key References for Future Work

- [Gershman (2020)](https://www.biorxiv.org/content/10.1101/2020.10.05.327007v2) - Volatility vs unpredictability
- [SurNoR (2021)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009070) - Full novelty + surprise framework
- [Meta-World](https://meta-world.github.io/) - Multi-task benchmark
- [ProcGen](https://openai.com/research/procgen-benchmark) - Generalization benchmark
