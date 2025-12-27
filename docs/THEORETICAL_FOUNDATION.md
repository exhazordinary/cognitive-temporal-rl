# Theoretical Foundation: Why Stabilization Works

## Executive Summary

Our experiments found that **high surprise → lower learning rate** (stabilization) outperforms the classic Pearce-Hall prediction (high surprise → higher learning rate). This document explains why, drawing from:

1. **Computational Neuroscience** - Volatility vs. Unpredictability distinction
2. **Optimization Theory** - Adam's signal-to-noise ratio
3. **Bayesian Learning** - Kalman filter principles

---

## The Key Insight: Volatility ≠ Unpredictability

The most important theoretical framework comes from [Soltani & Izquierdo (2019)](https://www.nature.com/articles/s41583-019-0180-y) and [Gershman's work](https://www.biorxiv.org/content/10.1101/2020.10.05.327007v2):

> "Learning should **speed up** as volatility increases but **slow down** as unpredictability increases."

### Definitions

| Term | Meaning | Effect on LR |
|------|---------|--------------|
| **Volatility** | Environment is actually changing | ↑ Increase LR |
| **Unpredictability** | Inherent stochasticity/noise | ↓ Decrease LR |

### The Problem

Our prediction error signal conflates both:
- A high prediction error could mean the environment changed (volatility)
- OR it could mean we just got unlucky with noise (unpredictability)

### Why Stabilization Wins

In LunarLander (and most environments):
- **Noise dominates** - physics is stochastic, actions have variable outcomes
- **True volatility is low** - the dynamics don't actually change

Therefore, high prediction error is mostly **unpredictability**, not volatility.
The correct response is to **decrease LR** (be skeptical of noisy signals).

---

## Evidence from Optimization Theory

### Adam Optimizer's Built-in Stabilization

From [Kingma & Ba (2014)](https://arxiv.org/abs/1412.6980):

> "The ratio m̂_t / √v̂_t is called the **signal-to-noise ratio (SNR)**. With a smaller SNR, the effective stepsize Δ_t will be closer to zero."

Adam automatically:
- Tracks gradient variance (second moment v_t)
- **Reduces effective LR when variance is high**
- This is exactly the stabilization principle!

### Mathematical Formulation

Adam's update: `θ_t = θ_{t-1} - α × m̂_t / (√v̂_t + ε)`

When `v̂_t` is large (high variance/uncertainty):
- The denominator grows
- The effective step size shrinks
- **High uncertainty → Lower effective LR**

---

## Evidence from Bayesian Learning

### Kalman Filter Principles

The Kalman filter is the optimal linear estimator under Gaussian assumptions:

> "Values with better (i.e., smaller) estimated uncertainty are 'trusted' more."

The **Kalman gain** K determines how much to update:
- High measurement noise → Low K → Small updates
- High process noise (volatility) → High K → Large updates

Our forward model's prediction error is analogous to measurement noise.
High error = high "measurement uncertainty" → should decrease trust → lower LR.

### Precision-Weighted Prediction Errors

From [Mathys et al. (2011)](https://www.frontiersin.org/articles/10.3389/fnhum.2011.00039):

> "Each prediction error is weighted by its precision (inverse variance)."

High variance prediction errors get **down-weighted**.
This is exactly what stabilization does.

---

## When Pearce-Hall IS Correct

Pearce-Hall (high surprise → high LR) is correct when:

1. **Surprise indicates volatility** (environment actually changed)
2. **Low baseline noise** (prediction errors are meaningful)
3. **Need to adapt quickly** (old knowledge is obsolete)

Examples:
- Reversal learning (reward contingencies flip)
- Non-stationary bandits
- Explicit change-point detection tasks

### The Distinction

| Scenario | Surprise Type | Correct Response |
|----------|---------------|------------------|
| Noisy environment, stable dynamics | Unpredictability | ↓ Stabilize |
| Clean environment, changing rules | Volatility | ↑ Pearce-Hall |
| Mixed | Need to disentangle | Context-dependent |

---

## Implications for Implementation

### Current Implementation (Correct for LunarLander)

```python
# High surprise → Lower LR (stabilization)
if self.invert:  # invert=True for stabilization
    multiplier = 2.0 - alpha  # alpha=1.5 → mult=0.5
```

### Future Improvement: Disentangle Volatility and Noise

A better approach would:
1. Track **prediction error variance over time**
2. High variance of PE = noise → decrease LR
3. Sudden change in mean PE = volatility → increase LR

```python
# Proposed improvement
pe_variance = running_var(prediction_errors)
pe_change = abs(mean(recent_PE) - mean(older_PE))

if pe_change > threshold:  # Volatility detected
    lr_multiplier = 1 + gamma * pe_change  # Increase LR
else:  # Just noise
    lr_multiplier = 1 - gamma * pe_variance  # Decrease LR
```

---

## Key References

### Computational Neuroscience
1. [Soltani & Izquierdo (2019)](https://www.nature.com/articles/s41583-019-0180-y) - "Adaptive learning under expected and unexpected uncertainty"
2. [Gershman (2020)](https://www.biorxiv.org/content/10.1101/2020.10.05.327007v2) - "Unpredictability vs. volatility and the control of learning"
3. [Behrens et al. (2007)](https://www.nature.com/articles/nn1954) - "Learning the value of information in an uncertain world"

### Optimization
4. [Kingma & Ba (2014)](https://arxiv.org/abs/1412.6980) - "Adam: A Method for Stochastic Optimization"
5. [Study of Gradient Variance (2020)](https://arxiv.org/abs/2007.04532) - "A Study of Gradient Variance in Deep Learning"

### Bayesian Learning
6. [Mathys et al. (2011)](https://www.frontiersin.org/articles/10.3389/fnhum.2011.00039) - "A Bayesian foundation for individual learning under uncertainty"
7. Kalman Filter literature

### Original Theories
8. [Pearce & Hall (1980)](https://psycnet.apa.org/record/1981-02676-001) - Classic associability theory
9. [SurNoR (2021)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009070) - Novelty vs. Surprise

---

## Summary

| Theory | Prediction | Our Finding | Explanation |
|--------|------------|-------------|-------------|
| Pearce-Hall | High PE → ↑ LR | ❌ Not optimal | Assumes PE = learning signal |
| Stabilization | High PE → ↓ LR | ✅ Works | PE dominated by noise |
| Adam optimizer | High variance → ↓ LR | ✅ Built-in | SNR-based step scaling |
| Kalman filter | High noise → ↓ gain | ✅ Same principle | Precision weighting |
| Gershman | Volatility↑ LR, Noise↓ LR | ✅ Context-dependent | Need to disentangle |

**Bottom line:** In noisy environments where prediction error is mostly stochastic noise (not meaningful volatility), reducing learning rate is the correct Bayesian response.
