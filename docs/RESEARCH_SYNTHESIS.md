# Research Synthesis: Why the Current Approach Fails and How to Fix It

**Date:** 2025-12-27
**Status:** Complete research synthesis with actionable recommendations

---

## Executive Summary

After extensive research into curiosity-driven exploration, intrinsic motivation, and learning rate modulation in RL, I've identified **5 critical issues** with the current implementation and **3 promising paths forward**.

**Key Finding:** The core hypothesis (entropy-based salience → LR modulation) has theoretical support from neuroscience (Pearce-Hall, SurNoR), but the current implementation has fundamental architectural flaws that prevent it from working.

---

## Critical Issues Identified

### Issue 1: LR Modulation Timing Bug (CRITICAL)

**Location:** `src/agents/temporal_ppo.py:66-69`

```python
# In _on_step() callback during rollout collection
self.salience_lr.update_optimizer_lr(
    self.model.policy.optimizer,
    clock_output.salience_score,
)
```

**Problem:** PPO collects experiences for `n_steps=2048` steps BEFORE doing ANY gradient updates. The LR changes during experience collection have **no effect** because:
1. During collection, no gradients are computed
2. Only the LR at the start of the PPO update phase matters
3. Per-step LR changes during rollout are overwritten before the update

**Evidence:** From FINDINGS.md - "LR modulation happens during rollout but PPO updates happen after."

**Impact:** The entire LR modulation mechanism is essentially no-op.

### Issue 2: Wrong Signal Type

**Current approach:** Uses covariance entropy of state window (how diverse are recent states?)

**What works in literature:**
- ICM/RND: Prediction error (how unexpected is this specific transition?)
- Count-based: Inverse visit frequency (how rarely have we seen this state?)
- TD-error: Value prediction error (how wrong was our estimate?)

**Problem:** State-window entropy measures "variety of recent experience" not "novelty of current moment." A sequence of diverse but familiar states scores high; a single surprising state in routine context scores low.

### Issue 3: Salience Signal Loses Direction

**Current implementation:**
```python
salience_score = abs(delta_entropy) / baseline_delta  # Always positive
```

**Problem:** Both entropy increases AND decreases are treated as equally salient. But they have different meanings:
- Entropy increase → entering novel/uncertain territory
- Entropy decrease → entering familiar/predictable territory

The absolute value conflates these distinct signals.

### Issue 4: Per-Step Modulation Too Noisy

From Pearce-Hall and SurNoR research, learning rate modulation should be **smoothed over time**:

**Pearce-Hall update:**
```
α_t = γ × |PE_t| + (1-γ) × α_{t-1}
```

This exponentially smooths the learning rate over trials. The current implementation applies raw, per-step salience directly.

### Issue 5: Deterministic Seeding Masks Effects

**From FINDINGS.md:** "All gamma sweep experiments show IDENTICAL results to baseline (151.27 ± 51.77)."

PPO with deterministic seeding produces identical action trajectories regardless of LR, because:
1. Same seed → same initial policy
2. Same policy → same action sampling
3. PPO clipping bounds policy changes

---

## What Works in Literature

### Approach 1: Intrinsic Reward Bonus (ICM, RND, Count-based)

**Pattern:**
```
total_reward = extrinsic_reward + β × intrinsic_reward
```

**How it works:**
- Novel states → higher intrinsic reward → agent seeks novelty
- Does NOT modulate learning rate
- Well-established, many successful implementations

**Implementations:**
- [RLeXplore](https://github.com/RLE-Foundation/RLeXplore) - 8 algorithms, SB3 compatible
- [CleanRL PPO-RND](https://docs.cleanrl.dev/rl-algorithms/ppo-rnd/) - Single-file reference

### Approach 2: Prioritized Replay (PER, NSPER)

**Pattern:**
```
sampling_priority ∝ |TD_error|^α  # or surprise score
```

**How it works:**
- Surprising transitions sampled more often
- More learning from surprising experiences
- Requires importance sampling correction

**Implementations:**
- [NSPER](https://github.com/UoA-CARES/NSPER) - Novelty/Surprise prioritization for TD3
- [RL-Adventure](https://github.com/higgsfield/RL-Adventure) - PER with DQN

### Approach 3: LR Modulation (SurNoR, Pearce-Hall)

**This is what you're trying to do!** Key papers:

1. **SurNoR (2021):** [Novelty is not surprise: Human exploratory and adaptive behavior](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009070)
   - Separates novelty (exploration) from surprise (learning rate)
   - Surprise = world-model prediction error
   - LR modulation based on surprise

2. **Pearce-Hall (1980):** Classic theory
   - Associability α updated by absolute prediction error
   - `α = γ|PE| + (1-γ)α_{prev}`
   - Smoothed, not per-step

---

## Three Paths Forward

### Path A: Fix the LR Modulation Approach (Hard, Novel)

**What to change:**

1. **Fix timing:** Modulate LR at PPO update time, not during rollout
```python
# In training loop, BEFORE optimizer.step():
avg_salience = np.mean(rollout_saliences)  # Average over rollout
lr = base_lr * salience_modulator(avg_salience)
for param_group in optimizer.param_groups:
    param_group['lr'] = lr
```

2. **Use prediction error, not state entropy:**
```python
# Train a simple next-state predictor
predicted_next = predictor(state, action)
surprise = ||actual_next - predicted_next||²
```

3. **Smooth the signal (Pearce-Hall style):**
```python
α_t = γ × surprise_t + (1-γ) × α_{t-1}
lr = base_lr × α_t
```

4. **Run without fixed seeds** or use 20+ seeds

### Path B: Switch to Intrinsic Reward (Easier, Proven)

**Replace LR modulation with reward bonus:**

1. Implement RND or ICM as intrinsic reward
2. Use your entropy clock to compute intrinsic reward instead of LR
3. Add `intrinsic_reward = salience_score × scale` to environment reward

**Advantage:** Well-established, many reference implementations
**Disadvantage:** Not testing the original hypothesis

### Path C: Hybrid Approach (SurNoR-inspired)

**Implement SurNoR's separation:**

1. **Novelty → Exploration bonus** (intrinsic reward)
   - Use count-based or RND-style novelty
   - Drives agent to novel states

2. **Surprise → LR modulation**
   - Use prediction error from world model
   - Modulates how much agent learns from each update
   - Implements Pearce-Hall associability

**This is the most theoretically grounded approach** and matches your original vision.

---

## Recommended Implementation Order

### Phase 1: Fix Critical Bugs (Week 1)

1. Move LR modulation to update phase
2. Average salience over rollout instead of per-step
3. Add Pearce-Hall smoothing
4. Run experiments without fixed seeds

### Phase 2: Improve Signal (Week 2)

1. Replace state-entropy with prediction error
   - Add small forward model (MLP predicting next state)
   - Use prediction error as surprise signal

2. Test both directions:
   - High surprise → higher LR (Pearce-Hall)
   - High surprise → lower LR (stabilization)

### Phase 3: Full SurNoR Implementation (Week 3-4)

1. Separate novelty from surprise
2. Add episodic count-based novelty
3. Add world model for surprise
4. Combine: novelty→reward, surprise→LR

---

## Key Papers to Read

### Must Read
1. **SurNoR:** [Novelty is not surprise](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009070) - Your theoretical foundation
2. **RND:** [Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894) - Best intrinsic motivation implementation
3. **Pearce-Hall Review:** [Prediction errors, attention and associative learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC4862921/)

### Useful References
4. **ICM:** [Curiosity-driven Exploration by Self-supervised Prediction](https://arxiv.org/abs/1705.05363)
5. **Temporal encoding in RL:** [Time cells emerge in deep RL](https://www.nature.com/articles/s41598-023-49847-y)
6. **Dopamine and TD:** [Gradual temporal shift of dopamine responses](https://www.nature.com/articles/s41593-022-01109-2)

---

## Relevant Repositories

| Purpose | Repository | Notes |
|---------|-----------|-------|
| Intrinsic rewards (SB3 compatible) | [RLeXplore](https://github.com/RLE-Foundation/RLeXplore) | 8 algorithms |
| RND reference | [CleanRL PPO-RND](https://docs.cleanrl.dev/rl-algorithms/ppo-rnd/) | Single file |
| ICM reference | [pathak22/noreward-rl](https://github.com/pathak22/noreward-rl) | Original |
| Surprise intrinsic motivation | [jachiam/surprise](https://github.com/jachiam/surprise) | Josh Achiam |
| Novelty/Surprise replay | [NSPER](https://github.com/UoA-CARES/NSPER) | TD3 based |
| Successor representations | [SuccessorOptions](https://github.com/rahul13ramesh/SuccessorOptions) | Temporal abstraction |

---

## Direction Debate: Which Way to Modulate?

**Open question:** Should high surprise INCREASE or DECREASE learning rate?

### Case for INCREASING LR (Pearce-Hall)
- Surprising events contain new information
- Should update weights more to incorporate
- Matches dopamine response patterns

### Case for DECREASING LR (Stabilization)
- During chaos, gradients may be unreliable
- Consolidate existing knowledge before updating
- Matches some uncertainty-weighting literature

### Resolution
Test both! The literature is genuinely split. Your experiment should include:
1. `invert_direction=False` (high surprise → high LR)
2. `invert_direction=True` (high surprise → low LR)

The Pearce-Hall direction has more theoretical support, but both are worth testing.

---

## Summary

| Problem | Solution |
|---------|----------|
| LR modulation during rollout | Modulate at update time |
| State entropy ≠ novelty | Use prediction error |
| Per-step too noisy | Smooth with Pearce-Hall EMA |
| Deterministic seeding | Use 20+ seeds or no seeds |
| Direction unclear | Test both, theory favors Pearce-Hall |

The core hypothesis is sound and has support from SurNoR, Pearce-Hall, and dopamine research. The implementation needs architectural fixes to actually test it.
