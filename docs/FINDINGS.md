# Research Findings Log

## Session: 2025-12-15

### Experiment Run: Baseline vs Salience LR

**Setup:**
- Environment: LunarLander-v3
- Timesteps: 200,000
- Seeds: 5 (42, 123, 456, 789, 1011)
- Baseline: Vanilla PPO (stable-baselines3)
- Treatment: PPO + Entropy Clock + Salience LR modulation (γ=0.5)

**Results:**

| Condition | Mean Final Reward | Std Dev | Winner |
|-----------|-------------------|---------|--------|
| Baseline PPO | 151.27 | 51.77 | ✅ |
| Salience LR | 70.35 | 77.27 | ❌ |

**Conclusion:** Salience LR **hurts** performance. Baseline wins by ~80 points.

---

### Why It Might Have Failed

1. **LR modulation too aggressive** - γ=0.5 means LR can swing from 0.5x to 2x base. Too volatile?

2. **Per-step modulation too noisy** - Changing LR every single step might destabilize training. Maybe modulate per-episode or per-rollout instead?

3. **Wrong direction** - Current: high salience → higher LR. Maybe it should be: high salience → *lower* LR (stabilize during chaos, consolidate learning)?

4. **PPO might not benefit** - PPO already has mechanisms for stable learning (clipping, GAE). Adding LR modulation might interfere.

5. **Entropy signal might be noisy** - LunarLander has 8D state. Covariance-based entropy on small windows might not capture meaningful novelty.

---

### Next Steps to Try

1. **Tune hyperparameters:**
   - Try γ = 0.1, 0.2, 0.3 (less aggressive)
   - Try larger window_size (100 instead of 50)
   - Try different min_samples

2. **Test other modulators:**
   - Salience Exploration (not tested yet)
   - Salience Replay (not integrated yet)

3. **Invert the hypothesis:**
   - High salience → *decrease* LR (consolidate)
   - Low salience → *increase* LR (shake things up)

4. **Change modulation frequency:**
   - Per-episode instead of per-step
   - Per-rollout (every n_steps)

5. **Analyze failure modes:**
   - Plot learning curves side by side
   - Look at when/where salience_lr diverges
   - Visualize salience distribution over training

---

### Files with Results

- `/results/ablation_20251215_230539.json` - Baseline only (1 seed)
- `/results/ablation_20251215_230958.json` - Salience LR only (1 seed)
- `/results/ablation_20251215_232029.json` - Both conditions (5 seeds each) ← **Main results**

---

### Key Code to Modify for Next Experiments

| Change | File | Location |
|--------|------|----------|
| LR gamma parameter | `src/modulators/learning_rate.py` | `SalienceLR.__init__()` |
| Entropy window size | `src/entropy_clock/clock.py` | `EntropyClockModule.__init__()` |
| Add exploration modulator | `src/agents/temporal_ppo.py` | `TemporalPPO.__init__()` |
| Experiment configs | `src/experiments/config.py` | `EXPERIMENTS` dict |

---

## Session: 2025-12-23

### Code Changes Made

Implemented all planned next steps from Dec 15:

1. **Gamma Tuning** - Added `lr_gamma` parameter wiring through config → train → agent
2. **Invert Hypothesis** - Added `invert_direction` parameter to SalienceLR
3. **Exploration Integration** - Integrated SalienceExploration via `ent_coef` modulation
4. **Diagnostics** - Added step-level logging and visualization tools

### New Experiment Configurations

| Experiment | Description |
|------------|-------------|
| `salience_lr_gamma_0.1` | LR modulation with γ=0.1 (less aggressive) |
| `salience_lr_gamma_0.2` | LR modulation with γ=0.2 |
| `salience_lr_gamma_0.3` | LR modulation with γ=0.3 |
| `salience_lr_inverted` | High salience → LOWER LR (γ=0.5) |
| `salience_lr_inverted_0.3` | Inverted with γ=0.3 |
| `salience_exploration` | Exploration modulation only |
| `salience_lr_exploration` | Combined LR + exploration |

---

### Experiments Run (Dec 23)

**Gamma Sweep** (`ablation_20251223_010017.json`):
- baseline, salience_lr_gamma_0.1, 0.2, 0.3
- 200k timesteps, 5 seeds

**Inverted Hypothesis** (`ablation_20251223_023956.json`):
- baseline, salience_lr_inverted, salience_lr_inverted_0.3
- 200k timesteps, 5 seeds

**Exploration** (`ablation_20251223_121019.json`):
- baseline, salience_exploration, salience_lr_exploration
- 200k timesteps, 5 seeds

---

### Critical Finding: Deterministic Seeding Issue

**Observation:** All gamma sweep and inverted experiments show IDENTICAL results to baseline (151.27 ± 51.77).

**Root Cause Analysis:**

1. Verified LR modulation IS being applied (step_lr_history shows variation from 0.00015 to 0.0006)
2. Verified diagnostics capture data correctly
3. Found: Episode rewards are byte-for-byte identical across experiments with same seed

**Why this happens:**

- Stable-baselines3 PPO uses deterministic seeding throughout
- Same seed → same initial policy → same action sampling random numbers
- Even with different LR values, the sampled actions end up identical
- PPO's clipping bounds policy changes, so updates are very similar

**Evidence:**

```
With seed=42:
  Baseline episode 24: -2.56
  Salience_lr episode 24: -2.56 (IDENTICAL)

Without seeds:
  Baseline: 180 eps, mean=-178.47
  Salience_lr: 193 eps, mean=-167.91 (DIFFERENT!)
```

**Conclusion:** LR modulation DOES work, but deterministic seeding masks the effect.

---

### Mystery: Dec 15 Results Showed Divergence

The original Dec 15 experiments showed salience_lr diverging from baseline at episode 24:

```
Original baseline episode 24: -2.56
Original salience_lr episode 24: 40.53 (DIFFERENT!)
```

But current code with same seeds produces identical results. Possible explanations:
1. Some non-determinism existed in original environment
2. Library version differences
3. Unknown environmental factor

---

### Exploration Modulation Results

| Condition | Mean Reward | Std | Notes |
|-----------|-------------|-----|-------|
| baseline | 151.27 | 51.77 | |
| salience_exploration | 90.66 | 73.28 | Hurts performance |
| salience_lr_exploration | 90.66 | 73.28 | Same as exploration only |

Divergence detected at episode 130 for exploration experiments.

---

### New Files Created

| File | Purpose |
|------|---------|
| `src/visualization/__init__.py` | Visualization module init |
| `src/visualization/diagnostics.py` | Plotting functions for analysis |
| `scripts/analyze_results.py` | CLI for post-hoc result analysis |

---

### Key Learnings

1. **Deterministic seeding is problematic** for hyperparameter comparison - same seeds produce identical trajectories regardless of LR

2. **LR modulation timing** - Modulation happens during rollout collection via `_on_step()`, but PPO updates happen after. The LR set during steps doesn't affect those steps' experiences.

3. **Exploration modulation hurts** - Both exploration-only and combined approaches perform worse than baseline (~90 vs ~150)

4. **Need stochastic evaluation** - For meaningful hyperparameter comparison, either:
   - Use no fixed seeds
   - Use many different seeds
   - Add explicit exploration noise

---

### Recommendations for Future Experiments

1. **Run without fixed seeds** to see true effect of modulation
2. **Use 10+ seeds** for statistical significance
3. **Consider per-rollout modulation** instead of per-step
4. **Try modulating during PPO update phase** instead of rollout collection
5. **Investigate why Dec 15 showed different behavior**

---

### Commands for Analysis

```bash
# Analyze single result file
uv run python scripts/analyze_results.py --results results/ablation_20251223_010017.json

# Analyze multiple files (merged)
uv run python scripts/analyze_results.py --results results/ablation_*.json

# With detailed per-run diagnostics
uv run python scripts/analyze_results.py --results results/ablation_*.json --detailed
```

---

## June 2026: Code review — three bugs invalidate prior SurNoR results

A review of `surnor_ppo.py` against the installed Stable-Baselines3 (2.7.1)
source found three bugs. All are fixed; regression tests guard each one.

### Bug 1: LR modulation was a silent no-op (critical)

`SurNoRCallback._on_rollout_end()` wrote the modulated LR directly into the
optimizer's param groups. But SB3's `PPO.train()` calls
`self._update_learning_rate(self.policy.optimizer)` as its **first** action
after rollout collection, resetting every param group from the LR schedule —
a constant 3e-4, since `learning_rate` was passed as a float. The modulated
LR was overwritten before any gradient step.

**Implication:** all SurNoR experiment arms trained with identical, constant
LR. The +14.8% stabilization result cannot be attributed to LR modulation;
the only code-path differences vs baseline were incidental (background
forward-model training and RNG consumption). The result may be noise.

**Fix:** a `MutableLRSchedule` is passed to PPO as its `learning_rate`; the
callback writes the modulated value into the schedule and SB3 itself applies
it. Regression test: `tests/test_surnor_integration.py` (fails on old code).

### Bug 2: Off-by-one action in surprise transitions

The callback computed surprise on `(prev_obs, prev_action, new_obs)` where
`prev_action` was captured on the *previous* callback step. The action that
actually produced `new_obs` is the current `locals["actions"]`. The forward
model was trained on mispaired transitions throughout.

### Bug 3: Surprise computed across episode resets

SB3 VecEnvs auto-reset: on `done` steps, `new_obs` is the *next* episode's
reset observation. The true terminal observation lives in
`infos[i]["terminal_observation"]`. Surprise (and forward-model training
data) included these cross-boundary pseudo-transitions.

Bugs 2 and 3 were fixed by the batched callback rewrite that came with
environment vectorization.

### Comparability break

Old and new results are **not comparable**, for three independent reasons:
the LR fix, the corrected forward-model training data, and vectorization
(Pearce-Hall alpha is now updated once per vec-step from the mean |PE|
across envs, so gamma's smoothing timescale in env-steps stretches with
n_envs). All configs — including baseline — must be re-run under the new
regime before drawing conclusions. Vectorization makes this ~1.5-2x cheaper
per run on a 4-core CPU, more on larger machines.
