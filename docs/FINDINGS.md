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
