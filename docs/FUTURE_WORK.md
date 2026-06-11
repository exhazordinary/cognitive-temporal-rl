# Future Work

Ideas for continuing this research, prioritized by potential impact.

---

## High Priority

### 0. Re-run all experiments post-bugfix (June 2026)

Three bugs were found and fixed (see `docs/FINDINGS.md`, "June 2026"); the
headline +14.8% stabilization result predates them and is unvalidated.

```bash
uv run python -m src.experiments.run_surnor \
  --experiments baseline surnor_pearce_hall surnor_stab_gamma_0.1 surnor_stabilize surnor_adaptive
uv run python scripts/analyze_results.py --results results/surnor_LunarLander-v3_*.json
```

---

### 1. ~~Test the VolatilityDetector~~ DONE (wired in, needs experiments)

The detector is now integrated as `lr_mode="adaptive"` with two configs
(`surnor_adaptive`, `surnor_adaptive_sensitive`) and per-update
volatility/noise/change-point logging in the results JSON.

```bash
uv run python -m src.experiments.run_surnor --experiments baseline surnor_adaptive surnor_adaptive_sensitive
```

**Open tuning question:** with default settings the detector flags many
change points early in training while the forward model is still learning
(its PEs drift, which looks like volatility). Consider a warmup period or a
higher `vol_change_threshold` if the adaptive arm over-boosts early.

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

### 4. ~~Statistical Significance Testing~~ DONE

`src/analysis/stats.py` + `scripts/analyze_results.py` now report bootstrap
CIs, Welch's t-test, IQM, and Hedges' g per environment.

---

### 4b. Faster vectorization backend (PufferLib)

Training now uses SB3 vectorized envs (DummyVecEnv default, `--vec-env
subproc` available). The SurNoR callback's only contracts with the env layer
are SB3's `locals` keys and the `terminal_observation` info convention, so a
faster backend like PufferLib's vectorization could later be swapped in
behind a thin SB3 `VecEnv` adapter without touching the surprise pipeline.
A full port to PufferLib's own PPO trainer was considered and deliberately
deferred: it would require reimplementing the LR modulation inside their
trainer and break comparability with SB3 runs.

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
| Fix LR modulation no-op + transition bugs | Done (June 2026) | `docs/FINDINGS.md`, regression tests |
| Vectorize training (8 envs default) | Done (June 2026) | `src/agents/surnor_ppo.py` |
| Wire VolatilityDetector into experiments | Done (June 2026) | `lr_mode="adaptive"` |
| Statistical analysis (CIs, Welch, IQM) | Done (June 2026) | `src/analysis/stats.py` |
| Multi-env configs (CartPole/Acrobot/MountainCar) | Done (June 2026) | `--env` flag |

---

## Key References for Future Work

- [Gershman (2020)](https://www.biorxiv.org/content/10.1101/2020.10.05.327007v2) - Volatility vs unpredictability
- [SurNoR (2021)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009070) - Full novelty + surprise framework
- [Meta-World](https://meta-world.github.io/) - Multi-task benchmark
- [ProcGen](https://openai.com/research/procgen-benchmark) - Generalization benchmark
