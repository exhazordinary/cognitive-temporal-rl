# Legacy: entropy-clock approach (deprecated)

This folder preserves the original **entropy-based salience clock** approach,
which was superseded by the SurNoR-inspired prediction-error approach in
`src/`. It is kept for reference because the negative result is part of the
research story — see `docs/FINDINGS.md` and `docs/RESEARCH_SYNTHESIS.md` for
why it failed.

**Do not build on this code.** It is unmaintained, its imports are not
guaranteed to work after the repo restructure, and `temporal_ppo.py` contains
a known bug (per-step LR modulation during rollout has no effect on PPO's
gradient updates).

Contents:

- `entropy_clock/` — covariance-entropy "internal clock" over rolling state windows
- `temporal_ppo.py` — PPO wrapper using the entropy clock (buggy LR timing)
- `modulators/` — per-step salience LR / exploration / replay modulators
- `experiments/` — original experiment runner and configs
- `tests/` — tests for the entropy clock (excluded from pytest collection)
- `scripts/watch.py` — visualization script tied to the entropy clock
