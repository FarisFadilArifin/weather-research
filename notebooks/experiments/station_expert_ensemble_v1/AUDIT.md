# Station Expert Ensemble V1 Audit

Status: implementation and static validation complete; full research training is pending.

## Contract checklist

- [x] Experiment-local configs and generator for KDAL, RKSI, and RJTT.
- [x] Four expert roles, target transforms, leakage routing, fold-owned 3% missingness, and physical floor.
- [x] Generic N-model simplex utilities with legacy two-/three-model compatibility.
- [x] Forward-only blend weights and frozen pre-2026 selection.
- [x] Research-only point bundle with expert contracts, audits, chronology, source identity, and verified SHA-256.
- [x] New common and Asia 61-feature probability profiles with declared base methods.
- [x] Native whole-Celsius probability path for Seoul and Tokyo.
- [x] KDAL 21/29/61 challenger feature sets and three frozen roles with no point-bucket override.
- [ ] KDAL full 30-trial clean-kernel research execution.
- [ ] Seoul full 30-trial clean-kernel research execution.
- [ ] Tokyo full 30-trial clean-kernel research execution.

## Validation log

Validation performed on 2026-08-11:

- Regenerated all three notebooks from the experiment-local generator.
- Parsed all three notebooks as notebook-format 4 JSON; each contains 23 cells.
- Parsed every ordinary Python code cell through `ast.parse` via the generator tests.
- Confirmed station identity, provider isolation, chronology/stage order, 30/15 Optuna settings, artifact paths, disabled live refits, bucket contracts, KDAL 21/29/61 arms, and JSON validity.
- Ran a real KDAL feature-snapshot preflight for `full_xgboost` through fit and prediction (one Optuna trial).
- Ran fit-and-prediction preflights for `forecast_huber`, `seasonal_ridge`, and `observation_catboost` on a controlled chronological frame.
- Ran the required and relevant combined suite: **57 passed**, with eight existing LightGBM feature-name warnings.
- Verified no stale experimental or absolute paths in the new source/notebooks/tests.
- Verified the README link target and ran `git diff --check` successfully; only existing Windows LF/CRLF notices were emitted.

The exact top-to-bottom notebook executions were not represented as completed. The measured one-trial real-data XGBoost preflight took about 99 seconds, while each notebook requires repeated outer/final fits with 30 trials for both nonlinear experts plus probability/challenger training. Completing all three is therefore a multi-hour research run. No partial or reduced-trial output is recorded as final evidence.

No result in this audit is a promotion decision. The 2026 output is exploratory; fresh shadow data remains mandatory before any promotion review.
