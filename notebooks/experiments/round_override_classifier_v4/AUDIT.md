# V4 Audit: Cost-Sensitive Half-Up Override

## Decision

**V4 is safe but does not yet add forecast value.** The stability gate disabled overrides for both
stations, so honest forward bucket accuracy exactly matches nearest half-up rounding. Do not enable
overrides until the recovery and damage heads show stable out-of-time separation.

## Honest 2024-2025 result

| Station | Half-up hit rate | V4 hit rate | Overrides | Recovered | Damaged | Net |
|---|---:|---:|---:|---:|---:|---:|
| KATL V20 | 49.93% | 49.93% | 0 | 0 | 0 | 0 |
| KDAL V20 no peak | 48.63% | 48.63% | 0 | 0 | 0 | 0 |

V4 also made zero overrides in each individual outer year and in exploratory sequential 2026.
This is the intended fallback when no policy satisfies all earlier stability requirements.

## Why the policy abstained

The two probability heads did not separate the actionable cases reliably out of time. Combined
2024-2025 ROC AUC was approximately 0.492 recovery / 0.510 damage for KATL and 0.520 recovery /
0.542 damage for KDAL. Those values are too close to random ranking for a high-cost override.

The earlier V3 system worsened bucket results because it acted on unstable ranking. V4 prevents
that failure by requiring every enabled policy to:

- use a damage penalty of at least 2;
- have positive `recoveries - 2 * damages` on prior policy data;
- have positive raw net wins in at least two of three chronological policy folds;
- have no harmful policy fold; and
- otherwise default completely to half-up.

## Model and feature contract

- Separate calibrated binary recovery and damage models.
- Models fit only actionable rows; non-actionable rows are automatically kept half-up.
- Compact inventories: 36 features for KATL and 28 for KDAL.
- Continuous bucket probabilities use 180-day and 365-day residual windows with `shift(1)`.
- KATL includes selected peak-timing context; KDAL excludes every peak feature.
- Logistic and shallow CatBoost hyperparameters are selected on an earlier inner window.
- Calibration and policy selection use separate 90-day windows.
- Recovery thresholds span the full 0.05-0.95 range.

## Integrity audit

All **50/50** executable checks passed. They cover exact target formulas, mutual exclusivity,
actionable-only fitting, prior-only rolling features, valid continuous probabilities, compact
feature counts, all-row scoring, exact utility reconstruction, disabled-policy abstention,
chronology, 90-day windows, full threshold coverage, three-fold policy stability, selection
eligibility for both outer and final exported policies, win accounting, forward point predictions,
unique dates, integer labels, and the KDAL no-peak contract.

## Acceptance conclusion

V4 fixes V3's unsafe behavior, but zero overrides means it has not demonstrated incremental skill.
Keep nearest half-up as production behavior. Treat the V4 heads as research diagnostics until new
features or more history produce positive net wins in each honest outer year without bypassing the
stability gate.
