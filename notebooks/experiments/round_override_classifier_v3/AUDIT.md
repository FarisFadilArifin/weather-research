# V3 Audit: Half-Up Override Classifier

## Decision

**Do not deploy V3 in its current form.** The target and chronological evaluation are correct,
but the learned override policy reduced bucket wins on both stations during honest 2024-2025
validation.

## Exact decision being modeled

The regression prediction is rounded to the nearest integer using half-up rounding by default.
The classifier estimates whether changing that default to the opposite floor/ceil integer would
turn a loss into a win:

- `override_target = 1` only when the alternative bucket wins and half-up loses.
- `override_target = 0` otherwise, including rows where floor and ceil settle to the same bucket.
- The final forecast switches only when the row is actionable and the calibrated probability is
  at least the threshold chosen entirely from earlier data.

No boundary-distance window is used. Every outer-validation row is scored.

## Honest forward results (2024-2025)

| Station | Half-up hit rate | V3 hit rate | Lift | Recovered | Damaged | Net wins |
|---|---:|---:|---:|---:|---:|---:|
| KATL V20 | 49.93% | 44.58% | -5.35 pp | 45 | 84 | -39 |
| KDAL V20 no peak | 48.63% | 46.15% | -2.47 pp | 43 | 61 | -18 |

The probabilities have useful ranking signal (ROC AUC 0.760 for KATL and 0.810 for KDAL), but
ranking is not enough: the chosen thresholds created more damaged wins than recovered losses.
KATL was also badly miscalibrated in 2024, with mean predicted probability 0.450 against an
observed override rate of 0.137.

| Station | V3 log loss | Continuous Student-t log loss |
|---|---:|---:|
| KATL V20 | 0.60828 | 0.31756 |
| KDAL V20 no peak | 0.33491 | 0.30578 |

The continuous-residual baseline is a comparison only; it is not leaked into V3 features.

## Exploratory 2026 result

This is not used for model selection. KATL made no overrides (net 0). KDAL made 28 overrides,
recovering 8 losses and damaging 11 wins (net -3).

## Integrity audit

All **28/28** programmatic checks passed, including:

- exact half-up default and opposite floor/ceil alternative;
- exact binary target reconstruction;
- all rows scored and no boundary-distance filtering;
- no overrides on non-actionable rows;
- fit < calibration < policy selection < outer validation chronology;
- forward-only point predictions and unique station/date keys;
- KDAL no-peak contract; and
- win-accounting identity (`final - default = recovered - damaged`).

The implementation is therefore internally consistent, but its honest outcome rejects the current
policy. The next version should optimize a more conservative, cost-sensitive override rule and
require positive net wins across multiple prior folds before enabling any override.
