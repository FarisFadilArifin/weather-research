# Directional Residual Audit V1

## Conclusion

The regression residual direction contains repeatable patterns, but those patterns do **not**
identify profitable bucket changes. Across KATL and KDAL, **zero alternative-bucket utility
subgroups passed both 2023-2025 development stability and untouched 2026 confirmation**.

This supports keeping nearest half-up rounding. It also explains why the correction classifiers
failed: continuous residual sign is not the same decision as whether the opposite 2-degree bucket
wins.

## Signal counts

| Station | Stable residual signals in 2023-2025 | Confirmed residual signals in 2026 | Stable utility signals in 2023-2025 | Confirmed utility signals in 2026 |
|---|---:|---:|---:|---:|
| KATL | 5 | 4 | 2 | **0** |
| KDAL | 6 | 4 | 0 | **0** |

## Confirmed continuous residual-direction patterns

`P(actual > prediction)` below 0.5 means the point prediction tends to be above the integer actual;
above 0.5 means it tends to be below it.

| Station | Diagnostic subgroup | 2023-2025 rate | 2026 rate | Direction |
|---|---|---:|---:|---|
| KATL | Prediction fraction `[0, 0.25]` | 29.1% | 27.9% | Actual below point |
| KATL | Base ensemble mean more than 0.25°F below point | 32.5% | 33.3% | Actual below point |
| KATL | Half-up direction is down | 37.6% | 30.0% | Actual below point |
| KATL | Fraction `[0, 0.25]` and base direction neutral | 28.8% | 28.6% | Actual below point |
| KDAL | Fraction `(0.75, 1)` and base mean above point | 71.0% | 72.5% | Actual above point |
| KDAL | Prediction fraction `(0.75, 1)` | 65.8% | 70.8% | Actual above point |
| KDAL | Fraction `(0.50, 0.75]` and base mean above point | 63.3% | 54.1% | Actual above point |
| KDAL | Winter and base mean above point | 62.2% | 56.1% | Actual above point |

Most fraction effects are partly mechanical: actual settlement labels are integers. When a point
prediction is just above an integer, matching that integer creates a negative residual; when it is
just below the next integer, matching that integer creates a positive residual. This predicts the
continuous residual sign without necessarily changing the winning 2-degree bucket.

## Direct bucket-utility result

Only two KATL utility patterns passed the 2023-2025 development criteria, and both favored keeping
half-up:

- When the model-vote balance favored the default bucket by at least two, the alternative recovered
  only 34.5% of decisive cases. In 2026 the share became 50.0%, so it did not confirm.
- When the rolling continuous model favored the default bucket by more than 0.10, the alternative
  recovered 35.9% of decisive cases. In 2026 it rose to 54.8%, reversing direction.

KDAL had no development-stable utility subgroup. Therefore no subgroup provides an audited basis
for overriding half-up.

## Statistical and leakage controls

- Fixed subgroup thresholds were defined before testing outcomes.
- All rows were retained; fractional groups are diagnostics, not eligibility windows.
- Development selection used 2023, 2024, and 2025 only.
- Each development year had to meet a minimum sample count and agree in direction.
- A minimum per-year directional edge was required.
- Binomial tests were corrected across searched subgroups with Benjamini-Hochberg FDR.
- 2026 was used only for confirmation with its own minimum sample requirement.
- Rolling residual features use `shift(1)` and point predictions are forward-only.

All **26/26** executable integrity checks passed.

## Recommendation

Do not build a new rounding-direction classifier from these subgroups. The surviving signal mostly
describes integer residual geometry, not profitable bucket switching. Keep half-up and focus future
research on a properly calibrated full bucket-probability distribution with more historical honest
out-of-fold data.
