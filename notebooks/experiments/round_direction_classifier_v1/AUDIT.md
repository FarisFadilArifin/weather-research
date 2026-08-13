# Binary Floor-or-Ceil Classifier Audit

The executed notebook was audited against the actual KATL V20 peak-timing and KDAL V20 no-peak
artifacts. All 28 executable checks passed.

## Confirmed implementation

- `floor_degree_f = floor(point_prediction_f)`.
- `ceil_degree_f = ceil(point_prediction_f)`.
- The binary target is `round_up = int(actual_high_f > point_prediction_f)`.
- Every 2024/2025 outer row receives a finite probability and exactly one class.
- The class threshold is fixed: probability `>= 0.5` selects ceil; otherwise floor.
- Corrected degrees are always exactly floor or ceil; there is no third action or abstention.
- There are no boundary-distance windows in the model feature inventory.
- Original point degrees are independently verified with half-up rounding.
- Point predictions are forward Ridge-stack predictions.
- Hyperparameter fitting, Platt calibration, and outer validation are strictly chronological.
- KDAL has zero overlap with KATL's peak-timing feature inventory.
- Corrected-minus-original bucket wins equals recovered-minus-damaged wins.

## Honest 2024–2025 result

| Station | Half-up hit rate | Binary-corrected hit rate | Recovered | Damaged | Net wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| KATL | 49.93% | 44.58% | 37 | 76 | -39 |
| KDAL | 48.63% | 46.02% | 52 | 71 | -19 |

The binary classifier underperformed ordinary half-up rounding for both stations. Direction
accuracy was `49.93%` for KATL and `51.92%` for KDAL, versus `60.63%` and `56.87%` respectively for
the existing deterministic half-up rule.

The classifier also underperformed the Student-t continuous-residual directional probability:

| Station | Binary classifier log loss | Continuous baseline log loss |
| --- | ---: | ---: |
| KATL | 0.78667 | 0.74389 |
| KDAL | 0.71682 | 0.70721 |

## Exploratory 2026

KATL gained two net bucket wins and KDAL gained one, but these small exploratory gains cannot
override losses of 39 and 19 wins on honest 2024–2025 data. The binary model should not be deployed.

## Conclusion

The corrected notebook now implements the requested pure binary floor/ceil task exactly. The audit
finds no label, rounding, chronology, station-feature, or accounting defect. The empirical result is
negative: the regression's existing half-up rounding is substantially better than replacing it with
this learned binary direction classifier.
