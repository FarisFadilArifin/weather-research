# v11 Settlement Prediction Report — July 12, 2026

## Executive summary

This report documents reconstructed same-day 11 AM predictions from the production-refit **v11 Wunderground settlement model** for KATL and KDAL.

| Station | Reconstructed prediction | Rounded market temperature | 11 AM high so far | Predicted remaining warming |
| --- | ---: | ---: | ---: | ---: |
| KATL | **93.09°F** | **93°F** | 86°F | 7.09°F |
| KDAL | **93.24°F** | **93°F** | 87°F | 6.24°F |

These are reconstructed estimates, not predictions recorded by the live bot on July 12. The historical 11 AM observations and direct NBM forecasts were retrieved successfully. Exact archived GFS and HRRR rows were unavailable within the reconstruction run, so their high-temperature inputs were conservatively set equal to NBM. This limitation materially reduces confidence in the decimal values.

## Model identification

| Property | Value |
| --- | --- |
| Model version | `station_high_regressor_v11_wunderground_settlement_stack` |
| Feature version | `v11` |
| Target source | `settlement_first` |
| Target modeled by base learners | Remaining warming above the observed high so far |
| Base learners | XGBoost, LightGBM, CatBoost |
| Final learner | Ridge regression stack |
| Forecast timing | Same-day 11 AM live-safe |
| Normal providers | GFS, HRRR, NBM |

The exported production base models were refitted on all available usable labels through June 21, 2026. The ridge stack was fitted on out-of-fold predictions from 2024–2025.

## Point-in-time observation inputs

The observation cutoff was the latest METAR inside the model's 10:40–11:00 AM local decision window.

| Input | KATL | KDAL |
| --- | ---: | ---: |
| Observation time, local | 10:52 AM EDT | 10:53 AM CDT |
| Observation age at 11 AM | 8 min | 7 min |
| Temperature | 86°F | 87°F |
| High through observation time | 86°F | 87°F |
| Dew point | 72°F | 72°F |
| Relative humidity | 63.13% | 61.15% |
| Wind | 4.60 mph, variable | 8.06 mph from 240° |
| Three-hour temperature change | +9°F | +5°F |
| Morning warm-up rate | 3.21°F/hour | 1.59°F/hour |
| Cloud cover | 75.0% | 37.5% |
| Sea-level pressure | 1019.8 mb | 1018.9 mb |

## Forecast inputs and reconstruction

The direct NOAA NBM 13Z cycle was selected using the model's live-safe cutoff rule.

| Input | KATL | KDAL |
| --- | ---: | ---: |
| NBM remaining-day high | 86.45°F | 92.03°F |
| NBM temperature at cutoff | 83.57°F | 87.17°F |
| Reconstructed GFS high | 86.45°F | 92.03°F |
| Reconstructed HRRR high | 86.45°F | 92.03°F |

The GFS and HRRR substitutions are not claims about what those models actually forecast. They provide a neutral zero-spread provider configuration that lets the fitted estimators produce a reproducible estimate while avoiding invented provider disagreement. RAP is irrelevant to this v11-only report.

Calendar fields, lagged station history, rolling bias features, climatology, and other non-live fields were initialized from the corresponding July 12 seasonal feature structure. Live observation fields, provider highs, provider aggregates, provider ranks, and forecast-minus-observation differences were replaced or recomputed for July 12, 2026.

## Base-model and stack outputs

### KATL

| Component | Predicted final high |
| --- | ---: |
| XGBoost | 92.54°F |
| LightGBM | 92.74°F |
| CatBoost | 92.77°F |
| Ridge stack | **93.09°F** |

The three base estimates span only 0.23°F. Their agreement is encouraging internally, but all three use the same reconstructed provider inputs and therefore do not constitute three independent confirmations.

KATL's ridge stack uses only the three base-model predictions. It has `alpha = 15.2255`, includes an intercept, and was fitted on 731 out-of-fold meta-training rows from 2024–2025.

### KDAL

| Component | Predicted final high |
| --- | ---: |
| XGBoost | 93.50°F |
| LightGBM | 93.56°F |
| CatBoost | 92.11°F |
| Ridge stack | **93.24°F** |

KDAL's base estimates span 1.45°F, indicating more model disagreement than KATL. Its ridge stack uses only the three base predictions, has `alpha = 35.9786`, no intercept, and was fitted on 730 out-of-fold meta-training rows from 2024–2025.

## Historical model performance

| Station | Evaluation period | Ridge-stack rows | MAE | RMSE |
| --- | --- | ---: | ---: | ---: |
| KATL | 2026 test through June 21 | 172 | 1.29°F | 1.77°F |
| KDAL | 2026 test through June 21 | 171 | 1.34°F | 1.80°F |

These error statistics describe normal model evaluation with complete historical features. They do not measure the additional error caused by substituting NBM for GFS and HRRR in this reconstruction. A rough model-performance reference band would be approximately ±1.8°F RMSE, but it is not a calibrated prediction interval and should be widened for this reconstructed case.

## Training provenance

| Station | Production refit rows | Training dates | Stack meta-training rows | Stack dates |
| --- | ---: | --- | ---: | --- |
| KATL | 1,996 | 2021-01-01 to 2026-06-21 | 731 | 2024-01-01 to 2025-12-31 |
| KDAL | 1,987 | 2021-01-01 to 2026-06-21 | 730 | 2024-01-01 to 2025-12-31 |

The base models use a full production refit. Expanding-year folds are used for hyperparameter selection and out-of-fold stack inputs; the deployed base estimators are not merely individual fold models.

## Interpretation

For KATL, the model estimates about 7.1°F of additional warming after the 86°F 11 AM high, producing 93.1°F. For KDAL, it estimates about 6.2°F above the 87°F high so far, producing 93.2°F. Both map to 93°F after whole-degree rounding.

KATL's prediction is notably above the reconstructed provider high of 86.45°F. This is driven by the observed rapid morning warming and the learned remaining-warmup relationship. KDAL's 93.24°F result is close to NBM's 92.03°F high and reflects a smaller adjustment.

## Limitations and decision status

1. Exact archived GFS and HRRR values were not recovered; NBM was substituted for both.
2. Provider spread was therefore forced to zero, affecting several v11 engineered features.
3. The results were calculated retrospectively and were not frozen at the original bot decision time.
4. Historical MAE and RMSE do not include reconstruction uncertainty.
5. The predictions should be treated as research estimates, not audit-grade evidence of what the live system emitted.

The most defensible reporting precision is therefore **approximately 93°F for both KATL and KDAL**, rather than relying on the hundredths-place outputs.

## Files used

- `data/calibration/station_stacking_v11_settlement/model_weights/`
- `data/calibration/station_stacking_v11_settlement/{STATION}_features.csv`
- `data/calibration/station_stacking_v11_settlement/{STATION}_year_split_scoreboard.csv`
- `data/calibration/sdk_current_obs_jul12/sdk_current_observations_11am.csv`
- `data/calibration/direct_nbm_13z_live_safe_jul12/direct_nbm_0h_cache.csv`
