# Polymarket 11 AM ML v7 Handoff

> Superseded for new bot handoff by `POLYMARKET_11AM_ML_V11_HANDOFF.md`.
> Keep this file only as historical context for the v7 experiment.

This is the current handoff for the updated station-stacking v7 model. It supersedes the older v2 deployment prompt for model and feature direction.

## Current Recommendation

Use station-stacking v7 as the current live-safe research contract, and use `ridge_stack` as the default prediction method.

Primary backtest column:

```text
data/calibration/backtests/station_stacking_v7_2026_oof_wide.csv
column: ridge_stack_predicted_high_f
```

Do not pick each station's best 2026 method in production. That is hindsight leakage. In the current v7 backtest, using `ridge_stack` everywhere beat choosing a method from 2024-2025 validation and applying that choice to 2026.

## Model Identity

Recommended production model version name after export:

```text
station_high_regressor_v7_live_safe_nbm
```

Model family:

```text
xgboost_lightgbm_catboost_ridge_stack_live_safe_gfs_hrrr_nbm
```

Target:

```text
actual_high_f
```

Prediction output:

```text
predictedHighF
```

Current research artifact directory:

```text
data/calibration/station_stacking_v7
```

Exported production model bundles:

```text
data/calibration/station_stacking_v7/model_weights/{STATION}_station_high_regressor_v7_live_safe_nbm.joblib
data/calibration/station_stacking_v7/model_weights/{STATION}_station_high_regressor_v7_live_safe_nbm.json
data/calibration/station_stacking_v7/model_weights/station_high_regressor_v7_live_safe_nbm_index.csv
```

Backtest outputs:

```text
data/calibration/backtests/station_stacking_v7_2026_oof_wide.csv
data/calibration/backtests/station_stacking_v7_2026_oof_method_metrics.csv
data/calibration/backtests/station_stacking_v6_v7_predictions_long.csv
```

## Supported Stations

```text
KATL
KAUS
KDAL
KHOU
KLAX
KLGA
KMIA
KORD
KSEA
```

Resolve Polymarket locations to these airport stations only. If the market cannot be resolved to one of these station IDs, return `predictionStatus = "unavailable"`.

## Timing Contract

Bot decision:

```text
11:15 AM local station time
```

Forecast snapshot:

```text
forecast_as_of = 11:00 AM local station time
timing_mode = same_day_11am_live_safe
```

Forecast valid window:

```text
valid_time >= 11:00 AM local
valid_time < next local midnight
```

Current observation contract:

```text
Use the latest station observation available for the 11:00 AM local decision snapshot.
Observation timestamp must be <= 11:00 AM local.
The bot may run at 11:15 AM local to allow late report arrival, but the observation itself must not be after 11:00 AM local unless the model is retrained for a wider observation window.
```

This is different from the older v2 deployment prompt that discussed a 10:50-11:10 observation window. The current v7 trend features were pulled as before-11 AM features, so do not use post-11:00 observations with v7.

## Forecast Cycle Rules

Use only model cycles that would have been available by the 11 AM local station-time decision.

HRRR:

```text
Use the latest hourly HRRR run available by 11:00 local.
Training used the 10:00 AM local-equivalent HRRR run:
Eastern stations: 14Z
Central stations: 15Z
Pacific stations: 17Z
```

GFS:

```text
Use the latest GFS cycle available by 11:00 local with availability buffer.
Training used:
Eastern/Central stations: 06Z
Pacific stations: 12Z
```

NBM:

```text
Use direct NOAA NBM, not SDK NBM.
For this 11 AM local bot, use the 13Z NBM cycle.
Source label should identify this as direct NBM.
```

Do not label NBM as NWS. Do not substitute city-center weather for airport-station weather.

## Feature Version

Feature version:

```text
feature_version = v7
```

Providers:

```text
("gfs", "hrrr", "nbm")
```

The authoritative feature list is each station's v7 artifact:

```text
data/calibration/station_stacking_v7/{STATION}_feature_columns.csv
```

Current feature counts:

```text
KATL-KORD except KSEA: 217 total, 6 categorical, 211 numeric
KSEA: 218 total, 6 categorical, 212 numeric
```

Do not hardcode a global feature count. Load the feature contract for the station or from the exported v7 model manifest.

Important v7 feature families:

```text
raw provider highs:
  gfs_high_f
  hrrr_high_f
  nbm_high_f

provider availability and ensemble:
  gfs_available, hrrr_available, nbm_available
  provider_count_available
  provider_mean_high_f
  provider_median_high_f
  provider_min_high_f
  provider_max_high_f
  provider_spread_high_f
  provider_std_high_f

provider timing:
  {provider}_issue_hour_utc
  {provider}_issue_hour_local
  {provider}_as_of_hour_local
  {provider}_forecast_lead_hours
  {provider}_forecast_window_hours

forecast weather:
  cloud cover
  wind speed and gusts
  dew point
  humidity
  precipitation totals, max 1h precip, precip hours, precip flag, precip intensity

current observation:
  observed_temp_at_as_of_f
  observed_high_temp_through_as_of_f
  observed_dewpoint_at_as_of_f
  observed_humidity_at_as_of
  observed_wind_speed_at_as_of
  observed_wind_direction_at_as_of
  observed_pressure_at_as_of
  observed_visibility_at_as_of
  observed_ceiling_at_as_of
  observed_cloud_cover_at_as_of
  observed_precip_recent_at_as_of
  observed_precip_intensity
  observed_precip_intensity_code
  observed_as_of_age_minutes

v7 observation trend inputs:
  observed_temp_change_last_1h_f
  observed_temp_change_last_3h_f
  observed_morning_warmup_rate_f_per_hour
  observed_high_so_far_change_since_9am_f

history and bias:
  actual_high_lag_1d
  actual_high_roll_7d_mean
  actual_high_roll_30d_mean
  provider error lags
  rolling provider bias/MAE
  prior-month provider bias/MAE

cross-model deltas:
  gfs_hrrr_high_f_diff_f
  gfs_nbm_high_f_diff_f
  hrrr_nbm_high_f_diff_f
  absolute variants of the same

v5/v6/v7 engineered features:
  v2_recent_heat_anomaly_f
  v2_recent_heat_momentum_f
  v2_morning_warmup_to_consensus_f
  v2_consensus_minus_7d_actual_f
  v2_spread_per_warmup_f
  v2_humidity_warmup_interaction
  v3_remaining_warmup_from_high_so_far_f
  v3_humidity_remaining_warmup_interaction
  v4 precipitation and observed-precip interaction features
```

Training/evaluation quality fields are not live features:

```text
actual_high_f
actual_source
actual_data_quality_flag
actual_raw_observation_count
strict_quality_ok
strict_quality_issues
```

The active-day bot must never use same-day `actual_high_f` or any final same-day actual summary.

## Training And Validation Setup

V7 notebook path:

```text
notebooks/station_stacking_v7/
```

Fold design:

```text
Fold 1: train 2021-2023 -> validate 2024
Fold 2: train 2021-2024 -> validate 2025
2026: OOF holdout/test
```

Optuna:

```text
base model trials: 50
stack trials: 50
startup/random trials: 20
metric: mae_f
hyperparameter space: wide
```

Base models:

```text
xgboost
lightgbm
catboost
```

Final/default output:

```text
ridge_stack
```

The stack feature set is station-specific and must be read from each exported bundle or manifest. Current exported stack feature sets:

```text
KATL, KAUS, KDAL, KHOU, KLGA, KMIA:
  xgboost_predicted_high_f
  lightgbm_predicted_high_f
  catboost_predicted_high_f

KLAX, KORD, KSEA:
  xgboost_predicted_high_f
  lightgbm_predicted_high_f
  catboost_predicted_high_f
  hrrr_raw_predicted_high_f
  gfs_raw_predicted_high_f
```

NBM is used by the base models as live-safe input features. NBM raw is evaluated as a baseline but is not currently part of the hardcoded raw stack feature set.

## Current v7 2026 Ridge-Stack Backtest

Strict 2026 OOF counts and errors:

```text
KATL  count 130  MAE 1.800  RMSE 2.444
KAUS  count  98  MAE 1.587  RMSE 2.184
KDAL  count 128  MAE 1.509  RMSE 1.964
KHOU  count 113  MAE 1.520  RMSE 2.109
KLAX  count  79  MAE 1.597  RMSE 1.959
KLGA  count  56  MAE 1.508  RMSE 1.958
KMIA  count 107  MAE 1.646  RMSE 2.305
KORD  count 128  MAE 1.702  RMSE 2.185
KSEA  count 114  MAE 1.396  RMSE 1.812
```

Aggregate v7 ridge-stack 2026 OOF:

```text
count: 953
weighted MAE: 1.593 F
weighted RMSE: 2.118 F
```

Known caveat:

```text
KLAX 2026 has only 79 strict clean rows because many final actual-high labels are sparse or inconsistent with the 11 AM observation. Do not over-interpret KLAX 2026 results until actual-high labels are repaired.
```

## Exported Production Bundles

The v7 production bundles were exported on 2026-06-14 with:

```powershell
.\.venv\Scripts\python.exe -m src.export_station_stacking_v2_models `
  --project-root . `
  --artifact-dir data/calibration/station_stacking_v7 `
  --model-version station_high_regressor_v7_live_safe_nbm `
  --timing-mode same_day_11am_live_safe `
  --providers gfs hrrr nbm `
  --feature-version v7 `
  --optuna-metric mae_f `
  --source-pipeline notebooks/station_stacking_v7 `
  --train-years all_available
```

These are production-refit bundles, not 2026 OOF bundles. They train on all strict completed rows available in the v7 artifact table. Use the 2026 OOF CSV files for backtesting; use the `.joblib` bundles for live inference going forward.

Exported training rows:

```text
KATL 1765
KAUS 1390
KDAL 1805
KHOU 1750
KLAX 1695
KLGA 1638
KMIA 1766
KORD 1827
KSEA 1790
```

## Inference Algorithm

For live production:

```python
bundle = joblib.load(model_bundle_path)

feature_row = build_station_stacking_v7_feature_row(
    station_id=station_id,
    contract_date=contract_date,
    timing_mode="same_day_11am_live_safe",
    providers=("gfs", "hrrr", "nbm"),
)

feature_row = ensure_columns(feature_row, bundle["feature_names"])

base_predictions = {
    "xgboost_predicted_high_f": bundle["base_models"]["xgboost"].predict(feature_row[bundle["feature_names"]])[0],
    "lightgbm_predicted_high_f": bundle["base_models"]["lightgbm"].predict(feature_row[bundle["feature_names"]])[0],
    "catboost_predicted_high_f": bundle["base_models"]["catboost"].predict(feature_row[bundle["feature_names"]])[0],
    "hrrr_raw_predicted_high_f": float(feature_row["hrrr_high_f"].iloc[0]),
    "gfs_raw_predicted_high_f": float(feature_row["gfs_high_f"].iloc[0]),
}

stack_row = pd.DataFrame([{name: base_predictions[name] for name in bundle["stack_features"]}])
predicted_high_f = float(bundle["stack_model"].predict(stack_row)[0])
```

Fail closed if required live data is missing. At minimum, require:

```text
station_id supported
contract_date valid
GFS high present and plausible
HRRR high present and plausible
NBM high present and plausible for v7, unless the exported v7 bundle and live policy explicitly allow missing NBM with imputation
current observation present by 11:00 local
observed_temp_at_as_of_f plausible
observed_high_temp_through_as_of_f plausible
trend features available or explicitly set to NaN under the exported model's imputation contract
```

If a live hard check fails, return `predictionStatus = "unavailable"` and log the missing field. Do not silently substitute another provider, another station, city weather, or a post-11:00 observation.

## Production Output Schema

Recommended JSON:

```json
{
  "stationId": "KSEA",
  "contractDate": "2026-06-06",
  "forecastAsOfLocal": "2026-06-06T11:00:00",
  "botDecisionTimeLocal": "2026-06-06T11:15:00",
  "forecastTimingMode": "same_day_11am_live_safe",
  "forecastCycles": {
    "hrrr": "latest_available_by_11_local",
    "gfs": "latest_available_by_11_local",
    "nbm": "13Z"
  },
  "hrrrHighF": 76.2,
  "gfsHighF": 74.8,
  "nbmHighF": 75.4,
  "observedTempAtAsOfF": 68.0,
  "observedHighTempThroughAsOfF": 70.0,
  "observedAtLocal": "2026-06-06T10:53:00",
  "modelVersion": "station_high_regressor_v7_live_safe_nbm",
  "modelFamily": "xgboost_lightgbm_catboost_ridge_stack_live_safe_gfs_hrrr_nbm",
  "primaryMethod": "ridge_stack",
  "predictedHighF": 75.6,
  "predictionIntervalF": null,
  "featuresPointInTimeSafe": true,
  "predictionStatus": "ok",
  "unavailableReason": null,
  "dataLineage": {
    "featurePipeline": "station_stacking_v7",
    "target": "airport_station_actual_high_f",
    "forecastProviders": ["gfs", "hrrr", "nbm"],
    "nbmSource": "direct_noaa_nbm_13z",
    "currentObservationRule": "latest_station_observation_at_or_before_1100_local",
    "stackOutput": "ridge_stack",
    "strictQualityOk": null,
    "strictQualityIssues": []
  }
}
```

If unavailable:

```json
{
  "stationId": "KSEA",
  "contractDate": "2026-06-06",
  "forecastAsOfLocal": "2026-06-06T11:00:00",
  "forecastTimingMode": "same_day_11am_live_safe",
  "modelVersion": "station_high_regressor_v7_live_safe_nbm",
  "predictionStatus": "unavailable",
  "unavailableReason": "missing_direct_nbm_13z_high_f",
  "featuresPointInTimeSafe": true
}
```

## Exporter

The exporter entry point is still named:

```text
src/export_station_stacking_v2_models.py
```

Despite the historical filename, it now supports configurable artifact directories, model versions, providers, timing modes, feature versions, and stack-selection metric. For v7, always pass the full v7 command shown above or read the already-exported bundles from `data/calibration/station_stacking_v7/model_weights`.

## Guardrails

```text
Never use same-day final actual_high_f as an input.
Never use a forecast cycle not available by 11 AM local.
Never use observations timestamped after 11:00 local for this v7 model.
Never use city-center weather for airport-station markets.
Never label direct NBM as NWS.
Never choose the best method by looking at 2026 outcomes for live trading.
Use ridge_stack as the default v7 output unless a future validation-only station policy beats it honestly.
```
