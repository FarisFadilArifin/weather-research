# Polymarket 11 AM HRRR/GFS ML Deployment Prompt

You are designing an actual ML deployment pipeline for a Polymarket weather bot. Build a production-grade regression system based on the repository's station-stacking v2 notebook pipeline. It predicts the final airport-station daily high temperature using an 11:00 AM local HRRR/GFS forecast snapshot plus a bounded station-observation window that closes at 11:10 AM local.

The final system should produce the best predicted final high temperature, not a bucket probability. Include uncertainty, but the primary output is `predictedHighF`.

## Data Contract

Input grain:

```text
one row per {station_id, contract_date}
```

Supported locations are airport stations, not city centers. Always resolve each Polymarket market to the official airport station ID used by the market resolution source.

Supported station IDs:

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

If the market cannot be resolved to one of these station IDs, return `predictionStatus = "unavailable"`.

Prediction target:

```text
actual_high_f
```

Forecast-as-of rule:

```text
forecast_as_of = 11:00 AM local station time
```

Observation-as-of rule:

```text
observation_window = 10:50 AM local through 11:10 AM local
bot_decision_time >= 11:10 AM local
```

Forecast window:

```text
valid_time >= 11:00 AM local
valid_time < next local midnight
```

Primary v2 artifact directory:

```text
data/calibration/station_stacking_v2
```

Production model weight bundles:

```text
data/calibration/station_stacking_v2/model_weights/{STATION}_station_high_regressor_v2.joblib
data/calibration/station_stacking_v2/model_weights/{STATION}_station_high_regressor_v2.json
```

Each station has its own separate model weight bundle. At inference time, resolve the Polymarket market to `station_id`, then load only that station's bundle. Do not use one station's model for another station.

The model bundle's `feature_names` list is the source of truth for inference features. The feature lists in this prompt describe the required feature families, but production code must read `feature_names` from the loaded bundle or its JSON manifest and build exactly those columns in that order.

Port the feature builder from the repository's `src/calibration/station_stacking.py` plus the v2 notebook feature formulas. Do not train or serve a reduced feature set unless a new model bundle is exported for that reduced feature set.

## Forecast Providers

Use only HRRR and GFS.

For each provider, fetch or select the forecast cycle that would have been available by 11:00 AM local station time. Do not use any forecast cycle issued after 11:00 AM local.

Compute these forecast features:

```text
hrrr_high_f
gfs_high_f
hrrr_dewpoint_mean_f
gfs_dewpoint_mean_f
hrrr_humidity_mean
gfs_humidity_mean
hrrr_wind_speed_mean
gfs_wind_speed_mean
hrrr_wind_speed_max
gfs_wind_speed_max
hrrr_precip_total_mm
gfs_precip_total_mm
hrrr_precip_max_1h_mm
gfs_precip_max_1h_mm
hrrr_precip_hours_count
gfs_precip_hours_count
hrrr_has_precip
gfs_has_precip
hrrr_precip_intensity_code
gfs_precip_intensity_code
```

Precipitation forecast fields are numeric model inputs when the exported bundle includes them. They are allowed because they are known at forecast time, but they are optional by bundle: if a given `feature_names` contract does not include them, leave them out or pass `NaN` and let the bundle contract decide.

## Current Observation Rule

The model does not require an observation timestamped exactly 11:00 AM.

Use the latest station observation whose actual observation time is within the 20-minute deployment window around the 11 AM forecast snapshot:

```text
10:50 AM local <= observed_at <= 11:10 AM local
```

Because METAR reports can arrive a few minutes late, separate observation time from bot decision time:

```text
observation_window_start = 10:50 AM local
observation_window_end = 11:10 AM local
bot_decision_time = 11:10 AM local or 11:15 AM local
```

Use only observations satisfying:

```text
10:50 AM local <= observed_at <= 11:10 AM local
received_at <= bot_decision_time
```

If `received_at` is unavailable from the data provider, run inference with a small delay, such as 11:10 or 11:15 local, and record:

```text
observed_as_of_time_local
observed_as_of_time_utc
observed_offset_minutes_from_11am
```

Reject or mark the prediction unavailable if no station observation exists in the valid 10:50-11:10 window.

Do not use observations with:

```text
observed_at > 11:10 AM local
observed_at < 10:50 AM local
```

Compute or use these current-observation features:

```text
observed_temp_at_as_of_f
observed_high_temp_through_as_of_f
observed_dewpoint_at_as_of_f
observed_humidity_at_as_of
observed_wind_speed_at_as_of
observed_wind_direction_at_as_of
observed_pressure_at_as_of
observed_visibility_at_as_of
observed_precip_recent_at_as_of
observed_precip_intensity_code
observed_precip_intensity
observed_as_of_age_minutes
observed_offset_minutes_from_11am
```

Feature types:

| Field | Type | Unit | Bot/inference use |
|---|---|---:|---|
| `observed_temp_at_as_of_f` | numeric feature | deg F | Required live feature |
| `observed_high_temp_through_as_of_f` | numeric feature | deg F | Required live sanity/feature field; must be `<= predicted final high` in reasonableness checks |
| `observed_dewpoint_at_as_of_f` | numeric feature | deg F | Live feature when present |
| `observed_humidity_at_as_of` | numeric feature | percent | Live feature when present |
| `observed_wind_speed_at_as_of` | numeric feature | mph | Live feature when present |
| `observed_wind_direction_at_as_of` | numeric feature | degrees | Live feature when present; also encoded as sine/cosine |
| `observed_pressure_at_as_of` | numeric feature | hPa/inHg normalized by builder | Live feature when present |
| `observed_visibility_at_as_of` | numeric feature | miles | Live feature when present |
| `observed_precip_recent_at_as_of` | numeric feature | inches or mm | Live feature when present; often sparse |
| `observed_precip_intensity_code` | numeric/categorical feature | code | Derived from recent precip and METAR present-weather codes |
| `observed_precip_intensity` | categorical/audit feature | text | Derived label, usually for lineage and feature engineering |
| `observed_as_of_age_minutes` | numeric feature/audit | minutes | Prefer signed offset from 11 AM if retained |
| `observed_offset_minutes_from_11am` | numeric feature/audit | minutes | Preferred explicit signed offset field |
| `observed_as_of_time_local` | lineage/audit | timestamp | Required for point-in-time validation, not a numeric model input |
| `observed_as_of_time_utc` | lineage/audit | timestamp | Required for point-in-time validation, not a numeric model input |

If the legacy feature name `observed_as_of_age_minutes` is retained, define it carefully because observations after 11:00 AM are now allowed. Either store signed minutes relative to 11:00 AM, where 10:53 is `+7` and 11:07 is `-7`, or add `observed_offset_minutes_from_11am` and retrain/export the model bundle with that feature contract.

## Derived Features

Build the station-stacking base features plus these deployment-critical derived features:

```text
provider_mean_high_f = mean(hrrr_high_f, gfs_high_f)
provider_median_high_f = median(hrrr_high_f, gfs_high_f)
provider_min_high_f = min(hrrr_high_f, gfs_high_f)
provider_max_high_f = max(hrrr_high_f, gfs_high_f)
provider_spread_high_f = provider_max_high_f - provider_min_high_f
provider_std_high_f = std(hrrr_high_f, gfs_high_f)
hrrr_high_minus_obs_temp_f = hrrr_high_f - observed_temp_at_as_of_f
gfs_high_minus_obs_temp_f = gfs_high_f - observed_temp_at_as_of_f
hrrr_high_minus_observed_high_temp_f = hrrr_high_f - observed_high_temp_through_as_of_f
gfs_high_minus_observed_high_temp_f = gfs_high_f - observed_high_temp_through_as_of_f
observed_dewpoint_depression_f = observed_temp_at_as_of_f - observed_dewpoint_at_as_of_f
observed_wind_dir_sin = sin(radians(observed_wind_direction_at_as_of))
observed_wind_dir_cos = cos(radians(observed_wind_direction_at_as_of))
observed_heat_index_at_as_of_f
observed_wind_chill_at_as_of_f
observed_is_raining_at_as_of
observed_is_fog_or_mist_at_as_of
observed_is_thunder_at_as_of
actual_high_lag_1d
actual_high_roll_7d_mean
actual_high_roll_30d_mean
month
year
day_of_year_sin
day_of_year_cos
station_id
```

Add the v2 notebook features exactly:

```text
v2_recent_heat_anomaly_f = actual_high_lag_1d - actual_high_roll_30d_mean
v2_recent_heat_momentum_f = actual_high_roll_7d_mean - actual_high_roll_30d_mean
v2_morning_warmup_to_consensus_f = provider_mean_high_f - observed_temp_at_as_of_f
v2_consensus_minus_7d_actual_f = provider_mean_high_f - actual_high_roll_7d_mean
v2_spread_per_warmup_f = provider_spread_high_f / abs(v2_morning_warmup_to_consensus_f).clip(lower=1.0)
v2_humidity_warmup_interaction = (observed_humidity_at_as_of / 100.0) * v2_morning_warmup_to_consensus_f
```

The full exported v2 bundle also uses additional lagged actual, rolling actual, provider-error, provider-bias, provider-difference, forecast-shape, observation-history, and observation-vs-forecast delta columns. Do not omit them if they appear in `bundle["feature_names"]`.

Important: widening observations from 10:50-11:00 to 10:50-11:10 changes the live feature distribution. Before production, retrain the station-stacking v2 artifacts and re-export the model bundles using the same 10:50-11:10 observation rule. Do not claim the current exported weights are fully aligned with post-11:00 observations unless they were rebuilt with this rule.

All lagged actual and provider-error features must use only completed dates strictly earlier than `contract_date`. For an active same-day market, `actual_high_f` for the current `contract_date` is unknown and must never be used as an input.

For any feature in `bundle["feature_names"]` that is structurally unavailable at inference time but not part of the hard-required live data, create the column with `null`/`NaN` and let the fitted preprocessing pipeline impute it. Do not invent substitute values from future data or city data.

## Strict Data Quality Contract

The current training/evaluation pipeline adds strict quality columns to station-stacking feature artifacts and calibration samples:

| Field | Type | Scope | Meaning | Bot/inference use |
|---|---|---|---|---|
| `strict_quality_ok` | boolean | training/eval artifact | `true` when the row passes label, current-observation, and forecast sanity checks | Do not pass to base model unless it appears in `bundle["feature_names"]`; normally audit/gating only |
| `strict_quality_issues` | semicolon-separated string | training/eval artifact | Names of failed checks, such as `actual_below_observed_high_so_far` or `actual_quality_not_ok` | Audit/logging only |
| `actual_source` | string | completed historical rows | Source lineage for the final high label | Training/eval lineage only; unavailable for active same-day inference |
| `actual_data_quality_flag` | string/categorical | completed historical rows | Label quality from actual-high construction, usually `ok` or `sparse_observations` | Training/eval gate only |
| `actual_raw_observation_count` | numeric | completed historical rows | Number of observations used to compute historical `actual_high_f` | Training/eval gate only |

Strict checks exclude rows from training/evaluation when they fail rules such as:

```text
actual_high_f is missing or outside plausible Fahrenheit range
actual_data_quality_flag != ok
actual_raw_observation_count < 18
observed_fetch_status is missing or not ok
observed_temp_at_as_of_f is missing when observed_fetch_status == ok
observed_high_temp_through_as_of_f is missing when observed_fetch_status == ok
actual_high_f < observed_high_temp_through_as_of_f
observed_temp_at_as_of_f > actual_high_f
observed_as_of_age_minutes > 20
hrrr_high_f or gfs_high_f outside plausible Fahrenheit range
```

These columns are not a market signal and are not a same-day target substitute. They define the benchmark population. Report metrics as `strict 2026 holdout` when these filters are active, and include excluded-row counts and reason summaries next to accuracy metrics.

At live bot inference time, `strict_quality_ok` for the current day cannot be fully known because `actual_high_f` is unknown. The bot should instead run the live subset of strict checks:

```text
station_id is supported
HRRR and GFS highs are present and plausible Fahrenheit values
current observation is inside 10:50-11:10 local
observed_fetch_status == ok
observed_temp_at_as_of_f is present and plausible
observed_high_temp_through_as_of_f is present and plausible
prediction should normally be >= observed_high_temp_through_as_of_f
```

If a live hard check fails, return `predictionStatus = "unavailable"` with the specific reason. If the model predicts below the already observed high-so-far, clamp only for downstream market-bracket sanity if the trading policy explicitly allows it; always log both raw and adjusted predictions.

## ML Model

Use the station-stacking v2 supervised regression model that predicts:

```text
predicted_high_f = model(features)
```

The exported v2 bundle contains these base regressors:

```text
LightGBMRegressor
XGBoostRegressor
CatBoostRegressor
```

The production prediction path should run all three base models and then use the exported `Ridge` stack as the final model. The stack input uses these exact feature names:

```text
xgboost_predicted_high_f
lightgbm_predicted_high_f
catboost_predicted_high_f
hrrr_raw_predicted_high_f
gfs_raw_predicted_high_f
```

Load the station bundle with `joblib`. The bundle includes:

```text
schema_version
model_version = station_high_regressor_v2
station_id
target = actual_high_f
base_models
stack_model
stack_features
categorical_features
numeric_features
feature_names
providers = (gfs, hrrr)
```

The base model objects in `bundle["base_models"]` are fitted scikit-learn pipelines that already include preprocessing and model weights. Pass a raw DataFrame with `bundle["feature_names"]` columns to each base model. Do not manually one-hot encode, scale, or impute before calling the pipeline.

Before predicting, validate:

```text
bundle.station_id == resolved station_id
bundle.model_version == station_high_regressor_v2
bundle.providers == (gfs, hrrr)
bundle.target == actual_high_f
```

## Inference Algorithm

Use this exact inference shape:

```python
bundle = joblib.load(model_bundle_path)

feature_row = build_station_stacking_v2_feature_row(...)
feature_row = ensure_columns(feature_row, bundle["feature_names"])

base_predictions = {
    "xgboost_predicted_high_f": bundle["base_models"]["xgboost"].predict(feature_row[bundle["feature_names"]])[0],
    "lightgbm_predicted_high_f": bundle["base_models"]["lightgbm"].predict(feature_row[bundle["feature_names"]])[0],
    "catboost_predicted_high_f": bundle["base_models"]["catboost"].predict(feature_row[bundle["feature_names"]])[0],
    "hrrr_raw_predicted_high_f": float(feature_row["hrrr_high_f"].iloc[0]),
    "gfs_raw_predicted_high_f": float(feature_row["gfs_high_f"].iloc[0]),
}

stack_row = DataFrame([{name: base_predictions[name] for name in bundle["stack_features"]}])
predicted_high_f = bundle["stack_model"].predict(stack_row)[0]
```

Do not re-fit models in the inference service. Use the exported `.joblib` weights.

Hard-required live inputs for inference:

```text
station_id
contract_date
hrrr_high_f
gfs_high_f
observed_temp_at_as_of_f
observed_high_temp_through_as_of_f
observed_as_of_time_local
10:50 AM local <= observed_at <= 11:10 AM local
```

If either HRRR or GFS high is unavailable, return unavailable. Do not fall back to NBM, Open-Meteo, city weather, or a single-provider model unless a separate model bundle is trained for that mode.

## Validation And Deployment Rule

The v2 notebooks tune against two fixed validation folds:

```text
train 2021-2023 -> validate 2024
train 2022-2024 -> validate 2025
```

They then train/test:

```text
train 2021-2025 -> test 2026
```

For production deployment, the exported `station_high_regressor_v2` bundles are refit on all available completed actuals in the v2 feature table, unless a strict holdout export is explicitly requested.

For every training, evaluation, or prediction date:

```text
training_data.contract_date < prediction_date
```

Never train or evaluate using future rows.

Hard leakage bans:

```text
Do not use actual_high_f as an input feature.
Do not use final same-day summaries.
Do not use observations outside the 10:50-11:10 local deployment window.
Do not use observations after 11:10 AM local.
Do not use forecast cycles issued after 11 AM.
Do not use market outcome or resolution data as an input feature.
Do not mix city-center weather with airport-station weather.
```

## Production Output Schema

Return this JSON for every prediction:

```json
{
  "stationId": "KSEA",
  "contractDate": "2026-06-06",
  "forecastAsOfLocal": "2026-06-06T11:00:00",
  "observationWindowLocal": {
    "start": "2026-06-06T10:50:00",
    "end": "2026-06-06T11:10:00"
  },
  "botDecisionTimeLocal": "2026-06-06T11:15:00",
  "hrrrHighF": 76.2,
  "gfsHighF": 74.8,
  "observedTempAtAsOfF": 68.0,
  "observedHighTempThroughAsOfF": 70.0,
  "observedAtLocal": "2026-06-06T11:07:00",
  "observedOffsetMinutesFrom11am": -7,
  "modelVersion": "station_high_regressor_v2",
  "modelFamily": "xgboost_lightgbm_catboost_ridge_stack",
  "modelBundlePath": "data/calibration/station_stacking_v2/model_weights/KSEA_station_high_regressor_v2.joblib",
  "predictedHighF": 75.6,
  "predictionIntervalF": [73.1, 78.0],
  "featuresPointInTimeSafe": true,
  "predictionStatus": "ok",
  "dataLineage": {
    "forecastProviders": ["hrrr", "gfs"],
    "observationRule": "latest_station_observation_between_1050_and_1110_local",
    "featurePipeline": "station_stacking_v2",
    "dataQualityPolicy": "strict_training_eval_live_subset_at_inference",
    "strictQualityOk": null,
    "strictQualityIssues": [],
    "target": "airport_station_actual_high_f",
    "stackFeatures": [
      "xgboost_predicted_high_f",
      "lightgbm_predicted_high_f",
      "catboost_predicted_high_f",
      "hrrr_raw_predicted_high_f",
      "gfs_raw_predicted_high_f"
    ]
  }
}
```

`predictionIntervalF` is not stored directly inside the `.joblib` bundle. If implemented, derive it from station/model validation or test residuals and document the method in the response. If no interval model is implemented, return `predictionIntervalF: null` rather than fabricating precision.

If required data is missing, return an unavailable response:

```json
{
  "stationId": "KSEA",
  "contractDate": "2026-06-06",
  "forecastAsOfLocal": "2026-06-06T11:00:00",
  "predictionStatus": "unavailable",
  "unavailableReason": "missing_valid_observation_between_1050_and_1110_local",
  "featuresPointInTimeSafe": true
}
```

## Engineering Requirements

Implement separate modules for:

1. market and station resolver
2. HRRR/GFS forecast fetcher
3. current-observation fetcher
4. feature builder
5. model trainer
6. model registry and versioning
7. inference service
8. prediction logger
9. backtest and walk-forward evaluator
10. v2 weight exporter and artifact manifest reader

The inference path must fail closed if required point-in-time data is missing. It may return `"predictionStatus": "unavailable"` with a reason, but it must not silently substitute future data, stale observations outside the 10:50-11:10 window, observations after 11:10 AM, other forecast providers, or city-center weather.

## Model Export Command

Export or refresh v2 station model weights from existing notebook artifacts:

```powershell
.\.venv\Scripts\python.exe -m src.export_station_stacking_v2_models --project-root .
```

Export strict 2021-2025 train-year bundles for 2026 holdout-style inference:

```powershell
.\.venv\Scripts\python.exe -m src.export_station_stacking_v2_models --project-root . --train-years 2021-2025
```
