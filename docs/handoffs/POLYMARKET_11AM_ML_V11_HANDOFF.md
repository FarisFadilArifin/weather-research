# Polymarket 11 AM ML v11 Handoff

This handoff supersedes `POLYMARKET_11AM_ML_V7_HANDOFF.md` for the station-stacking high-temperature model.

Use this document together with the exact per-feature inventory:

```text
POLYMARKET_11AM_ML_V11_FEATURE_INVENTORY.csv
POLYMARKET_11AM_ML_V11_ARTIFACT_SUMMARY.csv
```

Those CSVs were generated from the current v11 artifacts, not from a hand-written feature list.

Regenerate them with:

```powershell
.\.venv\Scripts\python.exe scripts\summarize_station_stacking_v11_handoff.py
```

## Recommendation

Use **station-stacking v11 ridge stack** as the current best production candidate.

```text
model_version: station_high_regressor_v11_huber_ridge_stack
primary_method: ridge_stack
feature_version: v11
timing_mode: same_day_11am_live_safe
providers: gfs, hrrr, nbm
target_mode: remaining_warmup
model_target: remaining_warmup_from_observed_high_so_far_f
selection_metric: mae_f
```

Supported stations:

```text
KATL KAUS KDAL KHOU KLAX KLGA KMIA KORD KSEA
```

If a market cannot be resolved to one of these airport station IDs, return `predictionStatus = "unavailable"`.

## Current Artifacts

Research artifact directory:

```text
data/calibration/station_stacking_v11
```

Exported model bundles:

```text
data/calibration/station_stacking_v11/model_weights/{STATION}_station_high_regressor_v11_huber_ridge_stack.joblib
data/calibration/station_stacking_v11/model_weights/{STATION}_station_high_regressor_v11_huber_ridge_stack.json
```

Use the files this way:

```text
.joblib = production inference bundle; load this with joblib.load(...) and call predict
.json   = manifest/metadata only; use it to validate schema, model contract, features, and lineage
```

Do not try to run inference from the `.json` manifest. It does not contain the fitted estimators.

All 9 stations currently have scoreboards and exported `.joblib`/`.json` bundles.

| station | feature_rows | feature_count | categorical_count | numeric_count | export_train_rows | stack_feature_set | test_2026_ridge_mae_f | test_2026_ridge_rmse_f | test_2026_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| KATL | 1964 | 197 | 3 | 194 | 1956 | models_only | 1.5319 | 2.2098 | 137 |
| KAUS | 1964 | 197 | 3 | 194 | 1954 | models_only | 1.5877 | 2.1388 | 137 |
| KDAL | 1964 | 197 | 3 | 194 | 1953 | models_only | 1.4300 | 1.9079 | 138 |
| KHOU | 1964 | 197 | 3 | 194 | 1953 | models_only | 1.4567 | 2.0139 | 138 |
| KLAX | 1964 | 197 | 3 | 194 | 1956 | models_plus_raw | 1.4394 | 1.7884 | 137 |
| KLGA | 1964 | 197 | 3 | 194 | 1945 | models_only | 1.6507 | 2.1434 | 138 |
| KMIA | 1964 | 197 | 3 | 194 | 1960 | models_plus_raw | 1.6119 | 2.1494 | 137 |
| KORD | 1964 | 197 | 3 | 194 | 1937 | models_only | 1.4385 | 1.9677 | 137 |
| KSEA | 1964 | 198 | 3 | 195 | 1953 | models_plus_raw | 1.3953 | 1.7308 | 136 |

Aggregate v11 ridge-stack 2026 test over all 9 completed stations:

```text
weighted MAE:  1.5048 F
weighted RMSE: 2.0058 F
p90 abs error: 3.1679 F
p95 abs error: 3.9853 F
p99 abs error: 6.1338 F
within 2 F:    72.47%
within 3 F:    88.34%
```

## Timing Contract

Bot decision time:

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

Current-observation rule:

```text
Use the latest station observation at or before 11:00 AM local station time.
The bot may run at 11:15 AM local to allow late report arrival.
Do not use an observation whose actual observation time is after 11:00 AM local unless the model is retrained.
```

Training rejects stale current observations through strict quality checks when `observed_as_of_age_minutes > 20`.

## Forecast Cycle Rules

Use only provider forecasts that would have been available by the 11 AM local station-time decision.

HRRR:

```text
Use the latest hourly HRRR run available by 11:00 local.
Training used live-safe 11 AM cache shards.
```

GFS:

```text
Use the latest GFS cycle available by 11:00 local with availability buffer.
Training used live-safe 11 AM cache shards.
```

NBM:

```text
Use direct NOAA NBM 13Z, not SDK NBM.
Set WEATHER_RESEARCH_INCLUDE_DIRECT_NBM=1 when rebuilding features outside the v11 runner.
```

Do not substitute city-center weather, another airport, a later provider cycle, or post-11 observations.

## Target And Model Shape

v11 does **not** train base learners directly on `actual_high_f`.

Base learner target:

```text
remaining_warmup_from_observed_high_so_far_f =
    actual_high_f - observed_high_temp_through_as_of_f
```

Base prediction transform:

```text
base_predicted_high_f =
    max(
        observed_high_temp_through_as_of_f,
        observed_high_temp_through_as_of_f + model_output_remaining_warmup
    )
```

The ridge stack is then fit on the transformed base high-temperature predictions against `actual_high_f`.

Important inference nuance:

```text
Do clamp each base model output back to at least observed_high_so_far.
Do not add an extra final clamp after ridge_stack if exact reproduction is required.
```

Adding a final post-stack clamp may be a reasonable production guardrail, but it is not the exact v11 artifact behavior and should be tracked as a new serving policy.

## Model Families

Base models:

```text
xgboost
lightgbm
catboost
```

v11 robust objectives:

```text
XGBoost:  objective = reg:pseudohubererror, eval_metric = mae
LightGBM: objective = huber, metric = mae, huber_alpha tuned by Optuna
CatBoost: loss_function = Huber:delta={huber_delta}, eval_metric = MAE
```

Final model:

```text
ridge_stack
```

Stack inputs are station-specific. Read `bundle["stack_features"]` or the JSON manifest. Current feature sets:

```text
models_only:
  xgboost_predicted_high_f
  lightgbm_predicted_high_f
  catboost_predicted_high_f

models_plus_raw:
  xgboost_predicted_high_f
  lightgbm_predicted_high_f
  catboost_predicted_high_f
  hrrr_raw_predicted_high_f
  gfs_raw_predicted_high_f
```

Current station stack feature sets:

```text
models_only:    KATL KAUS KDAL KHOU KLGA KORD
models_plus_raw: KLAX KMIA KSEA
```

NBM is used by base models as a live-safe input feature. NBM raw is not currently part of the stack's raw forecast feature set.

## Feature Construction

Main builder:

```text
src/calibration/station_stacking.py::build_station_wide_dataset
```

It merges:

```text
data/processed/actual_highs.csv
data/processed/station_registry.csv
data/calibration/sdk_current_obs_*/sdk_current_observations_11am.csv
data/calibration/sdk_11am_*/sdk_nwp_0h_cache.csv
data/calibration/direct_nbm_*/direct_nbm_0h_cache.csv
```

Then it adds feature blocks in this order:

```text
calendar
current-observation derived features
provider availability/time/ensemble features
forecast shape features
provider cross-model differences
lagged actual highs
lagged provider forecast errors
prior-month provider errors
forecast/history delta features
observation/history delta features
observation/forecast delta features
versioned v5/v8 feature engineering
v9/v10/v11 10-year climatology features
strict quality flags
```

v11 aliases the v9 feature contract:

```text
V11_FEATURE_COLUMNS = V9_FEATURE_COLUMNS
V11_DROPPED_FEATURE_COLUMNS = V9_DROPPED_FEATURE_COLUMNS
```

But this is **not** a closed production whitelist. The actual feature selector is exclusion-based.

The production source of truth is, per station:

```text
bundle["feature_names"]
bundle["categorical_features"]
bundle["numeric_features"]
manifest["features"]["all"]
data/calibration/station_stacking_v11/{STATION}_feature_columns.csv
```

Do not hardcode a global feature count. KSEA currently has one extra selected numeric feature (`observed_snow_depth_at_as_of`) because it has at least one non-null value after filtering.

## Used Feature Families

Current v11 selected-feature inventory, by family:

| family | kind | feature_count |
| --- | --- | --- |
| provider historical error/bias | numeric | 54 |
| current observation | numeric | 30 |
| prior actual history | numeric | 19 |
| provider forecast/raw | numeric | 19 |
| provider cross-model delta | numeric | 9 |
| v8 remaining-warmup engineered | numeric | 9 |
| observation history | numeric | 7 |
| v4 precipitation engineered | numeric | 7 |
| provider ensemble/climatology delta | numeric | 6 |
| provider-observation delta | numeric | 6 |
| v2 heat/warmup engineered | numeric | 6 |
| v3 high-so-far engineered | numeric | 6 |
| v9/v11 climatology | numeric | 6 |
| calendar | numeric | 4 |
| current observation trend | numeric | 4 |
| other numeric | numeric | 3 |
| current observation | categorical | 2 |
| calendar/categorical | categorical | 1 |

Categorical features currently used by every station:

```text
day_of_week
observed_weather_code_at_as_of
observed_precip_intensity
```

Core hard-required live numeric fields:

```text
gfs_high_f
hrrr_high_f
nbm_high_f
observed_temp_at_as_of_f
observed_high_temp_through_as_of_f
observed_as_of_age_minutes
```

Why these are hard required:

```text
Provider highs are required by _modeling_frame and are primary live signal.
observed_high_temp_through_as_of_f is required to transform remaining-warmup outputs back to predicted highs.
observed_temp_at_as_of_f and observed_as_of_age_minutes are strict current-observation sanity fields.
```

## NaN And Missingness Contract

NaNs are expected before preprocessing.

The fitted pipeline handles missing values:

```text
categorical: SimpleImputer(strategy="constant", fill_value="missing") + OneHotEncoder(handle_unknown="ignore")
numeric:     SimpleImputer(strategy="median")
```

Do not manually fill model-input NaNs differently in the bot. Pass null/NaN through to the exported pipeline unless the field is hard-required.

There are two useful missingness views:

```text
all feature rows: all rows in {STATION}_features.csv before strict quality/modeling filters
modeling rows: rows that passed strict quality and provider/target gates for fitting
```

All-feature-row missingness buckets, using each feature's worst station-level missing rate:

| severity by feature max NaN rate | feature_count |
| --- | --- |
| 0% everywhere | 64 |
| >0% to 1% | 111 |
| >1% to 5% | 12 |
| >5% to 20% | 3 |
| >20% to 80% | 1 |
| >80% | 7 |

Expected NaN policy counts from strict v11 modeling rows:

| expected_nan_policy | feature_count |
| --- | --- |
| expected_nan_early_history_or_missing_prior; numeric median imputer handles | 80 |
| may_be_null; categorical imputer fills "missing" | 3 |
| may_be_null; numeric median imputer handles | 33 |
| observed_non_null_in_v11_artifacts; still build column defensively | 70 |
| required_live_non_null; fail closed if missing | 6 |
| should_be_present_after_normals_join; investigate if high missingness | 6 |

Highest-missing selected features in current v11 all-feature-row artifacts:

| feature | kind | family | used_station_count | missing_pct_min_all_rows | missing_pct_max_all_rows | expected_nan_policy |
| --- | --- | --- | --- | --- | --- | --- |
| observed_snow_depth_at_as_of | numeric | current observation | 1 | 99.949 | 99.949 | may_be_null; numeric median imputer handles |
| observed_wind_chill_at_as_of_f | numeric | current observation | 9 | 57.739 | 99.796 | may_be_null; numeric median imputer handles |
| observed_heat_index_at_as_of_f | numeric | current observation | 9 | 32.892 | 99.338 | may_be_null; numeric median imputer handles |
| observed_peak_wind_gust_at_as_of | numeric | current observation | 9 | 88.849 | 98.676 | may_be_null; numeric median imputer handles |
| observed_peak_wind_direction_at_as_of | numeric | current observation | 9 | 88.849 | 98.676 | may_be_null; numeric median imputer handles |
| observed_wind_gust_at_as_of | numeric | current observation | 9 | 69.807 | 97.047 | may_be_null; numeric median imputer handles |
| observed_weather_code_at_as_of | categorical | current observation | 9 | 83.707 | 95.214 | may_be_null; categorical imputer fills "missing" |
| observed_ceiling_at_as_of | numeric | current observation | 9 | 24.745 | 54.481 | may_be_null; numeric median imputer handles |

Common expected-missing categories:

```text
early-history actual lags and rolling stats
provider lagged error/bias/MAE features
prior-month provider errors
weather fields that are sparse by nature, such as gusts, ceilings, heat index, wind chill, weather codes, snow depth
categorical weather code fields when METAR has no present weather
```

Climatology features should normally be present after the v9/v11 normals join. Investigate high missingness in:

```text
climatology_high_10y_f
climatology_high_10y_std_f
climatology_high_10y_count
provider_mean_minus_climatology_10y_f
observed_temp_minus_climatology_10y_f
observed_high_so_far_minus_climatology_10y_f
```

## Strict Quality Gates

Training/evaluation filters are applied by `add_strict_quality_flags` and `_modeling_frame`.

Rows are excluded from model fitting/evaluation when they fail checks such as:

```text
missing actual_high_f
actual_high_f outside plausible Fahrenheit range
actual_data_quality_flag present and not ok
actual_raw_observation_count < 18
observed_fetch_status missing or not ok
observed_temp_at_as_of_f missing when observed_fetch_status is ok
observed_temp_at_as_of_f outside plausible Fahrenheit range
observed_temp_at_as_of_f > actual_high_f
observed_high_temp_through_as_of_f missing when observed_fetch_status is ok
observed_high_temp_through_as_of_f outside plausible Fahrenheit range
actual_high_f < observed_high_temp_through_as_of_f
observed_as_of_age_minutes > 20
provider high outside plausible Fahrenheit range
```

Additional `_modeling_frame` gates:

```text
all provider highs must be present: gfs_high_f, hrrr_high_f, nbm_high_f
all_provider_highs_available must be true
remaining-warmup target must be non-null
strict_quality_ok must be true
```

For live inference, do not pass training-only columns:

```text
actual_high_f
remaining_warmup_from_observed_high_so_far_f
actual_source
actual_data_quality_flag
actual_raw_observation_count
strict_quality_ok
strict_quality_issues
```

The active-day bot must never use same-day `actual_high_f`. Actual-derived features are allowed only when they are strictly lagged/prior-history values.

## Inference Algorithm

Load the station-specific bundle:

```python
import joblib
import pandas as pd

bundle = joblib.load(
    "data/calibration/station_stacking_v11/model_weights/"
    f"{station_id}_station_high_regressor_v11_huber_ridge_stack.joblib"
)
```

Build exactly one live feature row for `{station_id, contract_date}` using the same v11 feature logic and point-in-time rules.

Before prediction:

```python
feature_row = build_station_stacking_v11_feature_row(...)

for column in bundle["feature_names"]:
    if column not in feature_row.columns:
        feature_row[column] = pd.NA

missing_hard = [
    column
    for column in [
        "gfs_high_f",
        "hrrr_high_f",
        "nbm_high_f",
        "observed_temp_at_as_of_f",
        "observed_high_temp_through_as_of_f",
        "observed_as_of_age_minutes",
    ]
    if column not in feature_row.columns or pd.isna(feature_row[column].iloc[0])
]
if missing_hard:
    return {"predictionStatus": "unavailable", "unavailableReason": f"missing {missing_hard}"}

x = feature_row[bundle["feature_names"]]
observed_high = float(feature_row[bundle["observed_high_so_far_column"]].iloc[0])
```

Base predictions:

```python
stack_inputs = {}

for method, model in bundle["base_models"].items():
    remaining_warmup = float(model.predict(x)[0])
    stack_inputs[f"{method}_predicted_high_f"] = max(
        observed_high,
        observed_high + remaining_warmup,
    )

stack_inputs["hrrr_raw_predicted_high_f"] = float(feature_row["hrrr_high_f"].iloc[0])
stack_inputs["gfs_raw_predicted_high_f"] = float(feature_row["gfs_high_f"].iloc[0])
```

Final v11 ridge-stack prediction:

```python
stack_row = pd.DataFrame([{name: stack_inputs[name] for name in bundle["stack_features"]}])
predicted_high_f = float(bundle["stack_model"].predict(stack_row)[0])
```

Return `predictedHighF = predicted_high_f`.

## Reproduce Pipeline

Use PowerShell from repo root:

```powershell
cd D:\dev\weather-research
```

Optional: refresh live-safe HRRR/GFS caches:

```powershell
.\scripts\backfill_live_safe_shards.ps1 `
  -StartDate "2021-01-01" `
  -EndDate "2026-06-10" `
  -Models @("hrrr", "gfs") `
  -TimingMode "same_day_11am_live_safe"
```

Optional: refresh direct NBM 13Z caches:

```powershell
.\scripts\backfill_direct_nbm_13z_shards.ps1 `
  -StartDate "2021-01-01" `
  -EndDate "2026-06-10" `
  -TimingMode "same_day_11am_live_safe"
```

Optional: refresh current-observation trend caches:

```powershell
.\scripts\backfill_current_obs_trend_shards.ps1 `
  -StartDate "2021-01-01" `
  -EndDate "2026-06-10"
```

Optional: rebuild 10-year rolling daily high climatology normals:

```powershell
.\.venv\Scripts\python.exe scripts\build_all_station_climatology_10y.py `
  --input-dir data\calibration\station_stacking_v8 `
  --output-root outputs\climatology_all_stations
```

Current v11 defaults reuse:

```text
data/calibration/station_stacking_v9/station_rolling_10y_daily_high_normals.csv
```

Train/evaluate/export v11:

```powershell
.\.venv\Scripts\python.exe scripts\run_station_stacking_v11.py --quiet-optuna
```

Run only selected stations:

```powershell
.\.venv\Scripts\python.exe scripts\run_station_stacking_v11.py --stations KATL,KAUS --quiet-optuna
```

Train/evaluate without exporting:

```powershell
.\.venv\Scripts\python.exe scripts\run_station_stacking_v11.py --skip-export --quiet-optuna
```

The runner defaults:

```text
output_dir = data/calibration/station_stacking_v11
base Optuna trials = 30
stack Optuna trials = 30
startup trials = 15
feature_version = v11
target_mode = remaining_warmup
optuna_metric = mae_f
hyperparameter_space = wide
year_split_folds = expanding 2021-2023 -> 2024, 2021-2024 -> 2025
test year = 2026
auto-export enabled unless --skip-export
```

Checkpointing:

```text
data/calibration/station_stacking_v11/{STATION}_optuna.sqlite3
```

Optuna studies use `load_if_exists=True`, so reruns resume from existing SQLite DBs. To rerun from scratch, move or delete the relevant `{STATION}_optuna.sqlite3` and station artifact files first.

## Export Only

If training artifacts already exist:

```powershell
.\.venv\Scripts\python.exe src\export_station_stacking_v2_models.py `
  --project-root . `
  --artifact-dir data\calibration\station_stacking_v11 `
  --model-version station_high_regressor_v11_huber_ridge_stack `
  --timing-mode same_day_11am_live_safe `
  --providers gfs hrrr nbm `
  --feature-version v11 `
  --optuna-metric mae_f `
  --target-mode remaining_warmup `
  --base-model-methods xgboost lightgbm catboost `
  --source-pipeline scripts/run_station_stacking_v11.py `
  --train-years all_available
```

Exporter inputs per station:

```text
{STATION}_features.csv
{STATION}_year_split_selected_hyperparameters.csv
{STATION}_year_split_validation_predictions.csv
{STATION}_year_split_stack_tuning.csv
```

Exporter outputs:

```text
data/calibration/station_stacking_v11/model_weights/{STATION}_station_high_regressor_v11_huber_ridge_stack.joblib
data/calibration/station_stacking_v11/model_weights/{STATION}_station_high_regressor_v11_huber_ridge_stack.json
data/calibration/station_stacking_v11/model_weights/station_high_regressor_v11_huber_ridge_stack_index.csv
```

## Verification Commands

Focused station-stacking tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_station_stacking.py -q
```

Current known passing result after v11 implementation:

```text
37 passed
```

Check current v11 output completeness:

```powershell
Get-ChildItem data\calibration\station_stacking_v11 -Filter "*_year_split_scoreboard.csv"
Get-ChildItem data\calibration\station_stacking_v11\model_weights -Filter "*station_high_regressor_v11_huber_ridge_stack*"
```

## Production Output Schema

Recommended bot JSON fields:

```json
{
  "stationId": "KSEA",
  "contractDate": "2026-06-06",
  "forecastAsOfLocal": "2026-06-06T11:00:00",
  "botDecisionTimeLocal": "2026-06-06T11:15:00",
  "forecastTimingMode": "same_day_11am_live_safe",
  "forecastProviders": ["gfs", "hrrr", "nbm"],
  "nbmSource": "direct_noaa_nbm_13z",
  "observedAtLocal": "2026-06-06T10:53:00",
  "observedHighTempThroughAsOfF": 70.0,
  "modelVersion": "station_high_regressor_v11_huber_ridge_stack",
  "featureVersion": "v11",
  "targetMode": "remaining_warmup",
  "primaryMethod": "ridge_stack",
  "predictedHighF": 75.6,
  "predictionStatus": "ok",
  "unavailableReason": null,
  "featuresPointInTimeSafe": true,
  "dataLineage": {
    "featurePipeline": "station_stacking_v11",
    "baseModels": ["xgboost", "lightgbm", "catboost"],
    "stackOutput": "ridge_stack",
    "currentObservationRule": "latest_station_observation_at_or_before_1100_local",
    "strictQualityIssues": []
  }
}
```

If unavailable:

```json
{
  "predictionStatus": "unavailable",
  "unavailableReason": "missing nbm_high_f",
  "predictedHighF": null
}
```

## Do Not Do These

```text
Do not use same-day actual_high_f for live inference.
Do not use observations after 11:00 local.
Do not use city-center weather for airport station markets.
Do not substitute one station's model for another station.
Do not hardcode the global feature count.
Do not assume V11_FEATURE_COLUMNS alone is the full schema.
Do not manually impute model-input NaNs with custom values.
Do not train or serve a reduced feature set without exporting a new bundle.
Do not pick per-station methods using 2026 test hindsight.
```
