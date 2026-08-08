# Weather Research Project Context

## Short Description

This project is a Python research pipeline for calibrating daily high-temperature forecasts for weather-market decisions. It compares raw forecast highs against observed airport-station highs, learns the historical forecast bias, and outputs an additive calibration value that can be used to adjust a raw forecast before turning it into market bucket probabilities.

The core target is:

```text
calibration_bias_f = actual_high_f - raw_forecast_high_f
calibrated_forecast_high_f = raw_forecast_high_f + predicted_calibration_bias_f
```

All temperature outputs are in Fahrenheit.

## Current Model Status

The project has two model tracks:

- Station-stacking v2 ML: `xgboost`, `lightgbm`, and `catboost` base models with a `ridge_stack` meta-model. This is the main deployment direction for predicting `actual_high_f` directly from GFS, HRRR, current observations, calendar features, lagged history, and v2 notebook-engineered features.
- Bias-calibration benchmark: `hierarchical_shrinkage`, a walk-forward historical bias calibration rule. This is the safer recommendation in the current generated calibration report.

For UI context, describe the project as using station-stacking v2 with XGBoost, LightGBM, CatBoost, and a Ridge stack. Hierarchical shrinkage remains a conservative calibration benchmark for the older additive-bias path.

## Conservative Calibration Rule

`hierarchical_shrinkage` estimates an additive forecast correction by blending historical bias at several levels:

```text
global provider bias
-> provider/model/timing bias
-> station/provider/model/timing bias
-> station/provider/model/month bias
-> station/provider/model/horizon/month bias
```

Groups with fewer samples are pulled back toward broader, more stable groups. This avoids overtrusting small station-month samples.

Current generated report summary:

- Dataset: `7,895` calibration samples.
- Stations: `9`.
- Providers: `gfs`, `hrrr`, `nbm`.
- Date range: `2022-01-01` to `2026-05-18`.
- Current report horizon: `0h`, forecast available at `06:00` local station time.
- Best baseline in the report: `global_provider_mean_bias`, MAE `2.506F` vs raw `2.650F`.
- Recommended rule: `hierarchical_shrinkage`, MAE `2.542F` vs raw `2.650F`.
- Best ML comparison: `elasticnet`, MAE `2.617F` vs raw `2.650F`.

The report recommends `hierarchical_shrinkage` because the ML comparison did not clear the robustness threshold for live trading calibration.

## ML Models In The Codebase

Primary station-stacking v2 ML models:

- `xgboost`
- `lightgbm`
- `catboost`
- `ridge_stack`

The station-level stacking workflow is for same-day 11 AM forecasts. The v2 notebooks write artifacts to `data/calibration/station_stacking_v2`, add notebook-level engineered features, tune `xgboost`, `lightgbm`, and `catboost` on fixed year splits, and use a `ridge_stack` meta-model as the final stacked prediction.

The deployable model weights are exported from the v2 artifacts:

```text
data/calibration/station_stacking_v2/model_weights/{STATION}_station_high_regressor_v2.joblib
data/calibration/station_stacking_v2/model_weights/{STATION}_station_high_regressor_v2.json
data/calibration/station_stacking_v2/model_weights/station_high_regressor_v2_index.csv
```

Each `.joblib` bundle contains:

- base models: `xgboost`, `lightgbm`, `catboost`
- final stack model: `ridge_stack`
- `stack_features`
- categorical and numeric feature names
- target: `actual_high_f`
- providers: `gfs`, `hrrr`

Stack feature names:

```text
xgboost_predicted_high_f
lightgbm_predicted_high_f
catboost_predicted_high_f
hrrr_raw_predicted_high_f
gfs_raw_predicted_high_f
```

Additional calibration ML comparison models:

- `ridge`
- `elasticnet`
- `random_forest`
- `extra_trees`
- `hist_gradient_boosting`

Treat these ML and stacking outputs as research artifacts until the training/evaluation process is finalized.

## Forecast Providers

Provider labels are source-truthful:

- `gfs`: Mostly Right `forecast_nwp(model="gfs")`.
- `hrrr`: Mostly Right `forecast_nwp(model="hrrr")`.
- `nbm`: NOAA NBM archive data for experiments and legacy comparison paths.
- `nws`: exact captured weather.gov/NWS forecasts only, if local capture files exist.

Do not treat NBM, MOS, Open-Meteo, or other proxies as exact `nws`.

## Stations

Current expanded research stations:

- `KATL`: Atlanta/Hartsfield-Jackson Intl
- `KAUS`: Austin/Bergstrom Intl
- `KDAL`: Dallas/Love Field
- `KHOU`: Houston/Hobby Airport
- `KLAX`: Los Angeles Intl
- `KLGA`: New York/La Guardia Airport
- `KMIA`: Miami Intl
- `KORD`: Chicago/O'Hare Intl
- `KSEA`: Seattle-Tacoma Intl

Station code matters. UI flows should use the resolved airport station, not only the city name from a market title.

## Main Inputs

Primary data inputs:

- Actual daily highs: `data/processed/actual_highs.csv`.
- Station metadata: `data/processed/station_registry.csv`.
- GFS/HRRR forecasts: Mostly Right NWP cache rows for the v2 deployment model.
- Current observations: timestamped station observations from the 10:50-11:10 AM local deployment window.
- NBM forecasts: direct NOAA NBM archive/TMAX or direct GRIB2 extraction for experiments and legacy comparisons.
- Optional exact NWS forecasts: weather.gov capture files, only if present.
- Optional Polymarket metadata: parsed daily high-temperature market events and bucket labels.

## Main Outputs

Primary calibration outputs are under `data/calibration/`:

- `calibration_samples.csv`: normalized station/date/provider training rows.
- `baseline_results.csv`: walk-forward baseline metrics.
- `ml_results.csv`: walk-forward ML comparison metrics.
- `recommended_calibration_rules.csv`: current additive calibration rules.
- `calibration_report.md`: generated research summary.
- `mostlyright_nwp_0h_cache.csv`: Mostly Right backfill cache when run.

Station stacking v2 outputs are under `data/calibration/station_stacking_v2/`:

- `{STATION}_features.csv`
- `{STATION}_year_split_validation_predictions.csv`
- `{STATION}_year_split_test_predictions.csv`
- `{STATION}_year_split_metrics.csv`
- `{STATION}_year_split_scoreboard.csv`
- `{STATION}_year_split_selected_hyperparameters.csv`
- `{STATION}_feature_columns.csv`

Station stacking v2 model weights are under `data/calibration/station_stacking_v2/model_weights/`:

- `{STATION}_station_high_regressor_v2.joblib`
- `{STATION}_station_high_regressor_v2.json`
- `station_high_regressor_v2_index.csv`

## Important Fields For UI

Use these fields for UI views and API payloads:

- `station_id`: airport station code, for example `KSEA`.
- `station_name` / `airport_name`: display labels.
- `provider`: `gfs`, `hrrr`, `nbm`, or exact `nws` if available.
- `model`: provider model label, for example `gfs`, `hrrr`, `nbm`, or `nbm_tmax`.
- `timing_mode`: forecast timing contract, for example `strict_6am` or `same_day_11am`.
- `contract_date`: local target date.
- `forecast_as_of`: timestamp when the forecast is considered available.
- `issued_at`: source model cycle or issue timestamp.
- `raw_forecast_high_f`: uncalibrated forecast high.
- `actual_high_f`: observed final high, only available after the day completes.
- `calibration_bias_f`: `actual_high_f - raw_forecast_high_f`.
- `predicted_high_f`: v2 station-stacking prediction of final high.
- `predicted_calibration_bias_f`: model-predicted additive correction for the older calibration path.
- `calibration_add_f`: recommended rule value from `recommended_calibration_rules.csv`.
- `calibrated_forecast_high_f`: `raw_forecast_high_f + calibration_add_f`.
- `sample_count`: historical sample count for the rule.
- `valid_after_contract_date`: latest date used when writing the rule.
- `mae_before_f`, `mae_after_f`, `mae_improvement_f`: evaluation metrics.
- `within_1f_after_pct`, `within_2f_after_pct`, `within_3f_after_pct`: accuracy-band metrics.

## Suggested UI Payload

```json
{
  "stationId": "KSEA",
  "stationName": "Seattle-Tacoma Intl",
  "contractDate": "2026-06-04",
  "provider": "ridge_stack",
  "model": "station_high_regressor_v2",
  "timingMode": "same_day_11am",
  "forecastAsOf": "2026-06-04T18:00:00Z",
  "observationWindowLocal": {
    "start": "2026-06-04T10:50:00",
    "end": "2026-06-04T11:10:00"
  },
  "hrrrHighF": 76.0,
  "gfsHighF": 75.0,
  "observedTempAtAsOfF": 68.0,
  "observedOffsetMinutesFrom11am": -7,
  "modelName": "station_high_regressor_v2",
  "modelStatus": "exported_model_weights",
  "baseModels": ["xgboost", "lightgbm", "catboost"],
  "stackingModel": "ridge_stack",
  "benchmarkModel": "hierarchical_shrinkage",
  "predictedHighF": 75.6,
  "modelBundlePath": "data/calibration/station_stacking_v2/model_weights/KSEA_station_high_regressor_v2.joblib",
  "uncertainty": {
    "predictionIntervalF": [73.1, 78.0]
  },
  "lineage": {
    "forecastSource": "mostlyright.weather.forecast_nwp_hrrr_gfs",
    "observationRule": "latest_station_observation_between_1050_and_1110_local",
    "actualSource": "airport_station_actual_high",
    "modelSource": "station_stacking_v2",
    "benchmarkSource": "walk_forward_hierarchical_shrinkage"
  }
}
```

## Probability Output

The project can convert a calibrated forecast into bucket probabilities with `src.bucket_probs.bucket_probabilities`.

Inputs:

- `forecast_high_f`: raw forecast high.
- `error_mean_f`: calibration add value.
- `error_std_f`: historical uncertainty for that station/provider/month.
- `buckets`: market bucket labels such as `70-71`, `72-73`, or `74 or above`.

Output shape:

```json
{
  "70-71": 0.14,
  "72-73": 0.31,
  "74 or above": 0.55
}
```

## Useful UI Screens

Good UI surfaces for another project:

- Forecast calibration card: station, date, provider, raw high, calibration add, calibrated high, sample count, and confidence/uncertainty.
- Provider comparison: GFS vs HRRR raw highs, spread, issue time, forecast availability, and provider lineage.
- Backtest metrics dashboard: MAE before/after, bias before/after, within 1F/2F/3F rates, filterable by station, provider, month, and timing mode.
- Bucket probability view: calibrated forecast distribution across market buckets.
- Rule table: `recommended_calibration_rules.csv` grouped by station/provider/month.
- Lineage/audit panel: source files, forecast cycle, forecast-as-of timestamp, and leakage guardrails.
- Model status panel: show `station_high_regressor_v2`, base models as `xgboost` + `lightgbm` + `catboost`, stack model as `ridge_stack`, and benchmark rule as `hierarchical_shrinkage`.

## Guardrails

- Never use future actuals when simulating historical decisions.
- Forecast features must be available at or before `forecast_as_of`.
- Current-observation features for 11 AM deployment must be observed from 10:50 AM local through 11:10 AM local.
- The 10:50-11:10 observation rule requires station-stacking v2 artifacts and model weights rebuilt with the same observation feature contract before production use.
- Do not silently label NBM as exact NWS.
- Do not infer station from city text alone; use the resolved airport station.
- Treat active-market actuals as unavailable until the local day is complete and observations are finalized.
- Load station-specific v2 model weights; do not use a model bundle trained for a different station.

## Refresh Commands

Build calibration samples:

```powershell
python -m src.build_calibration_dataset --project-root .
```

Train/evaluate baselines and optional ML:

```powershell
python -m src.train_calibration_models --calibration-dir data/calibration
```

Write the report:

```powershell
python -m src.write_calibration_report --calibration-dir data/calibration
```

Fast baseline-only run:

```powershell
python -m src.build_calibration_dataset --project-root . --providers nbm
python -m src.train_calibration_models --calibration-dir data/calibration --skip-ml
python -m src.write_calibration_report --calibration-dir data/calibration
```

Export station-stacking v2 model weights:

```powershell
python -m src.export_station_stacking_v2_models --project-root .
```

Export strict 2021-2025 train-year bundles:

```powershell
python -m src.export_station_stacking_v2_models --project-root . --train-years 2021-2025
```
