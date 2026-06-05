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

- Station-stacking ML: `xgboost`, `lightgbm`, and `catboost` base models, with an optional `ridge_stack` meta-model. This is the main gradient-boosted ML direction for combining GFS, HRRR, and NBM signals at the station/date level.
- Bias-calibration benchmark: `hierarchical_shrinkage`, a walk-forward historical bias calibration rule. This is the safer recommendation in the current generated calibration report.

For UI context, describe the project as using gradient-boosted station stacking with XGBoost, LightGBM, and CatBoost, while keeping hierarchical shrinkage as the conservative calibration benchmark until full training validation is finalized.

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

Primary station-stacking ML models:

- `xgboost`
- `lightgbm`
- `catboost`

The station-level stacking workflow is for same-day 11 AM forecasts. It builds wide station/date rows from GFS, HRRR, and NBM features, compares raw-provider baselines, trains `xgboost`, `lightgbm`, and `catboost` base models, then uses a `ridge_stack` meta-model when all required prediction columns are available.

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
- `nbm`: NOAA NBM archive data, currently the selected NBM path.
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
- NBM forecasts: direct NOAA NBM archive/TMAX or direct GRIB2 extraction.
- GFS/HRRR forecasts: Mostly Right NWP cache rows.
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

Station stacking outputs are under `data/calibration/station_stacking/`:

- `{STATION}_features.csv`
- `{STATION}_predictions.csv`
- `{STATION}_metrics.csv`
- `{STATION}_feature_columns.csv`

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
- `predicted_calibration_bias_f`: model-predicted additive correction.
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
  "provider": "nbm",
  "model": "nbm",
  "timingMode": "same_day_11am",
  "forecastAsOf": "2026-06-04T18:00:00Z",
  "issuedAt": "2026-06-04T12:00:00Z",
  "rawForecastHighF": 76.0,
  "modelName": "xgboost_lightgbm_catboost_station_stack",
  "modelStatus": "station_stacking_research",
  "baseModels": ["xgboost", "lightgbm", "catboost"],
  "stackingModel": "ridge_stack",
  "benchmarkModel": "hierarchical_shrinkage",
  "calibrationAddF": 0.5,
  "calibratedForecastHighF": 76.5,
  "sampleCount": 120,
  "uncertainty": {
    "maeAfterF": 2.5,
    "within2FAfterPct": 60.0
  },
  "lineage": {
    "forecastSource": "direct_noaa_nbm_archive_grib2",
    "actualSource": "airport_station_actual_high",
    "modelSource": "station_stacking",
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
- Provider comparison: GFS vs HRRR vs NBM raw highs, spread, issue time, forecast availability, and provider lineage.
- Backtest metrics dashboard: MAE before/after, bias before/after, within 1F/2F/3F rates, filterable by station, provider, month, and timing mode.
- Bucket probability view: calibrated forecast distribution across market buckets.
- Rule table: `recommended_calibration_rules.csv` grouped by station/provider/month.
- Lineage/audit panel: source files, forecast cycle, forecast-as-of timestamp, and leakage guardrails.
- Research status panel: show station stacking as `xgboost` + `lightgbm` + `catboost`, meta-model as `ridge_stack`, and benchmark rule as `hierarchical_shrinkage`.

## Guardrails

- Never use future actuals when simulating historical decisions.
- Forecast features must be available at or before `forecast_as_of`.
- Current-observation features for 11 AM workflows must be observed at or before 11 AM local.
- Do not silently label NBM as exact NWS.
- Do not infer station from city text alone; use the resolved airport station.
- Treat active-market actuals as unavailable until the local day is complete and observations are finalized.
- Treat current ML and stacking outputs as research until the training process is finalized.

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
