# Weather Forecast Calibration Research

Python research pipeline for daily high-temperature forecast markets. The current deployment direction is a same-day 11 AM station-stacking v2 workflow that combines GFS, HRRR, current station observations, calendar features, lagged history, and notebook-v2 engineered features.

The observed target variable is the final station daily high:

```text
actual_high_f
```

The calibration label is the additive forecast bias derived from that target:

```text
calibration_bias_f = actual_high_f - raw_forecast_high_f
calibrated_forecast_high_f = raw_forecast_high_f + predicted_calibration_bias_f
```

In the station-stacking workflow, models predict `actual_high_f` directly. In the conservative bias-calibration workflow, rules and comparison models predict `calibration_bias_f` as an additive correction.

## Current Workflow

The main research workflow is:

- Timing mode: `same_day_11am`
- Forecast snapshot: 11:00 AM local station time
- Forecast window: valid times from 11:00 AM local through the end of the local day
- Target forecast: max forecast temperature over that remaining-day window
- Target variable: final observed station high, `actual_high_f`

Primary station-stacking v2 models:

- `xgboost`
- `lightgbm`
- `catboost`
- `ridge_stack` meta-model as the final stacked prediction

The older `0h` / 06:00 local calibration report path still exists as a conservative benchmark, especially for `hierarchical_shrinkage`, but it is not the current main station-stacking direction.

## Providers

Provider labels are source-truthful:

- `gfs`: Mostly Right `forecast_nwp(model="gfs")`
- `hrrr`: Mostly Right `forecast_nwp(model="hrrr")`
- `nbm`: direct NOAA NBM archive GRIB2 extraction
- `nws`: exact captured weather.gov/NWS forecasts only, if local capture files exist

Do not label NBM, MOS, Open-Meteo, or other proxies as exact `nws`.

## Stations

Current research stations:

- `KATL`: Atlanta/Hartsfield-Jackson Intl
- `KAUS`: Austin/Bergstrom Intl
- `KDAL`: Dallas/Love Field
- `KHOU`: Houston/Hobby Airport
- `KLAX`: Los Angeles Intl
- `KLGA`: New York/La Guardia Airport
- `KMIA`: Miami Intl
- `KORD`: Chicago/O'Hare Intl
- `KSEA`: Seattle-Tacoma Intl

Use the resolved airport station code, not only a city name from a market title.

## Feature Rules

Forecast features must be available from the selected model cycle and valid times known at the 11 AM forecast snapshot. Do not use observed final highs, observations outside the deployment observation window, or same-day final actual-derived summaries as forecast features.

Current-observation features are allowed only when the observation timestamp is inside this deployment window:

```text
10:50 AM local <= observed_at <= 11:10 AM local
```

These include fields such as observed temperature, dew point, humidity, wind, pressure, visibility, ceiling, cloud cover, weather code, raw METAR lineage, and observation timing. For deployment, run inference at or after the window close, such as 11:10 or 11:15 local, so slightly delayed METAR reports can arrive without using observations after 11:10.

The 10:50-11:10 observation window is a widened production rule. Rebuild the station-stacking v2 artifacts and re-export model weights with that same rule before relying on post-11:00 observations.

Derived 11 AM observation features include:

- `observed_dewpoint_depression_f`
- `observed_heat_index_at_as_of_f`
- `observed_wind_chill_at_as_of_f`
- `observed_wind_dir_sin`
- `observed_wind_dir_cos`
- `observed_is_raining_at_as_of`
- `observed_is_fog_or_mist_at_as_of`
- `observed_is_thunder_at_as_of`

Audit fields such as raw METAR, observation timestamps, source text, and unavailable reasons are kept for lineage but should not be treated as ordinary numeric model features.

Station-stacking v2 adds notebook-level features:

- `v2_recent_heat_anomaly_f`
- `v2_recent_heat_momentum_f`
- `v2_morning_warmup_to_consensus_f`
- `v2_consensus_minus_7d_actual_f`
- `v2_spread_per_warmup_f`
- `v2_humidity_warmup_interaction`

## Install

```powershell
cd D:\dev\weather-research
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The Mostly Right NWP backfill uses `mostlyrightmd-weather[nwp]`, which depends on GRIB tooling such as `cfgrib`, `xarray`, `eccodes`, and `ecmwflibs`.

## Main Commands

Backfill 11 AM current observations:

```powershell
python -m src.backfill_mostlyright_current_observations --sdk-cache-dir data/calibration/sdk_current_obs_2021_2026 --stations $stations --start-date 2021-01-01 --end-date latest-complete --chunk-days 31 --retry-unavailable --request-retries 5 --retry-sleep-seconds 90 --sleep-between-chunks 8
```

Backfill 11 AM GFS/HRRR forecasts with weather features:

```powershell
python -m src.backfill_mostlyright_sdk_nwp --sdk-cache-dir data/calibration/sdk_11am_gfs_2025_2026 --stations $stations --models gfs --timing-mode same_day_11am --start-date 2025-01-01 --end-date latest-complete --include-weather-features --fxx-workers 3
python -m src.backfill_mostlyright_sdk_nwp --sdk-cache-dir data/calibration/sdk_11am_hrrr_2025_2026 --stations $stations --models hrrr --timing-mode same_day_11am --start-date 2025-01-01 --end-date latest-complete --include-weather-features --fxx-workers 3
```

Backfill direct NOAA NBM:

```powershell
python -m src.backfill_direct_nbm --sdk-cache-dir data/calibration/direct_nbm_2025_2026 --stations $stations --start-date 2025-01-01 --end-date latest-complete --include-weather-features
```

Run station-level stacking v2 notebooks:

```text
notebooks/station_stacking_v2/stacking_KATL_v2.ipynb
notebooks/station_stacking_v2/stacking_KAUS_v2.ipynb
...
```

Those notebooks patch in v2 feature engineering, call `src.calibration.station_stacking.run_station_year_split_experiment`, write wide station/date feature files, compare raw-provider baselines, tune XGBoost/LightGBM/CatBoost on fixed year splits, and train the Ridge stack.

Export station-stacking v2 model weights:

```powershell
.\.venv\Scripts\python.exe -m src.export_station_stacking_v2_models --project-root .
```

The exporter reads `data/calibration/station_stacking_v2` artifacts and writes deployable `.joblib` bundles plus JSON manifests. Use `--train-years 2021-2025` for strict 2026 holdout-style bundles; the default refits on all available completed actuals for production.

Legacy 6 AM calibration report commands:

```powershell
python -m src.build_calibration_dataset --project-root .
python -m src.train_calibration_models --calibration-dir data/calibration
python -m src.write_calibration_report --calibration-dir data/calibration
```

## Outputs

Station-stacking v2 outputs are written under `data/calibration/station_stacking_v2/`:

- `{STATION}_features.csv`
- `{STATION}_year_split_validation_predictions.csv`
- `{STATION}_year_split_test_predictions.csv`
- `{STATION}_year_split_metrics.csv`
- `{STATION}_year_split_scoreboard.csv`
- `{STATION}_year_split_selected_hyperparameters.csv`
- `{STATION}_feature_columns.csv`

Exported model weights are written under `data/calibration/station_stacking_v2/model_weights/`:

- `{STATION}_station_high_regressor_v2.joblib`
- `{STATION}_station_high_regressor_v2.json`
- `station_high_regressor_v2_index.csv`

Legacy calibration outputs are also under `data/calibration/`:

- `calibration_samples.csv`
- `baseline_results.csv`
- `ml_results.csv`
- `recommended_calibration_rules.csv`
- `calibration_report.md`

Large local data, logs, generated caches, GRIB files, and `.venv/` are intentionally ignored by Git.

## Guardrails

- Never use future actuals when simulating historical decisions.
- `actual_high_f` is the final target, not an input feature for same-day predictions.
- Forecast features must be available at or before `forecast_as_of`.
- 11 AM deployment observations must have timestamps from `10:50 AM local` through `11:10 AM local`.
- HRRR and GFS lineage must remain separate in the v2 deployment model.
- Direct NOAA NBM must not be labeled as SDK NBM or exact NWS.
- Treat active-market actuals as unavailable until the local day is complete and observations are finalized.
- Treat current ML and stacking outputs as research until training validation is finalized.

## Tests

```powershell
python -m pytest
```
