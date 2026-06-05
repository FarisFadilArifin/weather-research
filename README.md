# Weather Forecast Calibration Research

Python research pipeline for daily high-temperature forecast markets. The current direction is a same-day 11 AM station-stacking workflow that combines GFS, HRRR, direct NOAA NBM, current station observations, calendar features, and lagged history.

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

Primary station-stacking models:

- `xgboost`
- `lightgbm`
- `catboost`
- optional `ridge_stack` meta-model

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

Forecast features must be available from the selected model cycle and valid times known at the 11 AM snapshot. Do not use observed final highs, post-11 AM observations, or same-day final actual-derived summaries as forecast features.

Current-observation features are allowed only when the observation timestamp is at or before 11:00 AM local. These include fields such as observed temperature, dew point, humidity, wind, pressure, visibility, ceiling, cloud cover, weather code, raw METAR lineage, and observation age.

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

Run station-level stacking notebooks:

```text
notebooks/station_stacking/stacking_KATL.ipynb
notebooks/station_stacking/stacking_KAUS.ipynb
...
```

Those notebooks call `src.calibration.station_stacking.run_station_stacking_experiment`, write wide station/date feature files, compare raw-provider baselines, train base models, and optionally train the Ridge stack.

Legacy 6 AM calibration report commands:

```powershell
python -m src.build_calibration_dataset --project-root .
python -m src.train_calibration_models --calibration-dir data/calibration
python -m src.write_calibration_report --calibration-dir data/calibration
```

## Outputs

Station-stacking outputs are written under `data/calibration/station_stacking/`:

- `{STATION}_features.csv`
- `{STATION}_predictions.csv`
- `{STATION}_metrics.csv`
- `{STATION}_feature_columns.csv`

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
- 11 AM current observations must have timestamps `<= 11:00 AM local`.
- HRRR, GFS, and NBM lineage must remain separate.
- Direct NOAA NBM must not be labeled as SDK NBM or exact NWS.
- Treat active-market actuals as unavailable until the local day is complete and observations are finalized.
- Treat current ML and stacking outputs as research until training validation is finalized.

## Tests

```powershell
python -m pytest
```
