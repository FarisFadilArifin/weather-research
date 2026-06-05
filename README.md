# Weather Forecast Calibration Research

Calibration-first Python research project for daily high-temperature forecast markets.

The target is the additive forecast bias:

```text
calibration_bias_f = actual_high_f - raw_forecast_high_f
calibrated_forecast_high_f = raw_forecast_high_f + predicted_calibration_bias_f
```

The current research horizon is `0h`, defined as the forecast available at `06:00` local station time on the contract date.

## Provider Labels

Provider labels are source-truthful:

- `gfs`: Mostly Right `forecast_nwp(model="gfs")`
- `hrrr`: Mostly Right `forecast_nwp(model="hrrr")`
- `nbm`: archived NOAA NBM/TMAX data, including legacy rows previously stored in `nws_forecast_snapshots.csv`
- `nws`: exact captured weather.gov/NWS forecasts only

Do not treat NBM, MOS, Open-Meteo, or any other proxy as exact `nws`.

## Install

```powershell
cd D:\dev\weather-research
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The Mostly Right NWP backfill uses `mostlyrightmd-weather[nwp]`, which depends on GRIB tooling (`cfgrib`, `xarray`, `eccodes`, `ecmwflibs`).

## Commands

Build normalized calibration samples from local actuals, local NBM, exact NWS captures if present, and any Mostly Right backfill cache:

```powershell
python -m src.build_calibration_dataset --project-root .
```

Backfill missing GFS/HRRR 0h rows through Mostly Right:

```powershell
python -m src.backfill_mostlyright_nwp --project-root . --models hrrr gfs
```

Train walk-forward baselines from the CLI:

```powershell
python -m src.train_calibration_models --calibration-dir data/calibration --skip-ml
```

Run the ML comparison in the notebook:

```text
notebooks/calibration_ml_walkforward.ipynb
```

The notebook writes `ml_results.csv`, refreshes `recommended_calibration_rules.csv`, and rewrites `calibration_report.md`.

Run station-level 3-provider stacking notebooks:

```text
notebooks/station_stacking/stacking_KATL.ipynb
notebooks/station_stacking/stacking_KAUS.ipynb
...
```

These notebooks build one wide row per station/date from same-day 11am GFS, HRRR, and NBM provider caches, compare raw-provider baselines, then run XGBoost, LightGBM, CatBoost, and a Ridge stacking model once all three providers have overlapping rows.

Write the Markdown report:

```powershell
python -m src.write_calibration_report --calibration-dir data/calibration
```

Fast local-only run, useful before the Mostly Right full backfill completes:

```powershell
python -m src.build_calibration_dataset --project-root . --providers nbm
python -m src.train_calibration_models --calibration-dir data/calibration --skip-ml
python -m src.write_calibration_report --calibration-dir data/calibration
```

## Outputs

New research outputs are written only under `data/calibration/`:

- `calibration_samples.csv`
- `baseline_results.csv`
- `ml_results.csv`
- `recommended_calibration_rules.csv`
- `calibration_report.md`
- `mostlyright_nwp_0h_cache.csv` when backfill is run

Existing `data/raw`, `data/processed`, and legacy `data/outputs` files are read as inputs and are not overwritten by the calibration commands.

## Leakage Controls

- `forecast_as_of` is always `06:00` local station time converted to UTC.
- Walk-forward evaluation trains only on earlier `contract_date` values.
- Rolling bias and station/month climatology features use shifted historical values only.
- Exact `nws` rows are emitted only from explicit local weather.gov capture files.

## Tests

```powershell
python -m pytest
```
