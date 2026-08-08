# HKO V20 GFS-Only No-Peak Notebook

This notebook adapts the KDAL V20 no-peak training design for Hong Kong:

- official HKO daily maximum labels;
- official HKO Headquarters 1-minute temperature, maximum-since-midnight, and humidity snapshots through 11 AM;
- free historical snapshots from the DATA.GOV.HK archive and live snapshots from HKO Open Data;
- a same-station quality contract between the HKO high-so-far and official HKO daily maximum;
- exact prior-day 18Z GFS forecasts as the only model provider;
- the known access-blocked period through 2021-03-23, followed by required uninterrupted GFS coverage;
- four equal-weight validation folds for 2022–2025 and a 2026 holdout;
- XGBoost, LightGBM, CatBoost, and the ridge stack;
- Fahrenheit-native training with Fahrenheit and Celsius reporting; and
- 1°C buckets using floor intervals (`31.2°C` belongs to the `31°C` bucket).

The HKO model contract does not use the inherited paired 2°F Polymarket brackets. Its exported
probability policy, notebook metrics, and bracket artifacts use 1°C intervals `[n, n+1)`. The
two-bucket score selects the predicted floor bucket and the bucket immediately below it.

The notebook intentionally excludes peak-timing features and structurally degenerate one-provider
ensemble features. IFS and ICON acquisition/import utilities remain available, but they are not part
of this model contract.

Regenerate the notebook without executing its cells:

```powershell
.\.venv\Scripts\python.exe notebooks\experiments\station_stacking_v20_hko_no_peak\generate_station_notebook.py
```

Refresh the official HKO observations and HKO-point GFS files before rerunning:

```powershell
python scripts\run_hong_kong_11am_pipeline.py observations-backfill --force
python scripts\run_hong_kong_11am_pipeline.py gfs-backfill --force
python scripts\run_hong_kong_11am_pipeline.py audit
```

When the generated notebook is run, its experiment artifacts and model bundle are written beneath:

```text
data/calibration/hong_kong_11am/models/v20_hko_no_peak
```
