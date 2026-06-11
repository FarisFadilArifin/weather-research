# Notebooks

The current baseline notebook workflow is `station_stacking_v2`. The precipitation-feature experiments are `station_stacking_v4` and `station_stacking_v5`.

## Current

Use these for the active same-day 11 AM station-stacking research path:

```text
notebooks/station_stacking_v2/stacking_KATL_v2.ipynb
notebooks/station_stacking_v2/stacking_KAUS_v2.ipynb
notebooks/station_stacking_v2/stacking_KDAL_v2.ipynb
notebooks/station_stacking_v2/stacking_KHOU_v2.ipynb
notebooks/station_stacking_v2/stacking_KLAX_v2.ipynb
notebooks/station_stacking_v2/stacking_KLGA_v2.ipynb
notebooks/station_stacking_v2/stacking_KMIA_v2.ipynb
notebooks/station_stacking_v2/stacking_KORD_v2.ipynb
notebooks/station_stacking_v2/stacking_KSEA_v2.ipynb
```

These notebooks compare raw HRRR/GFS baselines with XGBoost, LightGBM, CatBoost, and the Ridge stacked meta-model.

## Precipitation Experiment

Use these for the SDK precipitation-feature training path:

```text
notebooks/station_stacking_v4/stacking_KATL_v4.ipynb
notebooks/station_stacking_v4/stacking_KAUS_v4.ipynb
notebooks/station_stacking_v4/stacking_KDAL_v4.ipynb
notebooks/station_stacking_v4/stacking_KHOU_v4.ipynb
notebooks/station_stacking_v4/stacking_KLAX_v4.ipynb
notebooks/station_stacking_v4/stacking_KLGA_v4.ipynb
notebooks/station_stacking_v4/stacking_KMIA_v4.ipynb
notebooks/station_stacking_v4/stacking_KORD_v4.ipynb
notebooks/station_stacking_v4/stacking_KSEA_v4.ipynb
```

These notebooks write artifacts to `data/calibration/station_stacking_v4`.

`notebooks/station_stacking_v5/` uses the same v4 feature engineering and 100/50 Optuna trial counts, but tunes Optuna against MAE instead of RMSE. It writes artifacts to `data/calibration/station_stacking_v5`.

## V6 Feature-Input Experiment

`notebooks/station_stacking_v6/` starts from the source-owned v5 feature engineering block, keeps the MAE Optuna setup, adds durable per-station Optuna SQLite storage, and includes the 11 AM observation trend columns when those cache fields are populated.

For v6, the authoritative training input contract is each station artifact:

```text
data/calibration/station_stacking_v6/{STATION}_feature_columns.csv
```

Those files list the categorical and numeric columns passed into the model pipeline. The matching `{STATION}_features.csv` file can be used to audit per-feature NaN percentages. Current v6 artifacts contain 237 candidate training inputs: 6 categorical and 231 numeric.

Do not infer the trained feature matrix only from constants such as `V6_FEATURE_COLUMNS`. The trainer removes numeric columns that are all-NaN for a fit, and the current historical cache leaves the four v6 observation trend fields empty, so those trend fields do not enter the saved v6 training feature list until the cache is repopulated with non-null values.

## Legacy / Reference

- `notebooks/station_stacking/`: older station-stacking notebooks retained for comparison.
- `notebooks/calibration_ml_walkforward.ipynb`: legacy additive-bias calibration workflow.
- `notebooks/exploratory_analysis.ipynb`: exploratory analysis and one-off inspection.

When adding new notebooks, use a short markdown cell at the top that states whether the notebook is current, experimental, or legacy.
