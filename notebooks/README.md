# Notebooks

The current notebook workflow is `station_stacking_v2`.

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

## Legacy / Reference

- `notebooks/station_stacking/`: older station-stacking notebooks retained for comparison.
- `notebooks/calibration_ml_walkforward.ipynb`: legacy additive-bias calibration workflow.
- `notebooks/exploratory_analysis.ipynb`: exploratory analysis and one-off inspection.

When adding new notebooks, use a short markdown cell at the top that states whether the notebook is current, experimental, or legacy.
