# KDAL V24 No-Peak Diverse-Ensemble Experiment

This notebook is an ablation of KDAL V20 no-peak. It preserves the V20 data and validation
contract and changes only the base-model families:

- XGBoost: sequential gradient-boosted trees;
- Extra Trees: independently randomized, averaged trees; and
- scaled Ridge: a regularized linear base learner.

The Ridge stack remains enabled and may combine the three model predictions with raw GFS,
HRRR, and NBM forecasts. The experiment uses four expanding validation folds for 2022-2025,
MAE tuning, a held-out 2026 test, and exports to
`data/calibration/station_stacking_v24_kdal_no_peak_diverse_ensemble`.

Regenerate the notebook with:

```powershell
.\.venv\Scripts\python.exe notebooks\experiments\station_stacking_v24_kdal_no_peak_diverse_ensemble\generate_station_notebook.py
```
