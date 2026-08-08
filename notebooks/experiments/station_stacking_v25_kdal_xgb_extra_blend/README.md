# KDAL V25 XGBoost + Extra Trees Constrained Blend

This experiment performs no base-model hyperparameter tuning. It reuses:

- the tuned XGBoost out-of-fold predictions and fitted model from KDAL V20 no-peak; and
- the tuned Extra Trees out-of-fold predictions and fitted model from KDAL V24.

The blend searches XGBoost weights from 0 to 1 in increments of 0.001. Extra Trees receives
the complementary weight, so both weights are non-negative and sum to one. Selection minimizes
the equal-weight mean MAE across the four expanding validation folds, with worst-fold MAE and
distance from an equal blend as deterministic tie breakers.

The 2026 holdout is evaluated only after weight selection. Outputs and the combined model bundle
are written to `data/calibration/station_stacking_v25_kdal_xgb_extra_blend`.

Regenerate the notebook with:

```powershell
.\.venv\Scripts\python.exe notebooks\experiments\station_stacking_v25_kdal_xgb_extra_blend\generate_station_notebook.py
```
