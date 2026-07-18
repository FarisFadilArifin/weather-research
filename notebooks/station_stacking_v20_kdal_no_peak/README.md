# KDAL V20-Aligned No-Peak Experiment

This KDAL-only notebook keeps the V11 Settlement Fix feature contract while aligning the surrounding experiment with KATL V20:

- same-day 11 AM live-safe inputs;
- `v11_settlement_fix_temp` features and no V20 HRRR/NBM peak-timing features;
- Wunderground-only settlement targets;
- 3% train-fold feature-missingness gate;
- explicit `training_profile="v20_aligned"`, which owns four equal-weight base folds for 2022-2025 and three expanding ridge meta-folds for 2023-2025;
- XGBoost, LightGBM, and CatBoost base learners with the ridge stack;
- aggregate ridge selection by mean MAE, worst-fold MAE, then trial key, using the same selector for evaluation and export;
- MAE-based tuning and a 2026 test refit; and
- enabled export to `data/calibration/station_stacking_v20_kdal_no_peak/model_weights` after a successful full notebook run.

Regenerate the notebook with:

```powershell
.\.venv\Scripts\python.exe notebooks\station_stacking_v20_kdal_no_peak\generate_station_notebook.py
```

The experiment uses the existing `v11_settlement_fix_temp` feature version intentionally. A new feature version is unnecessary because this arm changes labels and validation policy, not feature engineering.

The complete cross-repository handoff and ready-to-paste prompt are maintained at:

```text
D:\dev\polymarket-weather-prediction\docs\KDAL_V20_NO_PEAK_HANDOFF_PROMPT.md
```
