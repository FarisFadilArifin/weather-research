# KDAL V21 1 PM Bucket Challenger

This research-only challenger adds a bucket-probability layer to the completed V20 KDAL 1 PM no-peak point model.

Contract:

- station: `KDAL`
- timing: `same_day_1pm_live_safe`
- point feature version: `v20_kdal_1pm_no_peak`
- bucket feature profile: `kdal_1pm`
- point target: remaining warmup, reconstructed to final high
- bucket target: rounded actual degree offset from the honest OOF ridge point prediction
- development: strict forward 2024 and 2025 folds
- report-only holdout: 2026
- peak-timing and `v11sf` 11 AM alignment features: prohibited

Regenerate the notebook:

```powershell
.\.venv\Scripts\python.exe notebooks\experiments\station_stacking_v21_kdal_1pm_bucket\generate_notebook.py
```

Train from the command line:

```powershell
.\.venv\Scripts\python.exe scripts\train-bucket-probability.py `
  --station KDAL `
  --pipeline-dir data\calibration\station_stacking_v20_kdal_1pm_no_peak `
  --point-bundle data\calibration\station_stacking_v20_kdal_1pm_no_peak\model_weights\KDAL_station_high_regressor_v20_kdal_1pm_no_peak_stack.joblib `
  --point-model-version station_high_regressor_v20_kdal_1pm_no_peak_stack `
  --model-version station_bucket_v21_kdal_1pm `
  --feature-profile kdal_1pm `
  --output-dir data\calibration\station_stacking_v21_kdal_1pm_bucket
```

Audit:

```powershell
.\.venv\Scripts\python.exe scripts\audit_v21_kdal_1pm_bucket.py
```

The artifact must remain research-only whenever `historical_acceptance.passed` is false.
