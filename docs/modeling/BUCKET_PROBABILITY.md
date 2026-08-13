# Rounded-Degree Bucket Probability Models

The probability layer is trained only from out-of-sample point-model predictions. It predicts the
rounded settlement-degree residual in nine classes (`<=-4`, `-3` through `+3`, `>=+4`) and exports
an immutable station artifact plus manifest. KATL may use V20 Peak Timing features; KDAL must use
the V20-aligned No-Peak pipeline.

KATL training:

```powershell
.\.venv\Scripts\python.exe scripts\train-bucket-probability.py `
  --station KATL `
  --pipeline-dir data\calibration\station_stacking_v20_peak_timing `
  --point-bundle data\calibration\station_stacking_v20_peak_timing\model_weights\KATL_station_high_regressor_v20_peak_timing_stack.joblib `
  --point-model-version station_high_regressor_v20_peak_timing_stack `
  --output-dir data\calibration\station_bucket_probability\KATL `
  --include-peak-features
```

KDAL training (only after its No-Peak notebook has produced the feature, validation, test, and
point-model artifacts):

```powershell
.\.venv\Scripts\python.exe scripts\train-bucket-probability.py `
  --station KDAL `
  --pipeline-dir data\calibration\station_stacking_v20_kdal_no_peak `
  --point-bundle data\calibration\station_stacking_v20_kdal_no_peak\model_weights\KDAL_station_high_regressor_v20_kdal_no_peak_stack.joblib `
  --point-model-version station_high_regressor_v20_kdal_no_peak_stack `
  --output-dir data\calibration\station_bucket_probability\KDAL
```

The command writes forward 2024/2025 predictions, tuning results, a strict 2026 holdout report,
and the final research artifact. Export does not authorize production activation. Production must
first run in shadow and pass the station-specific evidence gates.

Optional live-safe inputs are median-imputed and each has an explicit `__missing` indicator.
Mandatory point/base predictions, provider highs, current/high-so-far observations, and observation
timing fail closed. An artifact exported from a dirty source tree is research-only; commit the
implementation and re-export from that clean commit before configuring production shadow mode.

The `v20_aligned` point-model profile also drops training rows missing any configured provider's
11 AM `*_forecast_temp_at_as_of_f`. Production requires all three of those GFS/HRRR/NBM values,
so a model intended for routing must be trained and evaluated on the same serveable population.
The probability-frame builder likewise excludes rows missing any mandatory runtime input or any
OOF point/base prediction before fitting or scoring.

Probability exports are bound to the exact point-bundle SHA-256. Any point-model retrain makes the
older probability artifact stale even when its own files are unchanged. A production candidate
must be re-exported from a clean research commit, reference the final point bundle, pass historical
acceptance, and then accumulate a fresh live-contract-mapped shadow report. The currently failed or
dirty provisional exports are evidence only and are not deployable weights.

## Two-stage bucket-correction challenger

The correction challenger leaves the point bucket unchanged unless both stages support an
adjacent-bucket override:

```text
point model -> P(point bucket is wrong) -> lower/same/upper -> stable override gate
```

Its policy requires at least ten forward switches, at least three in both 2024 and 2025, positive
switch lift over direct rounding in each year, and no reduction in full forward bucket accuracy.
If those conditions are not met, the artifact records `stable_forward_evidence = false`.

Train it with the same station arguments as the probability model, replacing the script and output
directory, for example:

```powershell
.\.venv\Scripts\python.exe scripts\train-bucket-correction.py `
  --station KDAL `
  --pipeline-dir data\calibration\station_stacking_v20_kdal_no_peak `
  --point-bundle data\calibration\station_stacking_v20_kdal_no_peak\model_weights\KDAL_station_high_regressor_v20_kdal_no_peak_stack.joblib `
  --point-model-version station_high_regressor_v20_kdal_no_peak_stack `
  --output-dir data\calibration\station_bucket_correction\KDAL
```

The challenger is research-only unless its forward stability and untouched 2026 holdout gates both
pass. A failed challenger is not added to the production router.
