# Regression Bucket Win Classifier

This research layer estimates the probability that the two-degree Polymarket bucket selected by
the existing point regression will settle. It does not replace the temperature regression or the
multiclass/continuous probability systems.

## Target and leakage controls

The binary target is `1` when the half-up rounded settlement temperature and the half-up rounded
point prediction map to the same canonical two-degree bucket. Point and base-model predictions in
training are forward, out-of-fold predictions. Label-derived rolling bias, MAE, and win-rate
features are shifted by one day. Model fitting, probability calibration, and validation are
strictly ordered:

1. Fit the classifier on history excluding the final 90 days.
2. Fit Platt or isotonic calibration on those final 90 prior days.
3. Evaluate on the following validation year.

Candidate selection uses combined 2024-2025 binary log loss, then Brier score, calibration error,
and model simplicity. The exported final model is fitted through the pre-calibration portion of
2025 and calibrated on the final 90 days of 2025. Results on 2026 are exploratory only.

KATL uses a compact set of live-safe v20 peak-timing variables. KDAL uses the v20 no-peak pipeline
and the code rejects attempts to add peak variables. Both stations compare regularized logistic
regression with shallow, strongly regularized CatBoost. The existing continuous-residual probability
for the regression-selected bucket is reported as a baseline when supplied; it is deliberately not
joined as a training feature because honest versions are not available for every earlier fold.

## Reproduction

```powershell
.\.venv\Scripts\python.exe scripts\train-win-classifier.py `
  --station KATL `
  --pipeline-dir data\calibration\station_stacking_v20_peak_timing `
  --point-bundle data\calibration\station_stacking_v20_peak_timing\model_weights\KATL_station_high_regressor_v20_peak_timing_stack.joblib `
  --point-model-version station_high_regressor_v20_peak_timing_stack `
  --continuous-forward data\calibration\station_continuous_residual_probability\KATL\KATL_forward_continuous_predictions.csv `
  --continuous-holdout data\calibration\station_continuous_residual_probability\KATL\KATL_2026_exploratory_predictions.csv `
  --output-dir data\calibration\station_win_classifier\KATL `
  --include-peak-features

.\.venv\Scripts\python.exe scripts\train-win-classifier.py `
  --station KDAL `
  --pipeline-dir data\calibration\station_stacking_v20_kdal_no_peak `
  --point-bundle data\calibration\station_stacking_v20_kdal_no_peak\model_weights\KDAL_station_high_regressor_v20_kdal_no_peak_stack.joblib `
  --point-model-version station_high_regressor_v20_kdal_no_peak_stack `
  --continuous-forward data\calibration\station_continuous_residual_probability\KDAL\KDAL_forward_continuous_predictions.csv `
  --continuous-holdout data\calibration\station_continuous_residual_probability\KDAL\KDAL_2026_exploratory_predictions.csv `
  --output-dir data\calibration\station_win_classifier\KDAL
```

`predict_win_bundle` returns the selected bucket and calibrated winning probability. With a market
implied probability it also returns the estimated probability edge and only marks `bet` when that
edge exceeds the requested minimum. Model acceptance remains research-only and requires at least a
0.002 out-of-time log-loss improvement and no worse calibration error than every supplied baseline.
