# Continuous Residual Probability Challenger

This research-only challenger leaves the existing nine-class rounded-offset model unchanged. Its
target is the unrounded residual `actual_high_f - point_prediction_f`. It fits a complete latent
temperature distribution, truncates it at the live observed-high physical floor, integrates each
whole-degree settlement cell, and aggregates the resulting degree masses with the existing
canonical two-degree bucket helper.

## Target and settlement semantics

The KATL and KDAL feature files use finalized `actual_high_f` settlement labels. The evaluation
script audits whether any labels are fractional; it never jitters integer targets. The repository's
`round_half_up` uses decimal `ROUND_HALF_UP`. In the positive Fahrenheit range, reported degree `d`
therefore receives continuous mass over `[d - 0.5, d + 0.5)`. Exact endpoints have zero probability
under every implemented continuous candidate.

The numerical degree support is selected by distribution quantiles at a `1e-12` tail tolerance.
Any remaining numerical tail mass is attached to the corresponding edge cell, so integration never
loses mass. The observed-high floor is applied as a truncation at the lower edge of its rounded
settlement degree and the distribution is renormalized before degree integration.

## Models and calibration

- `seasonal_empirical`: month-weighted kernel residual climatology, used as the simple baseline.
- `conditional_empirical`: shrinkage-weighted empirical residuals conditioned on month, rounding-
  boundary band, and provider spread.
- `gaussian`: regularized Ridge location and log-scale regression (EMOS/NGR-style benchmark).
- `student_t`: the same deterministic location-scale regression with five-degree-of-freedom tails.

Each outer 2024/2025 fold uses only earlier dates. The last 90 days of earlier history form a
strictly prior calibration window; a conservative scale/bandwidth multiplier is selected by CRPS.
The selected multiplier is part of the serialized bundle contract, and bundle prediction applies
it before settlement integration; export verifies the saved bundle against a generated evaluation
row. Manifests use strict JSON, representing unavailable metrics as `null` rather than `NaN`.
The point predictions are the existing honest year-split/cross-fitted Ridge-stack predictions.
Every prediction records model and calibration cutoffs, and the script asserts they precede its
contract date. The 2026 output is explicitly exploratory and never participates in selection.

Selection uses combined 2024-2025 evidence in this order: bucket log loss, continuous CRPS, bucket
Brier score, calibration error, then simpler family. Outputs include full serialized degree and
bucket probabilities, continuous scores, interval coverage, PIT histograms, bucket reliability,
boundary-distance diagnostics, all-serveable comparisons, and common-date comparisons with direct
rounding and the existing nine-class system. Log loss alone clips probability at `1e-12`.

## Reproduction

```powershell
.\.venv\Scripts\python.exe scripts\train-continuous-residual-probability.py `
  --station KATL `
  --pipeline-dir data\calibration\station_stacking_v20_peak_timing `
  --point-bundle data\calibration\station_stacking_v20_peak_timing\model_weights\KATL_station_high_regressor_v20_peak_timing_stack.joblib `
  --point-model-version station_high_regressor_v20_peak_timing_stack `
  --output-dir data\calibration\station_continuous_residual_probability\KATL `
  --include-peak-features

.\.venv\Scripts\python.exe scripts\train-continuous-residual-probability.py `
  --station KDAL `
  --pipeline-dir data\calibration\station_stacking_v20_kdal_no_peak `
  --point-bundle data\calibration\station_stacking_v20_kdal_no_peak\model_weights\KDAL_station_high_regressor_v20_kdal_no_peak_stack.joblib `
  --point-model-version station_high_regressor_v20_kdal_no_peak_stack `
  --output-dir data\calibration\station_continuous_residual_probability\KDAL
```

Important limitations: samples are station-specific and modest; empirical conditioning is
deliberately low-capacity; kernel CRPS is deterministic numerical approximation; 2026 was already
examined in adjacent policy research; and neither a selected artifact nor favorable metrics grant
trading or production authorization.
