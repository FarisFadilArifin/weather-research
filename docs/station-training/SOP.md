# Station Training SOP

## 1. Confirm the station contract

Review `configs/{STATION}.json`: station identity, timezone, providers,
settlement source, observation source, local 11 AM cutoff, native bucket unit,
chronological folds, and model versions must all be explicit.

KDAL uses GFS/HRRR/NBM and 2°F markets. RJTT, RKSI, and RKPK use
GFS/GEFS/JMA-MSM and native 1°C markets. Do not substitute providers or station
targets across these contracts.

## 2. Audit point-in-time data

Before training, verify that forecasts and observations were available at the
configured cutoff, settlement truth is finalized, station/date keys are
unique, and no final-high-derived field is used as an inference feature.

## 3. Generate the notebook

```powershell
python notebooks\station_training_baseline\generate_station_notebook.py `
  --config notebooks\station_training_baseline\configs\{STATION}.json
```

Use `--all` to regenerate KDAL, RJTT, RKSI, and RKPK together.

## 4. Run static checks

```powershell
python -m pytest tests\test_station_training_baseline.py
```

Confirm the notebook contains one XGBoost point model, the Gaussian benchmark,
four ordinal candidates, the three-member ensemble, 100 Optuna trials, 40
startup trials, and no point ensemble.

## 5. Execute top to bottom

```powershell
python scripts\execute_notebook_cells.py `
  notebooks\station_training_baseline\stations\{STATION}\train_{STATION}.ipynb
```

Optuna storage is persistent per station and resumes until 100 completed or
failed trials exist. Do not reuse a study whose station, feature, target,
metric, or search-space identity differs.

## 6. Review evidence

Review point MAE/RMSE and the Gaussian, four-candidate, and ensemble forward and
holdout comparison. Review member correlation, bucket agreement, two-of-three
gate coverage, monthly stability, and whether the native reference duplicates
the pure member. Probability promotion is based on log loss, multiclass Brier,
calibration, and stability across chronological periods—not P&L alone.

Check that every probability distribution sums to one and that probability
training dates precede their validation dates.

## 7. Verify artifacts

For each station, inspect:

```text
model_weights/evaluation/
model_weights/production/
probability/gaussian/evaluation/
probability/gaussian/production/
ordinal_candidates/evaluation/native_ordinal_reference/
ordinal_candidates/evaluation/blended_ordinal/
ordinal_candidates/evaluation/shared_slope_ordinal/
ordinal_candidates/evaluation/pure_ordinal/
ordinal_candidates/production/native_ordinal_reference/
ordinal_candidates/production/blended_ordinal/
ordinal_candidates/production/shared_slope_ordinal/
ordinal_candidates/production/pure_ordinal/
ordinal_ensemble/evaluation_manifest.json
ordinal_ensemble/production_manifest.json
reports/
```

Every loadable artifact needs a JSON manifest, matching SHA-256, source
identity, training cutoff, feature contract, role, and exact point-bundle
dependency.

Production artifacts emitted from a dirty or unreviewed source state are
candidates only. Promote them in a separate clean-commit review; never relabel
an inspected holdout as fresh out-of-sample evidence.
