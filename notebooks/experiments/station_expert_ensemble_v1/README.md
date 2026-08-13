# Station Expert Ensemble V1

This isolated, generator-backed experiment evaluates four deliberately different point experts for Dallas/KDAL, Seoul/RKSI, and Tokyo/RJTT:

1. `full_xgboost` — all fold-eligible live-safe numeric features; remaining-warmup target.
2. `forecast_huber` — forecast consensus, disagreement, lead-time, observation-gap, and historical provider-bias features; provider-consensus residual target; fixed five-knot splines.
3. `observation_catboost` — observations, morning movement, 11 AM forecast gaps, and weather/solar physics; remaining-warmup target; no provider final-high anchors or target-history families.
4. `seasonal_ridge` — periodic day-of-year spline and lagged/rolling/trend high history; direct final-high target.

The point prediction is a non-negative four-way simplex blend. Its grid step is 0.025, zero weights are legal, and selection first shortlists candidates within 0.01°F of the best equal-fold mean MAE. The shortlist is ordered by worst-fold MAE, distance from equal weighting, and deterministic method-order weights. Validation-year blend weights are learned only from earlier expert OOF years.

## Contracts

- KDAL providers: GFS, HRRR, NBM; evaluation refit through 2025; half-up 2°F buckets.
- RKSI/RJTT providers: GFS, GEFS, JMA MSM; evaluation refit through 2025; native whole-1°C half-up buckets.
- Every expert recomputes the 3% missingness gate inside every outer fit and on the exact final-refit population.
- Predictions are continuous Fahrenheit and floored at the observed high through the as-of time.
- The linked probability profiles contain 61 inputs and declare the four expert base methods explicitly.
- Point and probability bundles are research/shadow-only. Their manifests contain verified bundle SHA-256 values, and probability manifests bind the exact point bundle hash.
- The 2026 holdout is exploratory and cannot tune features, hyperparameters, blend weights, calibration, or policy.
- Live refitting is disabled. A future live path must use the frozen evaluation feature contracts and fail closed above 3% missingness.

The notebooks consume the canonical live-safe feature snapshots produced by the active station pipeline, but write only beneath `data/calibration/station_expert_ensemble_v1/{STATION}/`. They do not modify active baseline notebooks, configs, or bundles.

## Regeneration

```powershell
python notebooks/experiments/station_expert_ensemble_v1/generate_notebooks.py
```

The generator and files in `configs/` are the source of truth. Do not edit the generated notebooks directly.

## Execution

The default contract runs 30 Optuna trials with 15 startup trials for each XGBoost and observation CatBoost tuning fit. Execute each notebook in a clean kernel from the repository environment. Results and any environmental blockers are recorded in [AUDIT.md](AUDIT.md).
