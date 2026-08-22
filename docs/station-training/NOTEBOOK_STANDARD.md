# Station Training Notebook Standard

This standard applies to
`notebooks/station_training_baseline/stations/{STATION}/train_{STATION}.ipynb`.

## Source of truth

Notebooks are generated from the station JSON configs and
`generate_station_notebook.py`. Structural changes must be made in the
generator, shared implementation, or config before regenerating notebooks.

Station codes are canonical in directory, config, notebook, artifact, and
model identities: `KDAL`, `RJTT`, `RKSI`, and `RKPK`.

## Required stages

1. Environment and station contract.
2. Point-in-time 11 AM feature construction.
3. Single-XGBoost chronological tuning and validation.
4. Frozen evaluation point-model export.
5. Conditional Gaussian residual probability training and validation.
6. Native-reference, blended, shared-slope, and pure ordinal training and
   validation.
7. Canonical two-of-three vote and median selected-bucket ensemble evaluation.
8. Gaussian, candidate, ensemble, agreement, gate, and monthly reports.
9. Separately named production-candidate refits and manifests.
10. Artifact hash and point-model dependency checks.

## Point model

- The only learned point model is XGBoost.
- Optuna uses 100 total trials and 40 random startup trials through TPE.
- Every fold fits preprocessing, feature eligibility, and XGBoost only on its
  training history.
- The 3% feature-missingness ceiling is recomputed on the applicable fitting
  population.
- Production refits reuse the frozen evaluation feature contract.

## Probability models

All probability models consume honest, out-of-sample XGBoost point predictions.
They must never train against in-sample production predictions.

The Gaussian baseline models continuous residual mean and scale, calibrates a
scale multiplier chronologically, then integrates normal mass across native
market bucket boundaries.

The ordinal layer exports four candidates: a station-native reference, a full
cumulative ordinal blended with an empirical monthly residual prior, a compact
21-feature shared-slope ordinal, and a pure full cumulative ordinal. KDAL uses
2°F market aggregation with `<=-4` and `>=+4` tails. RJTT, RKSI, and RKPK use
native 1°C markets with `<=-3` and `>=+3` tails. Cumulative probabilities must
be monotonic and every exported market distribution must sum to one.

The native reference is comparison-only. Blended, shared-slope, and pure are
the voting members. All three must be available, two station-specific
confidence votes are required, and selected-bucket probability is their median.
The reference must not receive a vote unless a later promotion study changes
the frozen policy.

## Artifact roles

Evaluation artifacts are frozen to the configured pre-holdout training years.
Production candidates may refit on all completed labels, but probability
training still uses only chronological XGBoost predictions. Evaluation and
production artifacts have distinct model versions, directories, hashes, and
point-model dependencies.

Ordinal candidates are written under `ordinal_candidates/{evaluation,production}/{role}`.
The corresponding `ordinal_ensemble/evaluation_manifest.json` and
`production_manifest.json` bind all four candidate artifacts, identify the
three voting roles, and freeze the two-of-three policy.

A production candidate is not an approved live release. Dirty source,
unreviewed evidence, or a missing point-bundle hash keeps it unapproved.
