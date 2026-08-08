# Station Training Notebook Standard

This document defines the structure and artifact contract for notebooks under
`notebooks/station_training_baseline/stations/{STATION}/`. The KDAL notebook is
the reference implementation. `SOP.md` describes the procedure for creating and
running one.

## Source of truth

A station notebook is generated, not maintained as an independent hand-edited
copy:

```text
station config
  + station-specific point-notebook generator
  + station-training generator
  -> stations/{STATION}/train_{STATION}.ipynb
```

Change `generate_station_notebook.py`, the station config, or the referenced
station-specific generator first. Regenerate the notebook and commit both the
source change and generated `.ipynb`.

## Required notebook order

Every station notebook must keep these stages in order:

1. Environment and station constants.
2. Data-source, settlement-target, provider, and inference-cutoff contracts.
3. Feature construction and live-availability gates.
4. Base point-model training.
5. Honest chronological point-stack predictions.
6. Point-model metrics and diagnostic holdout analysis.
7. Point-model weight and manifest export.
8. Pure ordinal probability baseline training.
9. Chronological probability validation and exploratory holdout scoring.
10. Pure ordinal weight and manifest export.
11. Any approved station-specific challenger stage.
12. Challenger weight, manifest, prediction, comparison, and summary export.
13. In-notebook artifact-integrity assertions.

The ordinal stages must consume honest point predictions. They must never train
on in-sample point predictions.

## Standard naming

| Item | Standard |
|---|---|
| Notebook | `stations/{STATION}/train_{STATION}.ipynb` |
| Config | `configs/{STATION}.json` |
| Artifact root | `data/calibration/station_training_baseline/{STATION}/` |
| Point weights | `model_weights/*.joblib` plus matching `.json` |
| Pure ordinal outputs | `ordinal_probability/` |
| Challenger outputs | a named subdirectory such as `ordinal_challenger_v1/` |

Model versions must be station-specific, stable, lowercase identifiers. Every
loadable `.joblib` bundle must have a JSON manifest with:

- station and model version;
- point-model version and point-bundle SHA-256;
- feature names and preprocessing contract;
- selected family and hyperparameters;
- calibration/blend settings;
- training start, cutoff, and row count;
- holdout and promotion status; and
- SHA-256 of the loadable bundle.

The notebook must verify both the point-bundle binding and exported bundle hash
before completing.

## Chronology standard

For the current KDAL contract:

| Evaluation | Training history |
|---|---|
| 2024 forward validation | 2023 |
| 2025 forward validation | 2023–2024 |
| Final frozen models | 2023–2025 |
| 2026 exploratory holdout | frozen through 2025 |

The final 90 days inside each available training history form the inner
selection/calibration split. All preprocessing, model fitting, temperature
selection, blend selection, and policy selection happen inside the applicable
training history. A previously inspected holdout cannot become promotion
evidence.

New stations may use different calendar years only when their config and
documentation explicitly define an equivalent chronological contract.

For Seoul/RKSI and Tokyo/RJTT:

| Evaluation | Training history |
|---|---|
| 2025 probability forward validation | honest 2024 point-stack rows |
| Final frozen probability model | honest 2024–2025 point-stack rows |
| 2026 exploratory holdout | frozen through 2025 |

Their point workflow itself retains the expanding 2022→2023,
2022–2023→2024, and 2022–2024→2025 folds. Probability development begins in
2024 because the nested ridge stack needs an earlier point-validation year
before it can produce a leakage-free point prediction.

## Preprocessing standard

For ordinal logistic models, each chronological fold owns its preprocessing:

```text
median imputer -> standard scaler -> ordinal logistic model
```

Fit the imputer and scaler on the fold's training rows only. The current verified
contract does not use skew transforms, winsorization, or feature clipping.
Feature ablations select columns before fitting this pipeline.

## KDAL three-arm probability contract

KDAL must export exactly these roles, in this order:

1. `blended_ordinal`: independent cumulative logits blended with the empirical
   distribution, with model weight below `1.0`;
2. `shared_slope_ordinal`: the lower-variance shared-slope cumulative-logit
   family; and
3. `pure_ordinal`: independent cumulative logits with model weight `1.0`.

The best configuration for each role is selected by inner chronological
market-bucket log loss, with ranked probability score and offset log loss as
tie-breakers. Candidate search includes the 59-feature full contract and the
verified 27- and 21-feature ablations.

All three arms export `.joblib` weights and JSON manifests. Their probabilities
may be used for shadow evaluation, but the no-override policy keeps the V20 point
bucket as the recommendation.

## Seoul and Tokyo feature contract

Seoul and Tokyo use `asia_no_peak`, a 59-feature ordinal profile parallel to the
KDAL no-peak profile but with station-correct providers:

```text
gfs
gefs
jma_msm
```

The provider highs, provider-minus-point features, consensus statistics,
live-safe observations, calendar features, and optional missing indicators are
all computed at the Asia local 11 AM cutoff. HRRR and NBM must not appear in
these notebooks or probability artifacts.

Only the pure ordinal arm is currently enabled. The KDAL three-arm challenger is
not portable to Asia until a separate chronological challenger study is
implemented and approved.

Seoul and Tokyo are unit-contract exceptions to Ordinal Probabilities Model 2.
Their point models remain Fahrenheit-native, but each probability target is
`round_half_up(actual_high_c) - round_half_up(point_prediction_c)`, with
`round_half_up(value) = floor(value + 0.5)`. It fits the ordered classes
`<=-3, -2, -1, 0, +1, +2, >=+3` °C and exports exact whole-Celsius market
probabilities under `celsius_market_probability/`. All Seoul/Tokyo confidence fields
and decisions must be calculated from that distribution. The historical
rounded-Fahrenheit/2°F probability bundles remain comparison-only.

## Portability boundary

KDAL values are not universal defaults. Seoul and Tokyo demonstrate the required
provider/profile substitution. Before enabling a challenger for another station,
verify its providers, cutoff time, timezone, settlement source, feature
availability, unit conventions, market-bucket definition, and chronological
sample coverage. Until that work is complete, leave
`ordinal_challenger_enabled` false or absent in the station config.

## Acceptance checklist

A notebook meets this standard only when:

- regeneration is deterministic at the source/cell-contract level;
- all cells run top to bottom in a clean process;
- chronology and probability-sum assertions pass;
- expected row counts and missingness are reviewed;
- all required `.joblib` and `.json` files exist;
- manifest hashes match their bundles and point-model dependency;
- targeted tests pass; and
- all probability outputs remain marked exploratory/shadow-only unless a
  separate promotion review approves them.
