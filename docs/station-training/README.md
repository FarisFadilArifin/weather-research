# Station Training Baseline

This is the active, non-versioned station-training entry point. Use one notebook
per station. Do not add multiple stations to one notebook.

## Start here

| Purpose | Path |
|---|---|
| KDAL training notebook | `stations/KDAL/train_KDAL.ipynb` |
| KDAL station configuration | `configs/KDAL.json` |
| Seoul/RKSI training notebook | `stations/Seoul/train_Seoul.ipynb` |
| Seoul station configuration | `configs/Seoul.json` |
| Tokyo/RJTT training notebook | `stations/Tokyo/train_Tokyo.ipynb` |
| Tokyo station configuration | `configs/Tokyo.json` |
| Notebook structure and artifact contract | `NOTEBOOK_STANDARD.md` |
| New-station and retraining procedure | `SOP.md` |
| Current-pipeline research notebook procedure | `RESEARCH_NOTEBOOK_GUIDE.md` |
| Full pipeline, Celsius/Fahrenheit, bucket, and 3% missingness guide | `TEMPERATURE_UNITS_AND_BUCKETS.md` |
| Ordinal Model 2 contract | `ORDINAL_MODEL_2.md` |
| Immutable release provenance gate | `RELEASE_PROVENANCE.md` |
| Live point-model notebook lineage | `LIVE_MODEL_NOTEBOOK_LINEAGE.md` |
| KDAL/Tokyo research-production parity audit (2026-08-11) | `KDAL_TOKYO_RESEARCH_PRODUCTION_PARITY_AUDIT_2026-08-11.md` |
| Notebook generator | `generate_station_notebook.py` |

KDAL is the reference implementation for the Dallas probability challenger.
Seoul and Tokyo are the reference implementations for the Asia 11 AM no-peak
profile. Every station notebook contains, in order:

1. the station-specific V20 no-peak point-model training workflow;
2. XGBoost, LightGBM, and CatBoost base forecasts;
3. the Ridge stack and chronological point-model evaluation;
4. station-market probability training (**Ordinal Probabilities Model 2** for
   KDAL; the native whole-1°C ordinal model for Seoul/Tokyo);
5. chronological probability evaluation;
6. exploratory 2026 holdout scoring;
7. pure-ordinal model-weight export;
8. any enabled station-specific challenger; and
9. its predictions, comparisons, model weights, and manifests.

The probability model is intentionally in the same notebook as the station
training workflow. It is not a detached post-processing notebook.

## Active versus historical work

All `station_stacking_v*` directories are preserved in place as historical
research. They are inputs and evidence, not the starting point for a new
station. No historical notebook has been deleted or moved because existing
notebooks and scripts may reference those paths.

For new station work, copy a station configuration and generate a new notebook
under `stations/{STATION}/`. Station-specific providers, feature builders,
timing, labels, and source notebook builders must remain station-specific.

Generate KDAL:

```powershell
python notebooks\station_training_baseline\generate_station_notebook.py `
  --config notebooks\station_training_baseline\configs\KDAL.json
```

Generate Seoul and Tokyo:

```powershell
python notebooks\station_training_baseline\generate_station_notebook.py `
  --config notebooks\station_training_baseline\configs\Seoul.json

python notebooks\station_training_baseline\generate_station_notebook.py `
  --config notebooks\station_training_baseline\configs\Tokyo.json
```

Generated model artifacts go to:

```text
data/calibration/station_training_baseline/{STATION}/
```

The ordinal artifact remains research/shadow-only until it passes fresh,
station-specific promotion evidence.

Seoul and Tokyo use the `asia_no_peak` probability profile: GFS, GEFS, and JMA
MSM at the local 11 AM cutoff. Their honest point-stack probability history is
2024–2025, with 2025 as forward validation and 2026 as exploratory holdout.
Seoul and Tokyo probability targets and exported market distributions are
native whole degrees Celsius; their historical rounded-Fahrenheit/2°F artifacts
are retained only for comparison.

## Integrated KDAL probability challenger

The KDAL notebook now runs the ordinal challenger after the pure probability
baseline and exports exactly three roles: blended ordinal, shared-slope ordinal,
and pure ordinal. The shared implementation and research notes are documented at
`notebooks/experiments/kdal_ordinal_challenger_v1/README.md`. None of these arms can
override the V20 point bucket.
