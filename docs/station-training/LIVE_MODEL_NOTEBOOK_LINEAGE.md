# Live Model Notebook Lineage

This document records the training-notebook and clean-source provenance of the point models promoted after the Polymarket weather bot patch. These are the station high-temperature point models used by the live bot.

## Production lineage

| Station | Training notebook | Clean research commit | Live model identity | Frozen feature contract | Production bundle SHA-256 |
| --- | --- | --- | --- | ---: | --- |
| KDAL | `notebooks/station_training_baseline/stations/KDAL/train_KDAL.ipynb` | `a096fc5a9cb6dc4f27c21230d46ecc5fca2b7193` | `station_high_regressor_live_kdal_no_peak_stack_2026` | 198 ordered features | `8b559b595c16bf35cf7bec0e0719963bf13fda6cf5664a42297bbc925a78c974` |
| RJTT | `notebooks/station_training_baseline/stations/Tokyo/train_Tokyo.ipynb` | `46d3e9e31aa6bdf0b40a47a65d9250031ce7411b` | `station_high_regressor_live_tokyo_no_peak_stack_2026` | 293 ordered features | `2a30f116c188e4199950911523cdbe4cdb680a0e9cb8361092e23fd374c07d70` |

The commit in each row is the clean research source used for that station's release, not a claim that the generated model bundle is stored in Git. KDAL's release uses the KDAL notebook at its listed clean commit. RJTT's release uses the Tokyo notebook at its listed clean commit, after the notebook's Optuna tuning and refit completed.

The active baseline notebooks on `main` are byte-identical to the notebook blobs at both clean research commits above. Their station-relevant pipeline dependencies also match:

- Relative to the KDAL clean commit, the only later change inside the station-training and calibration pipeline scope is in `src/calibration/asia_station_stacking.py`, which the KDAL notebook does not import.
- Relative to the RJTT clean commit, the only later change inside that scope is in `src/calibration/kdal_ordinal_challenger.py`, which the Tokyo notebook does not import.

The non-versioned station baseline therefore reflects both production point-model pipelines. This is a station-scoped statement, not a claim that every shared pipeline file in the repository is identical across the two clean commits. No notebook replacement or regeneration is required for this documentation update.

## Training, evaluation, and live refit

The notebook workflow has three distinct roles:

1. **Training and tuning** builds the station-specific feature frame and completes model/stack hyperparameter selection. In particular, the RJTT release follows the completed Optuna tuning and refit in `train_Tokyo.ipynb`.
2. **Evaluation** trains against fixed historical windows for chronological validation and the separately labelled exploratory holdout. The evaluation bundle establishes the ordered feature contract. Evaluation or holdout metrics belong to that evaluation fit only.
3. **Live refit** retrains the point model on all completed actuals available at release time while reusing the evaluation bundle's exact ordered features. It may not add newly eligible columns. Consequently, the live refit has its own model identity and bundle hash and must not inherit an out-of-sample performance claim from the evaluation fit.

The frozen contracts are release invariants: 198 ordered features for KDAL and 293 ordered features for RJTT. Feature count alone is not sufficient verification; ordering and identity must match the released manifest or bundle contract.

## Point model versus probability artifact

Both rows above identify the **point models used live** to predict the final station high. RJTT notebook outputs also include probability-model research and shadow/evaluation artifacts. This lineage record does **not** say that an RJTT probability artifact is active, and such an artifact must not be presented as part of the live release unless a separate production promotion record explicitly establishes that status.

## Verification checklist

- Confirm the notebook path exists at the station's recorded commit, for example with `git show <commit>:<notebook-path>`.
- Confirm the checkout used to produce the bundle was clean and at the exact recorded research commit before executing the release refit.
- Confirm the live manifest reports the exact model identity shown above.
- Confirm the manifest reports a frozen evaluation feature contract with exactly 198 ordered features for KDAL or 293 for RJTT; compare the ordered feature list, not only its length.
- Compute SHA-256 over the production bundle bytes and compare it with the station's checksum above.
- Confirm runtime selection resolves to the point-model bundle for the intended station.
- For RJTT, confirm no probability artifact is described or selected as active unless a separate, explicit promotion record exists.
