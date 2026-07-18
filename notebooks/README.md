# Notebooks

The current baseline notebook workflow is `station_stacking_v2`. The precipitation-feature experiments are `station_stacking_v4` and `station_stacking_v5`.

## Current

For a repository-wide audit of all raw pulls, processed tables, calibration
datasets, exports, and model artifacts, use `notebooks/exploratory_analysis.ipynb`.

Use these for the active same-day 11 AM station-stacking research path:

```text
notebooks/station_stacking_v2/stacking_KATL_v2.ipynb
notebooks/station_stacking_v2/stacking_KAUS_v2.ipynb
notebooks/station_stacking_v2/stacking_KDAL_v2.ipynb
notebooks/station_stacking_v2/stacking_KHOU_v2.ipynb
notebooks/station_stacking_v2/stacking_KLAX_v2.ipynb
notebooks/station_stacking_v2/stacking_KLGA_v2.ipynb
notebooks/station_stacking_v2/stacking_KMIA_v2.ipynb
notebooks/station_stacking_v2/stacking_KORD_v2.ipynb
notebooks/station_stacking_v2/stacking_KSEA_v2.ipynb
```

These notebooks compare raw HRRR/GFS baselines with XGBoost, LightGBM, CatBoost, and the Ridge stacked meta-model.

## Precipitation Experiment

Use these for the SDK precipitation-feature training path:

```text
notebooks/station_stacking_v4/stacking_KATL_v4.ipynb
notebooks/station_stacking_v4/stacking_KAUS_v4.ipynb
notebooks/station_stacking_v4/stacking_KDAL_v4.ipynb
notebooks/station_stacking_v4/stacking_KHOU_v4.ipynb
notebooks/station_stacking_v4/stacking_KLAX_v4.ipynb
notebooks/station_stacking_v4/stacking_KLGA_v4.ipynb
notebooks/station_stacking_v4/stacking_KMIA_v4.ipynb
notebooks/station_stacking_v4/stacking_KORD_v4.ipynb
notebooks/station_stacking_v4/stacking_KSEA_v4.ipynb
```

These notebooks write artifacts to `data/calibration/station_stacking_v4`.

`notebooks/station_stacking_v5/` uses the same v4 feature engineering and 100/50 Optuna trial counts, but tunes Optuna against MAE instead of RMSE. It writes artifacts to `data/calibration/station_stacking_v5`.

## V6 Feature-Input Experiment

`notebooks/station_stacking_v6/` starts from the source-owned v5 feature engineering block, keeps the MAE Optuna setup, adds durable per-station Optuna SQLite storage, and includes the 11 AM observation trend columns when those cache fields are populated.

For v6, the authoritative training input contract is each station artifact:

```text
data/calibration/station_stacking_v6/{STATION}_feature_columns.csv
```

Those files list the categorical and numeric columns passed into the model pipeline. The matching `{STATION}_features.csv` file can be used to audit per-feature NaN percentages. Current v6 artifacts contain 237 candidate training inputs: 6 categorical and 231 numeric.

Do not infer the trained feature matrix only from constants such as `V6_FEATURE_COLUMNS`. The trainer removes numeric columns that are all-NaN for a fit, so the authoritative feature contract is the saved `{STATION}_feature_columns.csv` artifact.

## V7 Live-Safe NBM Experiment

`notebooks/station_stacking_v7/` is the current experimental path for retraining on corrected live-safe forecast timing. It uses `timing_mode="same_day_11am_live_safe"`, providers `("gfs", "hrrr", "nbm")`, direct 13Z NBM raw-high cache inclusion, v6/v7 11 AM observation trend inputs, expanding folds `2021-2023 -> 2024` and `2021-2024 -> 2025`, and 2026 as the OOF holdout year.

The v7 notebooks set Optuna to 50 base-model trials and 50 stack trials with 20 random startup trials, using the wider hyperparameter space. Artifacts are written to `data/calibration/station_stacking_v7`.

## V8 Remaining-Warmup Feature Experiment

`notebooks/station_stacking_v8/` is an experimental successor to v7. It keeps the v7 live-safe GFS/HRRR/NBM timing contract and direct `actual_high_f` target, adds source-owned remaining-warmup features, and conservatively drops fixed or consistently zero-importance model inputs. Artifacts are written to `data/calibration/station_stacking_v8`.

V8 is not the production default unless its 2026 OOF MAE, bucket accuracy, and large-miss behavior beat the v7 ridge-stack benchmark.

## V11 Wunderground Settlement Rerun

`notebooks/station_stacking_v11_settlement/` keeps the exact v11 model and feature contract for the full v11 station set, but uses settlement-first daily highs backed by exact Wunderground/Weather Company airport-station history. It writes isolated artifacts to `data/calibration/station_stacking_v11_settlement` so the original v11 benchmark remains unchanged.

## V11 Settlement Fix Temperature Experiment

`notebooks/station_stacking_v11_settlement_fix/` contains KATL and KDAL settlement-first notebooks that retain the v11 remaining-warmup model contract, apply a train-fold-only 3% feature-missingness gate, and add an expanded 11 AM forecast-temperature-versus-observation feature family. The notebooks run one experimental configuration, compare against the existing v11 settlement artifacts on common dates, and do not export model weights by default. An explicit KDAL candidate export was run on 2026-07-16 and written to `data/calibration/station_stacking_v11_settlement_fix/model_weights`; its manifest refit cutoff is 2026-06-21. This is a research candidate, not a live promotion.

## V12 Settlement-First Guarded Blend

`notebooks/station_stacking_v12/` is the current post-June-21 research path. It keeps the v11 live-safe GFS/HRRR/direct-NBM feature lineage, uses settlement-first labels, evaluates 1F/2F/3F provider-mean capped stack predictions, and writes artifacts to `data/calibration/station_stacking_v12`.

V12 candidate bundles are research exports only until `v12_candidate_model_handoff.md` passes the all-9-station 2026 provider-mean gate.

## V14 Curated Weather Experiment

`notebooks/station_stacking_v14/` is a controlled successor to v13. It keeps the v11 remaining-warmup, Huber, and ridge-stack lineage, computes weather aggregates from the enriched forecast caches, then trains only on the v11 feature base plus a curated aggregate weather allowlist that passes coverage checks. Artifacts are written to `data/calibration/station_stacking_v14`.

Use v14 to test whether precipitation, cloud, and forecast-temperature-at-as-of aggregates help without admitting raw provider weather fields or provider-difference feature sprawl.

## V19 Dense Settlement Modal-Bucket Experiment

V19 is split by method and station over the same dense v11 settlement backbone:

- `notebooks/station_stacking_v19/stacking_KATL_v19_b.ipynb`
- `notebooks/station_stacking_v19/stacking_KDAL_v19_b.ipynb`
- `notebooks/station_stacking_v19/stacking_KATL_v19_c.ipynb`
- `notebooks/station_stacking_v19/stacking_KDAL_v19_c.ipynb`

Each uses 30 base-model Optuna trials with 15 startup trials and 30 stack trials with 15 stack startup trials. Both methods use a train-only 3% feature-missingness gate and share the clean study path `data/calibration/station_stacking_v19_patched/{STATION}/dense_backbone`. V19-C uses cumulative-threshold ordinal logistic regression, empirically shaped censored tails, and strict forward-nested tuning. Method outputs remain separate under `v19_b` and `v19_c`; neither notebook exports a production bundle before promotion review. Use `scripts/run_station_stacking_v19_patched.py` to execute KATL then KDAL sequentially without concurrent study writes.

## V20 Peak-Timing Experiment

`notebooks/station_stacking_v20_peak_timing/` contains the single KATL/KDAL V20 arm. It combines the V11 Settlement Fix temperature features with curated live-safe HRRR/NBM afternoon peak, solar, cloud, and precipitation aggregates; uses Wunderground-only highs; and evaluates four expanding validation folds from 2022 through 2025 before the 2026 holdout. Its readiness cells can audit an in-progress shard pull but stop before tuning when station-year peak or target coverage exceeds 3% missingness. Artifacts are written to `data/calibration/station_stacking_v20_peak_timing`, and notebook model export remains disabled by default. An explicit KATL candidate export was run on 2026-07-16 and written to `data/calibration/station_stacking_v20_peak_timing/model_weights`; its manifest refit cutoff is 2026-06-21. Complete the intended July 14 Wunderground horizon and re-export before promotion review.

`notebooks/station_stacking_v20_kdal_fix/` is an isolated KDAL-only patch. It retains NBM temperature timing and HRRR solar/cloud/precipitation physics, excludes the HRRR temperature-curve family, and adds a capped monthly residual correction learned from forward 2023–2025 OOF stack predictions. It writes to `data/calibration/station_stacking_v20_kdal_fix`; KATL and the original V20 artifacts remain unchanged.

## Legacy / Reference

- `notebooks/station_stacking/`: older station-stacking notebooks retained for comparison.
- `notebooks/calibration_ml_walkforward.ipynb`: legacy additive-bias calibration workflow.

When adding new notebooks, use a short markdown cell at the top that states whether the notebook is current, experimental, or legacy.
