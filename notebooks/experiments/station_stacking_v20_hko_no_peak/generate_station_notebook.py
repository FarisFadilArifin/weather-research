from __future__ import annotations

import json
from pathlib import Path


def _markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def _code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def _notebook() -> dict:
    cells = [
        _markdown(
            """# HKO Station Stacking V20 GFS-Only No Peak

Hong Kong adaptation of the KDAL V20 no-peak experiment. It uses the official HKO daily maximum,
official HKO Headquarters observations through 11 AM Hong Kong time, and exact prior-day 18Z GFS forecasts.
The model remains Fahrenheit-native for compatibility and reports both Fahrenheit and Celsius results.
"""
        ),
        _code(
            """from pathlib import Path
import sys
import warnings

warnings.filterwarnings("ignore", message="IProgress not found.*")
warnings.filterwarnings("ignore", message="Skipping features without any observed values.*")

PROJECT_ROOT = Path.cwd().resolve()
while not (PROJECT_ROOT / "src" / "hong_kong_11am.py").exists():
    if PROJECT_ROOT.parent == PROJECT_ROOT:
        raise RuntimeError("Could not find project root containing src/hong_kong_11am.py")
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_ROOT = PROJECT_ROOT / "data" / "calibration" / "hong_kong_11am"
STATION_ID = "HKO"
PROVIDERS = ("gfs",)
TIMING_MODE = "hong_kong_same_day_11am_live_safe"
FEATURE_VERSION = "v20_hko_gfs_no_peak"
TARGET_SOURCE = "hko_daily_max"
FAST_MODE = False
OPTUNA_TRIALS = 30
MODEL_VERSION = "station_high_regressor_v20_hko_gfs_no_peak_stack"
EXPORT_MODEL_WEIGHTS = True
PROJECT_ROOT
"""
        ),
        _code(
            """import numpy as np
import pandas as pd

from src.export_station_stacking_v2_models import export_station_model_weights
from src.calibration.station_stacking import (
    TARGET_MODE_REMAINING_WARMUP,
    TRAINING_PROFILE_V20_ALIGNED,
    V20_EXPANDING_FOLDS,
    V20_HKO_GFS_NO_PEAK_DROPPED_FEATURE_COLUMNS,
    _fit_feature_columns,
    _modeling_frame,
    missing_model_dependencies,
)
from src.hong_kong_11am import (
    GFS_ALLOWED_GAP_END_DATE,
    GFS_USABLE_START_DATE,
    MODEL_PROVIDERS,
    OBSERVATION_SOURCE_CONTRACT,
    add_celsius_metric_columns,
    add_celsius_prediction_columns,
    hong_kong_stacking_config,
    provider_modeling_coverage,
    run_hong_kong_year_split_experiment,
)
"""
        ),
        _markdown(
            """## Contract

- HKO official daily maximum target (`hko_daily_max`)
- HKO Headquarters 1-minute temperature, maximum-since-midnight, and humidity snapshots by 11 AM
- Historical snapshots come from the free DATA.GOV.HK archive; live snapshots come from HKO Open Data
- Same-station quality contract: HKO high-so-far cannot exceed the official HKO daily maximum
- GFS-only forecast roster
- Known access-blocked gap through 2021-03-23; uninterrupted exact GFS required from 2021-03-24
- V20 expanding validation folds for 2022–2025 and 2026 out-of-fold testing
- No HRRR/NBM peak-timing feature family
"""
        ),
        _code(
            """config = hong_kong_stacking_config(
    PROJECT_ROOT,
    DATA_ROOT,
    fast_mode=FAST_MODE,
    optuna_trials=OPTUNA_TRIALS,
)
assert config.observation_target_same_station is True
assert config.observation_source == OBSERVATION_SOURCE_CONTRACT

fold_spec = pd.DataFrame(
    [
        {
            "fold": fold.name,
            "train_start_year": fold.train_start_year,
            "train_end_year": fold.train_end_year,
            "validation_year": fold.validation_year,
            "validation_weight": config.effective_year_split_validation_weights[fold.validation_year],
        }
        for fold in V20_EXPANDING_FOLDS
    ]
)
fold_spec
"""
        ),
        _markdown("## Data Readiness\n"),
        _code(
            """coverage = pd.DataFrame(
    [provider_modeling_coverage(DATA_ROOT, provider) for provider in MODEL_PROVIDERS]
)
coverage[
    [
        "provider",
        "usable_start_date",
        "usable_end_date",
        "ok_rows",
        "required_usable_rows",
        "allowed_early_gap_rows",
        "modeling_ready",
    ]
]
"""
        ),
        _markdown("## Train and Score\n"),
        _code(
            """missing_packages = missing_model_dependencies(config.effective_base_model_methods)
if missing_packages:
    raise ImportError(
        "Missing station-stacking ML packages: "
        + ", ".join(missing_packages)
        + ". Install them with: python -m pip install -r requirements.txt"
    )

if not coverage["modeling_ready"].all():
    raise RuntimeError("GFS coverage is not ready; inspect missing_usable_dates before training")

result = run_hong_kong_year_split_experiment(
    DATA_ROOT,
    project_root=PROJECT_ROOT,
    providers=MODEL_PROVIDERS,
    fast_mode=FAST_MODE,
    optuna_trials=OPTUNA_TRIALS,
)
result.scoreboard
"""
        ),
        _markdown("## Export Complete Base + Ridge Bundle\n"),
        _code(
            """if EXPORT_MODEL_WEIGHTS:
    exported_weights = export_station_model_weights(
        project_root=PROJECT_ROOT,
        station_id=STATION_ID,
        artifact_dir=config.resolved_output_dir(),
        model_version=MODEL_VERSION,
        timing_mode=TIMING_MODE,
        providers=MODEL_PROVIDERS,
        feature_version=FEATURE_VERSION,
        training_profile=TRAINING_PROFILE_V20_ALIGNED,
        optuna_metric="mae_f",
        target_mode=TARGET_MODE_REMAINING_WARMUP,
        target_source=TARGET_SOURCE,
        base_model_methods=("xgboost", "lightgbm", "catboost"),
        stack_enabled=True,
        source_pipeline="notebooks/experiments/station_stacking_v20_hko_no_peak",
        max_feature_missing_fraction=0.03,
        bucket_contract="floor_1c",
        observation_target_same_station=True,
        observation_source=OBSERVATION_SOURCE_CONTRACT,
    )
    exported_weights.bundle_path, exported_weights.manifest_path
else:
    print("Model export disabled.")
"""
        ),
        _markdown("## Included and Pruned Feature Audit\n"),
        _code(
            """included_features = result.feature_columns.sort_values(["kind", "feature"]).reset_index(drop=True)
pruned_single_provider_features = pd.DataFrame(
    {"feature": sorted(V20_HKO_GFS_NO_PEAK_DROPPED_FEATURE_COLUMNS)}
)
included_features, pruned_single_provider_features
"""
        ),
        _code(
            """feature_coverage = (
    result.features[included_features["feature"].tolist()]
    .notna()
    .mean()
    .mul(100)
    .sort_values(ascending=False)
    .rename("coverage_pct")
    .reset_index()
    .rename(columns={"index": "feature"})
)
feature_coverage
"""
        ),
        _markdown("## Train-Fold 3% Missingness Audit\n"),
        _code(
            """modeling_frame, candidate_categorical, candidate_numeric = _modeling_frame(result.features, config)
candidate_features = [*candidate_categorical, *candidate_numeric]
missingness_rows = []
years = pd.to_numeric(modeling_frame["year"], errors="coerce")

for fold in [*V20_EXPANDING_FOLDS, ("test_refit_2021_2025", 2021, 2025)]:
    if isinstance(fold, tuple):
        fold_name, train_start, train_end = fold
    else:
        fold_name, train_start, train_end = fold.name, fold.train_start_year, fold.train_end_year
    train = modeling_frame.loc[years.between(train_start, train_end)].copy()
    retained_categorical, retained_numeric = _fit_feature_columns(
        train,
        candidate_categorical,
        candidate_numeric,
        max_missing_fraction=0.03,
    )
    retained = set(retained_categorical) | set(retained_numeric)
    for feature in candidate_features:
        missingness_rows.append(
            {
                "fold": fold_name,
                "feature": feature,
                "missing_fraction": float(train[feature].isna().mean()),
                "retained": feature in retained,
            }
        )

fold_feature_missingness = pd.DataFrame(missingness_rows)
fold_feature_missingness.to_csv(
    config.resolved_output_dir() / "HKO_fold_feature_missingness.csv",
    index=False,
)
fold_feature_missingness.loc[~fold_feature_missingness["retained"]].sort_values(
    ["fold", "missing_fraction"], ascending=[True, False]
)
"""
        ),
        _markdown("## Feature Importance\n"),
        _code(
            """result.feature_importance.sort_values(
    ["method", "importance_mean_mae_f"],
    ascending=[True, False],
).groupby("method", as_index=False).head(20)
"""
        ),
        _markdown("## Fahrenheit and Celsius Metrics\n"),
        _code(
            """dual_metrics = add_celsius_metric_columns(result.metrics)
dual_scoreboard = add_celsius_metric_columns(result.scoreboard)
dual_metrics, dual_scoreboard
"""
        ),
        _markdown("## 2026 Celsius Buckets (Nearest Degree, Half-Up)\n"),
        _code(
            """dual_test_predictions = add_celsius_prediction_columns(result.test_predictions)
c_bucket_predictions = result.bracket_predictions
c_bucket_metrics = result.bracket_metrics

assert c_bucket_metrics["bucket_unit"].eq("celsius").all()
assert c_bucket_metrics["bucket_width_c"].eq(1.0).all()
assert c_bucket_metrics["rounding_rule"].eq("floor_integer_celsius").all()
c_bucket_metrics
"""
        ),
        _markdown("## 2026 Monthly Metrics\n"),
        _code(
            """monthly = dual_test_predictions.copy()
monthly["month"] = pd.to_datetime(monthly["contract_date"], errors="coerce").dt.month
monthly_metrics = (
    monthly.dropna(subset=["month", "error_f"])
    .groupby(["method", "month"], as_index=False)
    .agg(
        count=("error_f", "size"),
        mae_f=("absolute_error_f", "mean"),
        rmse_f=("error_f", lambda values: float(np.sqrt(np.mean(np.square(values))))),
        bias_f=("error_f", "mean"),
        mae_c=("absolute_error_c", "mean"),
        rmse_c=("error_c", lambda values: float(np.sqrt(np.mean(np.square(values))))),
        bias_c=("error_c", "mean"),
    )
)
monthly_metrics.to_csv(config.resolved_output_dir() / "HKO_2026_monthly_metrics_dual_units.csv", index=False)
monthly_metrics
"""
        ),
        _markdown("## GFS Raw Uplift\n"),
        _code(
            """test_mae = (
    dual_test_predictions.groupby("method", as_index=False)
    .agg(count=("absolute_error_f", "size"), mae_f=("absolute_error_f", "mean"), mae_c=("absolute_error_c", "mean"))
)
gfs_mae_f = float(test_mae.loc[test_mae["method"].eq("gfs_raw"), "mae_f"].iloc[0])
gfs_mae_c = float(test_mae.loc[test_mae["method"].eq("gfs_raw"), "mae_c"].iloc[0])
test_mae["mae_uplift_vs_gfs_f"] = gfs_mae_f - test_mae["mae_f"]
test_mae["mae_uplift_vs_gfs_c"] = gfs_mae_c - test_mae["mae_c"]
test_mae.sort_values("mae_f")
"""
        ),
        _markdown("## Performance by Warm/Cool 11 AM Forecast Delta\n"),
        _code(
            """delta_by_date = result.features[
    ["contract_date", "v11sf_forecast_temp_11am_minus_observed_f"]
].copy()
delta_by_date["forecast_temp_delta_c"] = (
    pd.to_numeric(delta_by_date["v11sf_forecast_temp_11am_minus_observed_f"], errors="coerce") * 5 / 9
)
delta_predictions = dual_test_predictions.merge(delta_by_date, on="contract_date", how="left")
delta_predictions["forecast_temp_delta_bucket"] = pd.cut(
    delta_predictions["forecast_temp_delta_c"],
    bins=[-np.inf, -1.0, -0.25, 0.25, 1.0, np.inf],
    labels=["cool_gt_1c", "cool_0.25_to_1c", "near_match", "warm_0.25_to_1c", "warm_gt_1c"],
)
warm_cool_metrics = (
    delta_predictions.dropna(subset=["forecast_temp_delta_bucket", "error_f"])
    .groupby(["method", "forecast_temp_delta_bucket"], observed=True, as_index=False)
    .agg(
        count=("error_f", "size"),
        mae_f=("absolute_error_f", "mean"),
        bias_f=("error_f", "mean"),
        mae_c=("absolute_error_c", "mean"),
        bias_c=("error_c", "mean"),
    )
)
warm_cool_metrics.to_csv(config.resolved_output_dir() / "HKO_warm_cool_delta_metrics.csv", index=False)
warm_cool_metrics
"""
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    path = Path(__file__).resolve().parent / "stacking_HKO_v20_no_peak.ipynb"
    path.write_text(json.dumps(_notebook(), indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
