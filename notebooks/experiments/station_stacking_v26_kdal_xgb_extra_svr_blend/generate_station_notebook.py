from __future__ import annotations

import json
from pathlib import Path


def _markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


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
            """# KDAL Station Stacking V26: XGBoost + Extra Trees + RBF-SVR

This experiment keeps the already tuned V20 XGBoost and V24 Extra Trees predictions fixed.
Only an RBF-SVR is tuned on the same four expanding folds. A three-model non-negative simplex
blend is then selected by equal-fold mean MAE. The 2026 results remain exploratory because that
period has already been examined in prior model-selection work.
"""
        ),
        _code(
            """from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from IPython.display import display

PROJECT_ROOT = Path.cwd().resolve()
while not (PROJECT_ROOT / "src" / "calibration" / "station_stacking.py").exists():
    if PROJECT_ROOT.parent == PROJECT_ROOT:
        raise RuntimeError("Could not find project root.")
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ["WEATHER_RESEARCH_INCLUDE_DIRECT_NBM"] = "1"

from src.calibration.constrained_blend import (
    blend_simplex_predictions,
    merge_multiple_prediction_sources,
    scan_three_model_simplex_weights,
    select_three_model_simplex_weights,
)
from src.calibration.station_stacking import (
    StationStackingConfig,
    V20_EXPANDING_FOLDS,
    _metric_row,
    _modeling_frame,
    _prediction_columns,
    tune_year_split_base_models,
    year_split_test_predictions,
)

STATION_ID = "KDAL"
METHODS = ("xgboost", "extra_trees", "svr")
BLEND_METHOD = "xgb_extra_svr_simplex_blend"
SVR_TRIALS = 30
SIMPLEX_STEP = 0.005

V20_DIR = PROJECT_ROOT / "data" / "calibration" / "station_stacking_v20_kdal_no_peak"
V24_DIR = PROJECT_ROOT / "data" / "calibration" / "station_stacking_v24_kdal_no_peak_diverse_ensemble"
V25_DIR = PROJECT_ROOT / "data" / "calibration" / "station_stacking_v25_kdal_xgb_extra_blend"
OUTPUT_DIR = PROJECT_ROOT / "data" / "calibration" / "station_stacking_v26_kdal_xgb_extra_svr_blend"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

config = StationStackingConfig(
    station_id=STATION_ID,
    project_root=PROJECT_ROOT,
    timing_mode="same_day_11am_live_safe",
    providers=("gfs", "hrrr", "nbm"),
    fast_mode=False,
    optuna_trials=SVR_TRIALS,
    optuna_startup_trials=15,
    optuna_metric="mae_f",
    optuna_verbose=False,
    feature_version="v11_settlement_fix_temp",
    training_profile="v20_aligned",
    target_mode="remaining_warmup",
    target_source="wunderground_only",
    max_feature_missing_fraction=0.03,
    base_model_methods=("svr",),
    stack_enabled=False,
    hyperparameter_space="wide",
    year_split_folds=V20_EXPANDING_FOLDS,
    year_split_validation_weights={2022: 1.0, 2023: 1.0, 2024: 1.0, 2025: 1.0},
    year_split_test_train_years=(2021, 2025),
    year_split_test_year=2026,
    output_dir=OUTPUT_DIR,
)
print(config.resolved_optuna_storage_path())
"""
        ),
        _markdown("## Tune only the new RBF-SVR base learner\n"),
        _code(
            """features = pd.read_csv(V24_DIR / f"{STATION_ID}_features.csv", low_memory=False)
modeling_frame, categorical, numeric = _modeling_frame(features, config)

svr_tuning, svr_validation, svr_selected = tune_year_split_base_models(
    modeling_frame,
    config,
    categorical,
    numeric,
    list(V20_EXPANDING_FOLDS),
)
svr_test = year_split_test_predictions(
    modeling_frame,
    config,
    categorical,
    numeric,
    svr_selected,
    train_years=(2021, 2025),
    test_year=2026,
)

svr_tuning.to_csv(OUTPUT_DIR / f"{STATION_ID}_svr_tuning.csv", index=False)
svr_selected.to_csv(OUTPUT_DIR / f"{STATION_ID}_svr_selected_hyperparameters.csv", index=False)
svr_validation.to_csv(OUTPUT_DIR / f"{STATION_ID}_svr_validation_predictions.csv", index=False)
svr_test.to_csv(OUTPUT_DIR / f"{STATION_ID}_svr_test_predictions.csv", index=False)
display(svr_selected)
"""
        ),
        _markdown("## Merge honest base predictions and select the three-model simplex weight\n"),
        _code(
            """v20_validation = pd.read_csv(V20_DIR / f"{STATION_ID}_year_split_validation_predictions.csv")
v20_test = pd.read_csv(V20_DIR / f"{STATION_ID}_year_split_test_predictions.csv")
v24_validation = pd.read_csv(V24_DIR / f"{STATION_ID}_year_split_validation_predictions.csv")
v24_test = pd.read_csv(V24_DIR / f"{STATION_ID}_year_split_test_predictions.csv")

validation = merge_multiple_prediction_sources(
    {
        "xgboost": v20_validation,
        "extra_trees": v24_validation,
        "svr": svr_validation,
    }
)
test = merge_multiple_prediction_sources(
    {
        "xgboost": v20_test,
        "extra_trees": v24_test,
        "svr": svr_test,
    }
)
simplex_scan = scan_three_model_simplex_weights(
    validation,
    methods=METHODS,
    grid_step=SIMPLEX_STEP,
)
selected = select_three_model_simplex_weights(simplex_scan, methods=METHODS)
weights = tuple(float(selected[f"{method}_weight"]) for method in METHODS)
display(pd.DataFrame([selected]))
print(dict(zip(METHODS, weights, strict=True)))
"""
        ),
        _markdown("## Evaluate and compare with the V20 and V25 point models\n"),
        _code(
            """validation_blend = blend_simplex_predictions(
    validation,
    methods=METHODS,
    weights=weights,
    method=BLEND_METHOD,
)
test_blend = blend_simplex_predictions(
    test,
    methods=METHODS,
    weights=weights,
    method=BLEND_METHOD,
)
validation_metrics = _metric_row(_prediction_columns(validation_blend))
validation_metrics.update({"evaluation_scope": "year_split_validation", "method": BLEND_METHOD})
test_metrics = _metric_row(_prediction_columns(test_blend))
test_metrics.update({"evaluation_scope": "year_split_test", "method": BLEND_METHOD})

v20_metrics = pd.read_csv(V20_DIR / f"{STATION_ID}_year_split_metrics.csv")
v25_metrics = pd.read_csv(V25_DIR / f"{STATION_ID}_metrics.csv")
v20 = v20_metrics.query("evaluation_scope == 'year_split_test' and method == 'ridge_stack'").iloc[0]
if len(v25_metrics) < 2:
    raise ValueError("V25 metrics must contain validation followed by test rows.")
v25 = v25_metrics.iloc[1]
svr_only = _metric_row(_prediction_columns(svr_test.loc[svr_test["method"].eq("svr")]))

comparison = pd.DataFrame(
    [
        {"model": "V20 boosted-tree ridge stack", **{k: float(v20[k]) for k in ("mae_f", "rmse_f", "bias_f", "bucket_log_loss", "bucket_accuracy_pct")}},
        {"model": "V25 constrained XGB + Extra Trees", **{k: float(v25[k]) for k in ("mae_f", "rmse_f", "bias_f", "bucket_log_loss", "bucket_accuracy_pct")}},
        {"model": "V26 RBF-SVR alone", **{k: float(svr_only[k]) for k in ("mae_f", "rmse_f", "bias_f", "bucket_log_loss", "bucket_accuracy_pct")}},
        {"model": "V26 XGB + Extra Trees + RBF-SVR", **{k: float(test_metrics[k]) for k in ("mae_f", "rmse_f", "bias_f", "bucket_log_loss", "bucket_accuracy_pct")}},
    ]
)
comparison["delta_mae_vs_v25_f"] = comparison["mae_f"] - float(v25["mae_f"])
comparison["delta_rmse_vs_v25_f"] = comparison["rmse_f"] - float(v25["rmse_f"])

simplex_scan.to_csv(OUTPUT_DIR / f"{STATION_ID}_simplex_weight_scan.csv", index=False)
validation_blend.to_csv(OUTPUT_DIR / f"{STATION_ID}_validation_predictions.csv", index=False)
test_blend.to_csv(OUTPUT_DIR / f"{STATION_ID}_test_predictions.csv", index=False)
pd.DataFrame([validation_metrics, test_metrics]).to_csv(
    OUTPUT_DIR / f"{STATION_ID}_metrics.csv",
    index=False,
)
comparison.to_csv(OUTPUT_DIR / f"{STATION_ID}_comparison.csv", index=False)

summary = {
    "station_id": STATION_ID,
    "experiment": "xgboost_extra_trees_rbf_svr_simplex_blend",
    "base_model_retraining": {"xgboost": False, "extra_trees": False, "svr": True},
    "svr_trials": SVR_TRIALS,
    "simplex_step": SIMPLEX_STEP,
    "weights": dict(zip(METHODS, weights, strict=True)),
    "validation_mean_fold_mae_f": float(selected["mean_fold_mae_f"]),
    "validation_worst_fold_mae_f": float(selected["worst_fold_mae_f"]),
    "test_metrics": {key: value for key, value in test_metrics.items()},
}
(OUTPUT_DIR / f"{STATION_ID}_summary.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)
display(comparison.sort_values(["mae_f", "rmse_f"]))
"""
        ),
        _markdown("## Final result\n"),
        _code(
            """winner = comparison.sort_values(["mae_f", "rmse_f"]).iloc[0]
print(f"Selected weights: {dict(zip(METHODS, weights, strict=True))}")
print(
    f"V26 holdout: MAE={test_metrics['mae_f']:.4f} F, "
    f"RMSE={test_metrics['rmse_f']:.4f} F, "
    f"bucket hit={test_metrics['bucket_accuracy_pct']:.2f}%"
)
print(f"Best exploratory 2026 point model: {winner['model']}")
"""
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": ".venv (Python 3)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.14"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "stacking_KDAL_v26_xgb_extra_svr_blend.ipynb"
    path.write_text(json.dumps(_notebook(), indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
