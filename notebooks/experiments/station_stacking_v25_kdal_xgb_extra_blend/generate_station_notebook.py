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
            """# KDAL Station Stacking V25: XGBoost + Extra Trees Constrained Blend

This no-retuning experiment reuses the best existing KDAL V20 XGBoost predictions and the
completed V24 Extra Trees predictions. It selects a non-negative two-model weight whose
weights sum to one, using equal-weight expanding-fold validation MAE. The untouched 2026
holdout is evaluated only after the weight is selected.
"""
        ),
        _code(
            """from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from IPython.display import display

PROJECT_ROOT = Path.cwd().resolve()
while not (PROJECT_ROOT / "src" / "calibration" / "constrained_blend.py").exists():
    if PROJECT_ROOT.parent == PROJECT_ROOT:
        raise RuntimeError("Could not find project root.")
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.constrained_blend import (
    FixedWeightBlendRegressor,
    blend_predictions,
    merge_prediction_sources,
    scan_two_model_weights,
    select_two_model_weight,
    selected_fold_metrics,
)
from src.calibration.station_stacking import _metric_row, _prediction_columns
from src.export_station_stacking_v2_models import _bucket_probability_policy

STATION_ID = "KDAL"
PRIMARY_METHOD = "xgboost"
SECONDARY_METHOD = "extra_trees"
BLEND_METHOD = "xgb_extra_constrained_blend"
GRID_STEP = 0.001
MODEL_VERSION = "station_high_regressor_v25_kdal_xgb_extra_constrained_blend"

V20_DIR = PROJECT_ROOT / "data" / "calibration" / "station_stacking_v20_kdal_no_peak"
V24_DIR = PROJECT_ROOT / "data" / "calibration" / "station_stacking_v24_kdal_no_peak_diverse_ensemble"
OUTPUT_DIR = PROJECT_ROOT / "data" / "calibration" / "station_stacking_v25_kdal_xgb_extra_blend"
MODEL_DIR = OUTPUT_DIR / "model_weights"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
"""
        ),
        _markdown("## Load already-tuned out-of-fold and holdout predictions\n"),
        _code(
            """v20_validation = pd.read_csv(V20_DIR / f"{STATION_ID}_year_split_validation_predictions.csv")
v20_test = pd.read_csv(V20_DIR / f"{STATION_ID}_year_split_test_predictions.csv")
v24_validation = pd.read_csv(V24_DIR / f"{STATION_ID}_year_split_validation_predictions.csv")
v24_test = pd.read_csv(V24_DIR / f"{STATION_ID}_year_split_test_predictions.csv")

validation = merge_prediction_sources(
    v20_validation,
    v24_validation,
    primary_method=PRIMARY_METHOD,
    secondary_method=SECONDARY_METHOD,
)
test = merge_prediction_sources(
    v20_test,
    v24_test,
    primary_method=PRIMARY_METHOD,
    secondary_method=SECONDARY_METHOD,
)

print(f"Validation rows: {len(validation):,}")
print(f"Validation folds: {validation['fold'].nunique()}")
print(f"Untouched 2026 test rows: {len(test):,}")
"""
        ),
        _markdown("## Select the convex weight on expanding-fold validation MAE\n"),
        _code(
            """weight_scan = scan_two_model_weights(
    validation,
    primary_method=PRIMARY_METHOD,
    secondary_method=SECONDARY_METHOD,
    grid_step=GRID_STEP,
)
selection = select_two_model_weight(
    weight_scan,
    primary_method=PRIMARY_METHOD,
    secondary_method=SECONDARY_METHOD,
    grid_step=GRID_STEP,
)
display(pd.DataFrame([selection.__dict__]))
"""
        ),
        _markdown("## Evaluate the selected weight without touching the test labels during selection\n"),
        _code(
            """validation_blend = blend_predictions(validation, selection, method=BLEND_METHOD)
test_blend = blend_predictions(test, selection, method=BLEND_METHOD)
fold_metrics = selected_fold_metrics(validation_blend, selection)

validation_metrics = _metric_row(_prediction_columns(validation_blend))
validation_metrics.update({"evaluation_scope": "year_split_validation", "method": BLEND_METHOD})
test_metrics = _metric_row(_prediction_columns(test_blend))
test_metrics.update({"evaluation_scope": "year_split_test", "method": BLEND_METHOD})
blend_metrics = pd.DataFrame([validation_metrics, test_metrics])

v20_metrics = pd.read_csv(V20_DIR / f"{STATION_ID}_year_split_metrics.csv")
v24_metrics = pd.read_csv(V24_DIR / f"{STATION_ID}_year_split_metrics.csv")
v20_final = v20_metrics.query("evaluation_scope == 'year_split_test' and method == 'ridge_stack'").iloc[0]
v24_final = v24_metrics.query("evaluation_scope == 'year_split_test' and method == 'ridge_stack'").iloc[0]

comparison = pd.DataFrame(
    [
        {
            "model": "V20 boosted-tree ridge stack",
            "mae_f": float(v20_final["mae_f"]),
            "rmse_f": float(v20_final["rmse_f"]),
            "bias_f": float(v20_final["bias_f"]),
            "bucket_log_loss": float(v20_final["bucket_log_loss"]),
            "bucket_accuracy_pct": float(v20_final["bucket_accuracy_pct"]),
        },
        {
            "model": "V24 XGB + Extra Trees + Ridge stack",
            "mae_f": float(v24_final["mae_f"]),
            "rmse_f": float(v24_final["rmse_f"]),
            "bias_f": float(v24_final["bias_f"]),
            "bucket_log_loss": float(v24_final["bucket_log_loss"]),
            "bucket_accuracy_pct": float(v24_final["bucket_accuracy_pct"]),
        },
        {
            "model": "V25 constrained XGB + Extra Trees blend",
            "mae_f": float(test_metrics["mae_f"]),
            "rmse_f": float(test_metrics["rmse_f"]),
            "bias_f": float(test_metrics["bias_f"]),
            "bucket_log_loss": float(test_metrics["bucket_log_loss"]),
            "bucket_accuracy_pct": float(test_metrics["bucket_accuracy_pct"]),
        },
    ]
)
comparison["delta_mae_vs_v20_f"] = comparison["mae_f"] - float(v20_final["mae_f"])
comparison["delta_rmse_vs_v20_f"] = comparison["rmse_f"] - float(v20_final["rmse_f"])
display(fold_metrics)
display(comparison.sort_values("mae_f"))
"""
        ),
        _markdown("## Save the audit artifacts and a deployable two-model bundle\n"),
        _code(
            """weight_scan.to_csv(OUTPUT_DIR / f"{STATION_ID}_weight_scan.csv", index=False)
fold_metrics.to_csv(OUTPUT_DIR / f"{STATION_ID}_validation_fold_metrics.csv", index=False)
validation_blend.to_csv(OUTPUT_DIR / f"{STATION_ID}_validation_predictions.csv", index=False)
test_blend.to_csv(OUTPUT_DIR / f"{STATION_ID}_test_predictions.csv", index=False)
blend_metrics.to_csv(OUTPUT_DIR / f"{STATION_ID}_metrics.csv", index=False)
comparison.to_csv(OUTPUT_DIR / f"{STATION_ID}_comparison.csv", index=False)

v20_bundle_path = (
    V20_DIR / "model_weights" / "KDAL_station_high_regressor_v20_kdal_no_peak_stack.joblib"
)
v24_bundle_path = (
    V24_DIR
    / "model_weights"
    / "KDAL_station_high_regressor_v24_kdal_no_peak_diverse_stack.joblib"
)
v20_manifest_path = v20_bundle_path.with_suffix(".json")
v24_manifest_path = v24_bundle_path.with_suffix(".json")
v20_bundle = joblib.load(v20_bundle_path)
v24_bundle = joblib.load(v24_bundle_path)

contract_keys = (
    "station_id",
    "target",
    "target_mode",
    "target_source",
    "model_target",
    "timing_mode",
    "providers",
    "feature_version",
    "training_profile",
    "feature_names",
)
for key in contract_keys:
    if v20_bundle.get(key) != v24_bundle.get(key):
        raise ValueError(f"Source bundles disagree on {key}.")

stack_features = (
    "xgboost_predicted_high_f",
    "extra_trees_predicted_high_f",
)
blend_model = FixedWeightBlendRegressor(
    stack_features,
    (selection.primary_weight, selection.secondary_weight),
)
residual = (
    pd.to_numeric(validation_blend["actual_high_f"], errors="coerce")
    - pd.to_numeric(validation_blend["predicted_high_f"], errors="coerce")
).dropna()
residual_calibrator = {
    "method": BLEND_METHOD,
    "source": "four_fold_out_of_fold_validation_predictions",
    "error_mean_f": float(residual.mean()),
    "error_std_f": max(0.25, float(residual.std(ddof=0))),
    "error_mean_c": float(residual.mean()) * 5.0 / 9.0,
    "error_std_c": max(0.25, float(residual.std(ddof=0))) * 5.0 / 9.0,
    "row_count": int(len(residual)),
    "first_contract_date": str(validation_blend.loc[residual.index, "contract_date"].min()),
    "last_contract_date": str(validation_blend.loc[residual.index, "contract_date"].max()),
}

bundle = dict(v24_bundle)
bundle.update(
    {
        "model_version": MODEL_VERSION,
        "base_model_methods": (PRIMARY_METHOD, SECONDARY_METHOD),
        "base_models": {
            PRIMARY_METHOD: v20_bundle["base_models"][PRIMARY_METHOD],
            SECONDARY_METHOD: v24_bundle["base_models"][SECONDARY_METHOD],
        },
        "stack_enabled": True,
        "final_model_method": BLEND_METHOD,
        "stack_model": blend_model,
        "stack_features": stack_features,
        "residual_calibrator": residual_calibrator,
        "bucket_probability_policy": _bucket_probability_policy(
            residual_calibrator,
            v24_bundle.get("bucket_contract", "polymarket_half_up_2f"),
        ),
    }
)
bundle_path = MODEL_DIR / f"{STATION_ID}_{MODEL_VERSION}.joblib"
joblib.dump(bundle, bundle_path)

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

bundle_digest = sha256(bundle_path)
v20_manifest = json.loads(v20_manifest_path.read_text(encoding="utf-8"))
v24_manifest = json.loads(v24_manifest_path.read_text(encoding="utf-8"))
base_manifests = {
    item["method"]: item
    for item in [*v20_manifest["base_models"], *v24_manifest["base_models"]]
}
manifest = dict(v24_manifest)
manifest.update(
    {
        "model_version": MODEL_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_pipeline": "notebooks/experiments/station_stacking_v25_kdal_xgb_extra_blend",
        "source_artifact_dir": str(OUTPUT_DIR.relative_to(PROJECT_ROOT)).replace("\\\\", "/"),
        "bundle_path": str(bundle_path.relative_to(PROJECT_ROOT)).replace("\\\\", "/"),
        "artifact_integrity": {"bundle_sha256": bundle_digest},
        "base_models": [base_manifests[PRIMARY_METHOD], base_manifests[SECONDARY_METHOD]],
        "stack_model": {
            "method": BLEND_METHOD,
            "selection_metric": "equal_fold_mean_mae_f",
            "constraint": "non_negative_weights_sum_to_one",
            "features": list(stack_features),
            "primary_weight": selection.primary_weight,
            "secondary_weight": selection.secondary_weight,
            "validation_mean_fold_mae_f": selection.mean_fold_mae_f,
            "validation_worst_fold_mae_f": selection.worst_fold_mae_f,
            "validation_row_mae_f": selection.row_mae_f,
            "grid_step": selection.grid_step,
        },
        "residual_calibrator": residual_calibrator,
        "bucket_probability_policy": bundle["bucket_probability_policy"],
    }
)
manifest["model_contract"].update(
    {
        "base_model_methods": [PRIMARY_METHOD, SECONDARY_METHOD],
        "stack_enabled": True,
        "final_model_method": BLEND_METHOD,
    }
)
manifest["inference"].update(
    {
        "final_model_method": BLEND_METHOD,
        "base_prediction_inputs": list(stack_features),
    }
)
manifest["training_validation"] = {
    "selection_source": "V20 XGBoost OOF plus V24 Extra Trees OOF",
    "selection_rule": "equal_fold_mean_mae_then_worst_mae_then_distance_from_equal",
    "fold_count": int(weight_scan["fold_count"].iloc[0]),
    "validation_row_count": int(weight_scan["row_count"].iloc[0]),
    "primary_weight": selection.primary_weight,
    "secondary_weight": selection.secondary_weight,
    "mean_fold_mae_f": selection.mean_fold_mae_f,
    "worst_fold_mae_f": selection.worst_fold_mae_f,
    "grid_step": selection.grid_step,
    "test_year": 2026,
    "test_row_count": int(test_metrics["count"]),
    "test_mae_f": float(test_metrics["mae_f"]),
    "test_rmse_f": float(test_metrics["rmse_f"]),
}
manifest_path = bundle_path.with_suffix(".json")
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

summary = {
    "station_id": STATION_ID,
    "model_version": MODEL_VERSION,
    "primary_source": str(v20_bundle_path.relative_to(PROJECT_ROOT)).replace("\\\\", "/"),
    "secondary_source": str(v24_bundle_path.relative_to(PROJECT_ROOT)).replace("\\\\", "/"),
    "primary_weight": selection.primary_weight,
    "secondary_weight": selection.secondary_weight,
    "validation_mean_fold_mae_f": selection.mean_fold_mae_f,
    "test_metrics": {key: value for key, value in test_metrics.items()},
    "bundle_path": str(bundle_path.relative_to(PROJECT_ROOT)).replace("\\\\", "/"),
    "bundle_sha256": bundle_digest,
    "manifest_path": str(manifest_path.relative_to(PROJECT_ROOT)).replace("\\\\", "/"),
}
(OUTPUT_DIR / f"{STATION_ID}_blend_summary.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)
print(bundle_path)
print(manifest_path)
"""
        ),
        _markdown("## Final result\n"),
        _code(
            """winner = comparison.sort_values(["mae_f", "rmse_f"]).iloc[0]
print(
    f"Selected weights: XGBoost={selection.primary_weight:.3f}, "
    f"Extra Trees={selection.secondary_weight:.3f}"
)
print(
    f"V25 holdout: MAE={test_metrics['mae_f']:.4f} F, "
    f"RMSE={test_metrics['rmse_f']:.4f} F"
)
print(f"Best tested holdout model: {winner['model']}")
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
    path = out_dir / "stacking_KDAL_v25_xgb_extra_blend.ipynb"
    path.write_text(json.dumps(_notebook(), indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
