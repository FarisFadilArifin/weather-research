from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.bucket_probability import FEATURE_PROFILE_KDAL_1PM, sha256_file
from src.calibration.station_stacking import (
    V20_ENGINEERED_FEATURE_COLUMNS,
    V20_KDAL_1PM_TEMP_FEATURE_COLUMNS,
    V20_PEAK_TIMING_RAW_FEATURE_COLUMNS,
)
from src.calibration.v19_bucket import crossfit_ridge_predictions
from src.calibration.win_classifier import (
    build_win_frame,
    predict_win_bundle,
    select_confidence_threshold,
    threshold_metrics,
)


POINT_VERSION = "station_high_regressor_v23_kdal_1pm_bucket_loss_stack"
SELECTOR_VERSION = "station_bucket_win_selector_v23_kdal_1pm_bucket_loss"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the complete KDAL V23 bucket-loss pipeline")
    parser.add_argument(
        "--pipeline-dir",
        type=Path,
        default=PROJECT_ROOT / "data/calibration/station_stacking_v23_kdal_1pm_bucket_loss",
    )
    parser.add_argument(
        "--selector-dir",
        type=Path,
        default=PROJECT_ROOT / "data/calibration/station_stacking_v23_kdal_1pm_bucket_loss_selector",
    )
    return parser.parse_args()


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    args = parse_args()
    pipeline = args.pipeline_dir.resolve()
    selector_dir = args.selector_dir.resolve()
    point_joblib = pipeline / "model_weights" / f"KDAL_{POINT_VERSION}.joblib"
    point_json = point_joblib.with_suffix(".json")
    selector_joblib = selector_dir / "model_weights" / f"KDAL_{SELECTOR_VERSION}.joblib"
    selector_json = selector_joblib.with_suffix(".json")
    point_manifest = json.loads(point_json.read_text(encoding="utf-8"))
    selector_manifest = json.loads(selector_json.read_text(encoding="utf-8"))
    selector = joblib.load(selector_joblib)
    point_sha = sha256_file(point_joblib)

    source_audit = json.loads(
        (PROJECT_ROOT / "data/calibration/station_stacking_v20_kdal_1pm_no_peak/audit/audit_result.json")
        .read_text(encoding="utf-8")
    )
    check(source_audit.get("passed") is True, "source pull audit failed")
    check(source_audit.get("timing_mode") == "same_day_1pm_live_safe", "source timing mismatch")
    check(int(source_audit.get("blocking_issue_count", -1)) == 0, "source audit has blockers")

    contract = point_manifest["model_contract"]
    expected_contract = {
        "timing_mode": "same_day_1pm_live_safe",
        "feature_version": "v20_kdal_1pm_no_peak",
        "training_profile": "v20_aligned",
        "optuna_metric": "bucket_log_loss",
        "target_mode": "remaining_warmup",
        "target_source": "wunderground_only",
        "final_model_method": "ridge_stack",
    }
    check(point_manifest["station_id"] == "KDAL", "point station mismatch")
    check(point_manifest["model_version"] == POINT_VERSION, "point version mismatch")
    for name, expected in expected_contract.items():
        check(contract.get(name) == expected, f"point contract mismatch for {name}")
    check(
        point_manifest["source_pipeline"].endswith("station_stacking_v23_kdal_1pm_bucket_loss"),
        "point source-pipeline mismatch",
    )
    check(point_manifest["artifact_integrity"]["bundle_sha256"] == point_sha, "point hash mismatch")

    features = pd.read_csv(pipeline / "KDAL_features.csv", low_memory=False)
    feature_columns = set(pd.read_csv(pipeline / "KDAL_feature_columns.csv")["feature"].astype(str))
    validation = pd.read_csv(pipeline / "KDAL_year_split_validation_predictions.csv")
    test = pd.read_csv(pipeline / "KDAL_year_split_test_predictions.csv")
    check(features["station_id"].astype(str).eq("KDAL").all(), "mixed station rows")
    check(set(V20_KDAL_1PM_TEMP_FEATURE_COLUMNS) <= feature_columns, "missing 1 PM point features")
    check(not any(name.startswith("v11sf_") for name in feature_columns), "11 AM feature leakage")
    peak = set(V20_PEAK_TIMING_RAW_FEATURE_COLUMNS) | set(V20_ENGINEERED_FEATURE_COLUMNS)
    check(feature_columns.isdisjoint(peak), "peak-timing feature leakage")
    for provider in ("gfs", "hrrr", "nbm"):
        labels = features[f"{provider}_source_label"].dropna().astype(str)
        check(labels.str.contains("same_day_1pm_live_safe", regex=False).all(), f"{provider} timing mismatch")
    clock = features["observed_as_of_time_local"].astype(str).str.extract(r"T(?P<hour>\d{2}):(?P<minute>\d{2})")
    minutes = pd.to_numeric(clock["hour"], errors="coerce") * 60 + pd.to_numeric(clock["minute"], errors="coerce")
    check(minutes.dropna().between(12 * 60 + 50, 13 * 60 + 10).all(), "observation-time mismatch")

    point_oof = crossfit_ridge_predictions(validation)
    check(set(point_oof["validation_year"].astype(int)) == {2023, 2024, 2025}, "point OOF years mismatch")
    check(
        point_oof["train_through_year"].astype(int).lt(point_oof["validation_year"].astype(int)).all(),
        "point OOF chronology leakage",
    )
    check(pd.to_datetime(point_oof["contract_date"]).dt.year.max() <= 2025, "2026 in point OOF")
    test_ridge = test.loc[test["method"].eq("ridge_stack")].copy()
    check(set(pd.to_datetime(test_ridge["contract_date"]).dt.year) == {2026}, "point holdout year mismatch")

    check(selector_manifest["station_id"] == "KDAL", "selector station mismatch")
    check(selector_manifest["model_version"] == SELECTOR_VERSION, "selector version mismatch")
    check(selector_manifest["feature_profile"] == FEATURE_PROFILE_KDAL_1PM, "selector profile mismatch")
    check(selector_manifest["point_model_version"] == POINT_VERSION, "selector point version mismatch")
    check(selector_manifest["point_bundle_sha256"] == point_sha, "selector bound to wrong point bundle")
    check(selector_manifest["artifact_integrity"]["bundle_sha256"] == sha256_file(selector_joblib), "selector hash mismatch")
    selected_features = set(selector["feature_names"])
    check(set(V20_KDAL_1PM_TEMP_FEATURE_COLUMNS) <= selected_features, "selector missing 1 PM features")
    check(not any(name.startswith("v11sf_") for name in selected_features), "selector 11 AM leakage")
    check(selected_features.isdisjoint(peak), "selector peak-feature leakage")

    forward = pd.read_csv(selector_dir / "KDAL_forward_win_predictions.csv")
    check(set(forward["validation_year"].astype(int)) == {2024, 2025}, "selector forward years mismatch")
    dates = pd.to_datetime(forward["contract_date"], errors="raise")
    check(pd.to_datetime(forward["model_training_cutoff"]).lt(dates).all(), "selector model chronology leakage")
    check(pd.to_datetime(forward["calibration_cutoff"]).lt(dates).all(), "selector calibration leakage")
    stored_policy = selector["confidence_policy"]
    rebuilt_policy = select_confidence_threshold(threshold_metrics(forward))
    check(stored_policy == rebuilt_policy, "stored selector threshold does not reproduce from forward rows")
    check(stored_policy["holdout_rows_used_for_selection"] == 0, "holdout used for threshold selection")
    check("2026" not in stored_policy["selection_data"], "threshold policy names holdout data")

    holdout = pd.read_csv(selector_dir / "KDAL_2026_win_predictions.csv")
    check(set(pd.to_datetime(holdout["contract_date"]).dt.year) == {2026}, "selector holdout year mismatch")
    check(len(holdout) == len(test_ridge), "selector/point holdout row mismatch")
    expected_decision = holdout["win_probability"].ge(float(stored_policy["threshold"]))
    check(
        expected_decision.eq(holdout["confidence_decision"].eq("eligible")).all(),
        "holdout decision/threshold mismatch",
    )

    holdout_frame = build_win_frame(
        features,
        test_ridge[["contract_date", "actual_high_f", "predicted_high_f"]],
        test,
        include_peak_features=False,
        feature_profile=FEATURE_PROFILE_KDAL_1PM,
    )
    smoke = holdout_frame.loc[holdout_frame["year"].eq(2026)].iloc[0].to_dict()
    result = predict_win_bundle(selector, smoke)
    check(result["status"] == "ok", "serialized selector inference failed")
    check(result["confidence_threshold"] == float(stored_policy["threshold"]), "serving threshold mismatch")

    selected = expected_decision
    summary = {
        "passed": True,
        "station": "KDAL",
        "point_model_version": POINT_VERSION,
        "selector_model_version": SELECTOR_VERSION,
        "timing_mode": contract["timing_mode"],
        "feature_version": contract["feature_version"],
        "optuna_metric": contract["optuna_metric"],
        "forward_rows": int(len(forward)),
        "holdout_rows": int(len(holdout)),
        "confidence_policy": stored_policy,
        "holdout_selected_count": int(selected.sum()),
        "holdout_coverage": float(selected.mean()),
        "holdout_selected_win_rate": float(holdout.loc[selected, "bucket_win"].mean()) if selected.any() else None,
        "checks": {
            "source_pull": "passed",
            "point_contract_and_hash": "passed",
            "point_feature_and_timing_profile": "passed",
            "point_oof_chronology": "passed",
            "2026_point_isolation": "passed",
            "selector_contract_and_hash": "passed",
            "selector_1pm_no_peak_profile": "passed",
            "selector_forward_chronology": "passed",
            "forward_only_threshold_reproduction": "passed",
            "2026_selector_isolation": "passed",
            "serialized_inference_parity": "passed",
        },
    }
    audit_dir = selector_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "audit_result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
