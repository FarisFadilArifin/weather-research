from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.bucket_probability import (
    FEATURE_PROFILE_KDAL_1PM,
    build_probability_frame,
    predict_probability_bundle,
    round_half_up,
    sha256_file,
)
from src.calibration.station_stacking import (
    V20_ENGINEERED_FEATURE_COLUMNS,
    V20_KDAL_1PM_TEMP_FEATURE_COLUMNS,
    V20_PEAK_TIMING_RAW_FEATURE_COLUMNS,
)
from src.calibration.v19_bucket import crossfit_ridge_predictions


POINT_VERSION = "station_high_regressor_v20_kdal_1pm_no_peak_stack"
BUCKET_VERSION = "station_bucket_v21_kdal_1pm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the V21 KDAL 1 PM bucket training pipeline")
    parser.add_argument(
        "--pipeline-dir",
        type=Path,
        default=PROJECT_ROOT / "data/calibration/station_stacking_v20_kdal_1pm_no_peak",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data/calibration/station_stacking_v21_kdal_1pm_bucket",
    )
    return parser.parse_args()


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    args = parse_args()
    pipeline = args.pipeline_dir.resolve()
    output = args.output_dir.resolve()
    point_dir = pipeline / "model_weights"
    point_joblib = point_dir / f"KDAL_{POINT_VERSION}.joblib"
    point_manifest_path = point_dir / f"KDAL_{POINT_VERSION}.json"
    bucket_dir = output / "model_weights"
    bucket_joblib = bucket_dir / f"KDAL_{BUCKET_VERSION}.joblib"
    bucket_manifest_path = bucket_dir / f"KDAL_{BUCKET_VERSION}.json"

    point_manifest = json.loads(point_manifest_path.read_text(encoding="utf-8"))
    bucket_manifest = json.loads(bucket_manifest_path.read_text(encoding="utf-8"))
    bucket = joblib.load(bucket_joblib)
    pull_audit = json.loads((pipeline / "audit/audit_result.json").read_text(encoding="utf-8"))

    _assert(bool(pull_audit.get("passed")), "source 1 PM pull audit did not pass")
    _assert(pull_audit.get("timing_mode") == "same_day_1pm_live_safe", "pull timing mismatch")
    _assert(int(pull_audit.get("blocking_issue_count", -1)) == 0, "blocking source-data issues")

    contract = point_manifest["model_contract"]
    _assert(point_manifest["station_id"] == "KDAL", "point station mismatch")
    _assert(point_manifest["model_version"] == POINT_VERSION, "point version mismatch")
    _assert(contract["timing_mode"] == "same_day_1pm_live_safe", "point timing mismatch")
    _assert(contract["feature_version"] == "v20_kdal_1pm_no_peak", "point feature version mismatch")
    _assert(contract["target_mode"] == "remaining_warmup", "point target mismatch")
    _assert(contract["target_source"] == "wunderground_only", "point target source mismatch")
    _assert(contract["training_profile"] == "v20_aligned", "point training profile mismatch")
    _assert(point_manifest["source_pipeline"].endswith("station_stacking_v20_kdal_1pm_no_peak"), "point source mismatch")
    point_sha = sha256_file(point_joblib)
    _assert(point_manifest["artifact_integrity"]["bundle_sha256"] == point_sha, "point bundle hash mismatch")

    _assert(bucket_manifest["station_id"] == "KDAL", "bucket station mismatch")
    _assert(bucket_manifest["model_version"] == BUCKET_VERSION, "bucket version mismatch")
    _assert(bucket_manifest["feature_profile"] == FEATURE_PROFILE_KDAL_1PM, "bucket profile mismatch")
    _assert(bucket_manifest["point_model_version"] == POINT_VERSION, "bucket point-version mismatch")
    _assert(bucket_manifest["point_bundle_sha256"] == point_sha, "bucket is bound to wrong point bundle")
    _assert(bucket_manifest["artifact_integrity"]["bundle_sha256"] == sha256_file(bucket_joblib), "bucket hash mismatch")
    _assert(bucket["feature_profile"] == FEATURE_PROFILE_KDAL_1PM, "serialized bucket profile mismatch")
    _assert(bucket["point_bundle_sha256"] == point_sha, "serialized point hash mismatch")

    selected_features = set(bucket["feature_names"])
    required_1pm = set(V20_KDAL_1PM_TEMP_FEATURE_COLUMNS) | {
        "observed_temp_change_last_1h_f",
        "observed_temp_change_last_3h_f",
        "observed_temp_change_since_11am_f",
        "observed_high_so_far_change_since_11am_f",
    }
    _assert(required_1pm <= selected_features, "missing 1 PM bucket features: " + str(sorted(required_1pm - selected_features)))
    _assert(not any(name.startswith("v11sf_") for name in selected_features), "11 AM alignment feature leakage")
    peak_features = set(V20_PEAK_TIMING_RAW_FEATURE_COLUMNS) | set(V20_ENGINEERED_FEATURE_COLUMNS)
    _assert(selected_features.isdisjoint(peak_features), "peak-timing feature leakage")

    features = pd.read_csv(pipeline / "KDAL_features.csv", low_memory=False)
    validation = pd.read_csv(pipeline / "KDAL_year_split_validation_predictions.csv")
    test = pd.read_csv(pipeline / "KDAL_year_split_test_predictions.csv")
    _assert(features["station_id"].astype(str).eq("KDAL").all(), "feature station mismatch")
    for provider in ("gfs", "hrrr", "nbm"):
        labels = features[f"{provider}_source_label"].dropna().astype(str)
        _assert(labels.str.contains("same_day_1pm_live_safe", regex=False).all(), f"{provider} timing leakage")
    observed_clock = features["observed_as_of_time_local"].astype(str).str.extract(r"T(?P<hour>\d{2}):(?P<minute>\d{2})")
    local_minutes = pd.to_numeric(observed_clock["hour"], errors="coerce") * 60 + pd.to_numeric(
        observed_clock["minute"], errors="coerce"
    )
    _assert(local_minutes.dropna().between(12 * 60 + 50, 13 * 60 + 10).all(), "observation outside 12:50-13:10")

    point_oof = crossfit_ridge_predictions(validation)
    _assert(set(point_oof["validation_year"].astype(int)) == {2023, 2024, 2025}, "unexpected point OOF years")
    _assert((point_oof["train_through_year"].astype(int) < point_oof["validation_year"].astype(int)).all(), "ridge OOF leakage")
    probability_frame = build_probability_frame(
        features,
        point_oof,
        validation,
        include_peak_features=False,
        feature_profile=FEATURE_PROFILE_KDAL_1PM,
    )
    expected_offset = probability_frame.apply(
        lambda row: round_half_up(row["actual_high_f"]) - round_half_up(row["point_prediction_f"]), axis=1
    )
    _assert(expected_offset.eq(probability_frame["exact_offset"].astype(int)).all(), "bucket target arithmetic mismatch")
    _assert(probability_frame["year"].max() <= 2025, "2026 leaked into bucket development frame")

    forward = pd.read_csv(output / "KDAL_forward_probability_predictions.csv")
    _assert(set(forward["validation_year"].astype(int)) == {2024, 2025}, "unexpected forward bucket folds")
    forward_dates = pd.to_datetime(forward["contract_date"], errors="raise")
    _assert((pd.to_datetime(forward["model_training_cutoff"]) < forward_dates).all(), "model fold leakage")
    _assert((pd.to_datetime(forward["calibration_validation_cutoff"]) < forward_dates).all(), "calibration fold leakage")
    _assert(pd.Timestamp(bucket["training_cutoff"]).year == 2025, "bucket final fit includes holdout")

    holdout_metrics = pd.read_csv(output / "KDAL_2026_probability_holdout_metrics.csv")
    _assert(len(holdout_metrics) == 1 and int(holdout_metrics.iloc[0]["count"]) > 0, "missing 2026 holdout")
    point_test = test.loc[
        test["method"].eq("ridge_stack"), ["contract_date", "actual_high_f", "predicted_high_f"]
    ]
    holdout_frame = build_probability_frame(
        features,
        point_test,
        test,
        include_peak_features=False,
        feature_profile=FEATURE_PROFILE_KDAL_1PM,
    )
    holdout_frame = holdout_frame.loc[holdout_frame["year"].eq(2026)].copy()
    _assert(len(holdout_frame) == int(holdout_metrics.iloc[0]["count"]), "holdout row-count mismatch")
    smoke_values = holdout_frame.iloc[0].to_dict()
    smoke_values["contract_date"] = pd.Timestamp(smoke_values["contract_date"]).date().isoformat()
    prediction = predict_probability_bundle(bucket, smoke_values)
    _assert(prediction["status"] == "ok", "serialized bucket smoke prediction failed")
    _assert(abs(sum(prediction["bucket_probabilities"].values()) - 1.0) < 1e-9, "bucket probabilities do not sum to one")
    _assert(
        min(int(label.split("-", 1)[0]) for label in prediction["bucket_probabilities"])
        >= 2 * (round_half_up(smoke_values["observed_high_temp_through_as_of_f"]) // 2),
        "physical observed-high floor violated",
    )

    result = {
        "passed": True,
        "station": "KDAL",
        "timing_mode": "same_day_1pm_live_safe",
        "point_model_version": POINT_VERSION,
        "bucket_model_version": BUCKET_VERSION,
        "feature_profile": FEATURE_PROFILE_KDAL_1PM,
        "source_feature_rows": int(len(features)),
        "point_oof_rows": int(len(point_oof)),
        "bucket_development_rows": int(len(probability_frame)),
        "forward_rows": int(len(forward)),
        "holdout_rows": int(len(holdout_frame)),
        "selected_family": bucket["selected_family"],
        "historical_acceptance": bucket.get("historical_acceptance", {}),
        "checks": {
            "source_pull": "passed",
            "point_contract_and_hash": "passed",
            "1pm_feature_profile": "passed",
            "no_11am_alignment_leakage": "passed",
            "no_peak_timing_leakage": "passed",
            "point_oof_chronology": "passed",
            "bucket_target_arithmetic": "passed",
            "bucket_forward_chronology": "passed",
            "2026_holdout_isolation": "passed",
            "serialized_prediction_smoke": "passed",
        },
    }
    audit_dir = output / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "audit_result.json").write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
