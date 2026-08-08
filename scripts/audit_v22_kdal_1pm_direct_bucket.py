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

from src.calibration.bucket_correction import (
    add_bucket_correction_targets,
    apply_override_policy,
    predict_bucket_correction,
)
from src.calibration.bucket_probability import (
    FEATURE_PROFILE_KDAL_1PM,
    build_probability_frame,
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
BUCKET_VERSION = "station_bucket_v22_kdal_1pm_direct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit V22 KDAL direct bucket training")
    parser.add_argument(
        "--pipeline-dir",
        type=Path,
        default=PROJECT_ROOT / "data/calibration/station_stacking_v20_kdal_1pm_no_peak",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data/calibration/station_stacking_v22_kdal_1pm_direct_bucket",
    )
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    args = parse_args()
    pipeline = args.pipeline_dir.resolve()
    output = args.output_dir.resolve()
    point_joblib = pipeline / "model_weights" / f"KDAL_{POINT_VERSION}.joblib"
    point_manifest_path = pipeline / "model_weights" / f"KDAL_{POINT_VERSION}.json"
    bucket_joblib = output / "model_weights" / f"KDAL_{BUCKET_VERSION}.joblib"
    bucket_manifest_path = output / "model_weights" / f"KDAL_{BUCKET_VERSION}.json"

    pull_audit = json.loads((pipeline / "audit/audit_result.json").read_text(encoding="utf-8"))
    point_manifest = json.loads(point_manifest_path.read_text(encoding="utf-8"))
    bucket_manifest = json.loads(bucket_manifest_path.read_text(encoding="utf-8"))
    bucket = joblib.load(bucket_joblib)

    require(pull_audit["passed"] and pull_audit["blocking_issue_count"] == 0, "source pull failed")
    require(pull_audit["timing_mode"] == "same_day_1pm_live_safe", "source timing mismatch")
    contract = point_manifest["model_contract"]
    require(contract["timing_mode"] == "same_day_1pm_live_safe", "point timing mismatch")
    require(contract["feature_version"] == "v20_kdal_1pm_no_peak", "point features mismatch")
    require(contract["target_mode"] == "remaining_warmup", "point target mismatch")
    require(contract["target_source"] == "wunderground_only", "point target source mismatch")
    require(contract["training_profile"] == "v20_aligned", "point profile mismatch")
    point_sha = sha256_file(point_joblib)
    require(point_manifest["artifact_integrity"]["bundle_sha256"] == point_sha, "point hash mismatch")

    require(bucket_manifest["station_id"] == "KDAL", "bucket station mismatch")
    require(bucket_manifest["model_version"] == BUCKET_VERSION, "bucket version mismatch")
    require(bucket_manifest["point_model_version"] == POINT_VERSION, "point version binding mismatch")
    require(bucket_manifest["point_bundle_sha256"] == point_sha, "point hash binding mismatch")
    require(bucket_manifest["feature_profile"] == FEATURE_PROFILE_KDAL_1PM, "feature profile mismatch")
    require(bucket_manifest["artifact_integrity"]["bundle_sha256"] == sha256_file(bucket_joblib), "bucket hash mismatch")
    require(bucket["relation_labels"] == ["lower", "same", "upper"], "not a direct three-class relation model")

    feature_names = set(bucket["feature_names"])
    required_1pm = set(V20_KDAL_1PM_TEMP_FEATURE_COLUMNS) | {
        "observed_temp_change_last_1h_f",
        "observed_temp_change_last_3h_f",
        "observed_temp_change_since_11am_f",
        "observed_high_so_far_change_since_11am_f",
    }
    require(required_1pm <= feature_names, "missing 1 PM features")
    require(not any(name.startswith("v11sf_") for name in feature_names), "11 AM alignment leakage")
    peak = set(V20_PEAK_TIMING_RAW_FEATURE_COLUMNS) | set(V20_ENGINEERED_FEATURE_COLUMNS)
    require(feature_names.isdisjoint(peak), "peak feature leakage")

    features = pd.read_csv(pipeline / "KDAL_features.csv", low_memory=False)
    validation = pd.read_csv(pipeline / "KDAL_year_split_validation_predictions.csv")
    test = pd.read_csv(pipeline / "KDAL_year_split_test_predictions.csv")
    for provider in ("gfs", "hrrr", "nbm"):
        labels = features[f"{provider}_source_label"].dropna().astype(str)
        require(labels.str.contains("same_day_1pm_live_safe", regex=False).all(), f"{provider} timing leakage")
    clock = features["observed_as_of_time_local"].astype(str).str.extract(r"T(?P<hour>\d{2}):(?P<minute>\d{2})")
    minutes = pd.to_numeric(clock["hour"], errors="coerce") * 60 + pd.to_numeric(clock["minute"], errors="coerce")
    require(minutes.dropna().between(770, 790).all(), "observation outside 12:50-13:10")

    point_oof = crossfit_ridge_predictions(validation)
    require(set(point_oof["validation_year"].astype(int)) == {2023, 2024, 2025}, "unexpected point OOF years")
    require((point_oof["train_through_year"] < point_oof["validation_year"]).all(), "point OOF leakage")
    frame = add_bucket_correction_targets(
        build_probability_frame(
            features,
            point_oof,
            validation,
            include_peak_features=False,
            feature_profile=FEATURE_PROFILE_KDAL_1PM,
        )
    )
    raw_point_index = frame["point_prediction_f"].map(round_half_up).astype(int) // 2
    observed_index = frame["observed_high_temp_through_as_of_f"].map(round_half_up).astype(int) // 2
    expected_point_index = pd.concat([raw_point_index, observed_index], axis=1).max(axis=1).astype(int)
    expected_actual_index = frame["actual_high_f"].map(round_half_up).astype(int) // 2
    expected_delta = expected_actual_index - expected_point_index
    expected_relation = np.select([expected_delta.lt(0), expected_delta.gt(0)], [0, 2], default=1)
    require(expected_point_index.eq(frame["point_bucket_index"]).all(), "point bucket target mismatch")
    require(expected_actual_index.eq(frame["actual_bucket_index"]).all(), "actual bucket target mismatch")
    require(expected_delta.eq(frame["bucket_delta"]).all(), "bucket delta mismatch")
    require(np.array_equal(expected_relation.astype(int), frame["bucket_relation_class"].to_numpy(dtype=int)), "relation target mismatch")
    require((expected_delta.ne(0).astype(int) == frame["point_bucket_wrong"]).all(), "risk target mismatch")
    require(frame["year"].max() <= 2025, "2026 leaked into development")

    forward = pd.read_csv(output / "KDAL_forward_bucket_correction_predictions.csv")
    require(set(forward["validation_year"].astype(int)) == {2024, 2025}, "unexpected correction folds")
    forward_dates = pd.to_datetime(forward["contract_date"], errors="raise")
    require((pd.to_datetime(forward["model_training_cutoff"]) < forward_dates).all(), "correction model leakage")
    require((pd.to_datetime(forward["calibration_validation_cutoff"]) < forward_dates).all(), "calibration leakage")
    require(pd.Timestamp(bucket["training_cutoff"]).year == 2025, "holdout included in final fit")
    relation = forward["relation_probabilities"].map(json.loads)
    require(relation.map(lambda values: abs(sum(values) - 1.0) < 1e-9).all(), "relation probabilities not normalized")

    physical_violations = 0
    for row, probabilities in zip(forward.to_dict(orient="records"), relation, strict=True):
        decision = apply_override_policy(
            float(row["risk_probability"]),
            probabilities,
            bucket["decision_thresholds"],
            point_bucket_index=int(row["point_bucket_index"]),
            observed_high_f=float(row["observed_high_temp_through_as_of_f"]),
        )
        recommended = int(row["point_bucket_index"]) + int(decision["direction"])
        observed_floor = round_half_up(row["observed_high_temp_through_as_of_f"]) // 2
        physical_violations += int(recommended < observed_floor)
    require(physical_violations == 0, "forward override violates observed-high floor")

    holdout_metrics = pd.read_csv(output / "KDAL_2026_bucket_correction_holdout_metrics.csv")
    require(len(holdout_metrics) == 1 and int(holdout_metrics.iloc[0]["count"]) > 0, "missing holdout")
    point_test = test.loc[test["method"].eq("ridge_stack"), ["contract_date", "actual_high_f", "predicted_high_f"]]
    holdout = add_bucket_correction_targets(
        build_probability_frame(
            features,
            point_test,
            test,
            include_peak_features=False,
            feature_profile=FEATURE_PROFILE_KDAL_1PM,
        )
    )
    holdout = holdout.loc[holdout["year"].eq(2026)].copy()
    require(len(holdout) == int(holdout_metrics.iloc[0]["count"]), "holdout count mismatch")
    smoke = holdout.iloc[0].to_dict()
    smoke["contract_date"] = pd.Timestamp(smoke["contract_date"]).date().isoformat()
    prediction = predict_bucket_correction(bucket, smoke)
    require(prediction["status"] == "ok", "serialized inference failed")
    require(abs(sum(prediction["relation_probabilities"].values()) - 1.0) < 1e-9, "inference probabilities not normalized")

    result = {
        "passed": True,
        "station": "KDAL",
        "timing_mode": "same_day_1pm_live_safe",
        "point_model_version": POINT_VERSION,
        "bucket_model_version": BUCKET_VERSION,
        "feature_profile": FEATURE_PROFILE_KDAL_1PM,
        "relation_labels": ["lower", "same", "upper"],
        "source_feature_rows": int(len(features)),
        "point_oof_rows": int(len(point_oof)),
        "development_rows": int(len(frame)),
        "forward_rows": int(len(forward)),
        "holdout_rows": int(len(holdout)),
        "historical_acceptance": bucket.get("historical_acceptance", {}),
        "checks": {
            "source_pull": "passed",
            "point_contract_and_hash": "passed",
            "direct_three_class_target": "passed",
            "1pm_feature_profile": "passed",
            "no_11am_alignment_leakage": "passed",
            "no_peak_timing_leakage": "passed",
            "point_oof_chronology": "passed",
            "correction_forward_chronology": "passed",
            "2026_holdout_isolation": "passed",
            "observed_high_physical_floor": "passed",
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
