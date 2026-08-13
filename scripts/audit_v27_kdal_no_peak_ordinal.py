from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.bucket_probability import sha256_file


STATION_ID = "KDAL"
POINT_MODEL_VERSION = "station_high_regressor_v20_kdal_no_peak_stack"
POINT_BUNDLE = (
    PROJECT_ROOT
    / "data"
    / "calibration"
    / "station_stacking_v20_kdal_no_peak"
    / "model_weights"
    / "KDAL_station_high_regressor_v20_kdal_no_peak_stack.joblib"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "calibration"
    / "station_stacking_v27_kdal_no_peak_ordinal"
)
ARMS = {
    "pure_ordinal": "station_bucket_v27_kdal_no_peak_ordinal_pure",
    "ordinal_empirical_blend": "station_bucket_v27_kdal_no_peak_ordinal_blend",
}


def main() -> int:
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    point_hash = sha256_file(POINT_BUNDLE)
    for arm, model_version in ARMS.items():
        arm_dir = OUTPUT_DIR / arm
        manifest_path = (
            arm_dir / "model_weights" / f"{STATION_ID}_{model_version}.json"
        )
        bundle_path = (
            arm_dir / "model_weights" / f"{STATION_ID}_{model_version}.joblib"
        )
        predictions_path = arm_dir / f"{STATION_ID}_forward_probability_predictions.csv"
        holdout_predictions_path = (
            arm_dir / f"{STATION_ID}_2026_probability_holdout_predictions.csv"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        predictions = pd.read_csv(predictions_path)
        probabilities = np.vstack(
            predictions["offset_probabilities"].map(json.loads).to_numpy()
        )
        contract_dates = pd.to_datetime(predictions["contract_date"])
        training_cutoffs = pd.to_datetime(predictions["model_training_cutoff"])
        calibration_cutoffs = pd.to_datetime(
            predictions["calibration_validation_cutoff"]
        )

        check(
            f"{arm}:bundle_hash",
            sha256_file(bundle_path)
            == manifest["artifact_integrity"]["bundle_sha256"],
            str(bundle_path.relative_to(PROJECT_ROOT)),
        )
        check(
            f"{arm}:point_identity",
            manifest["point_model_version"] == POINT_MODEL_VERSION
            and manifest["point_bundle_sha256"] == point_hash,
            manifest["point_model_version"],
        )
        check(
            f"{arm}:forced_ordinal",
            manifest["selected_family"] == "ordinal_logistic"
            and manifest["family_selection_mode"] == "forced"
            and manifest["forced_family"] == "ordinal_logistic",
            f"family={manifest['selected_family']}",
        )
        check(
            f"{arm}:no_peak_profile",
            manifest["feature_profile"] == "common_no_peak"
            and not any("peak" in name.lower() for name in manifest["feature_names"]),
            f"features={len(manifest['feature_names'])}",
        )
        check(
            f"{arm}:probability_simplex",
            bool(
                np.isfinite(probabilities).all()
                and (probabilities >= 0.0).all()
                and np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-10)
            ),
            f"rows={len(probabilities)}",
        )
        cumulative = probabilities.cumsum(axis=1)
        check(
            f"{arm}:ordered_cdf",
            bool((np.diff(cumulative, axis=1) >= -1e-12).all()),
            "cumulative class probabilities are monotone",
        )
        check(
            f"{arm}:forward_years",
            set(predictions["validation_year"].astype(int)) == {2024, 2025},
            str(sorted(predictions["validation_year"].unique().tolist())),
        )
        check(
            f"{arm}:chronology",
            bool(
                (training_cutoffs < contract_dates).all()
                and (calibration_cutoffs < contract_dates).all()
            ),
            "all model and calibration cutoffs precede scored dates",
        )
        check(
            f"{arm}:exploratory_holdout",
            manifest["holdout_status"] == "exploratory"
            and not manifest["historical_acceptance"]["passed"]
            and "fresh_shadow_data_required"
            in manifest["historical_acceptance"]["reasons"],
            manifest["holdout_status"],
        )
        holdout_predictions = pd.read_csv(holdout_predictions_path)
        holdout_bucket_probabilities = holdout_predictions[
            "bucket_probabilities"
        ].map(json.loads)
        check(
            f"{arm}:holdout_prediction_export",
            len(holdout_predictions) == int(manifest["holdout_metrics"]["count"])
            and {
                "contract_date",
                "point_prediction_f",
                "recommended_bucket",
                "recommended_bucket_probability",
                "offset_probabilities",
                "degree_probabilities",
                "bucket_probabilities",
            }.issubset(holdout_predictions.columns),
            str(holdout_predictions_path.relative_to(PROJECT_ROOT)),
        )
        check(
            f"{arm}:holdout_bucket_probability_simplex",
            all(
                np.isclose(sum(probabilities.values()), 1.0, atol=1e-10)
                for probabilities in holdout_bucket_probabilities
            ),
            f"rows={len(holdout_predictions)}",
        )

        if arm == "pure_ordinal":
            check(
                "pure_ordinal:fixed_weight",
                manifest["blend_weight"] == 1.0
                and manifest["blend_weight_candidates"] == [1.0],
                f"weight={manifest['blend_weight']}",
            )
        else:
            candidates = manifest["blend_weight_candidates"]
            check(
                "ordinal_empirical_blend:weight_grid",
                candidates == [0.25, 0.5, 0.75, 1.0]
                and manifest["blend_weight"] in candidates,
                f"selected={manifest['blend_weight']}",
            )

    check(
        "comparison_exists",
        (OUTPUT_DIR / f"{STATION_ID}_comparison.csv").is_file(),
        "arm comparison",
    )
    check(
        "summary_blocks_promotion",
        not json.loads(
            (OUTPUT_DIR / f"{STATION_ID}_summary.json").read_text(encoding="utf-8")
        )["promotion_approved"],
        "fresh shadow data required",
    )
    passed = all(bool(item["passed"]) for item in checks)
    result = {
        "experiment": "v27_kdal_no_peak_ordinal",
        "passed": passed,
        "check_count": len(checks),
        "passed_count": sum(bool(item["passed"]) for item in checks),
        "checks": checks,
    }
    audit_dir = OUTPUT_DIR / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "audit_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
