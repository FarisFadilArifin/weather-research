from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.bucket_probability import probability_metrics, sha256_file


STATION_ID = "KDAL"
POINT_MODEL_VERSION = "station_high_regressor_v20_kdal_no_peak_stack"
PIPELINE_DIR = (
    PROJECT_ROOT / "data" / "calibration" / "station_stacking_v20_kdal_no_peak"
)
POINT_BUNDLE = (
    PIPELINE_DIR
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
    "pure_ordinal": {
        "blend_weights": "1.0",
        "model_version": "station_bucket_v27_kdal_no_peak_ordinal_pure",
    },
    "ordinal_empirical_blend": {
        "blend_weights": "0.25,0.5,0.75,1.0",
        "model_version": "station_bucket_v27_kdal_no_peak_ordinal_blend",
    },
}


def _run_arm(arm: str, config: dict[str, str]) -> tuple[Path, Path]:
    arm_dir = OUTPUT_DIR / arm
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train-bucket-probability.py"),
        "--station",
        STATION_ID,
        "--pipeline-dir",
        str(PIPELINE_DIR),
        "--point-bundle",
        str(POINT_BUNDLE),
        "--point-model-version",
        POINT_MODEL_VERSION,
        "--model-version",
        config["model_version"],
        "--feature-profile",
        "common_no_peak",
        "--force-family",
        "ordinal_logistic",
        "--blend-weights",
        config["blend_weights"],
        "--holdout-status",
        "exploratory",
        "--output-dir",
        str(arm_dir),
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    manifest = (
        arm_dir
        / "model_weights"
        / f"{STATION_ID}_{config['model_version']}.json"
    )
    predictions = arm_dir / f"{STATION_ID}_forward_probability_predictions.csv"
    if not manifest.is_file() or not predictions.is_file():
        raise FileNotFoundError(f"{arm} did not produce its required artifacts")
    return manifest, predictions


def _comparison_row(arm: str, manifest: dict) -> dict[str, object]:
    forward = manifest["forward_metrics"]
    policy = manifest["forward_policy_metrics"]
    holdout = manifest["holdout_metrics"]
    return {
        "arm": arm,
        "selected_family": manifest["selected_family"],
        "family_selection_mode": manifest["family_selection_mode"],
        "selected_blend_weight": manifest["blend_weight"],
        "blend_weight_candidates": json.dumps(manifest["blend_weight_candidates"]),
        "forward_count": forward["count"],
        "forward_offset_log_loss": forward["log_loss"],
        "forward_ranked_probability_score": forward[
            "ranked_probability_score"
        ],
        "forward_multiclass_brier": forward["brier"],
        "forward_offset_accuracy": forward["offset_accuracy"],
        "forward_offset_top_two_accuracy": forward["top_two_accuracy"],
        "forward_actionable_coverage": policy["coverage"],
        "forward_actionable_accuracy": policy["accuracy"],
        "forward_point_accuracy": policy["full_point_accuracy"],
        "forward_switch_count": policy["switch_count"],
        "holdout_status": manifest["holdout_status"],
        "holdout_count": holdout["count"],
        "holdout_bucket_log_loss": holdout["bucket_log_loss"],
        "holdout_ranked_probability_score": holdout[
            "ranked_probability_score"
        ],
        "holdout_offset_accuracy": holdout["offset_accuracy"],
        "holdout_offset_top_two_accuracy": holdout["offset_top_two_accuracy"],
        "holdout_probability_bucket_accuracy": holdout[
            "probability_bucket_accuracy"
        ],
        "holdout_point_bucket_accuracy": holdout["point_bucket_accuracy"],
        "holdout_switch_count": holdout["switch_count"],
        "historical_acceptance_passed": manifest["historical_acceptance"]["passed"],
    }


def _forward_year_metrics(arm: str, path: Path) -> pd.DataFrame:
    predictions = pd.read_csv(path)
    predictions["offset_probabilities"] = predictions["offset_probabilities"].map(
        json.loads
    )
    rows = []
    for validation_year, group in predictions.groupby("validation_year"):
        metrics = probability_metrics(group).iloc[0].to_dict()
        metrics.update(
            {
                "arm": arm,
                "validation_year": int(validation_year),
                "first_contract_date": group["contract_date"].min(),
                "last_contract_date": group["contract_date"].max(),
            }
        )
        rows.append(metrics)
    return pd.DataFrame(rows)


def main() -> int:
    if not POINT_BUNDLE.is_file():
        raise SystemExit(f"V20 point bundle not found: {POINT_BUNDLE}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifests: dict[str, dict] = {}
    year_metrics = []
    for arm, config in ARMS.items():
        manifest_path, prediction_path = _run_arm(arm, config)
        manifests[arm] = json.loads(manifest_path.read_text(encoding="utf-8"))
        year_metrics.append(_forward_year_metrics(arm, prediction_path))

    comparison = pd.DataFrame(
        [_comparison_row(arm, manifest) for arm, manifest in manifests.items()]
    )
    comparison.to_csv(OUTPUT_DIR / f"{STATION_ID}_comparison.csv", index=False)
    pd.concat(year_metrics, ignore_index=True).to_csv(
        OUTPUT_DIR / f"{STATION_ID}_forward_year_metrics.csv", index=False
    )

    frozen_at = datetime.now(timezone.utc)
    summary = {
        "station_id": STATION_ID,
        "experiment": "v20_no_peak_forced_ordinal_distribution",
        "point_model_version": POINT_MODEL_VERSION,
        "point_bundle_path": str(POINT_BUNDLE.relative_to(PROJECT_ROOT)),
        "point_bundle_sha256": sha256_file(POINT_BUNDLE),
        "feature_profile": "common_no_peak",
        "target": "ordered rounded-degree residual offset",
        "offset_classes": [
            "le_-4",
            "-3",
            "-2",
            "-1",
            "0",
            "+1",
            "+2",
            "+3",
            "ge_+4",
        ],
        "arms": {
            arm: {
                "model_version": config["model_version"],
                "blend_weight_candidates": [
                    float(value) for value in config["blend_weights"].split(",")
                ],
            }
            for arm, config in ARMS.items()
        },
        "development_folds": [2024, 2025],
        "holdout_status": "2026_exploratory_previously_inspected",
        "promotion_approved": False,
        "promotion_blocker": "fresh_shadow_data_required",
        "artifact_frozen_at_utc": frozen_at.isoformat(),
        "fresh_shadow_start_contract_date": (
            date.today() + timedelta(days=1)
        ).isoformat(),
    }
    (OUTPUT_DIR / f"{STATION_ID}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "audit_v27_kdal_no_peak_ordinal.py"),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    print(comparison.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
