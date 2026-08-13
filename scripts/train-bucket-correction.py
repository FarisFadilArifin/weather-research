from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.bucket_correction import (
    add_bucket_correction_targets,
    correction_metrics,
    export_bucket_correction_bundle,
    fit_bucket_correction_system,
    predict_bucket_correction,
)
from src.calibration.bucket_probability import (
    FEATURE_PROFILE_COMMON_NO_PEAK,
    FEATURE_PROFILE_KDAL_1PM,
    FEATURE_PROFILE_PEAK_AUGMENTED,
    build_probability_frame,
    sha256_file,
)
from src.calibration.v19_bucket import crossfit_ridge_predictions


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a two-stage point-bucket correction challenger"
    )
    parser.add_argument("--station", required=True, choices=("KATL", "KDAL"))
    parser.add_argument("--pipeline-dir", required=True, type=Path)
    parser.add_argument("--point-bundle", required=True, type=Path)
    parser.add_argument("--point-model-version", required=True)
    parser.add_argument("--model-version", default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--include-peak-features", action="store_true")
    parser.add_argument(
        "--feature-profile",
        choices=(FEATURE_PROFILE_COMMON_NO_PEAK, FEATURE_PROFILE_PEAK_AUGMENTED, FEATURE_PROFILE_KDAL_1PM),
        default=None,
    )
    args = parser.parse_args()

    station = args.station.upper()
    if station == "KDAL" and args.include_peak_features:
        raise SystemExit("KDAL bucket correction must not include peak features")
    if args.feature_profile == FEATURE_PROFILE_KDAL_1PM and station != "KDAL":
        raise SystemExit("kdal_1pm bucket-correction profile is limited to KDAL")
    if args.feature_profile == FEATURE_PROFILE_KDAL_1PM and args.include_peak_features:
        raise SystemExit("kdal_1pm bucket-correction profile cannot include peak features")
    if not args.point_bundle.is_file():
        raise SystemExit(f"point bundle not found: {args.point_bundle}")

    features = pd.read_csv(
        args.pipeline_dir / f"{station}_features.csv", low_memory=False
    )
    validation = pd.read_csv(
        args.pipeline_dir / f"{station}_year_split_validation_predictions.csv"
    )
    point_oof = crossfit_ridge_predictions(validation)
    profiles = (
        [(False, FEATURE_PROFILE_COMMON_NO_PEAK), (True, FEATURE_PROFILE_PEAK_AUGMENTED)]
        if args.include_peak_features
        else [(False, args.feature_profile or FEATURE_PROFILE_COMMON_NO_PEAK)]
    )
    profile_results = []
    for include_peak, feature_profile in profiles:
        frame = build_probability_frame(
            features,
            point_oof,
            validation,
            include_peak_features=include_peak,
            feature_profile=feature_profile,
        )
        bundle, predictions, tuning = fit_bucket_correction_system(
            frame,
            station_id=station,
            point_model_version=args.point_model_version,
            point_bundle_sha256=sha256_file(args.point_bundle),
            include_peak_features=include_peak,
            feature_profile=feature_profile,
            model_version=args.model_version,
        )
        profile_results.append((bundle, predictions, tuning, include_peak, feature_profile))

    comparison = pd.DataFrame(
        [
            {
                "feature_profile": bundle["feature_profile"],
                "stable_forward_evidence": bundle["decision_thresholds"][
                    "stable_forward_evidence"
                ],
                "corrected_bucket_accuracy": bundle["forward_metrics"][
                    "corrected_bucket_accuracy"
                ],
                "point_bucket_accuracy": bundle["forward_metrics"][
                    "point_bucket_accuracy"
                ],
                "switch_count": bundle["forward_metrics"]["switch_count"],
                "switch_accuracy": bundle["forward_metrics"]["switch_accuracy"],
                "point_accuracy_on_switches": bundle["forward_metrics"][
                    "point_accuracy_on_switches"
                ],
                "relation_log_loss": bundle["forward_metrics"][
                    "relation_log_loss"
                ],
                "profile_rank": 1 if include_peak else 0,
            }
            for bundle, _, _, include_peak, _ in profile_results
        ]
    )
    winner = comparison.sort_values(
        [
            "stable_forward_evidence",
            "corrected_bucket_accuracy",
            "switch_accuracy",
            "relation_log_loss",
            "profile_rank",
        ],
        ascending=[False, False, False, True, True],
    ).iloc[0]
    bundle, forward_predictions, tuning, selected_include_peak, selected_feature_profile = next(
        item
        for item in profile_results
        if item[0]["feature_profile"] == winner["feature_profile"]
    )
    bundle["profile_comparison"] = comparison.drop(
        columns=["profile_rank"]
    ).to_dict(orient="records")

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    serializable = forward_predictions.copy()
    serializable["relation_probabilities"] = serializable[
        "relation_probabilities"
    ].map(json.dumps)
    serializable.to_csv(
        output / f"{station}_forward_bucket_correction_predictions.csv", index=False
    )
    tuning.to_csv(
        output / f"{station}_bucket_correction_tuning.csv", index=False
    )
    comparison.drop(columns=["profile_rank"]).to_csv(
        output / f"{station}_bucket_correction_profile_comparison.csv", index=False
    )
    pd.DataFrame([bundle["forward_metrics"]]).to_csv(
        output / f"{station}_forward_bucket_correction_metrics.csv", index=False
    )

    holdout = evaluate_2026_holdout(
        args.pipeline_dir,
        station,
        features,
        bundle,
        selected_include_peak,
        selected_feature_profile,
    )
    holdout.to_csv(
        output / f"{station}_2026_bucket_correction_holdout_metrics.csv",
        index=False,
    )
    bundle["holdout_metrics"] = holdout.iloc[0].to_dict() if not holdout.empty else {}
    bundle["historical_acceptance"] = historical_acceptance(bundle, holdout)
    bundle_path, manifest_path = export_bucket_correction_bundle(
        bundle,
        output / "model_weights",
        source_identity=source_identity(),
    )
    print(
        json.dumps(
            {
                "station": station,
                "bundle": str(bundle_path),
                "manifest": str(manifest_path),
                "selected_feature_profile": bundle["feature_profile"],
                "forward": bundle["forward_metrics"],
                "holdout": bundle["holdout_metrics"],
                "historical_acceptance": bundle["historical_acceptance"],
            },
            indent=2,
            allow_nan=True,
        )
    )
    return 0


def evaluate_2026_holdout(
    pipeline_dir: Path,
    station: str,
    features: pd.DataFrame,
    bundle: dict,
    include_peak_features: bool,
    feature_profile: str | None = None,
) -> pd.DataFrame:
    test_path = pipeline_dir / f"{station}_year_split_test_predictions.csv"
    if not test_path.is_file():
        return pd.DataFrame()
    test = pd.read_csv(test_path)
    point = test.loc[
        test["method"].eq("ridge_stack"),
        ["contract_date", "actual_high_f", "predicted_high_f"],
    ]
    frame = add_bucket_correction_targets(
        build_probability_frame(
            features,
            point,
            test,
            include_peak_features=include_peak_features,
            feature_profile=feature_profile,
        )
    )
    frame = frame.loc[frame["year"].eq(2026)].copy()
    if frame.empty:
        return pd.DataFrame()

    rows = []
    for _, row in frame.iterrows():
        values = row.to_dict()
        values["contract_date"] = pd.Timestamp(row["contract_date"]).date().isoformat()
        prediction = predict_bucket_correction(bundle, values)
        if prediction["status"] != "ok":
            continue
        direction = int(prediction["override_direction"])
        delta = int(row["bucket_delta"])
        point_hit = delta == 0
        corrected_hit = delta == direction if direction else point_hit
        relation = prediction["relation_probabilities"]
        relation_values = np.asarray(
            [relation["lower"], relation["same"], relation["upper"]], dtype=float
        )
        actual_relation = int(row["bucket_relation_class"])
        rows.append(
            {
                "point_bucket_wrong": int(row["point_bucket_wrong"]),
                "bucket_relation_class": actual_relation,
                "bucket_delta": delta,
                "point_bucket_index": int(row["point_bucket_index"]),
                "observed_high_temp_through_as_of_f": float(
                    row["observed_high_temp_through_as_of_f"]
                ),
                "risk_probability": float(prediction["risk_probability"]),
                "relation_probabilities": relation_values,
                "point_hit": point_hit,
                "corrected_hit": corrected_hit,
                "switch": direction != 0,
                "switch_hit": direction != 0 and delta == direction,
            }
        )
    if not rows:
        return pd.DataFrame()
    evaluated = pd.DataFrame(rows)
    metric_input = pd.DataFrame(
        {
            "risk_probability": evaluated["risk_probability"],
            "relation_probabilities": evaluated["relation_probabilities"],
            "point_bucket_wrong": evaluated["point_bucket_wrong"],
            "bucket_relation_class": evaluated["bucket_relation_class"],
            "bucket_delta": evaluated["bucket_delta"],
            "point_bucket_index": evaluated["point_bucket_index"],
            "observed_high_temp_through_as_of_f": evaluated[
                "observed_high_temp_through_as_of_f"
            ],
        }
    )
    metrics = correction_metrics(metric_input, bundle["decision_thresholds"])
    return pd.DataFrame([metrics])


def historical_acceptance(
    bundle: Mapping[str, object], holdout: pd.DataFrame
) -> dict[str, object]:
    if holdout.empty:
        return {"passed": False, "reasons": ["missing_2026_holdout"]}
    row = holdout.iloc[0]
    gates = {
        "stableForwardSwitchEvidence": bool(
            bundle["decision_thresholds"]["stable_forward_evidence"]  # type: ignore[index]
        ),
        "holdoutCorrectedAccuracyNoWorseThanPoint": float(
            row["corrected_bucket_accuracy"]
        )
        >= float(row["point_bucket_accuracy"]),
        "holdoutHasAtLeastFiveSwitches": int(row["switch_count"]) >= 5,
        "holdoutSwitchesBeatPoint": int(row["switch_count"]) > 0
        and float(row["switch_accuracy"])
        > float(row["point_accuracy_on_switches"]),
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "reasons": [name for name, passed in gates.items() if not passed],
    }


def source_identity() -> dict[str, object]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    return {
        "git_commit": run("rev-parse", "HEAD"),
        "git_dirty": bool(run("status", "--porcelain")),
    }


if __name__ == "__main__":
    raise SystemExit(main())
