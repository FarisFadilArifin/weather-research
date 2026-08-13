from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.v19_bucket import crossfit_ridge_predictions
from src.calibration.win_classifier import (
    attach_continuous_baseline,
    binary_metrics,
    build_win_frame,
    export_win_bundle,
    fit_win_classifier_system,
    predict_win_bundle,
    sha256_file,
    threshold_metrics,
)
from src.calibration.bucket_probability import FEATURE_PROFILES


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a leakage-safe binary classifier for whether the regression-selected 2F bucket wins"
    )
    parser.add_argument("--station", required=True, choices=("KATL", "KDAL"))
    parser.add_argument("--pipeline-dir", required=True, type=Path)
    parser.add_argument("--point-bundle", required=True, type=Path)
    parser.add_argument("--point-model-version", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--include-peak-features", action="store_true")
    parser.add_argument("--feature-profile", choices=FEATURE_PROFILES)
    parser.add_argument("--model-version")
    parser.add_argument("--continuous-forward", type=Path)
    parser.add_argument("--continuous-holdout", type=Path)
    args = parser.parse_args()

    station = args.station.upper()
    if station == "KDAL" and args.include_peak_features:
        raise SystemExit("KDAL v20 no-peak training must not include peak features")
    if station == "KATL" and not args.include_peak_features:
        raise SystemExit("KATL v20 training requires --include-peak-features")
    for path in (args.pipeline_dir, args.point_bundle):
        if not path.exists():
            raise SystemExit(f"required input not found: {path}")

    features = pd.read_csv(args.pipeline_dir / f"{station}_features.csv", low_memory=False)
    validation = pd.read_csv(
        args.pipeline_dir / f"{station}_year_split_validation_predictions.csv"
    )
    point = crossfit_ridge_predictions(validation)
    frame = build_win_frame(
        features,
        point,
        validation,
        include_peak_features=args.include_peak_features,
        feature_profile=args.feature_profile,
    )
    continuous = attach_continuous_baseline(
        [path for path in (args.continuous_forward,) if path is not None]
    )
    bundle, forward, comparison, tuning = fit_win_classifier_system(
        frame,
        station_id=station,
        point_model_version=args.point_model_version,
        point_bundle_sha256=sha256_file(args.point_bundle),
        include_peak_features=args.include_peak_features,
        feature_profile=args.feature_profile,
        continuous_baseline=continuous,
    )
    if args.model_version:
        bundle["model_version"] = args.model_version

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    forward.to_csv(output / f"{station}_forward_win_predictions.csv", index=False)
    comparison.to_csv(output / f"{station}_win_candidate_comparison.csv", index=False)
    tuning.to_csv(output / f"{station}_win_fold_tuning.csv", index=False)
    threshold_metrics(forward).to_csv(
        output / f"{station}_win_threshold_metrics.csv", index=False
    )
    pd.DataFrame(bundle["baseline_metrics"]).to_csv(
        output / f"{station}_win_baseline_comparison.csv", index=False
    )

    holdout = _evaluate_holdout(args, station, features, bundle)
    if not holdout.empty:
        holdout.to_csv(output / f"{station}_2026_win_predictions.csv", index=False)
        holdout_metrics = binary_metrics(holdout)
        bundle["exploratory_2026_metrics"] = holdout_metrics
        pd.DataFrame([holdout_metrics]).to_csv(
            output / f"{station}_2026_win_metrics.csv", index=False
        )
        threshold_metrics(holdout).to_csv(
            output / f"{station}_2026_win_threshold_metrics.csv", index=False
        )
    else:
        bundle["exploratory_2026_metrics"] = {}

    bundle_path, manifest_path = export_win_bundle(
        bundle, output / "model_weights", source_identity=_source_identity()
    )
    print(
        json.dumps(
            {
                "station": station,
                "bundle": str(bundle_path),
                "manifest": str(manifest_path),
                "selected_candidate": bundle["selected_candidate"],
                "forward_metrics": bundle["forward_metrics"],
                "baseline_metrics": bundle["baseline_metrics"],
                "historical_acceptance": bundle["historical_acceptance"],
                "exploratory_2026_metrics": bundle["exploratory_2026_metrics"],
            },
            indent=2,
        )
    )
    return 0


def _evaluate_holdout(
    args: argparse.Namespace, station: str, features: pd.DataFrame, bundle: dict
) -> pd.DataFrame:
    test_path = args.pipeline_dir / f"{station}_year_split_test_predictions.csv"
    if not test_path.is_file():
        return pd.DataFrame()
    test = pd.read_csv(test_path)
    point = test.loc[
        test["method"].eq("ridge_stack"),
        ["contract_date", "actual_high_f", "predicted_high_f"],
    ].copy()
    if point.empty:
        return pd.DataFrame()
    frame = build_win_frame(
        features,
        point,
        test,
        include_peak_features=bool(bundle["include_peak_features"]),
        feature_profile=str(bundle["feature_profile"]),
    )
    frame = frame.loc[frame["year"].eq(2026)].copy()
    if frame.empty:
        return pd.DataFrame()
    continuous = attach_continuous_baseline(
        [path for path in (args.continuous_holdout,) if path is not None]
    )
    rows = []
    for _, row in frame.iterrows():
        values = row.to_dict()
        result = predict_win_bundle(bundle, values)
        if result["status"] != "ok":
            continue
        rows.append(
            {
                "contract_date": row["contract_date"],
                "actual_high_f": row["actual_high_f"],
                "point_prediction_f": row["point_prediction_f"],
                "point_bucket_label": row["point_bucket_label"],
                "actual_bucket_label": row["actual_bucket_label"],
                "bucket_win": row["bucket_win"],
                "win_probability": result["probability_selected_bucket_wins"],
                "confidence_decision": result["confidence_decision"],
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty and not continuous.empty:
        out["contract_date"] = pd.to_datetime(out["contract_date"])
        out = out.merge(continuous, on="contract_date", how="left", validate="one_to_one")
    return out


def _source_identity() -> dict[str, str | None]:
    def git(*arguments: str) -> str | None:
        completed = subprocess.run(
            ["git", *arguments], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
        )
        value = completed.stdout.strip()
        return value or None

    return {"git_commit": git("rev-parse", "HEAD"), "git_branch": git("branch", "--show-current")}


if __name__ == "__main__":
    raise SystemExit(main())
