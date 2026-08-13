#!/usr/bin/env python3
"""Refit Tokyo's Celsius probability model for an exact serving point bundle.

The probability learner is trained only on honest pre-2026 point-stack rows. The
exact serving point bundle is then used to create a separate exploratory 2026
prediction ledger. Because the current serving point bundle was fitted through
July 2026, that ledger is explicitly not unseen promotion evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from src.calibration.celsius_market_probability import (
    build_celsius_probability_frame,
    celsius_calibration_table,
    celsius_probability_metrics,
    evaluate_celsius_probability_holdout,
    export_celsius_probability_bundle,
    fit_celsius_probability_system,
)
from src.calibration.v19_bucket import crossfit_ridge_predictions


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOKYO_ROOT = PROJECT_ROOT / "data/calibration/station_training_baseline/Tokyo"
DEFAULT_POINT_ROOT = Path(
    r"D:/dev/polymarket-weather-prediction/data/ml-candidates/"
    r"tokyo-station-training-live-v2/2a30f116c188e419"
)
STATION_ID = "RJTT"
FEATURE_PROFILE = "asia_no_peak"
PROVIDERS = ("gfs", "gefs", "jma_msm")
BASE_METHODS = ("xgboost", "lightgbm", "catboost")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--point-bundle",
        type=Path,
        default=DEFAULT_POINT_ROOT
        / "RJTT_station_high_regressor_live_tokyo_no_peak_stack_2026.joblib",
    )
    parser.add_argument(
        "--point-manifest",
        type=Path,
        default=DEFAULT_POINT_ROOT
        / "RJTT_station_high_regressor_live_tokyo_no_peak_stack_2026.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TOKYO_ROOT / "celsius_market_probability_live_refit",
    )
    parser.add_argument(
        "--model-version",
        default="station_bucket_live_tokyo_1c_market_ordinal_2026_r1",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_source_identity() -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip()
    )
    if dirty:
        raise ValueError("probability refit requires a clean source checkout")
    return {
        "git_commit": commit,
        "git_dirty": False,
        "pipeline": "scripts/refit-celsius-probability-release.py",
    }


def validate_point_artifact(
    bundle: Mapping[str, Any], manifest: Mapping[str, Any], bundle_hash: str
) -> None:
    if bundle.get("station_id") != STATION_ID:
        raise ValueError("point bundle station mismatch")
    if bundle.get("model_version") != manifest.get("model_version"):
        raise ValueError("point model version mismatch")
    if (manifest.get("artifact_integrity") or {}).get("bundle_sha256") != bundle_hash:
        raise ValueError("point bundle hash mismatch")
    if manifest.get("training", {}).get("last_contract_date", "") < "2026-07-25":
        raise ValueError("expected the August 12 production refit population")


def serving_point_predictions(
    features: pd.DataFrame, point_bundle: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = features.copy()
    frame["contract_date"] = pd.to_datetime(frame["contract_date"], errors="raise")
    frame = frame.loc[frame["contract_date"].dt.year.eq(2026)].copy()
    feature_names = list(point_bundle["feature_names"])
    observed = pd.to_numeric(
        frame["observed_high_temp_through_as_of_f"], errors="coerce"
    )
    frame = frame.loc[observed.notna()].copy()
    observed = observed.loc[frame.index]
    if frame.empty:
        raise ValueError("serving replay has no rows with an observed high")
    model_frame = frame.reindex(columns=feature_names)
    base_columns: dict[str, np.ndarray] = {}
    long_parts: list[pd.DataFrame] = []
    for method in BASE_METHODS:
        remaining = np.asarray(
            point_bundle["base_models"][method].predict(model_frame), dtype=float
        )
        predicted = np.maximum(observed.to_numpy(dtype=float), observed + remaining)
        name = f"{method}_predicted_high_f"
        base_columns[name] = predicted
        long_parts.append(
            pd.DataFrame(
                {
                    "contract_date": frame["contract_date"].to_numpy(),
                    "method": method,
                    "predicted_high_f": predicted,
                }
            )
        )
    stack_features = list(point_bundle["stack_features"])
    stack_frame = pd.DataFrame(base_columns).reindex(columns=stack_features)
    predicted_high = np.asarray(
        point_bundle["stack_model"].predict(stack_frame), dtype=float
    )
    if not np.isfinite(predicted_high).all():
        raise ValueError("serving point replay produced a non-finite prediction")
    point = pd.DataFrame(
        {
            "contract_date": frame["contract_date"].to_numpy(),
            "actual_high_f": pd.to_numeric(frame["actual_high_f"], errors="raise"),
            "predicted_high_f": predicted_high,
        }
    )
    return point, pd.concat(long_parts, ignore_index=True)


def write_prediction_csv(frame: pd.DataFrame, path: Path) -> None:
    output = frame.copy()
    for column in ("celsius_offset_probabilities", "market_bucket_probabilities_c"):
        output[column] = output[column].map(
            lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"))
        )
    output.to_csv(path, index=False)


def main() -> int:
    args = parse_args()
    source_identity = clean_source_identity()
    point_bundle_path = args.point_bundle.resolve()
    point_manifest_path = args.point_manifest.resolve()
    point_hash = sha256_file(point_bundle_path)
    point_bundle = joblib.load(point_bundle_path)
    point_manifest = json.loads(point_manifest_path.read_text(encoding="utf-8"))
    validate_point_artifact(point_bundle, point_manifest, point_hash)

    features = pd.read_csv(TOKYO_ROOT / "RJTT_features.csv", low_memory=False)
    base_oof = pd.read_csv(TOKYO_ROOT / "RJTT_year_split_validation_predictions.csv")
    honest_point = crossfit_ridge_predictions(base_oof, providers=PROVIDERS)
    training = build_celsius_probability_frame(
        features,
        honest_point,
        base_oof,
        include_peak_features=False,
        feature_profile=FEATURE_PROFILE,
    )
    probability_bundle, forward, tuning = fit_celsius_probability_system(
        training,
        station_id=STATION_ID,
        point_model_version=str(point_bundle["model_version"]),
        point_bundle_sha256=point_hash,
        feature_profile=FEATURE_PROFILE,
        model_version=args.model_version,
        development_years=(2024, 2025),
        forward_validation_years=(2025,),
    )

    live_point, live_base = serving_point_predictions(features, point_bundle)
    holdout, holdout_metrics, holdout_calibration = evaluate_celsius_probability_holdout(
        features, live_point, live_base, probability_bundle, holdout_year=2026
    )
    probability_bundle["holdout_metrics"] = holdout_metrics.iloc[0].to_dict()
    probability_bundle["holdout_status"] = (
        "exploratory_serving_point_in_sample_not_promotion_evidence"
    )
    probability_bundle["serving_refit"] = {
        "fitting_performed": True,
        "threshold_selection_performed": True,
        "point_model_version": point_bundle["model_version"],
        "point_bundle_sha256": point_hash,
        "probability_training_population": "honest_2024_2025_point_stack_rows",
        "serving_replay_warning": (
            "the point bundle was fitted through 2026-07-25; its 2026 replay is exploratory"
        ),
    }
    probability_bundle["historical_acceptance"] = {
        "passed": False,
        "reason": "economic_backtest_pending_and_2026_serving_point_replay_is_not_unseen",
    }

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    forward_path = output / "RJTT_forward_validation_predictions.csv"
    holdout_path = output / "RJTT_2026_serving_point_predictions.csv"
    forward_metrics_path = output / "RJTT_forward_validation_metrics.csv"
    holdout_metrics_path = output / "RJTT_2026_serving_point_metrics.csv"
    calibration_path = output / "RJTT_2026_serving_point_calibration.csv"
    tuning_path = output / "RJTT_pre_2026_tuning.csv"
    write_prediction_csv(forward, forward_path)
    write_prediction_csv(holdout, holdout_path)
    celsius_probability_metrics(forward).to_csv(forward_metrics_path, index=False)
    holdout_metrics.to_csv(holdout_metrics_path, index=False)
    holdout_calibration.to_csv(calibration_path, index=False)
    tuning.to_csv(tuning_path, index=False)
    artifact_paths = [
        forward_path,
        holdout_path,
        forward_metrics_path,
        holdout_metrics_path,
        calibration_path,
        tuning_path,
        point_manifest_path,
    ]
    bundle_path, manifest_path = export_celsius_probability_bundle(
        probability_bundle,
        output / "model_weights",
        source_identity=source_identity,
        artifact_paths=artifact_paths,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["serving_refit"] = probability_bundle["serving_refit"]
    manifest["historical_acceptance"] = probability_bundle["historical_acceptance"]
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "point_model_version": point_bundle["model_version"],
        "point_bundle_sha256": point_hash,
        "probability_model_version": args.model_version,
        "probability_bundle_sha256": sha256_file(bundle_path),
        "probability_manifest_sha256": sha256_file(manifest_path),
        "forward_metrics": celsius_probability_metrics(forward).iloc[0].to_dict(),
        "exploratory_2026_serving_point_metrics": holdout_metrics.iloc[0].to_dict(),
        "exploratory_2026_serving_point_rows": int(len(holdout)),
        "exploratory_2026_excluded_missing_observed_high": int(
            pd.to_datetime(features["contract_date"]).dt.year.eq(2026).sum() - len(live_point)
        ),
        "promotion_eligible": False,
        "promotion_blocker": (
            "fresh outcomes after the 2026-07-25 point-model training cutoff are required"
        ),
    }
    (output / "refit_summary.json").write_text(
        json.dumps(summary, indent=2, default=str, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, default=str, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
