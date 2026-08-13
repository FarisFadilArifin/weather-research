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

from src.calibration.bucket_probability import (
    OFFSET_LABELS,
    FEATURE_PROFILE_COMMON_NO_PEAK,
    FEATURE_PROFILE_KDAL_1PM,
    FEATURE_PROFILE_PEAK_AUGMENTED,
    build_probability_frame,
    default_candidate_specs,
    export_probability_bundle,
    predict_probability_bundle,
    probability_metrics,
    score_probabilities,
    sha256_file,
)
from src.calibration.v19_bucket import crossfit_ridge_predictions


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a station rounded-degree bucket-probability model")
    parser.add_argument("--station", required=True, choices=("KATL", "KDAL"))
    parser.add_argument("--pipeline-dir", required=True, type=Path)
    parser.add_argument("--point-bundle", required=True, type=Path)
    parser.add_argument("--point-model-version", required=True)
    parser.add_argument("--model-version", default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--force-family",
        choices=("empirical", "ordinal_logistic", "lightgbm_multiclass"),
        default=None,
        help="Freeze family selection for an isolated challenger.",
    )
    parser.add_argument(
        "--blend-weights",
        default=None,
        help="Comma-separated learned-model weights; use 1.0 for a pure learned distribution.",
    )
    parser.add_argument(
        "--holdout-status",
        choices=("acceptance", "exploratory"),
        default="acceptance",
        help="Whether the inspected 2026 period may participate in historical acceptance.",
    )
    parser.add_argument("--include-peak-features", action="store_true")
    parser.add_argument(
        "--feature-profile",
        choices=(FEATURE_PROFILE_COMMON_NO_PEAK, FEATURE_PROFILE_PEAK_AUGMENTED, FEATURE_PROFILE_KDAL_1PM),
        default=None,
    )
    args = parser.parse_args()

    station = args.station.upper()
    if station == "KDAL" and args.include_peak_features:
        raise SystemExit("KDAL probability training must not include peak features")
    if args.feature_profile == FEATURE_PROFILE_KDAL_1PM and station != "KDAL":
        raise SystemExit("kdal_1pm probability profile is limited to KDAL")
    if args.feature_profile == FEATURE_PROFILE_KDAL_1PM and args.include_peak_features:
        raise SystemExit("kdal_1pm probability profile cannot include peak features")
    if not args.point_bundle.is_file():
        raise SystemExit(f"point bundle not found: {args.point_bundle}")
    blend_weights = _parse_blend_weights(args.blend_weights)
    candidate_specs = default_candidate_specs()
    if args.force_family is not None:
        candidate_specs = [
            spec
            for spec in candidate_specs
            if spec.family in {"empirical", args.force_family}
        ]

    features = pd.read_csv(args.pipeline_dir / f"{station}_features.csv", low_memory=False)
    validation = pd.read_csv(args.pipeline_dir / f"{station}_year_split_validation_predictions.csv")
    residual_point = crossfit_ridge_predictions(validation)
    from src.calibration.bucket_probability import fit_probability_system

    profile_results = []
    profiles = (
        [(False, FEATURE_PROFILE_COMMON_NO_PEAK), (True, FEATURE_PROFILE_PEAK_AUGMENTED)]
        if args.include_peak_features
        else [(False, args.feature_profile or FEATURE_PROFILE_COMMON_NO_PEAK)]
    )
    for include_peak, feature_profile in profiles:
        frame = build_probability_frame(
            features,
            residual_point,
            validation,
            include_peak_features=include_peak,
            feature_profile=feature_profile,
        )
        bundle, predictions, tuning = fit_probability_system(
            frame,
            station_id=station,
            point_model_version=args.point_model_version,
            point_bundle_sha256=sha256_file(args.point_bundle),
            include_peak_features=include_peak,
            feature_profile=feature_profile,
            model_version=args.model_version,
            candidate_specs=candidate_specs,
            forced_family=args.force_family,
            blend_weights=blend_weights,
        )
        profile_results.append((bundle, predictions, tuning, include_peak, feature_profile))
    comparison = pd.DataFrame(
        [
            {
                "feature_profile": bundle["feature_profile"],
                "log_loss": bundle["forward_metrics"]["log_loss"],
                "brier": bundle["forward_metrics"]["brier"],
                "offset_accuracy": bundle["forward_metrics"]["offset_accuracy"],
                "top_two_accuracy": bundle["forward_metrics"]["top_two_accuracy"],
                "selected_family": bundle["selected_family"],
                "profile_rank": 1 if include_peak else 0,
            }
            for bundle, _, _, include_peak, _ in profile_results
        ]
    )
    winner = comparison.sort_values(
        ["log_loss", "brier", "profile_rank"], ascending=[True, True, True]
    ).iloc[0]
    bundle, forward_predictions, tuning, selected_include_peak, selected_feature_profile = next(
        item for item in profile_results if item[0]["feature_profile"] == winner["feature_profile"]
    )
    bundle["profile_comparison"] = comparison.drop(columns=["profile_rank"]).to_dict(orient="records")

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    forward_path = output / f"{station}_forward_probability_predictions.csv"
    serializable_forward = forward_predictions.copy()
    serializable_forward["offset_probabilities"] = serializable_forward["offset_probabilities"].map(json.dumps)
    serializable_forward.to_csv(forward_path, index=False)
    tuning.to_csv(output / f"{station}_probability_tuning.csv", index=False)
    comparison.drop(columns=["profile_rank"]).to_csv(
        output / f"{station}_probability_feature_profile_comparison.csv", index=False
    )
    probability_metrics(forward_predictions).to_csv(output / f"{station}_forward_probability_metrics.csv", index=False)

    holdout_predictions_path = (
        output / f"{station}_2026_probability_holdout_predictions.csv"
    )
    holdout_metrics = _evaluate_2026_holdout(
        args.pipeline_dir,
        station,
        features,
        bundle,
        selected_include_peak,
        selected_feature_profile,
        predictions_output_path=holdout_predictions_path,
    )
    holdout_metrics.to_csv(output / f"{station}_2026_probability_holdout_metrics.csv", index=False)
    bundle["holdout_metrics"] = (
        holdout_metrics.iloc[0].to_dict() if not holdout_metrics.empty else {}
    )
    bundle["holdout_status"] = args.holdout_status
    bundle["historical_acceptance"] = (
        _historical_acceptance(bundle, holdout_metrics)
        if args.holdout_status == "acceptance"
        else {
            "passed": False,
            "reasons": ["fresh_shadow_data_required"],
            "holdout_status": "exploratory_previously_inspected",
        }
    )
    source_identity = _source_identity()
    bundle_path, manifest_path = export_probability_bundle(bundle, output / "model_weights", source_identity=source_identity)
    print(
        json.dumps(
            {
                "station": station,
                "bundle": str(bundle_path),
                "manifest": str(manifest_path),
                "forward_rows": len(forward_predictions),
                "selected_feature_profile": bundle["feature_profile"],
                "selected_family": bundle["selected_family"],
                "family_selection_mode": bundle["family_selection_mode"],
                "blend_weight": bundle["blend_weight"],
                "blend_weight_candidates": bundle["blend_weight_candidates"],
                "holdout": holdout_metrics.iloc[0].to_dict() if not holdout_metrics.empty else {},
                "historical_acceptance": bundle["historical_acceptance"],
            },
            indent=2,
        )
    )
    return 0


def _evaluate_2026_holdout(
    pipeline_dir: Path,
    station: str,
    features: pd.DataFrame,
    bundle: dict,
    include_peak_features: bool,
    feature_profile: str | None = None,
    predictions_output_path: Path | None = None,
) -> pd.DataFrame:
    test_path = pipeline_dir / f"{station}_year_split_test_predictions.csv"
    if not test_path.is_file():
        return pd.DataFrame()
    test = pd.read_csv(test_path)
    point = test.loc[test["method"].eq("ridge_stack"), ["contract_date", "actual_high_f", "predicted_high_f"]]
    holdout = build_probability_frame(
        features,
        point,
        test,
        include_peak_features=include_peak_features,
        feature_profile=feature_profile,
    )
    holdout = holdout.loc[holdout["year"].eq(2026)].copy()
    if holdout.empty:
        return pd.DataFrame()

    results = []
    for _, row in holdout.iterrows():
        values = row.to_dict()
        values["contract_date"] = pd.Timestamp(row["contract_date"]).date().isoformat()
        result = predict_probability_bundle(bundle, values)
        if result["status"] != "ok":
            continue
        actual_bucket = _bucket(int(row["actual_degree_f"]))
        point_bucket = _bucket(int(row["point_degree_f"]))
        actual_offset_label = _offset_label(
            int(row["actual_degree_f"]) - int(row["point_degree_f"])
        )
        offset_probabilities = result["offset_probabilities"]
        offset_probability_vector = np.asarray(
            [float(offset_probabilities[label]) for label in OFFSET_LABELS],
            dtype=float,
        )
        actual_offset_class = OFFSET_LABELS.index(actual_offset_label)
        ranked_offsets = sorted(
            offset_probabilities.items(), key=lambda item: (-item[1], item[0])
        )
        ranked_buckets = sorted(
            result["bucket_probabilities"].items(),
            key=lambda item: (-item[1], item[0]),
        )
        offset_brier = sum(
            (float(probability) - (1.0 if label == actual_offset_label else 0.0)) ** 2
            for label, probability in offset_probabilities.items()
        )
        results.append(
            {
                "contract_date": values["contract_date"],
                "actual_high_f": float(row["actual_high_f"]),
                "actual_degree_f": int(row["actual_degree_f"]),
                "actual_bucket": actual_bucket,
                "point_prediction_f": float(row["point_prediction_f"]),
                "point_degree_f": int(row["point_degree_f"]),
                "point_bucket": point_bucket,
                "observed_high_temp_through_as_of_f": float(
                    row["observed_high_temp_through_as_of_f"]
                ),
                "point_hit": point_bucket == actual_bucket,
                "recommended_bucket": result["recommended_bucket_label"],
                "recommended_hit": result["recommended_bucket_label"] == actual_bucket,
                "recommended_bucket_probability": float(
                    result["recommended_bucket_probability"]
                ),
                "second_bucket": (
                    ranked_buckets[1][0] if len(ranked_buckets) > 1 else None
                ),
                "second_bucket_probability": float(
                    result["second_bucket_probability"]
                ),
                "point_bucket_probability": float(
                    result["bucket_probabilities"].get(point_bucket, 0.0)
                ),
                "actionable": result["probability_decision"] != "no_trade",
                "probability_decision": result["probability_decision"],
                "probability_decision_reason": result[
                    "probability_decision_reason"
                ],
                "switch": result["overrides_point_bucket"],
                "probability_advantage_over_point_bucket": float(
                    result["probability_advantage_over_point_bucket"]
                ),
                "actual_probability": float(result["bucket_probabilities"].get(actual_bucket, 0.0)),
                "actual_offset_label": actual_offset_label,
                "actual_offset_probability": float(
                    offset_probabilities.get(actual_offset_label, 0.0)
                ),
                "offset_brier": offset_brier,
                "offset_hit": ranked_offsets[0][0] == actual_offset_label,
                "offset_top_two_hit": actual_offset_label
                in {label for label, _ in ranked_offsets[:2]},
                "offset_top_confidence": float(ranked_offsets[0][1]),
                "actual_offset_class": actual_offset_class,
                "offset_probability_vector": offset_probability_vector,
                "offset_probabilities": offset_probabilities,
                "degree_probabilities": result["degree_probabilities"],
                "bucket_probabilities": result["bucket_probabilities"],
            }
        )
    if not results:
        return pd.DataFrame()
    evaluated = pd.DataFrame(results)
    if predictions_output_path is not None:
        predictions_output_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = evaluated.drop(
            columns=["offset_probability_vector"], errors="ignore"
        ).copy()
        for column in (
            "offset_probabilities",
            "degree_probabilities",
            "bucket_probabilities",
        ):
            serializable[column] = serializable[column].map(
                lambda value: json.dumps(value, sort_keys=True)
            )
        serializable.to_csv(predictions_output_path, index=False)
    offset_scores = score_probabilities(
        evaluated["actual_offset_class"].to_numpy(dtype=int),
        np.vstack(evaluated["offset_probability_vector"].to_numpy()),
    )
    actionable = evaluated["actionable"]
    switches = actionable & evaluated["switch"]
    calibration_error = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        selected = evaluated["offset_top_confidence"].ge(lower) & evaluated[
            "offset_top_confidence"
        ].lt(lower + 0.1)
        if selected.any():
            calibration_error += float(selected.mean()) * abs(
                float(evaluated.loc[selected, "offset_hit"].mean())
                - float(evaluated.loc[selected, "offset_top_confidence"].mean())
            )
    return pd.DataFrame(
        [
            {
                "count": len(evaluated),
                "bucket_log_loss": float(-np.log(evaluated["actual_probability"].clip(lower=1e-12)).mean()),
                "offset_log_loss": float(
                    -np.log(evaluated["actual_offset_probability"].clip(lower=1e-12)).mean()
                ),
                "multiclass_brier": float(evaluated["offset_brier"].mean()),
                "ranked_probability_score": float(
                    offset_scores["ranked_probability_score"]
                ),
                "calibration_error": calibration_error,
                "offset_accuracy": float(evaluated["offset_hit"].mean()),
                "offset_top_two_accuracy": float(evaluated["offset_top_two_hit"].mean()),
                "point_bucket_accuracy": float(evaluated["point_hit"].mean()),
                "probability_bucket_accuracy": float(evaluated["recommended_hit"].mean()),
                "actionable_coverage": float(actionable.mean()),
                "actionable_accuracy": float(evaluated.loc[actionable, "recommended_hit"].mean()) if actionable.any() else np.nan,
                "point_accuracy_on_actionable": float(
                    evaluated.loc[actionable, "point_hit"].mean()
                )
                if actionable.any()
                else np.nan,
                "switch_count": int(switches.sum()),
                "switch_accuracy": float(evaluated.loc[switches, "recommended_hit"].mean()) if switches.any() else np.nan,
                "point_accuracy_on_switches": float(evaluated.loc[switches, "point_hit"].mean()) if switches.any() else np.nan,
            }
        ]
    )


def _bucket(degree: int) -> str:
    low = degree if degree % 2 == 0 else degree - 1
    return f"{low}-{low + 1}"


def _offset_label(offset: int) -> str:
    if offset <= -4:
        return "le_-4"
    if offset >= 4:
        return "ge_+4"
    return f"{offset:+d}" if offset > 0 else str(offset)


def _historical_acceptance(bundle: dict, holdout_metrics: pd.DataFrame) -> dict[str, object]:
    if holdout_metrics.empty:
        return {"passed": False, "reasons": ["missing_2026_holdout"]}
    row = holdout_metrics.iloc[0]
    gates = {
        "forwardLogLossBeatsEmpirical": float(bundle["forward_metrics"]["log_loss"])
        < float(bundle["empirical_forward_metrics"]["log_loss"]),
        "holdoutCoverageWithin55To65Pct": 0.55
        <= float(row["actionable_coverage"])
        <= 0.65,
        "holdoutActionableAccuracyNoWorseThanPoint": float(row["actionable_accuracy"])
        >= float(row["point_accuracy_on_actionable"]),
        "holdoutHasSwitches": int(row["switch_count"]) > 0,
        "holdoutSwitchesBeatPoint": int(row["switch_count"]) > 0
        and float(row["switch_accuracy"]) > float(row["point_accuracy_on_switches"]),
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "reasons": [name for name, passed in gates.items() if not passed],
    }


def _parse_blend_weights(value: str | None) -> tuple[float, ...] | None:
    if value is None:
        return None
    try:
        weights = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise SystemExit("--blend-weights must be a comma-separated list of numbers") from exc
    if not weights:
        raise SystemExit("--blend-weights must contain at least one value")
    if any(not np.isfinite(weight) or weight <= 0.0 or weight > 1.0 for weight in weights):
        raise SystemExit("--blend-weights values must be in (0, 1]")
    return weights


def _source_identity() -> dict[str, object]:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], text=True).strip()

    try:
        commit = git("rev-parse", "HEAD")
        dirty = bool(git("status", "--porcelain"))
    except Exception:
        return {}
    return {"git_commit": commit, "git_dirty": dirty}


if __name__ == "__main__":
    raise SystemExit(main())
