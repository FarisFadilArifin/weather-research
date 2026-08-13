from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.bucket_probability import (
    canonical_two_degree_bucket,
    degree_to_bucket_probabilities,
    expand_offset_probabilities,
    fit_tail_policy,
    round_half_up,
)
from src.calibration.continuous_residual_probability import (
    ARTIFACT_TYPE,
    LOG_EPSILON,
    MODEL_FEATURES,
    PEAK_FEATURES,
    SCHEMA_VERSION,
    assert_cutoffs,
    boundary_metric_rows,
    bucket_log_loss,
    calibrated_prediction,
    common_date_comparison,
    distribution_cdf,
    distribution_pdf,
    fit_distribution,
    integrate_settlement_degrees,
    interval_coverage_rows,
    multiclass_brier,
    pit_histogram_rows,
    predict_continuous_bundle,
    predict_distributions,
    prepare_probability_frame,
    quantile_crps,
    ranked_probability_score,
    reliability_rows,
    settlement_interval,
    sha256_file,
    summarize_metrics,
    strict_json_data,
    truncated_cdf,
    truncated_ppf,
    DistributionPrediction,
)
from src.calibration.v19_bucket import crossfit_ridge_predictions


FAMILIES = ("seasonal_empirical", "conditional_empirical", "gaussian", "student_t")
SCALE_GRID = (0.75, 0.9, 1.0, 1.1, 1.25, 1.5)


def main() -> int:
    parser = argparse.ArgumentParser(description="Research-only continuous residual probability challenger")
    parser.add_argument("--station", required=True, choices=("KATL", "KDAL"))
    parser.add_argument("--pipeline-dir", required=True, type=Path)
    parser.add_argument("--point-bundle", required=True, type=Path)
    parser.add_argument("--point-model-version", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--include-peak-features", action="store_true")
    args = parser.parse_args()
    station = args.station.upper()
    _validate_inputs(args, station)
    point_hash = sha256_file(args.point_bundle)
    _validate_existing_manifest(args.output_dir, point_hash)

    features = pd.read_csv(args.pipeline_dir / f"{station}_features.csv", low_memory=False)
    validation = pd.read_csv(args.pipeline_dir / f"{station}_year_split_validation_predictions.csv", low_memory=False)
    test = pd.read_csv(args.pipeline_dir / f"{station}_year_split_test_predictions.csv", low_memory=False)
    validation_point = crossfit_ridge_predictions(validation)
    test_point = test.loc[test["method"].eq("ridge_stack"), ["contract_date", "actual_high_f", "predicted_high_f"]].copy()

    profiles = [False, True] if args.include_peak_features else [False]
    profile_results = []
    for include_peak in profiles:
        forward_frame = prepare_probability_frame(
            features, validation_point, validation, station_id=station, include_peak_features=include_peak
        )
        exploratory_frame = prepare_probability_frame(
            features, test_point, test, station_id=station, include_peak_features=include_peak
        )
        forward, tuning, states = _forward_evaluate(forward_frame, station, include_peak)
        profile_results.append((include_peak, forward_frame, exploratory_frame, forward, tuning, states))

    profile_comparison = pd.concat(
        [summarize_metrics(item[3]).assign(feature_profile="peak_augmented" if item[0] else "common_no_peak") for item in profile_results],
        ignore_index=True,
    )
    candidates = profile_comparison.loc[
        profile_comparison["period"].eq("2024-2025")
        & profile_comparison["model_family"].isin(("conditional_empirical", "gaussian", "student_t"))
    ].copy()
    candidates["family_rank"] = candidates["model_family"].map({"conditional_empirical": 0, "gaussian": 1, "student_t": 2})
    winner = candidates.sort_values(
        ["bucket_log_loss", "continuous_crps", "bucket_brier", "calibration_error", "family_rank"],
        na_position="last",
    ).iloc[0]
    selected_profile = winner["feature_profile"]
    selected_family = winner["model_family"]
    selected = next(item for item in profile_results if ("peak_augmented" if item[0] else "common_no_peak") == selected_profile)
    include_peak, forward_frame, exploratory_frame, forward, tuning, states = selected

    discrete_forward = _discrete_baseline(forward_frame, station, args.pipeline_dir, exploratory=False)
    direct_forward = _direct_baseline(forward_frame)
    all_forward = pd.concat([forward, discrete_forward, direct_forward], ignore_index=True, sort=False)
    assert_cutoffs(all_forward)
    common = common_date_comparison(all_forward)

    exploratory, exploratory_tuning, final_states = _exploratory_evaluate(
        forward_frame, exploratory_frame, station, include_peak
    )
    discrete_exploratory = _discrete_baseline(exploratory_frame, station, args.pipeline_dir, exploratory=True)
    direct_exploratory = _direct_baseline(exploratory_frame)
    all_exploratory = pd.concat([exploratory, discrete_exploratory, direct_exploratory], ignore_index=True, sort=False)
    assert_cutoffs(all_exploratory)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_outputs(
        args.output_dir, station, all_forward, all_exploratory,
        pd.concat([tuning, exploratory_tuning], ignore_index=True), profile_comparison, common,
    )
    selected_export = final_states[selected_family]
    bundle, manifest = _export_winner(
        args, station, point_hash, selected_family, selected_profile,
        selected_export["model_state"], selected_export["calibration_scale_multiplier"],
        all_forward, all_exploratory, profile_comparison, forward_frame, exploratory_frame,
    )
    print(json.dumps({
        "station": station, "research_only": True, "selected_family": selected_family,
        "selected_feature_profile": selected_profile, "forward_rows": len(all_forward),
        "exploratory_2026_rows": len(all_exploratory), "bundle": str(bundle), "manifest": str(manifest),
    }, indent=2))
    return 0


def _validate_inputs(args: argparse.Namespace, station: str) -> None:
    expected_suffix = "station_stacking_v20_peak_timing" if station == "KATL" else "station_stacking_v20_kdal_no_peak"
    if args.pipeline_dir.name.lower() != expected_suffix:
        raise SystemExit(f"wrong station/pipeline pairing: {station} requires {expected_suffix}")
    if station == "KDAL" and args.include_peak_features:
        raise SystemExit("KDAL cannot enable peak features")
    if not args.point_bundle.is_file():
        raise SystemExit(f"missing point bundle: {args.point_bundle}")
    for suffix in ("features", "year_split_validation_predictions", "year_split_test_predictions"):
        path = args.pipeline_dir / f"{station}_{suffix}.csv"
        if not path.is_file():
            raise SystemExit(f"missing required predictions/input: {path}")


def _validate_existing_manifest(output: Path, point_hash: str) -> None:
    manifests = list((output / "model_weights").glob("*_continuous_residual_probability_*.json")) if output.exists() else []
    for path in manifests:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        old_hash = manifest.get("point_bundle_sha256")
        if old_hash and old_hash.lower() != point_hash.lower():
            raise SystemExit(f"point-bundle hash mismatch with existing artifact: {path}")


def _forward_evaluate(frame: pd.DataFrame, station: str, include_peak: bool):
    prediction_frames = []
    tuning_rows = []
    latest_states = {}
    feature_names = [*MODEL_FEATURES, *(PEAK_FEATURES if include_peak else ())]
    for year in (2024, 2025):
        outer_train = frame.loc[frame["contract_date"].dt.year.lt(year)].copy()
        outer_valid = frame.loc[frame["contract_date"].dt.year.eq(year)].copy()
        if len(outer_train) < 180 or outer_valid.empty:
            continue
        calibration_start = outer_train["contract_date"].max() - pd.Timedelta(days=89)
        model_train = outer_train.loc[outer_train["contract_date"].lt(calibration_start)].copy()
        calibration = outer_train.loc[outer_train["contract_date"].ge(calibration_start)].copy()
        if len(model_train) < 90 or calibration.empty:
            raise ValueError(f"insufficient strictly prior calibration history for {year}")
        for family in FAMILIES:
            state = fit_distribution(family, model_train, feature_names=feature_names)
            cal_predictions = predict_distributions(state, calibration)
            multiplier, objective = _select_scale_multiplier(calibration, cal_predictions)
            final_state = fit_distribution(family, outer_train, feature_names=feature_names)
            latest_states[family] = final_state
            predictions = [calibrated_prediction(value, multiplier) for value in predict_distributions(final_state, outer_valid)]
            prediction_frames.append(_prediction_rows(
                outer_valid, predictions, station, family,
                model_cutoff=outer_train["contract_date"].max(), calibration_start=calibration["contract_date"].min(),
                calibration_cutoff=calibration["contract_date"].max(), scale_multiplier=multiplier,
            ))
            tuning_rows.append({
                "station_id": station, "validation_year": year, "model_family": family,
                "feature_profile": "peak_augmented" if include_peak else "common_no_peak",
                "model_training_start": model_train["contract_date"].min(), "model_training_cutoff": outer_train["contract_date"].max(),
                "calibration_training_start": calibration["contract_date"].min(), "calibration_training_cutoff": calibration["contract_date"].max(),
                "model_training_count": len(outer_train), "calibration_sample_count": len(calibration),
                "chosen_scale_multiplier": multiplier, "objective": "continuous_crps", "objective_value": objective,
            })
    if not prediction_frames:
        raise ValueError("no 2024-2025 forward folds produced")
    predictions = pd.concat(prediction_frames, ignore_index=True)
    assert_cutoffs(predictions)
    return predictions, pd.DataFrame(tuning_rows), latest_states


def _exploratory_evaluate(forward: pd.DataFrame, exploratory: pd.DataFrame, station: str, include_peak: bool):
    feature_names = [*MODEL_FEATURES, *(PEAK_FEATURES if include_peak else ())]
    predictions = []
    tuning = []
    states = {}
    train = forward.loc[forward["contract_date"].dt.year.le(2025)].copy()
    calibration_start = train["contract_date"].max() - pd.Timedelta(days=89)
    model_train = train.loc[train["contract_date"].lt(calibration_start)]
    calibration = train.loc[train["contract_date"].ge(calibration_start)]
    evaluation = exploratory.loc[exploratory["contract_date"].dt.year.eq(2026)].copy()
    for family in FAMILIES:
        calibration_state = fit_distribution(family, model_train, feature_names=feature_names)
        multiplier, objective = _select_scale_multiplier(calibration, predict_distributions(calibration_state, calibration))
        state = fit_distribution(family, train, feature_names=feature_names)
        states[family] = {
            "model_state": state,
            "calibration_scale_multiplier": multiplier,
        }
        family_predictions = [calibrated_prediction(value, multiplier) for value in predict_distributions(state, evaluation)]
        predictions.append(_prediction_rows(
            evaluation, family_predictions, station, family, model_cutoff=train["contract_date"].max(),
            calibration_start=calibration["contract_date"].min(), calibration_cutoff=calibration["contract_date"].max(),
            scale_multiplier=multiplier,
        ))
        tuning.append({"station_id": station, "validation_year": 2026, "model_family": family,
                       "feature_profile": "peak_augmented" if include_peak else "common_no_peak",
                       "model_training_start": train["contract_date"].min(), "model_training_cutoff": train["contract_date"].max(),
                       "calibration_training_start": calibration["contract_date"].min(), "calibration_training_cutoff": calibration["contract_date"].max(),
                       "model_training_count": len(train), "calibration_sample_count": len(calibration),
                       "chosen_scale_multiplier": multiplier, "objective": "continuous_crps", "objective_value": objective,
                       "status": "exploratory_only"})
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(tuning), states


def _select_scale_multiplier(frame: pd.DataFrame, predictions: list[DistributionPrediction]) -> tuple[float, float]:
    scores = []
    for multiplier in SCALE_GRID:
        values = []
        for (_, row), prediction in zip(frame.iterrows(), predictions, strict=True):
            adjusted = calibrated_prediction(prediction, multiplier)
            final = DistributionPrediction(adjusted.family, float(row["point_prediction_f"]) + adjusted.location, adjusted.scale, adjusted.state)
            values.append(quantile_crps(final, float(row["actual_high_f"]), floor=float(row["observed_high_temp_through_as_of_f"]), points=79))
        scores.append((float(np.mean(values)), multiplier))
    objective, multiplier = min(scores, key=lambda item: (item[0], abs(item[1] - 1.0)))
    return float(multiplier), float(objective)


def _prediction_rows(frame: pd.DataFrame, predictions: list[DistributionPrediction], station: str, family: str, *, model_cutoff, calibration_start, calibration_cutoff, scale_multiplier: float) -> pd.DataFrame:
    rows = []
    all_bucket_labels: set[str] = set()
    intermediate = []
    for (_, source), residual_prediction in zip(frame.iterrows(), predictions, strict=True):
        point = float(source["point_prediction_f"])
        actual = float(source["actual_high_f"])
        floor = float(source["observed_high_temp_through_as_of_f"])
        final = DistributionPrediction(residual_prediction.family, point + residual_prediction.location, residual_prediction.scale, residual_prediction.state)
        degree_probs = integrate_settlement_degrees(point, residual_prediction, observed_high_f=floor)
        bucket_probs = degree_to_bucket_probabilities(degree_probs)
        all_bucket_labels.update(bucket_probs)
        intermediate.append((source, residual_prediction, final, degree_probs, bucket_probs))
    labels = sorted(all_bucket_labels, key=lambda label: int(label.split("-", 1)[0]))
    for source, residual_prediction, final, degree_probs, bucket_probs in intermediate:
        normalized_buckets = {label: float(bucket_probs.get(label, 0.0)) for label in labels}
        ranked = sorted(normalized_buckets.items(), key=lambda item: (-item[1], int(item[0].split("-", 1)[0])))
        actual = float(source["actual_high_f"]); floor = float(source["observed_high_temp_through_as_of_f"])
        actual_bucket = canonical_two_degree_bucket(int(source["actual_degree_f"])); point_bucket = canonical_two_degree_bucket(int(source["point_degree_f"]))
        pit = truncated_cdf(final, actual, floor)
        density = distribution_pdf(final, actual)
        if density is not None:
            normalizer = max(1.0 - distribution_cdf(final, round_half_up(float(source["observed_high_temp_through_as_of_f"])) - 0.5), LOG_EPSILON)
            nll = -np.log(max(density / normalizer, LOG_EPSILON))
        else:
            nll = np.nan
        row = {
            "station_id": station, "contract_date": source["contract_date"], "validation_year": int(source["contract_date"].year),
            "model_family": family, "feature_profile": source["feature_profile"], "point_prediction_f": float(source["point_prediction_f"]),
            "actual_high_f": actual, "continuous_residual_f": float(source["continuous_residual_f"]),
            "predicted_residual_mean_f": residual_prediction.location, "predicted_residual_scale_f": residual_prediction.scale,
            "point_degree_f": int(source["point_degree_f"]), "actual_degree_f": int(source["actual_degree_f"]),
            "point_rounding_remainder_f": float(source["point_rounding_remainder_f"]),
            "point_distance_to_round_boundary_f": float(source["point_distance_to_round_boundary_f"]),
            "point_signed_distance_to_round_boundary_f": float(source["point_signed_distance_to_round_boundary_f"]),
            "actual_bucket": actual_bucket, "point_bucket": point_bucket, "recommended_bucket": ranked[0][0],
            "actual_bucket_probability": normalized_buckets.get(actual_bucket, 0.0), "point_bucket_probability": normalized_buckets.get(point_bucket, 0.0),
            "degree_probabilities_json": json.dumps({str(key): value for key, value in degree_probs.items()}, sort_keys=True),
            "bucket_probabilities_json": json.dumps(normalized_buckets, sort_keys=True),
            "model_training_cutoff": model_cutoff, "calibration_training_start": calibration_start,
            "calibration_training_cutoff": calibration_cutoff, "availability_status": "available", "unavailable_reason": "",
            "calibration_scale_multiplier": scale_multiplier, "continuous_crps": quantile_crps(final, actual, floor=floor, points=199),
            "continuous_nll": nll, "pit": pit, "predictive_mean_absolute_error_f": abs(final.location - actual),
            "bucket_log_loss": bucket_log_loss(actual_bucket, normalized_buckets),
            "bucket_brier": multiclass_brier(actual_bucket, normalized_buckets, labels),
            "ranked_probability_score": ranked_probability_score(actual_bucket, normalized_buckets, labels),
            "top_bucket_probability": ranked[0][1], "top_bucket_hit": ranked[0][0] == actual_bucket,
            "top_two_bucket_hit": actual_bucket in {item[0] for item in ranked[:2]}, "point_bucket_hit": point_bucket == actual_bucket,
        }
        for level in (50, 80, 90, 95):
            alpha = (1 - level / 100) / 2
            low = truncated_ppf(final, alpha, floor); high = truncated_ppf(final, 1 - alpha, floor)
            row[f"interval_{level}_lower_f"] = low; row[f"interval_{level}_upper_f"] = high
            row[f"interval_{level}_covered"] = low <= actual <= high; row[f"interval_{level}_width_f"] = high - low
        rows.append(row)
    return pd.DataFrame(rows)


def _direct_baseline(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, source in frame.loc[frame["year"].isin((2024, 2025, 2026))].iterrows():
        bucket = canonical_two_degree_bucket(int(source["point_degree_f"])); actual = canonical_two_degree_bucket(int(source["actual_degree_f"]))
        probs = {bucket: 1.0}
        rows.append(_baseline_row(source, "direct_point_rounding", probs, model_cutoff=pd.Timestamp(f"{int(source['year'])-1}-12-31")))
    return pd.DataFrame(rows)


def _discrete_baseline(frame: pd.DataFrame, station: str, pipeline_dir: Path, *, exploratory: bool) -> pd.DataFrame:
    root = pipeline_dir.parent / "station_bucket_probability" / station
    path = root / (f"{station}_2026_probability_predictions.csv" if exploratory else f"{station}_forward_probability_predictions.csv")
    if not path.is_file() and exploratory:
        # Existing script did not persist holdout rows, so reproduce predictions from its exported bundle.
        bundle_paths = list((root / "model_weights").glob("*.joblib"))
        if not bundle_paths:
            return pd.DataFrame()
        from src.calibration.bucket_probability import predict_probability_bundle
        bundle = joblib.load(bundle_paths[0])
        rows = []
        evaluation = frame.loc[frame["year"].eq(2026)]
        for _, source in evaluation.iterrows():
            result = predict_probability_bundle(bundle, source.to_dict())
            if result["status"] == "ok":
                rows.append(_baseline_row(source, "nine_class_discrete", result["bucket_probabilities"], model_cutoff=pd.Timestamp(bundle["training_cutoff"])))
        return pd.DataFrame(rows)
    if not path.is_file():
        return pd.DataFrame()
    discrete = pd.read_csv(path)
    discrete["contract_date"] = pd.to_datetime(discrete["contract_date"], errors="coerce")
    merged = frame.merge(discrete[["contract_date", "offset_probabilities", "model_training_cutoff", "calibration_training_cutoff"]], on="contract_date", how="inner")
    output = []
    for year, group in merged.groupby(merged["contract_date"].dt.year):
        history = frame.loc[frame["contract_date"].dt.year.lt(year)]
        tail = fit_tail_policy((history["actual_degree_f"] - history["point_degree_f"]).astype(int))
        for _, source in group.iterrows():
            offsets = json.loads(source["offset_probabilities"])
            degree = expand_offset_probabilities(int(source["point_degree_f"]), offsets, tail, observed_high_f=float(source["observed_high_temp_through_as_of_f"]))
            row = _baseline_row(source, "nine_class_discrete", degree_to_bucket_probabilities(degree), model_cutoff=pd.Timestamp(source["model_training_cutoff"]))
            row["calibration_training_start"] = pd.Timestamp(source["calibration_training_cutoff"])
            row["calibration_training_cutoff"] = pd.Timestamp(source["calibration_training_cutoff"])
            output.append(row)
    return pd.DataFrame(output)


def _baseline_row(source: pd.Series, family: str, probs: dict[str, float], *, model_cutoff: pd.Timestamp) -> dict:
    actual = canonical_two_degree_bucket(int(source["actual_degree_f"])); point = canonical_two_degree_bucket(int(source["point_degree_f"]))
    labels = sorted(set(probs) | {actual, point}, key=lambda label: int(label.split("-", 1)[0])); normalized = {label: float(probs.get(label, 0.0)) for label in labels}
    ranked = sorted(normalized.items(), key=lambda item: (-item[1], int(item[0].split("-", 1)[0])))
    return {"station_id": source["station_id"], "contract_date": source["contract_date"], "validation_year": int(source["year"]),
            "model_family": family, "feature_profile": source["feature_profile"], "point_prediction_f": source["point_prediction_f"],
            "actual_high_f": source["actual_high_f"], "continuous_residual_f": source["continuous_residual_f"],
            "predicted_residual_mean_f": np.nan, "predicted_residual_scale_f": np.nan, "point_degree_f": source["point_degree_f"], "actual_degree_f": source["actual_degree_f"],
            "point_rounding_remainder_f": source["point_rounding_remainder_f"], "point_distance_to_round_boundary_f": source["point_distance_to_round_boundary_f"],
            "point_signed_distance_to_round_boundary_f": source["point_signed_distance_to_round_boundary_f"], "actual_bucket": actual, "point_bucket": point,
            "recommended_bucket": ranked[0][0], "actual_bucket_probability": normalized.get(actual, 0.0), "point_bucket_probability": normalized.get(point, 0.0),
            "degree_probabilities_json": "{}", "bucket_probabilities_json": json.dumps(normalized, sort_keys=True), "model_training_cutoff": model_cutoff,
            "calibration_training_start": model_cutoff, "calibration_training_cutoff": model_cutoff, "availability_status": "available", "unavailable_reason": "",
            "continuous_crps": np.nan, "continuous_nll": np.nan, "pit": np.nan, "predictive_mean_absolute_error_f": np.nan,
            "bucket_log_loss": bucket_log_loss(actual, normalized), "bucket_brier": multiclass_brier(actual, normalized, labels),
            "ranked_probability_score": ranked_probability_score(actual, normalized, labels), "top_bucket_probability": ranked[0][1],
            "top_bucket_hit": ranked[0][0] == actual, "top_two_bucket_hit": actual in {item[0] for item in ranked[:2]}, "point_bucket_hit": point == actual}


def _write_outputs(output: Path, station: str, forward: pd.DataFrame, exploratory: pd.DataFrame, tuning: pd.DataFrame, profiles: pd.DataFrame, common: pd.DataFrame) -> None:
    forward.to_csv(output / f"{station}_forward_continuous_predictions.csv", index=False)
    summarize_metrics(forward).to_csv(output / f"{station}_forward_continuous_metrics.csv", index=False)
    profiles.to_csv(output / f"{station}_candidate_comparison.csv", index=False)
    common.to_csv(output / f"{station}_common_date_comparison.csv", index=False)
    reliability_rows(forward).to_csv(output / f"{station}_reliability_by_bucket.csv", index=False)
    pit_histogram_rows(forward).to_csv(output / f"{station}_pit_histogram.csv", index=False)
    interval_coverage_rows(forward).to_csv(output / f"{station}_interval_coverage.csv", index=False)
    boundary_metric_rows(forward).to_csv(output / f"{station}_boundary_distance_metrics.csv", index=False)
    exploratory.to_csv(output / f"{station}_2026_exploratory_predictions.csv", index=False)
    summarize_metrics(exploratory).to_csv(output / f"{station}_2026_exploratory_metrics.csv", index=False)
    tuning.to_csv(output / f"{station}_continuous_probability_tuning.csv", index=False)


def _export_winner(
    args, station, point_hash, family, profile, state, scale_multiplier,
    forward, exploratory_predictions, comparison, source_frame, exploratory_frame,
):
    output = args.output_dir / "model_weights"; output.mkdir(parents=True, exist_ok=True)
    stem = f"{station}_continuous_residual_probability_v1"; bundle_path = output / f"{stem}.joblib"; manifest_path = output / f"{stem}.json"
    bundle = {"schema_version": SCHEMA_VERSION, "artifact_type": ARTIFACT_TYPE, "research_only": True, "not_production_approved": True,
              "station_id": station, "selected_family": family, "feature_profile": profile, "model_state": state,
              "calibration_scale_multiplier": float(scale_multiplier),
              "point_model_version": args.point_model_version, "point_bundle_sha256": point_hash,
              "feature_names": state.get("feature_names", list(MODEL_FEATURES)), "training_start": str(source_frame.contract_date.min().date()),
              "training_cutoff": str(source_frame.loc[source_frame.year.le(2025), "contract_date"].max().date()), "settlement_rounding": "ROUND_HALF_UP; degree d integrates [d-0.5,d+0.5)",
              "target": "continuous_residual_f=actual_high_f-point_prediction_f", "target_source": sorted(source_frame.get("target_source", pd.Series(["actual_high_f"])).dropna().astype(str).unique().tolist()),
              "exploratory_2026": True, "no_trading_policy_authorization": True}
    joblib.dump(bundle, bundle_path)
    _assert_export_reproduces_evaluation_row(
        bundle_path, family, exploratory_predictions, exploratory_frame
    )
    source_identity = _git_identity()
    metrics = summarize_metrics(forward).loc[lambda value: value.model_family.eq(family)].to_dict(orient="records")
    manifest = {key: value for key, value in bundle.items() if key != "model_state"}
    manifest.update({"calibration_date_range": [str(source_frame.loc[source_frame.year.le(2025), "contract_date"].max().date() - pd.Timedelta(days=89)), str(source_frame.loc[source_frame.year.le(2025), "contract_date"].max().date())],
                     "training_sample_count": int(source_frame.year.le(2025).sum()), "comparison_metrics": metrics,
                     "candidate_comparison": comparison.to_dict(orient="records"), "source_identity": source_identity,
                     "artifact_integrity": {"bundle_sha256": sha256_file(bundle_path)}})
    manifest_path.write_text(
        json.dumps(
            strict_json_data(manifest),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return bundle_path, manifest_path


def _assert_export_reproduces_evaluation_row(
    bundle_path: Path,
    family: str,
    exploratory_predictions: pd.DataFrame,
    exploratory_frame: pd.DataFrame,
) -> None:
    expected = exploratory_predictions.loc[
        exploratory_predictions["model_family"].eq(family)
    ].iloc[0]
    contract_date = pd.Timestamp(expected["contract_date"])
    features = exploratory_frame.loc[
        exploratory_frame["contract_date"].eq(contract_date)
    ].iloc[0]
    result = predict_continuous_bundle(joblib.load(bundle_path), features.to_dict())
    expected_degrees = {
        str(key): float(value)
        for key, value in json.loads(expected["degree_probabilities_json"]).items()
    }
    if result["degree_probabilities"].keys() != expected_degrees.keys() or not all(
        np.isclose(result["degree_probabilities"][key], value, rtol=0.0, atol=1e-12)
        for key, value in expected_degrees.items()
    ):
        raise AssertionError(
            "exported calibrated bundle does not reproduce its evaluation row"
        )
    if not np.isclose(
        result["predicted_residual_scale_f"],
        float(expected["predicted_residual_scale_f"]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise AssertionError(
            "exported calibrated bundle does not reproduce evaluated scale"
        )


def _git_identity():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True).strip())
        return {"git_commit": commit, "git_dirty": dirty}
    except Exception:
        return {}


if __name__ == "__main__":
    raise SystemExit(main())
