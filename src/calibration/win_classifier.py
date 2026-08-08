from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .bucket_probability import (
    BASE_METHODS,
    FEATURE_PROFILE_COMMON_NO_PEAK,
    FEATURE_PROFILE_KDAL_1PM,
    FEATURE_PROFILE_PEAK_AUGMENTED,
    FEATURE_PROFILES,
    KDAL_1PM_SOURCE_FEATURES,
    MANDATORY_SOURCE_FEATURES,
    add_probability_features,
    build_probability_frame,
    canonical_two_degree_bucket,
    round_half_up,
)


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "station_regression_bucket_win_classifier_research"
MODEL_VERSION = "regression_bucket_win_classifier_v1"
HISTORY_FEATURES = (
    "prior_residual_bias_7d_f",
    "prior_residual_bias_30d_f",
    "prior_residual_mae_7d_f",
    "prior_residual_mae_30d_f",
    "prior_bucket_win_rate_30d",
    "prior_bucket_win_rate_90d",
    "prior_bucket_history_count_30d",
    "prior_bucket_history_count_90d",
)

COMMON_FEATURES = (
    "point_prediction_f",
    "rounded_point_degree_f",
    "point_rounding_remainder_f",
    "point_distance_to_round_boundary_f",
    "point_signed_distance_to_round_boundary_f",
    "point_bucket_lower_degree_f",
    "point_bucket_upper_degree_f",
    "point_distance_to_bucket_lower_edge_f",
    "point_distance_to_bucket_upper_edge_f",
    "point_position_within_bucket_f",
    "point_degree_is_bucket_upper",
    "xgboost_predicted_high_f",
    "lightgbm_predicted_high_f",
    "catboost_predicted_high_f",
    "base_prediction_mean_f",
    "base_prediction_spread_f",
    "base_prediction_std_f",
    "xgboost_minus_point_f",
    "lightgbm_minus_point_f",
    "catboost_minus_point_f",
    "gfs_high_f",
    "hrrr_high_f",
    "nbm_high_f",
    "provider_mean_high_f",
    "provider_median_high_f",
    "provider_spread_high_f",
    "provider_std_high_f",
    "gfs_minus_point_f",
    "hrrr_minus_point_f",
    "nbm_minus_point_f",
    "observed_temp_at_as_of_f",
    "observed_high_temp_through_as_of_f",
    "point_minus_observed_temp_f",
    "point_minus_observed_high_f",
    "observed_as_of_age_minutes",
    "observed_temp_change_last_1h_f",
    "observed_temp_change_last_3h_f",
    "observed_morning_warmup_rate_f_per_hour",
    "observed_high_so_far_change_since_9am_f",
    "observed_dewpoint_at_as_of_f",
    "observed_humidity_at_as_of",
    "observed_cloud_cover_at_as_of",
    "observed_precip_recent_at_as_of",
    "v11sf_forecast_temp_11am_minus_observed_f",
    "v11sf_forecast_temp_11am_spread_f",
    "v11sf_observation_adjusted_provider_high_f",
    "v11sf_forecast_warmup_after_11am_f",
    "day_of_year_sin",
    "day_of_year_cos",
    *HISTORY_FEATURES,
)

# KATL-only live-safe peak features. KDAL's no-peak profile never requests them.
KATL_PEAK_FEATURES = (
    "nbm_hour_of_max_local",
    "hrrr_hour_of_max_local",
    "hrrr_peak_at_window_end",
    "hrrr_slope_11_14_f",
    "hrrr_slope_14_to_peak_f",
    "hrrr_solar_energy_11_to_hrrr_peak_wh_m2",
    "hrrr_precip_total_11_to_hrrr_peak_mm",
    "hrrr_precip_wet_hours_11_to_hrrr_peak",
    "hrrr_tcc_11_to_hrrr_peak_mean_pct",
    "hrrr_tcc_11_to_hrrr_peak_max_pct",
    "v20_hrrr_t11_minus_observed_f",
    "v20_nbm_t11_minus_observed_f",
    "v20_hrrr_remaining_rise_f",
    "v20_nbm_remaining_rise_f",
    "v20_hrrr_observation_adjusted_high_f",
    "v20_nbm_observation_adjusted_high_f",
    "v20_adjusted_high_mean_f",
    "v20_adjusted_high_spread_f",
    "v20_model_high_difference_f",
    "v20_peak_hour_difference",
    "v20_solar_energy_11_14_wh_m2",
    "v20_solar_energy_15_18_wh_m2",
    "v20_tcc_change_11_to_hrrr_peak_pct",
    "v20_tcc_change_11_to_nbm_peak_pct",
    "v20_rain_before_hrrr_peak",
    "v20_rain_before_nbm_peak",
    "v20_rain_present_11_18",
    "v20_precip_onset_minus_hrrr_peak_hours_zero_filled",
    "v20_precip_onset_minus_nbm_peak_hours_zero_filled",
)


@dataclass(frozen=True)
class CandidateSpec:
    family: str
    params: Mapping[str, Any]
    calibration: str

    @property
    def key(self) -> str:
        params = json.dumps(dict(self.params), sort_keys=True, separators=(",", ":"))
        return f"{self.family}|{params}|{self.calibration}"


def default_candidate_specs() -> list[CandidateSpec]:
    specs: list[CandidateSpec] = []
    for c in (0.1, 1.0):
        for calibration in ("platt", "isotonic"):
            specs.append(CandidateSpec("logistic", {"C": c}, calibration))
    for depth, iterations, learning_rate, l2 in (
        (2, 150, 0.03, 10.0),
        (2, 300, 0.02, 20.0),
        (3, 200, 0.02, 20.0),
    ):
        for calibration in ("platt", "isotonic"):
            specs.append(
                CandidateSpec(
                    "catboost",
                    {
                        "max_depth": depth,
                        "max_iter": iterations,
                        "learning_rate": learning_rate,
                        "l2_regularization": l2,
                    },
                    calibration,
                )
            )
    return specs


def win_feature_names(*, include_peak_features: bool) -> list[str]:
    names = [*COMMON_FEATURES]
    if include_peak_features:
        names.extend(KATL_PEAK_FEATURES)
    return list(dict.fromkeys(names))


def win_feature_names_for_profile(
    *, include_peak_features: bool, feature_profile: str | None = None
) -> list[str]:
    profile = feature_profile or (
        FEATURE_PROFILE_PEAK_AUGMENTED
        if include_peak_features
        else FEATURE_PROFILE_COMMON_NO_PEAK
    )
    if profile not in FEATURE_PROFILES:
        raise ValueError(f"unknown win-classifier feature profile: {profile}")
    if include_peak_features != (profile == FEATURE_PROFILE_PEAK_AUGMENTED):
        raise ValueError("include_peak_features does not match win-classifier feature profile")
    names = [*win_feature_names(include_peak_features=include_peak_features)]
    if profile == FEATURE_PROFILE_KDAL_1PM:
        names = [
            name
            for name in names
            if not name.startswith("v11sf_")
        ]
        names.extend(KDAL_1PM_SOURCE_FEATURES)
    return list(dict.fromkeys(names))


def build_win_frame(
    feature_frame: pd.DataFrame,
    point_predictions: pd.DataFrame,
    base_validation_predictions: pd.DataFrame,
    *,
    include_peak_features: bool,
    feature_profile: str | None = None,
) -> pd.DataFrame:
    """Build a leakage-safe binary meta-label frame from honest point predictions."""
    frame = build_probability_frame(
        feature_frame,
        point_predictions,
        base_validation_predictions,
        include_peak_features=include_peak_features,
        feature_profile=feature_profile,
    )
    frame = add_win_geometry_features(frame)
    point_bucket = frame["point_degree_f"].astype(int).map(canonical_two_degree_bucket)
    actual_bucket = frame["actual_degree_f"].astype(int).map(canonical_two_degree_bucket)
    frame["point_bucket_label"] = point_bucket
    frame["actual_bucket_label"] = actual_bucket
    frame["bucket_win"] = point_bucket.eq(actual_bucket).astype(int)
    frame = add_strict_history_features(frame)
    for name in win_feature_names_for_profile(
        include_peak_features=include_peak_features, feature_profile=feature_profile
    ):
        if name not in frame:
            frame[name] = np.nan
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    return frame.sort_values("contract_date").reset_index(drop=True)


def add_win_geometry_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "point_prediction_f" not in out:
        raise ValueError("point_prediction_f is required")
    if "rounded_point_degree_f" not in out:
        out = add_probability_features(out)
    point = pd.to_numeric(out["point_prediction_f"], errors="coerce")
    degree = pd.to_numeric(out["rounded_point_degree_f"], errors="coerce")
    lower = degree.where(degree.mod(2).eq(0), degree - 1)
    upper = lower + 1
    lower_edge = lower - 0.5
    upper_edge = upper + 0.5
    out["point_bucket_lower_degree_f"] = lower
    out["point_bucket_upper_degree_f"] = upper
    out["point_distance_to_bucket_lower_edge_f"] = point - lower_edge
    out["point_distance_to_bucket_upper_edge_f"] = upper_edge - point
    out["point_position_within_bucket_f"] = (point - lower_edge) / 2.0
    out["point_degree_is_bucket_upper"] = degree.mod(2).eq(1).astype(float)
    return out


def add_strict_history_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add label-derived features using only rows strictly before each date."""
    out = frame.sort_values("contract_date").copy()
    residual = pd.to_numeric(out["actual_high_f"], errors="coerce") - pd.to_numeric(
        out["point_prediction_f"], errors="coerce"
    )
    win = pd.to_numeric(out["bucket_win"], errors="coerce")
    for window in (7, 30):
        prior_residual = residual.shift(1).rolling(window, min_periods=max(2, window // 5))
        out[f"prior_residual_bias_{window}d_f"] = prior_residual.mean()
        out[f"prior_residual_mae_{window}d_f"] = residual.abs().shift(1).rolling(
            window, min_periods=max(2, window // 5)
        ).mean()
    for window in (30, 90):
        prior_win = win.shift(1).rolling(window, min_periods=max(5, window // 5))
        out[f"prior_bucket_win_rate_{window}d"] = prior_win.mean()
        out[f"prior_bucket_history_count_{window}d"] = prior_win.count()
    return out.sort_index()


def fit_win_classifier_system(
    frame: pd.DataFrame,
    *,
    station_id: str,
    point_model_version: str,
    point_bundle_sha256: str,
    include_peak_features: bool,
    feature_profile: str | None = None,
    continuous_baseline: pd.DataFrame | None = None,
    candidate_specs: Sequence[CandidateSpec] | None = None,
    calibration_days: int = 90,
    min_fit_rows: int = 180,
    random_state: int = 42,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    station = station_id.strip().upper()
    if station == "KDAL" and include_peak_features:
        raise ValueError("KDAL v20 no-peak classifier cannot include peak features")
    if station not in {"KATL", "KDAL"}:
        raise ValueError("station_id must be KATL or KDAL")
    specs = list(candidate_specs or default_candidate_specs())
    if not specs:
        raise ValueError("at least one candidate is required")
    resolved_profile = feature_profile or (
        FEATURE_PROFILE_PEAK_AUGMENTED
        if include_peak_features
        else FEATURE_PROFILE_COMMON_NO_PEAK
    )
    feature_names = win_feature_names_for_profile(
        include_peak_features=include_peak_features, feature_profile=resolved_profile
    )
    development = frame.loc[frame["year"].between(2023, 2025)].copy()
    prediction_parts: dict[str, list[pd.DataFrame]] = {spec.key: [] for spec in specs}
    tuning_rows: list[dict[str, Any]] = []

    for validation_year in (2024, 2025):
        history = development.loc[development["year"].lt(validation_year)].copy()
        validation = development.loc[development["year"].eq(validation_year)].copy()
        fit, calibration = _calibration_split(
            history, calibration_days=calibration_days, min_fit_rows=min_fit_rows
        )
        if validation.empty:
            continue
        if calibration["contract_date"].max() >= validation["contract_date"].min():
            raise AssertionError("calibration data must be strictly before validation")
        for spec in specs:
            fitted = _fit_candidate(fit, feature_names, spec, random_state=random_state)
            calibrator = _fit_calibrator(
                spec.calibration,
                _predict_raw(fitted, calibration, feature_names),
                calibration["bucket_win"].to_numpy(dtype=int),
            )
            probability = _apply_calibrator(
                calibrator, _predict_raw(fitted, validation, feature_names)
            )
            predicted = _prediction_frame(
                validation,
                probability,
                validation_year=validation_year,
                spec=spec,
                model_training_cutoff=fit["contract_date"].max(),
                calibration_start=calibration["contract_date"].min(),
                calibration_cutoff=calibration["contract_date"].max(),
            )
            prediction_parts[spec.key].append(predicted)
            fold_score = binary_metrics(predicted)
            tuning_rows.append(
                {
                    "candidate_key": spec.key,
                    "family": spec.family,
                    "params_json": json.dumps(dict(spec.params), sort_keys=True),
                    "calibration": spec.calibration,
                    "validation_year": validation_year,
                    **fold_score,
                }
            )

    if not any(prediction_parts.values()):
        raise ValueError("no 2024/2025 chronological validation predictions were produced")
    comparison_rows = []
    for spec in specs:
        if not prediction_parts[spec.key]:
            continue
        predicted = pd.concat(prediction_parts[spec.key], ignore_index=True)
        comparison_rows.append(
            {
                "candidate_key": spec.key,
                "family": spec.family,
                "params_json": json.dumps(dict(spec.params), sort_keys=True),
                "calibration": spec.calibration,
                "complexity_rank": 0 if spec.family == "logistic" else 1,
                **binary_metrics(predicted),
            }
        )
    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["log_loss", "brier", "ece", "complexity_rank"], ignore_index=True
    )
    winner_key = str(comparison.iloc[0]["candidate_key"])
    winner = next(spec for spec in specs if spec.key == winner_key)
    forward = pd.concat(prediction_parts[winner_key], ignore_index=True).sort_values(
        "contract_date", ignore_index=True
    )

    baselines = _evaluate_baselines(development, forward, continuous_baseline)
    final_fit, final_calibration = _calibration_split(
        development, calibration_days=calibration_days, min_fit_rows=min_fit_rows
    )
    final_model = _fit_candidate(
        final_fit, feature_names, winner, random_state=random_state
    )
    final_calibrator = _fit_calibrator(
        winner.calibration,
        _predict_raw(final_model, final_calibration, feature_names),
        final_calibration["bucket_win"].to_numpy(dtype=int),
    )
    selected_metrics = binary_metrics(forward)
    acceptance = _acceptance(selected_metrics, baselines)
    threshold_table = threshold_metrics(forward)
    history_fallbacks = {
        name: _last_finite(development[name]) for name in HISTORY_FEATURES
    }
    bundle: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "model_version": MODEL_VERSION,
        "station_id": station,
        "point_model_version": point_model_version,
        "point_bundle_sha256": point_bundle_sha256,
        "feature_profile": resolved_profile,
        "include_peak_features": include_peak_features,
        "feature_names": feature_names,
        "mandatory_source_features": [
            *MANDATORY_SOURCE_FEATURES,
            *(f"{name}_predicted_high_f" for name in BASE_METHODS),
            "point_prediction_f",
        ],
        "selected_candidate": asdict(winner),
        "model_state": final_model,
        "calibrator_state": final_calibrator,
        "training_start": development["contract_date"].min().date().isoformat(),
        "model_training_cutoff": final_fit["contract_date"].max().date().isoformat(),
        "calibration_start": final_calibration["contract_date"].min().date().isoformat(),
        "training_cutoff": final_calibration["contract_date"].max().date().isoformat(),
        "training_rows": int(len(development)),
        "positive_rate": float(development["bucket_win"].mean()),
        "forward_metrics": selected_metrics,
        "baseline_metrics": baselines.to_dict(orient="records"),
        "candidate_comparison": comparison.drop(columns="complexity_rank").to_dict(orient="records"),
        "historical_acceptance": acceptance,
        "history_feature_fallbacks": history_fallbacks,
        "confidence_policy": select_confidence_threshold(threshold_table),
        "package_versions": package_versions(),
    }
    return bundle, forward, comparison.drop(columns="complexity_rank"), pd.DataFrame(tuning_rows)


def predict_win_bundle(
    bundle: Mapping[str, Any],
    feature_values: Mapping[str, Any],
    *,
    market_implied_probability: float | None = None,
    minimum_edge: float = 0.05,
) -> dict[str, Any]:
    missing = [
        name
        for name in bundle["mandatory_source_features"]
        if _finite_number(feature_values.get(name)) is None
    ]
    if missing:
        return {
            "status": "unavailable",
            "reason": "missing_required_features:" + ",".join(sorted(set(missing))),
        }
    frame = add_probability_features(pd.DataFrame([dict(feature_values)]))
    frame = add_win_geometry_features(frame)
    for name, fallback in bundle.get("history_feature_fallbacks", {}).items():
        if name not in frame or _finite_number(frame.iloc[0].get(name)) is None:
            frame[name] = fallback
    for name in bundle["feature_names"]:
        if name not in frame:
            frame[name] = np.nan
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    raw = _predict_raw(bundle["model_state"], frame, bundle["feature_names"])
    probability = float(_apply_calibrator(bundle["calibrator_state"], raw)[0])
    point_degree = round_half_up(float(feature_values["point_prediction_f"]))
    point_bucket = canonical_two_degree_bucket(point_degree)
    policy = bundle.get("confidence_policy", {})
    threshold = float(
        policy.get("threshold", bundle.get("default_confidence_threshold", 0.5))
    )
    result: dict[str, Any] = {
        "status": "ok",
        "model_version": bundle["model_version"],
        "station_id": bundle["station_id"],
        "point_prediction_f": float(feature_values["point_prediction_f"]),
        "rounded_point_high_f": point_degree,
        "selected_bucket": point_bucket,
        "probability_selected_bucket_wins": probability,
        "confidence_threshold": threshold,
        "confidence_decision": "eligible" if probability >= threshold else "skip",
    }
    if market_implied_probability is not None:
        market = float(market_implied_probability)
        if not 0.0 <= market <= 1.0:
            raise ValueError("market_implied_probability must be between 0 and 1")
        edge = probability - market
        result.update(
            {
                "market_implied_probability": market,
                "estimated_probability_edge": edge,
                "minimum_edge": float(minimum_edge),
                "trade_decision": "bet" if edge >= float(minimum_edge) else "skip",
            }
        )
    else:
        result["trade_decision"] = "market_price_required"
    return result


def binary_metrics(predictions: pd.DataFrame) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score

    y = pd.to_numeric(predictions["bucket_win"], errors="coerce").to_numpy(dtype=int)
    p = np.clip(
        pd.to_numeric(predictions["win_probability"], errors="coerce").to_numpy(dtype=float),
        1e-12,
        1.0 - 1e-12,
    )
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else math.nan
    return {
        "count": int(len(y)),
        "positive_rate": float(y.mean()),
        "mean_probability": float(p.mean()),
        "log_loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
        "brier": float(np.mean((p - y) ** 2)),
        "accuracy_at_0_5": float(np.mean((p >= 0.5) == y)),
        "roc_auc": auc,
        "ece": expected_calibration_error(y, p),
    }


def expected_calibration_error(
    actual: Sequence[int] | np.ndarray,
    probability: Sequence[float] | np.ndarray,
    *,
    bins: int = 10,
) -> float:
    y = np.asarray(actual, dtype=float)
    p = np.asarray(probability, dtype=float)
    total = len(y)
    if total == 0:
        return math.nan
    error = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        selected = (p >= lower) & ((p <= upper) if index == bins - 1 else (p < upper))
        if selected.any():
            error += float(selected.mean()) * abs(float(y[selected].mean()) - float(p[selected].mean()))
    return float(error)


def threshold_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold in np.arange(0.30, 0.701, 0.05):
        selected = predictions["win_probability"].ge(threshold)
        rows.append(
            {
                "threshold": float(round(threshold, 2)),
                "selected_count": int(selected.sum()),
                "coverage": float(selected.mean()),
                "mean_predicted_probability": float(predictions.loc[selected, "win_probability"].mean()) if selected.any() else math.nan,
                "realized_win_rate": float(predictions.loc[selected, "bucket_win"].mean()) if selected.any() else math.nan,
            }
        )
    return pd.DataFrame(rows)


def select_confidence_threshold(
    metrics: pd.DataFrame,
    *,
    target_win_rate: float = 0.70,
    minimum_coverage: float = 0.10,
    minimum_selected: int = 50,
) -> dict[str, Any]:
    """Freeze the highest-coverage forward-only rule that reaches the target.

    If no rule reaches the target, freeze the strongest eligible rule and mark the
    target unsupported. The 2026 holdout must never be passed to this function.
    """
    eligible = metrics.loc[
        metrics["selected_count"].ge(minimum_selected)
        & metrics["coverage"].ge(minimum_coverage)
        & metrics["realized_win_rate"].notna()
    ].copy()
    if eligible.empty:
        raise ValueError("no confidence threshold satisfies the minimum forward sample contract")
    passing = eligible.loc[eligible["realized_win_rate"].ge(target_win_rate)].copy()
    if not passing.empty:
        chosen = passing.sort_values(
            ["coverage", "realized_win_rate", "threshold"],
            ascending=[False, False, True],
        ).iloc[0]
        supported = True
    else:
        chosen = eligible.sort_values(
            ["realized_win_rate", "coverage", "threshold"],
            ascending=[False, False, False],
        ).iloc[0]
        supported = False
    return {
        "selection_data": "chronological_forward_validation_2024_2025_only",
        "threshold": float(chosen["threshold"]),
        "target_win_rate": float(target_win_rate),
        "target_supported_forward": supported,
        "forward_selected_count": int(chosen["selected_count"]),
        "forward_coverage": float(chosen["coverage"]),
        "forward_realized_win_rate": float(chosen["realized_win_rate"]),
        "minimum_coverage": float(minimum_coverage),
        "minimum_selected": int(minimum_selected),
        "holdout_rows_used_for_selection": 0,
    }


def attach_continuous_baseline(
    paths: Sequence[Path | str], *, selected_family: str = "student_t"
) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths if Path(path).is_file()]
    if not frames:
        return pd.DataFrame(columns=["contract_date", "continuous_point_bucket_probability"])
    frame = pd.concat(frames, ignore_index=True)
    if "model_family" in frame:
        frame = frame.loc[frame["model_family"].eq(selected_family)].copy()
    if "availability_status" in frame:
        frame = frame.loc[frame["availability_status"].eq("available")].copy()
    frame["contract_date"] = pd.to_datetime(frame["contract_date"], errors="coerce")
    frame["continuous_point_bucket_probability"] = pd.to_numeric(
        frame["point_bucket_probability"], errors="coerce"
    )
    columns = ["contract_date", "continuous_point_bucket_probability"]
    return frame.dropna(subset=columns).drop_duplicates("contract_date")[columns]


def export_win_bundle(
    bundle: Mapping[str, Any], output_dir: Path | str, *, source_identity: Mapping[str, Any] | None = None
) -> tuple[Path, Path]:
    import joblib

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = f"{bundle['station_id']}_{bundle['model_version']}"
    bundle_path = output / f"{stem}.joblib"
    manifest_path = output / f"{stem}.json"
    joblib.dump(dict(bundle), bundle_path)
    manifest = {
        key: value
        for key, value in bundle.items()
        if key not in {"model_state", "calibrator_state"}
    }
    manifest["source_identity"] = dict(source_identity or {})
    manifest["artifact_integrity"] = {"bundle_sha256": sha256_file(bundle_path)}
    manifest_path.write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return bundle_path, manifest_path


def _calibration_split(
    history: pd.DataFrame, *, calibration_days: int, min_fit_rows: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if history.empty:
        raise ValueError("training history is empty")
    cutoff = history["contract_date"].max() - pd.Timedelta(days=calibration_days - 1)
    fit = history.loc[history["contract_date"].lt(cutoff)].copy()
    calibration = history.loc[history["contract_date"].ge(cutoff)].copy()
    if len(fit) < min_fit_rows:
        raise ValueError("insufficient pre-calibration training rows")
    if calibration.empty or calibration["bucket_win"].nunique() < 2:
        raise ValueError("calibration window must contain both target classes")
    if fit["contract_date"].max() >= calibration["contract_date"].min():
        raise AssertionError("model fit data must be strictly before calibration")
    return fit, calibration


def _fit_candidate(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    spec: CandidateSpec,
    *,
    random_state: int,
) -> Any:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if frame["bucket_win"].nunique() < 2:
        raise ValueError("model fit data must contain both target classes")
    if spec.family == "logistic":
        estimator = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=float(spec.params["C"]),
                        max_iter=2000,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    elif spec.family == "catboost":
        from catboost import CatBoostClassifier

        estimator = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "classifier",
                    CatBoostClassifier(
                        depth=int(spec.params["max_depth"]),
                        iterations=int(spec.params["max_iter"]),
                        learning_rate=float(spec.params["learning_rate"]),
                        l2_leaf_reg=float(spec.params["l2_regularization"]),
                        loss_function="Logloss",
                        eval_metric="Logloss",
                        random_state=random_state,
                        verbose=False,
                        allow_writing_files=False,
                        thread_count=1,
                    ),
                ),
            ]
        )
    else:
        raise ValueError(f"unknown classifier family: {spec.family}")
    estimator.fit(frame[list(feature_names)], frame["bucket_win"].astype(int))
    return estimator


def _predict_raw(model: Any, frame: pd.DataFrame, feature_names: Sequence[str]) -> np.ndarray:
    return np.asarray(model.predict_proba(frame[list(feature_names)])[:, 1], dtype=float)


def _fit_calibrator(method: str, raw_probability: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    clipped = np.clip(np.asarray(raw_probability, dtype=float), 1e-6, 1 - 1e-6)
    if method == "platt":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(C=1.0, max_iter=1000)
        model.fit(_logit(clipped).reshape(-1, 1), actual)
        return {"method": method, "model": model}
    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        model = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
        model.fit(clipped, actual)
        return {"method": method, "model": model}
    raise ValueError(f"unknown calibration method: {method}")


def _apply_calibrator(state: Mapping[str, Any], raw_probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(raw_probability, dtype=float), 1e-6, 1 - 1e-6)
    if state["method"] == "platt":
        values = state["model"].predict_proba(_logit(clipped).reshape(-1, 1))[:, 1]
    elif state["method"] == "isotonic":
        values = state["model"].predict(clipped)
    else:
        raise ValueError(f"unknown calibration state: {state['method']}")
    return np.clip(np.asarray(values, dtype=float), 1e-6, 1 - 1e-6)


def _logit(probability: np.ndarray) -> np.ndarray:
    return np.log(probability / (1.0 - probability))


def _prediction_frame(
    frame: pd.DataFrame,
    probability: np.ndarray,
    *,
    validation_year: int,
    spec: CandidateSpec,
    model_training_cutoff: pd.Timestamp,
    calibration_start: pd.Timestamp,
    calibration_cutoff: pd.Timestamp,
) -> pd.DataFrame:
    out = frame[
        [
            "contract_date",
            "actual_high_f",
            "point_prediction_f",
            "point_degree_f",
            "actual_degree_f",
            "point_bucket_label",
            "actual_bucket_label",
            "bucket_win",
        ]
    ].copy()
    out["win_probability"] = probability
    out["validation_year"] = validation_year
    out["candidate_key"] = spec.key
    out["model_training_cutoff"] = model_training_cutoff
    out["calibration_start"] = calibration_start
    out["calibration_cutoff"] = calibration_cutoff
    return out


def _evaluate_baselines(
    development: pd.DataFrame,
    selected_predictions: pd.DataFrame,
    continuous_baseline: pd.DataFrame | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    climatology_parts = []
    for year in (2024, 2025):
        history = development.loc[development["year"].lt(year)]
        valid = selected_predictions.loc[selected_predictions["validation_year"].eq(year)].copy()
        if history.empty or valid.empty:
            continue
        global_rate = float(history["bucket_win"].mean())
        monthly = history.groupby("month")["bucket_win"].agg(["sum", "count"])
        months = valid["contract_date"].dt.month
        probability = []
        for month in months:
            wins = float(monthly.loc[month, "sum"]) if month in monthly.index else 0.0
            count = float(monthly.loc[month, "count"]) if month in monthly.index else 0.0
            probability.append((wins + 30.0 * global_rate) / (count + 30.0))
        valid["win_probability"] = probability
        climatology_parts.append(valid)
    if climatology_parts:
        rows.append({"baseline": "prior_month_shrunk_climatology", **binary_metrics(pd.concat(climatology_parts))})
    if continuous_baseline is not None and not continuous_baseline.empty:
        compared = selected_predictions.merge(
            continuous_baseline, on="contract_date", how="inner", validate="one_to_one"
        )
        if not compared.empty:
            compared["win_probability"] = compared["continuous_point_bucket_probability"]
            rows.append({"baseline": "continuous_residual_point_bucket_probability", **binary_metrics(compared)})
    return pd.DataFrame(rows)


def _acceptance(
    selected: Mapping[str, Any],
    baselines: pd.DataFrame,
    *,
    minimum_log_loss_improvement: float = 0.002,
) -> dict[str, Any]:
    gates: dict[str, bool] = {}
    for _, row in baselines.iterrows():
        label = str(row["baseline"])
        gates[f"log_loss_materially_better_than_{label}"] = (
            float(row["log_loss"]) - float(selected["log_loss"])
            >= minimum_log_loss_improvement
        )
        gates[f"ece_no_worse_than_{label}"] = float(selected["ece"]) <= float(row["ece"])
    return {
        "passed": bool(gates) and all(gates.values()),
        "minimum_log_loss_improvement": minimum_log_loss_improvement,
        "gates": gates,
        "reasons": [name for name, passed in gates.items() if not passed] or ([] if gates else ["no_baselines"]),
    }


def _last_finite(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(numeric.iloc[-1]) if not numeric.empty else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> dict[str, str]:
    return {
        package: importlib.metadata.version(distribution)
        for package, distribution in {
            "pandas": "pandas",
            "numpy": "numpy",
            "scikit-learn": "scikit-learn",
            "joblib": "joblib",
            "catboost": "catboost",
        }.items()
    }


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value
