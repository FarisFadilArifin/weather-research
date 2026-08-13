from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .bucket_probability import (
    ARTIFACT_TYPE,
    FEATURE_PROFILE_COMMON_NO_PEAK,
    OFFSET_LABELS,
    CandidateSpec,
    _fit_candidate,
    _predict_candidate,
    canonical_two_degree_bucket,
    degree_to_bucket_probabilities,
    empirical_probabilities,
    empirical_probabilities_from_state,
    expand_offset_probabilities,
    fit_empirical_state,
    fit_tail_policy,
    normalize_probabilities,
    probability_feature_names,
    probability_base_methods,
    probability_mandatory_feature_names,
    score_probabilities,
    temperature_scale,
)
from .v19_bucket import crossfit_ridge_predictions


MODEL_FAMILIES = (
    "ordinal_logistic",
    "shared_slope_ordinal_logistic",
)
MODEL_WEIGHTS = (0.25, 0.50, 0.75, 1.0)
REGULARIZATION_VALUES = (0.01, 0.03, 0.10)
TEMPERATURES = (0.75, 1.0, 1.25, 1.5, 2.0)
PRIOR_STRENGTHS = (15.0, 30.0, 60.0)
CALIBRATION_DAYS = 90
FRESH_SHADOW_START = "2026-07-31"
FROZEN_CANDIDATE_ROLES = (
    "blended_ordinal",
    "shared_slope_ordinal",
    "pure_ordinal",
)


@dataclass(frozen=True)
class ChallengerConfig:
    family: str
    feature_set: str
    c: float
    class_weight: str | None
    temperature: float
    model_weight: float
    prior_strength: float


def feature_sets(
    feature_profile: str = FEATURE_PROFILE_COMMON_NO_PEAK,
) -> dict[str, list[str]]:
    full = probability_feature_names(
        include_peak_features=False,
        feature_profile=feature_profile,
    )
    base_methods = probability_base_methods(feature_profile)
    compact = [
        "point_prediction_f",
        "rounded_point_degree_f",
        "point_rounding_remainder_f",
        "point_distance_to_round_boundary_f",
        "point_signed_distance_to_round_boundary_f",
        *(f"{method}_predicted_high_f" for method in base_methods),
        "base_prediction_mean_f",
        "base_prediction_spread_f",
        "base_prediction_std_f",
        *(f"{method}_minus_point_f" for method in base_methods),
        "gfs_minus_point_f",
        "hrrr_minus_point_f",
        "nbm_minus_point_f",
        "point_minus_observed_temp_f",
        "point_minus_observed_high_f",
        "gfs_high_f",
        "hrrr_high_f",
        "nbm_high_f",
        "observed_temp_at_as_of_f",
        "observed_high_temp_through_as_of_f",
        "observed_as_of_age_minutes",
        "day_of_year_sin",
        "day_of_year_cos",
    ]
    market_core = [
        "point_prediction_f",
        "rounded_point_degree_f",
        "point_rounding_remainder_f",
        "point_distance_to_round_boundary_f",
        "point_signed_distance_to_round_boundary_f",
        "base_prediction_mean_f",
        "base_prediction_spread_f",
        "base_prediction_std_f",
        "provider_spread_high_f",
        "provider_std_high_f",
        "point_minus_observed_temp_f",
        "point_minus_observed_high_f",
        "observed_temp_at_as_of_f",
        "observed_high_temp_through_as_of_f",
        "observed_as_of_age_minutes",
        "v11sf_forecast_temp_11am_minus_observed_f",
        "v11sf_forecast_temp_11am_spread_f",
        "v11sf_observation_adjusted_provider_high_f",
        "v11sf_forecast_warmup_after_11am_f",
        "day_of_year_sin",
        "day_of_year_cos",
    ]
    return {
        "market_core_21": market_core,
        f"compact_{len(compact)}": compact,
        f"full_{len(full)}": full,
    }


def build_frames(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    root = Path(project_root).resolve()
    output = root / "data" / "calibration" / "station_training_baseline" / "KDAL"
    features_path = output / "KDAL_features.csv"
    validation_path = output / "KDAL_year_split_validation_predictions.csv"
    test_path = output / "KDAL_year_split_test_predictions.csv"
    point_bundle_path = (
        output
        / "model_weights"
        / "KDAL_station_high_regressor_baseline_kdal_no_peak_stack.joblib"
    )
    required = (features_path, validation_path, test_path, point_bundle_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing baseline artifacts: " + ", ".join(missing))

    from .bucket_probability import build_probability_frame

    features = pd.read_csv(features_path)
    validation = pd.read_csv(validation_path)
    test = pd.read_csv(test_path)
    point_forward = crossfit_ridge_predictions(validation)
    if point_forward.empty:
        raise ValueError("baseline validation predictions produced no honest ridge rows")
    if not (
        point_forward["train_through_year"]
        < point_forward["validation_year"]
    ).all():
        raise AssertionError("point cross-fit chronology failed")
    development = build_probability_frame(
        features,
        point_forward,
        validation,
        include_peak_features=False,
        feature_profile=FEATURE_PROFILE_COMMON_NO_PEAK,
    )
    holdout_point = test.loc[
        test["method"].eq("ridge_stack"),
        ["contract_date", "actual_high_f", "predicted_high_f"],
    ]
    holdout = build_probability_frame(
        features,
        holdout_point,
        test,
        include_peak_features=False,
        feature_profile=FEATURE_PROFILE_COMMON_NO_PEAK,
    )
    holdout = holdout.loc[holdout["year"].eq(2026)].copy()
    return development, holdout, {
        "features": features_path,
        "validation_predictions": validation_path,
        "test_predictions": test_path,
        "point_bundle": point_bundle_path,
    }


def _blend(
    empirical: np.ndarray, model: np.ndarray, model_weight: float
) -> np.ndarray:
    return normalize_probabilities(
        (1.0 - float(model_weight)) * empirical
        + float(model_weight) * model
    )


def distribution_metrics(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    tail_policy: Mapping[str, Any],
) -> tuple[dict[str, float], pd.DataFrame]:
    offset_scores = score_probabilities(
        frame["offset_class"].to_numpy(dtype=int),
        probabilities,
    )
    rows: list[dict[str, Any]] = []
    for position, (_, row) in enumerate(frame.iterrows()):
        degree_probabilities = expand_offset_probabilities(
            int(row["point_degree_f"]),
            probabilities[position],
            tail_policy,
            observed_high_f=float(row["observed_high_temp_through_as_of_f"]),
        )
        bucket_probabilities = degree_to_bucket_probabilities(
            degree_probabilities
        )
        actual_bucket = canonical_two_degree_bucket(
            int(row["actual_degree_f"])
        )
        point_bucket = canonical_two_degree_bucket(
            int(row["point_degree_f"])
        )
        ranked = sorted(
            bucket_probabilities.items(),
            key=lambda item: (-item[1], item[0]),
        )
        actual_probability = float(
            bucket_probabilities.get(actual_bucket, 0.0)
        )
        rows.append(
            {
                "contract_date": pd.Timestamp(row["contract_date"]),
                "validation_year": int(row["year"]),
                "actual_high_f": float(row["actual_high_f"]),
                "actual_degree_f": int(row["actual_degree_f"]),
                "exact_offset": int(row["exact_offset"]),
                "offset_class": int(row["offset_class"]),
                "point_prediction_f": float(row["point_prediction_f"]),
                "point_degree_f": int(row["point_degree_f"]),
                "actual_bucket": actual_bucket,
                "point_bucket": point_bucket,
                "point_bucket_hit": point_bucket == actual_bucket,
                "probability_top_bucket": ranked[0][0],
                "probability_top_bucket_hit": ranked[0][0] == actual_bucket,
                "top_bucket_probability": float(ranked[0][1]),
                "top_two_margin": float(
                    ranked[0][1]
                    - (ranked[1][1] if len(ranked) > 1 else 0.0)
                ),
                "actual_bucket_probability": actual_probability,
                "offset_probabilities": probabilities[position].tolist(),
                "degree_probabilities": {
                    str(key): float(value)
                    for key, value in degree_probabilities.items()
                },
                "bucket_probabilities": {
                    str(key): float(value)
                    for key, value in bucket_probabilities.items()
                },
            }
        )
    predictions = pd.DataFrame(rows)
    metrics = {
        **offset_scores,
        "bucket_log_loss": float(
            -np.log(
                predictions["actual_bucket_probability"].clip(lower=1e-12)
            ).mean()
        ),
        "bucket_accuracy": float(
            predictions["probability_top_bucket_hit"].mean()
        ),
        "point_bucket_accuracy": float(
            predictions["point_bucket_hit"].mean()
        ),
        "count": int(len(predictions)),
    }
    return metrics, predictions


def _model_probabilities(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    config: ChallengerConfig,
    features: Sequence[str],
    *,
    random_state: int,
) -> tuple[dict[str, Any] | None, np.ndarray]:
    empirical = empirical_probabilities(
        train, valid, float(config.prior_strength)
    )
    if config.family == "empirical":
        return None, empirical
    spec = CandidateSpec(
        config.family,
        {"C": float(config.c), "class_weight": config.class_weight},
    )
    state = _fit_candidate(
        train,
        features,
        spec,
        random_state=random_state,
    )
    raw = _predict_candidate(state, valid, features)
    calibrated = temperature_scale(raw, float(config.temperature))
    return state, _blend(empirical, calibrated, config.model_weight)


def tune_candidates(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    random_state: int = 42,
    feature_profile: str = FEATURE_PROFILE_COMMON_NO_PEAK,
) -> pd.DataFrame:
    sets = feature_sets(feature_profile)
    tail_policy = fit_tail_policy(train["exact_offset"].astype(int))
    rows: list[dict[str, Any]] = []
    for prior_strength in PRIOR_STRENGTHS:
        empirical = empirical_probabilities(train, valid, prior_strength)
        metrics, _ = distribution_metrics(
            valid, empirical, tail_policy=tail_policy
        )
        rows.append(
            {
                **asdict(
                    ChallengerConfig(
                        "empirical",
                        "historical_only",
                        0.0,
                        None,
                        1.0,
                        0.0,
                        prior_strength,
                    )
                ),
                "feature_count": 0,
                **metrics,
            }
        )
    for family in MODEL_FAMILIES:
        for feature_set, features in sets.items():
            for c in REGULARIZATION_VALUES:
                spec = CandidateSpec(
                    family,
                    {"C": c, "class_weight": None},
                )
                state = _fit_candidate(
                    train, features, spec, random_state=random_state
                )
                raw = _predict_candidate(state, valid, features)
                empirical_by_prior = {
                    prior: empirical_probabilities(train, valid, prior)
                    for prior in PRIOR_STRENGTHS
                }
                for temperature in TEMPERATURES:
                    calibrated = temperature_scale(raw, temperature)
                    for prior_strength, empirical in empirical_by_prior.items():
                        for model_weight in MODEL_WEIGHTS:
                            probabilities = _blend(
                                empirical, calibrated, model_weight
                            )
                            metrics, _ = distribution_metrics(
                                valid,
                                probabilities,
                                tail_policy=tail_policy,
                            )
                            rows.append(
                                {
                                    **asdict(
                                        ChallengerConfig(
                                            family,
                                            feature_set,
                                            c,
                                            None,
                                            temperature,
                                            model_weight,
                                            prior_strength,
                                        )
                                    ),
                                    "feature_count": len(features),
                                    **metrics,
                                }
                            )
    candidates = pd.DataFrame(rows)
    family_rank = {
        "empirical": 0,
        "shared_slope_ordinal_logistic": 1,
        "ordinal_logistic": 2,
    }
    candidates["family_rank"] = candidates["family"].map(family_rank)
    return candidates.sort_values(
        [
            "bucket_log_loss",
            "ranked_probability_score",
            "log_loss",
            "feature_count",
            "family_rank",
            "c",
            "temperature",
            "model_weight",
            "prior_strength",
        ],
        ascending=True,
    ).reset_index(drop=True)


def row_to_config(row: Mapping[str, Any]) -> ChallengerConfig:
    return ChallengerConfig(
        family=str(row["family"]),
        feature_set=str(row["feature_set"]),
        c=float(row["c"]),
        class_weight=(
            None
            if pd.isna(row.get("class_weight"))
            else str(row["class_weight"])
        ),
        temperature=float(row["temperature"]),
        model_weight=float(row["model_weight"]),
        prior_strength=float(row["prior_strength"]),
    )


def fit_and_predict(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    config: ChallengerConfig,
    *,
    random_state: int = 42,
    feature_profile: str = FEATURE_PROFILE_COMMON_NO_PEAK,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    features = (
        []
        if config.family == "empirical"
        else feature_sets(feature_profile)[config.feature_set]
    )
    model_state, probabilities = _model_probabilities(
        train,
        valid,
        config,
        features,
        random_state=random_state,
    )
    tail_policy = fit_tail_policy(train["exact_offset"].astype(int))
    metrics, predictions = distribution_metrics(
        valid, probabilities, tail_policy=tail_policy
    )
    state = {
        "config": asdict(config),
        "feature_names": features,
        "model_state": model_state,
        "empirical_state": fit_empirical_state(train),
        "tail_policy": tail_policy,
        "training_start": pd.Timestamp(train["contract_date"].min())
        .date()
        .isoformat(),
        "training_cutoff": pd.Timestamp(train["contract_date"].max())
        .date()
        .isoformat(),
        "training_rows": int(len(train)),
    }
    return metrics, predictions, state


def tune_no_override_policy(
    predictions: pd.DataFrame, *, target_coverage: float = 0.60
) -> dict[str, float]:
    candidates: list[dict[str, float]] = []
    for minimum_top in np.arange(0.20, 0.701, 0.025):
        for minimum_margin in np.arange(0.0, 0.301, 0.025):
            selected = predictions["top_bucket_probability"].ge(
                minimum_top
            ) & predictions["top_two_margin"].ge(minimum_margin)
            if not selected.any():
                continue
            coverage = float(selected.mean())
            candidates.append(
                {
                    "minimum_top_probability": float(minimum_top),
                    "minimum_top_two_margin": float(minimum_margin),
                    "coverage": coverage,
                    "coverage_distance": abs(coverage - target_coverage),
                    "point_bucket_accuracy": float(
                        predictions.loc[selected, "point_bucket_hit"].mean()
                    ),
                }
            )
    if not candidates:
        raise ValueError("no confidence policy candidate was available")
    frame = pd.DataFrame(candidates)
    preferred = frame.loc[frame["coverage"].between(0.55, 0.65)].copy()
    if preferred.empty:
        preferred = frame
    selected = preferred.sort_values(
        [
            "point_bucket_accuracy",
            "coverage_distance",
            "minimum_top_probability",
            "minimum_top_two_margin",
        ],
        ascending=[False, True, True, True],
    ).iloc[0]
    return {
        "minimum_top_probability": float(
            selected["minimum_top_probability"]
        ),
        "minimum_top_two_margin": float(
            selected["minimum_top_two_margin"]
        ),
        "target_coverage": float(target_coverage),
        "overrides_enabled": False,
    }


def apply_no_override_policy(
    predictions: pd.DataFrame, policy: Mapping[str, Any]
) -> pd.DataFrame:
    out = predictions.copy()
    out["shadow_trade"] = (
        out["top_bucket_probability"].ge(
            float(policy["minimum_top_probability"])
        )
        & out["top_two_margin"].ge(
            float(policy["minimum_top_two_margin"])
        )
    )
    out["recommended_bucket"] = out["point_bucket"]
    out["recommended_bucket_hit"] = out["point_bucket_hit"]
    out["overrides_point_bucket"] = False
    return out


def policy_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    selected = predictions["shadow_trade"].astype(bool)
    return {
        "count": int(len(predictions)),
        "coverage": float(selected.mean()),
        "selected_count": int(selected.sum()),
        "selected_point_bucket_accuracy": (
            float(predictions.loc[selected, "point_bucket_hit"].mean())
            if selected.any()
            else math.nan
        ),
        "full_point_bucket_accuracy": float(
            predictions["point_bucket_hit"].mean()
        ),
        "override_count": 0,
    }


def _inner_split(
    history: pd.DataFrame, calibration_days: int = CALIBRATION_DAYS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_at = history["contract_date"].max() - pd.Timedelta(
        days=calibration_days - 1
    )
    train = history.loc[history["contract_date"].lt(split_at)].copy()
    valid = history.loc[history["contract_date"].ge(split_at)].copy()
    if train.empty or valid.empty:
        raise ValueError("insufficient data for inner calibration split")
    if train["contract_date"].max() >= valid["contract_date"].min():
        raise AssertionError("inner split is not chronological")
    return train, valid


def nested_forward_evaluation(
    development: pd.DataFrame,
    *,
    random_state: int = 42,
    feature_profile: str = FEATURE_PROFILE_COMMON_NO_PEAK,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    development = development.loc[
        development["year"].between(2023, 2025)
    ].copy()
    predictions: list[pd.DataFrame] = []
    selections: list[dict[str, Any]] = []
    tuning_frames: list[pd.DataFrame] = []
    for validation_year in (2024, 2025):
        outer_train = development.loc[
            development["year"].lt(validation_year)
        ].copy()
        outer_valid = development.loc[
            development["year"].eq(validation_year)
        ].copy()
        inner_train, inner_valid = _inner_split(outer_train)
        tuning = tune_candidates(
            inner_train,
            inner_valid,
            random_state=random_state,
            feature_profile=feature_profile,
        )
        tuning["outer_validation_year"] = validation_year
        tuning_frames.append(tuning)
        selected_config = row_to_config(tuning.iloc[0])
        _, inner_predictions, _ = fit_and_predict(
            inner_train,
            inner_valid,
            selected_config,
            random_state=random_state,
            feature_profile=feature_profile,
        )
        policy = tune_no_override_policy(inner_predictions)
        outer_metrics, outer_predictions, _ = fit_and_predict(
            outer_train,
            outer_valid,
            selected_config,
            random_state=random_state,
            feature_profile=feature_profile,
        )
        outer_predictions = apply_no_override_policy(
            outer_predictions, policy
        )
        outer_predictions["outer_validation_year"] = validation_year
        outer_predictions["model_training_cutoff"] = outer_train[
            "contract_date"
        ].max()
        predictions.append(outer_predictions)
        selections.append(
            {
                "outer_validation_year": validation_year,
                **asdict(selected_config),
                **{
                    f"outer_{key}": value
                    for key, value in outer_metrics.items()
                },
                **{
                    f"policy_{key}": value
                    for key, value in policy_metrics(
                        outer_predictions
                    ).items()
                },
                **{
                    f"threshold_{key}": value
                    for key, value in policy.items()
                    if key != "overrides_enabled"
                },
            }
        )
    return (
        pd.concat(predictions, ignore_index=True),
        pd.DataFrame(selections),
        pd.concat(tuning_frames, ignore_index=True),
    )


def frozen_candidate_rows(tuning: pd.DataFrame) -> pd.DataFrame:
    ordered = tuning.sort_values(
        [
            "bucket_log_loss",
            "ranked_probability_score",
            "log_loss",
            "feature_count",
        ]
    )
    role_filters = (
        (
            "blended_ordinal",
            ordered["family"].eq("ordinal_logistic")
            & ordered["model_weight"].lt(1.0),
        ),
        (
            "shared_slope_ordinal",
            ordered["family"].eq("shared_slope_ordinal_logistic"),
        ),
        (
            "pure_ordinal",
            ordered["family"].eq("ordinal_logistic")
            & ordered["model_weight"].eq(1.0),
        ),
    )
    chosen: list[pd.Series] = []
    for role, mask in role_filters:
        candidates = ordered.loc[mask]
        if candidates.empty:
            raise ValueError(f"no tuning candidate is available for {role}")
        selected = candidates.iloc[0].copy()
        selected["candidate_role"] = role
        chosen.append(selected)
    frame = pd.DataFrame(chosen).reset_index(drop=True)
    if tuple(frame["candidate_role"]) != FROZEN_CANDIDATE_ROLES:
        raise AssertionError("frozen challenger roles are not in contract order")
    return frame


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_frozen_candidate(
    output_dir: Path,
    *,
    station_id: str,
    point_model_version: str,
    point_bundle_path: Path,
    config: ChallengerConfig,
    state: Mapping[str, Any],
    policy: Mapping[str, Any],
    historical_metrics: Mapping[str, Any],
    candidate_name: str,
    feature_profile: str = FEATURE_PROFILE_COMMON_NO_PEAK,
) -> tuple[Path, Path]:
    import joblib

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    bundle = {
        "schema_version": 1,
        "artifact_type": "station_ordinal_probability_challenger",
        "station_id": station_id,
        "model_version": candidate_name,
        "point_model_version": point_model_version,
        "point_bundle_sha256": sha256_file(point_bundle_path),
        "feature_profile": feature_profile,
        "base_methods": list(probability_base_methods(feature_profile)),
        "feature_names": list(state["feature_names"]),
        "mandatory_source_features": list(
            probability_mandatory_feature_names(feature_profile)
        ),
        "offset_labels": list(OFFSET_LABELS),
        "selected_family": config.family,
        "selected_params": {
            "C": config.c,
            "class_weight": config.class_weight,
        },
        "model_state": state["model_state"],
        "temperature": config.temperature,
        "blend_weight": config.model_weight,
        "empirical_prior_strength": config.prior_strength,
        "empirical_state": state["empirical_state"],
        "tail_policy": state["tail_policy"],
        "decision_thresholds": dict(policy),
        "overrides_enabled": False,
        "training_start": state["training_start"],
        "training_cutoff": state["training_cutoff"],
        "training_rows": state["training_rows"],
        "historical_metrics": dict(historical_metrics),
        "holdout_status": "exploratory_previously_inspected",
        "promotion_approved": False,
        "promotion_blocker": "fresh_shadow_data_required",
        "fresh_shadow_start_contract_date": FRESH_SHADOW_START,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    bundle_path = output / f"{candidate_name}.joblib"
    manifest_path = output / f"{candidate_name}.json"
    joblib.dump(bundle, bundle_path)
    manifest = {
        key: value
        for key, value in bundle.items()
        if key
        not in {
            "model_state",
            "empirical_state",
        }
    }
    manifest["artifact_integrity"] = {
        "bundle_sha256": sha256_file(bundle_path)
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return bundle_path, manifest_path


def serialize_prediction_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in (
        "offset_probabilities",
        "degree_probabilities",
        "bucket_probabilities",
    ):
        if column in out:
            out[column] = out[column].map(
                lambda value: json.dumps(value, sort_keys=True)
            )
    if "contract_date" in out:
        out["contract_date"] = pd.to_datetime(
            out["contract_date"]
        ).dt.date.astype(str)
    return out
