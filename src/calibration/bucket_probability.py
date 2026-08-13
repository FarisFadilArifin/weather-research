from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .station_stacking import (
    V20_ENGINEERED_FEATURE_COLUMNS,
    V20_KDAL_1PM_TEMP_FEATURE_COLUMNS,
    V20_PEAK_TIMING_RAW_FEATURE_COLUMNS,
)


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "station_bucket_probability_model"
OFFSET_LABELS = ("le_-4", "-3", "-2", "-1", "0", "+1", "+2", "+3", "ge_+4")
CENTRAL_OFFSETS = (-3, -2, -1, 0, 1, 2, 3)
BASE_METHODS = ("xgboost", "lightgbm", "catboost")
EXPERT_ENSEMBLE_BASE_METHODS = (
    "full_xgboost",
    "forecast_huber",
    "observation_catboost",
    "seasonal_ridge",
)
PROVIDERS = ("gfs", "hrrr", "nbm")
ASIA_PROVIDERS = ("gfs", "gefs", "jma_msm")
EFFECTIVE_TIE_TOLERANCE = 1e-3
FEATURE_PROFILE_COMMON_NO_PEAK = "common_no_peak"
FEATURE_PROFILE_PEAK_AUGMENTED = "peak_augmented"
FEATURE_PROFILE_KDAL_1PM = "kdal_1pm"
FEATURE_PROFILE_ASIA_NO_PEAK = "asia_no_peak"
FEATURE_PROFILE_EXPERT_ENSEMBLE_COMMON_NO_PEAK = "expert_ensemble_common_no_peak"
FEATURE_PROFILE_EXPERT_ENSEMBLE_ASIA_NO_PEAK = "expert_ensemble_asia_no_peak"
FEATURE_PROFILES = (
    FEATURE_PROFILE_COMMON_NO_PEAK,
    FEATURE_PROFILE_PEAK_AUGMENTED,
    FEATURE_PROFILE_KDAL_1PM,
    FEATURE_PROFILE_ASIA_NO_PEAK,
    FEATURE_PROFILE_EXPERT_ENSEMBLE_COMMON_NO_PEAK,
    FEATURE_PROFILE_EXPERT_ENSEMBLE_ASIA_NO_PEAK,
)

MANDATORY_SOURCE_FEATURES = (
    "gfs_high_f",
    "hrrr_high_f",
    "nbm_high_f",
    "observed_temp_at_as_of_f",
    "observed_high_temp_through_as_of_f",
    "observed_as_of_age_minutes",
)

ASIA_MANDATORY_SOURCE_FEATURES = (
    "gfs_high_f",
    "gefs_high_f",
    "jma_msm_high_f",
    "observed_temp_at_as_of_f",
    "observed_high_temp_through_as_of_f",
    "observed_as_of_age_minutes",
)

COMMON_SOURCE_FEATURES = (
    *MANDATORY_SOURCE_FEATURES,
    "provider_mean_high_f",
    "provider_spread_high_f",
    "provider_std_high_f",
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
)

ASIA_SOURCE_FEATURES = (
    *ASIA_MANDATORY_SOURCE_FEATURES,
    "provider_mean_high_f",
    "provider_spread_high_f",
    "provider_std_high_f",
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
)

KDAL_1PM_SOURCE_FEATURES = (
    *MANDATORY_SOURCE_FEATURES,
    "provider_mean_high_f",
    "provider_spread_high_f",
    "provider_std_high_f",
    "observed_temp_change_last_1h_f",
    "observed_temp_change_last_3h_f",
    "observed_morning_warmup_rate_f_per_hour",
    "observed_high_so_far_change_since_9am_f",
    "observed_temp_change_since_11am_f",
    "observed_high_so_far_change_since_11am_f",
    "observed_dewpoint_at_as_of_f",
    "observed_humidity_at_as_of",
    "observed_cloud_cover_at_as_of",
    "observed_precip_recent_at_as_of",
    *V20_KDAL_1PM_TEMP_FEATURE_COLUMNS,
    "day_of_year_sin",
    "day_of_year_cos",
)

BASE_DERIVED_FEATURES = (
    "point_prediction_f",
    "rounded_point_degree_f",
    "point_rounding_remainder_f",
    "point_distance_to_round_boundary_f",
    "point_signed_distance_to_round_boundary_f",
    "xgboost_predicted_high_f",
    "lightgbm_predicted_high_f",
    "catboost_predicted_high_f",
    "base_prediction_mean_f",
    "base_prediction_spread_f",
    "base_prediction_std_f",
    "xgboost_minus_point_f",
    "lightgbm_minus_point_f",
    "catboost_minus_point_f",
)

DERIVED_FEATURES = (
    *BASE_DERIVED_FEATURES,
    "gfs_minus_point_f",
    "hrrr_minus_point_f",
    "nbm_minus_point_f",
    "point_minus_observed_temp_f",
    "point_minus_observed_high_f",
)

ASIA_DERIVED_FEATURES = (
    *BASE_DERIVED_FEATURES,
    "gfs_minus_point_f",
    "gefs_minus_point_f",
    "jma_msm_minus_point_f",
    "point_minus_observed_temp_f",
    "point_minus_observed_high_f",
)


def _base_derived_features(methods: Sequence[str]) -> tuple[str, ...]:
    return (
        "point_prediction_f",
        "rounded_point_degree_f",
        "point_rounding_remainder_f",
        "point_distance_to_round_boundary_f",
        "point_signed_distance_to_round_boundary_f",
        *(f"{method}_predicted_high_f" for method in methods),
        "base_prediction_mean_f",
        "base_prediction_spread_f",
        "base_prediction_std_f",
        *(f"{method}_minus_point_f" for method in methods),
    )


EXPERT_ENSEMBLE_DERIVED_FEATURES = (
    *_base_derived_features(EXPERT_ENSEMBLE_BASE_METHODS),
    "gfs_minus_point_f",
    "hrrr_minus_point_f",
    "nbm_minus_point_f",
    "point_minus_observed_temp_f",
    "point_minus_observed_high_f",
)

EXPERT_ENSEMBLE_ASIA_DERIVED_FEATURES = (
    *_base_derived_features(EXPERT_ENSEMBLE_BASE_METHODS),
    "gfs_minus_point_f",
    "gefs_minus_point_f",
    "jma_msm_minus_point_f",
    "point_minus_observed_temp_f",
    "point_minus_observed_high_f",
)

MISSING_INDICATOR_SUFFIX = "__missing"


def round_half_up(value: float) -> int:
    if not math.isfinite(float(value)):
        raise ValueError("temperature must be finite")
    return int(Decimal(str(float(value))).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def offset_class_index(exact_offset: int) -> int:
    if exact_offset <= -4:
        return 0
    if exact_offset >= 4:
        return 8
    return exact_offset + 4


def offset_label(exact_offset: int) -> str:
    return OFFSET_LABELS[offset_class_index(exact_offset)]


def canonical_two_degree_bucket(degree_f: int) -> str:
    low = degree_f if degree_f % 2 == 0 else degree_f - 1
    return f"{low}-{low + 1}"


def _resolve_feature_profile(*, include_peak_features: bool, feature_profile: str | None) -> str:
    profile = feature_profile or (
        FEATURE_PROFILE_PEAK_AUGMENTED if include_peak_features else FEATURE_PROFILE_COMMON_NO_PEAK
    )
    if profile not in FEATURE_PROFILES:
        raise ValueError("unknown probability feature profile: " + str(profile))
    if include_peak_features != (profile == FEATURE_PROFILE_PEAK_AUGMENTED):
        raise ValueError("include_peak_features does not match probability feature profile")
    return profile


def probability_provider_names(feature_profile: str) -> tuple[str, ...]:
    return (
        ASIA_PROVIDERS
        if feature_profile in (
            FEATURE_PROFILE_ASIA_NO_PEAK,
            FEATURE_PROFILE_EXPERT_ENSEMBLE_ASIA_NO_PEAK,
        )
        else PROVIDERS
    )


def probability_base_methods(feature_profile: str) -> tuple[str, ...]:
    if feature_profile in (
        FEATURE_PROFILE_EXPERT_ENSEMBLE_COMMON_NO_PEAK,
        FEATURE_PROFILE_EXPERT_ENSEMBLE_ASIA_NO_PEAK,
    ):
        return EXPERT_ENSEMBLE_BASE_METHODS
    return BASE_METHODS


def probability_mandatory_feature_names(
    feature_profile: str,
) -> tuple[str, ...]:
    return (
        ASIA_MANDATORY_SOURCE_FEATURES
        if feature_profile in (
            FEATURE_PROFILE_ASIA_NO_PEAK,
            FEATURE_PROFILE_EXPERT_ENSEMBLE_ASIA_NO_PEAK,
        )
        else MANDATORY_SOURCE_FEATURES
    )


def _profile_feature_contract(
    feature_profile: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if feature_profile == FEATURE_PROFILE_EXPERT_ENSEMBLE_ASIA_NO_PEAK:
        return EXPERT_ENSEMBLE_ASIA_DERIVED_FEATURES, ASIA_SOURCE_FEATURES
    if feature_profile == FEATURE_PROFILE_EXPERT_ENSEMBLE_COMMON_NO_PEAK:
        return EXPERT_ENSEMBLE_DERIVED_FEATURES, COMMON_SOURCE_FEATURES
    if feature_profile == FEATURE_PROFILE_ASIA_NO_PEAK:
        return ASIA_DERIVED_FEATURES, ASIA_SOURCE_FEATURES
    sources = (
        KDAL_1PM_SOURCE_FEATURES
        if feature_profile == FEATURE_PROFILE_KDAL_1PM
        else COMMON_SOURCE_FEATURES
    )
    return DERIVED_FEATURES, sources


def probability_feature_names(
    *, include_peak_features: bool, feature_profile: str | None = None
) -> list[str]:
    profile = _resolve_feature_profile(
        include_peak_features=include_peak_features, feature_profile=feature_profile
    )
    derived, sources = _profile_feature_contract(profile)
    names = [*derived, *sources]
    if profile == FEATURE_PROFILE_PEAK_AUGMENTED:
        names.extend(V20_PEAK_TIMING_RAW_FEATURE_COLUMNS)
        names.extend(V20_ENGINEERED_FEATURE_COLUMNS)
    core_names = list(dict.fromkeys(names))
    optional_sources = probability_optional_feature_names(
        include_peak_features=include_peak_features,
        feature_profile=profile,
    )
    return [
        *core_names,
        *(f"{name}{MISSING_INDICATOR_SUFFIX}" for name in optional_sources),
    ]


def probability_optional_feature_names(
    *, include_peak_features: bool, feature_profile: str | None = None
) -> list[str]:
    profile = _resolve_feature_profile(
        include_peak_features=include_peak_features, feature_profile=feature_profile
    )
    _, sources = _profile_feature_contract(profile)
    mandatory = probability_mandatory_feature_names(profile)
    names = list(sources[len(mandatory) :])
    if profile == FEATURE_PROFILE_PEAK_AUGMENTED:
        names.extend(V20_PEAK_TIMING_RAW_FEATURE_COLUMNS)
        names.extend(V20_ENGINEERED_FEATURE_COLUMNS)
    return list(dict.fromkeys(names))


def build_probability_frame(
    feature_frame: pd.DataFrame,
    point_predictions: pd.DataFrame,
    base_validation_predictions: pd.DataFrame,
    *,
    include_peak_features: bool,
    feature_profile: str | None = None,
) -> pd.DataFrame:
    """Join honest point/base OOF predictions to their same-day live-safe features."""
    required_point = {"contract_date", "actual_high_f", "predicted_high_f"}
    missing = sorted(required_point - set(point_predictions.columns))
    if missing:
        raise ValueError("point predictions missing: " + ",".join(missing))
    required_base = {"contract_date", "method", "predicted_high_f"}
    missing = sorted(required_base - set(base_validation_predictions.columns))
    if missing:
        raise ValueError("base predictions missing: " + ",".join(missing))

    point = point_predictions.copy()
    point["contract_date"] = pd.to_datetime(point["contract_date"], errors="coerce")
    point = point.dropna(subset=["contract_date", "actual_high_f", "predicted_high_f"])
    point = point.rename(columns={"predicted_high_f": "point_prediction_f"})

    features = feature_frame.copy()
    features["contract_date"] = pd.to_datetime(features["contract_date"], errors="coerce")
    if features["contract_date"].duplicated().any():
        raise ValueError("feature frame has duplicate contract dates")
    resolved_profile = _resolve_feature_profile(
        include_peak_features=include_peak_features,
        feature_profile=feature_profile,
    )
    base_methods = probability_base_methods(resolved_profile)
    base = base_validation_predictions.loc[
        base_validation_predictions["method"].isin(base_methods),
        ["contract_date", "method", "predicted_high_f"],
    ].copy()
    base["contract_date"] = pd.to_datetime(base["contract_date"], errors="coerce")
    duplicate = base.duplicated(["contract_date", "method"], keep=False)
    if duplicate.any():
        conflicts = base.loc[duplicate].groupby(["contract_date", "method"])["predicted_high_f"].nunique(dropna=False)
        if conflicts.gt(1).any():
            raise ValueError("conflicting base OOF predictions for station-date")
        base = base.drop_duplicates(["contract_date", "method"])
    base = base.pivot(index="contract_date", columns="method", values="predicted_high_f")
    base = base.rename(columns={name: f"{name}_predicted_high_f" for name in base_methods})
    base = base.reset_index()

    merged = point.merge(base, on="contract_date", how="inner", validate="one_to_one")
    merged = merged.merge(features, on="contract_date", how="left", validate="one_to_one", suffixes=("", "_feature"))
    merged = add_probability_features(
        merged,
        providers=probability_provider_names(resolved_profile),
        base_methods=base_methods,
    )
    core_feature_names = [
        name
        for name in probability_feature_names(
            include_peak_features=include_peak_features, feature_profile=feature_profile
        )
        if not name.endswith(MISSING_INDICATOR_SUFFIX)
    ]
    for name in core_feature_names:
        if name not in merged:
            merged[name] = np.nan
        merged[name] = pd.to_numeric(merged[name], errors="coerce")
    serving_required = [
        *probability_mandatory_feature_names(resolved_profile),
        "point_prediction_f",
        *(f"{name}_predicted_high_f" for name in base_methods),
    ]
    merged = merged.dropna(subset=serving_required).copy()
    for name in probability_optional_feature_names(
        include_peak_features=include_peak_features, feature_profile=feature_profile
    ):
        merged[f"{name}{MISSING_INDICATOR_SUFFIX}"] = merged[name].isna().astype(float)
    merged = merged.copy()
    merged["point_degree_f"] = merged["point_prediction_f"].map(round_half_up)
    merged["actual_degree_f"] = pd.to_numeric(merged["actual_high_f"], errors="coerce").map(round_half_up)
    merged["exact_offset"] = merged["actual_degree_f"] - merged["point_degree_f"]
    merged["offset_class"] = merged["exact_offset"].astype(int).map(offset_class_index)
    merged["year"] = merged["contract_date"].dt.year
    merged["month"] = merged["contract_date"].dt.month
    return merged.sort_values("contract_date").reset_index(drop=True)


def add_probability_features(
    frame: pd.DataFrame,
    *,
    providers: Sequence[str] = PROVIDERS,
    base_methods: Sequence[str] = BASE_METHODS,
) -> pd.DataFrame:
    out = frame.copy()
    point = pd.to_numeric(out["point_prediction_f"], errors="coerce")
    rounded = point.map(lambda value: round_half_up(value) if pd.notna(value) else np.nan)
    out["rounded_point_degree_f"] = rounded
    out["point_rounding_remainder_f"] = point - rounded
    out["point_distance_to_round_boundary_f"] = 0.5 - out["point_rounding_remainder_f"].abs()
    remainder = out["point_rounding_remainder_f"]
    out["point_signed_distance_to_round_boundary_f"] = np.where(
        remainder.ge(0.0), remainder - 0.5, remainder + 0.5
    )
    base_columns = [f"{name}_predicted_high_f" for name in base_methods]
    base = out.reindex(columns=base_columns).apply(pd.to_numeric, errors="coerce")
    out["base_prediction_mean_f"] = base.mean(axis=1)
    out["base_prediction_spread_f"] = base.max(axis=1) - base.min(axis=1)
    out["base_prediction_std_f"] = base.std(axis=1, ddof=0)
    for name in base_methods:
        out[f"{name}_minus_point_f"] = pd.to_numeric(out.get(f"{name}_predicted_high_f"), errors="coerce") - point
    for name in providers:
        out[f"{name}_minus_point_f"] = pd.to_numeric(out.get(f"{name}_high_f"), errors="coerce") - point
    out["point_minus_observed_temp_f"] = point - pd.to_numeric(out.get("observed_temp_at_as_of_f"), errors="coerce")
    out["point_minus_observed_high_f"] = point - pd.to_numeric(out.get("observed_high_temp_through_as_of_f"), errors="coerce")
    return out


@dataclass(frozen=True)
class CandidateSpec:
    family: str
    params: Mapping[str, Any]


def default_candidate_specs() -> list[CandidateSpec]:
    specs = [CandidateSpec("empirical", {"prior_strength": value}) for value in (15.0, 30.0, 60.0)]
    for c in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
        for class_weight in (None, "balanced"):
            specs.append(CandidateSpec("ordinal_logistic", {"C": c, "class_weight": class_weight}))
    for estimators in (50, 100, 200):
        for leaves in (5, 9, 15):
            for child_rows in (30, 60):
                for l2 in (5.0, 20.0):
                    specs.append(
                        CandidateSpec(
                            "lightgbm_multiclass",
                            {
                                "learning_rate": 0.03,
                                "n_estimators": estimators,
                                "num_leaves": leaves,
                                "min_child_samples": child_rows,
                                "reg_lambda": l2,
                                "subsample": 0.8,
                                "colsample_bytree": 0.8,
                            },
                        )
                    )
    return specs


def fit_probability_system(
    frame: pd.DataFrame,
    *,
    station_id: str,
    point_model_version: str,
    point_bundle_sha256: str,
    include_peak_features: bool,
    feature_profile: str | None = None,
    model_version: str | None = None,
    candidate_specs: Sequence[CandidateSpec] | None = None,
    forced_family: str | None = None,
    blend_weights: Sequence[float] | None = None,
    min_train_rows: int = 180,
    calibration_days: int = 90,
    random_state: int = 42,
    development_years: Sequence[int] = (2023, 2024, 2025),
    forward_validation_years: Sequence[int] = (2024, 2025),
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Select a station probability model using strict expanding folds."""
    station_id = station_id.strip().upper()
    specs = list(candidate_specs or default_candidate_specs())
    available_families = {spec.family for spec in specs}
    if forced_family is not None and forced_family not in available_families:
        raise ValueError(
            f"forced probability family {forced_family!r} is absent from candidate_specs"
        )
    resolved_blend_weights = tuple(
        float(value)
        for value in (blend_weights if blend_weights is not None else (0.25, 0.5, 0.75, 1.0))
    )
    if (
        not resolved_blend_weights
        or any(not math.isfinite(value) or value <= 0.0 or value > 1.0 for value in resolved_blend_weights)
    ):
        raise ValueError("blend_weights must contain finite values in (0, 1]")
    resolved_blend_weights = tuple(dict.fromkeys(resolved_blend_weights))
    resolved_profile = _resolve_feature_profile(
        include_peak_features=include_peak_features, feature_profile=feature_profile
    )
    feature_names = probability_feature_names(
        include_peak_features=include_peak_features, feature_profile=resolved_profile
    )
    resolved_development_years = tuple(
        dict.fromkeys(int(year) for year in development_years)
    )
    resolved_validation_years = tuple(
        dict.fromkeys(int(year) for year in forward_validation_years)
    )
    if not resolved_development_years:
        raise ValueError("development_years must not be empty")
    if (
        not resolved_validation_years
        or not set(resolved_validation_years).issubset(
            resolved_development_years
        )
    ):
        raise ValueError(
            "forward_validation_years must be a non-empty subset of "
            "development_years"
        )
    development = frame.loc[
        frame["year"].isin(resolved_development_years)
    ].copy()
    if len(development) < min_train_rows:
        raise ValueError("insufficient probability training rows")
    families = list(dict.fromkeys(spec.family for spec in specs))
    family_oof_rows: dict[str, list[pd.DataFrame]] = {family: [] for family in families}
    tuning_rows: list[dict[str, Any]] = []
    for valid_year in resolved_validation_years:
        outer_train = development.loc[development["year"].lt(valid_year)].copy()
        outer_valid = development.loc[development["year"].eq(valid_year)].copy()
        if len(outer_train) < min_train_rows or outer_valid.empty:
            continue
        split_at = outer_train["contract_date"].max() - pd.Timedelta(days=calibration_days - 1)
        inner_train = outer_train.loc[outer_train["contract_date"].lt(split_at)].copy()
        inner_valid = outer_train.loc[outer_train["contract_date"].ge(split_at)].copy()
        if len(inner_train) < max(60, min_train_rows // 2) or inner_valid.empty:
            raise ValueError(f"insufficient inner calibration history for {valid_year}")
        validation_start = outer_valid["contract_date"].min()
        if (
            outer_train["contract_date"].max() >= validation_start
            or inner_train["contract_date"].max() >= inner_valid["contract_date"].min()
            or inner_valid["contract_date"].max() >= validation_start
        ):
            raise AssertionError("probability fold history is not strictly before validation")
        for family in families:
            family_specs = [spec for spec in specs if spec.family == family]
            selected = _select_inner_candidate(
                inner_train,
                inner_valid,
                feature_names,
                family_specs,
                blend_weights=resolved_blend_weights,
                random_state=random_state,
            )
            fitted = _fit_candidate(
                outer_train, feature_names, selected["spec"], random_state=random_state
            )
            empirical = empirical_probabilities(
                outer_train, outer_valid, float(selected["prior_strength"])
            )
            if selected["spec"].family == "empirical":
                blended = empirical
            else:
                raw = _predict_candidate(fitted, outer_valid, feature_names)
                calibrated = temperature_scale(raw, float(selected["temperature"]))
                blended = blend_probabilities(
                    empirical, calibrated, float(selected["blend_weight"])
                )
            prediction_rows = _prediction_rows(
                outer_valid, blended, valid_year, selected
            )
            prediction_rows["model_training_cutoff"] = outer_train[
                "contract_date"
            ].max()
            prediction_rows["calibration_training_cutoff"] = inner_train[
                "contract_date"
            ].max()
            prediction_rows["calibration_validation_start"] = inner_valid[
                "contract_date"
            ].min()
            prediction_rows["calibration_validation_cutoff"] = inner_valid[
                "contract_date"
            ].max()
            family_oof_rows[family].append(prediction_rows)
            tuning_rows.append(
                {
                    "validation_year": valid_year,
                    "family": family,
                    "params_json": json.dumps(dict(selected["spec"].params), sort_keys=True),
                    "temperature": selected["temperature"],
                    "blend_weight": selected["blend_weight"],
                    "prior_strength": selected["prior_strength"],
                    "inner_log_loss": selected["log_loss"],
                    "inner_brier": selected["brier"],
                }
            )
    available_oof = {
        family: pd.concat(rows, ignore_index=True)
        for family, rows in family_oof_rows.items()
        if rows
    }
    if not available_oof:
        raise ValueError("no forward probability folds were produced")
    comparison_rows = []
    for family, predictions in available_oof.items():
        row = probability_metrics(predictions).iloc[0].to_dict()
        row["family"] = family
        comparison_rows.append(row)
    family_rank = {
        "empirical": 0,
        "shared_slope_ordinal_logistic": 1,
        "ordinal_logistic": 2,
        "ordinal_lightgbm": 3,
        "lightgbm_multiclass": 4,
    }
    comparison = pd.DataFrame(comparison_rows)
    comparison["family_rank"] = comparison["family"].map(family_rank).fillna(99)
    winning_family = (
        forced_family
        if forced_family is not None
        else str(_simplest_effective_tie(comparison, "family_rank")["family"])
    )
    oof = available_oof[winning_family]
    metrics = probability_metrics(oof)
    empirical_metrics = (
        probability_metrics(available_oof["empirical"])
        if "empirical" in available_oof
        else pd.DataFrame()
    )

    final_split = development["contract_date"].max() - pd.Timedelta(days=calibration_days - 1)
    final_inner_train = development.loc[development["contract_date"].lt(final_split)].copy()
    final_inner_valid = development.loc[development["contract_date"].ge(final_split)].copy()
    selected = _select_inner_candidate(
        final_inner_train,
        final_inner_valid,
        feature_names,
        [spec for spec in specs if spec.family == winning_family],
        blend_weights=resolved_blend_weights,
        random_state=random_state,
    )
    fitted = _fit_candidate(development, feature_names, selected["spec"], random_state=random_state)
    tail_policy = fit_tail_policy(development["exact_offset"].astype(int))
    thresholds, policy_rows = tune_decision_policy(oof, target_coverage=0.60)
    selected_policy = policy_rows.loc[
        policy_rows["minimum_top_probability"].eq(thresholds["minimum_top_probability"])
        & policy_rows["minimum_top_two_margin"].eq(thresholds["minimum_top_two_margin"])
        & policy_rows["minimum_switch_advantage"].eq(thresholds["minimum_switch_advantage"])
    ].iloc[0]
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "station_id": station_id,
        "model_version": model_version or f"{station_id.lower()}_rounded_degree_probability_v1",
        "point_model_version": point_model_version,
        "point_bundle_sha256": point_bundle_sha256.lower(),
        "feature_profile": resolved_profile,
        "base_methods": list(probability_base_methods(resolved_profile)),
        "feature_names": feature_names,
        "mandatory_source_features": list(
            probability_mandatory_feature_names(resolved_profile)
        ),
        "offset_labels": list(OFFSET_LABELS),
        "selected_family": selected["spec"].family,
        "family_selection_mode": "forced" if forced_family is not None else "automatic",
        "forced_family": forced_family,
        "selected_params": dict(selected["spec"].params),
        "model_state": fitted,
        "temperature": float(selected["temperature"]),
        "blend_weight": float(selected["blend_weight"]),
        "blend_weight_candidates": list(resolved_blend_weights),
        "empirical_prior_strength": float(selected["prior_strength"]),
        "empirical_state": fit_empirical_state(development),
        "tail_policy": tail_policy,
        "decision_thresholds": thresholds,
        "forward_policy_metrics": selected_policy.to_dict(),
        "training_start": development["contract_date"].min().date().isoformat(),
        "training_cutoff": development["contract_date"].max().date().isoformat(),
        "training_rows": int(len(development)),
        "development_years": list(resolved_development_years),
        "forward_validation_years": list(resolved_validation_years),
        "forward_metrics": metrics.iloc[0].to_dict(),
        "empirical_forward_metrics": (
            empirical_metrics.iloc[0].to_dict() if not empirical_metrics.empty else {}
        ),
        "candidate_comparison": comparison.drop(columns=["family_rank"]).to_dict(orient="records"),
        "package_versions": probability_package_versions(),
    }
    tuning = pd.DataFrame(tuning_rows)
    tuning = pd.concat(
        [tuning, comparison.drop(columns=["family_rank"]).assign(validation_year="family_comparison")],
        ignore_index=True,
        sort=False,
    )
    tuning = pd.concat([tuning, policy_rows.assign(validation_year="policy")], ignore_index=True, sort=False)
    return bundle, oof, tuning


def _select_inner_candidate(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_names: Sequence[str],
    specs: Sequence[CandidateSpec],
    *,
    blend_weights: Sequence[float],
    random_state: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        if spec.family == "empirical":
            prior_strength = float(spec.params["prior_strength"])
            raw = empirical_probabilities(train, valid, prior_strength)
            candidates = [(1.0, 0.0, raw)]
        else:
            fitted = _fit_candidate(train, feature_names, spec, random_state=random_state)
            uncalibrated = _predict_candidate(fitted, valid, feature_names)
            candidates = []
            for temperature in _temperature_grid():
                calibrated = temperature_scale(uncalibrated, temperature)
                for prior_strength in (15.0, 30.0, 60.0):
                    empirical = empirical_probabilities(train, valid, prior_strength)
                    for blend_weight in blend_weights:
                        candidates.append(
                            (temperature, blend_weight, blend_probabilities(empirical, calibrated, blend_weight), prior_strength)
                        )
        for item in candidates:
            if spec.family == "empirical":
                temperature, blend_weight, probs = item
                prior_strength = float(spec.params["prior_strength"])
            else:
                temperature, blend_weight, probs, prior_strength = item
            score = score_probabilities(valid["offset_class"].to_numpy(dtype=int), probs)
            rows.append(
                {
                    "spec": spec,
                    "temperature": float(temperature),
                    "blend_weight": float(blend_weight),
                    "prior_strength": float(prior_strength),
                    **score,
                }
            )
    if not rows:
        raise ValueError("no probability candidate could be evaluated")
    family_rank = {
        "empirical": 0,
        "shared_slope_ordinal_logistic": 1,
        "ordinal_logistic": 2,
        "ordinal_lightgbm": 3,
        "lightgbm_multiclass": 4,
    }
    best_log_loss = min(float(row["log_loss"]) for row in rows)
    near_log_loss = [
        row
        for row in rows
        if float(row["log_loss"]) <= best_log_loss + EFFECTIVE_TIE_TOLERANCE
    ]
    best_brier = min(float(row["brier"]) for row in near_log_loss)
    effective_ties = [
        row
        for row in near_log_loss
        if float(row["brier"]) <= best_brier + EFFECTIVE_TIE_TOLERANCE
    ]
    return min(
        effective_ties,
        key=lambda row: (
            family_rank[row["spec"].family],
            _candidate_complexity(row["spec"]),
        ),
    )


def _simplest_effective_tie(frame: pd.DataFrame, simplicity_column: str) -> pd.Series:
    best_log_loss = float(frame["log_loss"].min())
    near_log_loss = frame.loc[
        frame["log_loss"].le(best_log_loss + EFFECTIVE_TIE_TOLERANCE)
    ]
    best_brier = float(near_log_loss["brier"].min())
    effective_ties = near_log_loss.loc[
        near_log_loss["brier"].le(best_brier + EFFECTIVE_TIE_TOLERANCE)
    ]
    return effective_ties.sort_values(simplicity_column).iloc[0]


def _candidate_complexity(spec: CandidateSpec) -> tuple[Any, ...]:
    if spec.family == "empirical":
        return (0,)
    if spec.family in {
        "ordinal_logistic",
        "shared_slope_ordinal_logistic",
    }:
        return (
            1 if spec.params.get("class_weight") == "balanced" else 0,
            float(spec.params.get("C", 1.0)),
        )
    if spec.family == "ordinal_lightgbm":
        return (
            int(spec.params.get("n_estimators", 0)),
            int(spec.params.get("num_leaves", 0)),
            -int(spec.params.get("min_child_samples", 0)),
            -float(spec.params.get("reg_lambda", 0.0)),
        )
    return (
        int(spec.params.get("n_estimators", 0)),
        int(spec.params.get("num_leaves", 0)),
        -int(spec.params.get("min_child_samples", 0)),
        -float(spec.params.get("reg_lambda", 0.0)),
    )


def _fit_candidate(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    spec: CandidateSpec,
    *,
    random_state: int,
) -> dict[str, Any]:
    if spec.family == "empirical":
        return {"family": "empirical"}
    x = frame.reindex(columns=feature_names)
    y = frame["offset_class"].to_numpy(dtype=int)
    if spec.family == "ordinal_logistic":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        models: list[Any] = []
        for threshold in range(len(OFFSET_LABELS) - 1):
            binary = (y > threshold).astype(int)
            if np.unique(binary).size < 2:
                models.append(float(binary[0]))
                continue
            pipeline = Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median",
                            add_indicator=False,
                            keep_empty_features=True,
                        ),
                    ),
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            C=float(spec.params["C"]),
                            class_weight=spec.params.get("class_weight"),
                            solver="lbfgs",
                            max_iter=2_000,
                            random_state=random_state,
                        ),
                    ),
                ]
            )
            pipeline.fit(x, binary)
            models.append(pipeline)
        return {"family": spec.family, "threshold_models": models}
    if spec.family == "shared_slope_ordinal_logistic":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        imputer = SimpleImputer(
            strategy="median",
            add_indicator=False,
            keep_empty_features=True,
        )
        scaler = StandardScaler()
        transformed = scaler.fit_transform(imputer.fit_transform(x))
        threshold_count = len(OFFSET_LABELS) - 1
        expanded_features = np.repeat(transformed, threshold_count, axis=0)
        threshold_ids = np.tile(np.arange(threshold_count), len(frame))
        threshold_features = np.eye(threshold_count, dtype=float)[threshold_ids]
        design = np.column_stack([expanded_features, threshold_features])
        binary = np.concatenate(
            [(y > threshold).astype(int) for threshold in range(threshold_count)]
        ).reshape(threshold_count, len(frame)).T.reshape(-1)
        classifier = LogisticRegression(
            C=float(spec.params["C"]),
            class_weight=spec.params.get("class_weight"),
            solver="lbfgs",
            fit_intercept=False,
            max_iter=2_000,
            random_state=random_state,
        )
        classifier.fit(design, binary)
        return {
            "family": spec.family,
            "imputer": imputer,
            "scaler": scaler,
            "classifier": classifier,
            "threshold_count": threshold_count,
        }
    if spec.family == "ordinal_lightgbm":
        from lightgbm import LGBMClassifier
        from sklearn.impute import SimpleImputer

        imputer = SimpleImputer(
            strategy="median",
            add_indicator=False,
            keep_empty_features=True,
        )
        transformed = imputer.fit_transform(x)
        models: list[Any] = []
        for threshold in range(len(OFFSET_LABELS) - 1):
            binary = (y > threshold).astype(int)
            if np.unique(binary).size < 2:
                models.append(float(binary[0]))
                continue
            model = LGBMClassifier(
                objective="binary",
                random_state=random_state,
                verbosity=-1,
                n_jobs=1,
                **dict(spec.params),
            )
            model.fit(transformed, binary)
            models.append(model)
        return {
            "family": spec.family,
            "imputer": imputer,
            "threshold_models": models,
        }
    if spec.family == "lightgbm_multiclass":
        from lightgbm import LGBMClassifier
        from sklearn.impute import SimpleImputer

        imputer = SimpleImputer(
            strategy="median",
            add_indicator=False,
            keep_empty_features=True,
        )
        transformed = imputer.fit_transform(x)
        model = LGBMClassifier(
            objective="multiclass",
            num_class=len(OFFSET_LABELS),
            random_state=random_state,
            verbosity=-1,
            n_jobs=1,
            **dict(spec.params),
        )
        model.fit(transformed, y)
        return {"family": spec.family, "imputer": imputer, "classifier": model}
    raise ValueError(f"unsupported probability family: {spec.family}")


def _predict_candidate(state: Mapping[str, Any], frame: pd.DataFrame, feature_names: Sequence[str]) -> np.ndarray:
    family = state["family"]
    if family == "empirical":
        raise ValueError("empirical predictions require historical state")
    x = frame.reindex(columns=feature_names)
    if family == "ordinal_logistic":
        exceed = []
        for model in state["threshold_models"]:
            if isinstance(model, float):
                exceed.append(np.full(len(frame), model, dtype=float))
            else:
                exceed.append(model.predict_proba(x)[:, 1])
        q = np.column_stack(exceed)
        q = np.minimum.accumulate(q, axis=1)
        probabilities = np.zeros((len(frame), len(OFFSET_LABELS)), dtype=float)
        probabilities[:, 0] = 1.0 - q[:, 0]
        for index in range(1, len(OFFSET_LABELS) - 1):
            probabilities[:, index] = q[:, index - 1] - q[:, index]
        probabilities[:, -1] = q[:, -1]
        return normalize_probabilities(probabilities)
    if family == "shared_slope_ordinal_logistic":
        transformed = state["scaler"].transform(
            state["imputer"].transform(x)
        )
        threshold_count = int(state["threshold_count"])
        exceed = []
        for threshold in range(threshold_count):
            threshold_features = np.zeros(
                (len(frame), threshold_count), dtype=float
            )
            threshold_features[:, threshold] = 1.0
            design = np.column_stack([transformed, threshold_features])
            exceed.append(state["classifier"].predict_proba(design)[:, 1])
        q = np.minimum.accumulate(np.column_stack(exceed), axis=1)
        probabilities = np.zeros(
            (len(frame), len(OFFSET_LABELS)), dtype=float
        )
        probabilities[:, 0] = 1.0 - q[:, 0]
        for index in range(1, len(OFFSET_LABELS) - 1):
            probabilities[:, index] = q[:, index - 1] - q[:, index]
        probabilities[:, -1] = q[:, -1]
        return normalize_probabilities(probabilities)
    if family == "ordinal_lightgbm":
        transformed = state["imputer"].transform(x)
        exceed = []
        for model in state["threshold_models"]:
            if isinstance(model, float):
                exceed.append(np.full(len(frame), model, dtype=float))
            else:
                exceed.append(model.predict_proba(transformed)[:, 1])
        q = np.minimum.accumulate(np.column_stack(exceed), axis=1)
        probabilities = np.zeros(
            (len(frame), len(OFFSET_LABELS)), dtype=float
        )
        probabilities[:, 0] = 1.0 - q[:, 0]
        for index in range(1, len(OFFSET_LABELS) - 1):
            probabilities[:, index] = q[:, index - 1] - q[:, index]
        probabilities[:, -1] = q[:, -1]
        return normalize_probabilities(probabilities)
    if family == "lightgbm_multiclass":
        transformed = state["imputer"].transform(x)
        raw = state["classifier"].predict_proba(transformed)
        probabilities = np.zeros((len(frame), len(OFFSET_LABELS)), dtype=float)
        for source_index, class_index in enumerate(state["classifier"].classes_.astype(int)):
            probabilities[:, class_index] = raw[:, source_index]
        return normalize_probabilities(probabilities)
    raise ValueError(f"unsupported probability family: {family}")


def fit_empirical_state(frame: pd.DataFrame) -> dict[str, Any]:
    global_counts = np.bincount(frame["offset_class"].to_numpy(dtype=int), minlength=len(OFFSET_LABELS))
    month_counts = {
        str(month): np.bincount(group["offset_class"].to_numpy(dtype=int), minlength=len(OFFSET_LABELS)).tolist()
        for month, group in frame.groupby("month")
    }
    return {"global_counts": global_counts.tolist(), "month_counts": month_counts, "alpha": 0.5}


def empirical_probabilities(history: pd.DataFrame, rows: pd.DataFrame, prior_strength: float) -> np.ndarray:
    state = fit_empirical_state(history)
    return empirical_probabilities_from_state(state, rows["month"].to_numpy(dtype=int), prior_strength)


def empirical_probabilities_from_state(
    state: Mapping[str, Any], months: Iterable[int], prior_strength: float
) -> np.ndarray:
    alpha = float(state.get("alpha", 0.5))
    global_counts = np.asarray(state["global_counts"], dtype=float)
    global_probs = (global_counts + alpha) / (global_counts.sum() + alpha * len(OFFSET_LABELS))
    outputs = []
    for month in months:
        counts = np.asarray(state.get("month_counts", {}).get(str(int(month)), np.zeros(len(OFFSET_LABELS))), dtype=float)
        month_probs = (counts + alpha) / (counts.sum() + alpha * len(OFFSET_LABELS))
        weight = counts.sum() / (counts.sum() + prior_strength) if counts.sum() else 0.0
        outputs.append((1.0 - weight) * global_probs + weight * month_probs)
    return normalize_probabilities(np.asarray(outputs))


def temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("temperature must be positive and finite")
    logits = np.log(np.clip(probabilities, 1e-12, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return normalize_probabilities(exp)


def blend_probabilities(empirical: np.ndarray, model: np.ndarray, model_weight: float) -> np.ndarray:
    return normalize_probabilities((1.0 - model_weight) * empirical + model_weight * model)


def normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    values[~np.isfinite(values)] = 0.0
    values = np.clip(values, 0.0, None)
    totals = values.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("probabilities have zero mass")
    return values / totals


def _temperature_grid() -> tuple[float, ...]:
    return tuple(float(value) for value in np.linspace(0.5, 3.0, 11))


def score_probabilities(actual_classes: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    positions = np.arange(len(actual_classes))
    actual_probability = probabilities[positions, actual_classes]
    one_hot = np.eye(len(OFFSET_LABELS))[actual_classes]
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == actual_classes
    predicted_cdf = np.cumsum(probabilities, axis=1)[:, :-1]
    observed_cdf = np.cumsum(one_hot, axis=1)[:, :-1]
    calibration_error = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        selected = (confidence >= lower) & (confidence < lower + 0.1)
        if selected.any():
            calibration_error += float(selected.mean()) * abs(float(correct[selected].mean()) - float(confidence[selected].mean()))
    return {
        "log_loss": float(-np.log(np.clip(actual_probability, 1e-12, 1.0)).mean()),
        "brier": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "ranked_probability_score": float(
            np.square(predicted_cdf - observed_cdf).sum(axis=1).mean()
            / (len(OFFSET_LABELS) - 1)
        ),
        "offset_accuracy": float((probabilities.argmax(axis=1) == actual_classes).mean()),
        "top_two_accuracy": float(
            np.mean([actual in row for actual, row in zip(actual_classes, np.argsort(probabilities, axis=1)[:, -2:], strict=False)])
        ),
        "calibration_error": calibration_error,
    }


def _prediction_rows(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    validation_year: int,
    selected: Mapping[str, Any],
) -> pd.DataFrame:
    rows = frame[
        [
            "contract_date",
            "actual_high_f",
            "actual_degree_f",
            "point_prediction_f",
            "point_degree_f",
            "observed_high_temp_through_as_of_f",
            "exact_offset",
            "offset_class",
            "month",
        ]
    ].copy()
    rows["validation_year"] = validation_year
    rows["selected_family"] = selected["spec"].family
    rows["offset_probabilities"] = [row.tolist() for row in probabilities]
    return rows


def probability_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    probabilities = np.vstack(predictions["offset_probabilities"].to_numpy())
    actual = predictions["offset_class"].to_numpy(dtype=int)
    score = score_probabilities(actual, probabilities)
    score["count"] = int(len(predictions))
    return pd.DataFrame([score])


def fit_tail_policy(exact_offsets: Iterable[int], *, alpha: float = 0.5) -> dict[str, Any]:
    counts = Counter(int(value) for value in exact_offsets)
    low_min = min([value for value in counts if value <= -4], default=-4) - 1
    high_max = max([value for value in counts if value >= 4], default=4) + 1
    low_support = list(range(low_min, -3))
    high_support = list(range(4, high_max + 1))

    def weights(support: list[int]) -> dict[str, float]:
        raw = {value: counts.get(value, 0) + alpha for value in support}
        total = sum(raw.values())
        return {str(value): float(weight / total) for value, weight in raw.items()}

    return {
        "alpha": alpha,
        "low_exact_offset_weights": weights(low_support),
        "high_exact_offset_weights": weights(high_support),
    }


def expand_offset_probabilities(
    point_degree_f: int,
    offset_probabilities: Sequence[float],
    tail_policy: Mapping[str, Any],
    *,
    observed_high_f: float | None = None,
) -> dict[int, float]:
    probabilities = np.asarray(offset_probabilities, dtype=float)
    if probabilities.shape != (len(OFFSET_LABELS),):
        raise ValueError("expected nine offset probabilities")
    degree_probabilities: dict[int, float] = {}
    for class_index, exact_offset in enumerate(CENTRAL_OFFSETS, start=1):
        degree_probabilities[point_degree_f + exact_offset] = float(probabilities[class_index])
    for class_index, policy_key in ((0, "low_exact_offset_weights"), (8, "high_exact_offset_weights")):
        for raw_offset, weight in tail_policy[policy_key].items():
            degree = point_degree_f + int(raw_offset)
            degree_probabilities[degree] = degree_probabilities.get(degree, 0.0) + float(probabilities[class_index]) * float(weight)
    if observed_high_f is not None:
        minimum_degree = round_half_up(observed_high_f)
        degree_probabilities = {degree: probability for degree, probability in degree_probabilities.items() if degree >= minimum_degree}
    total = sum(degree_probabilities.values())
    if total <= 0:
        raise ValueError("physical floor removed all probability mass")
    return {degree: probability / total for degree, probability in sorted(degree_probabilities.items())}


def degree_to_bucket_probabilities(degree_probabilities: Mapping[int, float]) -> dict[str, float]:
    buckets: dict[str, float] = {}
    for degree, probability in degree_probabilities.items():
        label = canonical_two_degree_bucket(int(degree))
        buckets[label] = buckets.get(label, 0.0) + float(probability)
    return dict(sorted(buckets.items(), key=lambda item: int(item[0].split("-", 1)[0])))


def tune_decision_policy(
    predictions: pd.DataFrame, *, target_coverage: float
) -> tuple[dict[str, float], pd.DataFrame]:
    evaluated = []
    tail_policy = fit_tail_policy(predictions["exact_offset"].astype(int))
    for _, row in predictions.iterrows():
        probabilities = expand_offset_probabilities(
            int(row["point_degree_f"]),
            row["offset_probabilities"],
            tail_policy,
            observed_high_f=float(row["observed_high_temp_through_as_of_f"]),
        )
        buckets = degree_to_bucket_probabilities(probabilities)
        ranked = sorted(buckets.items(), key=lambda item: (-item[1], item[0]))
        point_bucket = canonical_two_degree_bucket(int(row["point_degree_f"]))
        actual_bucket = canonical_two_degree_bucket(int(row["actual_degree_f"]))
        evaluated.append(
            {
                "top_bucket": ranked[0][0],
                "top_probability": ranked[0][1],
                "margin": ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0.0),
                "advantage": ranked[0][1] - buckets.get(point_bucket, 0.0),
                "point_bucket": point_bucket,
                "actual_bucket": actual_bucket,
                "tail_ambiguous": tail_allocation_is_ambiguous(
                    int(row["point_degree_f"]),
                    row["offset_probabilities"],
                    tail_policy,
                    ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0.0),
                    observed_high_f=float(row["observed_high_temp_through_as_of_f"]),
                ),
            }
        )
    scored = pd.DataFrame(evaluated)
    candidates = []
    for minimum_top in np.arange(0.20, 0.701, 0.025):
        for minimum_margin in np.arange(0.0, 0.301, 0.025):
            for minimum_advantage in np.arange(0.0, 0.201, 0.025):
                confident = scored["top_probability"].ge(minimum_top) & scored["margin"].ge(minimum_margin)
                switch = scored["top_bucket"].ne(scored["point_bucket"])
                actionable = confident & (~switch | scored["advantage"].ge(minimum_advantage))
                actionable &= ~scored["tail_ambiguous"]
                coverage = float(actionable.mean())
                if not actionable.any():
                    continue
                accuracy = float(scored.loc[actionable, "top_bucket"].eq(scored.loc[actionable, "actual_bucket"]).mean())
                switch_count = int((actionable & switch).sum())
                point_accuracy = float(
                    scored.loc[actionable, "point_bucket"].eq(
                        scored.loc[actionable, "actual_bucket"]
                    ).mean()
                )
                switch_accuracy = (
                    float(scored.loc[actionable & switch, "top_bucket"].eq(scored.loc[actionable & switch, "actual_bucket"]).mean())
                    if switch_count
                    else np.nan
                )
                candidates.append(
                    {
                        "minimum_top_probability": float(minimum_top),
                        "minimum_top_two_margin": float(minimum_margin),
                        "minimum_switch_advantage": float(minimum_advantage),
                        "coverage": coverage,
                        "accuracy": accuracy,
                        "point_accuracy_on_actionable": point_accuracy,
                        "full_point_accuracy": float(
                            scored["point_bucket"].eq(scored["actual_bucket"]).mean()
                        ),
                        "switch_count": switch_count,
                        "switch_accuracy": switch_accuracy,
                    }
                )
    if not candidates:
        raise ValueError("unable to tune an actionable probability policy")
    candidate_frame = pd.DataFrame(candidates)
    preferred = candidate_frame.loc[candidate_frame["coverage"].between(0.55, 0.65)].copy()
    if preferred.empty:
        preferred = candidate_frame.copy()
        preferred["coverage_distance"] = (preferred["coverage"] - target_coverage).abs()
        policy = preferred.sort_values(
            ["coverage_distance", "accuracy", "switch_count"], ascending=[True, False, False]
        ).iloc[0]
    else:
        preferred["coverage_distance"] = (preferred["coverage"] - target_coverage).abs()
        policy = preferred.sort_values(
            ["accuracy", "coverage_distance", "switch_count"], ascending=[False, True, False]
        ).iloc[0]
    thresholds = {
        "minimum_top_probability": float(policy["minimum_top_probability"]),
        "minimum_top_two_margin": float(policy["minimum_top_two_margin"]),
        "minimum_switch_advantage": float(policy["minimum_switch_advantage"]),
        "target_coverage": target_coverage,
    }
    return thresholds, candidate_frame


def predict_probability_bundle(bundle: Mapping[str, Any], feature_values: Mapping[str, Any]) -> dict[str, Any]:
    missing = [
        name
        for name in bundle["mandatory_source_features"]
        if _finite_number(feature_values.get(name)) is None
    ]
    feature_profile = str(bundle.get("feature_profile", FEATURE_PROFILE_COMMON_NO_PEAK))
    base_methods = tuple(bundle.get("base_methods") or probability_base_methods(feature_profile))
    for name in ("point_prediction_f", *[f"{method}_predicted_high_f" for method in base_methods]):
        if _finite_number(feature_values.get(name)) is None:
            missing.append(name)
    if missing:
        return {"status": "unavailable", "reason": "missing_required_features:" + ",".join(sorted(set(missing)))}
    frame = add_probability_features(
        pd.DataFrame([dict(feature_values)]),
        providers=probability_provider_names(
            feature_profile
        ),
        base_methods=base_methods,
    )
    for name in bundle["feature_names"]:
        if name.endswith(MISSING_INDICATOR_SUFFIX):
            source_name = name[: -len(MISSING_INDICATOR_SUFFIX)]
            source = (
                frame[source_name]
                if source_name in frame
                else pd.Series([np.nan] * len(frame), index=frame.index)
            )
            frame[name] = source.map(
                lambda value: 1.0 if _finite_number(value) is None else 0.0
            )
        if name not in frame:
            frame[name] = np.nan
    point = float(feature_values["point_prediction_f"])
    month = int(feature_values.get("month") or pd.Timestamp(feature_values["contract_date"]).month)
    empirical = empirical_probabilities_from_state(
        bundle["empirical_state"], [month], float(bundle["empirical_prior_strength"])
    )
    if bundle["selected_family"] == "empirical":
        probabilities = empirical
    else:
        raw = _predict_candidate(bundle["model_state"], frame, bundle["feature_names"])
        calibrated = temperature_scale(raw, float(bundle["temperature"]))
        probabilities = blend_probabilities(empirical, calibrated, float(bundle["blend_weight"]))
    point_degree = round_half_up(point)
    degree_probs = expand_offset_probabilities(
        point_degree,
        probabilities[0],
        bundle["tail_policy"],
        observed_high_f=float(feature_values["observed_high_temp_through_as_of_f"]),
    )
    bucket_probs = degree_to_bucket_probabilities(degree_probs)
    ranked = sorted(bucket_probs.items(), key=lambda item: (-item[1], item[0]))
    point_bucket = canonical_two_degree_bucket(point_degree)
    top_bucket, top_probability = ranked[0]
    second_probability = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = top_probability - second_probability
    advantage = top_probability - bucket_probs.get(point_bucket, 0.0)
    thresholds = bundle["decision_thresholds"]
    overrides_enabled = bool(bundle.get("overrides_enabled", True))
    reason = "confidence_passed"
    decision = "shadow_trade"
    if tail_allocation_is_ambiguous(
        point_degree,
        probabilities[0],
        bundle["tail_policy"],
        margin,
        observed_high_f=float(feature_values["observed_high_temp_through_as_of_f"]),
    ):
        decision, reason = "no_trade", "tail_allocation_ambiguous"
    elif top_probability < float(thresholds["minimum_top_probability"]):
        decision, reason = "no_trade", "top_probability_below_threshold"
    elif margin < float(thresholds["minimum_top_two_margin"]):
        decision, reason = "no_trade", "top_two_margin_below_threshold"
    elif (
        overrides_enabled
        and top_bucket != point_bucket
        and advantage
        < float(thresholds.get("minimum_switch_advantage", 0.0))
    ):
        decision, reason = "no_trade", "switch_advantage_below_threshold"
    recommended_bucket = top_bucket if overrides_enabled else point_bucket
    recommended_probability = float(
        bucket_probs.get(recommended_bucket, 0.0)
    )
    return {
        "status": "ok",
        "model_version": bundle["model_version"],
        "rounded_point_high_f": point_degree,
        "offset_probabilities": {label: float(value) for label, value in zip(OFFSET_LABELS, probabilities[0], strict=True)},
        "degree_probabilities": {str(key): float(value) for key, value in degree_probs.items()},
        "bucket_probabilities": bucket_probs,
        "point_bucket_label": point_bucket,
        "probability_top_bucket_label": top_bucket,
        "probability_top_bucket_probability": float(top_probability),
        "recommended_bucket_label": recommended_bucket,
        "recommended_bucket_probability": recommended_probability,
        "second_bucket_probability": float(second_probability),
        "probability_advantage_over_point_bucket": float(advantage),
        "probability_thresholds": {
            "minimumTopProbability": float(thresholds["minimum_top_probability"]),
            "minimumTopTwoMargin": float(thresholds["minimum_top_two_margin"]),
            "minimumSwitchAdvantage": float(
                thresholds.get("minimum_switch_advantage", 0.0)
            ),
        },
        "probability_decision": decision,
        "probability_decision_reason": reason,
        "overrides_enabled": overrides_enabled,
        "overrides_point_bucket": recommended_bucket != point_bucket,
    }


def tail_allocation_is_ambiguous(
    point_degree_f: int,
    offset_probabilities: Sequence[float],
    tail_policy: Mapping[str, Any],
    top_two_margin: float,
    *,
    observed_high_f: float,
) -> bool:
    minimum_degree = round_half_up(observed_high_f)
    for class_index, key in ((0, "low_exact_offset_weights"), (8, "high_exact_offset_weights")):
        mass = float(offset_probabilities[class_index])
        buckets = {
            canonical_two_degree_bucket(point_degree_f + int(offset))
            for offset in tail_policy[key]
            if point_degree_f + int(offset) >= minimum_degree
        }
        if len(buckets) > 1 and mass + 1e-12 >= top_two_margin:
            return True
    return False


def evaluate_probability_holdout(
    feature_frame: pd.DataFrame,
    point_predictions: pd.DataFrame,
    base_predictions: pd.DataFrame,
    bundle: Mapping[str, Any],
    *,
    holdout_year: int = 2026,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score a frozen probability bundle on a later point-model holdout."""
    frame = build_probability_frame(
        feature_frame,
        point_predictions,
        base_predictions,
        include_peak_features=bundle["feature_profile"]
        == FEATURE_PROFILE_PEAK_AUGMENTED,
        feature_profile=str(bundle["feature_profile"]),
    )
    frame = frame.loc[frame["year"].eq(int(holdout_year))].copy()
    rows: list[dict[str, Any]] = []
    probability_vectors: list[np.ndarray] = []
    actual_classes: list[int] = []
    for _, row in frame.iterrows():
        values = row.to_dict()
        values["contract_date"] = (
            pd.Timestamp(row["contract_date"]).date().isoformat()
        )
        result = predict_probability_bundle(bundle, values)
        if result["status"] != "ok":
            continue
        actual_degree = int(row["actual_degree_f"])
        point_degree = int(row["point_degree_f"])
        actual_bucket = canonical_two_degree_bucket(actual_degree)
        point_bucket = canonical_two_degree_bucket(point_degree)
        actual_offset = offset_label(actual_degree - point_degree)
        vector = np.asarray(
            [
                float(result["offset_probabilities"][label])
                for label in OFFSET_LABELS
            ],
            dtype=float,
        )
        actual_class = OFFSET_LABELS.index(actual_offset)
        probability_vectors.append(vector)
        actual_classes.append(actual_class)
        rows.append(
            {
                "contract_date": values["contract_date"],
                "actual_high_f": float(row["actual_high_f"]),
                "actual_degree_f": actual_degree,
                "actual_bucket": actual_bucket,
                "point_prediction_f": float(row["point_prediction_f"]),
                "point_degree_f": point_degree,
                "point_bucket": point_bucket,
                "point_hit": point_bucket == actual_bucket,
                "recommended_bucket": result["recommended_bucket_label"],
                "recommended_bucket_probability": float(
                    result["recommended_bucket_probability"]
                ),
                "recommended_hit": result["recommended_bucket_label"]
                == actual_bucket,
                "actual_bucket_probability": float(
                    result["bucket_probabilities"].get(actual_bucket, 0.0)
                ),
                "actual_offset_label": actual_offset,
                "actual_offset_probability": float(
                    result["offset_probabilities"].get(actual_offset, 0.0)
                ),
                "probability_decision": result["probability_decision"],
                "probability_decision_reason": result[
                    "probability_decision_reason"
                ],
                "overrides_point_bucket": bool(result["overrides_point_bucket"]),
                "offset_probabilities": result["offset_probabilities"],
                "degree_probabilities": result["degree_probabilities"],
                "bucket_probabilities": result["bucket_probabilities"],
            }
        )
    predictions = pd.DataFrame(rows)
    if predictions.empty:
        return predictions, pd.DataFrame()
    scores = score_probabilities(
        np.asarray(actual_classes, dtype=int),
        np.vstack(probability_vectors),
    )
    metrics = pd.DataFrame(
        [
            {
                "holdout_year": int(holdout_year),
                "count": int(len(predictions)),
                "offset_log_loss": float(scores["log_loss"]),
                "multiclass_brier": float(scores["brier"]),
                "ranked_probability_score": float(
                    scores["ranked_probability_score"]
                ),
                "offset_accuracy": float(scores["offset_accuracy"]),
                "offset_top_two_accuracy": float(scores["top_two_accuracy"]),
                "bucket_log_loss": float(
                    -np.log(
                        predictions["actual_bucket_probability"].clip(lower=1e-12)
                    ).mean()
                ),
                "point_bucket_accuracy": float(predictions["point_hit"].mean()),
                "probability_bucket_accuracy": float(
                    predictions["recommended_hit"].mean()
                ),
                "switch_count": int(
                    predictions["overrides_point_bucket"].sum()
                ),
            }
        ]
    )
    return predictions, metrics


def export_probability_bundle(
    bundle: Mapping[str, Any], output_dir: Path | str, *, source_identity: Mapping[str, Any] | None = None
) -> tuple[Path, Path]:
    import joblib

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = f"{bundle['station_id']}_{bundle['model_version']}"
    bundle_path = output / f"{stem}.joblib"
    manifest_path = output / f"{stem}.json"
    joblib.dump(dict(bundle), bundle_path)
    bundle_hash = sha256_file(bundle_path)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "station_id": bundle["station_id"],
        "model_version": bundle["model_version"],
        "point_model_version": bundle["point_model_version"],
        "point_bundle_sha256": bundle["point_bundle_sha256"],
        "feature_profile": bundle["feature_profile"],
        "base_methods": bundle.get(
            "base_methods", probability_base_methods(str(bundle["feature_profile"]))
        ),
        "feature_names": bundle["feature_names"],
        "mandatory_source_features": bundle["mandatory_source_features"],
        "offset_labels": bundle["offset_labels"],
        "selected_family": bundle["selected_family"],
        "family_selection_mode": bundle.get("family_selection_mode", "automatic"),
        "forced_family": bundle.get("forced_family"),
        "selected_params": bundle["selected_params"],
        "temperature": bundle["temperature"],
        "blend_weight": bundle["blend_weight"],
        "blend_weight_candidates": bundle.get(
            "blend_weight_candidates", [0.25, 0.5, 0.75, 1.0]
        ),
        "empirical_prior_strength": bundle["empirical_prior_strength"],
        "tail_policy": bundle["tail_policy"],
        "decision_thresholds": bundle["decision_thresholds"],
        "forward_policy_metrics": bundle["forward_policy_metrics"],
        "training_start": bundle["training_start"],
        "training_cutoff": bundle["training_cutoff"],
        "training_rows": bundle["training_rows"],
        "forward_metrics": bundle["forward_metrics"],
        "empirical_forward_metrics": bundle["empirical_forward_metrics"],
        "candidate_comparison": bundle["candidate_comparison"],
        "profile_comparison": bundle.get("profile_comparison", []),
        "package_versions": bundle["package_versions"],
        "holdout_metrics": bundle.get("holdout_metrics", {}),
        "holdout_status": bundle.get("holdout_status", "acceptance"),
        "historical_acceptance": bundle.get(
            "historical_acceptance", {"passed": False, "reasons": ["not_evaluated"]}
        ),
        "source_identity": dict(source_identity or {}),
        "artifact_integrity": {"bundle_sha256": bundle_hash},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return bundle_path, manifest_path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probability_package_versions() -> dict[str, str]:
    return {
        package: importlib.metadata.version(distribution)
        for package, distribution in {
            "pandas": "pandas",
            "numpy": "numpy",
            "scikit-learn": "scikit-learn",
            "joblib": "joblib",
            "lightgbm": "lightgbm",
        }.items()
    }


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
