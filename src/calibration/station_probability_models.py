from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


SCHEMA_VERSION = 2
GAUSSIAN_ARTIFACT_TYPE = "station_gaussian_residual_probability_model"
ORDINAL_ARTIFACT_TYPE = "station_ordinal_residual_probability_model"
ORDINAL_ENSEMBLE_ARTIFACT_TYPE = "station_ordinal_probability_ensemble"
ORDINAL_POLICY_VERSION = "station_ordinal_two_of_three_xgboost_v1"
ORDINAL_MEMBER_ROLES = (
    "blended_ordinal",
    "shared_slope_ordinal",
    "pure_ordinal",
)
MODEL_FEATURES = (
    "point_prediction_native",
    "point_rounding_remainder_native",
    "point_distance_to_round_boundary_native",
    "provider_mean_minus_point_native",
    "provider_spread_native",
    "provider_std_native",
    "observed_temp_minus_point_native",
    "observed_high_minus_point_native",
    "observed_as_of_age_minutes",
    "observed_temp_change_last_1h_f",
    "observed_temp_change_last_3h_f",
    "observed_morning_warmup_rate_f_per_hour",
    "observed_high_so_far_change_since_9am_f",
    "observed_dewpoint_depression_f",
    "observed_humidity_at_as_of",
    "observed_wind_speed_at_as_of",
    "observed_pressure_at_as_of",
    "observed_visibility_at_as_of",
    "observed_ceiling_at_as_of",
    "observed_cloud_cover_at_as_of",
    "observed_precip_recent_at_as_of",
    "observed_is_raining_at_as_of",
    "observed_is_fog_or_mist_at_as_of",
    "observed_is_thunder_at_as_of",
    "day_of_year_sin",
    "day_of_year_cos",
    "month_sin",
    "month_cos",
)
ORDINAL_COMPACT_FEATURES = (
    "point_prediction_native",
    "point_rounding_remainder_native",
    "point_distance_to_round_boundary_native",
    "provider_mean_minus_point_native",
    "provider_spread_native",
    "provider_std_native",
    "observed_temp_minus_point_native",
    "observed_high_minus_point_native",
    "observed_as_of_age_minutes",
    "observed_temp_change_last_1h_f",
    "observed_temp_change_last_3h_f",
    "observed_morning_warmup_rate_f_per_hour",
    "observed_high_so_far_change_since_9am_f",
    "observed_dewpoint_depression_f",
    "observed_humidity_at_as_of",
    "observed_cloud_cover_at_as_of",
    "observed_precip_recent_at_as_of",
    "day_of_year_sin",
    "day_of_year_cos",
    "month_sin",
    "month_cos",
)
GAUSSIAN_ALPHA_GRID = (0.1, 1.0, 10.0, 100.0)
GAUSSIAN_SCALE_GRID = (0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75)
ORDINAL_C_GRID = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
ORDINAL_CLASS_WEIGHTS = (None, "balanced")
ORDINAL_TEMPERATURE_GRID = (0.65, 0.8, 1.0, 1.25, 1.5, 2.0)
ORDINAL_BLEND_WEIGHT_GRID = (0.5, 0.75, 0.9)
ORDINAL_PRIOR_STRENGTH_GRID = (15.0, 30.0, 60.0)


@dataclass(frozen=True)
class ProbabilityRun:
    gaussian_state: dict[str, Any]
    ordinal_states: dict[str, dict[str, Any]]
    forward_predictions: pd.DataFrame
    forward_metrics: pd.DataFrame
    tuning: pd.DataFrame


def round_half_up(value: float) -> int:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("cannot round a non-finite market temperature")
    return int(math.floor(numeric + 0.5))


def fahrenheit_to_celsius(value: float) -> float:
    return (float(value) - 32.0) * 5.0 / 9.0


def native_value(value_f: pd.Series, unit: str) -> pd.Series:
    values = pd.to_numeric(value_f, errors="coerce")
    return values if unit == "F" else (values - 32.0) * 5.0 / 9.0


def canonical_market_bucket(degree: int, *, unit: str, bucket_width: int) -> str:
    if unit == "C" and bucket_width == 1:
        return f"{int(degree)}C"
    if unit == "F" and bucket_width == 2:
        lower = int(degree) if int(degree) % 2 == 0 else int(degree) - 1
        return f"{lower}-{lower + 1}F"
    raise ValueError(f"unsupported market bucket contract: {bucket_width}{unit}")


def build_probability_frame(
    features: pd.DataFrame,
    point_predictions: pd.DataFrame,
    *,
    providers: Sequence[str],
    unit: str,
    bucket_width: int,
) -> pd.DataFrame:
    required = {"contract_date", "actual_high_f", "predicted_high_f"}
    missing = sorted(required - set(point_predictions.columns))
    if missing:
        raise ValueError("point predictions missing: " + ",".join(missing))
    unit = str(unit).strip().upper()
    if unit not in {"F", "C"}:
        raise ValueError("probability unit must be F or C")

    point = point_predictions.copy()
    point["contract_date"] = pd.to_datetime(point["contract_date"], errors="coerce")
    invalid_point_dates = int(point["contract_date"].isna().sum())
    incomplete_point_rows = int(point[["actual_high_f", "predicted_high_f"]].isna().any(axis=1).sum())
    if invalid_point_dates or incomplete_point_rows:
        raise ValueError("point predictions contain invalid dates or incomplete target/prediction rows")
    point = point.dropna(subset=["contract_date", "actual_high_f", "predicted_high_f"])
    if point["contract_date"].duplicated().any():
        raise ValueError("point predictions have duplicate contract dates; refusing ambiguous join")
    source = features.copy()
    source["contract_date"] = pd.to_datetime(source["contract_date"], errors="coerce")
    if source["contract_date"].duplicated().any():
        raise ValueError("feature frame has duplicate contract dates")
    out = point.merge(source, on="contract_date", how="left", validate="one_to_one", suffixes=("", "_feature"), indicator=True)
    unmatched = out["_merge"].ne("both")
    if unmatched.any():
        dates = out.loc[unmatched, "contract_date"].dt.date.astype(str).tolist()
        raise ValueError("point predictions have no matching feature row: " + ",".join(dates[:10]))
    out = out.drop(columns="_merge")
    feature_target = "actual_high_f_feature" if "actual_high_f_feature" in out else "actual_high_f"
    if feature_target in out:
        point_target = pd.to_numeric(out["actual_high_f"], errors="coerce")
        joined_target = pd.to_numeric(out[feature_target], errors="coerce")
        mismatch = joined_target.notna() & ~np.isclose(point_target, joined_target, rtol=0.0, atol=1e-8, equal_nan=False)
        if mismatch.any():
            dates = out.loc[mismatch, "contract_date"].dt.date.astype(str).tolist()
            raise ValueError("point prediction target differs from joined feature target: " + ",".join(dates[:10]))

    actual_f = pd.to_numeric(out["actual_high_f"], errors="coerce")
    if unit == "C":
        actual_native = pd.Series(np.nan, index=out.index, dtype=float)
        fahrenheit_source = _normalized_source_series(out, ("actual_source_feature", "actual_source", "settlement_source_feature", "settlement_source"))
        for column, source_columns in (
            (
                "actual_high_c",
                (
                    "actual_high_c_settlement_source",
                    "settlement_high_c_source",
                    "settlement_source_feature",
                    "settlement_source",
                ),
            ),
            (
                "settlement_high_c",
                (
                    "settlement_high_c_source",
                    "settlement_source_feature",
                    "settlement_source",
                ),
            ),
        ):
            if column in out:
                candidate = pd.to_numeric(out[column], errors="coerce")
                candidate_source = _normalized_source_series(out, source_columns)
                used = actual_native.isna() & candidate.notna()
                if used.any() and (candidate_source.loc[used].eq("").any() or fahrenheit_source.loc[used].eq("").any()):
                    raise ValueError("native Celsius target provenance is missing or ambiguous")
                if used.any() and candidate_source.loc[used].ne(fahrenheit_source.loc[used]).any():
                    raise ValueError("native Celsius target provenance differs from Fahrenheit settlement")
                actual_native = actual_native.fillna(candidate)
        actual_native = actual_native.fillna(native_value(actual_f, unit))
        converted_actual = native_value(actual_f, unit)
        contradictory = actual_native.notna() & converted_actual.notna() & ~np.isclose(
            actual_native, converted_actual, rtol=0.0, atol=0.15, equal_nan=False
        )
        if contradictory.any():
            dates = out.loc[contradictory, "contract_date"].dt.date.astype(str).tolist()
            raise ValueError("native Celsius target contradicts matched Fahrenheit settlement: " + ",".join(dates[:10]))
    else:
        actual_native = actual_f
    point_native = native_value(out["predicted_high_f"], unit)
    out["actual_high_native"] = actual_native
    out["point_prediction_native"] = point_native
    out["point_degree_native"] = point_native.map(round_half_up)
    out["actual_degree_native"] = actual_native.map(round_half_up)
    out["point_rounding_remainder_native"] = point_native - out["point_degree_native"]
    out["point_distance_to_round_boundary_native"] = 0.5 - out["point_rounding_remainder_native"].abs()
    out["residual_native"] = actual_native - point_native
    out["offset_native"] = out["actual_degree_native"] - out["point_degree_native"]
    out["actual_market_bucket"] = out["actual_degree_native"].map(
        lambda value: canonical_market_bucket(int(value), unit=unit, bucket_width=bucket_width)
    )

    provider_native: list[pd.Series] = []
    for provider in providers:
        column = f"{provider}_high_f"
        values = native_value(out[column], unit) if column in out else pd.Series(np.nan, index=out.index)
        out[f"{provider}_high_native"] = values
        out[f"{provider}_minus_point_native"] = values - point_native
        provider_native.append(values.rename(provider))
    provider_frame = pd.concat(provider_native, axis=1)
    out["provider_mean_minus_point_native"] = provider_frame.mean(axis=1) - point_native
    out["provider_spread_native"] = provider_frame.max(axis=1) - provider_frame.min(axis=1)
    out["provider_std_native"] = provider_frame.std(axis=1, ddof=0)

    observed_temp = native_value(out.get("observed_temp_at_as_of_f"), unit)
    observed_high = native_value(out.get("observed_high_temp_through_as_of_f"), unit)
    out["observed_temp_minus_point_native"] = observed_temp - point_native
    out["observed_high_minus_point_native"] = observed_high - point_native
    month = out["contract_date"].dt.month.astype(float)
    out["month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
    out["month_cos"] = np.cos(2.0 * np.pi * month / 12.0)

    feature_names = [*MODEL_FEATURES, *(f"{provider}_minus_point_native" for provider in providers)]
    for name in feature_names:
        if name not in out:
            out[name] = np.nan
        out[name] = pd.to_numeric(out[name], errors="coerce")
    out = out.dropna(
        subset=["actual_high_native", "point_prediction_native", "actual_degree_native", "point_degree_native"]
    ).copy()
    out["year"] = out["contract_date"].dt.year
    out.attrs["feature_names"] = tuple(feature_names)
    out.attrs["unit"] = unit
    out.attrs["bucket_width"] = int(bucket_width)
    out.attrs["row_completeness"] = {"point_input_rows": int(len(point_predictions)), "point_invalid_contract_date_rows": invalid_point_dates, "point_missing_target_or_prediction_rows": incomplete_point_rows, "joined_rows": int(len(out)), "feature_rows": int(len(source)), "unmatched_point_rows": 0}
    out.attrs["feature_missingness_before_imputation"] = _feature_missingness_audit(out, feature_names)
    return out.sort_values("contract_date").reset_index(drop=True)


def _normalized_source_series(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    result = pd.Series("", index=frame.index, dtype="string")
    for column in columns:
        if column not in frame:
            continue
        source = frame[column].astype("string").str.strip().str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")
        result = result.mask(result.eq("") & source.notna(), source)
    return result.fillna("")


def _feature_missingness_audit(frame: pd.DataFrame, feature_names: Sequence[str]) -> list[dict[str, Any]]:
    return [{"feature": name, "available": name in frame, "missing_count_before_imputation": int(pd.to_numeric(frame.get(name), errors="coerce").isna().sum()) if name in frame else int(len(frame)), "missing_fraction_before_imputation": float(pd.to_numeric(frame.get(name), errors="coerce").isna().mean()) if name in frame and len(frame) else 1.0} for name in feature_names]


def _pipeline(alpha: float):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=float(alpha))),
        ]
    )


def fit_gaussian(frame: pd.DataFrame, feature_names: Sequence[str], *, alpha: float, scale_multiplier: float) -> dict[str, Any]:
    if len(frame) < 60:
        raise ValueError("at least 60 rows are required for Gaussian residual fitting")
    x = frame.reindex(columns=feature_names)
    residual = frame["residual_native"].to_numpy(float)
    mean_model = _pipeline(alpha)
    mean_model.fit(x, residual)
    centered = residual - mean_model.predict(x)
    scale_model = _pipeline(alpha)
    scale_model.fit(x, np.log(np.maximum(np.abs(centered), 0.10)))
    return {
        "family": "conditional_gaussian_residual",
        "feature_names": list(feature_names),
        "alpha": float(alpha),
        "scale_multiplier": float(scale_multiplier),
        "mean_model": mean_model,
        "scale_model": scale_model,
        "absolute_residual_to_sigma": math.sqrt(math.pi / 2.0),
        "feature_availability_before_imputation": _feature_missingness_audit(frame, feature_names),
    }


def predict_gaussian(state: Mapping[str, Any], frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = frame.reindex(columns=state["feature_names"])
    mean = np.asarray(state["mean_model"].predict(x), dtype=float)
    sigma = np.exp(np.asarray(state["scale_model"].predict(x), dtype=float))
    sigma *= float(state["absolute_residual_to_sigma"]) * float(state["scale_multiplier"])
    return mean, np.maximum(sigma, 0.15)


def _ordinal_class(offset: int, tail: int) -> int:
    return int(np.clip(int(offset), -tail, tail) + tail)


def fit_ordinal(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    tail: int,
    c: float,
    class_weight: str | None,
    temperature: float,
) -> dict[str, Any]:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    y = frame["offset_native"].astype(int).map(lambda value: _ordinal_class(value, tail)).to_numpy(int)
    x = frame.reindex(columns=feature_names)
    models: list[Any] = []
    for threshold in range(2 * tail):
        binary = (y > threshold).astype(int)
        if np.unique(binary).size < 2:
            models.append(float(binary[0]))
            continue
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=float(c),
                        class_weight=class_weight,
                        solver="lbfgs",
                        max_iter=2_000,
                        random_state=42,
                    ),
                ),
            ]
        )
        model.fit(x, binary)
        models.append(model)
    return {
        "family": "cumulative_ordinal_logistic",
        "candidate_role": "pure_ordinal",
        "feature_profile": "full",
        "feature_names": list(feature_names),
        "tail": int(tail),
        "c": float(c),
        "class_weight": class_weight,
        "temperature": float(temperature),
        "threshold_models": models,
        "tail_offsets": _fit_tail_offsets(frame, tail),
        "feature_availability_before_imputation": _feature_missingness_audit(frame, feature_names),
    }


def fit_shared_slope_ordinal(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    tail: int,
    c: float,
    class_weight: str | None,
    temperature: float,
) -> dict[str, Any]:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if len(frame) < 60:
        raise ValueError("at least 60 rows are required for shared-slope ordinal fitting")
    y = frame["offset_native"].astype(int).map(lambda value: _ordinal_class(value, tail)).to_numpy(int)
    x = frame.reindex(columns=feature_names)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    transformed = scaler.fit_transform(imputer.fit_transform(x))
    threshold_count = 2 * int(tail)
    expanded_features = np.repeat(transformed, threshold_count, axis=0)
    threshold_ids = np.tile(np.arange(threshold_count), len(frame))
    threshold_features = np.eye(threshold_count, dtype=float)[threshold_ids]
    design = np.column_stack([expanded_features, threshold_features])
    binary = np.column_stack(
        [(y > threshold).astype(int) for threshold in range(threshold_count)]
    ).reshape(-1)
    classifier = LogisticRegression(
        C=float(c),
        class_weight=class_weight,
        solver="lbfgs",
        fit_intercept=False,
        max_iter=2_000,
        random_state=42,
    )
    classifier.fit(design, binary)
    return {
        "family": "shared_slope_ordinal_logistic",
        "candidate_role": "shared_slope_ordinal",
        "feature_profile": "compact_21",
        "feature_names": list(feature_names),
        "tail": int(tail),
        "c": float(c),
        "class_weight": class_weight,
        "temperature": float(temperature),
        "imputer": imputer,
        "scaler": scaler,
        "classifier": classifier,
        "threshold_count": threshold_count,
        "tail_offsets": _fit_tail_offsets(frame, tail),
        "feature_availability_before_imputation": _feature_missingness_audit(frame, feature_names),
    }


def _fit_tail_offsets(frame: pd.DataFrame, tail: int) -> dict[str, dict[int, float]]:
    offsets = frame["offset_native"].astype(int)
    output: dict[str, dict[int, float]] = {}
    for label, selected, fallback in (
        ("lower", offsets[offsets <= -tail], -tail),
        ("upper", offsets[offsets >= tail], tail),
    ):
        counts = selected.value_counts(normalize=True).sort_index()
        output[label] = (
            {int(key): float(value) for key, value in counts.items()}
            if not counts.empty
            else {int(fallback): 1.0}
        )
    return output


def predict_ordinal(state: Mapping[str, Any], frame: pd.DataFrame) -> np.ndarray:
    if state["family"] == "shared_slope_ordinal_logistic":
        return _predict_shared_slope_ordinal(state, frame)
    x = frame.reindex(columns=state["feature_names"])
    exceed = []
    for model in state["threshold_models"]:
        if isinstance(model, (int, float, np.number)):
            exceed.append(np.full(len(frame), float(model), dtype=float))
        else:
            exceed.append(model.predict_proba(x)[:, 1])
    q = np.minimum.accumulate(np.column_stack(exceed), axis=1)
    probabilities = np.zeros((len(frame), q.shape[1] + 1), dtype=float)
    probabilities[:, 0] = 1.0 - q[:, 0]
    for index in range(1, probabilities.shape[1] - 1):
        probabilities[:, index] = q[:, index - 1] - q[:, index]
    probabilities[:, -1] = q[:, -1]
    logits = np.log(np.clip(probabilities, 1e-12, 1.0)) / float(state["temperature"])
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    blend_weight = state.get("blend_weight")
    blend_weight = 1.0 if blend_weight is None else float(blend_weight)
    if blend_weight < 1.0:
        empirical = _empirical_probabilities_from_state(
            state["empirical_state"],
            frame,
            prior_strength=float(state["empirical_prior_strength"]),
        )
        probabilities = _normalize_probability_rows(
            blend_weight * probabilities + (1.0 - blend_weight) * empirical
        )
    return probabilities


def _predict_shared_slope_ordinal(state: Mapping[str, Any], frame: pd.DataFrame) -> np.ndarray:
    x = frame.reindex(columns=state["feature_names"])
    transformed = state["scaler"].transform(state["imputer"].transform(x))
    threshold_count = int(state["threshold_count"])
    exceed: list[np.ndarray] = []
    for threshold in range(threshold_count):
        threshold_features = np.zeros((len(frame), threshold_count), dtype=float)
        threshold_features[:, threshold] = 1.0
        design = np.column_stack([transformed, threshold_features])
        exceed.append(state["classifier"].predict_proba(design)[:, 1])
    q = np.minimum.accumulate(np.column_stack(exceed), axis=1)
    probabilities = np.zeros((len(frame), threshold_count + 1), dtype=float)
    probabilities[:, 0] = 1.0 - q[:, 0]
    for index in range(1, threshold_count):
        probabilities[:, index] = q[:, index - 1] - q[:, index]
    probabilities[:, -1] = q[:, -1]
    logits = np.log(np.clip(probabilities, 1e-12, 1.0)) / float(state["temperature"])
    logits -= logits.max(axis=1, keepdims=True)
    return _normalize_probability_rows(np.exp(logits))


def _fit_empirical_state(frame: pd.DataFrame, tail: int) -> dict[str, Any]:
    classes = frame["offset_native"].astype(int).map(lambda value: _ordinal_class(value, tail))
    class_count = 2 * int(tail) + 1
    global_counts = np.bincount(classes.to_numpy(int), minlength=class_count).astype(float)
    monthly_counts: dict[int, list[float]] = {}
    for month, group in frame.assign(_class=classes).groupby(frame["contract_date"].dt.month):
        monthly_counts[int(month)] = np.bincount(
            group["_class"].to_numpy(int), minlength=class_count
        ).astype(float).tolist()
    return {
        "tail": int(tail),
        "global_counts": global_counts.tolist(),
        "monthly_counts": monthly_counts,
    }


def _empirical_probabilities_from_state(
    state: Mapping[str, Any],
    frame: pd.DataFrame,
    *,
    prior_strength: float,
) -> np.ndarray:
    global_counts = np.asarray(state["global_counts"], dtype=float) + 0.5
    global_probabilities = global_counts / global_counts.sum()
    output = np.zeros((len(frame), len(global_probabilities)), dtype=float)
    months = pd.to_datetime(frame["contract_date"], errors="coerce").dt.month
    for position, month in enumerate(months):
        month_counts = np.asarray(
            state["monthly_counts"].get(int(month), np.zeros(len(global_probabilities))),
            dtype=float,
        )
        output[position] = (
            month_counts + float(prior_strength) * global_probabilities
        ) / (month_counts.sum() + float(prior_strength))
    return _normalize_probability_rows(output)


def _normalize_probability_rows(probabilities: np.ndarray) -> np.ndarray:
    clean = np.maximum(np.asarray(probabilities, dtype=float), 0.0)
    totals = clean.sum(axis=1, keepdims=True)
    if np.any(totals <= 0.0):
        raise ValueError("probability distribution has no mass")
    return clean / totals


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0)))


def _gaussian_degree_probabilities(mean: float, sigma: float) -> dict[int, float]:
    center = round_half_up(mean)
    lower = center - 24
    upper = center + 24
    output = {
        degree: _normal_cdf((degree + 0.5 - mean) / sigma)
        - _normal_cdf((degree - 0.5 - mean) / sigma)
        for degree in range(lower, upper + 1)
    }
    output[lower] += _normal_cdf((lower - 0.5 - mean) / sigma)
    output[upper] += 1.0 - _normal_cdf((upper + 0.5 - mean) / sigma)
    return _normalize_mapping(output)


def _ordinal_degree_probabilities(point_degree: int, probabilities: np.ndarray, state: Mapping[str, Any]) -> dict[int, float]:
    tail = int(state["tail"])
    output: dict[int, float] = {}
    for index, mass in enumerate(probabilities):
        clipped_offset = index - tail
        if clipped_offset == -tail:
            spread = state["tail_offsets"]["lower"]
        elif clipped_offset == tail:
            spread = state["tail_offsets"]["upper"]
        else:
            spread = {clipped_offset: 1.0}
        for offset, weight in spread.items():
            degree = int(point_degree) + int(offset)
            output[degree] = output.get(degree, 0.0) + float(mass) * float(weight)
    return _normalize_mapping(output)


def _normalize_mapping(values: Mapping[Any, float]) -> dict[Any, float]:
    clean = {key: max(0.0, float(value)) for key, value in values.items() if math.isfinite(float(value))}
    total = sum(clean.values())
    if total <= 0.0:
        raise ValueError("probability distribution has no mass")
    return {key: value / total for key, value in clean.items()}


def _market_probabilities(degrees: Mapping[int, float], *, unit: str, bucket_width: int) -> dict[str, float]:
    output: dict[str, float] = {}
    for degree, probability in degrees.items():
        bucket = canonical_market_bucket(int(degree), unit=unit, bucket_width=bucket_width)
        output[bucket] = output.get(bucket, 0.0) + float(probability)
    return _normalize_mapping(output)


def probability_predictions(
    frame: pd.DataFrame,
    *,
    family: str,
    state: Mapping[str, Any],
    unit: str,
    bucket_width: int,
    period: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if family == "gaussian":
        residual_mean, sigma = predict_gaussian(state, frame)
        ordinal_values = None
        output_family = "gaussian"
    elif family in {"ordinal", "native_ordinal_reference", *ORDINAL_MEMBER_ROLES}:
        residual_mean = np.full(len(frame), np.nan)
        sigma = np.full(len(frame), np.nan)
        ordinal_values = predict_ordinal(state, frame)
        output_family = str(state.get("candidate_role") or family)
    else:
        raise ValueError(f"unknown probability family: {family}")
    for position, (_, row) in enumerate(frame.iterrows()):
        if family == "gaussian":
            final_mean = float(row["point_prediction_native"]) + float(residual_mean[position])
            degrees = _gaussian_degree_probabilities(final_mean, float(sigma[position]))
        else:
            final_mean = math.nan
            degrees = _ordinal_degree_probabilities(
                int(row["point_degree_native"]), ordinal_values[position], state
            )
        markets = _market_probabilities(degrees, unit=unit, bucket_width=bucket_width)
        ranked = sorted(markets.items(), key=lambda item: (-item[1], item[0]))
        actual_bucket = str(row["actual_market_bucket"])
        point_bucket = canonical_market_bucket(
            int(row["point_degree_native"]), unit=unit, bucket_width=bucket_width
        )
        point_probability = float(markets.get(point_bucket, 0.0))
        strongest_alternative = max((value for bucket, value in markets.items() if bucket != point_bucket), default=0.0)
        point_margin = point_probability - float(strongest_alternative)
        top_two_margin = float(ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0.0))
        thresholds = state.get("decision_thresholds") or {}
        vote = (
            point_probability >= float(thresholds["minimum_top_probability"])
            and point_margin >= float(thresholds["minimum_top_two_margin"])
            if thresholds
            else None
        )
        rows.append(
            {
                "contract_date": pd.Timestamp(row["contract_date"]).date().isoformat(),
                "period": period,
                "family": output_family,
                "actual_high_native": float(row["actual_high_native"]),
                "point_prediction_native": float(row["point_prediction_native"]),
                "point_degree_native": int(row["point_degree_native"]),
                "actual_market_bucket": actual_bucket,
                "point_market_bucket": point_bucket,
                "point_market_probability": point_probability,
                "point_market_margin": point_margin,
                "top_market_bucket": ranked[0][0],
                "top_market_probability": float(ranked[0][1]),
                "top_two_margin": top_two_margin,
                "ordinal_vote": vote,
                "actual_market_probability": float(markets.get(actual_bucket, 0.0)),
                "top_market_hit": ranked[0][0] == actual_bucket,
                "predicted_residual_mean_native": None if not math.isfinite(final_mean) else float(final_mean - row["point_prediction_native"]),
                "predicted_sigma_native": None if not math.isfinite(float(sigma[position])) else float(sigma[position]),
                "degree_probabilities": json.dumps({str(key): value for key, value in sorted(degrees.items())}, sort_keys=True),
                "market_bucket_probabilities": json.dumps(markets, sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def ordinal_ensemble_predictions(
    frame: pd.DataFrame,
    states: Mapping[str, Mapping[str, Any]],
    *,
    unit: str,
    bucket_width: int,
    period: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = [role for role in ORDINAL_MEMBER_ROLES if role not in states]
    if missing:
        raise ValueError("ordinal ensemble missing members: " + ",".join(missing))
    member_frames = [
        probability_predictions(
            frame,
            family=role,
            state=states[role],
            unit=unit,
            bucket_width=bucket_width,
            period=period,
        )
        for role in ORDINAL_MEMBER_ROLES
    ]
    ensemble_rows: list[dict[str, Any]] = []
    for position, (_, source_row) in enumerate(frame.iterrows()):
        member_markets = [
            json.loads(member.iloc[position]["market_bucket_probabilities"])
            for member in member_frames
        ]
        keys = sorted(set().union(*(values.keys() for values in member_markets)))
        median_markets = _normalize_mapping(
            {
                key: float(np.median([values.get(key, 0.0) for values in member_markets]))
                for key in keys
            }
        )
        ranked = sorted(median_markets.items(), key=lambda item: (-item[1], item[0]))
        actual_bucket = str(source_row["actual_market_bucket"])
        point_bucket = canonical_market_bucket(
            int(source_row["point_degree_native"]), unit=unit, bucket_width=bucket_width
        )
        votes = sum(bool(member.iloc[position]["ordinal_vote"]) for member in member_frames)
        selected_probabilities = [
            float(member.iloc[position]["point_market_probability"]) for member in member_frames
        ]
        ensemble_rows.append(
            {
                "contract_date": pd.Timestamp(source_row["contract_date"]).date().isoformat(),
                "period": period,
                "family": "ordinal_ensemble_median",
                "actual_high_native": float(source_row["actual_high_native"]),
                "point_prediction_native": float(source_row["point_prediction_native"]),
                "point_degree_native": int(source_row["point_degree_native"]),
                "actual_market_bucket": actual_bucket,
                "point_market_bucket": point_bucket,
                "point_market_probability": float(np.median(selected_probabilities)),
                "top_market_bucket": ranked[0][0],
                "top_market_probability": float(ranked[0][1]),
                "top_two_margin": float(ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0.0)),
                "ordinal_vote": votes >= 2,
                "ordinal_votes": int(votes),
                "ordinal_approved": votes >= 2,
                "actual_market_probability": float(median_markets.get(actual_bucket, 0.0)),
                "top_market_hit": ranked[0][0] == actual_bucket,
                "predicted_residual_mean_native": None,
                "predicted_sigma_native": None,
                "degree_probabilities": None,
                "market_bucket_probabilities": json.dumps(median_markets, sort_keys=True),
            }
        )
    return pd.concat(member_frames, ignore_index=True), pd.DataFrame(ensemble_rows)


def probability_metrics(predictions: pd.DataFrame) -> dict[str, Any]:
    if predictions.empty:
        return {"count": 0}
    brier_rows = []
    calibration = []
    for _, row in predictions.iterrows():
        probabilities = json.loads(row["market_bucket_probabilities"])
        actual = str(row["actual_market_bucket"])
        keys = set(probabilities) | {actual}
        brier_rows.append(
            sum((float(probabilities.get(key, 0.0)) - float(key == actual)) ** 2 for key in keys)
        )
        calibration.append((float(row["top_market_probability"]), bool(row["top_market_hit"])))
    calibration_error = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        selected = [(prob, hit) for prob, hit in calibration if lower <= prob < lower + 0.1]
        if selected:
            calibration_error += len(selected) / len(calibration) * abs(
                np.mean([hit for _, hit in selected]) - np.mean([prob for prob, _ in selected])
            )
    actual_probability = predictions["actual_market_probability"].clip(lower=1e-12)
    return {
        "count": int(len(predictions)),
        "market_log_loss": float(-np.log(actual_probability).mean()),
        "market_brier": float(np.mean(brier_rows)),
        "top_market_accuracy": float(predictions["top_market_hit"].mean()),
        "top_market_calibration_error": float(calibration_error),
        "mean_top_probability": float(predictions["top_market_probability"].mean()),
    }


def _selection_split(frame: pd.DataFrame, calibration_days: int = 90) -> tuple[pd.DataFrame, pd.DataFrame]:
    split = frame["contract_date"].max() - pd.Timedelta(days=calibration_days - 1)
    train = frame.loc[frame["contract_date"].lt(split)].copy()
    calibration = frame.loc[frame["contract_date"].ge(split)].copy()
    if len(train) < 60 or calibration.empty:
        raise ValueError("insufficient chronological probability selection history")
    return train, calibration


def tune_gaussian(frame: pd.DataFrame, feature_names: Sequence[str], *, unit: str, bucket_width: int) -> tuple[float, float, pd.DataFrame]:
    train, calibration = _selection_split(frame)
    rows = []
    for alpha in GAUSSIAN_ALPHA_GRID:
        base_state = fit_gaussian(train, feature_names, alpha=alpha, scale_multiplier=1.0)
        for multiplier in GAUSSIAN_SCALE_GRID:
            state = {**base_state, "scale_multiplier": float(multiplier)}
            predictions = probability_predictions(
                calibration, family="gaussian", state=state, unit=unit, bucket_width=bucket_width, period="inner_calibration"
            )
            metrics = probability_metrics(predictions)
            rows.append({"family": "gaussian", "alpha": alpha, "scale_multiplier": multiplier, **metrics})
    tuning = pd.DataFrame(rows).sort_values(
        ["market_log_loss", "market_brier", "top_market_calibration_error", "alpha", "scale_multiplier"]
    ).reset_index(drop=True)
    selected = tuning.iloc[0]
    return float(selected["alpha"]), float(selected["scale_multiplier"]), tuning


def tune_ordinal(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    tail: int,
    unit: str,
    bucket_width: int,
) -> tuple[float, str | None, float, pd.DataFrame]:
    train, calibration = _selection_split(frame)
    rows = []
    for c in ORDINAL_C_GRID:
        for class_weight in ORDINAL_CLASS_WEIGHTS:
            base_state = fit_ordinal(
                train,
                feature_names,
                tail=tail,
                c=c,
                class_weight=class_weight,
                temperature=1.0,
            )
            for temperature in ORDINAL_TEMPERATURE_GRID:
                state = {**base_state, "temperature": float(temperature)}
                predictions = probability_predictions(
                    calibration, family="ordinal", state=state, unit=unit, bucket_width=bucket_width, period="inner_calibration"
                )
                metrics = probability_metrics(predictions)
                rows.append(
                    {
                        "family": "ordinal",
                        "c": c,
                        "class_weight": class_weight or "none",
                        "temperature": temperature,
                        **metrics,
                    }
                )
    tuning = pd.DataFrame(rows).sort_values(
        ["market_log_loss", "market_brier", "top_market_calibration_error", "c", "temperature"]
    ).reset_index(drop=True)
    selected = tuning.iloc[0]
    weight = None if selected["class_weight"] == "none" else str(selected["class_weight"])
    return float(selected["c"]), weight, float(selected["temperature"]), tuning


def tune_shared_slope_ordinal(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    tail: int,
    unit: str,
    bucket_width: int,
) -> tuple[float, str | None, float, pd.DataFrame]:
    train, calibration = _selection_split(frame)
    rows: list[dict[str, Any]] = []
    for c in ORDINAL_C_GRID:
        for class_weight in ORDINAL_CLASS_WEIGHTS:
            base_state = fit_shared_slope_ordinal(
                train,
                feature_names,
                tail=tail,
                c=c,
                class_weight=class_weight,
                temperature=1.0,
            )
            for temperature in ORDINAL_TEMPERATURE_GRID:
                state = {**base_state, "temperature": float(temperature)}
                predictions = probability_predictions(
                    calibration,
                    family="shared_slope_ordinal",
                    state=state,
                    unit=unit,
                    bucket_width=bucket_width,
                    period="inner_calibration",
                )
                rows.append(
                    {
                        "family": "shared_slope_ordinal",
                        "c": c,
                        "class_weight": class_weight or "none",
                        "temperature": temperature,
                        **probability_metrics(predictions),
                    }
                )
    tuning = pd.DataFrame(rows).sort_values(
        ["market_log_loss", "market_brier", "top_market_calibration_error", "c", "temperature"]
    ).reset_index(drop=True)
    selected = tuning.iloc[0]
    weight = None if selected["class_weight"] == "none" else str(selected["class_weight"])
    return float(selected["c"]), weight, float(selected["temperature"]), tuning


def tune_confidence_thresholds(
    predictions: pd.DataFrame,
    *,
    target_coverage: float = 0.60,
) -> tuple[dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    point_hit = predictions["point_market_bucket"].eq(predictions["actual_market_bucket"])
    for minimum_top in np.arange(0.20, 0.701, 0.025):
        for minimum_margin in np.arange(0.0, 0.301, 0.025):
            selected = predictions["point_market_probability"].ge(minimum_top) & predictions[
                "point_market_margin"
            ].ge(minimum_margin)
            if not selected.any():
                continue
            coverage = float(selected.mean())
            rows.append(
                {
                    "minimum_top_probability": float(minimum_top),
                    "minimum_top_two_margin": float(minimum_margin),
                    "coverage": coverage,
                    "coverage_distance": abs(coverage - float(target_coverage)),
                    "selected_point_bucket_accuracy": float(point_hit.loc[selected].mean()),
                    "selected_count": int(selected.sum()),
                }
            )
    if not rows:
        raise ValueError("no ordinal confidence policy candidate was available")
    tuning = pd.DataFrame(rows)
    preferred = tuning.loc[tuning["coverage"].between(0.55, 0.65)].copy()
    if preferred.empty:
        preferred = tuning
    selected = preferred.sort_values(
        [
            "selected_point_bucket_accuracy",
            "coverage_distance",
            "minimum_top_probability",
            "minimum_top_two_margin",
        ],
        ascending=[False, True, True, True],
    ).iloc[0]
    return (
        {
            "policy_version": ORDINAL_POLICY_VERSION,
            "minimum_top_probability": float(selected["minimum_top_probability"]),
            "minimum_top_two_margin": float(selected["minimum_top_two_margin"]),
            "target_coverage": float(target_coverage),
            "overrides_enabled": False,
        },
        tuning,
    )


def fit_ordinal_candidates(
    frame: pd.DataFrame,
    *,
    tail: int,
    unit: str,
    bucket_width: int,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    full_features = tuple(frame.attrs.get("feature_names") or ())
    compact_features = tuple(name for name in ORDINAL_COMPACT_FEATURES if name in full_features)
    if len(compact_features) != len(ORDINAL_COMPACT_FEATURES):
        missing = sorted(set(ORDINAL_COMPACT_FEATURES) - set(compact_features))
        raise ValueError("compact ordinal feature contract missing: " + ",".join(missing))
    inner_train, calibration = _selection_split(frame)

    c, class_weight, temperature, pure_tuning = tune_ordinal(
        inner_train, full_features, tail=tail, unit=unit, bucket_width=bucket_width
    )
    inner_pure = fit_ordinal(
        inner_train,
        full_features,
        tail=tail,
        c=c,
        class_weight=class_weight,
        temperature=temperature,
    )
    pure_calibration = probability_predictions(
        calibration,
        family="pure_ordinal",
        state=inner_pure,
        unit=unit,
        bucket_width=bucket_width,
        period="inner_calibration",
    )
    pure_policy, pure_policy_tuning = tune_confidence_thresholds(pure_calibration)
    pure = fit_ordinal(
        frame,
        full_features,
        tail=tail,
        c=c,
        class_weight=class_weight,
        temperature=temperature,
    )
    pure["decision_thresholds"] = pure_policy

    native_features = tuple(name for name in ("point_prediction_native", "point_rounding_remainder_native", "point_distance_to_round_boundary_native") if name in full_features)
    if len(native_features) != 3:
        raise ValueError("native ordinal reference feature contract missing")
    native_inner = fit_ordinal(inner_train, native_features, tail=tail, c=1.0, class_weight=None, temperature=1.0)
    native_policy, native_policy_tuning = tune_confidence_thresholds(probability_predictions(calibration, family="native_ordinal_reference", state=native_inner, unit=unit, bucket_width=bucket_width, period="inner_calibration"))
    native_reference = fit_ordinal(frame, native_features, tail=tail, c=1.0, class_weight=None, temperature=1.0)
    native_reference.update({"family": "native_cumulative_ordinal_logistic", "candidate_role": "native_ordinal_reference", "feature_profile": "native_minimal_reference", "reference_contract": "single_xgboost_native_unit_ordinal_reference_v2", "decision_thresholds": native_policy})

    blend_rows: list[dict[str, Any]] = []
    blend_train, blend_calibration = _selection_split(inner_train)
    blend_pure = fit_ordinal(
        blend_train, full_features, tail=tail, c=c, class_weight=class_weight, temperature=temperature,
    )
    empirical_inner = _fit_empirical_state(inner_train, tail)
    for prior_strength in ORDINAL_PRIOR_STRENGTH_GRID:
        for blend_weight in ORDINAL_BLEND_WEIGHT_GRID:
            candidate = {
                **blend_pure,
                "family": "blended_cumulative_ordinal_logistic",
                "candidate_role": "blended_ordinal",
                "feature_profile": "full",
                "blend_weight": float(blend_weight),
                "empirical_prior_strength": float(prior_strength),
                "empirical_state": empirical_inner,
            }
            candidate_predictions = probability_predictions(
                blend_calibration,
                family="blended_ordinal",
                state=candidate,
                unit=unit,
                bucket_width=bucket_width,
                period="inner_calibration",
            )
            blend_rows.append(
                {
                    "family": "blended_ordinal",
                    "blend_weight": blend_weight,
                    "empirical_prior_strength": prior_strength,
                    **probability_metrics(candidate_predictions),
                }
            )
    blend_tuning = pd.DataFrame(blend_rows).sort_values(
        [
            "market_log_loss",
            "market_brier",
            "top_market_calibration_error",
            "blend_weight",
            "empirical_prior_strength",
        ]
    ).reset_index(drop=True)
    selected_blend = blend_tuning.iloc[0]
    inner_blended = {
        **inner_pure,
        "family": "blended_cumulative_ordinal_logistic",
        "candidate_role": "blended_ordinal",
        "feature_profile": "full",
        "blend_weight": float(selected_blend["blend_weight"]),
        "empirical_prior_strength": float(selected_blend["empirical_prior_strength"]),
        "empirical_state": empirical_inner,
    }
    blended_calibration = probability_predictions(
        calibration,
        family="blended_ordinal",
        state=inner_blended,
        unit=unit,
        bucket_width=bucket_width,
        period="inner_calibration",
    )
    blended_policy, blended_policy_tuning = tune_confidence_thresholds(blended_calibration)
    blended = {
        **pure,
        "family": "blended_cumulative_ordinal_logistic",
        "candidate_role": "blended_ordinal",
        "feature_profile": "full",
        "blend_weight": float(selected_blend["blend_weight"]),
        "empirical_prior_strength": float(selected_blend["empirical_prior_strength"]),
        "empirical_state": _fit_empirical_state(frame, tail),
        "decision_thresholds": blended_policy,
    }

    shared_c, shared_weight, shared_temperature, shared_tuning = tune_shared_slope_ordinal(
        inner_train,
        compact_features,
        tail=tail,
        unit=unit,
        bucket_width=bucket_width,
    )
    inner_shared = fit_shared_slope_ordinal(
        inner_train,
        compact_features,
        tail=tail,
        c=shared_c,
        class_weight=shared_weight,
        temperature=shared_temperature,
    )
    shared_calibration = probability_predictions(
        calibration,
        family="shared_slope_ordinal",
        state=inner_shared,
        unit=unit,
        bucket_width=bucket_width,
        period="inner_calibration",
    )
    shared_policy, shared_policy_tuning = tune_confidence_thresholds(shared_calibration)
    shared = fit_shared_slope_ordinal(
        frame,
        compact_features,
        tail=tail,
        c=shared_c,
        class_weight=shared_weight,
        temperature=shared_temperature,
    )
    shared["decision_thresholds"] = shared_policy

    pure_tuning = pure_tuning.assign(candidate_role="pure_ordinal", tuning_stage="model")
    shared_tuning = shared_tuning.assign(candidate_role="shared_slope_ordinal", tuning_stage="model")
    blend_tuning = blend_tuning.assign(candidate_role="blended_ordinal", tuning_stage="blend")
    policy_frames = []
    for role, policy_tuning in (
        ("blended_ordinal", blended_policy_tuning),
        ("shared_slope_ordinal", shared_policy_tuning),
        ("pure_ordinal", pure_policy_tuning),
        ("native_ordinal_reference", native_policy_tuning),
    ):
        policy_frames.append(
            policy_tuning.assign(candidate_role=role, family=role, tuning_stage="confidence_policy")
        )
    tuning = pd.concat(
        [pure_tuning, shared_tuning, blend_tuning, *policy_frames],
        ignore_index=True,
        sort=False,
    )
    return (
        {
            "native_ordinal_reference": native_reference,
            "blended_ordinal": blended,
            "shared_slope_ordinal": shared,
            "pure_ordinal": pure,
        },
        tuning,
    )


def run_probability_walk_forward(
    frame: pd.DataFrame,
    *,
    station_id: str,
    development_years: Sequence[int],
    validation_years: Sequence[int],
    tail: int,
    unit: str,
    bucket_width: int,
) -> ProbabilityRun:
    feature_names = tuple(frame.attrs.get("feature_names") or ())
    development = frame.loc[frame["year"].isin([int(value) for value in development_years])].copy()
    predictions: list[pd.DataFrame] = []
    tuning: list[pd.DataFrame] = []
    completed_years: list[int] = []
    required_years = [int(year) for year in validation_years]
    unavailable_years: list[int] = []
    for required_year in required_years:
        required_train = development.loc[development["year"].lt(required_year)]
        required_valid = development.loc[development["year"].eq(required_year)]
        if len(required_train) < 180 or required_valid.empty:
            unavailable_years.append(required_year)
    if unavailable_years:
        raise ValueError(f"probability forward folds unavailable for {station_id}: {unavailable_years}")
    for year in validation_years:
        train = development.loc[development["year"].lt(int(year))].copy()
        valid = development.loc[development["year"].eq(int(year))].copy()
        if len(train) < 180 or valid.empty:
            continue
        alpha, multiplier, gaussian_tuning = tune_gaussian(train, feature_names, unit=unit, bucket_width=bucket_width)
        gaussian_state = fit_gaussian(train, feature_names, alpha=alpha, scale_multiplier=multiplier)
        gaussian_predictions = probability_predictions(
            valid, family="gaussian", state=gaussian_state, unit=unit, bucket_width=bucket_width, period=f"forward_{year}"
        )
        ordinal_states, ordinal_tuning = fit_ordinal_candidates(
            train, tail=tail, unit=unit, bucket_width=bucket_width
        )
        native_predictions = probability_predictions(
            valid,
            family="native_ordinal_reference",
            state=ordinal_states["native_ordinal_reference"],
            unit=unit,
            bucket_width=bucket_width,
            period=f"forward_{year}",
        )
        member_predictions, ensemble_predictions = ordinal_ensemble_predictions(
            valid,
            ordinal_states,
            unit=unit,
            bucket_width=bucket_width,
            period=f"forward_{year}",
        )
        predictions.extend(
            [gaussian_predictions, native_predictions, member_predictions, ensemble_predictions]
        )
        gaussian_tuning["validation_year"] = int(year)
        ordinal_tuning["validation_year"] = int(year)
        tuning.extend([gaussian_tuning, ordinal_tuning])
        completed_years.append(int(year))
    if not predictions:
        raise ValueError(f"no probability forward folds produced for {station_id}")
    if completed_years != required_years:
        raise ValueError(f"incomplete probability forward folds for {station_id}: expected={required_years} completed={completed_years}")
    alpha, multiplier, final_gaussian_tuning = tune_gaussian(
        development, feature_names, unit=unit, bucket_width=bucket_width
    )
    final_gaussian = fit_gaussian(development, feature_names, alpha=alpha, scale_multiplier=multiplier)
    final_ordinals, final_ordinal_tuning = fit_ordinal_candidates(
        development, tail=tail, unit=unit, bucket_width=bucket_width
    )
    final_gaussian_tuning["validation_year"] = "final"
    final_ordinal_tuning["validation_year"] = "final"
    tuning.extend([final_gaussian_tuning, final_ordinal_tuning])
    forward = pd.concat(predictions, ignore_index=True)
    metrics = pd.DataFrame(
        [
            {
                "station_id": station_id,
                "period": period,
                "family": family,
                **probability_metrics(group),
            }
            for (period, family), group in forward.groupby(["period", "family"], sort=True)
        ]
    )
    return ProbabilityRun(
        gaussian_state=final_gaussian,
        ordinal_states=final_ordinals,
        forward_predictions=forward,
        forward_metrics=metrics,
        tuning=pd.concat(tuning, ignore_index=True),
    )


def fit_production_probability_models(
    frame: pd.DataFrame,
    *,
    tail: int,
    unit: str,
    bucket_width: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], pd.DataFrame]:
    feature_names = tuple(frame.attrs.get("feature_names") or ())
    alpha, multiplier, gaussian_tuning = tune_gaussian(frame, feature_names, unit=unit, bucket_width=bucket_width)
    gaussian = fit_gaussian(frame, feature_names, alpha=alpha, scale_multiplier=multiplier)
    ordinals, ordinal_tuning = fit_ordinal_candidates(
        frame, tail=tail, unit=unit, bucket_width=bucket_width
    )
    return gaussian, ordinals, pd.concat([gaussian_tuning, ordinal_tuning], ignore_index=True)


def export_probability_artifact(
    state: Mapping[str, Any],
    output_dir: str | Path,
    *,
    artifact_type: str,
    station_id: str,
    model_version: str,
    point_model_version: str,
    point_bundle_sha256: str,
    unit: str,
    bucket_width: int,
    training_frame: pd.DataFrame,
    validation_metrics: Sequence[Mapping[str, Any]],
    external_evaluation_evidence: Sequence[Mapping[str, Any]] | None = None,
    source_identity: Mapping[str, Any],
    release_role: str,
) -> tuple[Path, Path]:
    import joblib

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": artifact_type,
        "station_id": station_id,
        "model_version": model_version,
        "point_model_version": point_model_version,
        "point_bundle_sha256": point_bundle_sha256,
        "unit": unit,
        "bucket_width": int(bucket_width),
        "release_role": release_role,
        "feature_names": list(state["feature_names"]),
        "candidate_role": state.get("candidate_role"),
        "family": state.get("family"),
        "feature_profile": state.get("feature_profile"),
        "decision_thresholds": dict(state.get("decision_thresholds") or {}),
        "model_state": dict(state),
    }
    stem = f"{station_id}_{model_version}"
    bundle_path = output / f"{stem}.joblib"
    manifest_path = output / f"{stem}.json"
    joblib.dump(bundle, bundle_path)
    manifest = {
        key: bundle[key]
        for key in (
            "schema_version",
            "artifact_type",
            "station_id",
            "model_version",
            "point_model_version",
            "point_bundle_sha256",
            "unit",
            "bucket_width",
            "release_role",
            "feature_names",
            "candidate_role",
            "family",
            "feature_profile",
            "decision_thresholds",
        )
    }
    manifest.update(
        {
            "selected_parameters": _state_parameters(state),
            "training": {
                "start": pd.Timestamp(training_frame["contract_date"].min()).date().isoformat(),
                "cutoff": pd.Timestamp(training_frame["contract_date"].max()).date().isoformat(),
                "rows": int(len(training_frame)),
                "point_predictions_are_out_of_sample": True,
            },
            "validation_metrics": [dict(value) for value in validation_metrics],
            "external_evaluation_evidence": [dict(value) for value in (external_evaluation_evidence or ())],
            "feature_availability_before_imputation": list(state.get("feature_availability_before_imputation") or ()),
            "row_completeness": dict(training_frame.attrs.get("row_completeness") or {}),
            "source_identity": dict(source_identity),
            "package_versions": _package_versions(),
            "artifact_integrity": {"bundle_sha256": sha256_file(bundle_path)},
            "approval_status": (
                "unapproved_dirty_source_candidate"
                if bool(source_identity.get("git_dirty"))
                else (
                    "frozen_evaluation_artifact"
                    if release_role == "frozen_evaluation"
                    else "unapproved_production_candidate"
                )
            ),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return bundle_path, manifest_path


def export_ordinal_ensemble_manifest(
    output_path: str | Path,
    *,
    station_id: str,
    point_model_version: str,
    point_bundle_sha256: str,
    unit: str,
    bucket_width: int,
    member_artifacts: Mapping[str, tuple[Path, Path]],
    source_identity: Mapping[str, Any],
    release_role: str,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    expected = {"native_ordinal_reference", *ORDINAL_MEMBER_ROLES}
    missing = sorted(expected - set(member_artifacts))
    if missing:
        raise ValueError("ordinal ensemble manifest missing candidates: " + ",".join(missing))
    members: dict[str, Any] = {}
    seen_bundle_hashes: set[str] = set()
    training_contracts: set[tuple[Any, Any, Any]] = set()
    for role in ("native_ordinal_reference", *ORDINAL_MEMBER_ROLES):
        bundle_path, manifest_path = member_artifacts[role]
        candidate_manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        expected_family = {
            "native_ordinal_reference": "native_cumulative_ordinal_logistic",
            "blended_ordinal": "blended_cumulative_ordinal_logistic",
            "shared_slope_ordinal": "shared_slope_ordinal_logistic",
            "pure_ordinal": "cumulative_ordinal_logistic",
        }[role]
        expected = {"station_id": station_id, "candidate_role": role, "family": expected_family,
                    "point_model_version": point_model_version, "point_bundle_sha256": point_bundle_sha256,
                    "unit": unit, "bucket_width": int(bucket_width), "release_role": release_role}
        bad = [key for key, value in expected.items() if candidate_manifest.get(key) != value]
        if bad:
            raise ValueError("ordinal candidate manifest contract mismatch for " + role + ":" + ",".join(bad))
        training = candidate_manifest.get("training")
        if not isinstance(training, Mapping) or not {"start", "cutoff", "rows"}.issubset(training):
            raise ValueError("ordinal candidate manifest missing training contract: " + role)
        training_contracts.add((training["start"], training["cutoff"], training["rows"]))
        bundle_hash = sha256_file(bundle_path)
        declared_hash = (candidate_manifest.get("artifact_integrity") or {}).get("bundle_sha256")
        if declared_hash != bundle_hash:
            raise ValueError("ordinal candidate bundle SHA-256 mismatch: " + role)
        import joblib
        bundle = joblib.load(bundle_path)
        bundle_bad = [key for key in ("station_id", "model_version", "point_model_version", "point_bundle_sha256", "unit", "bucket_width", "candidate_role", "family", "release_role") if bundle.get(key) != candidate_manifest.get(key)]
        if bundle_bad:
            raise ValueError("ordinal candidate bundle/manifest identity mismatch for " + role + ":" + ",".join(bundle_bad))
        _validate_ordinal_learned_state(role, bundle.get("model_state"))
        if bundle_hash in seen_bundle_hashes:
            raise ValueError("ordinal ensemble candidate artifacts must be distinct")
        seen_bundle_hashes.add(bundle_hash)
        members[role] = {
            "model_version": candidate_manifest["model_version"],
            "candidate_role": role,
            "voting_member": role in ORDINAL_MEMBER_ROLES,
            "bundle_path": Path(os.path.relpath(bundle_path.resolve(), output.parent.resolve())).as_posix(),
            "manifest_path": Path(os.path.relpath(manifest_path.resolve(), output.parent.resolve())).as_posix(),
            "bundle_sha256": bundle_hash,
            "manifest_sha256": sha256_file(manifest_path),
            "feature_profile": candidate_manifest.get("feature_profile"),
            "decision_thresholds": candidate_manifest.get("decision_thresholds", {}),
        }
    if len(training_contracts) != 1:
        raise ValueError("ordinal ensemble candidates do not share a training date/row contract")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ORDINAL_ENSEMBLE_ARTIFACT_TYPE,
        "station_id": station_id,
        "release_role": release_role,
        "policy_version": ORDINAL_POLICY_VERSION,
        "required_votes": 2,
        "require_all_models": True,
        "voting_roles": list(ORDINAL_MEMBER_ROLES),
        "reference_role": "native_ordinal_reference",
        "reference_required_for_voting": False,
        "aggregation": "median_selected_bucket",
        "point_model_version": point_model_version,
        "point_bundle_sha256": point_bundle_sha256,
        "unit": unit,
        "bucket_width": int(bucket_width),
        "members": members,
        "source_identity": dict(source_identity),
        "approval_status": (
            "unapproved_dirty_source_candidate"
            if bool(source_identity.get("git_dirty"))
            else (
                "frozen_evaluation_artifact"
                if release_role == "frozen_evaluation"
                else "unapproved_production_candidate"
            )
        ),
    }
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _validate_ordinal_learned_state(role: str, state: Any) -> None:
    if not isinstance(state, Mapping):
        raise ValueError("ordinal candidate missing learned state: " + role)
    family = state.get("family")
    profile = state.get("feature_profile")
    features = list(state.get("feature_names") or ())
    if not features or int(state.get("tail", 0)) <= 0:
        raise ValueError("ordinal learned state missing feature/tail contract")
    temperature = float(state.get("temperature", float("nan")))
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("ordinal learned state has invalid temperature")
    if state.get("candidate_role") not in {None, role}:
        raise ValueError("ordinal learned state candidate role mismatch")
    threshold_models = list(state.get("threshold_models") or ())
    if any((isinstance(model, (int, float, np.number)) and (not math.isfinite(float(model)) or not 0.0 <= float(model) <= 1.0)) or (not isinstance(model, (int, float, np.number)) and not callable(getattr(model, "predict_proba", None))) for model in threshold_models):
        raise ValueError("ordinal learned state has unfitted threshold model")
    if any(getattr(model, "n_features_in_", len(features)) != len(features) for model in threshold_models if not isinstance(model, (int, float, np.number))):
        raise ValueError("ordinal learned state threshold feature dimensions mismatch")
    tail = int(state.get("tail", 0))
    tail_offsets = state.get("tail_offsets")
    if not isinstance(tail_offsets, Mapping) or set(tail_offsets) != {"lower", "upper"}:
        raise ValueError("ordinal learned state has invalid tail-offset structure")
    for label, direction in (("lower", -1), ("upper", 1)):
        masses = tail_offsets[label]
        if not isinstance(masses, Mapping) or not masses:
            raise ValueError("ordinal learned state has empty tail offsets")
        try:
            keys = [int(key) for key in masses]
            values = [float(value) for value in masses.values()]
        except (TypeError, ValueError) as exc:
            raise ValueError("ordinal learned state has non-numeric tail offsets") from exc
        if any(value < 0 or not math.isfinite(value) for value in values) or not math.isclose(sum(values), 1.0, abs_tol=1e-6):
            raise ValueError("ordinal learned state has invalid tail-offset mass")
        out_of_tail = any(key > -tail for key in keys) if direction < 0 else any(key < tail for key in keys)
        if out_of_tail:
            raise ValueError("ordinal learned state tail offsets are outside their tails")
    if role == "native_ordinal_reference":
        if family != "native_cumulative_ordinal_logistic" or profile != "native_minimal_reference" or len(features) != 3 or len(state.get("threshold_models") or ()) != 2 * tail:
            raise ValueError("native ordinal learned-state contract mismatch")
    elif role == "blended_ordinal":
        empirical = state.get("empirical_state")
        global_counts = empirical.get("global_counts") if isinstance(empirical, Mapping) else None
        monthly_counts = empirical.get("monthly_counts") if isinstance(empirical, Mapping) else None
        if family != "blended_cumulative_ordinal_logistic" or not isinstance(empirical, Mapping) or not isinstance(monthly_counts, Mapping) or int(empirical.get("tail", -1)) != tail or not isinstance(global_counts, (list, tuple)) or len(global_counts) != 2 * tail + 1 or any(not math.isfinite(float(value)) or float(value) < 0 for value in global_counts) or sum(float(value) for value in global_counts) <= 0 or any(not isinstance(values, (list, tuple)) or len(values) != 2 * tail + 1 or any(not math.isfinite(float(value)) or float(value) < 0 for value in values) for values in monthly_counts.values()) or not math.isfinite(float(state.get("empirical_prior_strength", float("nan")))) or float(state.get("empirical_prior_strength")) <= 0 or not 0 <= float(state.get("blend_weight", 1.0)) < 1 or len(state.get("threshold_models") or ()) != 2 * tail:
            raise ValueError("blended ordinal learned-state contract mismatch")
    elif role == "shared_slope_ordinal":
        if family != "shared_slope_ordinal_logistic" or profile != "compact_21" or not callable(getattr(state.get("imputer"), "transform", None)) or not callable(getattr(state.get("scaler"), "transform", None)) or not callable(getattr(state.get("classifier"), "predict_proba", None)) or getattr(state.get("imputer"), "n_features_in_", len(features)) != len(features) or getattr(state.get("scaler"), "n_features_in_", len(features)) != len(features) or getattr(state.get("classifier"), "n_features_in_", len(features) + int(state.get("threshold_count", 0))) != len(features) + int(state.get("threshold_count", 0)) or int(state.get("threshold_count", 0)) != 2 * tail:
            raise ValueError("shared-slope ordinal learned-state contract mismatch")
    elif role == "pure_ordinal":
        if family != "cumulative_ordinal_logistic" or profile != "full" or state.get("empirical_state") is not None or len(state.get("threshold_models") or ()) != 2 * tail:
            raise ValueError("pure ordinal learned-state contract mismatch")


def _state_parameters(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: state[key]
        for key in (
            "family",
            "candidate_role",
            "feature_profile",
            "alpha",
            "scale_multiplier",
            "tail",
            "c",
            "class_weight",
            "temperature",
            "blend_weight",
            "empirical_prior_strength",
            "reference_contract",
        )
        if key in state
    }


def _package_versions() -> dict[str, str]:
    distributions = {
        "numpy": "numpy",
        "pandas": "pandas",
        "scikit_learn": "scikit-learn",
        "joblib": "joblib",
    }
    return {key: importlib.metadata.version(value) for key, value in distributions.items()}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
