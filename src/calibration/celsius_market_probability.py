from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .bucket_probability import (
    FEATURE_PROFILE_PEAK_AUGMENTED,
    MISSING_INDICATOR_SUFFIX,
    add_probability_features,
    build_probability_frame,
    probability_feature_names,
    probability_base_methods,
    probability_mandatory_feature_names,
    probability_provider_names,
)


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "station_celsius_market_probability_model"
OFFSET_LABELS_C = ("<=-3", "-2", "-1", "0", "+1", "+2", ">=+3")
CENTRAL_OFFSETS_C = (-2, -1, 0, 1, 2)
TAIL_BOUNDARY_C = 3
MODEL_FAMILY = "celsius_offset_ordinal_logistic"
TARGET_CONTRACT = (
    "point_bucket_c=round_half_up((point_prediction_f-32)*5/9); "
    "actual_bucket_c=round_half_up(actual_high_c); "
    "offset_c=actual_bucket_c-point_bucket_c"
)


def round_half_up(value: float) -> int:
    """Tokyo market rounding: floor(value + 0.5), including negative values."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("temperature must be finite")
    return int(math.floor(number + 0.5))


def fahrenheit_to_celsius(value_f: float) -> float:
    number = float(value_f)
    if not math.isfinite(number):
        raise ValueError("temperature must be finite")
    return (number - 32.0) * 5.0 / 9.0


def offset_class_index_c(offset_c: int) -> int:
    offset = int(offset_c)
    if offset <= -TAIL_BOUNDARY_C:
        return 0
    if offset >= TAIL_BOUNDARY_C:
        return len(OFFSET_LABELS_C) - 1
    return offset + TAIL_BOUNDARY_C


def offset_label_c(offset_c: int) -> str:
    return OFFSET_LABELS_C[offset_class_index_c(offset_c)]


def build_celsius_probability_frame(
    feature_frame: pd.DataFrame,
    point_predictions: pd.DataFrame,
    base_validation_predictions: pd.DataFrame,
    *,
    include_peak_features: bool,
    feature_profile: str | None = None,
) -> pd.DataFrame:
    """Build the live-safe feature frame with a whole-Celsius market target."""
    frame = build_probability_frame(
        feature_frame,
        point_predictions,
        base_validation_predictions,
        include_peak_features=include_peak_features,
        feature_profile=feature_profile,
    )
    actual_c = pd.Series(np.nan, index=frame.index, dtype=float)
    source = pd.Series("actual_high_f_converted_to_c", index=frame.index, dtype=object)
    # These are the only raw-C columns accepted as settlement-equivalent targets.
    # iem_daily_high_c is deliberately excluded because Tokyo's current target is
    # sourced from Wunderground, not the diagnostic IEM daily-high field.
    for column in ("actual_high_c", "settlement_high_c"):
        if column in frame:
            candidate = pd.to_numeric(frame[column], errors="coerce")
            use = actual_c.isna() & candidate.notna()
            actual_c.loc[use] = candidate.loc[use]
            source.loc[use] = column
    fallback = pd.to_numeric(frame["actual_high_f"], errors="coerce").map(
        lambda value: fahrenheit_to_celsius(value) if pd.notna(value) else np.nan
    )
    actual_c = actual_c.fillna(fallback)
    frame["actual_high_c"] = actual_c
    frame["actual_high_c_source"] = source
    frame["point_prediction_c"] = pd.to_numeric(
        frame["point_prediction_f"], errors="coerce"
    ).map(fahrenheit_to_celsius)
    frame["point_bucket_c"] = frame["point_prediction_c"].map(round_half_up)
    frame["actual_bucket_c"] = frame["actual_high_c"].map(round_half_up)
    frame["offset_c"] = (
        frame["actual_bucket_c"].astype(int) - frame["point_bucket_c"].astype(int)
    )
    frame["offset_class_c"] = frame["offset_c"].map(offset_class_index_c)
    return frame.sort_values("contract_date").reset_index(drop=True)


@dataclass(frozen=True)
class CelsiusCandidate:
    c: float
    class_weight: str | None


def default_celsius_candidates() -> tuple[CelsiusCandidate, ...]:
    return tuple(
        CelsiusCandidate(c, class_weight)
        for c in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
        for class_weight in (None, "balanced")
    )


def _normalize(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    values[~np.isfinite(values)] = 0.0
    values = np.clip(values, 0.0, None)
    totals = values.sum(axis=1, keepdims=True)
    if np.any(totals <= 0.0):
        raise ValueError("probabilities have zero mass")
    return values / totals


def _fit_ordinal(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    candidate: CelsiusCandidate,
    *,
    random_state: int,
) -> dict[str, Any]:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    x = frame.reindex(columns=feature_names)
    y = frame["offset_class_c"].to_numpy(dtype=int)
    models: list[Any] = []
    for threshold in range(len(OFFSET_LABELS_C) - 1):
        binary = (y > threshold).astype(int)
        if np.unique(binary).size < 2:
            models.append(float(binary[0]))
            continue
        pipeline = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median", add_indicator=False, keep_empty_features=True
                    ),
                ),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=float(candidate.c),
                        class_weight=candidate.class_weight,
                        solver="lbfgs",
                        max_iter=2_000,
                        random_state=random_state,
                    ),
                ),
            ]
        )
        pipeline.fit(x, binary)
        models.append(pipeline)
    return {"threshold_models": models}


def _predict_ordinal(
    state: Mapping[str, Any], frame: pd.DataFrame, feature_names: Sequence[str]
) -> np.ndarray:
    x = frame.reindex(columns=feature_names)
    exceed = []
    for model in state["threshold_models"]:
        if isinstance(model, float):
            exceed.append(np.full(len(frame), model, dtype=float))
        else:
            exceed.append(model.predict_proba(x)[:, 1])
    q = np.minimum.accumulate(np.column_stack(exceed), axis=1)
    probabilities = np.zeros((len(frame), len(OFFSET_LABELS_C)), dtype=float)
    probabilities[:, 0] = 1.0 - q[:, 0]
    for index in range(1, len(OFFSET_LABELS_C) - 1):
        probabilities[:, index] = q[:, index - 1] - q[:, index]
    probabilities[:, -1] = q[:, -1]
    return _normalize(probabilities)


def _temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probabilities, 1e-12, 1.0)) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    return _normalize(np.exp(logits))


def score_offset_probabilities(
    actual_classes: np.ndarray, probabilities: np.ndarray
) -> dict[str, float]:
    actual = np.asarray(actual_classes, dtype=int)
    probs = _normalize(probabilities)
    positions = np.arange(len(actual))
    one_hot = np.eye(len(OFFSET_LABELS_C))[actual]
    predicted_cdf = np.cumsum(probs, axis=1)[:, :-1]
    actual_cdf = np.cumsum(one_hot, axis=1)[:, :-1]
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == actual
    calibration_error = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        selected = (confidence >= lower) & (confidence < lower + 0.1)
        if selected.any():
            calibration_error += float(selected.mean()) * abs(
                float(correct[selected].mean()) - float(confidence[selected].mean())
            )
    return {
        "offset_log_loss": float(
            -np.log(np.clip(probs[positions, actual], 1e-12, 1.0)).mean()
        ),
        "offset_brier": float(np.square(probs - one_hot).sum(axis=1).mean()),
        "ranked_probability_score": float(
            np.square(predicted_cdf - actual_cdf).sum(axis=1).mean()
            / (len(OFFSET_LABELS_C) - 1)
        ),
        "offset_accuracy": float(correct.mean()),
        "offset_top_two_accuracy": float(
            np.mean(
                [
                    class_index in row
                    for class_index, row in zip(
                        actual, np.argsort(probs, axis=1)[:, -2:], strict=True
                    )
                ]
            )
        ),
        "offset_calibration_error": float(calibration_error),
        "count": int(len(actual)),
    }


def fit_tail_policy_c(offsets: Sequence[int]) -> dict[str, Any]:
    values = np.asarray(list(offsets), dtype=int)
    if values.size == 0:
        raise ValueError("tail policy requires training offsets")

    def weights(mask: np.ndarray, fallback: int) -> dict[str, float]:
        selected = values[mask]
        if selected.size == 0:
            selected = np.asarray([fallback], dtype=int)
        labels, counts = np.unique(selected, return_counts=True)
        smoothed = counts.astype(float) + 0.5
        smoothed /= smoothed.sum()
        return {str(int(label)): float(weight) for label, weight in zip(labels, smoothed, strict=True)}

    return {
        "low_exact_offset_weights": weights(values <= -TAIL_BOUNDARY_C, -TAIL_BOUNDARY_C),
        "high_exact_offset_weights": weights(values >= TAIL_BOUNDARY_C, TAIL_BOUNDARY_C),
        "fitted_min_offset_c": int(values.min()),
        "fitted_max_offset_c": int(values.max()),
        "smoothing_alpha": 0.5,
    }


def offset_to_market_bucket_probabilities_c(
    point_bucket_c: int,
    offset_probabilities: Mapping[str, float] | Sequence[float],
    tail_policy: Mapping[str, Any],
) -> dict[int, float]:
    if isinstance(offset_probabilities, Mapping):
        vector = np.asarray(
            [float(offset_probabilities[label]) for label in OFFSET_LABELS_C], dtype=float
        )
    else:
        vector = np.asarray(offset_probabilities, dtype=float)
    vector = _normalize(vector.reshape(1, -1))[0]
    exact: dict[int, float] = {}
    for label, offset in zip(OFFSET_LABELS_C[1:-1], CENTRAL_OFFSETS_C, strict=True):
        exact[offset] = exact.get(offset, 0.0) + float(vector[OFFSET_LABELS_C.index(label)])
    for class_index, key in (
        (0, "low_exact_offset_weights"),
        (len(OFFSET_LABELS_C) - 1, "high_exact_offset_weights"),
    ):
        weights = {int(k): float(v) for k, v in tail_policy[key].items()}
        total = sum(weights.values())
        for offset, weight in weights.items():
            exact[offset] = exact.get(offset, 0.0) + float(vector[class_index]) * weight / total
    buckets = {
        int(point_bucket_c) + int(offset): float(probability)
        for offset, probability in exact.items()
    }
    total = sum(buckets.values())
    return dict(sorted((bucket, probability / total) for bucket, probability in buckets.items()))


def market_tail_ambiguity_c(
    offset_probabilities: Mapping[str, float] | Sequence[float],
    tail_policy: Mapping[str, Any],
    top_two_margin: float,
) -> bool:
    vector = (
        [float(offset_probabilities[label]) for label in OFFSET_LABELS_C]
        if isinstance(offset_probabilities, Mapping)
        else list(offset_probabilities)
    )
    for index, key in (
        (0, "low_exact_offset_weights"),
        (len(OFFSET_LABELS_C) - 1, "high_exact_offset_weights"),
    ):
        if len(tail_policy[key]) > 1 and float(vector[index]) + 1e-12 >= float(top_two_margin):
            return True
    return False


def _market_fields(
    point_bucket_c: int,
    actual_bucket_c: int | None,
    vector: Sequence[float],
    tail_policy: Mapping[str, Any],
) -> dict[str, Any]:
    offset_probabilities = {
        label: float(value) for label, value in zip(OFFSET_LABELS_C, vector, strict=True)
    }
    market = offset_to_market_bucket_probabilities_c(
        point_bucket_c, offset_probabilities, tail_policy
    )
    ranked = sorted(market.items(), key=lambda item: (-item[1], item[0]))
    top_bucket, top_probability = ranked[0]
    second_probability = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = float(top_probability - second_probability)
    point_probability = float(market.get(int(point_bucket_c), 0.0))
    return {
        "point_bucket_c": int(point_bucket_c),
        "actual_bucket_c": int(actual_bucket_c) if actual_bucket_c is not None else None,
        "recommended_bucket_c": int(top_bucket),
        "recommended_bucket_probability_c": float(top_probability),
        "actual_bucket_probability_c": (
            float(market.get(int(actual_bucket_c), 0.0))
            if actual_bucket_c is not None
            else None
        ),
        "market_top_probability_c": float(top_probability),
        "market_top_two_margin_c": margin,
        "market_switch_advantage_c": float(top_probability - point_probability),
        "market_tail_ambiguity_c": market_tail_ambiguity_c(
            offset_probabilities, tail_policy, margin
        ),
        "celsius_offset_probabilities": offset_probabilities,
        "market_bucket_probabilities_c": {str(k): float(v) for k, v in market.items()},
    }


def _decision(fields: Mapping[str, Any], thresholds: Mapping[str, float]) -> tuple[str, str]:
    if bool(fields["market_tail_ambiguity_c"]):
        return "no_trade", "tail_allocation_ambiguous"
    if float(fields["market_top_probability_c"]) < float(thresholds["minimum_top_probability"]):
        return "no_trade", "top_probability_below_threshold"
    if float(fields["market_top_two_margin_c"]) < float(thresholds["minimum_top_two_margin"]):
        return "no_trade", "top_two_margin_below_threshold"
    if (
        int(fields["recommended_bucket_c"]) != int(fields["point_bucket_c"])
        and float(fields["market_switch_advantage_c"])
        < float(thresholds["minimum_switch_advantage"])
    ):
        return "no_trade", "switch_advantage_below_threshold"
    return "shadow_trade", "confidence_passed"


def tune_celsius_decision_policy(
    predictions: pd.DataFrame, *, target_coverage: float = 0.60
) -> tuple[dict[str, float | bool], pd.DataFrame]:
    candidates: list[dict[str, Any]] = []
    for minimum_top in np.arange(0.20, 0.701, 0.025):
        for minimum_margin in np.arange(0.0, 0.301, 0.025):
            for minimum_advantage in np.arange(0.0, 0.201, 0.025):
                thresholds = {
                    "minimum_top_probability": float(minimum_top),
                    "minimum_top_two_margin": float(minimum_margin),
                    "minimum_switch_advantage": float(minimum_advantage),
                }
                decisions = predictions.apply(
                    lambda row: _decision(row, thresholds)[0], axis=1
                )
                actionable = decisions.eq("shadow_trade")
                if not actionable.any():
                    continue
                hits = predictions["recommended_bucket_c"].eq(
                    predictions["actual_bucket_c"]
                )
                switches = predictions["recommended_bucket_c"].ne(
                    predictions["point_bucket_c"]
                )
                candidates.append(
                    {
                        **thresholds,
                        "tail_ambiguity_rule_enabled": True,
                        "coverage": float(actionable.mean()),
                        "accuracy": float(hits[actionable].mean()),
                        "switch_count": int((actionable & switches).sum()),
                        "point_accuracy_on_actionable": float(
                            predictions.loc[actionable, "point_bucket_c"]
                            .eq(predictions.loc[actionable, "actual_bucket_c"])
                            .mean()
                        ),
                    }
                )
    frame = pd.DataFrame(candidates)
    if frame.empty:
        raise ValueError("unable to tune Celsius market decision policy")
    preferred = frame.loc[frame["coverage"].between(0.55, 0.65)].copy()
    if preferred.empty:
        preferred = frame.copy()
    preferred["coverage_distance"] = (preferred["coverage"] - target_coverage).abs()
    selected = preferred.sort_values(
        ["accuracy", "coverage_distance", "switch_count"],
        ascending=[False, True, False],
    ).iloc[0]
    thresholds: dict[str, float | bool] = {
        "minimum_top_probability": float(selected["minimum_top_probability"]),
        "minimum_top_two_margin": float(selected["minimum_top_two_margin"]),
        "minimum_switch_advantage": float(selected["minimum_switch_advantage"]),
        "tail_ambiguity_rule_enabled": True,
        "target_coverage": float(target_coverage),
    }
    return thresholds, frame


def _select_candidate(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_names: Sequence[str],
    candidates: Sequence[CelsiusCandidate],
    *,
    random_state: int,
    temperature_grid: Sequence[float] = tuple(np.linspace(0.5, 3.0, 11)),
) -> tuple[CelsiusCandidate, float, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    for candidate in candidates:
        state = _fit_ordinal(train, feature_names, candidate, random_state=random_state)
        raw = _predict_ordinal(state, valid, feature_names)
        states.append(state)
        for temperature in temperature_grid:
            score = score_offset_probabilities(
                valid["offset_class_c"].to_numpy(dtype=int),
                _temperature_scale(raw, float(temperature)),
            )
            rows.append(
                {
                    "candidate_index": len(states) - 1,
                    "C": float(candidate.c),
                    "class_weight": candidate.class_weight,
                    "temperature": float(temperature),
                    **score,
                }
            )
    tuning = pd.DataFrame(rows)
    best_loss = float(tuning["offset_log_loss"].min())
    near = tuning.loc[tuning["offset_log_loss"].le(best_loss + 1e-3)].copy()
    best_brier = float(near["offset_brier"].min())
    near = near.loc[near["offset_brier"].le(best_brier + 1e-3)].copy()
    near["balanced_rank"] = near["class_weight"].notna().astype(int)
    selected = near.sort_values(["balanced_rank", "C", "temperature"]).iloc[0]
    return (
        candidates[int(selected["candidate_index"])],
        float(selected["temperature"]),
        tuning,
    )


def _prediction_frame(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    tail_policy: Mapping[str, Any],
    *,
    thresholds: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (_, source), vector in zip(frame.iterrows(), probabilities, strict=True):
        fields = _market_fields(
            int(source["point_bucket_c"]), int(source["actual_bucket_c"]), vector, tail_policy
        )
        decision, reason = (
            _decision(fields, thresholds) if thresholds is not None else (None, None)
        )
        rows.append(
            {
                "contract_date": pd.Timestamp(source["contract_date"]),
                "actual_high_f": float(source["actual_high_f"]),
                "actual_high_c": float(source["actual_high_c"]),
                "actual_high_c_source": str(source["actual_high_c_source"]),
                "point_prediction_f": float(source["point_prediction_f"]),
                "point_prediction_c": float(source["point_prediction_c"]),
                "offset_c": int(source["offset_c"]),
                "offset_class_c": int(source["offset_class_c"]),
                **fields,
                "market_probability_decision": decision,
                "market_probability_decision_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def celsius_probability_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    offset_vectors = np.vstack(
        predictions["celsius_offset_probabilities"].map(
            lambda value: [float(value[label]) for label in OFFSET_LABELS_C]
        )
    )
    scores = score_offset_probabilities(
        predictions["offset_class_c"].to_numpy(dtype=int), offset_vectors
    )
    market_brier = predictions.apply(
        lambda row: sum(
            (
                float(probability)
                - (1.0 if int(bucket) == int(row["actual_bucket_c"]) else 0.0)
            )
            ** 2
            for bucket, probability in row["market_bucket_probabilities_c"].items()
        ),
        axis=1,
    )
    decided = predictions.get(
        "market_probability_decision", pd.Series(index=predictions.index, dtype=object)
    ).eq("shadow_trade")
    return pd.DataFrame(
        [
            {
                **scores,
                "market_bucket_accuracy": float(
                    predictions["recommended_bucket_c"].eq(predictions["actual_bucket_c"]).mean()
                ),
                "point_bucket_accuracy": float(
                    predictions["point_bucket_c"].eq(predictions["actual_bucket_c"]).mean()
                ),
                "market_bucket_log_loss": float(
                    -np.log(
                        predictions["actual_bucket_probability_c"].clip(lower=1e-12)
                    ).mean()
                ),
                "market_bucket_brier": float(market_brier.mean()),
                "decision_coverage": float(decided.mean()),
                "decision_count": int(decided.sum()),
                "decision_accuracy": (
                    float(
                        predictions.loc[decided, "recommended_bucket_c"]
                        .eq(predictions.loc[decided, "actual_bucket_c"])
                        .mean()
                    )
                    if decided.any()
                    else math.nan
                ),
            }
        ]
    )


def celsius_calibration_table(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame["confidence_bin"] = pd.cut(
        frame["market_top_probability_c"],
        bins=np.linspace(0.0, 1.0, 11),
        include_lowest=True,
    )
    frame["correct"] = frame["recommended_bucket_c"].eq(frame["actual_bucket_c"])
    return (
        frame.groupby("confidence_bin", observed=False)
        .agg(
            count=("correct", "size"),
            mean_predicted_probability=("market_top_probability_c", "mean"),
            realized_accuracy=("correct", "mean"),
        )
        .reset_index()
        .assign(confidence_bin=lambda value: value["confidence_bin"].astype(str))
    )


def fit_celsius_probability_system(
    frame: pd.DataFrame,
    *,
    station_id: str,
    point_model_version: str,
    point_bundle_sha256: str,
    feature_profile: str,
    model_version: str,
    development_years: Sequence[int] = (2024, 2025),
    forward_validation_years: Sequence[int] = (2025,),
    calibration_days: int = 90,
    min_train_rows: int = 180,
    random_state: int = 42,
    candidates: Sequence[CelsiusCandidate] | None = None,
    temperature_grid: Sequence[float] = tuple(np.linspace(0.5, 3.0, 11)),
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    feature_names = probability_feature_names(
        include_peak_features=feature_profile == FEATURE_PROFILE_PEAK_AUGMENTED,
        feature_profile=feature_profile,
    )
    development_years = tuple(dict.fromkeys(int(year) for year in development_years))
    validation_years = tuple(dict.fromkeys(int(year) for year in forward_validation_years))
    if not set(validation_years).issubset(development_years):
        raise ValueError("forward validation years must be development years")
    development = frame.loc[frame["year"].isin(development_years)].copy()
    if len(development) < min_train_rows:
        raise ValueError("insufficient Celsius probability training rows")
    candidates = tuple(candidates or default_celsius_candidates())
    forward_parts: list[pd.DataFrame] = []
    tuning_parts: list[pd.DataFrame] = []
    for validation_year in validation_years:
        outer_train = development.loc[development["year"].lt(validation_year)].copy()
        outer_valid = development.loc[development["year"].eq(validation_year)].copy()
        if len(outer_train) < min_train_rows or outer_valid.empty:
            continue
        split = outer_train["contract_date"].max() - pd.Timedelta(days=calibration_days - 1)
        inner_train = outer_train.loc[outer_train["contract_date"].lt(split)].copy()
        inner_valid = outer_train.loc[outer_train["contract_date"].ge(split)].copy()
        if inner_train.empty or inner_valid.empty:
            raise ValueError("insufficient inner chronological calibration history")
        if not (
            inner_train["contract_date"].max() < inner_valid["contract_date"].min()
            and inner_valid["contract_date"].max() < outer_valid["contract_date"].min()
        ):
            raise AssertionError("Celsius probability chronology is invalid")
        candidate, temperature, tuning = _select_candidate(
            inner_train,
            inner_valid,
            feature_names,
            candidates,
            random_state=random_state,
            temperature_grid=temperature_grid,
        )
        state = _fit_ordinal(outer_train, feature_names, candidate, random_state=random_state)
        probabilities = _temperature_scale(
            _predict_ordinal(state, outer_valid, feature_names), temperature
        )
        predictions = _prediction_frame(
            outer_valid, probabilities, fit_tail_policy_c(outer_train["offset_c"])
        )
        predictions["validation_year"] = int(validation_year)
        predictions["model_training_cutoff"] = outer_train["contract_date"].max()
        predictions["calibration_training_cutoff"] = inner_train["contract_date"].max()
        predictions["calibration_validation_start"] = inner_valid["contract_date"].min()
        predictions["calibration_validation_cutoff"] = inner_valid["contract_date"].max()
        forward_parts.append(predictions)
        tuning["validation_year"] = int(validation_year)
        tuning_parts.append(tuning)
    if not forward_parts:
        raise ValueError("no Celsius forward-validation folds were produced")
    forward = pd.concat(forward_parts, ignore_index=True)
    thresholds, policy_tuning = tune_celsius_decision_policy(forward)
    decisions = forward.apply(lambda row: _decision(row, thresholds), axis=1)
    forward["market_probability_decision"] = decisions.map(lambda value: value[0])
    forward["market_probability_decision_reason"] = decisions.map(lambda value: value[1])

    split = development["contract_date"].max() - pd.Timedelta(days=calibration_days - 1)
    inner_train = development.loc[development["contract_date"].lt(split)].copy()
    inner_valid = development.loc[development["contract_date"].ge(split)].copy()
    candidate, temperature, final_tuning = _select_candidate(
        inner_train,
        inner_valid,
        feature_names,
        candidates,
        random_state=random_state,
        temperature_grid=temperature_grid,
    )
    state = _fit_ordinal(development, feature_names, candidate, random_state=random_state)
    tail_policy = fit_tail_policy_c(development["offset_c"])
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "station_id": station_id.strip().upper(),
        "model_version": model_version,
        "point_model_version": point_model_version,
        "point_bundle_sha256": point_bundle_sha256.lower(),
        "feature_profile": feature_profile,
        "base_methods": list(probability_base_methods(feature_profile)),
        "feature_names": feature_names,
        "mandatory_source_features": list(probability_mandatory_feature_names(feature_profile)),
        "selected_family": MODEL_FAMILY,
        "selected_params": {"C": candidate.c, "class_weight": candidate.class_weight},
        "model_state": state,
        "temperature": float(temperature),
        "offset_labels_c": list(OFFSET_LABELS_C),
        "offset_class_contract": "<=-3, -2, -1, 0, +1, +2, >=+3 Celsius degrees",
        "target_contract": TARGET_CONTRACT,
        "market_bucket_contract": "Tokyo Polymarket whole 1C integer buckets",
        "actual_celsius_source_priority": [
            "actual_high_c",
            "settlement_high_c",
            "actual_high_f_converted_to_c",
        ],
        "tail_policy": tail_policy,
        "decision_thresholds": thresholds,
        "policy_selection_data": "pre-2026 forward validation only",
        "training_start": development["contract_date"].min().date().isoformat(),
        "training_cutoff": development["contract_date"].max().date().isoformat(),
        "training_rows": int(len(development)),
        "final_calibration_training_cutoff": inner_train["contract_date"].max().date().isoformat(),
        "final_calibration_validation_start": inner_valid["contract_date"].min().date().isoformat(),
        "final_calibration_validation_cutoff": inner_valid["contract_date"].max().date().isoformat(),
        "development_years": list(development_years),
        "forward_validation_years": list(validation_years),
        "forward_validation_start": forward["contract_date"].min().date().isoformat(),
        "forward_validation_cutoff": forward["contract_date"].max().date().isoformat(),
        "holdout_year": 2026,
        "selection_excludes_holdout": True,
        "forward_metrics": celsius_probability_metrics(forward).iloc[0].to_dict(),
        "package_versions": _package_versions(),
        "holdout_status": "exploratory_shadow_only",
        "overrides_enabled": True,
    }
    tuning = pd.concat(tuning_parts, ignore_index=True)
    final_tuning["validation_year"] = "final_pre_2026_calibration"
    policy_tuning["validation_year"] = "pre_2026_policy"
    tuning = pd.concat([tuning, final_tuning, policy_tuning], ignore_index=True, sort=False)
    return bundle, forward, tuning


def predict_celsius_probability_bundle(
    bundle: Mapping[str, Any], feature_values: Mapping[str, Any]
) -> dict[str, Any]:
    missing = [
        name
        for name in bundle["mandatory_source_features"]
        if _finite_number(feature_values.get(name)) is None
    ]
    base_methods = tuple(
        bundle.get("base_methods")
        or probability_base_methods(str(bundle["feature_profile"]))
    )
    for name in (
        "point_prediction_f",
        *(f"{method}_predicted_high_f" for method in base_methods),
    ):
        if _finite_number(feature_values.get(name)) is None:
            missing.append(name)
    if missing:
        return {
            "status": "unavailable",
            "reason": "missing_required_features:" + ",".join(sorted(set(missing))),
        }
    frame = add_probability_features(
        pd.DataFrame([dict(feature_values)]),
        providers=probability_provider_names(str(bundle["feature_profile"])),
        base_methods=base_methods,
    )
    for name in bundle["feature_names"]:
        if name.endswith(MISSING_INDICATOR_SUFFIX):
            source_name = name[: -len(MISSING_INDICATOR_SUFFIX)]
            frame[name] = (
                frame[source_name].isna().astype(float)
                if source_name in frame
                else 1.0
            )
        elif name not in frame:
            frame[name] = np.nan
    vector = _temperature_scale(
        _predict_ordinal(bundle["model_state"], frame, bundle["feature_names"]),
        float(bundle["temperature"]),
    )[0]
    point_prediction_f = float(feature_values["point_prediction_f"])
    point_prediction_c = fahrenheit_to_celsius(point_prediction_f)
    fields = _market_fields(
        round_half_up(point_prediction_c),
        int(feature_values["actual_bucket_c"])
        if feature_values.get("actual_bucket_c") is not None
        else None,
        vector,
        bundle["tail_policy"],
    )
    decision, reason = _decision(fields, bundle["decision_thresholds"])
    return {
        "status": "ok",
        "model_version": bundle["model_version"],
        "point_prediction_c": point_prediction_c,
        **fields,
        "market_probability_decision": decision,
        "market_probability_decision_reason": reason,
        "probability_thresholds": dict(bundle["decision_thresholds"]),
    }


def evaluate_celsius_probability_holdout(
    feature_frame: pd.DataFrame,
    point_predictions: pd.DataFrame,
    base_predictions: pd.DataFrame,
    bundle: Mapping[str, Any],
    *,
    holdout_year: int = 2026,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = build_celsius_probability_frame(
        feature_frame,
        point_predictions,
        base_predictions,
        include_peak_features=bundle["feature_profile"] == FEATURE_PROFILE_PEAK_AUGMENTED,
        feature_profile=str(bundle["feature_profile"]),
    )
    frame = frame.loc[frame["year"].eq(int(holdout_year))].copy()
    probabilities = _temperature_scale(
        _predict_ordinal(bundle["model_state"], frame, bundle["feature_names"]),
        float(bundle["temperature"]),
    )
    predictions = _prediction_frame(
        frame, probabilities, bundle["tail_policy"], thresholds=bundle["decision_thresholds"]
    )
    return predictions, celsius_probability_metrics(predictions), celsius_calibration_table(predictions)


def export_celsius_probability_bundle(
    bundle: Mapping[str, Any],
    output_dir: Path | str,
    *,
    source_identity: Mapping[str, Any],
    artifact_paths: Sequence[Path | str] = (),
) -> tuple[Path, Path]:
    import joblib

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = f"{bundle['station_id']}_{bundle['model_version']}"
    bundle_path = output / f"{stem}.joblib"
    manifest_path = output / f"{stem}.json"
    joblib.dump(dict(bundle), bundle_path)
    hashes = {
        Path(path).name: sha256_file(Path(path))
        for path in artifact_paths
        if Path(path).is_file()
    }
    manifest = {
        key: bundle[key]
        for key in (
            "schema_version",
            "artifact_type",
            "station_id",
            "model_version",
            "point_model_version",
            "point_bundle_sha256",
            "feature_profile",
            "feature_names",
            "mandatory_source_features",
            "selected_family",
            "selected_params",
            "temperature",
            "offset_labels_c",
            "offset_class_contract",
            "target_contract",
            "market_bucket_contract",
            "actual_celsius_source_priority",
            "tail_policy",
            "decision_thresholds",
            "policy_selection_data",
            "training_start",
            "training_cutoff",
            "training_rows",
            "final_calibration_training_cutoff",
            "final_calibration_validation_start",
            "final_calibration_validation_cutoff",
            "development_years",
            "forward_validation_years",
            "forward_validation_start",
            "forward_validation_cutoff",
            "holdout_year",
            "selection_excludes_holdout",
            "forward_metrics",
            "package_versions",
            "holdout_status",
        )
    }
    manifest["base_methods"] = list(
        bundle.get("base_methods")
        or probability_base_methods(str(bundle["feature_profile"]))
    )
    manifest["holdout_metrics"] = dict(bundle.get("holdout_metrics", {}))
    manifest["source_identity"] = dict(source_identity)
    manifest["artifact_integrity"] = {
        "bundle_sha256": sha256_file(bundle_path),
        "artifact_sha256": hashes,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return bundle_path, manifest_path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_versions() -> dict[str, str]:
    return {
        package: importlib.metadata.version(distribution)
        for package, distribution in {
            "pandas": "pandas",
            "numpy": "numpy",
            "scikit-learn": "scikit-learn",
            "joblib": "joblib",
        }.items()
    }


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
