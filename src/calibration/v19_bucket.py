from __future__ import annotations

from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
import json
import math
import re
from typing import Iterable

import numpy as np
import pandas as pd

from .station_stacking import (
    STACK_METHOD,
    TARGET,
    _stack_features_for_set,
    _year_split_stack_source_frame,
)


def feature_missingness_audit(
    frame: pd.DataFrame,
    categorical: Iterable[str],
    numeric: Iterable[str],
    *,
    train_years: tuple[int, int] = (2021, 2025),
    max_missing_fraction: float = 0.03,
) -> pd.DataFrame:
    """Report the station-specific, training-only V19 feature gate."""
    years = pd.to_numeric(frame.get("year"), errors="coerce")
    train = frame.loc[years.between(*train_years)].copy() if years.notna().any() else frame.copy()
    rows: list[dict[str, object]] = []
    for kind, columns in (("categorical", categorical), ("numeric", numeric)):
        for column in columns:
            if column not in train:
                missing_fraction = 1.0
                non_null_rows = 0
            else:
                values = pd.to_numeric(train[column], errors="coerce") if kind == "numeric" else train[column]
                missing_fraction = float(values.isna().mean()) if len(values) else 1.0
                non_null_rows = int(values.notna().sum())
            rows.append(
                {
                    "feature": column,
                    "kind": kind,
                    "train_rows": int(len(train)),
                    "non_null_train_rows": non_null_rows,
                    "missing_fraction": missing_fraction,
                    "missing_pct": missing_fraction * 100.0,
                    "keep_v19": bool(missing_fraction <= max_missing_fraction and non_null_rows > 0),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["keep_v19", "missing_fraction", "feature"],
        ascending=[True, False, True],
        ignore_index=True,
    )


def crossfit_ridge_predictions(
    validation_predictions: pd.DataFrame,
    stack_tuning: pd.DataFrame | None = None,
    *,
    base_model_methods: tuple[str, ...] = ("xgboost", "lightgbm", "catboost"),
    providers: tuple[str, ...] = ("gfs", "hrrr", "nbm"),
    min_train_rows: int = 60,
    alpha_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0),
) -> pd.DataFrame:
    """Strictly nested cross-fit of the ridge stack across validation years.

    Base predictions must already be forward/out-of-fold. The ridge for each
    validation year is fitted only on earlier validation years. Ridge feature
    set and alpha are selected using an inner split within those earlier years;
    the first usable year falls back to fixed, predeclared parameters.

    ``stack_tuning`` is accepted for backward compatibility but intentionally
    ignored so global pre-2026 tuning cannot leak into earlier cross-fit years.
    """
    columns = [
        "contract_date",
        TARGET,
        "predicted_high_f",
        "residual_f",
        "train_through_year",
        "validation_year",
        "ridge_feature_set",
        "ridge_alpha",
        "inner_validation_year",
    ]
    if validation_predictions.empty:
        return pd.DataFrame(columns=columns)
    stack_methods = [*base_model_methods, *(f"{provider}_raw" for provider in providers)]
    source = _year_split_stack_source_frame(validation_predictions, stack_methods)
    if source.empty:
        return pd.DataFrame(columns=columns)
    source["contract_date"] = pd.to_datetime(source["contract_date"], errors="coerce")
    source["validation_year"] = source["contract_date"].dt.year
    source = source.dropna(subset=["contract_date", "validation_year", TARGET])
    if source.empty:
        return pd.DataFrame(columns=columns)

    from sklearn.linear_model import Ridge

    rows: list[pd.DataFrame] = []
    for validation_year in sorted(source["validation_year"].astype(int).unique()):
        train = source.loc[source["validation_year"].lt(validation_year)].copy()
        valid = source.loc[source["validation_year"].eq(validation_year)].copy()
        if len(train) < min_train_rows or valid.empty:
            continue
        feature_set, alpha, inner_validation_year = _select_nested_ridge_parameters(
            train,
            base_model_methods=base_model_methods,
            providers=providers,
            alpha_grid=alpha_grid,
            min_train_rows=min_train_rows,
        )
        stack_features = _stack_features_for_set(feature_set, base_model_methods, providers)
        train = train.dropna(subset=[*stack_features, TARGET])
        valid = valid.dropna(subset=[*stack_features, TARGET])
        if len(train) < min_train_rows or valid.empty:
            continue
        model = Ridge(
            alpha=alpha,
            fit_intercept=True,
        )
        model.fit(train[stack_features], train[TARGET])
        out = valid[["contract_date", TARGET]].copy()
        out["predicted_high_f"] = model.predict(valid[stack_features])
        out["residual_f"] = out[TARGET] - out["predicted_high_f"]
        out["train_through_year"] = int(validation_year) - 1
        out["validation_year"] = int(validation_year)
        out["ridge_feature_set"] = feature_set
        out["ridge_alpha"] = alpha
        out["inner_validation_year"] = inner_validation_year
        rows.append(out[columns])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=columns)


def _select_nested_ridge_parameters(
    history: pd.DataFrame,
    *,
    base_model_methods: tuple[str, ...],
    providers: tuple[str, ...],
    alpha_grid: tuple[float, ...],
    min_train_rows: int,
) -> tuple[str, float, int | None]:
    years = sorted(pd.to_numeric(history["validation_year"], errors="coerce").dropna().astype(int).unique())
    if len(years) < 2:
        return "models_plus_raw", 1.0, None
    inner_validation_year = years[-1]
    inner_train = history.loc[history["validation_year"].lt(inner_validation_year)].copy()
    inner_valid = history.loc[history["validation_year"].eq(inner_validation_year)].copy()
    if len(inner_train) < min_train_rows or inner_valid.empty:
        return "models_plus_raw", 1.0, inner_validation_year

    from sklearn.linear_model import Ridge

    candidates: list[tuple[float, str, float]] = []
    for feature_set in ("models_only", "models_plus_raw"):
        features = _stack_features_for_set(feature_set, base_model_methods, providers)
        train = inner_train.dropna(subset=[*features, TARGET])
        valid = inner_valid.dropna(subset=[*features, TARGET])
        if len(train) < min_train_rows or valid.empty:
            continue
        for alpha in alpha_grid:
            model = Ridge(alpha=float(alpha), fit_intercept=True)
            model.fit(train[features], train[TARGET])
            mae = float(np.abs(valid[TARGET].to_numpy(dtype=float) - model.predict(valid[features])).mean())
            candidates.append((mae, feature_set, float(alpha)))
    if not candidates:
        return "models_plus_raw", 1.0, inner_validation_year
    _, feature_set, alpha = min(candidates, key=lambda item: (item[0], item[1], item[2]))
    return feature_set, alpha, inner_validation_year


def empirical_modal_bucket_decisions(
    test_predictions: pd.DataFrame,
    residual_predictions: pd.DataFrame,
    *,
    method: str = STACK_METHOD,
    monthly_shrinkage: float = 60.0,
    epsilon: float = 1e-12,
) -> pd.DataFrame:
    """Turn a point forecast into train-only empirical bucket probabilities."""
    columns = [
        "contract_date",
        TARGET,
        "predicted_high_f",
        "actual_bucket",
        "point_bucket",
        "modal_bucket",
        "point_bucket_hit",
        "modal_bucket_hit",
        "top_bucket_probability",
        "second_bucket_probability",
        "probability_margin",
        "actual_bucket_probability",
        "bucket_log_loss",
        "month_residual_count",
        "monthly_weight",
        "bucket_probabilities_json",
    ]
    if test_predictions.empty or residual_predictions.empty:
        return pd.DataFrame(columns=columns)
    test = test_predictions.loc[test_predictions["method"].eq(method)].copy()
    if test.empty:
        return pd.DataFrame(columns=columns)
    residuals = residual_predictions.copy()
    residuals["residual_f"] = pd.to_numeric(residuals["residual_f"], errors="coerce")
    residuals["month"] = pd.to_datetime(residuals["contract_date"], errors="coerce").dt.month
    residuals = residuals.dropna(subset=["residual_f", "month"])
    global_residuals = residuals["residual_f"].to_numpy(dtype=float)
    if not len(global_residuals):
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for _, row in test.iterrows():
        predicted = _finite_float(row.get("predicted_high_f"))
        actual = _finite_float(row.get(TARGET))
        date = pd.to_datetime(row.get("contract_date"), errors="coerce")
        if predicted is None or actual is None or pd.isna(date):
            continue
        month_values = residuals.loc[residuals["month"].eq(int(date.month)), "residual_f"].to_numpy(dtype=float)
        monthly_weight = float(len(month_values) / (len(month_values) + monthly_shrinkage)) if len(month_values) else 0.0
        global_probs = _bucket_counts(predicted, global_residuals)
        month_probs = _bucket_counts(predicted, month_values) if len(month_values) else {}
        labels = set(global_probs) | set(month_probs)
        probabilities = {
            label: (1.0 - monthly_weight) * global_probs.get(label, 0.0)
            + monthly_weight * month_probs.get(label, 0.0)
            for label in labels
        }
        total = sum(probabilities.values())
        probabilities = {label: value / total for label, value in probabilities.items()} if total else global_probs
        ranked = sorted(
            probabilities.items(),
            key=lambda item: (-item[1], abs(_bucket_center(item[0]) - predicted), item[0]),
        )
        modal_bucket, top_probability = ranked[0]
        second_probability = ranked[1][1] if len(ranked) > 1 else 0.0
        actual_bucket = temperature_bucket_label(actual)
        point_bucket = temperature_bucket_label(predicted)
        actual_probability = float(probabilities.get(actual_bucket, 0.0))
        rows.append(
            {
                "contract_date": date.date().isoformat(),
                TARGET: actual,
                "predicted_high_f": predicted,
                "actual_bucket": actual_bucket,
                "point_bucket": point_bucket,
                "modal_bucket": modal_bucket,
                "point_bucket_hit": point_bucket == actual_bucket,
                "modal_bucket_hit": modal_bucket == actual_bucket,
                "top_bucket_probability": float(top_probability),
                "second_bucket_probability": float(second_probability),
                "probability_margin": float(top_probability - second_probability),
                "actual_bucket_probability": actual_probability,
                "bucket_log_loss": float(-math.log(max(epsilon, actual_probability))),
                "month_residual_count": int(len(month_values)),
                "monthly_weight": monthly_weight,
                "bucket_probabilities_json": json.dumps(dict(sorted(probabilities.items())), sort_keys=True),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def bucket_decision_metrics(decisions: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty:
        return pd.DataFrame(columns=["decision", "count", "bucket_accuracy_pct", "bucket_log_loss"])
    count = int(len(decisions))
    return pd.DataFrame(
        [
            {
                "decision": "rounded_point",
                "count": count,
                "bucket_accuracy_pct": float(decisions["point_bucket_hit"].mean() * 100.0),
                "bucket_log_loss": np.nan,
            },
            {
                "decision": "empirical_modal",
                "count": count,
                "bucket_accuracy_pct": float(decisions["modal_bucket_hit"].mean() * 100.0),
                "bucket_log_loss": float(decisions["bucket_log_loss"].mean()),
            },
        ]
    )


def ordinal_blend_bucket_decisions(
    validation_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    residual_predictions: pd.DataFrame,
    *,
    base_model_methods: tuple[str, ...] = ("xgboost", "lightgbm", "catboost"),
    providers: tuple[str, ...] = ("gfs", "hrrr", "nbm"),
    monthly_shrinkage: float = 60.0,
    blend_weights: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    min_train_rows: int = 60,
    random_state: int = 42,
    epsilon: float = 1e-12,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Fit a cross-validated ordinal bucket-offset classifier and blend it.

    Blend weights are chosen on forward classifier OOF rows only. A weight of
    zero is the empirical V19-B distribution and one is the ordinal model.
    """
    train_meta = _ordinal_meta_frame(
        validation_predictions,
        residual_predictions,
        base_model_methods=base_model_methods,
        providers=providers,
    )
    decision_columns = [
        "contract_date",
        TARGET,
        "predicted_high_f",
        "actual_bucket",
        "empirical_bucket",
        "ordinal_bucket",
        "blended_bucket",
        "empirical_bucket_hit",
        "ordinal_bucket_hit",
        "blended_bucket_hit",
        "selected_ordinal_weight",
        "top_bucket_probability",
        "second_bucket_probability",
        "probability_margin",
        "actual_bucket_probability",
        "bucket_log_loss",
        "bucket_probabilities_json",
    ]
    tuning_columns = ["ordinal_weight", "count", "bucket_log_loss", "bucket_accuracy_pct"]
    if train_meta.empty:
        return pd.DataFrame(columns=decision_columns), pd.DataFrame(columns=tuning_columns), {}
    feature_names = _ordinal_feature_columns(train_meta, base_model_methods, providers)
    classifier_oof = _crossfit_ordinal_probabilities(
        train_meta,
        feature_names,
        min_train_rows=min_train_rows,
        random_state=random_state,
    )
    tuning = _tune_ordinal_blend(
        classifier_oof,
        residual_predictions,
        blend_weights=blend_weights,
        monthly_shrinkage=monthly_shrinkage,
        epsilon=epsilon,
    )
    if tuning.empty:
        return pd.DataFrame(columns=decision_columns), tuning, {"feature_names": feature_names}
    selected_weight = float(tuning.sort_values(["bucket_log_loss", "bucket_accuracy_pct", "ordinal_weight"], ascending=[True, False, True]).iloc[0]["ordinal_weight"])

    classifier = _fit_ordinal_classifier(train_meta, feature_names, random_state=random_state)
    test_meta = _ordinal_test_meta_frame(
        test_predictions,
        base_model_methods=base_model_methods,
        providers=providers,
    )
    if classifier is None or test_meta.empty:
        return pd.DataFrame(columns=decision_columns), tuning, {"feature_names": feature_names}
    class_probabilities = classifier.predict_proba(test_meta[feature_names])
    classes = classifier.classes_.astype(int)
    residual_history = residual_predictions.copy()
    residual_history["residual_f"] = pd.to_numeric(residual_history["residual_f"], errors="coerce")
    residual_history["month"] = pd.to_datetime(residual_history["contract_date"], errors="coerce").dt.month
    residual_history = residual_history.dropna(subset=["residual_f", "month"])

    rows: list[dict[str, object]] = []
    for position, (_, row) in enumerate(test_meta.iterrows()):
        predicted = float(row["predicted_high_f"])
        actual = float(row[TARGET])
        date = pd.Timestamp(row["contract_date"])
        empirical = _shrunk_empirical_probabilities(
            predicted,
            residual_history,
            month=int(date.month),
            monthly_shrinkage=monthly_shrinkage,
        )
        ordinal = _offset_probabilities_to_buckets(
            predicted,
            classes,
            class_probabilities[position],
            empirical_reference=empirical,
        )
        blended = _blend_probabilities(empirical, ordinal, selected_weight)
        empirical_bucket = _top_bucket(empirical, predicted)
        ordinal_bucket = _top_bucket(ordinal, predicted)
        blended_bucket = _top_bucket(blended, predicted)
        ranked = sorted(blended.items(), key=lambda item: (-item[1], abs(_bucket_center(item[0]) - predicted), item[0]))
        actual_bucket = temperature_bucket_label(actual)
        actual_probability = float(blended.get(actual_bucket, 0.0))
        rows.append(
            {
                "contract_date": date.date().isoformat(),
                TARGET: actual,
                "predicted_high_f": predicted,
                "actual_bucket": actual_bucket,
                "empirical_bucket": empirical_bucket,
                "ordinal_bucket": ordinal_bucket,
                "blended_bucket": blended_bucket,
                "empirical_bucket_hit": empirical_bucket == actual_bucket,
                "ordinal_bucket_hit": ordinal_bucket == actual_bucket,
                "blended_bucket_hit": blended_bucket == actual_bucket,
                "selected_ordinal_weight": selected_weight,
                "top_bucket_probability": float(ranked[0][1]),
                "second_bucket_probability": float(ranked[1][1]) if len(ranked) > 1 else 0.0,
                "probability_margin": float(ranked[0][1] - ranked[1][1]) if len(ranked) > 1 else float(ranked[0][1]),
                "actual_bucket_probability": actual_probability,
                "bucket_log_loss": float(-math.log(max(epsilon, actual_probability))),
                "bucket_probabilities_json": json.dumps(dict(sorted(blended.items())), sort_keys=True),
            }
        )
    metadata = {
        "feature_names": feature_names,
        "classifier_type": "cumulative_threshold_ordinal_logistic",
        "tail_policy": "censored_endpoint_mass_distributed_by_empirical_tail_shape",
        "tuning_policy": "strict_forward_nested",
        "selected_ordinal_weight": selected_weight,
        "train_rows": int(len(train_meta)),
        "classifier_oof_rows": int(len(classifier_oof)),
        "classes": classes.tolist(),
    }
    return pd.DataFrame(rows, columns=decision_columns), tuning[tuning_columns], metadata


def ordinal_blend_metrics(decisions: pd.DataFrame) -> pd.DataFrame:
    columns = ["decision", "count", "bucket_accuracy_pct", "bucket_log_loss"]
    if decisions.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for decision, hit_column in (
        ("empirical_modal", "empirical_bucket_hit"),
        ("ordinal_only", "ordinal_bucket_hit"),
        ("empirical_ordinal_blend", "blended_bucket_hit"),
    ):
        rows.append(
            {
                "decision": decision,
                "count": int(len(decisions)),
                "bucket_accuracy_pct": float(decisions[hit_column].mean() * 100.0),
                "bucket_log_loss": float(decisions["bucket_log_loss"].mean()) if decision == "empirical_ordinal_blend" else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _ordinal_meta_frame(
    validation_predictions: pd.DataFrame,
    residual_predictions: pd.DataFrame,
    *,
    base_model_methods: tuple[str, ...],
    providers: tuple[str, ...],
) -> pd.DataFrame:
    methods = [*base_model_methods, *(f"{provider}_raw" for provider in providers)]
    source = _year_split_stack_source_frame(validation_predictions, methods)
    if source.empty or residual_predictions.empty:
        return pd.DataFrame()
    point = residual_predictions[["contract_date", TARGET, "predicted_high_f", "validation_year"]].copy()
    source["contract_date"] = pd.to_datetime(source["contract_date"], errors="coerce")
    point["contract_date"] = pd.to_datetime(point["contract_date"], errors="coerce")
    source = source.drop(columns=[TARGET], errors="ignore")
    out = point.merge(source, on="contract_date", how="inner")
    return _add_ordinal_features(out, base_model_methods, providers)


def _ordinal_test_meta_frame(
    test_predictions: pd.DataFrame,
    *,
    base_model_methods: tuple[str, ...],
    providers: tuple[str, ...],
) -> pd.DataFrame:
    methods = [*base_model_methods, *(f"{provider}_raw" for provider in providers)]
    source = _year_split_stack_source_frame(test_predictions, methods)
    point = test_predictions.loc[
        test_predictions["method"].eq(STACK_METHOD),
        ["contract_date", TARGET, "predicted_high_f"],
    ].copy()
    if source.empty or point.empty:
        return pd.DataFrame()
    source["contract_date"] = pd.to_datetime(source["contract_date"], errors="coerce")
    point["contract_date"] = pd.to_datetime(point["contract_date"], errors="coerce")
    source = source.drop(columns=[TARGET], errors="ignore")
    out = point.merge(source, on="contract_date", how="inner")
    return _add_ordinal_features(out, base_model_methods, providers)


def _add_ordinal_features(
    frame: pd.DataFrame,
    base_model_methods: tuple[str, ...],
    providers: tuple[str, ...],
) -> pd.DataFrame:
    out = frame.copy()
    date = pd.to_datetime(out["contract_date"], errors="coerce")
    out["month_sin"] = np.sin(2.0 * np.pi * date.dt.month / 12.0)
    out["month_cos"] = np.cos(2.0 * np.pi * date.dt.month / 12.0)
    model_columns = [f"{method}_predicted_high_f" for method in base_model_methods if f"{method}_predicted_high_f" in out]
    provider_columns = [f"{provider}_raw_predicted_high_f" for provider in providers if f"{provider}_raw_predicted_high_f" in out]
    out["base_model_spread_f"] = out[model_columns].max(axis=1) - out[model_columns].min(axis=1)
    out["provider_spread_f"] = out[provider_columns].max(axis=1) - out[provider_columns].min(axis=1)
    out["point_minus_bucket_center_f"] = out["predicted_high_f"].map(
        lambda value: float(value) - _bucket_center(temperature_bucket_label(float(value)))
    )
    if TARGET in out:
        out["bucket_offset_class"] = [
            int(np.clip(_bucket_index(actual) - _bucket_index(predicted), -2, 2))
            for actual, predicted in zip(out[TARGET], out["predicted_high_f"], strict=False)
        ]
    return out


def _ordinal_feature_columns(
    frame: pd.DataFrame,
    base_model_methods: tuple[str, ...],
    providers: tuple[str, ...],
) -> list[str]:
    candidates = [
        "predicted_high_f",
        *(f"{method}_predicted_high_f" for method in base_model_methods),
        *(f"{provider}_raw_predicted_high_f" for provider in providers),
        "base_model_spread_f",
        "provider_spread_f",
        "point_minus_bucket_center_f",
        "month_sin",
        "month_cos",
    ]
    return [column for column in candidates if column in frame]


class CumulativeOrdinalClassifier:
    """Cumulative-threshold ordinal logistic model for offsets −2..+2."""

    def __init__(self, *, c: float = 0.3, max_iter: int = 2000, random_state: int = 42):
        self.c = float(c)
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)
        self.classes_ = np.asarray([-2, -1, 0, 1, 2], dtype=int)
        self._threshold_models: list[object] = []

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "CumulativeOrdinalClassifier":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        target = pd.to_numeric(y, errors="raise").astype(int).to_numpy()
        self._threshold_models = []
        for threshold in self.classes_[:-1]:
            binary = (target > threshold).astype(int)
            if np.unique(binary).size == 1:
                self._threshold_models.append(float(binary[0]))
                continue
            model = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            C=self.c,
                            max_iter=self.max_iter,
                            random_state=self.random_state,
                        ),
                    ),
                ]
            )
            model.fit(x, binary)
            self._threshold_models.append(model)
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        if len(self._threshold_models) != len(self.classes_) - 1:
            raise ValueError("CumulativeOrdinalClassifier must be fitted before prediction")
        exceedance_columns: list[np.ndarray] = []
        for model in self._threshold_models:
            if isinstance(model, float):
                exceedance_columns.append(np.full(len(x), model, dtype=float))
            else:
                exceedance_columns.append(np.asarray(model.predict_proba(x)[:, 1], dtype=float))
        exceedance = np.column_stack(exceedance_columns)
        exceedance = np.minimum.accumulate(exceedance, axis=1)
        probabilities = np.column_stack(
            [
                1.0 - exceedance[:, 0],
                exceedance[:, 0] - exceedance[:, 1],
                exceedance[:, 1] - exceedance[:, 2],
                exceedance[:, 2] - exceedance[:, 3],
                exceedance[:, 3],
            ]
        )
        probabilities = np.clip(probabilities, 0.0, 1.0)
        totals = probabilities.sum(axis=1, keepdims=True)
        return np.divide(probabilities, totals, out=np.zeros_like(probabilities), where=totals > 0)


def _fit_ordinal_classifier(frame: pd.DataFrame, feature_names: list[str], *, random_state: int):
    if frame.empty or not feature_names or frame["bucket_offset_class"].nunique() < 2:
        return None
    classifier = CumulativeOrdinalClassifier(c=0.3, max_iter=2000, random_state=random_state)
    classifier.fit(frame[feature_names], frame["bucket_offset_class"].astype(int))
    return classifier


def _crossfit_ordinal_probabilities(
    frame: pd.DataFrame,
    feature_names: list[str],
    *,
    min_train_rows: int,
    random_state: int,
) -> pd.DataFrame:
    rows = []
    for validation_year in sorted(pd.to_numeric(frame["validation_year"], errors="coerce").dropna().astype(int).unique()):
        train = frame.loc[pd.to_numeric(frame["validation_year"], errors="coerce").lt(validation_year)].copy()
        valid = frame.loc[pd.to_numeric(frame["validation_year"], errors="coerce").eq(validation_year)].copy()
        if len(train) < min_train_rows or valid.empty:
            continue
        classifier = _fit_ordinal_classifier(train, feature_names, random_state=random_state)
        if classifier is None:
            continue
        probabilities = classifier.predict_proba(valid[feature_names])
        classes = classifier.classes_.astype(int)
        for position, (_, row) in enumerate(valid.iterrows()):
            rows.append(
                {
                    "contract_date": row["contract_date"],
                    TARGET: float(row[TARGET]),
                    "predicted_high_f": float(row["predicted_high_f"]),
                    "validation_year": int(validation_year),
                    "ordinal_probabilities": {int(label): float(probability) for label, probability in zip(classes, probabilities[position], strict=False)},
                }
            )
    return pd.DataFrame(rows)


def _tune_ordinal_blend(
    classifier_oof: pd.DataFrame,
    residual_predictions: pd.DataFrame,
    *,
    blend_weights: tuple[float, ...],
    monthly_shrinkage: float,
    epsilon: float,
) -> pd.DataFrame:
    columns = ["ordinal_weight", "count", "bucket_log_loss", "bucket_accuracy_pct"]
    if classifier_oof.empty:
        return pd.DataFrame(columns=columns)
    residuals = residual_predictions.copy()
    residuals["contract_date"] = pd.to_datetime(residuals["contract_date"], errors="coerce")
    residuals["month"] = residuals["contract_date"].dt.month
    residuals["residual_f"] = pd.to_numeric(residuals["residual_f"], errors="coerce")
    residuals = residuals.dropna(subset=["contract_date", "month", "residual_f"])
    rows = []
    for weight in blend_weights:
        losses: list[float] = []
        hits: list[bool] = []
        for _, row in classifier_oof.iterrows():
            year = int(row["validation_year"])
            history = residuals.loc[residuals["contract_date"].dt.year.lt(year)]
            if history.empty:
                continue
            date = pd.Timestamp(row["contract_date"])
            predicted = float(row["predicted_high_f"])
            actual_bucket = temperature_bucket_label(float(row[TARGET]))
            empirical = _shrunk_empirical_probabilities(
                predicted,
                history,
                month=int(date.month),
                monthly_shrinkage=monthly_shrinkage,
            )
            offset_probabilities = row["ordinal_probabilities"]
            ordinal = _offset_probabilities_to_buckets(
                predicted,
                np.asarray(list(offset_probabilities), dtype=int),
                np.asarray(list(offset_probabilities.values()), dtype=float),
                empirical_reference=empirical,
            )
            blended = _blend_probabilities(empirical, ordinal, float(weight))
            losses.append(-math.log(max(epsilon, blended.get(actual_bucket, 0.0))))
            hits.append(_top_bucket(blended, predicted) == actual_bucket)
        if losses:
            rows.append(
                {
                    "ordinal_weight": float(weight),
                    "count": int(len(losses)),
                    "bucket_log_loss": float(np.mean(losses)),
                    "bucket_accuracy_pct": float(np.mean(hits) * 100.0),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _shrunk_empirical_probabilities(
    predicted: float,
    residuals: pd.DataFrame,
    *,
    month: int,
    monthly_shrinkage: float,
) -> dict[str, float]:
    global_values = residuals["residual_f"].to_numpy(dtype=float)
    month_values = residuals.loc[residuals["month"].eq(month), "residual_f"].to_numpy(dtype=float)
    global_probs = _bucket_counts(predicted, global_values)
    month_probs = _bucket_counts(predicted, month_values) if len(month_values) else {}
    weight = float(len(month_values) / (len(month_values) + monthly_shrinkage)) if len(month_values) else 0.0
    labels = set(global_probs) | set(month_probs)
    probabilities = {
        label: (1.0 - weight) * global_probs.get(label, 0.0) + weight * month_probs.get(label, 0.0)
        for label in labels
    }
    total = sum(probabilities.values())
    return {label: value / total for label, value in probabilities.items()} if total else global_probs


def _offset_probabilities_to_buckets(
    predicted: float,
    classes: np.ndarray,
    probabilities: np.ndarray,
    *,
    empirical_reference: dict[str, float] | None = None,
) -> dict[str, float]:
    point_index = _bucket_index(predicted)
    output: dict[str, float] = {}
    for offset, probability in zip(classes, probabilities, strict=False):
        offset = int(offset)
        probability = float(probability)
        if offset in {-2, 2} and empirical_reference:
            eligible = {
                label: value
                for label, value in empirical_reference.items()
                if (_bucket_label_index(label) - point_index <= -2 if offset == -2 else _bucket_label_index(label) - point_index >= 2)
            }
            eligible_total = float(sum(eligible.values()))
            if eligible_total > 0:
                for label, reference_probability in eligible.items():
                    output[label] = output.get(label, 0.0) + probability * float(reference_probability) / eligible_total
                continue
        lower = 2 * (point_index + int(offset))
        label = f"{lower}-{lower + 1}"
        output[label] = output.get(label, 0.0) + probability
    return output


def _blend_probabilities(empirical: dict[str, float], ordinal: dict[str, float], ordinal_weight: float) -> dict[str, float]:
    labels = set(empirical) | set(ordinal)
    output = {
        label: (1.0 - ordinal_weight) * empirical.get(label, 0.0) + ordinal_weight * ordinal.get(label, 0.0)
        for label in labels
    }
    total = sum(output.values())
    return {label: value / total for label, value in output.items()} if total else output


def _top_bucket(probabilities: dict[str, float], predicted: float) -> str:
    return min(probabilities, key=lambda label: (-probabilities[label], abs(_bucket_center(label) - predicted), label))


def paired_bootstrap_bucket_gain(
    decisions: pd.DataFrame,
    *,
    repetitions: int = 5000,
    random_state: int = 42,
) -> pd.Series:
    if decisions.empty:
        return pd.Series(dtype="float64")
    return paired_bootstrap_accuracy_gain(
        decisions["modal_bucket_hit"],
        decisions["point_bucket_hit"],
        repetitions=repetitions,
        random_state=random_state,
    )


def paired_bootstrap_accuracy_gain(
    candidate_hits: Iterable[object],
    baseline_hits: Iterable[object],
    *,
    repetitions: int = 5000,
    random_state: int = 42,
) -> pd.Series:
    candidate = pd.Series(candidate_hits, dtype="boolean")
    baseline = pd.Series(baseline_hits, dtype="boolean")
    valid = candidate.notna() & baseline.notna()
    paired_gain = (
        candidate.loc[valid].astype(float).to_numpy()
        - baseline.loc[valid].astype(float).to_numpy()
    )
    if not len(paired_gain):
        return pd.Series(dtype="float64")
    rng = np.random.default_rng(random_state)
    sampled = rng.choice(paired_gain, size=(repetitions, len(paired_gain)), replace=True).mean(axis=1) * 100.0
    return pd.Series(
        {
            "count": int(len(paired_gain)),
            "gain_pp": float(paired_gain.mean() * 100.0),
            "ci_low_pp": float(np.quantile(sampled, 0.025)),
            "ci_high_pp": float(np.quantile(sampled, 0.975)),
            "probability_gain_gt_zero": float((sampled > 0.0).mean()),
        }
    )


def temperature_bucket_label(value: float) -> str:
    rounded = int(Decimal(str(float(value))).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    lower = rounded if rounded % 2 == 0 else rounded - 1
    return f"{lower}-{lower + 1}"


def _bucket_index(value: float) -> int:
    label = temperature_bucket_label(float(value))
    return _bucket_label_index(label)


def _bucket_label_index(label: str) -> int:
    match = re.fullmatch(r"(-?\d+)-(-?\d+)", label)
    if match is None:
        raise ValueError(f"Invalid bucket label: {label!r}")
    return int(match.group(1)) // 2


def _bucket_counts(predicted: float, residuals: np.ndarray) -> dict[str, float]:
    if not len(residuals):
        return {}
    counts = Counter(temperature_bucket_label(predicted + residual) for residual in residuals)
    total = float(sum(counts.values()))
    return {label: count / total for label, count in counts.items()}


def _bucket_center(label: str) -> float:
    match = re.fullmatch(r"(-?\d+)-(-?\d+)", label)
    if match is None:
        raise ValueError(f"Invalid bucket label: {label!r}")
    return (float(match.group(1)) + float(match.group(2))) / 2.0


def _coerce_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _finite_float(value: object) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if np.isfinite(converted) else None
