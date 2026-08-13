from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np
import pandas as pd


TARGET = "actual_high_f"
PREDICTION = "predicted_high_f"


@dataclass(frozen=True)
class TwoModelBlendSelection:
    primary_method: str
    secondary_method: str
    primary_weight: float
    secondary_weight: float
    mean_fold_mae_f: float
    worst_fold_mae_f: float
    row_mae_f: float
    grid_step: float


class FixedWeightBlendRegressor:
    def __init__(self, feature_names: tuple[str, str], weights: tuple[float, float]) -> None:
        if len(feature_names) != 2 or len(weights) != 2:
            raise ValueError("FixedWeightBlendRegressor requires exactly two features and two weights.")
        if any(weight < 0.0 for weight in weights) or not np.isclose(sum(weights), 1.0):
            raise ValueError("Blend weights must be non-negative and sum to one.")
        self.feature_names = tuple(feature_names)
        self.weights = tuple(float(weight) for weight in weights)
        self.coef_ = np.asarray(self.weights, dtype=float)
        self.intercept_ = 0.0

    def predict(self, values) -> np.ndarray:
        if isinstance(values, pd.DataFrame):
            missing = [feature for feature in self.feature_names if feature not in values]
            if missing:
                raise ValueError(f"Blend prediction frame is missing features: {missing}")
            matrix = values.loc[:, self.feature_names].to_numpy(dtype=float)
        else:
            matrix = np.asarray(values, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != 2:
            raise ValueError("Blend prediction input must have exactly two columns.")
        return matrix @ self.coef_


class FixedWeightSimplexBlendRegressor:
    def __init__(self, feature_names: tuple[str, ...], weights: tuple[float, ...]) -> None:
        if len(feature_names) < 2 or len(feature_names) != len(weights):
            raise ValueError("Simplex blend requires matching feature and weight vectors.")
        if any(weight < 0.0 for weight in weights) or not np.isclose(sum(weights), 1.0):
            raise ValueError("Blend weights must be non-negative and sum to one.")
        self.feature_names = tuple(feature_names)
        self.weights = tuple(float(weight) for weight in weights)
        self.coef_ = np.asarray(self.weights, dtype=float)
        self.intercept_ = 0.0

    def predict(self, values) -> np.ndarray:
        if isinstance(values, pd.DataFrame):
            missing = [feature for feature in self.feature_names if feature not in values]
            if missing:
                raise ValueError(f"Blend prediction frame is missing features: {missing}")
            matrix = values.loc[:, self.feature_names].to_numpy(dtype=float)
        else:
            matrix = np.asarray(values, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.weights):
            raise ValueError(
                f"Blend prediction input must have exactly {len(self.weights)} columns."
            )
        return matrix @ self.coef_


def merge_multiple_prediction_sources(
    prediction_sources: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    if len(prediction_sources) < 2:
        raise ValueError("At least two prediction sources are required.")
    prepared = [
        _method_prediction_frame(predictions, method)
        for method, predictions in prediction_sources.items()
    ]
    join_columns = [
        column
        for column in ("contract_date", TARGET, "evaluation_scope", "fold")
        if all(column in frame for frame in prepared)
    ]
    if "contract_date" not in join_columns or TARGET not in join_columns:
        raise ValueError("Prediction sources must share contract_date and actual_high_f.")
    merged = prepared[0]
    for frame in prepared[1:]:
        merged = merged.merge(frame, on=join_columns, how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError("Prediction sources have no common evaluation rows.")
    return merged.sort_values("contract_date").reset_index(drop=True)


def scan_three_model_simplex_weights(
    frame: pd.DataFrame,
    *,
    methods: tuple[str, str, str],
    grid_step: float = 0.01,
) -> pd.DataFrame:
    if len(methods) != 3 or len(set(methods)) != 3:
        raise ValueError("Three distinct model methods are required.")
    units = int(round(1.0 / grid_step))
    if not 0.0 < grid_step <= 1.0 or not np.isclose(units * grid_step, 1.0):
        raise ValueError("grid_step must divide one exactly.")
    prediction_columns = [_prediction_column(method) for method in methods]
    required = [TARGET, "fold", *prediction_columns]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Blend frame is missing required columns: {missing}")
    clean = frame.dropna(subset=required).reset_index(drop=True)
    if clean.empty:
        raise ValueError("Blend frame has no complete rows.")
    target = pd.to_numeric(clean[TARGET], errors="coerce").to_numpy(dtype=float)
    predictions = clean.loc[:, prediction_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    fold_indices = [np.asarray(index, dtype=int) for index in clean.groupby("fold", sort=True).indices.values()]
    rows: list[dict[str, float | int]] = []
    for first_units in range(units + 1):
        for second_units in range(units - first_units + 1):
            third_units = units - first_units - second_units
            weights = np.asarray(
                [first_units / units, second_units / units, third_units / units],
                dtype=float,
            )
            predicted = predictions @ weights
            absolute_error = np.abs(predicted - target)
            fold_maes = [float(absolute_error[index].mean()) for index in fold_indices]
            rows.append(
                {
                    f"{methods[0]}_weight": float(weights[0]),
                    f"{methods[1]}_weight": float(weights[1]),
                    f"{methods[2]}_weight": float(weights[2]),
                    "mean_fold_mae_f": float(np.mean(fold_maes)),
                    "worst_fold_mae_f": float(np.max(fold_maes)),
                    "row_mae_f": float(absolute_error.mean()),
                    "fold_count": len(fold_maes),
                    "row_count": len(clean),
                }
            )
    return pd.DataFrame(rows)


def select_three_model_simplex_weights(
    scan: pd.DataFrame,
    *,
    methods: tuple[str, str, str],
) -> pd.Series:
    weight_columns = [f"{method}_weight" for method in methods]
    required = [*weight_columns, "mean_fold_mae_f", "worst_fold_mae_f", "row_mae_f"]
    missing = [column for column in required if column not in scan]
    if missing or scan.empty:
        raise ValueError(f"Simplex scan is empty or missing required columns: {missing}")
    equal_weight = 1.0 / len(methods)
    distance = sum(
        (pd.to_numeric(scan[column], errors="coerce") - equal_weight).abs()
        for column in weight_columns
    )
    return (
        scan.assign(distance_from_equal=distance)
        .sort_values(
            ["mean_fold_mae_f", "worst_fold_mae_f", "distance_from_equal", *weight_columns],
            kind="stable",
        )
        .iloc[0]
    )


def blend_simplex_predictions(
    frame: pd.DataFrame,
    *,
    methods: tuple[str, ...],
    weights: tuple[float, ...],
    method: str,
) -> pd.DataFrame:
    model = FixedWeightSimplexBlendRegressor(
        tuple(_prediction_column(name) for name in methods),
        weights,
    )
    required = ["contract_date", TARGET, *model.feature_names]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Blend frame is missing required columns: {missing}")
    out_columns = [
        column
        for column in ("contract_date", TARGET, "evaluation_scope", "fold")
        if column in frame
    ]
    out = frame.loc[:, out_columns].copy()
    out["method"] = method
    out[PREDICTION] = model.predict(frame.loc[:, model.feature_names])
    return out


def merge_prediction_sources(
    primary_predictions: pd.DataFrame,
    secondary_predictions: pd.DataFrame,
    *,
    primary_method: str = "xgboost",
    secondary_method: str = "extra_trees",
) -> pd.DataFrame:
    primary = _method_prediction_frame(primary_predictions, primary_method)
    secondary = _method_prediction_frame(secondary_predictions, secondary_method)
    join_columns = [
        column
        for column in ("contract_date", TARGET, "evaluation_scope", "fold")
        if column in primary and column in secondary
    ]
    if "contract_date" not in join_columns or TARGET not in join_columns:
        raise ValueError("Prediction sources must share contract_date and actual_high_f.")
    merged = primary.merge(
        secondary,
        on=join_columns,
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("Prediction sources have no common evaluation rows.")
    return merged.sort_values("contract_date").reset_index(drop=True)


def scan_two_model_weights(
    frame: pd.DataFrame,
    *,
    primary_method: str = "xgboost",
    secondary_method: str = "extra_trees",
    grid_step: float = 0.001,
) -> pd.DataFrame:
    if not 0.0 < grid_step <= 1.0:
        raise ValueError("grid_step must be in (0, 1].")
    primary_column = _prediction_column(primary_method)
    secondary_column = _prediction_column(secondary_method)
    required = [TARGET, primary_column, secondary_column, "fold"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Blend frame is missing required columns: {missing}")
    clean = frame.dropna(subset=required).copy()
    if clean.empty:
        raise ValueError("Blend frame has no complete rows.")

    target = pd.to_numeric(clean[TARGET], errors="coerce").to_numpy(dtype=float)
    primary = pd.to_numeric(clean[primary_column], errors="coerce").to_numpy(dtype=float)
    secondary = pd.to_numeric(clean[secondary_column], errors="coerce").to_numpy(dtype=float)
    fold_indices = [np.asarray(index, dtype=int) for index in clean.groupby("fold", sort=True).indices.values()]
    steps = int(round(1.0 / grid_step))
    weights = np.linspace(0.0, 1.0, steps + 1)
    rows: list[dict[str, float | int]] = []
    for primary_weight in weights:
        predicted = primary_weight * primary + (1.0 - primary_weight) * secondary
        absolute_error = np.abs(predicted - target)
        fold_maes = [float(absolute_error[index].mean()) for index in fold_indices]
        rows.append(
            {
                "primary_weight": float(primary_weight),
                "secondary_weight": float(1.0 - primary_weight),
                "mean_fold_mae_f": float(np.mean(fold_maes)),
                "worst_fold_mae_f": float(np.max(fold_maes)),
                "row_mae_f": float(absolute_error.mean()),
                "fold_count": len(fold_maes),
                "row_count": len(clean),
            }
        )
    return pd.DataFrame(rows)


def select_two_model_weight(
    scan: pd.DataFrame,
    *,
    primary_method: str = "xgboost",
    secondary_method: str = "extra_trees",
    grid_step: float = 0.001,
) -> TwoModelBlendSelection:
    required = ["primary_weight", "secondary_weight", "mean_fold_mae_f", "worst_fold_mae_f", "row_mae_f"]
    missing = [column for column in required if column not in scan]
    if missing or scan.empty:
        raise ValueError(f"Weight scan is empty or missing required columns: {missing}")
    ranked = scan.assign(
        distance_from_equal=(pd.to_numeric(scan["primary_weight"], errors="coerce") - 0.5).abs()
    ).sort_values(
        ["mean_fold_mae_f", "worst_fold_mae_f", "distance_from_equal", "primary_weight"],
        kind="stable",
    )
    row = ranked.iloc[0]
    return TwoModelBlendSelection(
        primary_method=primary_method,
        secondary_method=secondary_method,
        primary_weight=float(row["primary_weight"]),
        secondary_weight=float(row["secondary_weight"]),
        mean_fold_mae_f=float(row["mean_fold_mae_f"]),
        worst_fold_mae_f=float(row["worst_fold_mae_f"]),
        row_mae_f=float(row["row_mae_f"]),
        grid_step=float(grid_step),
    )


def blend_predictions(
    frame: pd.DataFrame,
    selection: TwoModelBlendSelection,
    *,
    method: str = "xgb_extra_constrained_blend",
) -> pd.DataFrame:
    primary_column = _prediction_column(selection.primary_method)
    secondary_column = _prediction_column(selection.secondary_method)
    required = ["contract_date", TARGET, primary_column, secondary_column]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Blend frame is missing required columns: {missing}")
    out_columns = [
        column
        for column in ("contract_date", TARGET, "evaluation_scope", "fold")
        if column in frame
    ]
    out = frame.loc[:, out_columns].copy()
    out["method"] = method
    out[PREDICTION] = (
        selection.primary_weight * pd.to_numeric(frame[primary_column], errors="coerce")
        + selection.secondary_weight * pd.to_numeric(frame[secondary_column], errors="coerce")
    )
    return out


def selected_fold_metrics(
    blended_predictions: pd.DataFrame,
    selection: TwoModelBlendSelection,
) -> pd.DataFrame:
    if "fold" not in blended_predictions:
        raise ValueError("Blended predictions must include a fold column.")
    rows: list[dict[str, float | int | str]] = []
    for fold, part in blended_predictions.groupby("fold", sort=True):
        error = pd.to_numeric(part[PREDICTION], errors="coerce") - pd.to_numeric(
            part[TARGET], errors="coerce"
        )
        rows.append(
            {
                "fold": str(fold),
                "count": int(error.notna().sum()),
                "mae_f": float(error.abs().mean()),
                "rmse_f": float(np.sqrt(np.mean(np.square(error.dropna())))),
                "bias_f": float(error.mean()),
                "primary_method": selection.primary_method,
                "secondary_method": selection.secondary_method,
                "primary_weight": selection.primary_weight,
                "secondary_weight": selection.secondary_weight,
            }
        )
    return pd.DataFrame(rows)


def _method_prediction_frame(predictions: pd.DataFrame, method: str) -> pd.DataFrame:
    required = ["contract_date", TARGET, "method", PREDICTION]
    missing = [column for column in required if column not in predictions]
    if missing:
        raise ValueError(f"Prediction source is missing required columns: {missing}")
    selected = predictions.loc[predictions["method"].astype(str).eq(method)].copy()
    if selected.empty:
        raise ValueError(f"Prediction source has no rows for method {method!r}.")
    keep = [
        column
        for column in ("contract_date", TARGET, "evaluation_scope", "fold", PREDICTION)
        if column in selected
    ]
    return selected.loc[:, keep].rename(columns={PREDICTION: _prediction_column(method)})


def _prediction_column(method: str) -> str:
    return f"{method}_predicted_high_f"
