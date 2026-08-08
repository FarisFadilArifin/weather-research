from __future__ import annotations

from itertools import product
import json
import math
from statistics import median_low
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


YES_PROBABILITY_MAXIMUMS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
NO_PRICE_CAPS = (0.55, 0.65, 0.75, 0.80, 0.85, 0.90, 0.95)
MINIMUM_NO_EDGES = (0.00, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20)
PROVIDER_SPREAD_CAPS_F = (4.0, 6.0, 8.0, 10.0, 100.0)
BUCKET_SCOPES = ("all", "point", "recommended")


def parameter_grid() -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "max_yes_probability": max_yes_probability,
            "max_no_price": max_no_price,
            "min_no_edge": min_no_edge,
            "max_provider_spread_f": max_provider_spread_f,
            "require_confidence_gate": require_confidence_gate,
            "exclude_open_tail": exclude_open_tail,
            "bucket_scope": bucket_scope,
        }
        for (
            max_yes_probability,
            max_no_price,
            min_no_edge,
            max_provider_spread_f,
            require_confidence_gate,
            exclude_open_tail,
            bucket_scope,
        ) in product(
            YES_PROBABILITY_MAXIMUMS,
            NO_PRICE_CAPS,
            MINIMUM_NO_EDGES,
            PROVIDER_SPREAD_CAPS_F,
            (False, True),
            (False, True),
            BUCKET_SCOPES,
        )
    )


def validate_candidate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "contract_date",
        "model_yes_probability",
        "no_entry_cost",
        "no_edge",
        "provider_spread_high_f",
        "confidence_gate_passed",
        "is_open_tail",
        "is_point_bucket",
        "is_recommended_bucket",
        "no_available",
        "no_win",
        "no_net_return",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"NO candidate frame is missing columns: {missing}")
    prepared = frame.copy()
    prepared["contract_date"] = pd.to_datetime(prepared["contract_date"], errors="raise")
    return prepared.sort_values(["contract_date", "market_slug"], ignore_index=True)


def filter_candidates(frame: pd.DataFrame, parameters: Mapping[str, Any]) -> pd.DataFrame:
    mask = (
        frame["no_available"].astype(bool)
        & frame["model_yes_probability"].le(float(parameters["max_yes_probability"]))
        & frame["no_entry_cost"].le(float(parameters["max_no_price"]))
        & frame["no_edge"].ge(float(parameters["min_no_edge"]))
        & frame["provider_spread_high_f"].le(float(parameters["max_provider_spread_f"]))
    )
    if bool(parameters["require_confidence_gate"]):
        mask &= frame["confidence_gate_passed"].astype(bool)
    if bool(parameters["exclude_open_tail"]):
        mask &= ~frame["is_open_tail"].astype(bool)
    scope = str(parameters["bucket_scope"])
    if scope == "point":
        mask &= frame["is_point_bucket"].astype(bool)
    elif scope == "recommended":
        mask &= frame["is_recommended_bucket"].astype(bool)
    elif scope != "all":
        raise ValueError(f"unsupported bucket scope: {scope}")
    eligible = frame.loc[mask].copy()
    if eligible.empty:
        return eligible
    eligible = eligible.sort_values(
        ["contract_date", "no_edge", "model_yes_probability", "no_entry_cost"],
        ascending=[True, False, True, True],
    )
    return eligible.groupby("contract_date", sort=False, as_index=False).head(1)


def economic_metrics(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        return {
            "trade_count": 0,
            "wins": 0,
            "win_rate": math.nan,
            "total_pnl_per_1usd": 0.0,
            "mean_return": math.nan,
            "return_std": math.nan,
            "mean_return_lcb_95": math.nan,
            "profit_factor": math.nan,
            "max_drawdown_per_1usd": math.nan,
        }
    returns = pd.to_numeric(trades["no_net_return"], errors="raise").to_numpy(dtype=float)
    count = len(returns)
    mean = float(returns.mean())
    std = float(returns.std(ddof=1)) if count > 1 else 0.0
    lcb = mean - 1.645 * std / math.sqrt(count) if count > 1 else mean
    positive = float(returns[returns > 0].sum())
    negative = float(-returns[returns < 0].sum())
    cumulative = np.concatenate(([0.0], np.cumsum(returns)))
    drawdown = np.maximum.accumulate(cumulative) - cumulative
    return {
        "trade_count": count,
        "wins": int(trades["no_win"].astype(bool).sum()),
        "win_rate": float(trades["no_win"].astype(bool).mean()),
        "total_pnl_per_1usd": float(returns.sum()),
        "mean_return": mean,
        "return_std": std,
        "mean_return_lcb_95": lcb,
        "profit_factor": positive / negative if negative else math.inf,
        "max_drawdown_per_1usd": float(drawdown.max()),
    }


def _selection_arrays(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "date": pd.factorize(frame["contract_date"], sort=True)[0],
        "probability": frame["model_yes_probability"].to_numpy(dtype=float),
        "cost": frame["no_entry_cost"].to_numpy(dtype=float),
        "edge": frame["no_edge"].to_numpy(dtype=float),
        "spread": frame["provider_spread_high_f"].to_numpy(dtype=float),
        "confidence": frame["confidence_gate_passed"].to_numpy(dtype=bool),
        "tail": frame["is_open_tail"].to_numpy(dtype=bool),
        "point": frame["is_point_bucket"].to_numpy(dtype=bool),
        "recommended": frame["is_recommended_bucket"].to_numpy(dtype=bool),
        "available": frame["no_available"].to_numpy(dtype=bool),
        "win": frame["no_win"].to_numpy(dtype=bool),
        "return": frame["no_net_return"].to_numpy(dtype=float),
    }


def _fast_selected_positions(
    arrays: Mapping[str, np.ndarray], parameters: Mapping[str, Any]
) -> np.ndarray:
    mask = (
        arrays["available"]
        & (arrays["probability"] <= float(parameters["max_yes_probability"]))
        & (arrays["cost"] <= float(parameters["max_no_price"]))
        & (arrays["edge"] >= float(parameters["min_no_edge"]))
        & (arrays["spread"] <= float(parameters["max_provider_spread_f"]))
    )
    if bool(parameters["require_confidence_gate"]):
        mask &= arrays["confidence"]
    if bool(parameters["exclude_open_tail"]):
        mask &= ~arrays["tail"]
    scope = str(parameters["bucket_scope"])
    if scope == "point":
        mask &= arrays["point"]
    elif scope == "recommended":
        mask &= arrays["recommended"]
    positions = np.flatnonzero(mask)
    if not len(positions):
        return positions
    order = np.lexsort(
        (
            arrays["cost"][positions],
            arrays["probability"][positions],
            -arrays["edge"][positions],
            arrays["date"][positions],
        )
    )
    ranked = positions[order]
    _, first = np.unique(arrays["date"][ranked], return_index=True)
    return ranked[np.sort(first)]


def _fast_metrics(arrays: Mapping[str, np.ndarray], positions: np.ndarray) -> dict[str, Any]:
    if not len(positions):
        return economic_metrics(pd.DataFrame())
    returns = arrays["return"][positions]
    count = len(returns)
    mean = float(returns.mean())
    std = float(returns.std(ddof=1)) if count > 1 else 0.0
    positive = float(returns[returns > 0].sum())
    negative = float(-returns[returns < 0].sum())
    cumulative = np.concatenate(([0.0], np.cumsum(returns)))
    drawdown = np.maximum.accumulate(cumulative) - cumulative
    return {
        "trade_count": count,
        "wins": int(arrays["win"][positions].sum()),
        "win_rate": float(arrays["win"][positions].mean()),
        "total_pnl_per_1usd": float(returns.sum()),
        "mean_return": mean,
        "return_std": std,
        "mean_return_lcb_95": mean - 1.645 * std / math.sqrt(count) if count > 1 else mean,
        "profit_factor": positive / negative if negative else math.inf,
        "max_drawdown_per_1usd": float(drawdown.max()),
    }


def monthly_walk_forward(
    frame: pd.DataFrame,
    *,
    grid: Sequence[Mapping[str, Any]] | None = None,
    minimum_train_trades: int = 8,
    minimum_train_coverage: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prepared = validate_candidate_frame(frame)
    grid = tuple(grid or parameter_grid())
    months = sorted(prepared["contract_date"].dt.to_period("M").unique())
    fold_rows: list[dict[str, Any]] = []
    trade_rows: list[pd.DataFrame] = []
    for train_month, test_month in zip(months, months[1:]):
        train = prepared.loc[prepared["contract_date"].dt.to_period("M").eq(train_month)]
        test = prepared.loc[prepared["contract_date"].dt.to_period("M").eq(test_month)]
        train_arrays = _selection_arrays(train)
        available_days = int(train.loc[train["no_available"], "contract_date"].nunique())
        required = max(minimum_train_trades, math.ceil(minimum_train_coverage * available_days))
        evaluated: list[dict[str, Any]] = []
        for parameters in grid:
            positions = _fast_selected_positions(train_arrays, parameters)
            metrics = _fast_metrics(train_arrays, positions)
            if metrics["trade_count"] < required:
                continue
            evaluated.append({"parameters": dict(parameters), **metrics})
        if not evaluated:
            raise ValueError(f"no eligible NO parameters for training month {train_month}")
        evaluated.sort(
            key=lambda row: (
                row["mean_return_lcb_95"],
                row["mean_return"],
                row["total_pnl_per_1usd"],
                row["trade_count"],
                json.dumps(row["parameters"], sort_keys=True),
            ),
            reverse=True,
        )
        selected = evaluated[0]
        parameters = selected["parameters"]
        test_trades = filter_candidates(test, parameters)
        metrics = economic_metrics(test_trades)
        serialized = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
        if not test_trades.empty:
            test_trades = test_trades.copy()
            test_trades["fold"] = str(test_month)
            test_trades["selected_parameters"] = serialized
            trade_rows.append(test_trades)
        fold_rows.append(
            {
                "train_month": str(train_month),
                "test_month": str(test_month),
                "selected_parameters": serialized,
                "train_trade_count": selected["trade_count"],
                "train_mean_return": selected["mean_return"],
                "train_mean_return_lcb_95": selected["mean_return_lcb_95"],
                **metrics,
            }
        )
    return pd.DataFrame(fold_rows), (
        pd.concat(trade_rows, ignore_index=True) if trade_rows else pd.DataFrame()
    )


def frozen_median_parameters(folds: pd.DataFrame) -> dict[str, Any]:
    selections = [json.loads(value) for value in folds["selected_parameters"]]
    numeric = (
        "max_yes_probability",
        "max_no_price",
        "min_no_edge",
        "max_provider_spread_f",
    )
    result: dict[str, Any] = {
        key: float(median_low(sorted(float(selection[key]) for selection in selections)))
        for key in numeric
    }
    for key in ("require_confidence_gate", "exclude_open_tail", "bucket_scope"):
        counts = pd.Series([selection[key] for selection in selections]).value_counts()
        result[key] = str(counts.index[0]) if key == "bucket_scope" else bool(counts.index[0])
    return result
