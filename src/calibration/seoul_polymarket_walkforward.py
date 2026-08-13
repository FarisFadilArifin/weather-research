from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EconomicFold:
    name: str
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str


@dataclass(frozen=True)
class EconomicFilterFamily:
    name: str
    action: str
    parameter_grid: tuple[Mapping[str, float], ...]
    eligible_for_selection: bool = True


DEFAULT_FOLDS = (
    EconomicFold("february", "2026-01-01", "2026-01-31", "2026-02-01", "2026-02-28"),
    EconomicFold("march", "2026-02-01", "2026-02-28", "2026-03-01", "2026-03-31"),
    EconomicFold("april", "2026-03-01", "2026-03-31", "2026-04-01", "2026-04-30"),
    EconomicFold("may", "2026-04-01", "2026-04-30", "2026-05-01", "2026-05-31"),
    EconomicFold("june", "2026-05-01", "2026-05-31", "2026-06-01", "2026-06-30"),
    EconomicFold("july", "2026-06-01", "2026-06-30", "2026-07-01", "2026-07-25"),
)


def _grid(**values: Sequence[float]) -> tuple[Mapping[str, float], ...]:
    names = tuple(values)
    return tuple(
        dict(zip(names, combination, strict=True))
        for combination in product(*(values[name] for name in names))
    )


PRICE_CAPS = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80)
EDGES = (-0.05, 0.00, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25)
CONFIDENCE = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
PROVIDER_SPREADS = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0)


DEFAULT_FAMILIES = (
    EconomicFilterFamily("point_all", "point", ({},), eligible_for_selection=False),
    EconomicFilterFamily("probability_all", "probability", ({},), eligible_for_selection=False),
    EconomicFilterFamily("point_price_cap", "point", _grid(max_price=PRICE_CAPS)),
    EconomicFilterFamily("probability_price_cap", "probability", _grid(max_price=PRICE_CAPS)),
    EconomicFilterFamily("point_edge", "point", _grid(min_edge=EDGES)),
    EconomicFilterFamily("probability_edge", "probability", _grid(min_edge=EDGES)),
    EconomicFilterFamily(
        "point_price_edge",
        "point",
        _grid(max_price=(0.25, 0.35, 0.45, 0.55, 0.65, 0.75), min_edge=(0.00, 0.05, 0.10, 0.15, 0.20)),
    ),
    EconomicFilterFamily(
        "probability_price_edge",
        "probability",
        _grid(max_price=(0.25, 0.35, 0.45, 0.55, 0.65, 0.75), min_edge=(0.00, 0.05, 0.10, 0.15, 0.20)),
    ),
    EconomicFilterFamily("probability_confidence", "probability", _grid(min_confidence=CONFIDENCE)),
    EconomicFilterFamily(
        "probability_confidence_edge",
        "probability",
        _grid(min_confidence=(0.30, 0.35, 0.40, 0.45, 0.50, 0.55), min_edge=(0.00, 0.05, 0.10, 0.15, 0.20)),
    ),
    EconomicFilterFamily("point_provider_spread", "point", _grid(max_provider_spread=PROVIDER_SPREADS)),
    EconomicFilterFamily("probability_provider_spread", "probability", _grid(max_provider_spread=PROVIDER_SPREADS)),
    EconomicFilterFamily(
        "point_provider_spread_edge",
        "point",
        _grid(max_provider_spread=PROVIDER_SPREADS, min_edge=(0.00, 0.05, 0.10, 0.15, 0.20)),
    ),
    EconomicFilterFamily(
        "probability_provider_spread_edge",
        "probability",
        _grid(max_provider_spread=PROVIDER_SPREADS, min_edge=(0.00, 0.05, 0.10, 0.15, 0.20)),
    ),
    EconomicFilterFamily(
        "probability_confidence_spread_edge",
        "probability",
        _grid(
            min_confidence=(0.30, 0.35, 0.40, 0.45, 0.50),
            max_provider_spread=(2.0, 3.0, 4.0, 5.0, 6.0),
            min_edge=(0.00, 0.05, 0.10, 0.15, 0.20),
        ),
    ),
    EconomicFilterFamily(
        "notebook_policy_reference",
        "probability",
        ({"notebook_policy": 1.0},),
        eligible_for_selection=False,
    ),
)


def expand_families_with_price_caps(
    families: Sequence[EconomicFilterFamily] = DEFAULT_FAMILIES,
    price_caps: Sequence[float] = PRICE_CAPS,
) -> tuple[EconomicFilterFamily, ...]:
    expanded = list(families)
    for family in families:
        if not family.eligible_for_selection:
            continue
        if any("max_price" in parameters for parameters in family.parameter_grid):
            continue
        capped_grid = tuple(
            {**dict(parameters), "max_price": float(cap)}
            for parameters in family.parameter_grid
            for cap in price_caps
        )
        expanded.append(
            EconomicFilterFamily(
                name=f"{family.name}_price_cap",
                action=family.action,
                parameter_grid=capped_grid,
            )
        )
    return tuple(expanded)


def family_by_name(
    name: str, families: Sequence[EconomicFilterFamily] = DEFAULT_FAMILIES
) -> EconomicFilterFamily:
    for family in families:
        if family.name == name:
            return family
    raise KeyError(name)


def validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "contract_date",
        "market_top_probability_c",
        "provider_spread_high_f",
        "market_probability_decision",
    }
    for action in ("point", "probability"):
        required.update(
            {
                f"{action}_available",
                f"{action}_entry_cost",
                f"{action}_model_probability",
                f"{action}_edge",
                f"{action}_net_return",
                f"{action}_win",
            }
        )
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"economic frame is missing columns: {missing}")
    prepared = frame.copy()
    prepared["contract_date"] = pd.to_datetime(prepared["contract_date"], errors="raise")
    if prepared["contract_date"].duplicated().any():
        raise ValueError("economic frame must contain one row per Seoul contract date")
    return prepared.sort_values("contract_date").reset_index(drop=True)


def filter_mask(
    frame: pd.DataFrame,
    family: EconomicFilterFamily,
    parameters: Mapping[str, float],
) -> pd.Series:
    action = family.action
    mask = frame[f"{action}_available"].astype(bool).copy()
    if "max_price" in parameters:
        mask &= frame[f"{action}_entry_cost"].le(parameters["max_price"])
    if "min_edge" in parameters:
        mask &= frame[f"{action}_edge"].ge(parameters["min_edge"])
    if "min_confidence" in parameters:
        mask &= frame["market_top_probability_c"].ge(parameters["min_confidence"])
    if "max_provider_spread" in parameters:
        mask &= frame["provider_spread_high_f"].le(parameters["max_provider_spread"])
    if "notebook_policy" in parameters:
        mask &= frame["market_probability_decision"].eq("shadow_trade")
    return mask


def economic_metrics(returns: pd.Series, wins: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(returns, errors="coerce").dropna().to_numpy(dtype=float)
    if not len(values):
        return {
            "trade_count": 0,
            "wins": 0,
            "win_rate": math.nan,
            "total_pnl_per_1usd": 0.0,
            "mean_return": math.nan,
            "return_std": math.nan,
            "mean_return_lcb_95": math.nan,
            "profit_factor": math.nan,
            "max_drawdown": math.nan,
        }
    win_values = pd.Series(wins).astype(bool).to_numpy()
    count = len(values)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if count > 1 else 0.0
    lcb = mean - 1.645 * std / math.sqrt(count) if count > 1 else mean
    positive = float(values[values > 0].sum())
    negative = float(-values[values < 0].sum())
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    drawdown = np.maximum.accumulate(cumulative) - cumulative
    return {
        "trade_count": count,
        "wins": int(win_values.sum()),
        "win_rate": float(win_values.mean()),
        "total_pnl_per_1usd": float(values.sum()),
        "mean_return": mean,
        "return_std": std,
        "mean_return_lcb_95": lcb,
        "profit_factor": positive / negative if negative > 0 else math.inf,
        "max_drawdown": float(drawdown.max()),
    }


def evaluate_candidate(
    frame: pd.DataFrame,
    family: EconomicFilterFamily,
    parameters: Mapping[str, float],
) -> dict[str, Any]:
    mask = filter_mask(frame, family, parameters)
    available_count = int(frame[f"{family.action}_available"].sum())
    metrics = economic_metrics(
        frame.loc[mask, f"{family.action}_net_return"],
        frame.loc[mask, f"{family.action}_win"],
    )
    return {
        "parameters": json.dumps(dict(parameters), sort_keys=True, separators=(",", ":")),
        "available_count": available_count,
        "coverage": metrics["trade_count"] / available_count if available_count else math.nan,
        **metrics,
    }


def select_candidate(
    train: pd.DataFrame,
    family: EconomicFilterFamily,
    *,
    minimum_selected: int | None = None,
    minimum_coverage: float = 0.15,
    maximum_coverage: float = 0.85,
) -> dict[str, Any]:
    available = int(train[f"{family.action}_available"].sum())
    required = minimum_selected or max(12, math.ceil(minimum_coverage * available))
    strict: list[dict[str, Any]] = []
    fallback: list[dict[str, Any]] = []
    for parameters in family.parameter_grid:
        row = {
            **evaluate_candidate(train, family, parameters),
            "parameter_values": dict(parameters),
        }
        if row["trade_count"] > 0:
            fallback.append(row)
        if family.eligible_for_selection and (
            row["trade_count"] < required or row["coverage"] > maximum_coverage
        ):
            continue
        strict.append(row)
    rows = strict or fallback
    if not rows:
        raise ValueError(f"no candidate in {family.name} selects a trade")
    rows.sort(
        key=lambda row: (
            row["mean_return_lcb_95"],
            row["mean_return"],
            row["total_pnl_per_1usd"],
            row["trade_count"],
            row["parameters"],
        ),
        reverse=True,
    )
    return {**rows[0], "selection_contract_relaxed": not bool(strict)}


def validate_folds(frame: pd.DataFrame, folds: Sequence[EconomicFold]) -> None:
    previous_end: pd.Timestamp | None = None
    for fold in folds:
        train_start = pd.Timestamp(fold.train_start)
        train_end = pd.Timestamp(fold.train_end)
        validation_start = pd.Timestamp(fold.validation_start)
        validation_end = pd.Timestamp(fold.validation_end)
        if not train_start <= train_end < validation_start <= validation_end:
            raise ValueError(f"invalid chronology in fold {fold.name}")
        if previous_end is not None and validation_start <= previous_end:
            raise ValueError("validation folds must be ordered and non-overlapping")
        train = frame[frame["contract_date"].between(train_start, train_end)]
        validation = frame[frame["contract_date"].between(validation_start, validation_end)]
        if train.empty or validation.empty:
            raise ValueError(f"fold {fold.name} has no train or validation rows")
        if train["contract_date"].max() >= validation["contract_date"].min():
            raise AssertionError(f"fold {fold.name} leaks future returns")
        previous_end = validation_end


def walk_forward_backtest(
    frame: pd.DataFrame,
    *,
    families: Sequence[EconomicFilterFamily] = DEFAULT_FAMILIES,
    folds: Sequence[EconomicFold] = DEFAULT_FOLDS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prepared = validate_frame(frame)
    validate_folds(prepared, folds)
    fold_rows: list[dict[str, Any]] = []
    trade_rows: list[pd.DataFrame] = []
    for family in families:
        for fold in folds:
            train = prepared[prepared["contract_date"].between(fold.train_start, fold.train_end)]
            validation = prepared[
                prepared["contract_date"].between(fold.validation_start, fold.validation_end)
            ]
            selected = select_candidate(train, family)
            parameters = selected["parameter_values"]
            metrics = evaluate_candidate(validation, family, parameters)
            mask = filter_mask(validation, family, parameters)
            trades = validation.loc[
                mask,
                [
                    "contract_date",
                    f"{family.action}_market_slug",
                    f"{family.action}_bucket_label",
                    f"{family.action}_entry_cost",
                    f"{family.action}_model_probability",
                    f"{family.action}_edge",
                    f"{family.action}_win",
                    f"{family.action}_net_return",
                ],
            ].copy()
            trades.columns = [
                "contract_date",
                "market_slug",
                "bucket_label",
                "entry_cost",
                "model_probability",
                "edge",
                "win",
                "net_return",
            ]
            trades["family"] = family.name
            trades["action"] = family.action
            trades["fold"] = fold.name
            trades["selected_parameters"] = selected["parameters"]
            trade_rows.append(trades)
            fold_rows.append(
                {
                    "family": family.name,
                    "action": family.action,
                    "eligible_for_selection": family.eligible_for_selection,
                    "fold": fold.name,
                    "train_start": train["contract_date"].min().date().isoformat(),
                    "train_end": train["contract_date"].max().date().isoformat(),
                    "validation_start": validation["contract_date"].min().date().isoformat(),
                    "validation_end": validation["contract_date"].max().date().isoformat(),
                    "selected_parameters": selected["parameters"],
                    "selection_contract_relaxed": selected["selection_contract_relaxed"],
                    "train_trade_count": selected["trade_count"],
                    "train_mean_return": selected["mean_return"],
                    "train_mean_return_lcb_95": selected["mean_return_lcb_95"],
                    **metrics,
                }
            )
    fold_results = pd.DataFrame(fold_rows)
    trade_results = pd.concat(trade_rows, ignore_index=True) if trade_rows else pd.DataFrame()
    summaries: list[dict[str, Any]] = []
    for family, group in fold_results.groupby("family", sort=False):
        family_trades = trade_results.loc[trade_results["family"].eq(family)].sort_values(
            "contract_date"
        )
        metrics = economic_metrics(family_trades["net_return"], family_trades["win"])
        available_count = int(group["available_count"].sum())
        summaries.append(
            {
                "family": family,
                "action": group["action"].iloc[0],
                "eligible_for_selection": bool(group["eligible_for_selection"].iloc[0]),
                "fold_count": len(group),
                "folds_with_trades": int(group["trade_count"].gt(0).sum()),
                "available_count": available_count,
                "coverage": metrics["trade_count"] / available_count if available_count else math.nan,
                "minimum_fold_trades": int(group["trade_count"].min()),
                "relaxed_selection_folds": int(group["selection_contract_relaxed"].sum()),
                "unique_parameter_sets": int(group["selected_parameters"].nunique()),
                "selected_parameters_by_fold": " | ".join(group["selected_parameters"]),
                **metrics,
            }
        )
    summary = pd.DataFrame(summaries)
    summary["winner_eligible"] = (
        summary["eligible_for_selection"]
        & summary["trade_count"].ge(25)
        & summary["coverage"].between(0.15, 0.85)
        & summary["folds_with_trades"].ge(4)
        & summary["relaxed_selection_folds"].eq(0)
    )
    summary["robust_positive"] = summary["winner_eligible"] & summary[
        "mean_return_lcb_95"
    ].ge(0.0)
    summary = summary.sort_values(
        ["winner_eligible", "mean_return_lcb_95", "mean_return", "total_pnl_per_1usd"],
        ascending=[False, False, False, False],
        ignore_index=True,
    )
    return fold_results, trade_results, summary


def select_winner(summary: pd.DataFrame) -> str:
    eligible = summary.loc[summary["winner_eligible"]]
    if eligible.empty:
        raise ValueError("no economic filter family satisfies the winner contract")
    return str(eligible.iloc[0]["family"])
