from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.calibration.seoul_polymarket_walkforward import (
    DEFAULT_FOLDS,
    EconomicFold,
    economic_metrics,
    expand_families_with_price_caps,
    family_by_name,
    filter_mask,
    validate_folds,
    validate_frame,
    walk_forward_backtest,
)


def _frame() -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", "2026-07-25", freq="D")
    index = np.arange(len(dates))
    good = index % 3 != 0
    frame = pd.DataFrame(
        {
            "contract_date": dates,
            "market_top_probability_c": np.where(good, 0.55, 0.32),
            "provider_spread_high_f": np.where(good, 2.0, 6.0),
            "market_probability_decision": np.where(good, "shadow_trade", "no_trade"),
        }
    )
    for action in ("point", "probability"):
        frame[f"{action}_available"] = True
        frame[f"{action}_entry_cost"] = np.where(good, 0.30, 0.60)
        frame[f"{action}_model_probability"] = np.where(good, 0.60, 0.35)
        frame[f"{action}_edge"] = (
            frame[f"{action}_model_probability"] - frame[f"{action}_entry_cost"]
        )
        frame[f"{action}_win"] = good
        frame[f"{action}_net_return"] = np.where(good, 1.0 / 0.30 - 1.0, -1.0)
        frame[f"{action}_market_slug"] = "synthetic"
        frame[f"{action}_bucket_label"] = "20°C"
    return frame


def test_economic_metrics_use_fixed_one_dollar_risk_and_drawdown() -> None:
    metrics = economic_metrics(pd.Series([1.0, -1.0, 2.0]), pd.Series([True, False, True]))
    assert metrics["trade_count"] == 3
    assert metrics["total_pnl_per_1usd"] == 2.0
    assert metrics["win_rate"] == pytest.approx(2 / 3)
    assert metrics["max_drawdown"] == 1.0


def test_price_edge_filter_applies_to_realized_action_columns() -> None:
    frame = validate_frame(_frame())
    family = family_by_name("probability_price_edge")
    mask = filter_mask(frame, family, {"max_price": 0.35, "min_edge": 0.20})
    assert mask.sum() == int((np.arange(len(frame)) % 3 != 0).sum())


def test_price_caps_are_crossed_with_non_price_filter_families() -> None:
    families = expand_families_with_price_caps(price_caps=(0.40, 0.55))
    family = family_by_name("probability_provider_spread_edge_price_cap", families)
    assert len(family.parameter_grid) == 6 * 5 * 2
    assert {row["max_price"] for row in family.parameter_grid} == {0.40, 0.55}


def test_walk_forward_prefers_positive_economic_filter() -> None:
    folds, trades, summary = walk_forward_backtest(_frame())
    assert not folds.empty
    assert not trades.empty
    winner = summary.loc[summary["winner_eligible"]].iloc[0]
    assert winner["mean_return"] > 0
    assert winner["total_pnl_per_1usd"] > 0


def test_leaking_fold_is_rejected() -> None:
    frame = validate_frame(_frame())
    folds = (
        EconomicFold("bad", "2026-01-01", "2026-04-01", "2026-03-01", "2026-03-31"),
    )
    with pytest.raises(ValueError, match="invalid chronology"):
        validate_folds(frame, folds)


def test_default_folds_use_only_the_immediately_previous_month() -> None:
    assert [fold.name for fold in DEFAULT_FOLDS] == [
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
    ]
    for fold in DEFAULT_FOLDS:
        train_end = pd.Timestamp(fold.train_end)
        validation_start = pd.Timestamp(fold.validation_start)
        assert train_end + pd.Timedelta(days=1) == validation_start
        assert pd.Timestamp(fold.train_start).month == train_end.month
