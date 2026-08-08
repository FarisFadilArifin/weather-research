from __future__ import annotations

import pandas as pd

from src.calibration.weather_no_walkforward import (
    filter_candidates,
    frozen_median_parameters,
    monthly_walk_forward,
)


def _frame() -> pd.DataFrame:
    rows = []
    for date in pd.date_range("2026-01-01", "2026-03-31"):
        for bucket, probability, cost in (("low", 0.10, 0.70), ("high", 0.30, 0.60)):
            no_win = bucket == "low"
            rows.append(
                {
                    "contract_date": date,
                    "market_slug": bucket,
                    "model_yes_probability": probability,
                    "no_entry_cost": cost,
                    "no_edge": 1.0 - probability - cost,
                    "provider_spread_high_f": 4.0,
                    "confidence_gate_passed": True,
                    "is_open_tail": False,
                    "is_point_bucket": bucket == "high",
                    "is_recommended_bucket": bucket == "high",
                    "no_available": True,
                    "no_win": no_win,
                    "no_net_return": (1.0 / cost - 1.0) if no_win else -1.0,
                }
            )
    return pd.DataFrame(rows)


def test_filter_selects_one_highest_edge_no_trade_per_day() -> None:
    parameters = {
        "max_yes_probability": 0.20,
        "max_no_price": 0.80,
        "min_no_edge": 0.05,
        "max_provider_spread_f": 8.0,
        "require_confidence_gate": True,
        "exclude_open_tail": True,
        "bucket_scope": "all",
    }
    selected = filter_candidates(_frame(), parameters)
    assert selected["contract_date"].is_unique
    assert selected["market_slug"].eq("low").all()


def test_walk_forward_uses_previous_month_and_freezes_medians() -> None:
    grid = (
        {
            "max_yes_probability": 0.20,
            "max_no_price": 0.80,
            "min_no_edge": 0.05,
            "max_provider_spread_f": 8.0,
            "require_confidence_gate": True,
            "exclude_open_tail": True,
            "bucket_scope": "all",
        },
    )
    folds, trades = monthly_walk_forward(_frame(), grid=grid)
    assert folds[["train_month", "test_month"]].values.tolist() == [
        ["2026-01", "2026-02"],
        ["2026-02", "2026-03"],
    ]
    assert len(trades) == 59
    assert frozen_median_parameters(folds) == grid[0]
