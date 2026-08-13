from __future__ import annotations

import pandas as pd
import pytest

from src.calibration.temperature_buckets import (
    POLYMARKET_CELSIUS_1C,
    POLYMARKET_FAHRENHEIT_2F,
    point_bucket_metrics,
    point_bucket_predictions,
)


def test_tokyo_point_bucket_uses_celsius_half_up_after_conversion() -> None:
    predictions = pd.DataFrame(
        {
            "contract_date": ["2026-08-09", "2026-08-10"],
            "actual_high_f": [91.04, 89.6],
            "actual_high_c": [32.8, 32.0],
            "predicted_high_f": [89.24, 90.14],
        }
    )

    scored = point_bucket_predictions(predictions, POLYMARKET_CELSIUS_1C)
    metrics = point_bucket_metrics(predictions, POLYMARKET_CELSIUS_1C).iloc[0]

    assert scored["actual_market_bucket"].tolist() == [33, 32]
    assert scored["predicted_market_bucket"].tolist() == [32, 32]
    assert scored["bucket_hit"].tolist() == [False, True]
    assert metrics["bucket_hits"] == 1
    assert metrics["exact_bucket_hit_rate"] == pytest.approx(0.5)


def test_fahrenheit_market_contract_groups_rounded_degrees_into_two_f_cells() -> None:
    predictions = pd.DataFrame(
        {
            "actual_high_f": [91.0, 92.0],
            "predicted_high_f": [90.6, 93.2],
        }
    )

    scored = point_bucket_predictions(predictions, POLYMARKET_FAHRENHEIT_2F)

    assert scored["actual_market_bucket"].tolist() == [90, 92]
    assert scored["predicted_market_bucket"].tolist() == [90, 92]
    assert scored["bucket_hit"].all()
