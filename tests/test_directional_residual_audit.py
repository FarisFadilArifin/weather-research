from __future__ import annotations

import numpy as np
import pandas as pd

from src.calibration.directional_residual_audit import (
    _benjamini_hochberg,
    _stability_table,
    add_directional_audit_groups,
)


def test_benjamini_hochberg_is_bounded_and_not_below_raw_p() -> None:
    raw = np.array([0.001, 0.02, 0.04, 0.8])
    adjusted = _benjamini_hochberg(raw)
    assert np.all((adjusted >= 0) & (adjusted <= 1))
    assert np.all(adjusted >= raw)


def test_directional_groups_are_fixed_and_preserve_all_rows() -> None:
    source = pd.DataFrame(
        {
            "contract_date": pd.to_datetime(["2024-01-01", "2024-07-01"]),
            "prediction_fraction_f": [0.1, 0.8],
            "default_half_up": [0, 1],
            "base_prediction_spread_f": [0.2, 1.5],
            "provider_spread_high_f": [2.0, 7.0],
            "base_mean_minus_point_f": [-0.5, 0.5],
            "provider_mean_minus_point_f": [-2.0, 2.0],
            "prior_residual_bias_30d_f": [-0.7, 0.7],
            "point_minus_observed_high_f": [4.0, 12.0],
            "observed_cloud_cover_at_as_of": [10.0, 90.0],
            "observed_precip_recent_at_as_of": [0.0, 1.0],
            "point_prediction_f": [70.1, 96.8],
            "models_supporting_alternative_bucket": [1.0, 4.0],
            "models_supporting_default_bucket": [4.0, 1.0],
            "continuous_alternative_probability_advantage_180d": [-0.2, 0.2],
        }
    )
    grouped = add_directional_audit_groups(source)
    assert len(grouped) == len(source)
    assert grouped["audit_season"].tolist() == ["DJF", "JJA"]
    assert grouped["audit_fraction_quarter"].tolist() == ["Q1_[0,.25]", "Q4_(.75,1)"]
    assert grouped["audit_bucket_support_balance"].tolist() == [
        "default_leads_by_2+",
        "alternative_leads_by_2+",
    ]


def test_2026_only_confirms_and_cannot_change_development_selection() -> None:
    rows = []
    for year in (2023, 2024, 2025):
        rows.append(
            {
                "year": year,
                "group_name": "example",
                "group_value": "signal",
                "row_count": 100,
                "underprediction_count": 70,
                "underprediction_rate": 0.70,
                "decisive_override_count": 20,
                "recovery_count": 14,
                "alternative_recovery_share": 0.70,
            }
        )
    rows.append(
        {
            "year": 2026,
            "group_name": "example",
            "group_value": "signal",
            "row_count": 20,
            "underprediction_count": 5,
            "underprediction_rate": 0.25,
            "decisive_override_count": 10,
            "recovery_count": 3,
            "alternative_recovery_share": 0.30,
        }
    )
    rejected_confirmation = _stability_table(
        pd.DataFrame(rows), metric="residual", minimum_year_count=20, minimum_confirmation_count=15
    ).iloc[0]
    assert rejected_confirmation["stable_development_signal"]
    assert not rejected_confirmation["confirmed_stable_signal"]

    rows[-1]["underprediction_count"] = 15
    rows[-1]["underprediction_rate"] = 0.75
    accepted_confirmation = _stability_table(
        pd.DataFrame(rows), metric="residual", minimum_year_count=20, minimum_confirmation_count=15
    ).iloc[0]
    assert accepted_confirmation["stable_development_signal"]
    assert accepted_confirmation["confirmed_stable_signal"]
    assert accepted_confirmation["development_rate"] == rejected_confirmation["development_rate"]
