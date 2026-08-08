from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.calibration.bucket_correction import (
    LogisticSpec,
    add_bucket_correction_targets,
    apply_override_policy,
    fit_bucket_correction_system,
    predict_bucket_correction,
    tune_override_policy,
)


def test_override_policy_enforces_observed_high_physical_floor() -> None:
    thresholds = {
        "minimum_risk_probability": 0.5,
        "minimum_direction_probability": 0.5,
        "minimum_direction_margin": 0.1,
        "minimum_advantage_over_same": 0.1,
    }
    blocked = apply_override_policy(
        0.9,
        [0.8, 0.1, 0.1],
        thresholds,
        point_bucket_index=46,
        observed_high_f=92.0,
    )
    assert blocked == {"direction": 0, "reason": "observed_high_physical_floor"}


def test_bucket_correction_target_maps_916_and_91_to_lower_bucket() -> None:
    frame = pd.DataFrame(
        [
            {
                "point_prediction_f": 91.6,
                "point_degree_f": 92,
                "actual_high_f": 91.0,
                "actual_degree_f": 91,
            }
        ]
    )
    result = add_bucket_correction_targets(frame)
    assert result.loc[0, "point_bucket_index"] == 46
    assert result.loc[0, "actual_bucket_index"] == 45
    assert result.loc[0, "bucket_delta"] == -1
    assert result.loc[0, "point_bucket_wrong"] == 1
    assert result.loc[0, "bucket_relation_class"] == 0


def test_bucket_correction_point_bucket_is_floored_by_observed_high() -> None:
    frame = pd.DataFrame(
        [{
            "point_prediction_f": 66.4,
            "point_degree_f": 66,
            "observed_high_temp_through_as_of_f": 68.0,
            "actual_high_f": 69.0,
            "actual_degree_f": 69,
        }]
    )
    result = add_bucket_correction_targets(frame)
    assert result.loc[0, "point_bucket_index"] == 34
    assert result.loc[0, "actual_bucket_index"] == 34
    assert result.loc[0, "bucket_delta"] == 0


def test_policy_requires_positive_switch_lift_in_each_forward_year() -> None:
    rows = []
    for year in (2024, 2025):
        for _ in range(10):
            rows.append(_policy_row(year, -1, 0.90, [0.85, 0.10, 0.05]))
            rows.append(_policy_row(year, 1, 0.90, [0.05, 0.10, 0.85]))
        for _ in range(20):
            rows.append(_policy_row(year, 0, 0.10, [0.05, 0.90, 0.05]))
    thresholds, policies = tune_override_policy(pd.DataFrame(rows))
    assert thresholds["stable_forward_evidence"]
    selected = policies.loc[policies["selected"]].iloc[0]
    assert selected["switch_count"] >= 10
    assert selected["switch_accuracy"] == 1.0
    assert selected["point_accuracy_on_switches"] == 0.0


def test_policy_marks_harmful_switches_unstable() -> None:
    rows = []
    for year in (2024, 2025):
        for _ in range(10):
            rows.append(_policy_row(year, -1, 0.90, [0.05, 0.10, 0.85]))
        for _ in range(20):
            rows.append(_policy_row(year, 0, 0.10, [0.05, 0.90, 0.05]))
    thresholds, _ = tune_override_policy(pd.DataFrame(rows))
    assert not thresholds["stable_forward_evidence"]


def test_small_two_stage_fit_is_forward_only_and_fails_closed() -> None:
    rows = []
    for date in pd.date_range("2023-01-01", "2025-12-31", freq="D"):
        point = 80.0 + math.sin(date.dayofyear / 30.0)
        point_degree = int(np.floor(point + 0.5))
        bucket_shift = (-1, 0, 1)[date.dayofyear % 3]
        actual_degree = point_degree + 2 * bucket_shift
        rows.append(
            {
                "contract_date": date,
                "year": date.year,
                "month": date.month,
                "point_prediction_f": point,
                "point_degree_f": point_degree,
                "actual_high_f": float(actual_degree),
                "actual_degree_f": actual_degree,
                "observed_high_temp_through_as_of_f": float(point_degree - 4),
            }
        )
    bundle, oof, _ = fit_bucket_correction_system(
        pd.DataFrame(rows),
        station_id="KATL",
        point_model_version="point-v20",
        point_bundle_sha256="a" * 64,
        include_peak_features=False,
        specs=[LogisticSpec(c=0.1, class_weight=None)],
    )
    validation_dates = pd.to_datetime(oof["contract_date"])
    assert (pd.to_datetime(oof["model_training_cutoff"]) < validation_dates).all()
    assert (
        pd.to_datetime(oof["calibration_validation_cutoff"]) < validation_dates
    ).all()
    unavailable = predict_bucket_correction(
        bundle,
        {
            "contract_date": "2026-01-01",
            "point_prediction_f": 91.6,
            "xgboost_predicted_high_f": 91.5,
            "lightgbm_predicted_high_f": 91.7,
            "catboost_predicted_high_f": 91.8,
        },
    )
    assert unavailable["status"] == "unavailable"
    assert "gfs_high_f" in unavailable["reason"]


def _policy_row(
    year: int,
    bucket_delta: int,
    risk_probability: float,
    relation_probabilities: list[float],
) -> dict[str, object]:
    relation_class = 0 if bucket_delta < 0 else 2 if bucket_delta > 0 else 1
    return {
        "validation_year": year,
        "bucket_delta": bucket_delta,
        "point_bucket_wrong": int(bucket_delta != 0),
        "bucket_relation_class": relation_class,
        "point_bucket_index": 46,
        "observed_high_temp_through_as_of_f": 89.0,
        "risk_probability": risk_probability,
        "relation_probabilities": relation_probabilities,
    }
