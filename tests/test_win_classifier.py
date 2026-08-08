from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.calibration.win_classifier import (
    CandidateSpec,
    KATL_PEAK_FEATURES,
    add_strict_history_features,
    add_win_geometry_features,
    expected_calibration_error,
    fit_win_classifier_system,
    select_confidence_threshold,
    win_feature_names,
    win_feature_names_for_profile,
)


def test_bucket_geometry_respects_half_up_and_two_degree_cells() -> None:
    frame = pd.DataFrame(
        {
            "point_prediction_f": [91.49, 91.50, 92.49, 92.50],
            "rounded_point_degree_f": [91, 92, 92, 93],
        }
    )
    result = add_win_geometry_features(frame)
    assert result["point_bucket_lower_degree_f"].tolist() == [90, 92, 92, 92]
    assert result["point_bucket_upper_degree_f"].tolist() == [91, 93, 93, 93]
    assert result["point_degree_is_bucket_upper"].tolist() == [1.0, 0.0, 0.0, 1.0]
    assert result.loc[1, "point_distance_to_bucket_lower_edge_f"] == pytest.approx(0.0)
    assert result.loc[3, "point_distance_to_bucket_upper_edge_f"] == pytest.approx(1.0)


def test_history_features_are_strictly_prior() -> None:
    frame = pd.DataFrame(
        {
            "contract_date": pd.date_range("2025-01-01", periods=40),
            "actual_high_f": np.arange(40, dtype=float) + 2.0,
            "point_prediction_f": np.arange(40, dtype=float),
            "bucket_win": [0, 1] * 20,
        }
    )
    result = add_strict_history_features(frame)
    assert pd.isna(result.loc[0, "prior_residual_bias_7d_f"])
    assert result.loc[10, "prior_residual_bias_7d_f"] == pytest.approx(2.0)
    assert result.loc[30, "prior_bucket_win_rate_30d"] == pytest.approx(0.5)
    changed = frame.copy()
    changed.loc[30, ["actual_high_f", "bucket_win"]] = [999.0, 1]
    changed_result = add_strict_history_features(changed)
    assert changed_result.loc[30, "prior_residual_bias_30d_f"] == result.loc[30, "prior_residual_bias_30d_f"]
    assert changed_result.loc[30, "prior_bucket_win_rate_30d"] == result.loc[30, "prior_bucket_win_rate_30d"]


def test_kdal_inventory_excludes_every_peak_feature() -> None:
    no_peak = set(win_feature_names(include_peak_features=False))
    assert no_peak.isdisjoint(KATL_PEAK_FEATURES)
    assert set(KATL_PEAK_FEATURES).issubset(win_feature_names(include_peak_features=True))


def test_kdal_1pm_inventory_excludes_11am_alignment_and_includes_1pm() -> None:
    names = set(
        win_feature_names_for_profile(
            include_peak_features=False, feature_profile="kdal_1pm"
        )
    )
    assert not any(name.startswith("v11sf_") for name in names)
    assert "v13sf_forecast_temp_1pm_minus_observed_f" in names
    assert "observed_temp_change_since_11am_f" in names


def test_confidence_threshold_is_frozen_from_forward_metrics() -> None:
    metrics = pd.DataFrame(
        {
            "threshold": [0.50, 0.55, 0.60],
            "selected_count": [100, 80, 60],
            "coverage": [0.50, 0.40, 0.30],
            "realized_win_rate": [0.65, 0.70, 0.75],
        }
    )
    policy = select_confidence_threshold(metrics, minimum_selected=50)
    assert policy["threshold"] == pytest.approx(0.55)
    assert policy["target_supported_forward"] is True
    assert policy["holdout_rows_used_for_selection"] == 0


def test_expected_calibration_error_is_zero_for_matching_bins() -> None:
    actual = np.array([0, 0, 1, 1])
    probability = np.array([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error(actual, probability) == pytest.approx(0.0)


def test_fit_uses_strict_chronological_cutoffs_and_rejects_kdal_peak() -> None:
    dates = pd.date_range("2023-01-01", "2025-12-31", freq="D")
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({"contract_date": dates})
    frame["year"] = frame["contract_date"].dt.year
    frame["month"] = frame["contract_date"].dt.month
    frame["bucket_win"] = (rng.random(len(frame)) < 0.5).astype(int)
    frame["actual_high_f"] = 80.0 + frame["bucket_win"]
    frame["point_prediction_f"] = 80.0
    frame["point_degree_f"] = 80
    frame["actual_degree_f"] = frame["actual_high_f"].astype(int)
    frame["point_bucket_label"] = "80-81"
    frame["actual_bucket_label"] = "80-81"
    for feature in win_feature_names(include_peak_features=False):
        frame[feature] = rng.normal(size=len(frame))
    specs = [CandidateSpec("logistic", {"C": 0.1}, "platt")]
    bundle, forward, _, _ = fit_win_classifier_system(
        frame,
        station_id="KATL",
        point_model_version="test_point",
        point_bundle_sha256="abc",
        include_peak_features=False,
        candidate_specs=specs,
    )
    assert set(forward["validation_year"]) == {2024, 2025}
    assert (
        pd.to_datetime(forward["calibration_cutoff"])
        < pd.to_datetime(forward["contract_date"])
    ).all()
    assert bundle["training_cutoff"] == "2025-12-31"
    with pytest.raises(ValueError, match="no-peak"):
        fit_win_classifier_system(
            frame,
            station_id="KDAL",
            point_model_version="test_point",
            point_bundle_sha256="abc",
            include_peak_features=True,
            candidate_specs=specs,
        )
