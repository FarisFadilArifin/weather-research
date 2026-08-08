from __future__ import annotations

import json
import math

import joblib
import numpy as np
import pandas as pd
import pytest

from src.calibration.bucket_probability import canonical_two_degree_bucket
from src.calibration.continuous_residual_probability import (
    DistributionPrediction,
    assert_cutoffs,
    boundary_band,
    bucket_log_loss,
    common_date_comparison,
    continuous_residual,
    fit_distribution,
    integrate_settlement_degrees,
    multiclass_brier,
    predict_continuous_bundle,
    predict_distributions,
    prepare_probability_frame,
    quantile_crps,
    ranked_probability_score,
    reliability_rows,
    settlement_interval,
    strict_json_data,
)


def test_continuous_residual_is_not_rounded() -> None:
    assert continuous_residual(91.0, 89.37) == pytest.approx(1.63)


def test_half_up_settlement_geometry_at_exact_boundaries() -> None:
    assert settlement_interval(91) == (90.5, 91.5)
    prediction = DistributionPrediction("gaussian", 91.5, 0.01, {})
    probabilities = integrate_settlement_degrees(0.0, prediction)
    assert probabilities[91] == pytest.approx(0.5, abs=1e-8)
    assert probabilities[92] == pytest.approx(0.5, abs=1e-8)


def test_degree_and_bucket_probabilities_sum_to_one() -> None:
    prediction = DistributionPrediction("gaussian", 0.0, 1.2, {})
    degrees = integrate_settlement_degrees(80.2, prediction)
    buckets = {}
    for degree, probability in degrees.items():
        bucket = canonical_two_degree_bucket(degree)
        buckets[bucket] = buckets.get(bucket, 0.0) + probability
    assert sum(degrees.values()) == pytest.approx(1.0)
    assert sum(buckets.values()) == pytest.approx(1.0)
    assert all(math.isfinite(value) and value >= 0 for value in degrees.values())


def test_open_ended_tails_retain_probability_mass() -> None:
    prediction = DistributionPrediction("student_t", 0.0, 2.0, {"df": 3.0})
    degrees = integrate_settlement_degrees(80.0, prediction, tail_tolerance=1e-8)
    assert min(degrees) < 70 and max(degrees) > 90
    assert degrees[min(degrees)] > 0 and degrees[max(degrees)] > 0


def test_observed_high_floor_truncates_and_renormalizes() -> None:
    prediction = DistributionPrediction("gaussian", 0.0, 2.0, {})
    degrees = integrate_settlement_degrees(80.0, prediction, observed_high_f=82.1)
    assert min(degrees) == 82
    assert sum(degrees.values()) == pytest.approx(1.0)


def test_gaussian_candidate_has_reasonable_probabilities() -> None:
    prediction = DistributionPrediction("gaussian", 0.0, 1.0, {})
    degrees = integrate_settlement_degrees(80.0, prediction)
    assert degrees[80] == pytest.approx(0.3829249, rel=1e-5)
    assert degrees[79] == pytest.approx(degrees[81], rel=1e-10)


def test_empirical_candidate_behaves_on_toy_distribution() -> None:
    frame = _model_frame(60)
    frame["continuous_residual_f"] = np.tile([-1.0, 1.0], 30)
    state = fit_distribution("conditional_empirical", frame, bandwidth=0.2)
    prediction = predict_distributions(state, frame.iloc[:1])[0]
    assert prediction.location == pytest.approx(0.0, abs=0.2)
    assert prediction.state["weights"].sum() == pytest.approx(1.0)


def test_crps_known_degenerate_limit() -> None:
    prediction = DistributionPrediction("gaussian", 2.0, 1e-5, {})
    assert quantile_crps(prediction, 5.0) == pytest.approx(3.0, abs=1e-4)


def test_bucket_log_loss_and_brier_known_examples() -> None:
    probabilities = {"80-81": 0.75, "82-83": 0.25}
    assert bucket_log_loss("80-81", probabilities) == pytest.approx(-math.log(0.75))
    assert multiclass_brier("80-81", probabilities) == pytest.approx(0.125)


def test_ranked_probability_score_respects_ordering() -> None:
    labels = ["78-79", "80-81", "82-83"]
    near = {"78-79": 0.0, "80-81": 1.0, "82-83": 0.0}
    far = {"78-79": 1.0, "80-81": 0.0, "82-83": 0.0}
    assert ranked_probability_score("82-83", near, labels) < ranked_probability_score("82-83", far, labels)


def test_reliability_bin_aggregation_keeps_low_counts() -> None:
    frame = pd.DataFrame([{"station_id": "KATL", "validation_year": 2024, "model_family": "gaussian",
                           "actual_bucket": "80-81", "bucket_probabilities_json": json.dumps({"80-81": 0.7, "82-83": 0.3})}])
    result = reliability_rows(frame)
    assert result["sample_count"].sum() == 2
    assert result["low_count"].all()


def test_boundary_distance_bands() -> None:
    result = boundary_band(pd.Series([0.0, 0.099, 0.1, 0.249, 0.25, 0.5])).astype(str).tolist()
    assert result == ["[0.00,0.10)", "[0.00,0.10)", "[0.10,0.25)", "[0.10,0.25)", "[0.25,0.50]", "[0.25,0.50]"]


def test_missing_mandatory_features_fail_closed() -> None:
    features, point, base = _join_frames()
    features = features.drop(columns=["gfs_high_f"])
    with pytest.raises(ValueError, match="mandatory"):
        prepare_probability_frame(features, point, base, station_id="KATL", include_peak_features=False)


def test_kdal_cannot_enable_peak_features() -> None:
    features, point, base = _join_frames()
    with pytest.raises(ValueError, match="KDAL"):
        prepare_probability_frame(features, point, base, station_id="KDAL", include_peak_features=True)


def test_model_and_calibration_cutoffs_precede_dates() -> None:
    valid = pd.DataFrame({"contract_date": ["2024-01-01"], "model_training_cutoff": ["2023-12-31"],
                          "calibration_training_start": ["2023-10-01"], "calibration_training_cutoff": ["2023-12-31"]})
    assert_cutoffs(valid)
    invalid = valid.assign(model_training_cutoff="2024-01-01")
    with pytest.raises(AssertionError):
        assert_cutoffs(invalid)


def test_common_date_comparison_uses_intersection() -> None:
    rows = []
    for family, dates in {"a": ["2024-01-01", "2024-01-02"], "b": ["2024-01-02"]}.items():
        for date in dates:
            rows.append(_metric_row(family, date))
    result = common_date_comparison(pd.DataFrame(rows))
    assert set(result["count"]) == {1}


def test_exported_calibrated_bundle_reproduces_evaluation_probabilities(tmp_path) -> None:
    frame = _model_frame(60)
    state = fit_distribution("gaussian", frame)
    feature_values = frame.iloc[0].to_dict()
    bundle = {
        "model_state": state,
        "calibration_scale_multiplier": 1.25,
    }
    raw = predict_distributions(state, frame.iloc[:1])[0]
    calibrated = DistributionPrediction(raw.family, raw.location, raw.scale * 1.25, raw.state)
    expected = integrate_settlement_degrees(
        feature_values["point_prediction_f"],
        calibrated,
        observed_high_f=feature_values["observed_high_temp_through_as_of_f"],
    )
    path = tmp_path / "model.joblib"
    joblib.dump(bundle, path)
    result = predict_continuous_bundle(joblib.load(path), feature_values)
    assert result["calibration_scale_multiplier"] == 1.25
    assert result["predicted_residual_scale_f"] == pytest.approx(raw.scale * 1.25)
    assert result["degree_probabilities"] == pytest.approx(
        {str(key): value for key, value in expected.items()}
    )


def test_manifest_data_is_strict_json() -> None:
    cleaned = strict_json_data(
        {"continuous_nll": np.nan, "nested": [np.inf, np.float64(1.25)]}
    )
    encoded = json.dumps(cleaned, allow_nan=False)
    assert json.loads(encoded) == {
        "continuous_nll": None,
        "nested": [None, 1.25],
    }


def test_actual_source_values_are_not_silently_jittered() -> None:
    features, point, base = _join_frames()
    result = prepare_probability_frame(features, point, base, station_id="KATL", include_peak_features=False)
    assert result["actual_high_f"].tolist() == point["actual_high_f"].tolist()


def _model_frame(count: int) -> pd.DataFrame:
    dates = pd.date_range("2022-01-01", periods=count)
    x = np.linspace(-1, 1, count)
    return pd.DataFrame({"contract_date": dates, "month": dates.month, "point_prediction_f": 80 + x,
                         "provider_spread_high_f": 2 + x, "base_prediction_spread_f": 1 + x / 2,
                         "observed_temp_at_as_of_f": 70 + x, "observed_high_temp_through_as_of_f": 72 + x,
                         "point_rounding_remainder_f": x / 3, "point_distance_to_round_boundary_f": 0.5 - abs(x / 3),
                         "day_of_year_sin": np.sin(2 * np.pi * dates.dayofyear / 365.25),
                         "day_of_year_cos": np.cos(2 * np.pi * dates.dayofyear / 365.25), "continuous_residual_f": x})


def _join_frames():
    dates = pd.date_range("2023-01-01", periods=2)
    features = pd.DataFrame({"contract_date": dates, "station_id": "KATL", "gfs_high_f": [80, 81], "hrrr_high_f": [81, 82],
                             "nbm_high_f": [79, 80], "observed_temp_at_as_of_f": [70, 71],
                             "observed_high_temp_through_as_of_f": [72, 73], "observed_as_of_age_minutes": [2, 3]})
    point = pd.DataFrame({"contract_date": dates, "actual_high_f": [80.0, 82.0], "predicted_high_f": [80.25, 81.75]})
    base = pd.DataFrame([{"contract_date": date, "method": method, "predicted_high_f": 80 + index}
                         for date in dates for index, method in enumerate(("xgboost", "lightgbm", "catboost"))])
    return features, point, base


def _metric_row(family: str, date: str) -> dict:
    return {"model_family": family, "contract_date": date, "validation_year": 2024, "continuous_crps": 1.0,
            "continuous_nll": 1.0, "pit": 0.5, "predictive_mean_absolute_error_f": 1.0,
            "bucket_log_loss": 1.0, "bucket_brier": 1.0, "ranked_probability_score": 1.0,
            "top_bucket_accuracy": 1.0, "top_bucket_hit": True, "top_two_bucket_hit": True, "point_bucket_hit": True,
            "actual_bucket_probability": 0.5, "top_bucket_probability": 0.5}
