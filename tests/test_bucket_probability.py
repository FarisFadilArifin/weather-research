from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.calibration.bucket_probability import (
    CandidateSpec,
    FEATURE_PROFILE_ASIA_NO_PEAK,
    FEATURE_PROFILE_KDAL_1PM,
    FEATURE_PROFILE_EXPERT_ENSEMBLE_ASIA_NO_PEAK,
    FEATURE_PROFILE_EXPERT_ENSEMBLE_COMMON_NO_PEAK,
    EXPERT_ENSEMBLE_BASE_METHODS,
    OFFSET_LABELS,
    build_probability_frame,
    canonical_two_degree_bucket,
    expand_offset_probabilities,
    fit_probability_system,
    fit_tail_policy,
    offset_class_index,
    predict_probability_bundle,
    round_half_up,
    probability_feature_names,
    score_probabilities,
)


def test_asia_probability_profile_uses_gfs_gefs_and_jma_contract() -> None:
    features = probability_feature_names(
        include_peak_features=False,
        feature_profile=FEATURE_PROFILE_ASIA_NO_PEAK,
    )
    assert len(features) == 59
    assert "gfs_high_f" in features
    assert "gefs_high_f" in features
    assert "jma_msm_high_f" in features
    assert "gefs_minus_point_f" in features
    assert "jma_msm_minus_point_f" in features
    assert "hrrr_high_f" not in features
    assert "nbm_high_f" not in features
    assert not any("peak" in name.lower() for name in features)


def test_expert_ensemble_probability_profiles_have_61_inputs() -> None:
    common = probability_feature_names(
        include_peak_features=False,
        feature_profile=FEATURE_PROFILE_EXPERT_ENSEMBLE_COMMON_NO_PEAK,
    )
    asia = probability_feature_names(
        include_peak_features=False,
        feature_profile=FEATURE_PROFILE_EXPERT_ENSEMBLE_ASIA_NO_PEAK,
    )
    assert len(common) == len(asia) == 61
    for method in EXPERT_ENSEMBLE_BASE_METHODS:
        assert f"{method}_predicted_high_f" in common
        assert f"{method}_minus_point_f" in common
    assert "hrrr_minus_point_f" in common and "hrrr_minus_point_f" not in asia
    assert "jma_msm_minus_point_f" in asia and "jma_msm_minus_point_f" not in common


def test_expert_ensemble_probability_frame_uses_declared_four_methods() -> None:
    date = pd.Timestamp("2025-07-01")
    features = pd.DataFrame([{
        "contract_date": date,
        "gfs_high_f": 91.0,
        "hrrr_high_f": 92.0,
        "nbm_high_f": 90.0,
        "observed_temp_at_as_of_f": 84.0,
        "observed_high_temp_through_as_of_f": 85.0,
        "observed_as_of_age_minutes": 0.0,
    }])
    point = pd.DataFrame([{"contract_date": date, "actual_high_f": 92.0, "predicted_high_f": 91.5}])
    base = pd.DataFrame([
        {"contract_date": date, "method": method, "predicted_high_f": 90.0 + index}
        for index, method in enumerate(EXPERT_ENSEMBLE_BASE_METHODS)
    ])
    result = build_probability_frame(
        features, point, base, include_peak_features=False,
        feature_profile=FEATURE_PROFILE_EXPERT_ENSEMBLE_COMMON_NO_PEAK,
    )
    assert len(result) == 1
    assert math.isclose(result.loc[0, "base_prediction_mean_f"], 91.5)
    assert all(result.loc[0, f"{method}_predicted_high_f"] == 90.0 + index for index, method in enumerate(EXPERT_ENSEMBLE_BASE_METHODS))


def test_build_asia_probability_frame_uses_asia_mandatory_providers() -> None:
    dates = pd.date_range("2024-01-01", periods=2, freq="D")
    features = pd.DataFrame(
        {
            "contract_date": dates,
            "gfs_high_f": [50.0, 51.0],
            "gefs_high_f": [49.0, 50.0],
            "jma_msm_high_f": [51.0, np.nan],
            "observed_temp_at_as_of_f": [45.0, 46.0],
            "observed_high_temp_through_as_of_f": [46.0, 47.0],
            "observed_as_of_age_minutes": [0.0, 1.0],
        }
    )
    point = pd.DataFrame(
        {
            "contract_date": dates,
            "actual_high_f": [51.0, 52.0],
            "predicted_high_f": [50.6, 51.4],
        }
    )
    base = pd.DataFrame(
        [
            {
                "contract_date": date,
                "method": method,
                "predicted_high_f": 50.0 + index,
            }
            for date in dates
            for index, method in enumerate(
                ("xgboost", "lightgbm", "catboost")
            )
        ]
    )
    result = build_probability_frame(
        features,
        point,
        base,
        include_peak_features=False,
        feature_profile=FEATURE_PROFILE_ASIA_NO_PEAK,
    )
    assert len(result) == 1
    assert math.isclose(result.iloc[0]["gefs_minus_point_f"], -1.6)
    assert math.isclose(result.iloc[0]["jma_msm_minus_point_f"], 0.4)


def test_kdal_1pm_probability_profile_uses_1pm_features_without_11am_or_peak_leakage() -> None:
    features = set(
        probability_feature_names(
            include_peak_features=False,
            feature_profile=FEATURE_PROFILE_KDAL_1PM,
        )
    )
    assert "v13sf_forecast_warmup_after_1pm_f" in features
    assert "v13sf_forecast_temp_1pm_minus_observed_f" in features
    assert "observed_temp_change_since_11am_f" in features
    assert "observed_high_so_far_change_since_11am_f" in features
    assert not any(name.startswith("v11sf_") for name in features)
    assert not any("peak_timing" in name for name in features)


def test_half_up_rounding_and_offset_example() -> None:
    assert round_half_up(91.6) == 92
    assert round_half_up(91.5) == 92
    assert round_half_up(91.4) == 91
    exact_offset = round_half_up(91.0) - round_half_up(91.6)
    assert exact_offset == -1
    assert OFFSET_LABELS[offset_class_index(exact_offset)] == "-1"
    assert canonical_two_degree_bucket(round_half_up(91.0)) == "90-91"


def test_tail_expansion_and_physical_floor_sum_to_one() -> None:
    tail = fit_tail_policy([-8, -5, -4, -3, 0, 1, 4, 5, 7])
    class_probabilities = np.asarray([0.10, 0.05, 0.10, 0.20, 0.30, 0.10, 0.05, 0.04, 0.06])
    degrees = expand_offset_probabilities(
        92,
        class_probabilities,
        tail,
        observed_high_f=91.6,
    )
    assert min(degrees) >= 92
    assert math.isclose(sum(degrees.values()), 1.0, abs_tol=1e-12)
    assert all(math.isfinite(value) and value >= 0 for value in degrees.values())


def test_ranked_probability_score_penalizes_farther_ordered_misses() -> None:
    actual = np.asarray([4])
    near = np.zeros((1, len(OFFSET_LABELS)))
    far = np.zeros((1, len(OFFSET_LABELS)))
    near[0, 3] = 1.0
    far[0, 0] = 1.0
    assert (
        score_probabilities(actual, near)["ranked_probability_score"]
        < score_probabilities(actual, far)["ranked_probability_score"]
    )


def test_forced_family_must_be_present_and_blend_weights_are_bounded() -> None:
    frame = pd.DataFrame()
    with np.testing.assert_raises_regex(ValueError, "absent from candidate_specs"):
        fit_probability_system(
            frame,
            station_id="KDAL",
            point_model_version="point-v20",
            point_bundle_sha256="a" * 64,
            include_peak_features=False,
            candidate_specs=[CandidateSpec("empirical", {"prior_strength": 30.0})],
            forced_family="ordinal_logistic",
        )
    with np.testing.assert_raises_regex(ValueError, "blend_weights"):
        fit_probability_system(
            frame,
            station_id="KDAL",
            point_model_version="point-v20",
            point_bundle_sha256="a" * 64,
            include_peak_features=False,
            candidate_specs=[CandidateSpec("empirical", {"prior_strength": 30.0})],
            blend_weights=(0.0,),
        )


def test_build_probability_frame_uses_honest_point_and_base_rows() -> None:
    dates = pd.date_range("2023-01-01", periods=2, freq="D")
    features = pd.DataFrame(
        {
            "contract_date": dates,
            "gfs_high_f": [90, 91],
            "hrrr_high_f": [91, 92],
            "nbm_high_f": [89, 90],
            "observed_temp_at_as_of_f": [80, 81],
            "observed_high_temp_through_as_of_f": [82, 83],
            "observed_as_of_age_minutes": [0, 1],
        }
    )
    point = pd.DataFrame(
        {
            "contract_date": dates,
            "actual_high_f": [91, 92],
            "predicted_high_f": [91.6, 91.2],
        }
    )
    base = pd.DataFrame(
        [
            {"contract_date": date, "method": method, "predicted_high_f": 90 + index}
            for date in dates
            for index, method in enumerate(("xgboost", "lightgbm", "catboost"))
        ]
    )
    result = build_probability_frame(features, point, base, include_peak_features=False)
    assert len(result) == 2
    assert result.loc[0, "exact_offset"] == -1
    assert result.loc[0, "offset_class"] == offset_class_index(-1)
    assert result["point_rounding_remainder_f"].notna().all()
    assert math.isclose(
        result.loc[0, "point_signed_distance_to_round_boundary_f"], 0.1
    )
    assert result.loc[0, "observed_cloud_cover_at_as_of__missing"] == 1.0


def test_build_probability_frame_excludes_rows_runtime_cannot_serve() -> None:
    dates = pd.date_range("2023-01-01", periods=2, freq="D")
    features = pd.DataFrame(
        {
            "contract_date": dates,
            "gfs_high_f": [90.0, np.nan],
            "hrrr_high_f": [91.0, 92.0],
            "nbm_high_f": [89.0, 90.0],
            "observed_temp_at_as_of_f": [80.0, 81.0],
            "observed_high_temp_through_as_of_f": [82.0, 83.0],
            "observed_as_of_age_minutes": [0.0, 1.0],
        }
    )
    point = pd.DataFrame(
        {
            "contract_date": dates,
            "actual_high_f": [91.0, 92.0],
            "predicted_high_f": [91.6, 91.2],
        }
    )
    base = pd.DataFrame(
        [
            {"contract_date": date, "method": method, "predicted_high_f": 90.0 + index}
            for date in dates
            for index, method in enumerate(("xgboost", "lightgbm", "catboost"))
        ]
    )

    result = build_probability_frame(features, point, base, include_peak_features=False)

    assert result["contract_date"].tolist() == [dates[0]]


def test_small_empirical_system_fails_closed_for_missing_mandatory_feature() -> None:
    rows = []
    for date in pd.date_range("2023-01-01", "2025-12-31", freq="D"):
        point = 80.0 + math.sin(date.dayofyear / 20)
        actual = round_half_up(point) + ((date.dayofyear % 3) - 1)
        rows.append(
            {
                "contract_date": date,
                "actual_high_f": actual,
                "actual_degree_f": actual,
                "point_prediction_f": point,
                "point_degree_f": round_half_up(point),
                "observed_high_temp_through_as_of_f": point - 3,
                "exact_offset": actual - round_half_up(point),
                "offset_class": offset_class_index(actual - round_half_up(point)),
                "year": date.year,
                "month": date.month,
            }
        )
    frame = pd.DataFrame(rows)
    bundle, oof, _ = fit_probability_system(
        frame,
        station_id="KATL",
        point_model_version="point-v20",
        point_bundle_sha256="a" * 64,
        include_peak_features=False,
        candidate_specs=[CandidateSpec("empirical", {"prior_strength": 30.0})],
        min_train_rows=180,
    )
    assert set(oof["validation_year"]) == {2024, 2025}
    validation_dates = pd.to_datetime(oof["contract_date"])
    assert (pd.to_datetime(oof["model_training_cutoff"]) < validation_dates).all()
    assert (
        pd.to_datetime(oof["calibration_training_cutoff"])
        < pd.to_datetime(oof["calibration_validation_start"])
    ).all()
    assert (
        pd.to_datetime(oof["calibration_validation_cutoff"]) < validation_dates
    ).all()
    result = predict_probability_bundle(
        bundle,
        {
            "contract_date": "2026-01-01",
            "point_prediction_f": 91.6,
            "xgboost_predicted_high_f": 91.5,
            "lightgbm_predicted_high_f": 91.8,
            "catboost_predicted_high_f": 91.7,
        },
    )
    assert result["status"] == "unavailable"
    assert "gfs_high_f" in result["reason"]
