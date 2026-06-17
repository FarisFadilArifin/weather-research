from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from src import export_station_stacking_v2_models
from src.calibration import station_stacking
from src.calibration.dataset import CALIBRATION_COLUMNS, WEATHER_NUMERIC_COLUMNS
from src.calibration.modeling import _feature_columns
from src.calibration.station_stacking import (
    OBSERVED_CATEGORICAL_FEATURES,
    OBSERVED_NUMERIC_COLUMNS,
    PROVIDER_NUMERIC_COLUMNS,
    StationStackingConfig,
    build_station_wide_dataset,
    feature_columns,
    load_current_observation_features,
    load_same_day_provider_forecasts,
    missing_expected_model_methods,
    provider_availability,
    raw_baseline_predictions,
    tune_year_split_base_models,
    year_split_feature_importance,
    year_split_test_predictions,
)


def test_modeling_frame_quarantines_bad_labels_and_forecasts() -> None:
    frame = pd.DataFrame(
        [
            {
                "contract_date": "2026-01-01",
                "actual_high_f": 80.0,
                "actual_data_quality_flag": "ok",
                "actual_raw_observation_count": 24,
                "observed_temp_at_as_of_f": 72.0,
                "observed_high_temp_through_as_of_f": 73.0,
                "observed_fetch_status": "ok",
                "observed_as_of_age_minutes": 5,
                "gfs_high_f": 79.0,
                "hrrr_high_f": 81.0,
                "all_provider_highs_available": True,
            },
            {
                "contract_date": "2026-01-02",
                "actual_high_f": 80.0,
                "actual_data_quality_flag": "sparse_observations",
                "actual_raw_observation_count": 4,
                "observed_temp_at_as_of_f": 72.0,
                "observed_high_temp_through_as_of_f": 73.0,
                "observed_fetch_status": "ok",
                "observed_as_of_age_minutes": 5,
                "gfs_high_f": 79.0,
                "hrrr_high_f": 81.0,
                "all_provider_highs_available": True,
            },
            {
                "contract_date": "2026-01-03",
                "actual_high_f": 72.0,
                "actual_data_quality_flag": "ok",
                "actual_raw_observation_count": 24,
                "observed_temp_at_as_of_f": 71.0,
                "observed_high_temp_through_as_of_f": 74.0,
                "observed_fetch_status": "ok",
                "observed_as_of_age_minutes": 5,
                "gfs_high_f": 73.0,
                "hrrr_high_f": 74.0,
                "all_provider_highs_available": True,
            },
            {
                "contract_date": "2026-01-04",
                "actual_high_f": 80.0,
                "actual_data_quality_flag": "ok",
                "actual_raw_observation_count": 24,
                "observed_temp_at_as_of_f": 72.0,
                "observed_high_temp_through_as_of_f": 73.0,
                "observed_fetch_status": "ok",
                "observed_as_of_age_minutes": 5,
                "gfs_high_f": 274.0,
                "hrrr_high_f": 81.0,
                "all_provider_highs_available": True,
            },
        ]
    )
    config = StationStackingConfig(station_id="KATL", providers=("gfs", "hrrr"))

    modeling_frame, _, _ = station_stacking._modeling_frame(frame, config)

    assert list(modeling_frame["contract_date"]) == ["2026-01-01"]


FORECAST_AT_AS_OF_COLUMNS = {
    "forecast_temp_at_as_of_f",
    "dewpoint_at_as_of_f",
    "humidity_at_as_of",
    "wind_speed_at_as_of",
    "wind_direction_at_as_of",
}

EXPERIMENTAL_FORECAST_COLUMNS = {
    "cloud_cover_mean",
    "cloud_cover_max",
    "pressure_mslp_mean",
    "pressure_surface_mean",
    "visibility_mean",
    "ceiling_min",
}


def test_station_stacking_config_accepts_mae_optuna_metric() -> None:
    assert StationStackingConfig(station_id="KATL", optuna_metric="mae").effective_optuna_metric == "mae_f"
    assert StationStackingConfig(station_id="KATL", optuna_metric="rmse").effective_optuna_metric == "rmse_f"
    with pytest.raises(ValueError, match="optuna_metric"):
        StationStackingConfig(station_id="KATL", optuna_metric="mape").effective_optuna_metric


def test_station_stacking_config_accepts_remaining_warmup_target_mode() -> None:
    assert StationStackingConfig(station_id="KATL").effective_target_mode == "actual_high"
    assert (
        StationStackingConfig(station_id="KATL", target_mode="remaining_warmup").effective_target_mode
        == "remaining_warmup"
    )
    with pytest.raises(ValueError, match="target_mode"):
        StationStackingConfig(station_id="KATL", target_mode="afternoon_magic").effective_target_mode


def test_station_stacking_config_accepts_v8_v9_v10_and_v11_feature_versions() -> None:
    assert StationStackingConfig(station_id="KATL", feature_version="v8").effective_feature_version == "v8"
    assert StationStackingConfig(station_id="KATL", feature_version="v9").effective_feature_version == "v9"
    assert StationStackingConfig(station_id="KATL", feature_version="v10").effective_feature_version == "v10"
    assert StationStackingConfig(station_id="KATL", feature_version="v11").effective_feature_version == "v11"
    assert (
        StationStackingConfig(station_id="KATL", output_dir="calibration_v10", feature_version="v10")
        .resolved_optuna_storage_path()
        .name
        == "KATL_optuna.sqlite3"
    )
    assert (
        StationStackingConfig(station_id="KATL", output_dir="calibration_v11", feature_version="v11")
        .resolved_optuna_storage_path()
        .name
        == "KATL_optuna.sqlite3"
    )


def test_station_stacking_config_accepts_catboost_only_methods() -> None:
    config = StationStackingConfig(
        station_id="KATL",
        base_model_methods=("catboost", "catboost"),
        stack_enabled=False,
    )

    assert config.effective_base_model_methods == ("catboost",)
    assert not config.stack_enabled
    with pytest.raises(ValueError, match="base_model_methods"):
        StationStackingConfig(station_id="KATL", base_model_methods=("random_forest",)).effective_base_model_methods


def test_selected_hyperparameters_can_sort_by_mae() -> None:
    tuning = pd.DataFrame(
        [
            {
                "method": "xgboost",
                "param_key": "lower_mae",
                "fold": "fold_2024",
                "fold_weight": 1.0,
                "status": "ok",
                "mae_f": 1.0,
                "rmse_f": 10.0,
                "param_max_depth": 2,
            },
            {
                "method": "xgboost",
                "param_key": "lower_rmse",
                "fold": "fold_2024",
                "fold_weight": 1.0,
                "status": "ok",
                "mae_f": 2.0,
                "rmse_f": 3.0,
                "param_max_depth": 3,
            },
        ]
    )

    by_rmse = station_stacking._selected_hyperparameters(tuning, metric_col="rmse_f")
    by_mae = station_stacking._selected_hyperparameters(tuning, metric_col="mae_f")

    assert by_rmse.iloc[0]["param_key"] == "lower_rmse"
    assert by_mae.iloc[0]["param_key"] == "lower_mae"


def test_remaining_warmup_target_mode_fits_remaining_and_returns_high(monkeypatch) -> None:
    seen_targets = []

    class RemainingWarmupEstimator:
        def fit(self, x, y):
            seen_targets.append(list(pd.Series(y)))
            return self

        def predict(self, x):
            return [2.0] * len(x)

    monkeypatch.setattr(
        station_stacking,
        "_build_base_model_estimator",
        lambda *args, **kwargs: RemainingWarmupEstimator(),
    )
    train = pd.DataFrame(
        {
            "actual_high_f": [80.0, 82.0],
            "observed_high_temp_through_as_of_f": [75.0, 76.0],
            "driver_feature": [1.0, 2.0],
        }
    )
    valid = pd.DataFrame(
        {
            "actual_high_f": [83.0],
            "observed_high_temp_through_as_of_f": [78.0],
            "driver_feature": [3.0],
        }
    )
    config = StationStackingConfig(station_id="KATL", target_mode="remaining_warmup")

    predicted, metadata = station_stacking._fit_predict_base_model(
        config=config,
        categorical=[],
        numeric=["driver_feature"],
        method="xgboost",
        params={},
        train=train,
        valid=valid,
        early_stopping=False,
    )

    assert seen_targets == [[5.0, 6.0]]
    assert list(predicted) == [80.0]
    assert metadata["target_mode"] == "remaining_warmup"
    assert metadata["model_target"] == station_stacking.REMAINING_WARMUP_TARGET


def _write_station_stacking_fixture(root: Path, days: int = 8) -> None:
    processed = root / "data" / "processed"
    calibration = root / "data" / "calibration"
    processed.mkdir(parents=True)
    calibration.mkdir(parents=True)
    dates = pd.date_range("2026-01-01", periods=days, freq="D").date
    pd.DataFrame(
        {
            "station_code": ["KATL"],
            "station_name": ["Atlanta/Hartsfield-Jackson Intl"],
            "airport_name": ["Atlanta/Hartsfield-Jackson Intl"],
            "city_label": ["Atlanta"],
            "lat": [33.62972],
            "lon": [-84.44223],
            "timezone": ["America/New_York"],
            "country": ["US"],
        }
    ).to_csv(processed / "station_registry.csv", index=False)
    pd.DataFrame(
        {
            "station_code": ["KATL"] * days,
            "date_local": [day.isoformat() for day in dates],
            "actual_high_f": [70 + i for i in range(days)],
        }
    ).to_csv(processed / "actual_highs.csv", index=False)

    offsets = {"gfs": -1.0, "hrrr": 0.0, "nbm": 1.0}
    for provider, offset in offsets.items():
        cache_dir = calibration / f"sdk_11am_{provider}_fixture"
        cache_file = "sdk_nwp_0h_cache.csv"
        if provider == "nbm":
            cache_dir = calibration / "direct_nbm_fixture"
            cache_file = "direct_nbm_0h_cache.csv"
        cache_dir.mkdir()
        rows = []
        for i, day in enumerate(dates):
            rows.append(
                {
                    "station_id": "KATL",
                    "station_name": "Atlanta/Hartsfield-Jackson Intl",
                    "airport_name": "Atlanta/Hartsfield-Jackson Intl",
                    "provider": provider,
                    "model": provider,
                    "source_label": f"fixture_{provider}",
                    "timing_mode": "same_day_11am",
                    "cycle_selection_policy": "fixture",
                    "contract_date": day.isoformat(),
                    "forecast_as_of": f"{day.isoformat()}T16:00:00+00:00",
                    "issued_at": f"{day.isoformat()}T12:00:00+00:00",
                    "forecast_window_start": f"{day.isoformat()}T16:00:00+00:00",
                    "forecast_window_end": f"{day.isoformat()}T23:00:00+00:00",
                    "horizon_hours": 0,
                    "raw_forecast_high_f": 70 + i + offset,
                    "forecast_hour_min": 4,
                    "forecast_hour_max": 16,
                    "grid_dist_km_mean": 10,
                    "cloud_cover_mean": 40 + i,
                    "cloud_cover_max": 60 + i,
                    "precip_amount": 0.01 * i,
                    "forecast_precip_total_mm": 0.01 * i,
                    "forecast_precip_max_1h_mm": 0.005 * i,
                    "forecast_precip_hours_count": 1 if i else 0,
                    "forecast_has_precip": 1 if i else 0,
                    "forecast_precip_intensity_code": 1 if i else 0,
                    "forecast_precip_intensity": "light_rain" if i else "dry",
                    "wind_speed_mean": 5 + i,
                    "wind_speed_max": 8 + i,
                    "wind_speed_at_as_of": 6 + i,
                    "wind_direction_mean": 180,
                    "wind_direction_at_as_of": 190,
                    "wind_gust_max": 12 + i,
                    "dewpoint_mean_f": 50,
                    "dewpoint_at_as_of_f": 49,
                    "humidity_mean": 55,
                    "humidity_at_as_of": 56,
                    "data_source": "fixture",
                    "source_file_or_url": "fixture",
                    "fetch_status": "ok",
                    "unavailable_reason": "",
                }
            )
        pd.DataFrame(rows).to_csv(cache_dir / cache_file, index=False)

    obs_dir = calibration / "sdk_current_obs_fixture"
    obs_dir.mkdir()
    obs_rows = []
    for i, day in enumerate(dates):
        obs_rows.append(
            {
                "station_id": "KATL",
                "station_name": "Atlanta/Hartsfield-Jackson Intl",
                "airport_name": "Atlanta/Hartsfield-Jackson Intl",
                "contract_date": day.isoformat(),
                "timing_mode": "same_day_11am",
                "observed_temp_at_as_of_f": 70 + i,
                "observed_high_temp_through_as_of_f": 70 + i,
                "observed_dewpoint_at_as_of_f": 60 + i,
                "observed_humidity_at_as_of": 55,
                "observed_wind_speed_at_as_of": 8,
                "observed_wind_direction_at_as_of": 180,
                "observed_wind_gust_at_as_of": 12,
                "observed_peak_wind_gust_at_as_of": 15,
                "observed_peak_wind_direction_at_as_of": 190,
                "observed_peak_wind_time_utc": f"{day.isoformat()}T15:30:00Z",
                "observed_pressure_at_as_of": 1012,
                "observed_pressure_source": "sea_level_pressure_mb",
                "observed_altimeter_inhg_at_as_of": 29.9,
                "observed_sea_level_pressure_mb_at_as_of": 1012,
                "observed_visibility_at_as_of": 2 if i == 0 else 10,
                "observed_ceiling_at_as_of": 1500,
                "observed_cloud_cover_at_as_of": 75,
                "observed_weather_code_at_as_of": "-RA BR" if i == 0 else "",
                "observed_precip_recent_at_as_of": 0.02 if i == 0 else 0,
                "observed_snow_depth_at_as_of": "",
                "observed_temp_change_last_1h_f": 1.0 + i,
                "observed_temp_change_last_3h_f": 3.0 + i,
                "observed_morning_warmup_rate_f_per_hour": 2.0 + (0.1 * i),
                "observed_high_so_far_change_since_9am_f": 4.0 + i,
                "observed_as_of_time_local": f"{day.isoformat()}T10:55:00-05:00",
                "observed_as_of_time_utc": f"{day.isoformat()}T15:55:00Z",
                "observed_as_of_age_minutes": 5,
                "observed_source": "fixture",
                "observed_observation_type": "METAR",
                "observed_qc_field": "",
                "observed_raw_metar": "fixture metar",
                "observed_data_source": "fixture",
                "observed_fetch_status": "ok",
                "observed_unavailable_reason": "",
            }
        )
    pd.DataFrame(obs_rows).to_csv(obs_dir / "sdk_current_observations_11am.csv", index=False)


def test_station_wide_features_are_provider_wide_and_lag_safe(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("WEATHER_RESEARCH_INCLUDE_DIRECT_NBM", "1")
    _write_station_stacking_fixture(tmp_path)

    frame = build_station_wide_dataset(tmp_path, station_id="KATL", providers=("gfs", "hrrr", "nbm"))

    assert frame["all_provider_highs_available"].all()
    assert {"gfs_high_f", "hrrr_high_f", "nbm_high_f"}.issubset(frame.columns)
    for provider in ("gfs", "hrrr", "nbm"):
        expected = {
            f"{provider}_{column}"
            for column in PROVIDER_NUMERIC_COLUMNS
            if column != "raw_forecast_high_f"
        }
        assert expected.issubset(frame.columns)
        excluded = {f"{provider}_{column}" for column in FORECAST_AT_AS_OF_COLUMNS}
        assert excluded.isdisjoint(frame.columns)
    assert "gfs_error_f" not in frame.columns
    assert frame.loc[1, "actual_high_lag_1d"] == 70
    assert frame.loc[1, "gfs_error_lag_1d_f"] == 1
    assert frame.loc[1, "provider_mean_high_f"] == 71
    assert frame.loc[1, "gfs_minus_actual_high_lag_1d_f"] == 0
    assert frame.loc[0, "gfs_prior_month_bias_f"] != frame.loc[0, "gfs_prior_month_bias_f"]
    assert frame.loc[1, "gfs_prior_month_bias_f"] != frame.loc[1, "gfs_prior_month_bias_f"]
    assert frame.loc[2, "gfs_prior_month_bias_f"] == 1
    assert frame.loc[2, "gfs_high_plus_prior_month_bias_f"] == frame.loc[2, "actual_high_f"]
    assert frame.loc[0, "gfs_hrrr_high_f_diff_f"] == -1
    assert frame.loc[0, "gfs_hrrr_high_f_abs_diff_f"] == 1
    assert frame.loc[0, "gfs_high_minus_observed_temp_f"] == -1
    assert frame.loc[0, "hrrr_high_minus_observed_temp_f"] == 0
    assert frame.loc[0, "hrrr_high_minus_observed_high_temp_f"] == 0
    assert "gfs_forecast_precip_max_1h_mm" in frame.columns
    assert "gfs_hrrr_forecast_precip_total_mm_abs_diff_f" in frame.columns


def test_station_wide_features_include_current_observations(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("WEATHER_RESEARCH_INCLUDE_DIRECT_NBM", "1")
    _write_station_stacking_fixture(tmp_path)

    observations = load_current_observation_features(tmp_path, station_id="KATL")
    frame = build_station_wide_dataset(tmp_path, station_id="KATL")
    config = StationStackingConfig(station_id="KATL", project_root=tmp_path)
    categorical, numeric = feature_columns(frame, config)

    assert len(observations) == 8
    assert set(OBSERVED_NUMERIC_COLUMNS).issubset(frame.columns)
    assert frame.loc[0, "observed_dewpoint_depression_f"] == 10
    assert frame.loc[0, "observed_high_temp_minus_temp_at_as_of_f"] == 0
    assert round(frame.loc[0, "observed_wind_dir_sin"], 6) == 0
    assert round(frame.loc[0, "observed_wind_dir_cos"], 6) == -1
    assert frame.loc[0, "observed_is_raining_at_as_of"]
    assert not frame.loc[0, "observed_is_drizzle_at_as_of"]
    assert not frame.loc[0, "observed_is_snowing_at_as_of"]
    assert frame.loc[0, "observed_is_fog_or_mist_at_as_of"]
    assert frame.loc[0, "observed_precip_intensity"] == "light"
    assert frame.loc[0, "observed_precip_intensity_code"] == 1
    assert frame.loc[0, "observed_temp_change_last_1h_f"] == 1.0
    assert frame.loc[0, "observed_temp_change_last_3h_f"] == 3.0
    assert frame.loc[0, "observed_morning_warmup_rate_f_per_hour"] == 2.0
    assert frame.loc[0, "observed_high_so_far_change_since_9am_f"] == 4.0
    assert frame.loc[1, "observed_temp_minus_actual_high_lag_1d_f"] == 1
    assert frame.loc[1, "observed_temp_minus_actual_high_roll_7d_mean_f"] == 1
    assert set(OBSERVED_CATEGORICAL_FEATURES).issubset(categorical)
    assert "observed_temp_at_as_of_f" in numeric
    assert "observed_high_temp_through_as_of_f" in numeric
    assert "observed_dewpoint_depression_f" in numeric
    assert "observed_temp_change_last_1h_f" not in numeric
    assert "observed_raw_metar" not in numeric


def test_station_feature_version_v5_matches_source_helper(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("WEATHER_RESEARCH_INCLUDE_DIRECT_NBM", "1")
    _write_station_stacking_fixture(tmp_path)

    base = build_station_wide_dataset(tmp_path, station_id="KATL", providers=("gfs", "hrrr", "nbm"))
    v5 = build_station_wide_dataset(
        tmp_path,
        station_id="KATL",
        providers=("gfs", "hrrr", "nbm"),
        feature_version="v5",
    )
    expected = station_stacking.add_v5_feature_engineering(base, providers=("gfs", "hrrr", "nbm"))

    assert set(station_stacking.V5_FEATURE_COLUMNS).issubset(v5.columns)
    pd.testing.assert_frame_equal(
        v5[station_stacking.V5_FEATURE_COLUMNS],
        expected[station_stacking.V5_FEATURE_COLUMNS],
        check_dtype=False,
    )


def test_station_feature_version_v6_enables_morning_trend_inputs(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("WEATHER_RESEARCH_INCLUDE_DIRECT_NBM", "1")
    _write_station_stacking_fixture(tmp_path)

    frame = build_station_wide_dataset(
        tmp_path,
        station_id="KATL",
        providers=("gfs", "hrrr", "nbm"),
        feature_version="v6",
    )
    config = StationStackingConfig(station_id="KATL", project_root=tmp_path, feature_version="v6")
    _, numeric = feature_columns(frame, config)
    trend_columns = {
        "observed_temp_change_last_1h_f",
        "observed_temp_change_last_3h_f",
        "observed_morning_warmup_rate_f_per_hour",
        "observed_high_so_far_change_since_9am_f",
    }

    assert set(station_stacking.V6_FEATURE_COLUMNS).issubset(frame.columns)
    assert trend_columns.issubset(numeric)


def test_live_safe_timing_loads_existing_11am_current_observation_trends(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("WEATHER_RESEARCH_INCLUDE_DIRECT_NBM", "1")
    _write_station_stacking_fixture(tmp_path)

    observations = load_current_observation_features(
        tmp_path,
        station_id="KATL",
        timing_mode="same_day_11am_live_safe",
    )

    assert len(observations) == 8
    assert observations.loc[0, "observed_temp_change_last_1h_f"] == 1.0
    assert observations.loc[0, "observed_morning_warmup_rate_f_per_hour"] == 2.0


def test_current_observation_loader_prefers_usable_row_over_newer_bad_cache(tmp_path) -> None:
    old_dir = tmp_path / "data" / "calibration" / "sdk_current_obs_2021_2026"
    new_dir = tmp_path / "data" / "calibration" / "sdk_current_obs_trends_KORD_20250101_20251231"
    old_dir.mkdir(parents=True)
    new_dir.mkdir(parents=True)
    columns = [
        "station_id",
        "contract_date",
        "timing_mode",
        "observed_temp_at_as_of_f",
        "observed_high_temp_through_as_of_f",
        "observed_as_of_time_local",
        "observed_as_of_age_minutes",
        "observed_fetch_status",
        "observed_raw_metar",
    ]
    pd.DataFrame(
        [
            {
                "station_id": "KORD",
                "contract_date": "2025-12-28",
                "timing_mode": "same_day_11am",
                "observed_temp_at_as_of_f": 54.0,
                "observed_high_temp_through_as_of_f": 54.0,
                "observed_as_of_time_local": "2025-12-28T10:50:00-06:00",
                "observed_as_of_age_minutes": 10,
                "observed_fetch_status": "ok",
                "observed_raw_metar": "older valid METAR",
            }
        ],
        columns=columns,
    ).to_csv(old_dir / "sdk_current_observations_11am.csv", index=False)
    pd.DataFrame(
        [
            {
                "station_id": "KORD",
                "contract_date": "2025-12-28",
                "timing_mode": "same_day_11am",
                "observed_temp_at_as_of_f": pd.NA,
                "observed_high_temp_through_as_of_f": 54.0,
                "observed_as_of_time_local": "2025-12-28T10:50:00-06:00",
                "observed_as_of_age_minutes": 10,
                "observed_fetch_status": "ok",
                "observed_raw_metar": "newer missing-temp METAR",
            }
        ],
        columns=columns,
    ).to_csv(new_dir / "sdk_current_observations_11am.csv", index=False)
    os.utime(old_dir / "sdk_current_observations_11am.csv", (100, 100))
    os.utime(new_dir / "sdk_current_observations_11am.csv", (200, 200))

    observations = load_current_observation_features(tmp_path, station_id="KORD")

    assert len(observations) == 1
    assert observations.loc[0, "observed_temp_at_as_of_f"] == 54.0
    assert observations.loc[0, "observed_raw_metar"] == "older valid METAR"


def test_station_feature_version_v7_keeps_morning_trend_inputs(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("WEATHER_RESEARCH_INCLUDE_DIRECT_NBM", "1")
    _write_station_stacking_fixture(tmp_path)

    frame = build_station_wide_dataset(
        tmp_path,
        station_id="KATL",
        providers=("gfs", "hrrr", "nbm"),
        feature_version="v7",
    )
    config = StationStackingConfig(station_id="KATL", project_root=tmp_path, feature_version="v7")
    _, numeric = feature_columns(frame, config)
    trend_columns = {
        "observed_temp_change_last_1h_f",
        "observed_temp_change_last_3h_f",
        "observed_morning_warmup_rate_f_per_hour",
        "observed_high_so_far_change_since_9am_f",
    }

    assert set(station_stacking.V7_FEATURE_COLUMNS).issubset(frame.columns)
    assert trend_columns.issubset(numeric)


def test_station_feature_version_v8_adds_warmup_features_and_drops_zero_importance_inputs(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("WEATHER_RESEARCH_INCLUDE_DIRECT_NBM", "1")
    _write_station_stacking_fixture(tmp_path)

    frame = build_station_wide_dataset(
        tmp_path,
        station_id="KATL",
        providers=("gfs", "hrrr", "nbm"),
        feature_version="v8",
    )
    config = StationStackingConfig(station_id="KATL", project_root=tmp_path, feature_version="v8")
    categorical, numeric = feature_columns(frame, config)
    selected_features = set(categorical) | set(numeric)
    trend_columns = {
        "observed_temp_change_last_1h_f",
        "observed_temp_change_last_3h_f",
        "observed_morning_warmup_rate_f_per_hour",
        "observed_high_so_far_change_since_9am_f",
    }

    assert set(station_stacking.V8_FEATURE_COLUMNS).issubset(frame.columns)
    assert trend_columns.issubset(numeric)
    assert selected_features.isdisjoint(station_stacking.V8_DROPPED_FEATURE_COLUMNS)
    assert "provider_mean_high_f" in selected_features
    assert "provider_spread_high_f" in selected_features
    assert "observed_high_temp_through_as_of_f" in selected_features


def test_station_feature_version_v9_adds_climatology_features_and_excludes_diagnostic(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("WEATHER_RESEARCH_INCLUDE_DIRECT_NBM", "1")
    _write_station_stacking_fixture(tmp_path)
    normals_dir = tmp_path / "outputs" / "climatology_all_stations"
    normals_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "station_code": ["KATL"] * 8,
            "target_year": [2026] * 8,
            "month_day": [f"01-{day:02d}" for day in range(1, 9)],
            "climatology_high_10y_f": [60.0 + day for day in range(8)],
            "climatology_high_10y_std_f": [5.0] * 8,
            "climatology_high_10y_count": [10] * 8,
            "climatology_source_start_year": [2016] * 8,
            "climatology_source_end_year": [2025] * 8,
        }
    ).to_csv(normals_dir / "station_rolling_10y_daily_high_normals.csv", index=False)

    frame = build_station_wide_dataset(
        tmp_path,
        station_id="KATL",
        providers=("gfs", "hrrr", "nbm"),
        feature_version="v9",
    )
    config = StationStackingConfig(station_id="KATL", project_root=tmp_path, feature_version="v9")
    categorical, numeric = feature_columns(frame, config)
    selected_features = set(categorical) | set(numeric)

    assert set(station_stacking.V9_FEATURE_COLUMNS).issubset(frame.columns)
    assert set(station_stacking.V9_CLIMATOLOGY_FEATURE_COLUMNS).issubset(numeric)
    assert selected_features.isdisjoint(station_stacking.V9_DROPPED_FEATURE_COLUMNS)
    assert "actual_minus_climatology_10y_f_DIAGNOSTIC_ONLY" in frame.columns

    v10_frame = build_station_wide_dataset(
        tmp_path,
        station_id="KATL",
        providers=("gfs", "hrrr", "nbm"),
        feature_version="v10",
    )
    v10_config = StationStackingConfig(station_id="KATL", project_root=tmp_path, feature_version="v10")
    v10_categorical, v10_numeric = feature_columns(v10_frame, v10_config)
    v10_selected = set(v10_categorical) | set(v10_numeric)

    assert station_stacking.V10_FEATURE_COLUMNS == station_stacking.V9_FEATURE_COLUMNS
    assert station_stacking.V10_DROPPED_FEATURE_COLUMNS == station_stacking.V9_DROPPED_FEATURE_COLUMNS
    assert set(station_stacking.V10_FEATURE_COLUMNS).issubset(v10_frame.columns)
    assert set(station_stacking.V9_CLIMATOLOGY_FEATURE_COLUMNS).issubset(v10_numeric)
    assert v10_selected.isdisjoint(station_stacking.V10_DROPPED_FEATURE_COLUMNS)

    v11_frame = build_station_wide_dataset(
        tmp_path,
        station_id="KATL",
        providers=("gfs", "hrrr", "nbm"),
        feature_version="v11",
    )
    v11_config = StationStackingConfig(station_id="KATL", project_root=tmp_path, feature_version="v11")
    v11_categorical, v11_numeric = feature_columns(v11_frame, v11_config)
    v11_selected = set(v11_categorical) | set(v11_numeric)

    assert station_stacking.V11_FEATURE_COLUMNS == station_stacking.V9_FEATURE_COLUMNS
    assert station_stacking.V11_DROPPED_FEATURE_COLUMNS == station_stacking.V9_DROPPED_FEATURE_COLUMNS
    assert set(station_stacking.V11_FEATURE_COLUMNS).issubset(v11_frame.columns)
    assert set(station_stacking.V9_CLIMATOLOGY_FEATURE_COLUMNS).issubset(v11_numeric)
    assert v11_selected.isdisjoint(station_stacking.V11_DROPPED_FEATURE_COLUMNS)
    assert "actual_minus_climatology_10y_f_DIAGNOSTIC_ONLY" not in selected_features
    assert frame.loc[0, "provider_mean_minus_climatology_10y_f"] == pytest.approx(
        frame.loc[0, "provider_mean_high_f"] - frame.loc[0, "climatology_high_10y_f"]
    )


def test_station_feature_version_v8_historical_warmup_features_are_shifted() -> None:
    frame = pd.DataFrame(
        {
            "month": [1, 1, 1, 1],
            "actual_high_f": [80.0, 82.0, 90.0, 100.0],
            "observed_high_temp_through_as_of_f": [70.0, 75.0, 80.0, 85.0],
            "provider_mean_high_f": [78.0, 83.0, 90.0, 92.0],
            "provider_max_high_f": [79.0, 84.0, 92.0, 95.0],
            "provider_min_high_f": [77.0, 81.0, 88.0, 89.0],
            "provider_median_high_f": [78.0, 82.0, 90.0, 91.0],
            "provider_spread_high_f": [2.0, 3.0, 4.0, 6.0],
        }
    )

    out = station_stacking.add_v8_feature_engineering(frame, providers=("gfs", "hrrr", "nbm"))

    assert pd.isna(out.loc[0, "v8_month_remaining_warmup_mean_f"])
    assert pd.isna(out.loc[1, "v8_month_remaining_warmup_mean_f"])
    assert out.loc[2, "v8_month_remaining_warmup_mean_f"] == 8.5
    assert out.loc[3, "v8_month_remaining_warmup_mean_f"] == 9.0
    assert out.loc[2, "v8_recent_remaining_warmup_7d_mean_f"] == 8.5
    assert out.loc[3, "v8_recent_remaining_warmup_7d_mean_f"] == 9.0
    assert list(out["v8_month_remaining_warmup_count"]) == [0.0, 1.0, 2.0, 3.0]


def test_optuna_sqlite_resume_treats_trials_as_target_total(tmp_path) -> None:
    pytest.importorskip("optuna")
    config = StationStackingConfig(
        station_id="KATL",
        output_dir=tmp_path,
        feature_version="v6",
        optuna_trials=5,
        optuna_startup_trials=30,
        stack_optuna_startup_trials=30,
    )
    study = station_stacking._create_optuna_study(config, "xgboost")
    assert getattr(study.sampler, "_n_startup_trials", None) == 30
    stack_study = station_stacking._create_stack_optuna_study(config)
    assert getattr(stack_study.sampler, "_n_startup_trials", None) == 30

    study.optimize(lambda trial: float(trial.number), n_trials=3, show_progress_bar=False)
    reloaded = station_stacking._create_optuna_study(config, "xgboost")

    lower_remaining = station_stacking._remaining_optuna_trials(reloaded, 2)
    assert lower_remaining == 0
    if lower_remaining > 0:
        reloaded.optimize(lambda trial: 0.0, n_trials=lower_remaining, show_progress_bar=False)
    assert len(station_stacking._study_trials(reloaded)) == 3

    remaining = station_stacking._remaining_optuna_trials(reloaded, config.effective_optuna_trials)
    assert remaining == 2
    reloaded.optimize(lambda trial: float(trial.number), n_trials=remaining, show_progress_bar=False)
    final = station_stacking._create_optuna_study(config, "xgboost")
    assert station_stacking._remaining_optuna_trials(final, config.effective_optuna_trials) == 0
    assert len(station_stacking._study_trials(final)) == 5


def test_optuna_trial_checkpoint_attrs_round_trip_sqlite(tmp_path) -> None:
    pytest.importorskip("optuna")
    config = StationStackingConfig(station_id="KAUS", output_dir=tmp_path, feature_version="v6")
    study = station_stacking._create_optuna_study(config, "lightgbm")
    trial = study.ask()
    rows = [
        {
            "method": "lightgbm",
            "trial_number": trial.number,
            "param_key": "trial_0",
            "fold": "fold_2021_2023_to_2024",
            "fold_weight": 0.35,
            "mae_f": 1.2,
            "rmse_f": 1.5,
            "count": 10,
            "status": "ok",
            "error": "",
        }
    ]
    fit_metadata = [{"fold": "fold_2021_2023_to_2024", "numeric_features": "x,y", "best_iteration": 7}]

    station_stacking._set_trial_checkpoint_attrs(
        trial,
        method="lightgbm",
        param_key="trial_0",
        params={"learning_rate": 0.1},
        rows=rows,
        fit_metadata=fit_metadata,
        status="ok",
        error="",
        objective_value=1.5,
    )
    study.tell(trial, 1.5)

    reloaded = station_stacking._create_optuna_study(config, "lightgbm")
    stored_trial = station_stacking._study_trials(reloaded)[0]
    attrs = stored_trial.user_attrs

    assert attrs["status"] == "ok"
    assert attrs["fold_metrics"][0]["mae_f"] == 1.2
    assert attrs["fold_metrics"][0]["fold_weight"] == 0.35
    assert attrs["fit_metadata"][0]["numeric_features"] == "x,y"
    assert attrs["objective_value"] == 1.5
    assert station_stacking._study_tuning_rows(reloaded)[0]["param_key"] == "trial_0"


def test_station_notebook_generators_use_source_owned_feature_versions() -> None:
    root = Path(__file__).resolve().parents[1]
    v5_source = (root / "notebooks" / "station_stacking_v5" / "generate_station_notebooks.py").read_text()
    v6_source = (root / "notebooks" / "station_stacking_v6" / "generate_station_notebooks.py").read_text()
    v7_source = (root / "notebooks" / "station_stacking_v7" / "generate_station_notebooks.py").read_text()
    v8_source = (root / "notebooks" / "station_stacking_v8" / "generate_station_notebooks.py").read_text()
    v9_source = (root / "notebooks" / "station_stacking_v9" / "generate_station_notebooks.py").read_text()
    v10_source = (root / "notebooks" / "station_stacking_v10" / "generate_station_notebooks.py").read_text()
    v11_source = (root / "notebooks" / "station_stacking_v11" / "generate_station_notebooks.py").read_text()

    assert 'feature_version="v5"' in v5_source
    assert 'feature_version="v6"' in v6_source
    assert 'feature_version="v7"' in v7_source
    assert 'feature_version="v8"' in v8_source
    assert 'feature_version="v9"' in v9_source
    assert 'feature_version="v10"' in v10_source
    assert 'feature_version="v11"' in v11_source
    assert 'timing_mode=TIMING_MODE' in v7_source
    assert 'providers=PROVIDERS' in v7_source
    assert 'year_split_folds=YEAR_SPLIT_EXPANDING_FOLDS' in v7_source
    assert 'hyperparameter_space="wide"' in v7_source
    assert 'timing_mode=TIMING_MODE' in v8_source
    assert 'providers=PROVIDERS' in v8_source
    assert 'year_split_folds=YEAR_SPLIT_EXPANDING_FOLDS' in v8_source
    assert 'hyperparameter_space="wide"' in v8_source
    assert 'timing_mode=TIMING_MODE' in v9_source
    assert 'providers=PROVIDERS' in v9_source
    assert 'target_mode="remaining_warmup"' in v9_source
    assert "export_station_model_weights" in v9_source
    assert "station_high_regressor_v9_remaining_warmup" in v9_source
    assert 'year_split_folds=YEAR_SPLIT_EXPANDING_FOLDS' in v9_source
    assert 'hyperparameter_space="wide"' in v9_source
    assert "OPTUNA_TRIALS = 30" in v9_source
    assert "OPTUNA_STARTUP_TRIALS = 15" in v9_source
    assert 'timing_mode=TIMING_MODE' in v10_source
    assert 'providers=PROVIDERS' in v10_source
    assert 'target_mode="remaining_warmup"' in v10_source
    assert 'base_model_methods=("catboost",)' in v10_source
    assert "stack_enabled=False" in v10_source
    assert "station_high_regressor_v10_catboost_huber" in v10_source
    assert "export_station_model_weights" in v10_source
    assert 'year_split_folds=YEAR_SPLIT_EXPANDING_FOLDS' in v10_source
    assert 'hyperparameter_space="wide"' in v10_source
    assert 'timing_mode=TIMING_MODE' in v11_source
    assert 'providers=PROVIDERS' in v11_source
    assert 'target_mode="remaining_warmup"' in v11_source
    assert 'base_model_methods=("xgboost", "lightgbm", "catboost")' in v11_source
    assert "stack_enabled=True" in v11_source
    assert "station_high_regressor_v11_huber_ridge_stack" in v11_source
    assert "export_station_model_weights" in v11_source
    assert 'year_split_folds=YEAR_SPLIT_EXPANDING_FOLDS' in v11_source
    assert 'hyperparameter_space="wide"' in v11_source
    assert "build_station_wide_dataset =" not in v5_source
    assert "build_station_wide_dataset =" not in v6_source
    assert "build_station_wide_dataset =" not in v7_source
    assert "build_station_wide_dataset =" not in v8_source
    assert "build_station_wide_dataset =" not in v9_source
    assert "build_station_wide_dataset =" not in v10_source
    assert "build_station_wide_dataset =" not in v11_source


def test_station_stacking_loads_direct_nbm_cache(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("WEATHER_RESEARCH_INCLUDE_DIRECT_NBM", "1")
    _write_station_stacking_fixture(tmp_path)

    forecasts = load_same_day_provider_forecasts(tmp_path, providers=("gfs", "hrrr", "nbm"))

    nbm = forecasts.loc[forecasts["provider"].eq("nbm")]
    assert not nbm.empty
    assert set(nbm["source_cache_dir"]) == {"direct_nbm_fixture"}


def test_provider_availability_and_raw_baselines(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("WEATHER_RESEARCH_INCLUDE_DIRECT_NBM", "1")
    _write_station_stacking_fixture(tmp_path)

    availability = provider_availability(tmp_path, providers=("gfs", "hrrr", "nbm"))
    assert set(availability["provider"]) == {"gfs", "hrrr", "nbm"}
    assert set(availability["row_count"]) == {8}

    frame = build_station_wide_dataset(tmp_path, station_id="KATL", providers=("gfs", "hrrr", "nbm"))
    config = StationStackingConfig(
        station_id="KATL",
        project_root=tmp_path,
        providers=("gfs", "hrrr", "nbm"),
        min_train_rows=2,
        refit_days=1,
    )
    predictions = raw_baseline_predictions(frame, config)

    assert {"gfs_raw", "hrrr_raw", "nbm_raw", "provider_mean", "provider_median", "best_raw_provider"}.issubset(
        set(predictions["method"])
    )
    assert predictions.loc[predictions["method"].eq("best_raw_provider"), "contract_date"].min() == "2026-01-03"


def test_missing_expected_model_methods_flags_baseline_only_metrics() -> None:
    metrics = pd.DataFrame(
        {
            "method": ["gfs_raw", "hrrr_raw", "provider_mean", "provider_median"],
            "mae_f": [1.0, 1.1, 0.9, 0.9],
        }
    )

    missing = missing_expected_model_methods(metrics)

    assert missing == ["xgboost", "lightgbm", "catboost", "ridge_stack"]


def test_selected_hyperparameters_uses_rmse_not_mae() -> None:
    tuning = pd.DataFrame(
        [
            {"method": "xgboost", "param_key": "trial_low_mae", "status": "ok", "mae_f": 1.0, "rmse_f": 10.0},
            {"method": "xgboost", "param_key": "trial_low_mae", "status": "ok", "mae_f": 1.2, "rmse_f": 9.0},
            {"method": "xgboost", "param_key": "trial_low_rmse", "status": "ok", "mae_f": 4.0, "rmse_f": 2.0},
            {"method": "xgboost", "param_key": "trial_low_rmse", "status": "ok", "mae_f": 4.2, "rmse_f": 2.5},
        ]
    )

    selected = station_stacking._selected_hyperparameters(tuning)

    assert selected.iloc[0]["param_key"] == "trial_low_rmse"
    assert selected.iloc[0]["mean_validation_rmse_f"] == 2.25


def test_year_split_tuning_stack_scoreboard_and_brackets(monkeypatch) -> None:
    class FakeTrial:
        def __init__(self, number: int):
            self.number = number

        def suggest_float(self, name, low, high, log=False):
            return 1.0

        def suggest_categorical(self, name, choices):
            if "models_plus_raw" in choices:
                return "models_plus_raw"
            if True in choices:
                return True
            return choices[0]

    class FakeStudy:
        def optimize(self, objective, n_trials, show_progress_bar=False, catch=()):
            for number in range(n_trials):
                objective(FakeTrial(number))

    class MeanEstimator:
        def fit(self, x, y):
            self.mean_ = float(pd.Series(y).mean())
            return self

        def predict(self, x):
            return [self.mean_] * len(x)

    def fake_fit_predict_base_model(**kwargs):
        train = kwargs["train"]
        valid = kwargs["valid"]
        predicted = [float(pd.Series(train["actual_high_f"]).mean())] * len(valid)
        metadata = {
            "numeric_features": ",".join(kwargs["numeric"]),
            "categorical_features": ",".join(kwargs["categorical"]),
            "best_iteration": 1,
        }
        return predicted, metadata

    monkeypatch.setattr(station_stacking, "_create_optuna_study", lambda *args, **kwargs: FakeStudy())
    monkeypatch.setattr(station_stacking, "_create_stack_optuna_study", lambda *args, **kwargs: FakeStudy())
    monkeypatch.setattr(station_stacking, "_suggest_hyperparameters", lambda *args, **kwargs: {})
    monkeypatch.setattr(station_stacking, "_fit_predict_base_model", fake_fit_predict_base_model)
    rows = []
    for year in range(2021, 2027):
        for month in range(1, 4):
            actual = 70 + (year - 2021) + month
            rows.append(
                {
                    "contract_date": f"{year}-{month:02d}-01",
                    "actual_high_f": actual,
                    "year": year,
                    "month": month,
                    "day_of_week": "Monday",
                    "gfs_high_f": actual - 1,
                    "hrrr_high_f": actual + 1,
                    "all_provider_highs_available": True,
                    "provider_mean_high_f": actual,
                    "provider_median_high_f": actual,
                    "observed_temp_at_as_of_f": actual - 5,
                }
            )
    frame = pd.DataFrame(rows)
    config = StationStackingConfig(station_id="KATL", providers=("gfs", "hrrr"), fast_mode=True, min_meta_train_rows=2)
    categorical, numeric = ["day_of_week"], ["gfs_high_f", "hrrr_high_f", "observed_temp_at_as_of_f"]

    tuning, validation_predictions, selected = tune_year_split_base_models(frame, config, categorical, numeric)
    baseline_validation = station_stacking.year_split_baseline_predictions(frame, config)
    validation_predictions = pd.concat([baseline_validation, validation_predictions], ignore_index=True)
    test_predictions = year_split_test_predictions(frame, config, categorical, numeric, selected)
    stack_predictions = station_stacking.year_split_stack_test_predictions(validation_predictions, test_predictions, config)
    test_predictions = pd.concat([test_predictions, stack_predictions], ignore_index=True)
    scoreboard = station_stacking.year_split_scoreboard(validation_predictions, test_predictions)
    bracket_predictions = station_stacking.year_split_bracket_predictions(test_predictions)
    bracket_metrics = station_stacking.year_split_bracket_metrics(bracket_predictions)

    assert set(tuning["fold"].dropna()) == {"fold_2021_2023_to_2024", "fold_2022_2024_to_2025"}
    assert set(validation_predictions["contract_date"].str[:4]) == {"2024", "2025"}
    assert set(test_predictions["contract_date"].str[:4]) == {"2026"}
    assert {"xgboost", "lightgbm", "catboost"}.issubset(set(selected["method"]))
    assert set(stack_predictions["method"]) == {"ridge_stack"}
    assert set(scoreboard["method"]) == {"xgboost", "lightgbm", "catboost", "ridge_stack", "hrrr_raw", "gfs_raw"}
    assert set(bracket_metrics["method"]) == {"xgboost", "lightgbm", "catboost", "ridge_stack", "hrrr_raw", "gfs_raw"}


def test_v10_catboost_only_pipeline_skips_stack(monkeypatch) -> None:
    class FakeTrial:
        def __init__(self, number: int):
            self.number = number

        def suggest_int(self, name, low, high):
            return low

        def suggest_float(self, name, low, high, log=False):
            return low

        def suggest_categorical(self, name, choices):
            return tuple(choices)[0]

    class FakeStudy:
        def optimize(self, objective, n_trials, show_progress_bar=False, catch=()):
            for number in range(n_trials):
                objective(FakeTrial(number))

    def fake_fit_predict_base_model(**kwargs):
        method = kwargs["method"]
        valid = kwargs["valid"]
        return list(valid["actual_high_f"]), {
            "numeric_features": ",".join(kwargs["numeric"]),
            "categorical_features": ",".join(kwargs["categorical"]),
            "best_iteration": 1,
            "target_mode": kwargs["config"].effective_target_mode,
            "model_target": station_stacking._model_target_column(kwargs["config"]),
            "method": method,
        }

    monkeypatch.setattr(station_stacking, "_create_optuna_study", lambda *args, **kwargs: FakeStudy())
    monkeypatch.setattr(station_stacking, "_suggest_hyperparameters", lambda *args, **kwargs: {"huber_delta": 1.0})
    monkeypatch.setattr(station_stacking, "_fit_predict_base_model", fake_fit_predict_base_model)
    rows = []
    for year in range(2021, 2027):
        for month in range(1, 4):
            actual = 70 + (year - 2021) + month
            rows.append(
                {
                    "contract_date": f"{year}-{month:02d}-01",
                    "actual_high_f": actual,
                    "year": year,
                    "month": month,
                    "day_of_week": "Monday",
                    "gfs_high_f": actual - 1,
                    "hrrr_high_f": actual + 1,
                    "nbm_high_f": actual,
                    "all_provider_highs_available": True,
                    "provider_mean_high_f": actual,
                    "provider_median_high_f": actual,
                    "observed_temp_at_as_of_f": actual - 5,
                    "observed_high_temp_through_as_of_f": actual - 4,
                }
            )
    frame = pd.DataFrame(rows)
    config = StationStackingConfig(
        station_id="KATL",
        providers=("gfs", "hrrr", "nbm"),
        feature_version="v10",
        target_mode="remaining_warmup",
        base_model_methods=("catboost",),
        stack_enabled=False,
        fast_mode=True,
        min_meta_train_rows=2,
    )
    categorical, numeric = ["day_of_week"], [
        "gfs_high_f",
        "hrrr_high_f",
        "nbm_high_f",
        "observed_temp_at_as_of_f",
        "observed_high_temp_through_as_of_f",
    ]

    tuning, validation_predictions, selected = tune_year_split_base_models(frame, config, categorical, numeric)
    baseline_validation = station_stacking.year_split_baseline_predictions(frame, config)
    test_predictions = year_split_test_predictions(frame, config, categorical, numeric, selected)
    stack_predictions, stack_tuning = station_stacking.tune_year_split_stack_model(
        validation_predictions=pd.concat([baseline_validation, validation_predictions], ignore_index=True),
        test_predictions=test_predictions,
        config=config,
    )
    scoreboard = station_stacking.year_split_scoreboard(
        pd.concat([baseline_validation, validation_predictions], ignore_index=True),
        test_predictions,
    )

    assert set(tuning["method"]) == {"catboost"}
    assert set(selected["method"]) == {"catboost"}
    assert set(validation_predictions["method"]) == {"catboost"}
    assert stack_predictions.empty
    assert stack_tuning.empty
    assert "ridge_stack" not in set(scoreboard["method"])
    assert {"xgboost", "lightgbm"}.isdisjoint(set(scoreboard["method"]))


def test_year_split_tuning_accepts_v7_expanding_folds(monkeypatch) -> None:
    class FakeTrial:
        def __init__(self, number: int):
            self.number = number

    class FakeStudy:
        def optimize(self, objective, n_trials, show_progress_bar=False, catch=()):
            for number in range(n_trials):
                objective(FakeTrial(number))

    fit_windows = []

    def fake_fit_predict_base_model(**kwargs):
        train = kwargs["train"]
        valid = kwargs["valid"]
        fit_windows.append(
            {
                "train_start_year": int(train["year"].min()),
                "train_end_year": int(train["year"].max()),
                "validation_year": int(valid["year"].iloc[0]),
            }
        )
        return list(valid["actual_high_f"]), {
            "numeric_features": ",".join(kwargs["numeric"]),
            "categorical_features": ",".join(kwargs["categorical"]),
            "best_iteration": 1,
        }

    monkeypatch.setattr(station_stacking, "_create_optuna_study", lambda *args, **kwargs: FakeStudy())
    monkeypatch.setattr(station_stacking, "_suggest_hyperparameters", lambda *args, **kwargs: {})
    monkeypatch.setattr(station_stacking, "_fit_predict_base_model", fake_fit_predict_base_model)
    rows = []
    for year in range(2021, 2026):
        for month in range(1, 4):
            actual = 70 + (year - 2021) + month
            rows.append(
                {
                    "contract_date": f"{year}-{month:02d}-01",
                    "actual_high_f": actual,
                    "year": year,
                    "month": month,
                    "day_of_week": "Monday",
                    "gfs_high_f": actual - 1,
                    "hrrr_high_f": actual + 1,
                    "all_provider_highs_available": True,
                    "observed_temp_at_as_of_f": actual - 5,
                }
            )
    frame = pd.DataFrame(rows)
    config = StationStackingConfig(
        station_id="KATL",
        providers=("gfs", "hrrr"),
        fast_mode=True,
        year_split_folds=station_stacking.YEAR_SPLIT_EXPANDING_FOLDS,
    )

    tuning, _, _ = tune_year_split_base_models(
        frame,
        config,
        categorical=["day_of_week"],
        numeric=["gfs_high_f", "hrrr_high_f", "observed_temp_at_as_of_f"],
        folds=config.effective_year_split_folds,
    )

    assert set(tuning["fold"].dropna()) == {"fold_2021_2023_to_2024", "fold_2021_2024_to_2025"}
    assert {
        "train_start_year": 2021,
        "train_end_year": 2024,
        "validation_year": 2025,
    } in fit_windows


def test_wide_hyperparameter_space_expands_base_and_stack_ranges() -> None:
    class RecordingTrial:
        number = 0

        def __init__(self):
            self.calls = []

        def suggest_int(self, name, low, high):
            self.calls.append(("int", name, low, high))
            return low

        def suggest_float(self, name, low, high, log=False):
            self.calls.append(("float", name, low, high, log))
            return low

        def suggest_categorical(self, name, choices):
            self.calls.append(("categorical", name, tuple(choices)))
            return tuple(choices)[0]

    config = StationStackingConfig(station_id="KATL", hyperparameter_space="wide")
    xgb_trial = RecordingTrial()
    lgbm_trial = RecordingTrial()
    cat_trial = RecordingTrial()
    stack_trial = RecordingTrial()

    station_stacking._suggest_hyperparameters("xgboost", xgb_trial, config)
    station_stacking._suggest_hyperparameters("lightgbm", lgbm_trial, config)
    station_stacking._suggest_hyperparameters("catboost", cat_trial, config)
    station_stacking._suggest_stack_hyperparameters(stack_trial, config)

    assert ("int", "n_estimators", 50, 3500) in xgb_trial.calls
    assert ("float", "learning_rate", 0.001, 0.25, True) in xgb_trial.calls
    assert ("int", "num_leaves", 4, 512) in lgbm_trial.calls
    assert ("int", "iterations", 50, 3500) in cat_trial.calls
    assert ("float", "alpha", 1e-06, 100000.0, True) in stack_trial.calls


def test_v10_catboost_uses_tuned_huber_loss() -> None:
    class RecordingTrial:
        number = 0

        def __init__(self):
            self.calls = []

        def suggest_int(self, name, low, high):
            self.calls.append(("int", name, low, high))
            return low

        def suggest_float(self, name, low, high, log=False):
            self.calls.append(("float", name, low, high, log))
            return low

        def suggest_categorical(self, name, choices):
            self.calls.append(("categorical", name, tuple(choices)))
            return tuple(choices)[0]

    trial = RecordingTrial()
    config = StationStackingConfig(
        station_id="KATL",
        feature_version="v10",
        base_model_methods=("catboost",),
        stack_enabled=False,
        hyperparameter_space="wide",
    )

    params = station_stacking._suggest_hyperparameters("catboost", trial, config)
    v10_estimator = station_stacking._build_base_model_estimator(
        config,
        "catboost",
        {**params, "huber_delta": 2.0},
    )
    v9_estimator = station_stacking._build_base_model_estimator(
        StationStackingConfig(station_id="KATL", feature_version="v9"),
        "catboost",
        {},
    )

    assert ("categorical", "huber_delta", (0.5, 1.0, 1.5, 2.0, 3.0, 5.0)) in trial.calls
    assert params["huber_delta"] == 0.5
    assert v10_estimator.get_params()["loss_function"] == "Huber:delta=2"
    assert v10_estimator.get_params()["eval_metric"] == "MAE"
    assert v9_estimator.get_params()["loss_function"] == "RMSE"


def test_v11_base_models_use_huber_objectives() -> None:
    class RecordingTrial:
        number = 0

        def __init__(self):
            self.calls = []

        def suggest_int(self, name, low, high):
            self.calls.append(("int", name, low, high))
            return low

        def suggest_float(self, name, low, high, log=False):
            self.calls.append(("float", name, low, high, log))
            return low

        def suggest_categorical(self, name, choices):
            self.calls.append(("categorical", name, tuple(choices)))
            return tuple(choices)[0]

    config = StationStackingConfig(station_id="KATL", feature_version="v11", hyperparameter_space="wide")
    xgb_trial = RecordingTrial()
    lgbm_trial = RecordingTrial()
    cat_trial = RecordingTrial()

    xgb_params = station_stacking._suggest_hyperparameters("xgboost", xgb_trial, config)
    lgbm_params = station_stacking._suggest_hyperparameters("lightgbm", lgbm_trial, config)
    cat_params = station_stacking._suggest_hyperparameters("catboost", cat_trial, config)

    xgb_estimator = station_stacking._build_base_model_estimator(config, "xgboost", xgb_params)
    lgbm_estimator = station_stacking._build_base_model_estimator(config, "lightgbm", lgbm_params)
    cat_estimator = station_stacking._build_base_model_estimator(
        config,
        "catboost",
        {**cat_params, "huber_delta": 3.0},
    )
    v9_xgb_estimator = station_stacking._build_base_model_estimator(
        StationStackingConfig(station_id="KATL", feature_version="v9"),
        "xgboost",
        {},
    )

    assert xgb_params["objective"] == "reg:pseudohubererror"
    assert lgbm_params["objective"] == "huber"
    assert ("categorical", "huber_alpha", (0.75, 0.85, 0.9, 0.95)) in lgbm_trial.calls
    assert ("categorical", "huber_delta", (0.5, 1.0, 1.5, 2.0, 3.0, 5.0)) in cat_trial.calls
    assert xgb_estimator.get_params()["objective"] == "reg:pseudohubererror"
    assert xgb_estimator.get_params()["eval_metric"] == "mae"
    assert lgbm_estimator.get_params()["objective"] == "huber"
    assert lgbm_estimator.get_params()["metric"] == "mae"
    assert lgbm_estimator.get_params()["alpha"] == 0.75
    assert cat_estimator.get_params()["loss_function"] == "Huber:delta=3"
    assert cat_estimator.get_params()["eval_metric"] == "MAE"
    assert v9_xgb_estimator.get_params()["objective"] == "reg:squarederror"


def test_polymarket_temperature_bracket_rounds_half_up_to_two_degree_bins() -> None:
    assert station_stacking.polymarket_temperature_bracket(80) == "80-81"
    assert station_stacking.polymarket_temperature_bracket(81) == "80-81"
    assert station_stacking.polymarket_temperature_bracket(82) == "82-83"
    assert station_stacking.polymarket_temperature_bracket(84.5) == "84-85"
    assert station_stacking.round_temperature_half_up(84.5) == 85


def test_year_split_feature_importance_uses_2026_test(monkeypatch) -> None:
    from sklearn.base import BaseEstimator

    class FirstFeatureEstimator(BaseEstimator):
        def fit(self, x, y):
            return self

        def predict(self, x):
            return pd.to_numeric(x.iloc[:, 0], errors="coerce").fillna(0)

    monkeypatch.setattr(station_stacking, "_build_base_model_pipeline", lambda *args, **kwargs: FirstFeatureEstimator())
    rows = []
    for year in range(2021, 2027):
        for month in range(1, 11):
            driver = float(month * 10)
            rows.append(
                {
                    "contract_date": f"{year}-{month:02d}-01",
                    "actual_high_f": driver,
                    "year": year,
                    "driver_feature": driver,
                    "noise_feature": 1.0,
                }
            )
    frame = pd.DataFrame(rows)
    config = StationStackingConfig(
        station_id="KATL",
        providers=("gfs", "hrrr"),
        fast_mode=True,
        feature_importance_repeats=5,
    )
    selected = pd.DataFrame({"method": ["xgboost"], "param_key": ["trial_0"], "mean_validation_rmse_f": [0.0]})

    importance = year_split_feature_importance(frame, config, [], ["driver_feature", "noise_feature"], selected)

    assert not importance.empty
    assert set(importance["test_year"]) == {2026}
    assert set(importance["train_start_year"]) == {2021}
    assert set(importance["train_end_year"]) == {2025}
    assert importance.iloc[0]["feature"] == "driver_feature"
    assert importance.iloc[0]["importance_mean_mae_f"] > 0


def test_export_station_stacking_v2_model_weights(tmp_path, monkeypatch) -> None:
    from sklearn.dummy import DummyRegressor
    import json
    import joblib

    artifact_dir = tmp_path / "data" / "calibration" / "station_stacking_v2"
    artifact_dir.mkdir(parents=True)
    rows = []
    for year in range(2021, 2027):
        for month in range(1, 3):
            actual = 70 + (year - 2021) + month
            rows.append(
                {
                    "station_id": "KATL",
                    "contract_date": f"{year}-{month:02d}-01",
                    "actual_high_f": actual,
                    "year": year,
                    "month": month,
                    "day_of_week": "Monday",
                    "gfs_high_f": actual - 1,
                    "hrrr_high_f": actual + 1,
                    "provider_mean_high_f": actual,
                    "all_provider_highs_available": True,
                    "observed_temp_at_as_of_f": actual - 4,
                    "v2_recent_heat_anomaly_f": 0.5,
                }
            )
    pd.DataFrame(rows).to_csv(artifact_dir / "KATL_features.csv", index=False)
    pd.DataFrame(
        {
            "method": ["xgboost", "lightgbm", "catboost"],
            "param_key": ["trial_0", "trial_0", "trial_0"],
            "mean_validation_rmse_f": [1.0, 1.1, 1.2],
            "mean_validation_mae_f": [0.8, 0.9, 1.0],
        }
    ).to_csv(artifact_dir / "KATL_year_split_selected_hyperparameters.csv", index=False)

    validation_rows = []
    for date, actual in [("2024-01-01", 74.0), ("2025-01-01", 75.0)]:
        predictions = {
            "xgboost": actual,
            "lightgbm": actual + 0.1,
            "catboost": actual - 0.1,
            "hrrr_raw": actual + 1.0,
            "gfs_raw": actual - 1.0,
        }
        for method, predicted in predictions.items():
            validation_rows.append(
                {
                    "contract_date": date,
                    "method": method,
                    "actual_high_f": actual,
                    "predicted_high_f": predicted,
                }
            )
    pd.DataFrame(validation_rows).to_csv(artifact_dir / "KATL_year_split_validation_predictions.csv", index=False)
    pd.DataFrame(
        {
            "method": ["ridge_stack"],
            "trial_number": [0],
            "param_key": ["stack_trial_0"],
            "feature_set": ["models_plus_raw"],
            "alpha": [1.0],
            "fit_intercept": [True],
            "mae_f": [0.5],
            "rmse_f": [0.6],
            "count": [2],
            "status": ["ok"],
            "error": [""],
        }
    ).to_csv(artifact_dir / "KATL_year_split_stack_tuning.csv", index=False)

    monkeypatch.setattr(
        export_station_stacking_v2_models,
        "_build_base_model_pipeline",
        lambda *args, **kwargs: DummyRegressor(strategy="mean"),
    )

    exported = export_station_stacking_v2_models.export_station_model_weights(
        project_root=tmp_path,
        station_id="KATL",
        artifact_dir=artifact_dir,
    )
    bundle = joblib.load(exported.bundle_path)
    manifest = json.loads(exported.manifest_path.read_text(encoding="utf-8"))

    assert exported.bundle_path.exists()
    assert exported.manifest_path.exists()
    assert set(bundle["base_models"]) == {"xgboost", "lightgbm", "catboost"}
    assert bundle["stack_features"] == [
        "xgboost_predicted_high_f",
        "lightgbm_predicted_high_f",
        "catboost_predicted_high_f",
        "hrrr_raw_predicted_high_f",
        "gfs_raw_predicted_high_f",
    ]
    assert manifest["station_id"] == "KATL"
    assert manifest["training"]["mode"] == "production_refit_all_available_actuals"
    assert manifest["model_contract"]["target_mode"] == "actual_high"
    assert bundle["target_mode"] == "actual_high"
    assert bundle["model_target"] == "actual_high_f"


def test_export_station_stacking_remaining_warmup_model_weights(tmp_path, monkeypatch) -> None:
    from sklearn.dummy import DummyRegressor
    import json
    import joblib

    artifact_dir = tmp_path / "data" / "calibration" / "station_stacking_v9"
    artifact_dir.mkdir(parents=True)
    rows = []
    expected_targets = []
    for year in range(2021, 2027):
        for month in range(1, 3):
            actual = 70 + (year - 2021) + month
            observed_high = actual - month
            expected_targets.append(float(actual - observed_high))
            rows.append(
                {
                    "station_id": "KATL",
                    "contract_date": f"{year}-{month:02d}-01",
                    "actual_high_f": actual,
                    "actual_data_quality_flag": "ok",
                    "actual_raw_observation_count": 24,
                    "observed_fetch_status": "ok",
                    "observed_as_of_age_minutes": 5,
                    "observed_high_temp_through_as_of_f": observed_high,
                    "year": year,
                    "month": month,
                    "day_of_week": "Monday",
                    "gfs_high_f": actual - 1,
                    "hrrr_high_f": actual + 1,
                    "nbm_high_f": actual,
                    "provider_mean_high_f": actual,
                    "all_provider_highs_available": True,
                    "observed_temp_at_as_of_f": observed_high - 1,
                    "v9_climatology_normals_10y_count": 10,
                }
            )
    pd.DataFrame(rows).to_csv(artifact_dir / "KATL_features.csv", index=False)
    pd.DataFrame(
        {
            "method": ["xgboost", "lightgbm", "catboost"],
            "param_key": ["trial_0", "trial_0", "trial_0"],
            "mean_validation_rmse_f": [1.0, 1.1, 1.2],
            "mean_validation_mae_f": [0.8, 0.9, 1.0],
        }
    ).to_csv(artifact_dir / "KATL_year_split_selected_hyperparameters.csv", index=False)

    validation_rows = []
    for date, actual in [("2024-01-01", 74.0), ("2025-01-01", 75.0)]:
        predictions = {
            "xgboost": actual,
            "lightgbm": actual + 0.1,
            "catboost": actual - 0.1,
            "hrrr_raw": actual + 1.0,
            "gfs_raw": actual - 1.0,
        }
        for method, predicted in predictions.items():
            validation_rows.append(
                {
                    "contract_date": date,
                    "method": method,
                    "actual_high_f": actual,
                    "predicted_high_f": predicted,
                }
            )
    pd.DataFrame(validation_rows).to_csv(artifact_dir / "KATL_year_split_validation_predictions.csv", index=False)
    pd.DataFrame(
        {
            "method": ["ridge_stack"],
            "trial_number": [0],
            "param_key": ["stack_trial_0"],
            "feature_set": ["models_plus_raw"],
            "alpha": [1.0],
            "fit_intercept": [True],
            "mae_f": [0.5],
            "rmse_f": [0.6],
            "count": [2],
            "status": ["ok"],
            "error": [""],
        }
    ).to_csv(artifact_dir / "KATL_year_split_stack_tuning.csv", index=False)

    monkeypatch.setattr(
        export_station_stacking_v2_models,
        "_build_base_model_pipeline",
        lambda *args, **kwargs: DummyRegressor(strategy="mean"),
    )

    exported = export_station_stacking_v2_models.export_station_model_weights(
        project_root=tmp_path,
        station_id="KATL",
        artifact_dir=artifact_dir,
        model_version="station_high_regressor_v9_remaining_warmup",
        providers=("gfs", "hrrr", "nbm"),
        feature_version="v9",
        optuna_metric="mae_f",
        target_mode="remaining_warmup",
    )
    bundle = joblib.load(exported.bundle_path)
    manifest = json.loads(exported.manifest_path.read_text(encoding="utf-8"))
    learned_target = float(bundle["base_models"]["xgboost"].constant_.ravel()[0])

    assert learned_target == pytest.approx(sum(expected_targets) / len(expected_targets))
    assert bundle["target_mode"] == "remaining_warmup"
    assert bundle["model_target"] == station_stacking.REMAINING_WARMUP_TARGET
    assert bundle["observed_high_so_far_column"] == station_stacking.OBSERVED_HIGH_SO_FAR_COLUMN
    assert manifest["model_contract"]["target_mode"] == "remaining_warmup"
    assert manifest["training"]["model_target"] == station_stacking.REMAINING_WARMUP_TARGET
    assert manifest["inference"]["base_prediction_transform"].startswith("predicted_high_f=max")


def test_export_catboost_only_model_weights_without_stack_tuning(tmp_path, monkeypatch) -> None:
    from sklearn.dummy import DummyRegressor
    import json
    import joblib

    artifact_dir = tmp_path / "data" / "calibration" / "station_stacking_v10"
    artifact_dir.mkdir(parents=True)
    rows = []
    for year in range(2021, 2027):
        for month in range(1, 3):
            actual = 70 + (year - 2021) + month
            rows.append(
                {
                    "station_id": "KATL",
                    "contract_date": f"{year}-{month:02d}-01",
                    "actual_high_f": actual,
                    "actual_data_quality_flag": "ok",
                    "actual_raw_observation_count": 24,
                    "observed_fetch_status": "ok",
                    "observed_as_of_age_minutes": 5,
                    "observed_high_temp_through_as_of_f": actual - 3,
                    "year": year,
                    "month": month,
                    "day_of_week": "Monday",
                    "gfs_high_f": actual - 1,
                    "hrrr_high_f": actual + 1,
                    "nbm_high_f": actual,
                    "provider_mean_high_f": actual,
                    "all_provider_highs_available": True,
                    "observed_temp_at_as_of_f": actual - 4,
                    "v9_climatology_normals_10y_count": 10,
                }
            )
    pd.DataFrame(rows).to_csv(artifact_dir / "KATL_features.csv", index=False)
    pd.DataFrame(
        {
            "method": ["catboost"],
            "param_key": ["trial_0"],
            "mean_validation_rmse_f": [1.2],
            "mean_validation_mae_f": [0.9],
            "param_huber_delta": [1.5],
        }
    ).to_csv(artifact_dir / "KATL_year_split_selected_hyperparameters.csv", index=False)
    pd.DataFrame(
        [
            {
                "contract_date": "2024-01-01",
                "method": "catboost",
                "actual_high_f": 74.0,
                "predicted_high_f": 74.1,
            }
        ]
    ).to_csv(artifact_dir / "KATL_year_split_validation_predictions.csv", index=False)

    monkeypatch.setattr(
        export_station_stacking_v2_models,
        "_build_base_model_pipeline",
        lambda *args, **kwargs: DummyRegressor(strategy="mean"),
    )

    exported = export_station_stacking_v2_models.export_station_model_weights(
        project_root=tmp_path,
        station_id="KATL",
        artifact_dir=artifact_dir,
        model_version="station_high_regressor_v10_catboost_huber",
        providers=("gfs", "hrrr", "nbm"),
        feature_version="v10",
        optuna_metric="mae_f",
        target_mode="remaining_warmup",
        base_model_methods=("catboost",),
        stack_enabled=False,
    )
    bundle = joblib.load(exported.bundle_path)
    manifest = json.loads(exported.manifest_path.read_text(encoding="utf-8"))

    assert set(bundle["base_models"]) == {"catboost"}
    assert bundle["stack_model"] is None
    assert bundle["stack_features"] == []
    assert bundle["base_model_methods"] == ("catboost",)
    assert bundle["final_model_method"] == "catboost"
    assert manifest["model_contract"]["final_model_method"] == "catboost"
    assert manifest["model_contract"]["stack_enabled"] is False
    assert manifest["base_models"][0]["params"]["huber_delta"] == 1.5
    assert manifest["stack_model"]["features"] == []


def test_export_stack_model_selection_can_use_mae_metric() -> None:
    validation_rows = []
    for date, actual in [("2024-01-01", 74.0), ("2025-01-01", 75.0)]:
        for method, predicted in {
            "xgboost": actual,
            "lightgbm": actual + 0.1,
            "catboost": actual - 0.1,
            "hrrr_raw": actual + 1.0,
            "gfs_raw": actual - 1.0,
        }.items():
            validation_rows.append(
                {
                    "contract_date": date,
                    "method": method,
                    "actual_high_f": actual,
                    "predicted_high_f": predicted,
                }
            )
    validation_predictions = pd.DataFrame(validation_rows)
    stack_tuning = pd.DataFrame(
        [
            {
                "method": "ridge_stack",
                "param_key": "best_rmse",
                "feature_set": "models_only",
                "alpha": 1.0,
                "fit_intercept": True,
                "mae_f": 2.0,
                "rmse_f": 0.5,
                "status": "ok",
            },
            {
                "method": "ridge_stack",
                "param_key": "best_mae",
                "feature_set": "models_only",
                "alpha": 2.0,
                "fit_intercept": True,
                "mae_f": 0.5,
                "rmse_f": 2.0,
                "status": "ok",
            },
        ]
    )

    _, mae_manifest = export_station_stacking_v2_models._fit_stack_model(
        validation_predictions,
        stack_tuning,
        metric_col="mae_f",
    )
    _, rmse_manifest = export_station_stacking_v2_models._fit_stack_model(
        validation_predictions,
        stack_tuning,
        metric_col="rmse_f",
    )

    assert mae_manifest["param_key"] == "best_mae"
    assert mae_manifest["selection_metric"] == "mae_f"
    assert rmse_manifest["param_key"] == "best_rmse"
    assert rmse_manifest["selection_metric"] == "rmse_f"


def test_forecast_at_as_of_columns_are_not_calibration_features() -> None:
    frame = pd.DataFrame(
        {
            "station_id": ["KATL"],
            "provider": ["gfs"],
            "model": ["gfs"],
            "timing_mode": ["same_day_11am"],
            "month": [1],
            "day_of_week": [1],
            "rain_regime": ["dry"],
            "cloud_regime": ["clear"],
            "raw_forecast_high_f": [72.0],
            **{column: [1.0] for column in FORECAST_AT_AS_OF_COLUMNS},
        }
    )

    _, numeric = _feature_columns(frame)

    assert FORECAST_AT_AS_OF_COLUMNS.isdisjoint(CALIBRATION_COLUMNS)
    assert FORECAST_AT_AS_OF_COLUMNS.isdisjoint(WEATHER_NUMERIC_COLUMNS)
    assert FORECAST_AT_AS_OF_COLUMNS.isdisjoint(PROVIDER_NUMERIC_COLUMNS)
    assert FORECAST_AT_AS_OF_COLUMNS.isdisjoint(numeric)
    assert EXPERIMENTAL_FORECAST_COLUMNS.isdisjoint(CALIBRATION_COLUMNS)
    assert EXPERIMENTAL_FORECAST_COLUMNS.isdisjoint(WEATHER_NUMERIC_COLUMNS)
    assert EXPERIMENTAL_FORECAST_COLUMNS.isdisjoint(PROVIDER_NUMERIC_COLUMNS)
    assert EXPERIMENTAL_FORECAST_COLUMNS.isdisjoint(numeric)
