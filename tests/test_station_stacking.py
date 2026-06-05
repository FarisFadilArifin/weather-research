from __future__ import annotations

from pathlib import Path

import pandas as pd

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
    provider_availability,
    raw_baseline_predictions,
)


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

    frame = build_station_wide_dataset(tmp_path, station_id="KATL")

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
    assert round(frame.loc[0, "observed_wind_dir_sin"], 6) == 0
    assert round(frame.loc[0, "observed_wind_dir_cos"], 6) == -1
    assert frame.loc[0, "observed_is_raining_at_as_of"]
    assert frame.loc[0, "observed_is_fog_or_mist_at_as_of"]
    assert set(OBSERVED_CATEGORICAL_FEATURES).issubset(categorical)
    assert "observed_temp_at_as_of_f" in numeric
    assert "observed_dewpoint_depression_f" in numeric
    assert "observed_raw_metar" not in numeric


def test_station_stacking_loads_direct_nbm_cache(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("WEATHER_RESEARCH_INCLUDE_DIRECT_NBM", "1")
    _write_station_stacking_fixture(tmp_path)

    forecasts = load_same_day_provider_forecasts(tmp_path)

    nbm = forecasts.loc[forecasts["provider"].eq("nbm")]
    assert not nbm.empty
    assert set(nbm["source_cache_dir"]) == {"direct_nbm_fixture"}


def test_provider_availability_and_raw_baselines(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("WEATHER_RESEARCH_INCLUDE_DIRECT_NBM", "1")
    _write_station_stacking_fixture(tmp_path)

    availability = provider_availability(tmp_path)
    assert set(availability["provider"]) == {"gfs", "hrrr", "nbm"}
    assert set(availability["row_count"]) == {8}

    frame = build_station_wide_dataset(tmp_path, station_id="KATL")
    config = StationStackingConfig(station_id="KATL", project_root=tmp_path, min_train_rows=2, refit_days=1)
    predictions = raw_baseline_predictions(frame, config)

    assert {"gfs_raw", "hrrr_raw", "nbm_raw", "provider_mean", "provider_median", "best_raw_provider"}.issubset(
        set(predictions["method"])
    )
    assert predictions.loc[predictions["method"].eq("best_raw_provider"), "contract_date"].min() == "2026-01-03"


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
