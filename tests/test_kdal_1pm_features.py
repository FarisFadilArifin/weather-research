from __future__ import annotations

import pandas as pd

from src.calibration.station_stacking import (
    StationStackingConfig,
    V11_SETTLEMENT_FIX_TEMP_FEATURE_COLUMNS,
    V20_KDAL_1PM_FORECAST_WEATHER_FEATURE_COLUMNS,
    V20_KDAL_1PM_TEMP_FEATURE_COLUMNS,
    add_v20_kdal_1pm_temp_feature_engineering,
    feature_columns,
)


def test_kdal_1pm_temperature_alignment_features_and_contract() -> None:
    frame = pd.DataFrame(
        {
            "station_id": ["KDAL"],
            "day_of_week": [1],
            "observed_weather_code_at_as_of": [""],
            "observed_precip_intensity": ["dry"],
            "observed_temp_at_as_of_f": [90.0],
            "observed_high_temp_through_as_of_f": [91.0],
            "observed_temp_change_since_11am_f": [8.0],
            "observed_high_so_far_change_since_11am_f": [7.0],
            "gfs_forecast_temp_at_as_of_f": [92.0],
            "hrrr_forecast_temp_at_as_of_f": [90.0],
            "nbm_forecast_temp_at_as_of_f": [91.0],
            "provider_mean_high_f": [96.0],
            "gfs_high_f": [97.0],
            "actual_high_lag_1d": [95.0],
            "remaining_warmup_after_1pm_f": [4.0],
            "v11sf_forecast_temp_11am_mean_f": [999.0],
            "v4_any_forecast_precip": [False],
            "v4_observed_precip_any": [True],
            "v8_forecast_dewpoint_mean_f": [70.0],
        }
    )
    engineered = add_v20_kdal_1pm_temp_feature_engineering(frame, providers=("gfs", "hrrr", "nbm"))
    assert engineered.loc[0, "v13sf_forecast_temp_1pm_mean_f"] == 91.0
    assert engineered.loc[0, "v13sf_forecast_temp_1pm_minus_observed_f"] == 1.0
    assert engineered.loc[0, "v13sf_forecast_temp_1pm_spread_f"] == 2.0
    assert engineered.loc[0, "v13sf_forecast_warmup_after_1pm_f"] == 5.0

    config = StationStackingConfig(
        station_id="KDAL",
        timing_mode="same_day_1pm_live_safe",
        providers=("gfs", "hrrr", "nbm"),
        feature_version="v20_kdal_1pm_no_peak",
        training_profile="v20_aligned",
        target_mode="remaining_warmup",
        target_source="wunderground_only",
    )
    categorical, numeric = feature_columns(engineered, config)
    selected = set(categorical) | set(numeric)
    assert set(V20_KDAL_1PM_TEMP_FEATURE_COLUMNS).issubset(selected)
    assert "observed_temp_change_since_11am_f" in selected
    assert "observed_high_so_far_change_since_11am_f" in selected
    assert selected.isdisjoint(V11_SETTLEMENT_FIX_TEMP_FEATURE_COLUMNS)
    assert selected.isdisjoint(V20_KDAL_1PM_FORECAST_WEATHER_FEATURE_COLUMNS)
    assert "v4_observed_precip_any" in selected
    assert "gfs_high_f" in selected
    assert "actual_high_lag_1d" in selected
    assert "remaining_warmup_after_1pm_f" not in selected
