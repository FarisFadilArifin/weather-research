from __future__ import annotations

import pandas as pd
import pytest

from src.current_observations import summarize_current_observations


def test_current_observation_uses_latest_row_before_11am_local() -> None:
    rows = [
        {
            "station_code": "ATL",
            "observed_at": "2024-01-01T15:30:00Z",
            "observation_type": "METAR",
            "source": "iem",
            "temp_f": 55.0,
        },
        {
            "station_code": "ATL",
            "observed_at": "2024-01-01T15:52:00Z",
            "observation_type": "METAR",
            "source": "iem",
            "temp_f": 50.0,
            "dewpoint_f": 40.0,
            "wind_speed_kt": 10,
            "wind_gust_kt": 15,
            "wind_dir_degrees": 250,
            "altimeter_inhg": 30.0,
            "sea_level_pressure_mb": None,
            "visibility_miles": 10.0,
            "sky_cover_1": "SCT",
            "sky_base_1_ft": 2500,
            "sky_cover_2": "BKN",
            "sky_base_2_ft": 5000,
            "weather_codes": "RA",
            "precip_1hr_inches": 0.01,
            "raw_metar": "KATL sample before cutoff",
        },
        {
            "station_code": "ATL",
            "observed_at": "2024-01-01T16:30:00Z",
            "observation_type": "METAR",
            "source": "iem",
            "temp_f": 99.0,
            "dewpoint_f": 70.0,
            "wind_speed_kt": 30,
            "raw_metar": "KATL sample after cutoff",
        },
    ]

    out = summarize_current_observations(
        rows,
        station_id="KATL",
        station_name="Atlanta/Hartsfield-Jackson Intl",
        airport_name="Atlanta/Hartsfield-Jackson Intl",
        timezone="America/New_York",
        contract_dates=["2024-01-01"],
    )

    row = out[0]
    assert row["observed_fetch_status"] == "ok"
    assert row["observed_temp_at_as_of_f"] == 50.0
    assert row["observed_high_temp_through_as_of_f"] == 55.0
    assert round(row["observed_wind_speed_at_as_of"], 2) == 11.51
    assert row["observed_ceiling_at_as_of"] == 5000.0
    assert row["observed_cloud_cover_at_as_of"] == 75.0
    assert row["observed_weather_code_at_as_of"] == "RA"
    assert row["observed_as_of_age_minutes"] == 8.0


def test_current_observation_adds_morning_temperature_trends() -> None:
    rows = [
        {"station_code": "ATL", "observed_at": "2024-01-01T12:55:00Z", "source": "iem", "temp_f": 38.0},
        {"station_code": "ATL", "observed_at": "2024-01-01T13:55:00Z", "source": "iem", "temp_f": 42.0},
        {"station_code": "ATL", "observed_at": "2024-01-01T14:00:00Z", "source": "iem", "temp_f": 44.0},
        {"station_code": "ATL", "observed_at": "2024-01-01T14:55:00Z", "source": "iem", "temp_f": 50.0},
        {"station_code": "ATL", "observed_at": "2024-01-01T15:55:00Z", "source": "iem", "temp_f": 56.0},
        {"station_code": "ATL", "observed_at": "2024-01-01T16:30:00Z", "source": "iem", "temp_f": 90.0},
    ]

    out = summarize_current_observations(
        rows,
        station_id="KATL",
        station_name="Atlanta/Hartsfield-Jackson Intl",
        airport_name="Atlanta/Hartsfield-Jackson Intl",
        timezone="America/New_York",
        contract_dates=["2024-01-01"],
    )

    row = out[0]
    assert row["observed_fetch_status"] == "ok"
    assert row["observed_temp_at_as_of_f"] == 56.0
    assert row["observed_temp_change_last_1h_f"] == 6.0
    assert row["observed_temp_change_last_3h_f"] == 18.0
    assert row["observed_morning_warmup_rate_f_per_hour"] == pytest.approx(12.0 / (115.0 / 60.0))
    assert row["observed_high_so_far_change_since_9am_f"] == 12.0


def test_current_observation_9am_uses_latest_row_inside_window() -> None:
    rows = [
        {"station_code": "ATL", "observed_at": "2024-01-01T13:30:00Z", "source": "iem", "temp_f": 55.0},
        {"station_code": "ATL", "observed_at": "2024-01-01T13:49:00Z", "source": "iem", "temp_f": 40.0},
        {"station_code": "ATL", "observed_at": "2024-01-01T13:55:00Z", "source": "iem", "temp_f": 45.0},
        {"station_code": "ATL", "observed_at": "2024-01-01T14:05:00Z", "source": "iem", "temp_f": 50.0},
        {"station_code": "ATL", "observed_at": "2024-01-01T14:11:00Z", "source": "iem", "temp_f": 99.0},
    ]

    out = summarize_current_observations(
        rows,
        station_id="KATL",
        station_name="Atlanta/Hartsfield-Jackson Intl",
        airport_name="Atlanta/Hartsfield-Jackson Intl",
        timezone="America/New_York",
        contract_dates=["2024-01-01"],
        timing_mode="same_day_9am_live_safe",
        as_of_hour_local=9,
    )

    row = out[0]
    assert row["observed_fetch_status"] == "ok"
    assert row["observed_temp_at_as_of_f"] == 50.0
    assert row["observed_high_temp_through_as_of_f"] == 55.0
    assert row["observed_as_of_time_local"] == "2024-01-01T09:05:00-05:00"
    assert row["observed_as_of_age_minutes"] == -5.0


def test_current_observation_9am_writes_unavailable_when_window_is_empty() -> None:
    out = summarize_current_observations(
        [
            {"station_code": "ATL", "observed_at": "2024-01-01T13:49:00Z", "source": "iem", "temp_f": 55.0},
            {"station_code": "ATL", "observed_at": "2024-01-01T14:11:00Z", "source": "iem", "temp_f": 60.0},
        ],
        station_id="KATL",
        station_name="Atlanta/Hartsfield-Jackson Intl",
        airport_name="Atlanta/Hartsfield-Jackson Intl",
        timezone="America/New_York",
        contract_dates=["2024-01-01"],
        timing_mode="same_day_9am_live_safe",
        as_of_hour_local=9,
    )

    assert out[0]["observed_fetch_status"] == "unavailable"
    assert pd.isna(out[0]["observed_temp_at_as_of_f"])
    assert pd.isna(out[0]["observed_high_temp_through_as_of_f"])


def test_current_observation_1pm_uses_1250_to_1310_window_and_11am_deltas() -> None:
    rows = [
        {"station_code": "DAL", "observed_at": "2026-06-15T15:55:00Z", "source": "iem", "temp_f": 80.0},
        {"station_code": "DAL", "observed_at": "2026-06-15T16:05:00Z", "source": "iem", "temp_f": 82.0},
        {"station_code": "DAL", "observed_at": "2026-06-15T17:49:00Z", "source": "iem", "temp_f": 85.0},
        {"station_code": "DAL", "observed_at": "2026-06-15T17:55:00Z", "source": "iem", "temp_f": 88.0},
        {"station_code": "DAL", "observed_at": "2026-06-15T18:08:00Z", "source": "iem", "temp_f": 90.0},
        {"station_code": "DAL", "observed_at": "2026-06-15T18:11:00Z", "source": "iem", "temp_f": 99.0},
    ]
    out = summarize_current_observations(
        rows,
        station_id="KDAL",
        station_name="Dallas Love Field",
        airport_name="Dallas Love Field",
        timezone="America/Chicago",
        contract_dates=["2026-06-15"],
        timing_mode="same_day_1pm_live_safe",
        as_of_hour_local=13,
    )
    row = out[0]
    assert row["observed_fetch_status"] == "ok"
    assert row["observed_temp_at_as_of_f"] == 90.0
    assert row["observed_high_temp_through_as_of_f"] == 90.0
    assert row["observed_as_of_time_local"] == "2026-06-15T13:08:00-05:00"
    assert row["observed_as_of_age_minutes"] == -8.0
    assert row["observed_temp_change_since_11am_f"] == 10.0
    assert row["observed_high_so_far_change_since_11am_f"] == 10.0


def test_current_observation_trends_are_missing_without_history() -> None:
    out = summarize_current_observations(
        [
            {
                "station_code": "ATL",
                "observed_at": "2024-01-01T15:55:00Z",
                "source": "iem",
                "temp_f": 56.0,
            }
        ],
        station_id="KATL",
        station_name="Atlanta/Hartsfield-Jackson Intl",
        airport_name="Atlanta/Hartsfield-Jackson Intl",
        timezone="America/New_York",
        contract_dates=["2024-01-01"],
    )

    row = out[0]
    assert pd.isna(row["observed_temp_change_last_1h_f"])
    assert pd.isna(row["observed_temp_change_last_3h_f"])
    assert pd.isna(row["observed_morning_warmup_rate_f_per_hour"])
    assert pd.isna(row["observed_high_so_far_change_since_9am_f"])


def test_current_observation_writes_unavailable_when_no_prior_obs() -> None:
    out = summarize_current_observations(
        [
            {
                "station_code": "ATL",
                "observed_at": "2024-01-01T18:00:00Z",
                "source": "iem",
                "temp_f": 70.0,
            }
        ],
        station_id="KATL",
        station_name="Atlanta/Hartsfield-Jackson Intl",
        airport_name="Atlanta/Hartsfield-Jackson Intl",
        timezone="America/New_York",
        contract_dates=["2024-01-01"],
    )

    assert out[0]["observed_fetch_status"] == "unavailable"
    assert pd.isna(out[0]["observed_temp_at_as_of_f"])
    assert pd.isna(out[0]["observed_high_temp_through_as_of_f"])
