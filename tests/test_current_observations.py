from __future__ import annotations

import pandas as pd

from src.current_observations import summarize_current_observations


def test_current_observation_uses_latest_row_before_11am_local() -> None:
    rows = [
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
    assert round(row["observed_wind_speed_at_as_of"], 2) == 11.51
    assert row["observed_ceiling_at_as_of"] == 5000.0
    assert row["observed_cloud_cover_at_as_of"] == 75.0
    assert row["observed_weather_code_at_as_of"] == "RA"
    assert row["observed_as_of_age_minutes"] == 8.0


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
