from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import UTC, datetime

from src.calibration.station_stacking import YearSplitFold
from src.calibration.v11_settlement_enrichment import (
    add_cross_provider_features,
    coverage_inventory,
    expanding_fold_coverage_inventory,
    normalize_hourly_forecast,
    parity_report,
    summarize_hourly_forecast,
    summarize_observation_day,
)
from src.direct_nwp_fetch import (
    _available_direct_nwp_fxx_hours,
    _interpolate_legacy_gfs_values,
    direct_nwp_file_url,
)


def test_forecast_summary_has_no_gust_and_keeps_uv_vector() -> None:
    hourly = normalize_hourly_forecast(
        {
            1: {"temp_k_2m": 300, "dewpoint_k_2m": 290, "relative_humidity_pct_2m": 50, "precip_mm_1h": 0, "cloud_cover_pct": 25, "wind_u_ms_10m": 3, "wind_v_ms_10m": 4},
            2: {"temp_k_2m": 302, "dewpoint_k_2m": 292, "relative_humidity_pct_2m": 55, "precip_mm_1h": 2, "cloud_cover_pct": 75, "wind_u_ms_10m": 0, "wind_v_ms_10m": 5},
        },
        provider="gfs",
        station_id="KATL",
        contract_date="2025-07-01",
        issue_utc="2025-07-01T12:00:00Z",
    )
    summary = summarize_hourly_forecast(hourly)
    assert summary["precip_total_mm"] == 2
    assert summary["precip_wet_hour_count"] == 1
    assert summary["cloud_at_11am_pct"] == 25
    assert np.isclose(summary["wind_speed_at_11am_mph"], 5 * 2.2369362921)
    assert not any("gust" in key for key in summary)


def test_observation_summary_uses_strict_window_and_never_emits_precip() -> None:
    times = pd.to_datetime(["2025-07-01T13:00:00Z", "2025-07-01T14:00:00Z", "2025-07-01T14:50:00Z"], utc=True)
    observations = pd.DataFrame(
        {
            "observed_at": times,
            "temp_f": [70, 75, 80],
            "dewpoint_f": [60, 61, 62],
            "wind_speed_kt": [0, 5, 10],
            "wind_dir_degrees": [180, 190, 200],
            "visibility_miles": [10, 10, 9],
            "altimeter_inhg": [30.0, 30.0, 29.99],
            "raw_metar": ["CLR", "SCT", "BKN050"],
            "precip_1hr_inches": [0, 0, 1.2],
        }
    )
    summary = summarize_observation_day(observations, contract_date="2025-07-01", timezone="America/New_York")
    assert summary["observed_temp_at_as_of_f"] == 80
    assert summary["observed_high_temp_through_as_of_f"] == 80
    assert summary["observed_ceiling_present"]
    assert not any("precip" in key for key in summary)


def test_coverage_gate_fails_one_sparse_station_year() -> None:
    frame = pd.DataFrame(
        {
            "station_id": ["KATL"] * 10 + ["KDAL"] * 10,
            "contract_date": ["2021-01-01"] * 20,
            "year": [2021] * 20,
            "feature": list(range(10)) + [1] * 8 + [np.nan] * 2,
        }
    )
    inventory = coverage_inventory(frame, ["feature"], years=[2021], threshold=0.90)
    assert not bool(inventory.iloc[0]["admitted"])
    assert inventory.iloc[0]["minimum_station_year_coverage"] == 0.8


def test_expanding_gate_does_not_use_validation_year() -> None:
    frame = pd.DataFrame(
        {
            "station_id": ["KATL", "KDAL", "KATL", "KDAL"],
            "contract_date": ["2021-01-01", "2021-01-01", "2022-01-01", "2022-01-01"],
            "year": [2021, 2021, 2022, 2022],
            "feature": [1.0, 1.0, np.nan, np.nan],
        }
    )
    inventory = expanding_fold_coverage_inventory(
        frame,
        ["feature"],
        folds=[YearSplitFold("2021_to_2022", 2021, 2021, 2022)],
    )
    assert bool(inventory.iloc[0]["admitted_all_folds"])


def test_cross_provider_requires_all_three_admitted_parents() -> None:
    frame = pd.DataFrame({"gfs_cloud_at_11am_pct": [10], "hrrr_cloud_at_11am_pct": [20], "nbm_cloud_at_11am_pct": [30]})
    partial = add_cross_provider_features(frame, admitted=["gfs_cloud_at_11am_pct", "hrrr_cloud_at_11am_pct"])
    assert "provider_mean_cloud_at_11am_pct" not in partial
    complete = add_cross_provider_features(frame)
    assert complete.loc[0, "provider_mean_cloud_at_11am_pct"] == 20


def test_parity_gate_detects_source_missingness_mismatch() -> None:
    iem = pd.DataFrame({"station_id": ["KATL", "KDAL"], "contract_date": ["2026-07-01", "2026-07-01"], "temp": [80, 90]})
    awc = pd.DataFrame({"station_id": ["KATL", "KDAL"], "contract_date": ["2026-07-01", "2026-07-01"], "temp": [80, np.nan]})
    report = parity_report(iem, awc, ["temp"], tolerance={"temp": 0.2})
    assert not bool(report.iloc[0]["parity_pass"])


def test_legacy_gfs_uses_old_layout_and_three_hour_source_grid() -> None:
    issue = datetime(2021, 1, 1, 12, tzinfo=UTC)
    assert "/atmos/" not in direct_nwp_file_url("gfs", issue, 6)
    assert _available_direct_nwp_fxx_hours("gfs", issue, [4, 5, 6, 7, 8, 9]) == [3, 6, 9]
    modern = datetime(2021, 3, 23, 6, tzinfo=UTC)
    assert "/atmos/" in direct_nwp_file_url("gfs", modern, 6)


def test_legacy_gfs_interpolates_continuous_fields_and_recomputes_wind() -> None:
    values = {
        "KATL": {
            3: {"temp_k_2m": 300.0, "precip_mm_1h": 0.0, "wind_u_ms_10m": 0.0, "wind_v_ms_10m": 3.0},
            6: {"temp_k_2m": 306.0, "precip_mm_1h": 3.0, "wind_u_ms_10m": 3.0, "wind_v_ms_10m": 0.0},
        }
    }
    interpolated = _interpolate_legacy_gfs_values(values, [4, 5, 6])
    assert interpolated["KATL"][4]["temp_k_2m"] == 302.0
    assert interpolated["KATL"][5]["precip_mm_1h"] == 1.0
    assert interpolated["KATL"][4]["_precip_is_incremental"] == 1.0
    assert np.isclose(interpolated["KATL"][4]["wind_speed_ms_10m"], np.hypot(1.0, 2.0))
