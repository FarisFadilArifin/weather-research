from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from src.backfill_nbm_rap_features import FeatureRequest
from src.backfill_peak_timing_features import (
    DEFAULT_LOCAL_HOURS,
    HRRR_FIELDS,
    SCHEMA_VERSION,
    _append_rows_with_lock_retry,
    _completed_keys,
    _features_relative_to_nbm_peak,
    _normalize_stations,
    _seed_nbm_rows,
    summarize_hrrr_peak_timing,
)


def _request() -> FeatureRequest:
    cycle = datetime(2026, 7, 1, 14, tzinfo=UTC)
    return FeatureRequest(
        station_id="KATL",
        contract_date="2026-07-01",
        timezone="America/New_York",
        lat=33.62972,
        lon=-84.44223,
        source="hrrr",
        cycle=cycle,
        fxx_hours=tuple(range(1, 9)),
        forecast_as_of=cycle + timedelta(hours=1),
        forecast_window_start=cycle + timedelta(hours=1),
        forecast_window_end=cycle + timedelta(hours=9),
        cycle_selection_policy="test_live_safe",
    )


def _hourly_values() -> dict[int, dict[str, float]]:
    temperatures_f = [86.0, 88.0, 90.0, 92.0, 94.0, 94.0, 93.0, 91.0]
    precipitation = [0.0, 0.0, 0.0, 0.2, 0.4, 0.0, 0.0, 0.0]
    values: dict[int, dict[str, float]] = {}
    for fxx, (temp_f, precip) in enumerate(zip(temperatures_f, precipitation, strict=True), start=1):
        hour = 10 + fxx
        values[fxx] = {
            "temp_k_2m": (temp_f - 32.0) * 5.0 / 9.0 + 273.15,
            "shortwave_radiation_w_m2": 100.0 * fxx,
            "precip_mm_1h": precip,
            "cloud_cover_pct": float(hour),
            "boundary_layer_cloud_cover_pct": float(hour + 1),
            "low_cloud_cover_pct": float(hour + 2),
            "mid_cloud_cover_pct": float(hour + 3),
            "high_cloud_cover_pct": float(hour + 4),
        }
    return values


def test_peak_timing_summary_preserves_hourly_values_and_uses_earliest_tied_peak() -> None:
    summary = summarize_hrrr_peak_timing(_request(), _hourly_values())

    assert summary["hrrr_fetch_status"] == "ok"
    assert summary["hrrr_profile_complete"] == 1
    assert summary["hrrr_required_value_count"] == len(DEFAULT_LOCAL_HOURS) * len(HRRR_FIELDS)
    assert summary["hrrr_t11l_f"] == pytest.approx(86.0)
    assert summary["hrrr_dswrf_18l_w_m2"] == pytest.approx(800.0)
    assert summary["hrrr_precip_14l_mm"] == pytest.approx(0.2)
    assert summary["hrrr_hour_of_max_local"] == 15
    assert summary["hrrr_peak_at_window_end"] == 0


def test_peak_relative_features_are_inclusive_and_precip_timing_is_signed() -> None:
    summary = summarize_hrrr_peak_timing(_request(), _hourly_values())

    assert summary["hrrr_solar_energy_11_to_hrrr_peak_wh_m2"] == pytest.approx(1500.0)
    assert summary["hrrr_precip_total_11_to_hrrr_peak_mm"] == pytest.approx(0.6)
    assert summary["hrrr_precip_wet_hours_11_to_hrrr_peak"] == 2
    assert summary["hrrr_tcc_11_to_hrrr_peak_mean_pct"] == pytest.approx(13.0)
    assert summary["hrrr_precip_onset_hour_local"] == 14
    assert summary["hrrr_precip_onset_minus_hrrr_peak_hours"] == -1


def test_nbm_peak_relative_features_use_the_saved_hourly_profile() -> None:
    summary = summarize_hrrr_peak_timing(_request(), _hourly_values())
    features = _features_relative_to_nbm_peak(
        {"nbm_hour_of_max_local": 16},
        summary,
        DEFAULT_LOCAL_HOURS,
    )

    assert features["hrrr_solar_energy_11_to_nbm_peak_wh_m2"] == pytest.approx(2100.0)
    assert features["hrrr_precip_total_11_to_nbm_peak_mm"] == pytest.approx(0.6)
    assert features["hrrr_precip_onset_minus_nbm_peak_hours"] == -2


def test_missing_hour_is_preserved_and_suppresses_incomplete_aggregates() -> None:
    values = _hourly_values()
    del values[3]["cloud_cover_pct"]
    summary = summarize_hrrr_peak_timing(_request(), values)

    assert summary["hrrr_fetch_status"] == "partial"
    assert summary["hrrr_profile_complete"] == 0
    assert pd.isna(summary["hrrr_tcc_11_to_hrrr_peak_mean_pct"])
    assert summary["hrrr_lcc_11_to_hrrr_peak_mean_pct"] == pytest.approx(15.0)


def test_resume_requires_current_schema_and_complete_sources() -> None:
    frame = pd.DataFrame(
        [
            {
                "station_id": "KATL",
                "contract_date": "2026-07-01",
                "schema_version": SCHEMA_VERSION,
                "nbm_core_fetch_status": "ok",
                "hrrr_fetch_status": "ok",
            },
            {
                "station_id": "KDAL",
                "contract_date": "2026-07-01",
                "schema_version": SCHEMA_VERSION,
                "nbm_core_fetch_status": "ok",
                "hrrr_fetch_status": "partial",
            },
        ]
    )

    assert _completed_keys(frame) == {("KATL", "2026-07-01")}


def test_station_scope_is_katl_and_kdal_only() -> None:
    assert _normalize_stations(["katl", "KDAL", "katl"]) == ["KATL", "KDAL"]
    with pytest.raises(ValueError, match="limited"):
        _normalize_stations(["KHOU"])


def test_existing_nbm_curve_can_seed_new_peak_timing_row() -> None:
    seed = pd.DataFrame(
        [
            {
                "station_id": "KATL",
                "contract_date": "2026-07-01",
                "nbm_core_fetch_status": "ok",
                "nbm_hour_of_max_local": 16,
                "nbm_t11l_f": 88.25,
                "rap_dswrf_12_17_sum": 9999.0,
            }
        ]
    ).set_index(["station_id", "contract_date"])
    rows = {("KATL", "2026-07-01"): {"station_id": "KATL", "contract_date": "2026-07-01"}}

    _seed_nbm_rows(rows, seed)

    assert rows[("KATL", "2026-07-01")]["nbm_core_fetch_status"] == "ok"
    assert rows[("KATL", "2026-07-01")]["nbm_hour_of_max_local"] == 16
    assert "rap_dswrf_12_17_sum" not in rows[("KATL", "2026-07-01")]


def test_csv_append_retries_transient_windows_lock(tmp_path, monkeypatch) -> None:
    attempts = 0

    def append(path, existing, rows):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("temporarily locked")
        return pd.DataFrame(rows)

    monkeypatch.setattr("src.backfill_peak_timing_features._append_rows", append)
    result = _append_rows_with_lock_retry(
        tmp_path / "features.csv",
        pd.DataFrame(),
        [{"station_id": "KATL"}],
        retry_sleep_seconds=0,
    )

    assert attempts == 3
    assert result.loc[0, "station_id"] == "KATL"
