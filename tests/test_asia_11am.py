from __future__ import annotations

from datetime import UTC, date, datetime
from types import SimpleNamespace

import pandas as pd
import src.asia_11am as asia_11am

from src.asia_11am import (
    CITY_PROFILES,
    GEFS_MEMBERS,
    _gfs_rows,
    _jma_output_column,
    audit_pipeline,
    backfill_gefs_day,
    forecast_timing,
    gfs_day_workers,
    gefs_file_url,
    normalize_jma_payload,
    resolve_date_bounds,
    run_gfs_backfill,
)


def test_asia_profiles_and_local_cutoff() -> None:
    assert CITY_PROFILES["seoul"].station_id == "RKSI"
    assert CITY_PROFILES["tokyo"].station_id == "RJTT"
    timing = forecast_timing("2022-07-03")
    assert timing["issue_utc"] == datetime(2022, 7, 2, 18, tzinfo=UTC)
    assert timing["as_of_utc"] == datetime(2022, 7, 3, 2, tzinfo=UTC)
    assert timing["gfs_forecast_hours"] == tuple(range(8, 21))
    assert gfs_day_workers(12) == 3
    assert gfs_day_workers(16) == 4
    assert gfs_day_workers(2) == 1


def test_gefs_url_uses_expected_member_archive_layout() -> None:
    url = gefs_file_url(datetime(2022, 7, 2, 18, tzinfo=UTC), "p07", 12)
    assert "gefs.20220702/18/atmos/pgrb2sp25/gep07" in url
    assert url.endswith(".f012")


def test_gefs_fields_from_same_file_share_one_index(monkeypatch, tmp_path) -> None:
    index_requests: list[str] = []
    field_calls: list[tuple[str, str]] = []

    def fake_get(url, *, timeout, headers=None):
        index_requests.append(url)
        return SimpleNamespace(text="shared-index")

    def fake_field(
        data_root,
        issue_time,
        member_id,
        fxx,
        field,
        *,
        force,
        idx_text=None,
    ):
        field_calls.append((field, idx_text))
        return tmp_path / f"{field}.grib2"

    monkeypatch.setattr(asia_11am, "_get_with_retries", fake_get)
    monkeypatch.setattr(asia_11am, "_download_gefs_field", fake_field)

    result = asia_11am._download_gefs_file_fields(
        tmp_path,
        datetime(2022, 7, 2, 18, tzinfo=UTC),
        "p01",
        12,
        ("temp_2m_c", "tmax_3h_c"),
        force=False,
    )

    assert len(index_requests) == 1
    assert field_calls == [
        ("temp_2m_c", "shared-index"),
        ("tmax_3h_c", "shared-index"),
    ]
    assert set(result) == {"temp_2m_c", "tmax_3h_c"}


def test_jma_previous_day1_normalization_keeps_only_11_to_23_local() -> None:
    profile = CITY_PROFILES["tokyo"]
    hours = [
        f"2022-07-03T{hour:02d}:00"
        for hour in range(24)
    ]
    payload = {
        "hourly": {
            "time": hours,
            "temperature_2m_previous_day1": list(range(24)),
            "dew_point_2m_previous_day1": [10] * 24,
            "relative_humidity_2m_previous_day1": [50] * 24,
            "precipitation_previous_day1": [0] * 24,
            "cloud_cover_previous_day1": [20] * 24,
            "wind_speed_10m_previous_day1": [5] * 24,
            "wind_direction_10m_previous_day1": [180] * 24,
            "wind_gusts_10m_previous_day1": [8] * 24,
        }
    }
    frame = normalize_jma_payload(
        payload,
        profile,
        [date(2022, 7, 3)],
        historical=True,
        fetched_at_utc=datetime(2022, 7, 2, 18, tzinfo=UTC),
        source_url="https://example.test",
        source_checksum="abc",
    )
    assert len(frame) == 13
    assert frame["lineage"].eq("jma_msm_previous_day1").all()
    assert frame["valid_time_local"].str[11:13].astype(int).between(11, 23).all()
    assert _jma_output_column("temperature_2m") == "temp_2m_c"


def test_gfs_rows_use_celsius_and_expected_issue() -> None:
    profile = CITY_PROFILES["seoul"]
    values = {
        profile.city_id: {
            8: {"temp_k_2m": 300.15, "dewpoint_k_2m": 295.15},
            20: {"temp_k_2m": 301.15},
        }
    }
    frame = _gfs_rows(date(2022, 7, 3), [profile], values)[profile.city_id]
    assert frame.loc[frame["forecast_hour"].eq(8), "temp_c_2m"].iloc[0] == 27.0
    assert frame["issued_at_utc"].eq("2022-07-02T18:00:00Z").all()
    assert len(frame) == 13


def test_resolve_date_bounds_rejects_pre_jma_history() -> None:
    try:
        resolve_date_bounds("2022-07-02", "2022-07-03")
    except ValueError as exc:
        assert "starts on" in str(exc)
    else:
        raise AssertionError("Expected pre-history date to be rejected")


def test_gfs_backfill_compacts_each_month_before_starting_next(
    monkeypatch,
    tmp_path,
) -> None:
    events: list[tuple[str, str]] = []

    def fake_day(data_root, profiles, contract_date, *, force=False):
        events.append(("day", contract_date.isoformat()))
        return [{"status": "complete"}]

    def fake_compact(
        data_root,
        profiles,
        provider,
        start_date,
        end_date,
        *,
        expected_rows_per_day,
    ):
        events.append(("compact", f"{start_date}/{end_date}"))
        return [{"status": "complete", "month": start_date.strftime("%Y-%m")}]

    monkeypatch.setattr("src.asia_11am.backfill_gfs_day", fake_day)
    monkeypatch.setattr("src.asia_11am._compact_forecast_months", fake_compact)

    result = run_gfs_backfill(
        tmp_path,
        [CITY_PROFILES["tokyo"]],
        date(2022, 7, 30),
        date(2022, 8, 2),
        workers=1,
    )

    assert events == [
        ("day", "2022-07-30"),
        ("day", "2022-07-31"),
        ("compact", "2022-07-30/2022-07-31"),
        ("day", "2022-08-01"),
        ("day", "2022-08-02"),
        ("compact", "2022-08-01/2022-08-02"),
    ]
    assert result["status"] == "complete"
