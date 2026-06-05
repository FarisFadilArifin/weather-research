from datetime import UTC, datetime

import pandas as pd

from src.hrrr_fetch import _choose_hrrr_issue_time, _fill_hrrr_snapshot_requests


def test_choose_hrrr_uses_synoptic_cycle_for_longer_window():
    issue, fxx = _choose_hrrr_issue_time(
        datetime(2026, 5, 16, 15, tzinfo=UTC),
        "2026-05-17",
        "America/Chicago",
        max_fxx=48,
        search_hours=18,
        min_coverage_hours=18,
    )
    assert issue == datetime(2026, 5, 16, 12, tzinfo=UTC)
    assert max(fxx) <= 48
    assert len(fxx) >= 18


def test_choose_hrrr_allows_near_full_24h_window():
    issue, fxx = _choose_hrrr_issue_time(
        datetime(2026, 5, 16, 3, tzinfo=UTC),
        "2026-05-17",
        "America/Chicago",
        max_fxx=48,
        search_hours=18,
        min_coverage_hours=18,
    )
    assert issue == datetime(2026, 5, 16, 0, tzinfo=UTC)
    assert max(fxx) == 48
    assert len(fxx) == 20


def test_fill_hrrr_requests_batches_same_run_for_multiple_stations(monkeypatch, tmp_path):
    calls = []

    def fake_extract(stations, settings, raw_dir, issue_utc, fxx_hours, force_refresh):
        calls.append((set(stations), tuple(fxx_hours)))
        return {
            "KDAL": {1: 80.0, 2: 84.0},
            "KATL": {1: 70.0, 2: 72.0},
        }

    monkeypatch.setattr("src.hrrr_fetch._extract_hrrr_run_points", fake_extract)
    issue = datetime(2026, 5, 16, 12, tzinfo=UTC)
    requests = [
        {
            "station": pd.Series(
                {
                    "station_code": "KDAL",
                    "station_name": "Dallas/Love Fld",
                    "airport_name": "Dallas/Love Fld",
                    "lat": 32.8,
                    "lon": -96.8,
                }
            ),
            "station_code": "KDAL",
            "target_date": "2026-05-17",
            "horizon": 12,
            "issue_utc": issue,
            "issue_local": issue,
            "fxx_hours": [1, 2],
        },
        {
            "station": pd.Series(
                {
                    "station_code": "KATL",
                    "station_name": "Atlanta/Hartsfield-Jackson Intl",
                    "airport_name": "Atlanta/Hartsfield-Jackson Intl",
                    "lat": 33.6,
                    "lon": -84.4,
                }
            ),
            "station_code": "KATL",
            "target_date": "2026-05-17",
            "horizon": 12,
            "issue_utc": issue,
            "issue_local": issue,
            "fxx_hours": [1, 2],
        },
    ]

    rows = _fill_hrrr_snapshot_requests(requests, {}, tmp_path, force_refresh=False)

    assert len(calls) == 1
    assert calls[0] == ({"KDAL", "KATL"}, (1, 2))
    assert {row["station_code"]: row["forecast_high_f"] for row in rows} == {"KDAL": 84.0, "KATL": 72.0}
