from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import actuals_fetch


def _observations(local_times: list[str], temps: list[float], timezone: str, source: str) -> pd.DataFrame:
    valid_local = pd.to_datetime(local_times).tz_localize(timezone)
    frame = pd.DataFrame(
        {
            "valid_utc": valid_local.tz_convert("UTC"),
            "valid_local": valid_local,
            "tmpf": temps,
        }
    )
    frame.attrs["source"] = source
    return frame


def test_sparse_1min_actuals_fall_back_to_complete_hourly(monkeypatch, tmp_path: Path) -> None:
    station_map = pd.DataFrame(
        [
            {
                "station_code": "KZZZ",
                "target_date_local": "2024-07-01",
                "needs_manual_review": False,
                "station_name": "Test Station",
                "airport_name": "Test Airport",
                "timezone": "America/New_York",
            }
        ]
    )
    registry = station_map[["station_code", "station_name", "airport_name", "timezone"]].copy()

    sparse_1min = _observations(
        [f"2024-07-01 0{hour}:00" for hour in range(8)],
        [55.0, 56.0, 56.0, 58.0, 57.0, 59.0, 60.0, 59.0],
        "America/New_York",
        "iem_asos_1min",
    )
    complete_hourly = _observations(
        [f"2024-07-01 {hour:02d}:00" for hour in range(24)],
        [50.0 + hour for hour in range(24)],
        "America/New_York",
        "iem_asos_hourly",
    )

    monkeypatch.setattr(actuals_fetch, "fetch_iem_asos_1min_range", lambda *args, **kwargs: sparse_1min)
    monkeypatch.setattr(actuals_fetch, "fetch_iem_asos_hourly_range", lambda *args, **kwargs: complete_hourly)

    actuals = actuals_fetch.fetch_actual_highs(station_map, registry, tmp_path)

    assert len(actuals) == 1
    row = actuals.iloc[0]
    assert row["source"] == "iem_asos_hourly"
    assert row["data_quality_flag"] == "ok"
    assert row["raw_observation_count"] == 24
    assert row["actual_high_f"] == 73.0


def test_sparse_1min_actuals_are_kept_when_hourly_is_unavailable(monkeypatch, tmp_path: Path) -> None:
    station_map = pd.DataFrame(
        [
            {
                "station_code": "KZZZ",
                "target_date_local": "2024-07-01",
                "needs_manual_review": False,
                "station_name": "Test Station",
                "airport_name": "Test Airport",
                "timezone": "America/New_York",
            }
        ]
    )
    registry = station_map[["station_code", "station_name", "airport_name", "timezone"]].copy()

    sparse_1min = _observations(
        [f"2024-07-01 0{hour}:00" for hour in range(8)],
        [55.0, 56.0, 56.0, 58.0, 57.0, 59.0, 60.0, 59.0],
        "America/New_York",
        "iem_asos_1min",
    )

    monkeypatch.setattr(actuals_fetch, "fetch_iem_asos_1min_range", lambda *args, **kwargs: sparse_1min)
    monkeypatch.setattr(actuals_fetch, "fetch_iem_asos_hourly_range", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(actuals_fetch, "fetch_iem_asos_1min", lambda *args, **kwargs: sparse_1min)
    monkeypatch.setattr(actuals_fetch, "fetch_iem_asos_hourly", lambda *args, **kwargs: pd.DataFrame())

    actuals = actuals_fetch.fetch_actual_highs(station_map, registry, tmp_path)

    assert len(actuals) == 1
    row = actuals.iloc[0]
    assert row["source"] == "iem_asos_1min"
    assert row["data_quality_flag"] == "sparse_observations"
    assert row["actual_high_f"] == 60.0


def test_suspicious_complete_1min_spike_falls_back_to_hourly(monkeypatch, tmp_path: Path) -> None:
    station_map = pd.DataFrame(
        [
            {
                "station_code": "KZZZ",
                "target_date_local": "2024-07-01",
                "needs_manual_review": False,
                "station_name": "Test Station",
                "airport_name": "Test Airport",
                "timezone": "America/New_York",
            }
        ]
    )
    registry = station_map[["station_code", "station_name", "airport_name", "timezone"]].copy()

    minute_times = pd.date_range("2024-07-01 00:00", periods=1440, freq="min").strftime("%Y-%m-%d %H:%M").tolist()
    minute_temps = [82.0] * 1440
    minute_temps[746] = 122.0
    complete_1min_with_spike = _observations(
        minute_times,
        minute_temps,
        "America/New_York",
        "iem_asos_1min",
    )
    complete_hourly = _observations(
        [f"2024-07-01 {hour:02d}:00" for hour in range(24)],
        [70.0 + min(hour, 14) for hour in range(24)],
        "America/New_York",
        "iem_asos_hourly",
    )

    monkeypatch.setattr(actuals_fetch, "fetch_iem_asos_1min_range", lambda *args, **kwargs: complete_1min_with_spike)
    monkeypatch.setattr(actuals_fetch, "fetch_iem_asos_hourly_range", lambda *args, **kwargs: complete_hourly)

    actuals = actuals_fetch.fetch_actual_highs(station_map, registry, tmp_path)

    assert len(actuals) == 1
    row = actuals.iloc[0]
    assert row["source"] == "iem_asos_hourly"
    assert row["data_quality_flag"] == "ok"
    assert row["actual_high_f"] == 84.0
