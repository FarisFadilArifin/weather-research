from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src import wunderground_history as module


def test_tokyo_metric_history_is_converted_and_written_atomically(tmp_path, monkeypatch):
    start = datetime(2026, 8, 1, tzinfo=UTC)
    observations = [
        {
            "valid_time_gmt": int((start + timedelta(hours=offset)).timestamp()),
            "temp": 25.0 + offset / 11 * 5.0,
        }
        for offset in range(12)
    ]

    def fetch(_self, station, start_date, end_date, *, country, units):
        assert (station, start_date, end_date, country, units) == (
            "RJTT",
            "2026-08-01",
            "2026-08-01",
            "JP",
            "m",
        )
        return observations

    monkeypatch.setattr(module.WeatherCompanyStationHistoryClient, "fetch_observations", fetch)
    output = tmp_path / "settlements.csv"
    frame = module.backfill_wunderground_station_history(
        output,
        stations=["RJTT"],
        station_timezones={"RJTT": "Asia/Tokyo"},
        station_countries={"RJTT": "JP"},
        station_units={"RJTT": "m"},
        start_date="2026-08-01",
        end_date="2026-08-01",
        api_key="test-public-key",
    )

    row = frame.iloc[0]
    assert row["quality_flag"] == "ok"
    assert row["settlement_high_c"] == pytest.approx(30.0)
    assert row["settlement_high_f"] == pytest.approx(86.0)
    assert ":9:JP/" in row["source_url"]
    assert "apiKey" not in row["source_url"]
    assert output.is_file()
