import pandas as pd

from src.main import write_station_calendar_outputs
from src.station_registry import build_station_map, build_station_registry


def _market_row(slug, date, city, station_code, station_name):
    phrase = station_name or ""
    return {
        "slug": slug,
        "target_date_local": date,
        "city_text_from_title": city,
        "parsed_station_code": station_code,
        "parsed_station_name": station_name,
        "parsed_airport_name": phrase.replace(" Station", "") or None,
        "parsed_country": "US",
        "resolution_source": "",
        "description": f"Recorded at the {station_name}.",
        "rules": "",
        "market_start_time_utc": "2026-05-01T00:00:00Z",
        "market_end_time_utc": "2026-05-02T00:00:00Z",
        "parse_confidence": 0.9,
        "needs_manual_review": False,
    }


def test_manual_override_supplies_exact_station_truth(tmp_path):
    overrides = tmp_path / "manual.yaml"
    overrides.write_text(
        """
overrides:
  - market_slug: highest-temperature-in-example-on-june-1-2026
    station_code: KABC
    station_name: Example Airport Station
    airport_name: Example Airport
    lat: 40.0
    lon: -75.0
    timezone: America/New_York
    country: US
    mapping_confidence: 0.99
""",
        encoding="utf-8",
    )
    markets = pd.DataFrame(
        [
            _market_row(
                "highest-temperature-in-example-on-june-1-2026",
                "2026-06-01",
                "Example",
                None,
                None,
            )
        ]
    )
    station_map = build_station_map(markets, tmp_path, overrides)
    row = station_map.iloc[0]
    assert row["station_code"] == "KABC"
    assert row["lat"] == 40.0
    assert bool(row["needs_manual_review"]) is False


def test_station_mapping_is_date_specific_for_same_city(tmp_path, monkeypatch):
    def fake_enrich(code, raw_dir, force_refresh=False):
        return {"lat": 1.0, "lon": 2.0, "timezone": "America/Chicago", "country": "US"}

    monkeypatch.setattr("src.station_registry.enrich_station", fake_enrich)
    markets = pd.DataFrame(
        [
            _market_row("event-a", "2026-05-17", "Dallas", "KDAL", "Dallas Love Field Station"),
            _market_row("event-b", "2026-05-18", "Dallas", "KDFW", "Dallas-Fort Worth International Airport Station"),
        ]
    )
    station_map = build_station_map(markets, tmp_path, tmp_path / "missing.yaml")
    assert set(station_map["station_code"]) == {"KDAL", "KDFW"}
    assert station_map.groupby("polymarket_city")["station_code"].nunique().loc["Dallas"] == 2


def test_registry_marks_us_actuals_priority(tmp_path, monkeypatch):
    def fake_enrich(code, raw_dir, force_refresh=False):
        return {"lat": 32.8, "lon": -96.8, "timezone": "America/Chicago", "country": "US"}

    monkeypatch.setattr("src.station_registry.enrich_station", fake_enrich)
    markets = pd.DataFrame([_market_row("event-a", "2026-05-17", "Dallas", "KDAL", "Dallas Love Field Station")])
    station_map = build_station_map(markets, tmp_path, tmp_path / "missing.yaml")
    registry = build_station_registry(station_map)
    assert registry.iloc[0]["station_code"] == "KDAL"
    assert "asos_1min" in registry.iloc[0]["actuals_source_priority"]


def test_station_calendar_outputs_skip_polymarket_discovery(tmp_path, monkeypatch):
    def fake_enrich(code, raw_dir, force_refresh=False):
        return {
            "station_code": code,
            "station_name": "Dallas/Love Fld",
            "airport_name": "Dallas/Love Fld",
            "lat": 32.83836,
            "lon": -96.83584,
            "timezone": "America/Chicago",
            "country": "US",
        }

    monkeypatch.setattr("src.main.enrich_station", fake_enrich)
    markets, station_map, registry = write_station_calendar_outputs(
        ["kdal"],
        "2026-05-17",
        "2026-05-18",
        tmp_path / "processed",
        tmp_path / "raw_actuals",
    )
    assert len(markets) == 2
    assert len(station_map) == 2
    assert set(station_map["station_code"]) == {"KDAL"}
    assert station_map.iloc[0]["resolution_source_text"].startswith("station-calendar direct")
    assert bool(station_map.iloc[0]["needs_manual_review"]) is False
    assert bool(registry.iloc[0]["is_active_polymarket_station"]) is False
