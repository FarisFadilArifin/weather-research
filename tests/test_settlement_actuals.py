from __future__ import annotations

import json

import pandas as pd

from src.settlement_actuals import (
    SOURCE_PRIORITY,
    WeatherCompanyStationHistoryClient,
    _extract_weather_company_daily_high_f,
    _month_chunks,
    _station_history_daily_highs,
    infer_polymarket_settlement_bounds,
    import_manual_settlement_csv,
    merge_settlement_actuals,
    normalize_manual_settlement_frame,
    parse_temperature_bucket,
    write_missing_settlement_template,
)


def test_normalize_manual_settlement_frame_accepts_aliases() -> None:
    frame = pd.DataFrame(
        {
            "station_code": ["katl"],
            "target_date_local": ["2026-06-15"],
            "wunderground_high_f": ["82"],
            "market_url": ["https://example.test/market"],
        }
    )

    out = normalize_manual_settlement_frame(frame, default_source="manual_wunderground")

    assert out.iloc[0]["station_id"] == "KATL"
    assert out.iloc[0]["contract_date"] == "2026-06-15"
    assert out.iloc[0]["settlement_high_f"] == 82
    assert out.iloc[0]["settlement_source"] == "manual_wunderground"


def test_import_manual_settlement_csv_merges_by_station_date(tmp_path) -> None:
    existing = pd.DataFrame(
        {
            "station_id": ["KATL"],
            "contract_date": ["2026-06-15"],
            "settlement_high_f": [80],
            "settlement_source": ["weather_company_pws_history_daily"],
        }
    )
    output = tmp_path / "settlement_actual_highs.csv"
    existing.to_csv(output, index=False)
    incoming = tmp_path / "manual.csv"
    pd.DataFrame(
        {
            "station_id": ["KATL"],
            "contract_date": ["2026-06-15"],
            "settlement_high_f": [82],
        }
    ).to_csv(incoming, index=False)

    out = import_manual_settlement_csv(incoming, output, default_source="manual_polymarket")

    assert len(out) == 1
    assert out.iloc[0]["settlement_high_f"] == 82
    assert out.iloc[0]["settlement_source"] == "manual_polymarket"


def test_write_missing_settlement_template_skips_existing_labels(tmp_path) -> None:
    existing = tmp_path / "settlement_actual_highs.csv"
    pd.DataFrame(
        {
            "station_id": ["KATL"],
            "contract_date": ["2026-06-20"],
            "settlement_high_f": [88.0],
            "settlement_source": ["manual_polymarket"],
        }
    ).to_csv(existing, index=False)

    out = write_missing_settlement_template(
        tmp_path / "missing.csv",
        settlement_path=existing,
        stations=["KATL"],
        start_date="2026-06-19",
        end_date="2026-06-21",
    )

    assert list(out["contract_date"]) == ["2026-06-19", "2026-06-21"]
    assert set(out["station_id"]) == {"KATL"}
    assert out["settlement_high_f"].isna().all()


def test_merge_settlement_actuals_prefers_higher_priority_source() -> None:
    existing = pd.DataFrame(
        {
            "station_id": ["KATL"],
            "contract_date": ["2026-06-15"],
            "settlement_high_f": [80],
            "settlement_source": ["iem_fallback"],
        }
    )
    incoming = pd.DataFrame(
        {
            "station_id": ["KATL"],
            "contract_date": ["2026-06-15"],
            "settlement_high_f": [82],
            "settlement_source": ["manual_wunderground"],
        }
    )

    out = merge_settlement_actuals(existing, incoming)

    assert out.iloc[0]["settlement_high_f"] == 82


def test_extract_weather_company_daily_high_handles_common_shapes() -> None:
    assert _extract_weather_company_daily_high_f({"imperial": {"tempHigh": 81}}) == 81
    assert _extract_weather_company_daily_high_f({"observations": [{"imperial": {"tempHigh": "82.5"}}]}) == 82.5
    assert pd.isna(_extract_weather_company_daily_high_f({"observations": [{"imperial": {}}]}))


def test_station_history_daily_highs_uses_station_local_date() -> None:
    observations = [
        {"valid_time_gmt": 1777351980, "temp": 77},
        {"valid_time_gmt": 1777410000, "temp": 85},
        {"valid_time_gmt": 1777434780, "temp": 79},
    ]

    daily = _station_history_daily_highs(observations, "America/New_York")

    assert daily["2026-04-28"]["high_f"] == 85
    assert daily["2026-04-28"]["observation_count"] == 3


def test_station_history_client_requests_airport_history(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"observations": [{"valid_time_gmt": 1, "temp": 82}]}

    def fake_get(url, params, timeout):
        captured.update({"url": url, "params": params, "timeout": timeout})
        return FakeResponse()

    monkeypatch.setattr("src.settlement_actuals.requests.get", fake_get)
    client = WeatherCompanyStationHistoryClient(api_key="test-key")

    observations = client.fetch_observations("kmia", "2026-04-01", "2026-04-30")

    assert observations[0]["temp"] == 82
    assert captured["url"].endswith("/KMIA:9:US/observations/historical.json")
    assert captured["params"]["startDate"] == "20260401"
    assert captured["params"]["endDate"] == "20260430"
    assert captured["params"]["apiKey"] == "test-key"


def test_station_history_client_supports_international_metric_station(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        url = "https://api.weather.com/v1/location/RKSI:9:KR/observations/historical.json"

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"observations": [{"valid_time_gmt": 1, "temp": 30}]}

    def fake_get(url, params, timeout):
        captured.update({"url": url, "params": params, "timeout": timeout})
        return FakeResponse()

    monkeypatch.setattr("src.settlement_actuals.requests.get", fake_get)
    client = WeatherCompanyStationHistoryClient(api_key="test-key")
    observations = client.fetch_observations(
        "rksi",
        "2026-04-01",
        "2026-04-30",
        country="KR",
        units="m",
    )

    assert observations[0]["temp"] == 30
    assert captured["url"].endswith("/RKSI:9:KR/observations/historical.json")
    assert captured["params"]["units"] == "m"


def test_wunderground_station_history_has_priority_over_fallback() -> None:
    assert SOURCE_PRIORITY["wunderground_station_history"] > SOURCE_PRIORITY["iem_fallback"]
    assert _month_chunks("2026-01-30", "2026-03-02") == [
        ("2026-01-30", "2026-01-31"),
        ("2026-02-01", "2026-02-28"),
        ("2026-03-01", "2026-03-02"),
    ]


def test_parse_temperature_bucket_handles_exact_ranges_and_celsius() -> None:
    assert parse_temperature_bucket("82°F", "F") == (82.0, 82.0, "F", "exact")
    assert parse_temperature_bucket("74-75°F", "F") == (74.0, 75.0, "F", "interval")
    assert parse_temperature_bucket("20°C", "C") == (68.0, 68.0, "C", "exact")
    assert parse_temperature_bucket("73°F or below", "F") == (None, 73.0, "F", "upper_censored")


def test_infer_polymarket_settlement_bounds_from_raw_event(tmp_path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    payload = [
        {
            "id": "event-1",
            "title": "Highest temperature in Dallas on May 17?",
            "description": (
                "This market will resolve to the temperature range that contains the highest temperature recorded "
                "at the Dallas Love Field Station in degrees Fahrenheit on 17 May '26. "
                "https://www.wunderground.com/history/daily/us/tx/dallas/KDAL"
            ),
            "resolutionSource": "https://www.wunderground.com/history/daily/us/tx/dallas/KDAL",
            "closed": True,
            "markets": [
                {
                    "id": "m1",
                    "groupItemTitle": "80°F",
                    "outcomes": '["Yes", "No"]',
                    "outcomePrices": '["1", "0"]',
                },
                {
                    "id": "m2",
                    "groupItemTitle": "81°F",
                    "outcomes": '["Yes", "No"]',
                    "outcomePrices": '["0", "1"]',
                },
            ],
        }
    ]
    (raw / "events_test.json").write_text(json.dumps(payload), encoding="utf-8")

    bounds, exact = infer_polymarket_settlement_bounds(
        raw,
        tmp_path / "bounds.csv",
        exact_output_path=tmp_path / "settlements.csv",
    )

    assert len(bounds) == 1
    assert bounds.iloc[0]["station_id"] == "KDAL"
    assert bounds.iloc[0]["contract_date"] == "2026-05-17"
    assert bounds.iloc[0]["settlement_high_f"] == 80
    assert len(exact) == 1
