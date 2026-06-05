import json

from src.polymarket_parse import (
    infer_final_outcome,
    normalize_event_market,
    parse_city_from_title,
    parse_station_from_resolution,
    parse_target_date_from_text,
)


def test_parse_wunderground_station_url_and_airport_phrase():
    description = (
        "This market will resolve to the temperature range that contains the highest temperature "
        "recorded at the Dallas Love Field Station in degrees Fahrenheit on 17 May '26.\n\n"
        "The resolution source will be information from Wunderground, available here: "
        "https://www.wunderground.com/history/daily/us/tx/dallas/KDAL."
    )
    parsed = parse_station_from_resolution(description=description)
    assert parsed.station_code == "KDAL"
    assert parsed.station_name == "Dallas Love Field Station"
    assert parsed.airport_name == "Dallas Love Field"
    assert parsed.temperature_unit == "F"
    assert parsed.needs_manual_review is False


def test_parse_general_icao_from_wunderground_url():
    description = (
        "Recorded at the Benito Juárez International Airport Station in degrees Celsius. "
        "https://www.wunderground.com/history/daily/mx/mexico-city/MMMX."
    )
    parsed = parse_station_from_resolution(description=description)
    assert parsed.station_code == "MMMX"
    assert parsed.country == "MX"
    assert parsed.temperature_unit == "C"


def test_parse_named_non_us_official_station():
    description = (
        "This market will resolve to the highest temperature recorded by the Hong Kong Observatory "
        'in degrees Celsius, specifically the "Absolute Daily Max (deg. C)" in the Daily Extract.'
    )
    parsed = parse_station_from_resolution(description=description)
    assert parsed.station_code == "HKO"
    assert parsed.station_name == "Hong Kong Observatory"
    assert parsed.needs_manual_review is False


def test_city_and_target_date_are_title_metadata_only():
    title = "Highest temperature in Dallas on May 17?"
    description = "recorded at the Dallas Love Field Station in degrees Fahrenheit on 17 May '26."
    assert parse_city_from_title(title) == "Dallas"
    assert parse_target_date_from_text(title, description) == "2026-05-17"


def test_target_date_parses_abbreviated_resolution_date():
    title = "Highest temperature in Dallas on December 5?"
    description = "recorded at the Dallas Love Field Station in degrees Fahrenheit on 5 Dec '25."
    assert parse_target_date_from_text(title, description) == "2025-12-05"


def test_final_outcome_from_yes_bucket_price():
    event = {
        "markets": [
            {"groupItemTitle": "84-85°F", "outcomes": '["Yes","No"]', "outcomePrices": '["0","1"]'},
            {"groupItemTitle": "86-87°F", "outcomes": '["Yes","No"]', "outcomePrices": '["0.9975","0.0025"]'},
        ]
    }
    assert infer_final_outcome(event) == "86-87°F"


def test_normalize_event_market_outputs_expected_station_fields():
    event = {
        "id": "486828",
        "slug": "highest-temperature-in-dallas-on-may-17-2026",
        "title": "Highest temperature in Dallas on May 17?",
        "description": "recorded at the Dallas Love Field Station in degrees Fahrenheit on 17 May '26.",
        "resolutionSource": "https://www.wunderground.com/history/daily/us/tx/dallas/KDAL",
        "closed": True,
        "active": True,
        "markets": [
            {
                "id": "1",
                "question": "Will the highest temperature in Dallas be 86°F or higher on May 17?",
                "groupItemTitle": "86°F or higher",
                "outcomes": json.dumps(["Yes", "No"]),
                "outcomePrices": json.dumps(["1", "0"]),
            }
        ],
    }
    row = normalize_event_market(event, event["markets"][0])
    assert row["parsed_station_code"] == "KDAL"
    assert row["city_text_from_title"] == "Dallas"
    assert row["target_date_local"] == "2026-05-17"
    assert row["final_outcome"] == "86°F or higher"
