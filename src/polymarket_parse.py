from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any


MONTHS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}

KNOWN_OFFICIAL_STATIONS = {
    "hong kong observatory": {
        "station_code": "HKO",
        "station_name": "Hong Kong Observatory",
        "airport_name": "",
        "country": "HK",
        "timezone": "Asia/Hong_Kong",
        "lat": 22.302,
        "lon": 114.174,
        "confidence": 0.8,
    }
}


@dataclass(frozen=True)
class ParsedStation:
    station_code: str | None = None
    station_name: str | None = None
    airport_name: str | None = None
    country: str | None = None
    temperature_unit: str | None = None
    confidence: float = 0.0
    needs_manual_review: bool = True
    resolution_source_text: str = ""


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    replacements = {
        "Â°F": "°F",
        "Â°C": "°C",
        "â": '"',
        "â": '"',
        "â": "'",
        "\u00a0": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return re.sub(r"\s+", " ", text).strip()


def parse_jsonish(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (list, dict)):
        return value
    if not isinstance(value, str):
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def event_is_daily_high_temperature(event: dict[str, Any]) -> bool:
    text = " ".join(
        clean_text(event.get(key))
        for key in ("title", "slug", "description", "category", "subcategory")
    ).lower()
    return (
        "highest temperature" in text
        or "daily high" in text
        or re.search(r"\bhigh temperature\b", text) is not None
    ) and not any(blocked in text for blocked in ("space weather", "hottest year", "sea ice"))


def parse_city_from_title(title: str) -> str | None:
    title = clean_text(title)
    patterns = [
        r"Highest temperature in (?P<city>.+?) on ",
        r"high temperature in (?P<city>.+?) (?:be|on)",
        r"temperature in (?P<city>.+?) on ",
    ]
    for pattern in patterns:
        match = re.search(pattern, title, flags=re.IGNORECASE)
        if match:
            return match.group("city").strip(" ?")
    return None


def parse_target_date_from_text(*texts: str) -> str | None:
    combined = " ".join(clean_text(text) for text in texts if text)
    month_pattern = (
        "January|Jan|February|Feb|March|Mar|April|Apr|May|June|Jun|July|Jul|"
        "August|Aug|September|Sept|Sep|October|Oct|November|Nov|December|Dec"
    )
    patterns = [
        rf"\b(?P<day>\d{{1,2}}) (?P<month>{month_pattern}) ['’]?(?P<year>\d{{2,4}})\b",
        rf"\b(?P<month>{month_pattern}) (?P<day>\d{{1,2}})(?:st|nd|rd|th)?,? (?P<year>\d{{4}})\b",
        rf"\b(?P<month>{month_pattern}) (?P<day>\d{{1,2}})\b.*?\b(?P<year>20\d{{2}})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, combined, flags=re.IGNORECASE)
        if not match:
            continue
        month = MONTHS[match.group("month").lower()]
        day = int(match.group("day"))
        year = int(match.group("year"))
        if year < 100:
            year += 2000
        return f"{year:04d}-{month:02d}-{day:02d}"
    return None


def parse_temperature_unit(*texts: str) -> str | None:
    text = " ".join(clean_text(t) for t in texts if t).lower()
    if "fahrenheit" in text or "°f" in text:
        return "F"
    if "celsius" in text or "°c" in text:
        return "C"
    return None


def parse_station_from_resolution(
    description: str = "",
    rules: str = "",
    resolution_source: str = "",
) -> ParsedStation:
    source_text = "\n".join(clean_text(t) for t in (resolution_source, description, rules) if t)
    text = clean_text(source_text)
    unit = parse_temperature_unit(text)

    code = _extract_station_code_from_url(text)
    phrase_name = _extract_station_phrase(text)
    if code:
        confidence = 0.95 if "wunderground.com" in text.lower() else 0.88
        return ParsedStation(
            station_code=code,
            station_name=phrase_name,
            airport_name=_airport_name_from_station_phrase(phrase_name),
            country=_country_from_code(code),
            temperature_unit=unit,
            confidence=confidence,
            needs_manual_review=False,
            resolution_source_text=text,
        )

    code = _extract_icao_code(text)
    if code:
        return ParsedStation(
            station_code=code,
            station_name=phrase_name,
            airport_name=_airport_name_from_station_phrase(phrase_name),
            country=_country_from_code(code),
            temperature_unit=unit,
            confidence=0.82,
            needs_manual_review=False,
            resolution_source_text=text,
        )

    lower = text.lower()
    for needle, info in KNOWN_OFFICIAL_STATIONS.items():
        if needle in lower:
            return ParsedStation(
                station_code=info["station_code"],
                station_name=info["station_name"],
                airport_name=info["airport_name"],
                country=info["country"],
                temperature_unit=unit,
                confidence=float(info["confidence"]),
                needs_manual_review=False,
                resolution_source_text=text,
            )

    if phrase_name:
        return ParsedStation(
            station_name=phrase_name,
            airport_name=_airport_name_from_station_phrase(phrase_name),
            temperature_unit=unit,
            confidence=0.55,
            needs_manual_review=True,
            resolution_source_text=text,
        )

    return ParsedStation(
        temperature_unit=unit,
        confidence=0.0,
        needs_manual_review=True,
        resolution_source_text=text,
    )


def _extract_station_code_from_url(text: str) -> str | None:
    patterns = [
        r"wunderground\.com/history/daily/[^\s)]+/([A-Z0-9]{3,5})(?:[.?&\s)]|$)",
        r"/history/daily/[^\s)]+/([A-Z0-9]{3,5})(?:[.?&\s)]|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def _extract_icao_code(text: str) -> str | None:
    us_match = re.search(r"\b(K[A-Z]{3})\b", text)
    if us_match:
        return us_match.group(1)
    candidates = re.findall(r"\b[A-Z]{4}\b", text)
    blocked = {"THIS", "WILL", "FROM", "ONCE", "DATA", "DATE", "YES", "NOAA", "NWS"}
    for candidate in candidates:
        if candidate not in blocked and re.search(rf"(station|airport|observations?|source).{{0,80}}\b{candidate}\b|\b{candidate}\b.{{0,80}}(station|airport|observations?)", text, flags=re.IGNORECASE):
            return candidate
    return None


def _extract_station_phrase(text: str) -> str | None:
    patterns = [
        r"recorded at the (?P<name>.+? Station)\b",
        r"Forecast for the (?P<name>.+? Station)\b",
        r"observations? at (?P<name>.+?)(?: once|,|\.|\n)",
        r"from (?P<name>.+? International Airport)\b",
        r"at (?P<name>.+? International Airport)\b",
        r"at (?P<name>.+? Airport)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return clean_text(match.group("name")).strip(" .")
    return None


def _airport_name_from_station_phrase(name: str | None) -> str | None:
    if not name:
        return None
    cleaned = re.sub(r"\s+Station$", "", name, flags=re.IGNORECASE)
    return cleaned.strip() or None


def _country_from_code(code: str | None) -> str | None:
    if not code:
        return None
    if re.fullmatch(r"K[A-Z]{3}", code):
        return "US"
    if re.fullmatch(r"P[A-Z]{3}", code):
        return "US"
    if code.startswith("C"):
        return "CA"
    if code.startswith("MM"):
        return "MX"
    return None


def parse_outcome_buckets(event: dict[str, Any]) -> list[str]:
    buckets: list[str] = []
    for market in event.get("markets") or []:
        title = clean_text(market.get("groupItemTitle"))
        if title:
            buckets.append(title)
            continue
        question = clean_text(market.get("question"))
        bucket = parse_bucket_from_question(question)
        if bucket:
            buckets.append(bucket)
    return buckets


def parse_bucket_from_question(question: str) -> str | None:
    text = clean_text(question)
    match = re.search(r"be (?P<bucket>.+?) on ", text, flags=re.IGNORECASE)
    if match:
        return match.group("bucket")
    return None


def infer_final_outcome(event: dict[str, Any]) -> str | None:
    for market in event.get("markets") or []:
        prices = parse_jsonish(market.get("outcomePrices"), [])
        outcomes = parse_jsonish(market.get("outcomes"), [])
        if not prices or not outcomes:
            continue
        try:
            yes_index = [str(o).lower() for o in outcomes].index("yes")
            yes_price = float(prices[yes_index])
        except (ValueError, TypeError, IndexError):
            continue
        if yes_price >= 0.99:
            return clean_text(market.get("groupItemTitle")) or parse_bucket_from_question(market.get("question", ""))
    return None


def normalize_event_market(event: dict[str, Any], market: dict[str, Any]) -> dict[str, Any]:
    event_description = clean_text(event.get("description"))
    market_description = clean_text(market.get("description"))
    rules = clean_text(market.get("rules") or event.get("rules"))
    resolution_source = clean_text(market.get("resolutionSource") or event.get("resolutionSource"))
    parsed = parse_station_from_resolution(
        description=market_description or event_description,
        rules=rules,
        resolution_source=resolution_source,
    )
    title = clean_text(event.get("title") or market.get("question"))
    buckets = parse_outcome_buckets(event)
    return {
        "event_id": event.get("id"),
        "market_id": market.get("id"),
        "slug": event.get("slug") or market.get("slug"),
        "title": title,
        "description": market_description or event_description,
        "rules": rules,
        "resolution_source": resolution_source,
        "market_start_time_utc": market.get("startDate") or event.get("startDate"),
        "market_end_time_utc": market.get("endDate") or event.get("endDate"),
        "target_date_local": parse_target_date_from_text(title, market_description, event_description),
        "city_text_from_title": parse_city_from_title(title),
        "parsed_station_name": parsed.station_name,
        "parsed_station_code": parsed.station_code,
        "parsed_airport_name": parsed.airport_name,
        "parsed_country": parsed.country,
        "temperature_unit": parsed.temperature_unit,
        "outcome_buckets": json.dumps(buckets, ensure_ascii=False),
        "final_outcome": infer_final_outcome(event),
        "is_resolved": bool(event.get("closed") or market.get("closed")),
        "is_active": bool(event.get("active") or market.get("active")),
        "parse_confidence": parsed.confidence,
        "needs_manual_review": parsed.needs_manual_review,
    }


def parse_api_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None
