from __future__ import annotations

import os
import json
import re
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from .polymarket_parse import (
    clean_text,
    event_is_daily_high_temperature,
    parse_api_datetime,
    parse_jsonish,
    parse_station_from_resolution,
    parse_target_date_from_text,
    parse_temperature_unit,
)


SETTLEMENT_ACTUALS_FILE = "settlement_actual_highs.csv"
POLYMARKET_IMPLIED_BOUNDS_FILE = "polymarket_implied_settlement_bounds.csv"

SETTLEMENT_COLUMNS = [
    "station_id",
    "contract_date",
    "settlement_high_f",
    "settlement_source",
    "source_url",
    "quality_flag",
    "raw_value",
    "notes",
    "fetched_at_utc",
]

SOURCE_PRIORITY = {
    "manual_polymarket": 100,
    "wunderground_station_history": 98,
    "manual_wunderground": 95,
    "polymarket_implied_exact": 92,
    "weather_company_pws_history_daily": 90,
    "wunderground_manual": 85,
    "polymarket_manual": 85,
    "iem_fallback": 10,
}

POLYMARKET_BOUNDS_COLUMNS = [
    "event_id",
    "market_id",
    "event_title",
    "station_id",
    "contract_date",
    "temperature_unit",
    "winning_bucket",
    "lower_bound_f",
    "upper_bound_f",
    "settlement_high_f",
    "inference_quality",
    "source_url",
    "notes",
    "inferred_at_utc",
]

STATION_ALIASES = ("station_id", "station_code", "market_station_code", "airport_code")
DATE_ALIASES = ("contract_date", "date", "date_local", "target_date", "target_date_local")
HIGH_ALIASES = (
    "settlement_high_f",
    "settlement_high",
    "wu_high_f",
    "wunderground_high_f",
    "polymarket_high_f",
    "actual_high_f",
    "high_f",
    "high_temp_f",
)
SOURCE_ALIASES = ("settlement_source", "source", "actual_source")
URL_ALIASES = ("source_url", "url", "market_url", "wunderground_url", "polymarket_url")
QUALITY_ALIASES = ("quality_flag", "data_quality_flag")
NOTES_ALIASES = ("notes", "note", "comment", "comments")


@dataclass(frozen=True)
class WeatherCompanyPwsClient:
    api_key: str
    base_url: str = "https://api.weather.com/v2/pws/history/daily"
    api_query_param: str = "apiKey"
    timeout_seconds: int = 60

    def fetch_daily_high(self, station_id: str, contract_date: str, *, units: str = "e") -> dict[str, Any]:
        day = _parse_date(contract_date).strftime("%Y%m%d")
        params = {
            "stationId": station_id,
            "format": "json",
            "units": units,
            "date": day,
            "numericPrecision": "decimal",
            self.api_query_param: self.api_key,
        }
        response = requests.get(self.base_url, params=params, timeout=self.timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        high = _extract_weather_company_daily_high_f(payload)
        return {
            "station_id": station_id,
            "contract_date": _parse_date(contract_date).isoformat(),
            "settlement_high_f": high,
            "settlement_source": "weather_company_pws_history_daily",
            "source_url": _url_without_secret(response.url, self.api_query_param),
            "quality_flag": "ok" if pd.notna(high) else "missing_temp_high",
            "raw_value": high,
            "notes": pd.NA,
            "fetched_at_utc": datetime.now(UTC).isoformat(),
        }


@dataclass(frozen=True)
class WeatherCompanyStationHistoryClient:
    api_key: str
    base_url: str = "https://api.weather.com/v1/location/{station_id}:9:US/observations/historical.json"
    timeout_seconds: int = 60

    def fetch_observations(self, station_id: str, start_date: str, end_date: str) -> list[dict[str, Any]]:
        url = self.base_url.format(station_id=station_id.upper())
        params = {
            "apiKey": self.api_key,
            "units": "e",
            "startDate": _parse_date(start_date).strftime("%Y%m%d"),
            "endDate": _parse_date(end_date).strftime("%Y%m%d"),
        }
        response = requests.get(url, params=params, timeout=self.timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        observations = payload.get("observations", []) if isinstance(payload, dict) else []
        return [item for item in observations if isinstance(item, dict)]


def discover_weather_company_public_api_key(
    page_url: str = "https://www.wunderground.com/history/daily/us/fl/miami/KMIA",
) -> str:
    response = requests.get(page_url, timeout=60)
    response.raise_for_status()
    matches = re.findall(r"apiKey=([A-Za-z0-9]{24,64})", response.text)
    if not matches:
        raise ValueError("Could not discover the public Weather Company API key from Wunderground.")
    return matches[0]


def backfill_wunderground_station_history(
    output_path: str | Path,
    *,
    stations: Iterable[str],
    station_timezones: dict[str, str],
    start_date: str,
    end_date: str,
    api_key: str | None = None,
    api_key_env: str = "WEATHER_COMPANY_API_KEY",
    sleep_seconds: float = 0.0,
    force_refresh: bool = False,
) -> pd.DataFrame:
    wanted = sorted({str(station).upper().strip() for station in stations if str(station).strip()})
    missing_timezones = [station for station in wanted if not station_timezones.get(station)]
    if missing_timezones:
        raise ValueError(f"Missing station timezones for: {', '.join(missing_timezones)}")

    key = api_key or os.environ.get(api_key_env) or discover_weather_company_public_api_key()
    client = WeatherCompanyStationHistoryClient(api_key=key)
    existing = _read_existing(output_path)
    completed: set[tuple[str, str]] = set()
    if not force_refresh and not existing.empty:
        direct = existing.loc[
            existing["settlement_source"].eq("wunderground_station_history")
            & existing["settlement_high_f"].notna()
            & existing["quality_flag"].eq("ok")
        ]
        completed = set(zip(direct["station_id"], direct["contract_date"], strict=False))

    rows: list[dict[str, Any]] = []
    requested_days = date_range(start_date, end_date)
    for station in wanted:
        timezone = station_timezones[station]
        for chunk_start, chunk_end in _month_chunks(start_date, end_date):
            chunk_days = [day for day in requested_days if chunk_start <= day <= chunk_end]
            missing_days = [day for day in chunk_days if (station, day) not in completed]
            if not missing_days:
                continue
            try:
                observations = client.fetch_observations(station, missing_days[0], missing_days[-1])
                daily = _station_history_daily_highs(observations, timezone)
                fetched_at = datetime.now(UTC).isoformat()
                for day in missing_days:
                    summary = daily.get(day)
                    high = summary["high_f"] if summary else pd.NA
                    count = int(summary["observation_count"]) if summary else 0
                    quality_flag = "ok" if pd.notna(high) and count >= 12 else "sparse_or_unavailable"
                    rows.append(
                        {
                            "station_id": station,
                            "contract_date": day,
                            "settlement_high_f": high if quality_flag == "ok" else pd.NA,
                            "settlement_source": "wunderground_station_history",
                            "source_url": _station_history_source_url(client, station, day, day),
                            "quality_flag": quality_flag,
                            "raw_value": high,
                            "notes": f"observation_count={count}; timezone={timezone}",
                            "fetched_at_utc": fetched_at,
                        }
                    )
            except Exception as exc:  # noqa: BLE001 - preserve failed days for resumable backfills.
                fetched_at = datetime.now(UTC).isoformat()
                for day in missing_days:
                    rows.append(
                        {
                            "station_id": station,
                            "contract_date": day,
                            "settlement_high_f": pd.NA,
                            "settlement_source": "wunderground_station_history",
                            "source_url": pd.NA,
                            "quality_flag": "unavailable",
                            "raw_value": pd.NA,
                            "notes": str(exc),
                            "fetched_at_utc": fetched_at,
                        }
                    )
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    incoming = pd.DataFrame(rows, columns=SETTLEMENT_COLUMNS)
    merged = merge_settlement_actuals(existing, incoming)
    write_settlement_actuals(merged, output_path)
    return merged


def import_manual_settlement_csv(
    input_path: str | Path,
    output_path: str | Path,
    *,
    default_source: str = "manual_polymarket",
) -> pd.DataFrame:
    incoming = normalize_manual_settlement_frame(pd.read_csv(input_path), default_source=default_source)
    existing = _read_existing(output_path)
    merged = merge_settlement_actuals(existing, incoming)
    write_settlement_actuals(merged, output_path)
    return merged


def write_missing_settlement_template(
    output_path: str | Path,
    *,
    settlement_path: str | Path | None = None,
    stations: Iterable[str],
    start_date: str,
    end_date: str,
    default_source: str = "manual_polymarket",
) -> pd.DataFrame:
    existing = _read_existing(settlement_path) if settlement_path is not None else pd.DataFrame(columns=SETTLEMENT_COLUMNS)
    existing_keys = set()
    if not existing.empty:
        complete = existing.loc[existing["settlement_high_f"].notna()].copy()
        existing_keys = set(zip(complete["station_id"], complete["contract_date"], strict=False))

    rows: list[dict[str, Any]] = []
    for station in sorted({str(item).upper().strip() for item in stations if str(item).strip()}):
        for day in date_range(start_date, end_date):
            if (station, day) in existing_keys:
                continue
            rows.append(
                {
                    "station_id": station,
                    "contract_date": day,
                    "settlement_high_f": pd.NA,
                    "settlement_source": default_source,
                    "source_url": pd.NA,
                    "quality_flag": "ok",
                    "raw_value": pd.NA,
                    "notes": "fill settlement_high_f before importing",
                    "fetched_at_utc": pd.NA,
                }
            )
    template = pd.DataFrame(rows, columns=SETTLEMENT_COLUMNS)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(path, index=False)
    return template


def normalize_manual_settlement_frame(frame: pd.DataFrame, *, default_source: str = "manual_polymarket") -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=SETTLEMENT_COLUMNS)

    station_col = _first_existing(frame, STATION_ALIASES)
    date_col = _first_existing(frame, DATE_ALIASES)
    high_col = _first_existing(frame, HIGH_ALIASES)
    if not station_col or not date_col or not high_col:
        raise ValueError(
            "Manual settlement CSV must include station/date/high columns. "
            f"Accepted station={STATION_ALIASES}, date={DATE_ALIASES}, high={HIGH_ALIASES}."
        )

    source_col = _first_existing(frame, SOURCE_ALIASES)
    url_col = _first_existing(frame, URL_ALIASES)
    quality_col = _first_existing(frame, QUALITY_ALIASES)
    notes_col = _first_existing(frame, NOTES_ALIASES)

    out = pd.DataFrame(
        {
            "station_id": frame[station_col].astype(str).str.upper().str.strip(),
            "contract_date": pd.to_datetime(frame[date_col], errors="coerce").dt.date.astype("string"),
            "settlement_high_f": pd.to_numeric(frame[high_col], errors="coerce"),
            "settlement_source": frame[source_col].astype(str).str.strip() if source_col else default_source,
            "source_url": frame[url_col].astype(str).str.strip() if url_col else pd.NA,
            "quality_flag": frame[quality_col].astype(str).str.strip() if quality_col else "ok",
            "raw_value": frame[high_col],
            "notes": frame[notes_col].astype(str).str.strip() if notes_col else pd.NA,
            "fetched_at_utc": datetime.now(UTC).isoformat(),
        }
    )
    out = out.dropna(subset=["station_id", "contract_date"])
    out = out.loc[out["settlement_high_f"].notna()].copy()
    return _coerce_settlement_columns(out)


def backfill_weather_company_pws_history(
    output_path: str | Path,
    *,
    stations: Iterable[str],
    start_date: str,
    end_date: str,
    api_key: str | None = None,
    api_key_env: str = "WEATHER_COMPANY_API_KEY",
    api_query_param: str = "apiKey",
    sleep_seconds: float = 0.0,
    force_refresh: bool = False,
) -> pd.DataFrame:
    key = api_key or os.environ.get(api_key_env)
    if not key:
        raise ValueError(f"Missing Weather Company API key. Set --api-key or environment variable {api_key_env}.")

    existing = _read_existing(output_path)
    client = WeatherCompanyPwsClient(api_key=key, api_query_param=api_query_param)
    rows: list[dict[str, Any]] = []
    existing_keys = set()
    if not existing.empty and not force_refresh:
        existing_keys = set(zip(existing["station_id"], existing["contract_date"], strict=False))

    for station in sorted({str(s).upper().strip() for s in stations if str(s).strip()}):
        for day in date_range(start_date, end_date):
            key_tuple = (station, day)
            if key_tuple in existing_keys:
                continue
            try:
                rows.append(client.fetch_daily_high(station, day))
            except Exception as exc:  # noqa: BLE001 - preserve unavailable days and continue long backfills.
                rows.append(
                    {
                        "station_id": station,
                        "contract_date": day,
                        "settlement_high_f": pd.NA,
                        "settlement_source": "weather_company_pws_history_daily",
                        "source_url": pd.NA,
                        "quality_flag": "unavailable",
                        "raw_value": pd.NA,
                        "notes": str(exc),
                        "fetched_at_utc": datetime.now(UTC).isoformat(),
                    }
                )
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    incoming = pd.DataFrame(rows, columns=SETTLEMENT_COLUMNS)
    merged = merge_settlement_actuals(existing, incoming)
    write_settlement_actuals(merged, output_path)
    return merged


def infer_polymarket_settlement_bounds(
    raw_dir: str | Path,
    bounds_output_path: str | Path,
    *,
    exact_output_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for event in _iter_polymarket_events(raw_dir):
        row = _infer_polymarket_event_settlement(event)
        if row:
            rows.append(row)

    bounds = pd.DataFrame(rows, columns=POLYMARKET_BOUNDS_COLUMNS)
    if not bounds.empty:
        bounds = bounds.sort_values(["station_id", "contract_date", "event_id"]).drop_duplicates(
            subset=["station_id", "contract_date", "event_id"],
            keep="first",
        )
    Path(bounds_output_path).parent.mkdir(parents=True, exist_ok=True)
    bounds.to_csv(bounds_output_path, index=False)

    exact = _exact_settlement_rows_from_bounds(bounds)
    if exact_output_path is not None:
        existing = _read_existing(exact_output_path)
        merged = merge_settlement_actuals(existing, exact)
        write_settlement_actuals(merged, exact_output_path)
        exact = merged
    return bounds, exact


def merge_settlement_actuals(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    frames = [_coerce_settlement_columns(frame) for frame in (existing, incoming) if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame(columns=SETTLEMENT_COLUMNS)

    combined = pd.concat(frames, ignore_index=True)
    combined["source_priority"] = combined["settlement_source"].map(SOURCE_PRIORITY).fillna(50)
    combined["has_high"] = combined["settlement_high_f"].notna().astype(int)
    combined = combined.sort_values(
        ["station_id", "contract_date", "has_high", "source_priority", "fetched_at_utc"],
        ascending=[True, True, False, False, False],
    )
    combined = combined.drop_duplicates(subset=["station_id", "contract_date"], keep="first")
    combined = combined.drop(columns=["source_priority", "has_high"])
    return combined[SETTLEMENT_COLUMNS].sort_values(["station_id", "contract_date"]).reset_index(drop=True)


def write_settlement_actuals(frame: pd.DataFrame, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _coerce_settlement_columns(frame).to_csv(path, index=False)


def default_output_path(processed_dir: str | Path = "data/processed") -> Path:
    return Path(processed_dir) / SETTLEMENT_ACTUALS_FILE


def default_polymarket_bounds_output_path(processed_dir: str | Path = "data/processed") -> Path:
    return Path(processed_dir) / POLYMARKET_IMPLIED_BOUNDS_FILE


def date_range(start_date: str, end_date: str) -> list[str]:
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    if end < start:
        raise ValueError("end_date must be on or after start_date")
    days: list[str] = []
    current = start
    while current <= end:
        days.append(current.isoformat())
        current += timedelta(days=1)
    return days


def _month_chunks(start_date: str, end_date: str) -> list[tuple[str, str]]:
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    chunks: list[tuple[str, str]] = []
    cursor = start
    while cursor <= end:
        next_month = (cursor.replace(day=28) + timedelta(days=4)).replace(day=1)
        chunk_end = min(end, next_month - timedelta(days=1))
        chunks.append((cursor.isoformat(), chunk_end.isoformat()))
        cursor = chunk_end + timedelta(days=1)
    return chunks


def _station_history_daily_highs(
    observations: Iterable[dict[str, Any]],
    timezone: str,
) -> dict[str, dict[str, float | int]]:
    tz = ZoneInfo(timezone)
    temperatures_by_day: dict[str, list[float]] = {}
    for observation in observations:
        timestamp = pd.to_numeric(pd.Series([observation.get("valid_time_gmt")]), errors="coerce").iloc[0]
        temperature = pd.to_numeric(pd.Series([observation.get("temp")]), errors="coerce").iloc[0]
        if pd.isna(timestamp) or pd.isna(temperature):
            continue
        local_day = datetime.fromtimestamp(float(timestamp), tz=UTC).astimezone(tz).date().isoformat()
        temperatures_by_day.setdefault(local_day, []).append(float(temperature))
    return {
        day: {"high_f": max(values), "observation_count": len(values)}
        for day, values in temperatures_by_day.items()
        if values
    }


def _station_history_source_url(
    client: WeatherCompanyStationHistoryClient,
    station_id: str,
    start_date: str,
    end_date: str,
) -> str:
    base_url = client.base_url.format(station_id=station_id.upper())
    return (
        f"{base_url}?units=e&startDate={_parse_date(start_date):%Y%m%d}"
        f"&endDate={_parse_date(end_date):%Y%m%d}"
    )


def parse_temperature_bucket(bucket: str, default_unit: str | None) -> tuple[float | None, float | None, str | None, str]:
    text = clean_text(bucket)
    unit = _bucket_unit(text) or default_unit
    normalized = text.replace("°", "").replace("–", "-").replace("—", "-")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    unit_suffix = r"(?:\s*[FC])?"
    number = r"-?\d+(?:\.\d+)?"

    match = re.search(rf"(?P<value>{number}){unit_suffix}\s+or\s+below", normalized, flags=re.IGNORECASE)
    if match:
        upper = _to_f(float(match.group("value")), unit)
        return None, upper, unit, "upper_censored"

    match = re.search(rf"(?P<value>{number}){unit_suffix}\s+(?:or\s+)?(?:above|higher|more|\+)", normalized, flags=re.IGNORECASE)
    if match:
        lower = _to_f(float(match.group("value")), unit)
        return lower, None, unit, "lower_censored"

    match = re.search(rf"(?P<lower>{number}){unit_suffix}\s*-\s*(?P<upper>{number}){unit_suffix}", normalized, flags=re.IGNORECASE)
    if match:
        lower = _to_f(float(match.group("lower")), unit)
        upper = _to_f(float(match.group("upper")), unit)
        quality = "exact" if lower == upper else "interval"
        return lower, upper, unit, quality

    match = re.fullmatch(rf"\s*(?P<value>{number}){unit_suffix}\s*", normalized, flags=re.IGNORECASE)
    if match:
        value = _to_f(float(match.group("value")), unit)
        return value, value, unit, "exact"

    return None, None, unit, "unparsed"


def _extract_weather_company_daily_high_f(payload: Any) -> float | pd.NA:
    records = payload.get("observations") if isinstance(payload, dict) else None
    if records is None:
        records = payload.get("summaries") if isinstance(payload, dict) else None
    if records is None:
        records = payload if isinstance(payload, list) else [payload]

    for record in records:
        if not isinstance(record, dict):
            continue
        units = record.get("imperial") or record.get("metric") or {}
        high = units.get("tempHigh") if isinstance(units, dict) else None
        if high is None:
            high = record.get("tempHigh") or record.get("temperatureHigh")
        if high is not None:
            value = pd.to_numeric(pd.Series([high]), errors="coerce").iloc[0]
            if pd.notna(value):
                return float(value)
    return pd.NA


def _iter_polymarket_events(raw_dir: str | Path) -> list[dict[str, Any]]:
    raw_path = Path(raw_dir)
    events_by_id: dict[str, dict[str, Any]] = {}
    for path in sorted([*raw_path.glob("events_*.json"), *raw_path.glob("public_search_*.json")]):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            page = payload.get("events", payload.get("data", []))
        else:
            page = payload
        if not isinstance(page, list):
            continue
        for event in page:
            if not isinstance(event, dict):
                continue
            event_id = str(event.get("id") or event.get("slug") or hash(json.dumps(event, sort_keys=True, default=str)))
            events_by_id.setdefault(event_id, event)
    return list(events_by_id.values())


def _infer_polymarket_event_settlement(event: dict[str, Any]) -> dict[str, Any] | None:
    if not event_is_daily_high_temperature(event):
        return None
    title = clean_text(event.get("title"))
    description = clean_text(event.get("description"))
    rules = clean_text(event.get("rules"))
    resolution_source = clean_text(event.get("resolutionSource"))
    parsed = parse_station_from_resolution(description=description, rules=rules, resolution_source=resolution_source)
    station_id = parsed.station_code
    contract_date = parse_target_date_from_text(title, description, rules)
    if not station_id or not contract_date:
        return None

    market = _winning_yes_market(event)
    if not market:
        return None
    bucket = clean_text(market.get("groupItemTitle")) or _bucket_from_binary_question(clean_text(market.get("question")))
    if not bucket:
        return None

    default_unit = parsed.temperature_unit or parse_temperature_unit(title, description, rules, bucket)
    lower, upper, unit, quality = parse_temperature_bucket(bucket, default_unit)
    settlement = lower if quality == "exact" and lower == upper else pd.NA
    notes = pd.NA if quality != "unparsed" else "Could not parse winning bucket"
    return {
        "event_id": event.get("id"),
        "market_id": market.get("id"),
        "event_title": title,
        "station_id": station_id.upper(),
        "contract_date": contract_date,
        "temperature_unit": unit,
        "winning_bucket": bucket,
        "lower_bound_f": lower,
        "upper_bound_f": upper,
        "settlement_high_f": settlement,
        "inference_quality": quality,
        "source_url": resolution_source or market.get("resolutionSource"),
        "notes": notes,
        "inferred_at_utc": datetime.now(UTC).isoformat(),
    }


def _winning_yes_market(event: dict[str, Any]) -> dict[str, Any] | None:
    for market in event.get("markets") or []:
        outcomes = parse_jsonish(market.get("outcomes"), [])
        prices = parse_jsonish(market.get("outcomePrices"), [])
        if not outcomes or not prices:
            continue
        try:
            yes_index = [str(outcome).lower() for outcome in outcomes].index("yes")
            yes_price = float(prices[yes_index])
        except (ValueError, TypeError, IndexError):
            continue
        if yes_price >= 0.99:
            return market
    return None


def _bucket_from_binary_question(question: str) -> str | None:
    patterns = [
        r"be (?P<bucket>-?\d+(?:\.\d+)?\s*°?\s*[FC]?\s+or\s+higher)\b",
        r"be (?P<bucket>-?\d+(?:\.\d+)?\s*°?\s*[FC]?\s+or\s+above)\b",
        r"be (?P<bucket>-?\d+(?:\.\d+)?\s*°?\s*[FC]?\s+or\s+below)\b",
        r"be (?P<bucket>-?\d+(?:\.\d+)?\s*°?\s*[FC]?)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match:
            return clean_text(match.group("bucket"))
    return None


def _exact_settlement_rows_from_bounds(bounds: pd.DataFrame) -> pd.DataFrame:
    if bounds.empty:
        return pd.DataFrame(columns=SETTLEMENT_COLUMNS)
    exact = bounds.loc[bounds["inference_quality"].astype(str).eq("exact")].copy()
    if exact.empty:
        return pd.DataFrame(columns=SETTLEMENT_COLUMNS)
    out = pd.DataFrame(
        {
            "station_id": exact["station_id"],
            "contract_date": exact["contract_date"],
            "settlement_high_f": exact["settlement_high_f"],
            "settlement_source": "polymarket_implied_exact",
            "source_url": exact["source_url"],
            "quality_flag": "ok",
            "raw_value": exact["winning_bucket"],
            "notes": exact["event_title"],
            "fetched_at_utc": exact["inferred_at_utc"],
        }
    )
    return _coerce_settlement_columns(out)


def _bucket_unit(text: str) -> str | None:
    if re.search(r"\bF\b|°F", text, flags=re.IGNORECASE):
        return "F"
    if re.search(r"\bC\b|°C", text, flags=re.IGNORECASE):
        return "C"
    return None


def _to_f(value: float, unit: str | None) -> float:
    if str(unit).upper() == "C":
        return round(value * 9 / 5 + 32, 2)
    return float(value)


def _read_existing(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=SETTLEMENT_COLUMNS)
    return _coerce_settlement_columns(pd.read_csv(path))


def _coerce_settlement_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in SETTLEMENT_COLUMNS:
        if column not in out:
            out[column] = pd.NA
    out["station_id"] = out["station_id"].astype(str).str.upper().str.strip()
    out["contract_date"] = pd.to_datetime(out["contract_date"], errors="coerce").dt.date.astype("string")
    out["settlement_high_f"] = pd.to_numeric(out["settlement_high_f"], errors="coerce")
    return out[SETTLEMENT_COLUMNS]


def _parse_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    parsed = pd.to_datetime(value, errors="raise")
    return parsed.date()


def _first_existing(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lower_to_actual = {column.lower(): column for column in frame.columns}
    return next((lower_to_actual[candidate.lower()] for candidate in candidates if candidate.lower() in lower_to_actual), None)


def _url_without_secret(url: str, secret_param: str) -> str:
    marker = f"{secret_param}="
    if marker not in url:
        return url
    prefix, suffix = url.split(marker, 1)
    tail = suffix.split("&", 1)
    return f"{prefix}{marker}<redacted>" + (f"&{tail[1]}" if len(tail) == 2 else "")
