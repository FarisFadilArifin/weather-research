from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml

from .polymarket_parse import KNOWN_OFFICIAL_STATIONS, clean_text


STATION_MAP_COLUMNS = [
    "polymarket_city",
    "market_slug",
    "target_date_local",
    "station_code",
    "station_name",
    "airport_name",
    "lat",
    "lon",
    "timezone",
    "country",
    "resolution_source_text",
    "first_seen_market_date",
    "last_seen_market_date",
    "mapping_confidence",
    "needs_manual_review",
]

REGISTRY_COLUMNS = [
    "station_code",
    "station_name",
    "airport_name",
    "city_label",
    "lat",
    "lon",
    "timezone",
    "country",
    "actuals_source_priority",
    "is_active_polymarket_station",
]


def load_overrides(path: str | Path) -> list[dict[str, Any]]:
    override_path = Path(path)
    if not override_path.exists():
        return []
    data = yaml.safe_load(override_path.read_text(encoding="utf-8")) or {}
    return data.get("overrides", []) or []


def build_station_map(
    markets: pd.DataFrame,
    raw_actuals_dir: str | Path,
    overrides_path: str | Path = "config/manual_station_overrides.yaml",
    force_refresh: bool = False,
) -> pd.DataFrame:
    overrides = load_overrides(overrides_path)
    rows: list[dict[str, Any]] = []
    if markets.empty:
        return pd.DataFrame(columns=STATION_MAP_COLUMNS)

    event_rows = markets.drop_duplicates(subset=["slug", "target_date_local", "parsed_station_code", "parsed_station_name"])
    for _, row in event_rows.iterrows():
        base = _row_to_station_map(row)
        override = _find_override(base, overrides)
        if override:
            base.update(_override_to_map_values(override))
        if base["station_code"] and (pd.isna(base["lat"]) or pd.isna(base["lon"])):
            base.update(enrich_station(base["station_code"], raw_actuals_dir, force_refresh=force_refresh))
        known = _known_station_values(base["station_code"])
        if known:
            base.update(known)
        base["needs_manual_review"] = bool(
            base.get("needs_manual_review")
            or not base.get("station_code")
            or pd.isna(base.get("lat"))
            or pd.isna(base.get("lon"))
            or not base.get("timezone")
        )
        rows.append(base)

    frame = pd.DataFrame(rows)
    for column in STATION_MAP_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    return frame[STATION_MAP_COLUMNS]


def _row_to_station_map(row: pd.Series) -> dict[str, Any]:
    return {
        "polymarket_city": row.get("city_text_from_title"),
        "market_slug": row.get("slug"),
        "target_date_local": row.get("target_date_local"),
        "station_code": _none_if_na(row.get("parsed_station_code")),
        "station_name": _none_if_na(row.get("parsed_station_name")),
        "airport_name": _none_if_na(row.get("parsed_airport_name")),
        "lat": pd.NA,
        "lon": pd.NA,
        "timezone": pd.NA,
        "country": _none_if_na(row.get("parsed_country")),
        "resolution_source_text": " ".join(
            clean_text(row.get(key)) for key in ("resolution_source", "description", "rules") if clean_text(row.get(key))
        ),
        "first_seen_market_date": row.get("market_start_time_utc"),
        "last_seen_market_date": row.get("market_end_time_utc"),
        "mapping_confidence": row.get("parse_confidence"),
        "needs_manual_review": bool(row.get("needs_manual_review")),
    }


def _none_if_na(value: Any) -> Any:
    return None if pd.isna(value) else value


def _find_override(base: dict[str, Any], overrides: list[dict[str, Any]]) -> dict[str, Any] | None:
    for override in overrides:
        if override.get("market_slug") and override["market_slug"] == base.get("market_slug"):
            return override
        if override.get("station_code") and override.get("polymarket_city") == base.get("polymarket_city"):
            start = override.get("start_date")
            end = override.get("end_date")
            date = base.get("target_date_local")
            if (not start or date >= start) and (not end or date <= end):
                return override
    return None


def _override_to_map_values(override: dict[str, Any]) -> dict[str, Any]:
    return {
        "station_code": override.get("station_code"),
        "station_name": override.get("station_name"),
        "airport_name": override.get("airport_name"),
        "lat": override.get("lat"),
        "lon": override.get("lon"),
        "timezone": override.get("timezone"),
        "country": override.get("country"),
        "mapping_confidence": override.get("mapping_confidence", 0.95),
        "needs_manual_review": False,
        "resolution_source_text": override.get("resolution_source_text"),
    }


def _known_station_values(code: str) -> dict[str, Any]:
    info = {}
    for candidate in KNOWN_OFFICIAL_STATIONS.values():
        if candidate.get("station_code") == code:
            info = candidate
            break
    if not info:
        return {}
    return {
        "station_name": info.get("station_name"),
        "airport_name": info.get("airport_name"),
        "lat": info.get("lat"),
        "lon": info.get("lon"),
        "timezone": info.get("timezone"),
        "country": info.get("country"),
        "needs_manual_review": False,
    }


def enrich_station(station_code: str, raw_dir: str | Path, force_refresh: bool = False) -> dict[str, Any]:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache = raw_dir / f"stationinfo_{station_code.upper()}.json"
    if cache.exists() and not force_refresh:
        data = json.loads(cache.read_text(encoding="utf-8"))
    else:
        params = {"ids": station_code.upper(), "format": "json"}
        try:
            response = requests.get(
                "https://aviationweather.gov/api/data/stationinfo",
                params=params,
                timeout=30,
                headers={"User-Agent": "weather-research/0.1"},
            )
            response.raise_for_status()
            data = response.json()
            cache.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 - enrichment should not stop the whole research run.
            logging.warning("Could not enrich station %s: %s", station_code, exc)
            return {}
    records = data if isinstance(data, list) else data.get("value", data)
    if isinstance(records, dict):
        records = [records]
    if not records:
        return {}
    item = records[0]
    lat = item.get("lat")
    lon = item.get("lon")
    return {
        "station_code": item.get("icaoId") or station_code.upper(),
        "station_name": item.get("site") or item.get("name"),
        "airport_name": item.get("site") or item.get("name"),
        "lat": lat,
        "lon": lon,
        "timezone": timezone_from_latlon(lat, lon),
        "country": item.get("country"),
    }


def timezone_from_latlon(lat: Any, lon: Any) -> str | None:
    if pd.isna(lat) or pd.isna(lon):
        return None
    try:
        from timezonefinder import TimezoneFinder

        return TimezoneFinder().timezone_at(lat=float(lat), lng=float(lon))
    except Exception:  # noqa: BLE001 - optional dependency fallback.
        return None


def build_station_registry(station_map: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if station_map.empty:
        return pd.DataFrame(columns=REGISTRY_COLUMNS)
    for code, group in station_map.dropna(subset=["station_code"]).groupby("station_code", dropna=False):
        first = group.iloc[0]
        country = first.get("country")
        source_priority = (
            "asos_1min,metar_asos,noaa_lcd"
            if country == "US"
            else "official_station_source,metar_asos,manual_verified"
        )
        rows.append(
            {
                "station_code": code,
                "station_name": first.get("station_name"),
                "airport_name": first.get("airport_name"),
                "city_label": first.get("polymarket_city"),
                "lat": first.get("lat"),
                "lon": first.get("lon"),
                "timezone": first.get("timezone"),
                "country": country,
                "actuals_source_priority": source_priority,
                "is_active_polymarket_station": True,
            }
        )
    frame = pd.DataFrame(rows)
    for column in REGISTRY_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    return frame[REGISTRY_COLUMNS]


def write_station_outputs(
    markets: pd.DataFrame,
    processed_dir: str | Path,
    raw_actuals_dir: str | Path,
    overrides_path: str | Path,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    processed = Path(processed_dir)
    processed.mkdir(parents=True, exist_ok=True)
    station_map = build_station_map(markets, raw_actuals_dir, overrides_path, force_refresh=force_refresh)
    registry = build_station_registry(station_map)
    station_map.to_csv(processed / "polymarket_station_map.csv", index=False)
    registry.to_csv(processed / "station_registry.csv", index=False)
    return station_map, registry


def source_cache_path(raw_dir: str | Path, prefix: str, params: dict[str, Any]) -> Path:
    digest = hashlib.sha1(json.dumps(params, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]
    return Path(raw_dir) / f"{prefix}_{digest}.json"
