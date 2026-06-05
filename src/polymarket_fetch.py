from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal environments.
    def tqdm(iterable=None, **kwargs):
        class _Noop:
            def update(self, *_args, **_kwargs):
                return None

            def close(self):
                return None

        return iterable if iterable is not None else _Noop()

from .polymarket_parse import event_is_daily_high_temperature, normalize_event_market


MARKET_COLUMNS = [
    "event_id",
    "market_id",
    "slug",
    "title",
    "description",
    "rules",
    "resolution_source",
    "market_start_time_utc",
    "market_end_time_utc",
    "target_date_local",
    "city_text_from_title",
    "parsed_station_name",
    "parsed_station_code",
    "parsed_airport_name",
    "parsed_country",
    "temperature_unit",
    "outcome_buckets",
    "final_outcome",
    "is_resolved",
    "is_active",
    "parse_confidence",
    "needs_manual_review",
]


def load_settings(path: str | Path = "config/settings.yaml") -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def request_json(
    url: str,
    params: dict[str, Any] | None = None,
    retries: int = 3,
    timeout: int = 30,
) -> Any:
    headers = {"User-Agent": "weather-research/0.1"}
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=timeout, headers=headers)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001 - retry wrapper should keep original failure context.
            last_error = exc
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url} with params={params}: {last_error}") from last_error


def save_raw_json(raw_dir: str | Path, prefix: str, payload: Any, params: dict[str, Any] | None = None) -> Path:
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(json.dumps(params or {}, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = raw_path / f"{prefix}_{timestamp}_{digest}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return path


def discover_weather_events(
    settings: dict[str, Any],
    raw_dir: str | Path,
    markets_lookback_days: int = 90,
    include_active_markets: bool = False,
    include_resolved_markets: bool = True,
    start_date: str | None = None,
    end_date: str | None = None,
    force_refresh: bool = False,
) -> list[dict[str, Any]]:
    base = settings["polymarket"]["gamma_base_url"].rstrip("/")
    events_by_id: dict[str, dict[str, Any]] = {}
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    closed_options: list[bool] = []
    if include_resolved_markets:
        closed_options.append(True)
    if include_active_markets:
        closed_options.append(False)
    if not closed_options:
        closed_options = [True]

    for closed in closed_options:
        events = _fetch_events_by_weather_tag(
            base=base,
            settings=settings,
            raw_dir=raw_dir,
            closed=closed,
            markets_lookback_days=markets_lookback_days,
            force_refresh=force_refresh,
        )
        for event in events:
            if _event_in_requested_window(event, start_date, end_date) and event_is_daily_high_temperature(event):
                events_by_id[str(event.get("id"))] = event

    for term in settings["polymarket"].get("search_terms", []):
        payload = _fetch_public_search(base, raw_dir, term, force_refresh=force_refresh)
        for event in payload.get("events", []) if isinstance(payload, dict) else []:
            if _event_in_requested_window(event, start_date, end_date) and event_is_daily_high_temperature(event):
                events_by_id[str(event.get("id"))] = event

    logging.info("Discovered %s daily high-temperature Polymarket events", len(events_by_id))
    return list(events_by_id.values())


def _fetch_events_by_weather_tag(
    base: str,
    settings: dict[str, Any],
    raw_dir: Path,
    closed: bool,
    markets_lookback_days: int,
    force_refresh: bool,
) -> list[dict[str, Any]]:
    limit = int(settings["polymarket"].get("page_limit", 100))
    tag_id = settings["polymarket"].get("weather_tag_id", 84)
    cutoff = datetime.now(UTC) - timedelta(days=markets_lookback_days)
    events: list[dict[str, Any]] = []
    offset = 0
    pbar = tqdm(desc=f"Polymarket weather tag closed={closed}", unit="page", leave=False)
    while True:
        params = {
            "tag_id": tag_id,
            "related_tags": "true",
            "closed": str(closed).lower(),
            "limit": limit,
            "offset": offset,
            "order": "end_date",
            "ascending": "false",
        }
        if not closed:
            params["active"] = "true"
        cache = _cache_path(raw_dir, "events", params)
        if cache.exists() and not force_refresh:
            payload = json.loads(cache.read_text(encoding="utf-8"))
        else:
            payload = request_json(f"{base}/events", params=params)
            cache.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        page = payload.get("events", payload) if isinstance(payload, dict) else payload
        if not page:
            break
        events.extend(page)
        pbar.update(1)
        oldest = min((_parse_dt(e.get("endDate")) for e in page if _parse_dt(e.get("endDate"))), default=None)
        if oldest and oldest < cutoff:
            break
        has_more = payload.get("pagination", {}).get("hasMore") if isinstance(payload, dict) else len(page) == limit
        if not has_more:
            break
        offset += limit
    pbar.close()
    return events


def _fetch_public_search(base: str, raw_dir: Path, term: str, force_refresh: bool) -> dict[str, Any]:
    params = {"q": term, "limit": 100}
    cache = _cache_path(raw_dir, "public_search", params)
    if cache.exists() and not force_refresh:
        return json.loads(cache.read_text(encoding="utf-8"))
    payload = request_json(f"{base}/public-search", params=params)
    cache.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def _cache_path(raw_dir: Path, prefix: str, params: dict[str, Any]) -> Path:
    digest = hashlib.sha1(json.dumps(params, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]
    return raw_dir / f"{prefix}_{digest}.json"


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _event_in_requested_window(event: dict[str, Any], start_date: str | None, end_date: str | None) -> bool:
    end = _parse_dt(event.get("endDate"))
    if not end:
        return True
    if start_date and end.date() < datetime.fromisoformat(start_date).date():
        return False
    if end_date and end.date() > datetime.fromisoformat(end_date).date() + timedelta(days=2):
        return False
    return True


def normalize_markets(events: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event in events:
        markets = event.get("markets") or [event]
        for market in markets:
            rows.append(normalize_event_market(event, market))
    frame = pd.DataFrame(rows)
    for column in MARKET_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    return frame[MARKET_COLUMNS]


def write_polymarket_markets(events: list[dict[str, Any]], processed_dir: str | Path) -> pd.DataFrame:
    processed = Path(processed_dir)
    processed.mkdir(parents=True, exist_ok=True)
    frame = normalize_markets(events)
    frame.to_csv(processed / "polymarket_markets.csv", index=False)
    return frame
