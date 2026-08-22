from __future__ import annotations

import os
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import pandas as pd
import requests


SETTLEMENT_COLUMNS = [
    "station_id",
    "contract_date",
    "settlement_high_f",
    "settlement_high_c",
    "settlement_unit",
    "settlement_source",
    "source_url",
    "quality_flag",
    "raw_value",
    "notes",
    "fetched_at_utc",
]

WUNDERGROUND_HISTORY_PAGE = (
    "https://www.wunderground.com/history/daily/"
    "{country}/{city}/{station}/date/{day}"
)
_WUNDERGROUND_TEMP_CELL_RE = re.compile(
    r'<td[^>]*class=["\'][^"\']*\btemp\b[^"\']*["\'][^>]*>\s*(.*?)\s*</td>',
    re.IGNORECASE | re.DOTALL,
)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
# Keep this pattern ASCII-safe: Wunderground may render the degree separator as
# a literal symbol, an HTML entity, or a replacement glyph depending on the
# response encoding.
_TEMPERATURE_RE = re.compile(
    r"([-+]?\d+(?:\.\d+)?)\s*(?:[^A-Za-z0-9\s]+\s*)?([CF])",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class WeatherCompanyStationHistoryClient:
    api_key: str
    base_url: str = (
        "https://api.weather.com/v1/location/"
        "{station_id}:9:{country}/observations/historical.json"
    )
    timeout_seconds: int = 60

    def fetch_observations(
        self,
        station_id: str,
        start_date: str,
        end_date: str,
        *,
        country: str,
        units: str,
    ) -> list[dict[str, Any]]:
        url = self.base_url.format(
            station_id=station_id.upper(), country=str(country).upper()
        )
        response = requests.get(
            url,
            params={
                "apiKey": self.api_key,
                "units": units,
                "startDate": _parse_date(start_date).strftime("%Y%m%d"),
                "endDate": _parse_date(end_date).strftime("%Y%m%d"),
            },
            timeout=self.timeout_seconds,
        )
        if not response.ok:
            raise RuntimeError(
                f"Weather Company station history returned HTTP {response.status_code}"
            )
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
        raise ValueError("Could not discover the public Weather Company API key from Wunderground")
    return matches[0]


def backfill_wunderground_station_history(
    output_path: str | Path,
    *,
    stations: Iterable[str],
    station_timezones: dict[str, str],
    station_countries: dict[str, str] | None = None,
    station_units: dict[str, str] | None = None,
    station_slugs: dict[str, str] | None = None,
    start_date: str,
    end_date: str,
    api_key: str | None = None,
    api_key_env: str = "WEATHER_COMPANY_API_KEY",
    sleep_seconds: float = 0.0,
    force_refresh: bool = False,
    workers: int = 4,
) -> pd.DataFrame:
    wanted = sorted(
        {str(station).upper().strip() for station in stations if str(station).strip()}
    )
    missing_timezones = [station for station in wanted if not station_timezones.get(station)]
    if missing_timezones:
        raise ValueError(f"Missing station timezones for: {', '.join(missing_timezones)}")
    countries = {
        str(station).upper(): str(country).upper()
        for station, country in (station_countries or {}).items()
    }
    units_by_station = {
        str(station).upper(): str(units).lower()
        for station, units in (station_units or {}).items()
    }
    slugs = {
        str(station).upper(): str(slug).strip().lower()
        for station, slug in (station_slugs or {}).items()
    }
    key = api_key or os.environ.get(api_key_env) or discover_weather_company_public_api_key()
    client = WeatherCompanyStationHistoryClient(api_key=key)
    existing = _read_existing(output_path)
    requested_days = _date_range(start_date, end_date)
    requested = set(requested_days)
    if force_refresh and not existing.empty:
        existing = existing.loc[
            ~(
                existing["station_id"].isin(wanted)
                & existing["contract_date"].astype(str).isin(requested)
            )
        ].copy()
    completed = set()
    if not existing.empty:
        direct = existing.loc[
            existing["settlement_source"].eq("wunderground_station_history")
            & existing["settlement_high_f"].notna()
            & existing["quality_flag"].eq("ok")
        ]
        completed = set(zip(direct["station_id"], direct["contract_date"], strict=False))

    rows: list[dict[str, Any]] = []
    for station in wanted:
        timezone = station_timezones[station]
        country = countries.get(station, "US")
        units = units_by_station.get(station, "e")
        if units not in {"e", "m"}:
            raise ValueError(f"Unsupported Weather Company units for {station}: {units!r}")
        for chunk_start, chunk_end in _month_chunks(start_date, end_date):
            missing = [
                day
                for day in requested_days
                if chunk_start <= day <= chunk_end and (station, day) not in completed
            ]
            if not missing:
                continue
            fetched_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
            try:
                observations = client.fetch_observations(
                    station,
                    missing[0],
                    missing[-1],
                    country=country,
                    units=units,
                )
                daily = _station_history_daily_highs(observations, timezone, units=units)
                for day in missing:
                    summary = daily.get(day)
                    count = int(summary["observation_count"]) if summary else 0
                    quality = "ok" if summary and count >= 12 else "sparse_or_unavailable"
                    rows.append(
                        _settlement_row(
                            client,
                            station,
                            day,
                            country,
                            units,
                            timezone,
                            summary,
                            quality,
                            count,
                            fetched_at,
                        )
                    )
            except Exception as exc:  # noqa: BLE001 - preserve unavailable dates.
                # The Weather Company endpoint used by the public Wunderground
                # page can return 401 even when the page itself is readable.
                # Fall back to the exact Wunderground Daily Observations table,
                # preserving the Polymarket settlement source and native unit.
                page_rows = _fallback_wunderground_rows(
                    client,
                    station,
                    missing,
                    country=country,
                    city=slugs.get(station, ""),
                    units=units,
                    timezone=timezone,
                    fetched_at=fetched_at,
                    api_error=str(exc),
                    workers=workers,
                )
                rows.extend(page_rows)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    incoming = pd.DataFrame(rows, columns=SETTLEMENT_COLUMNS)
    merged = _merge(existing, incoming)
    _atomic_write_csv(merged, Path(output_path))
    return merged


def _settlement_row(
    client: WeatherCompanyStationHistoryClient,
    station: str,
    day: str,
    country: str,
    units: str,
    timezone: str,
    summary: dict[str, float | int] | None,
    quality: str,
    count: int,
    fetched_at: str,
    *,
    source_url: str | None = None,
    notes_extra: str | None = None,
) -> dict[str, Any]:
    return {
        "station_id": station,
        "contract_date": day,
        "settlement_high_f": summary["high_f"] if summary and quality == "ok" else pd.NA,
        "settlement_high_c": summary["high_c"] if summary and quality == "ok" else pd.NA,
        "settlement_unit": "C" if units == "m" else "F",
        "settlement_source": "wunderground_station_history",
        "source_url": source_url
        or _station_history_source_url(client, station, day, day, country, units),
        "quality_flag": quality,
        "raw_value": summary["high_native"] if summary else pd.NA,
        "notes": "; ".join(
            part
            for part in (
                f"observation_count={count}",
                f"timezone={timezone}",
                f"country={country}",
                f"units={units}",
                notes_extra,
            )
            if part
        ),
        "fetched_at_utc": fetched_at,
    }


def _fallback_wunderground_rows(
    client: WeatherCompanyStationHistoryClient,
    station: str,
    days: list[str],
    *,
    country: str,
    city: str,
    units: str,
    timezone: str,
    fetched_at: str,
    api_error: str,
    workers: int,
) -> list[dict[str, Any]]:
    def fetch(day: str) -> dict[str, Any]:
        try:
            summary, source_url = _fetch_wunderground_page_daily_high(
                station,
                day,
                country=country,
                city=city,
                units=units,
            )
            count = int(summary["observation_count"])
            quality = "ok" if count >= 12 else "sparse_or_unavailable"
            return _settlement_row(
                client,
                station,
                day,
                country,
                units,
                timezone,
                summary if quality == "ok" else None,
                quality,
                count,
                fetched_at,
                source_url=source_url,
                notes_extra="source=html_daily_observations; api_fallback=" + api_error,
            )
        except Exception as page_error:  # noqa: BLE001 - preserve unavailable dates.
            row = _settlement_row(
                client,
                station,
                day,
                country,
                units,
                timezone,
                None,
                "unavailable",
                0,
                fetched_at,
                notes_extra=f"api_error={api_error}; page_error={page_error}",
            )
            return row

    if not city:
        return [fetch(day) for day in days]
    with ThreadPoolExecutor(max_workers=max(1, min(int(workers), len(days)))) as executor:
        return list(executor.map(fetch, days))


def _fetch_wunderground_page_daily_high(
    station: str,
    day: str,
    *,
    country: str,
    city: str,
    units: str,
) -> tuple[dict[str, float | int], str]:
    if not city:
        raise ValueError(f"No Wunderground city slug configured for {station}")
    source_url = WUNDERGROUND_HISTORY_PAGE.format(
        country=country.lower(), city=city.lower(), station=station.upper(), day=day
    )
    response = requests.get(
        source_url,
        headers={"User-Agent": "Mozilla/5.0 weather-research settlement backfill"},
        timeout=60,
    )
    if not response.ok:
        raise RuntimeError(f"Wunderground history page returned HTTP {response.status_code}")
    expected_unit = "C" if units == "m" else "F"
    values: list[float] = []
    for cell in _WUNDERGROUND_TEMP_CELL_RE.findall(response.text):
        text = re.sub(r"\s+", " ", _HTML_TAG_RE.sub(" ", cell)).strip()
        match = _TEMPERATURE_RE.search(text)
        if match and match.group(2).upper() == expected_unit:
            values.append(float(match.group(1)))
    if not values:
        raise ValueError(f"No {expected_unit} temperature rows found for {station} {day}")
    high_native = max(values)
    high_c = high_native if units == "m" else (high_native - 32) * 5 / 9
    high_f = high_native * 9 / 5 + 32 if units == "m" else high_native
    return (
        {
            "high_native": high_native,
            "high_c": high_c,
            "high_f": high_f,
            "observation_count": len(values),
        },
        source_url,
    )


def _station_history_daily_highs(
    observations: Iterable[dict[str, Any]], timezone: str, *, units: str
) -> dict[str, dict[str, float | int]]:
    tz = ZoneInfo(timezone)
    temperatures: dict[str, list[float]] = {}
    for observation in observations:
        timestamp = pd.to_numeric(observation.get("valid_time_gmt"), errors="coerce")
        temperature = pd.to_numeric(observation.get("temp"), errors="coerce")
        if pd.isna(timestamp) or pd.isna(temperature):
            continue
        day = datetime.fromtimestamp(float(timestamp), tz=UTC).astimezone(tz).date().isoformat()
        temperatures.setdefault(day, []).append(float(temperature))
    output: dict[str, dict[str, float | int]] = {}
    for day, values in temperatures.items():
        high_native = max(values)
        high_c = high_native if units == "m" else (high_native - 32) * 5 / 9
        high_f = high_native * 9 / 5 + 32 if units == "m" else high_native
        output[day] = {
            "high_native": high_native,
            "high_f": high_f,
            "high_c": high_c,
            "observation_count": len(values),
        }
    return output


def _station_history_source_url(
    client: WeatherCompanyStationHistoryClient,
    station: str,
    start: str,
    end: str,
    country: str,
    units: str,
) -> str:
    base = client.base_url.format(station_id=station.upper(), country=country.upper())
    return (
        f"{base}?units={units}&startDate={_parse_date(start):%Y%m%d}"
        f"&endDate={_parse_date(end):%Y%m%d}"
    )


def _read_existing(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        return pd.DataFrame(columns=SETTLEMENT_COLUMNS)
    return _coerce(pd.read_csv(source))


def _coerce(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in SETTLEMENT_COLUMNS:
        if column not in output:
            output[column] = pd.NA
    output["station_id"] = output["station_id"].astype(str).str.upper().str.strip()
    output["contract_date"] = pd.to_datetime(
        output["contract_date"], errors="coerce"
    ).dt.date.astype("string")
    output["settlement_high_f"] = pd.to_numeric(
        output["settlement_high_f"], errors="coerce"
    )
    output["settlement_high_c"] = pd.to_numeric(
        output["settlement_high_c"], errors="coerce"
    )
    return output[SETTLEMENT_COLUMNS]


def _merge(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    frames = [_coerce(frame) for frame in (existing, incoming) if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=SETTLEMENT_COLUMNS)
    combined = pd.concat(frames, ignore_index=True)
    combined["complete"] = (
        combined["settlement_high_f"].notna() & combined["quality_flag"].eq("ok")
    ).astype(int)
    combined = combined.sort_values(
        ["station_id", "contract_date", "complete", "fetched_at_utc"]
    ).drop_duplicates(["station_id", "contract_date"], keep="last")
    return combined.drop(columns="complete")[SETTLEMENT_COLUMNS].reset_index(drop=True)


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        frame.to_csv(temporary, index=False)
        temporary.flush()
        os.fsync(temporary.fileno())
    os.replace(temporary_path, path)


def _parse_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _date_range(start: str, end: str) -> list[str]:
    first = _parse_date(start)
    last = _parse_date(end)
    if last < first:
        raise ValueError("end_date must be on or after start_date")
    return [
        (first + timedelta(days=offset)).isoformat()
        for offset in range((last - first).days + 1)
    ]


def _month_chunks(start: str, end: str) -> list[tuple[str, str]]:
    first = _parse_date(start)
    last = _parse_date(end)
    chunks: list[tuple[str, str]] = []
    cursor = first
    while cursor <= last:
        next_month = (cursor.replace(day=28) + timedelta(days=4)).replace(day=1)
        chunk_end = min(last, next_month - timedelta(days=1))
        chunks.append((cursor.isoformat(), chunk_end.isoformat()))
        cursor = chunk_end + timedelta(days=1)
    return chunks
