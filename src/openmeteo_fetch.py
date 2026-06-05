from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from .actuals_fetch import local_day_window


FORECAST_COLUMNS = [
    "station_code",
    "station_name",
    "airport_name",
    "provider",
    "model",
    "issue_time_utc",
    "issue_time_local",
    "target_date_local",
    "forecast_horizon_hours",
    "forecast_high_f",
    "forecast_low_f",
    "forecast_hourly_max_f",
    "cloud_cover_mean",
    "cloud_cover_max",
    "precip_amount",
    "wind_speed_mean",
    "wind_speed_max",
    "wind_direction_mean",
    "dewpoint_mean_f",
    "humidity_mean",
    "source_file_or_url",
]


OPENMETEO_HOURLY = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "cloud_cover",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
]


def target_cutoff_utc(target_date_local: str, timezone: str, horizon_hours: int, lag_minutes: int) -> tuple[datetime, datetime]:
    tz = ZoneInfo(timezone)
    day_start = datetime.fromisoformat(target_date_local).replace(tzinfo=tz)
    if horizon_hours == 0:
        cutoff_local = day_start.replace(hour=6)
    else:
        cutoff_local = day_start - timedelta(hours=horizon_hours)
    available_utc = cutoff_local.astimezone(UTC) - timedelta(minutes=lag_minutes)
    return cutoff_local, available_utc.replace(minute=0, second=0, microsecond=0)


def fetch_openmeteo_snapshots(
    station_map: pd.DataFrame,
    settings: dict[str, Any],
    raw_dir: str | Path,
    horizons: list[int],
    force_refresh: bool = False,
) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    if station_map.empty:
        return pd.DataFrame(columns=FORECAST_COLUMNS)
    lag = int(settings.get("providers", {}).get("publication_lag_minutes", {}).get("openmeteo", 60))
    eligible = station_map.dropna(subset=["station_code", "target_date_local"]).copy()
    eligible = eligible.loc[eligible["target_date_local"].astype(str).str.len() > 0]
    for _, station in eligible.drop_duplicates(subset=["station_code", "target_date_local"]).iterrows():
        if bool(station.get("needs_manual_review")) or pd.isna(station.get("lat")) or pd.isna(station.get("lon")):
            continue
        for horizon in horizons:
            rows.append(_openmeteo_snapshot_row(station, settings, raw_dir, int(horizon), lag, force_refresh))
    frame = pd.DataFrame(rows)
    for column in FORECAST_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    return frame[FORECAST_COLUMNS]


def _openmeteo_snapshot_row(
    station: pd.Series,
    settings: dict[str, Any],
    raw_dir: Path,
    horizon: int,
    lag_minutes: int,
    force_refresh: bool,
) -> dict[str, Any]:
    timezone = station.get("timezone")
    target_date = str(station.get("target_date_local"))
    issue_local, issue_utc = target_cutoff_utc(target_date, timezone, horizon, lag_minutes)
    params = {
        "latitude": float(station["lat"]),
        "longitude": float(station["lon"]),
        "start_date": target_date,
        "end_date": target_date,
        "hourly": ",".join(OPENMETEO_HOURLY),
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": timezone,
    }
    cache = raw_dir / f"openmeteo_{station['station_code']}_{target_date}_{horizon}.json"
    source_url = settings.get("openmeteo", {}).get("historical_forecast_url", "https://historical-forecast-api.open-meteo.com/v1/forecast")
    try:
        if cache.exists() and not force_refresh:
            payload = json.loads(cache.read_text(encoding="utf-8"))
        else:
            response = requests.get(source_url, params=params, timeout=45, headers={"User-Agent": "weather-research/0.1"})
            response.raise_for_status()
            payload = response.json()
            cache.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        summary = summarize_openmeteo_payload(payload, target_date)
        quality_source = str(response.url) if "response" in locals() else str(cache)
    except Exception as exc:  # noqa: BLE001
        logging.warning("Open-Meteo unavailable for %s %s h%s: %s", station["station_code"], target_date, horizon, exc)
        summary = {}
        quality_source = f"unavailable: {exc}"
    return _base_forecast_row(station, "openmeteo", "historical_forecast", issue_utc, issue_local, target_date, horizon, summary, quality_source)


def summarize_openmeteo_payload(payload: dict[str, Any], target_date: str) -> dict[str, float]:
    hourly = payload.get("hourly", {})
    if not hourly:
        return {}
    frame = pd.DataFrame(hourly)
    if frame.empty or "time" not in frame:
        return {}
    frame["date"] = pd.to_datetime(frame["time"], errors="coerce").dt.date.astype(str)
    frame = frame.loc[frame["date"] == target_date]
    if frame.empty:
        return {}
    return {
        "forecast_high_f": _max(frame, "temperature_2m"),
        "forecast_low_f": _min(frame, "temperature_2m"),
        "forecast_hourly_max_f": _max(frame, "temperature_2m"),
        "cloud_cover_mean": _mean(frame, "cloud_cover"),
        "cloud_cover_max": _max(frame, "cloud_cover"),
        "precip_amount": _sum(frame, "precipitation"),
        "wind_speed_mean": _mean(frame, "wind_speed_10m"),
        "wind_speed_max": _max(frame, "wind_speed_10m"),
        "wind_direction_mean": _mean(frame, "wind_direction_10m"),
        "dewpoint_mean_f": _mean(frame, "dew_point_2m"),
        "humidity_mean": _mean(frame, "relative_humidity_2m"),
    }


def _base_forecast_row(
    station: pd.Series,
    provider: str,
    model: str,
    issue_utc: datetime,
    issue_local: datetime,
    target_date: str,
    horizon: int,
    summary: dict[str, Any],
    source: str,
) -> dict[str, Any]:
    return {
        "station_code": station.get("station_code"),
        "station_name": station.get("station_name"),
        "airport_name": station.get("airport_name"),
        "provider": provider,
        "model": model,
        "issue_time_utc": issue_utc.isoformat(),
        "issue_time_local": issue_local.isoformat(),
        "target_date_local": target_date,
        "forecast_horizon_hours": horizon,
        "forecast_high_f": summary.get("forecast_high_f"),
        "forecast_low_f": summary.get("forecast_low_f"),
        "forecast_hourly_max_f": summary.get("forecast_hourly_max_f"),
        "cloud_cover_mean": summary.get("cloud_cover_mean"),
        "cloud_cover_max": summary.get("cloud_cover_max"),
        "precip_amount": summary.get("precip_amount"),
        "wind_speed_mean": summary.get("wind_speed_mean"),
        "wind_speed_max": summary.get("wind_speed_max"),
        "wind_direction_mean": summary.get("wind_direction_mean"),
        "dewpoint_mean_f": summary.get("dewpoint_mean_f"),
        "humidity_mean": summary.get("humidity_mean"),
        "source_file_or_url": source,
    }


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _mean(frame: pd.DataFrame, column: str) -> float | None:
    series = _num(frame, column).dropna()
    return None if series.empty else float(series.mean())


def _max(frame: pd.DataFrame, column: str) -> float | None:
    series = _num(frame, column).dropna()
    return None if series.empty else float(series.max())


def _min(frame: pd.DataFrame, column: str) -> float | None:
    series = _num(frame, column).dropna()
    return None if series.empty else float(series.min())


def _sum(frame: pd.DataFrame, column: str) -> float | None:
    series = _num(frame, column).dropna()
    return None if series.empty else float(series.sum())


def write_openmeteo_snapshots(
    station_map: pd.DataFrame,
    settings: dict[str, Any],
    processed_dir: str | Path,
    raw_dir: str | Path,
    horizons: list[int],
    force_refresh: bool = False,
) -> pd.DataFrame:
    processed = Path(processed_dir)
    processed.mkdir(parents=True, exist_ok=True)
    frame = fetch_openmeteo_snapshots(station_map, settings, raw_dir, horizons, force_refresh=force_refresh)
    frame.to_csv(processed / "openmeteo_forecast_snapshots.csv", index=False)
    return frame
