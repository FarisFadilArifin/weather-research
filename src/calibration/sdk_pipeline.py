from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .time_rules import forecast_as_of_utc, forecast_hours_for_local_day, local_day_utc_bounds


SDK_CACHE_DIRNAME = "sdk"
SDK_STATION_REGISTRY_FILE = "sdk_station_registry.csv"
SDK_ACTUALS_FILE = "sdk_actual_highs.csv"
SDK_NWP_FILE = "sdk_nwp_0h_cache.csv"
DIRECT_NBM_FILE = "direct_nbm_0h_cache.csv"
SDK_AVAILABILITY_FILE = "sdk_archive_availability.csv"
SDK_COVERAGE_SUMMARY_FILE = "sdk_coverage_summary.csv"
SDK_MISSING_COVERAGE_FILE = "sdk_missing_coverage.csv"
WEATHER_FEATURE_FLAG = "weather_features_included"
DIRECT_NBM_WEATHER_FEATURE_FLAG = WEATHER_FEATURE_FLAG

TARGET_STATIONS = [
    "KATL",
    "KAUS",
    "KORD",
    "KDAL",
    "KHOU",
    "KLAX",
    "KMIA",
    "KLGA",
    "KSEA",
]

STATION_METADATA: dict[str, dict[str, Any]] = {
    "KATL": {
        "station_name": "Atlanta/Hartsfield-Jackson Intl",
        "airport_name": "Atlanta/Hartsfield-Jackson Intl",
        "city_label": "Atlanta",
        "lat": 33.62972,
        "lon": -84.44223,
        "timezone": "America/New_York",
        "country": "US",
    },
    "KAUS": {
        "station_name": "Austin/Bergstrom Intl",
        "airport_name": "Austin/Bergstrom Intl",
        "city_label": "Austin",
        "lat": 30.18310,
        "lon": -97.68063,
        "timezone": "America/Chicago",
        "country": "US",
    },
    "KORD": {
        "station_name": "Chicago/O'Hare Intl",
        "airport_name": "Chicago/O'Hare Intl",
        "city_label": "Chicago",
        "lat": 41.96017,
        "lon": -87.93161,
        "timezone": "America/Chicago",
        "country": "US",
    },
    "KDAL": {
        "station_name": "Dallas/Love Fld",
        "airport_name": "Dallas/Love Fld",
        "city_label": "Dallas",
        "lat": 32.83836,
        "lon": -96.83584,
        "timezone": "America/Chicago",
        "country": "US",
    },
    "KHOU": {
        "station_name": "Houston/Hobby Arpt",
        "airport_name": "Houston/Hobby Arpt",
        "city_label": "Houston",
        "lat": 29.64582,
        "lon": -95.28214,
        "timezone": "America/Chicago",
        "country": "US",
    },
    "KLAX": {
        "station_name": "Los Angeles Intl",
        "airport_name": "Los Angeles Intl",
        "city_label": "Los Angeles",
        "lat": 33.93817,
        "lon": -118.38660,
        "timezone": "America/Los_Angeles",
        "country": "US",
    },
    "KMIA": {
        "station_name": "Miami Intl",
        "airport_name": "Miami Intl",
        "city_label": "Miami",
        "lat": 25.78806,
        "lon": -80.31692,
        "timezone": "America/New_York",
        "country": "US",
    },
    "KLGA": {
        "station_name": "New York/La Guardia Arpt",
        "airport_name": "New York/La Guardia Arpt",
        "city_label": "NYC",
        "lat": 40.77945,
        "lon": -73.88027,
        "timezone": "America/New_York",
        "country": "US",
    },
    "KSEA": {
        "station_name": "Seattle-Tacoma Intl",
        "airport_name": "Seattle-Tacoma Intl",
        "city_label": "Seattle",
        "lat": 47.44467,
        "lon": -122.31442,
        "timezone": "America/Los_Angeles",
        "country": "US",
    },
}

NWP_MODELS = ["hrrr", "gfs", "nbm"]
NWP_ARCHIVE_FALLBACK_START = {
    "hrrr": date(2014, 7, 30),
    "gfs": date(2021, 1, 1),
    "nbm": date(2020, 1, 1),
}
MODEL_LAG_MINUTES = {"hrrr": 75, "gfs": 240, "nbm": 120}
MODEL_MAX_FXX = {"hrrr": 48, "gfs": 120, "nbm": 72}
MODEL_SEARCH_HOURS = {"hrrr": 18, "gfs": 36, "nbm": 36}
GFS_CYCLES = {0, 6, 12, 18}
HRRR_LONG_CYCLES = {0, 6, 12, 18}
MODEL_CYCLE_HOURS_FALLBACK = {
    "hrrr": tuple(range(24)),
    "gfs": (0, 6, 12, 18),
    "nbm": tuple(range(24)),
}
TIMING_MODE_STRICT_6AM = "strict_6am"
TIMING_MODE_FRESH_AFTER_6AM = "fresh_after_6am"
TIMING_MODE_SAME_DAY_11AM = "same_day_11am"
TIMING_MODES = [TIMING_MODE_STRICT_6AM, TIMING_MODE_FRESH_AFTER_6AM, TIMING_MODE_SAME_DAY_11AM]
TIMING_MODE_SOURCE_LABELS = {
    TIMING_MODE_STRICT_6AM: "strict_6am",
    TIMING_MODE_FRESH_AFTER_6AM: "fresh_after_6am_remaining_day",
    TIMING_MODE_SAME_DAY_11AM: "same_day_11am_remaining_day",
}
NBM_SAME_DAY_11AM_RECENT_DAYS = 2
NWP_FXX_FETCH_RETRIES = 2
NWP_FXX_RETRY_SLEEP_SECONDS = 5.0


@dataclass(frozen=True)
class NwpRequest:
    station_id: str
    station_name: str
    airport_name: str
    timezone: str
    contract_date: str
    model: str
    forecast_as_of: datetime
    cycle: datetime
    fxx_hours: tuple[int, ...]
    timing_mode: str
    cycle_selection_policy: str
    forecast_window_start: datetime
    forecast_window_end: datetime


def default_sdk_cache_dir(calibration_dir: str | Path = "data/calibration") -> Path:
    return Path(calibration_dir) / SDK_CACHE_DIRNAME


def resolve_contract_end(value: str | None) -> date:
    if value is None or value.lower() in {"latest", "latest-complete", "latest_complete"}:
        return datetime.now(UTC).date() - timedelta(days=1)
    return date.fromisoformat(value[:10])


def latest_complete_contract_date() -> date:
    return datetime.now(UTC).date() - timedelta(days=1)


def nbm_same_day_11am_recent_floor(today_utc: date | None = None) -> date:
    today = today_utc or datetime.now(UTC).date()
    return today - timedelta(days=NBM_SAME_DAY_11AM_RECENT_DAYS)


def nbm_same_day_11am_supported_date(contract_day: date, today_utc: date | None = None) -> bool:
    today = today_utc or datetime.now(UTC).date()
    latest_complete = today - timedelta(days=1)
    return nbm_same_day_11am_recent_floor(today) <= contract_day <= latest_complete


def date_range(start_date: str | date, end_date: str | date) -> list[str]:
    start = date.fromisoformat(str(start_date)[:10]) if not isinstance(start_date, date) else start_date
    end = date.fromisoformat(str(end_date)[:10]) if not isinstance(end_date, date) else end_date
    if end < start:
        return []
    return [(start + timedelta(days=i)).isoformat() for i in range((end - start).days + 1)]


def local_datetime_utc(contract_date: str | date, timezone: str, hour: int) -> datetime:
    day = date.fromisoformat(str(contract_date)[:10]) if not isinstance(contract_date, date) else contract_date
    local = datetime.combine(day, datetime.min.time(), tzinfo=ZoneInfo(timezone)).replace(hour=hour)
    return local.astimezone(UTC)


def forecast_as_of_for_timing(contract_date: str | date, timezone: str, timing_mode: str) -> datetime:
    if timing_mode == TIMING_MODE_FRESH_AFTER_6AM:
        return local_datetime_utc(contract_date, timezone, 7)
    if timing_mode == TIMING_MODE_SAME_DAY_11AM:
        return local_datetime_utc(contract_date, timezone, 11)
    if timing_mode == TIMING_MODE_STRICT_6AM:
        return forecast_as_of_utc(contract_date, timezone)
    raise ValueError(f"Unsupported timing_mode: {timing_mode}")


def forecast_window_for_timing(
    contract_date: str | date,
    timezone: str,
    timing_mode: str,
    forecast_as_of: datetime,
) -> tuple[datetime, datetime]:
    start_utc, end_utc = local_day_utc_bounds(contract_date, timezone)
    if timing_mode in {TIMING_MODE_FRESH_AFTER_6AM, TIMING_MODE_SAME_DAY_11AM}:
        return forecast_as_of.astimezone(UTC), end_utc
    if timing_mode == TIMING_MODE_STRICT_6AM:
        return start_utc, end_utc
    raise ValueError(f"Unsupported timing_mode: {timing_mode}")


def forecast_hours_for_utc_window(cycle_utc: datetime, start_utc: datetime, end_utc: datetime) -> list[int]:
    if cycle_utc.tzinfo is None or cycle_utc.tzinfo.utcoffset(cycle_utc) is None:
        raise ValueError("cycle_utc must be timezone-aware")
    cycle_utc = cycle_utc.astimezone(UTC)
    start_utc = start_utc.astimezone(UTC)
    end_utc = end_utc.astimezone(UTC)
    hours: list[int] = []
    valid = start_utc
    while valid < end_utc:
        fxx = int((valid - cycle_utc).total_seconds() // 3600)
        if fxx >= 0:
            hours.append(fxx)
        valid += timedelta(hours=1)
    return hours


def model_cycle_hours(model: str) -> tuple[int, ...]:
    try:
        from mostlyright.weather._fetchers._nwp_cycle_chunks import CYCLE_HOURS

        hours = CYCLE_HOURS.get(model)
        if hours:
            return tuple(int(hour) for hour in hours)
    except Exception:
        pass
    return MODEL_CYCLE_HOURS_FALLBACK.get(model, tuple(range(24)))


def choose_fresh_after_6am_cycle(
    model: str,
    contract_date: str,
    timezone: str,
) -> tuple[datetime | None, tuple[int, ...], datetime, datetime, datetime]:
    model = model.lower()
    earliest_cycle = local_datetime_utc(contract_date, timezone, 6)
    seven_am = local_datetime_utc(contract_date, timezone, 7)
    _, local_midnight = local_day_utc_bounds(contract_date, timezone)
    allowed_hours = set(model_cycle_hours(model))
    candidate = earliest_cycle.replace(minute=0, second=0, microsecond=0)
    if candidate < earliest_cycle:
        candidate += timedelta(hours=1)
    cycle: datetime | None = None
    while candidate < local_midnight:
        if candidate.hour in allowed_hours:
            cycle = candidate
            break
        candidate += timedelta(hours=1)
    forecast_as_of = max(seven_am, cycle) if cycle is not None else seven_am
    if cycle is None:
        return None, (), forecast_as_of, forecast_as_of, local_midnight
    max_fxx = _fresh_model_max_fxx(model, cycle)
    fxx_hours = tuple(
        f for f in forecast_hours_for_utc_window(cycle, forecast_as_of, local_midnight) if 0 <= f <= max_fxx
    )
    return cycle, fxx_hours, forecast_as_of, forecast_as_of, local_midnight


def chunk_dates(dates: list[str], chunk_days: int) -> Iterable[tuple[str, str]]:
    chunk_days = max(1, int(chunk_days))
    for start in range(0, len(dates), chunk_days):
        chunk = dates[start : start + chunk_days]
        if chunk:
            yield chunk[0], chunk[-1]


def station_registry_frame(stations: Iterable[str] | None = None) -> pd.DataFrame:
    wanted = [s.upper() for s in stations] if stations else TARGET_STATIONS
    rows: list[dict[str, Any]] = []
    for station_id in wanted:
        if station_id not in STATION_METADATA:
            raise ValueError(f"Missing SDK station metadata for {station_id}")
        rows.append({"station_id": station_id, **STATION_METADATA[station_id]})
    return pd.DataFrame(rows)


def write_station_registry(sdk_cache_dir: str | Path, stations: Iterable[str] | None = None) -> pd.DataFrame:
    out_dir = Path(sdk_cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / SDK_STATION_REGISTRY_FILE
    requested = station_registry_frame(stations)
    frame = requested.copy()
    if path.exists():
        existing = pd.read_csv(path)
        if "station_id" in existing.columns:
            existing["station_id"] = existing["station_id"].astype(str).str.upper()
            frame = pd.concat([existing, frame], ignore_index=True)
            frame = frame.drop_duplicates(subset=["station_id"], keep="last")
            order = {station_id: idx for idx, station_id in enumerate(TARGET_STATIONS)}
            frame["_order"] = frame["station_id"].map(order).fillna(len(order))
            frame = frame.sort_values(["_order", "station_id"]).drop(columns=["_order"]).reset_index(drop=True)
    frame.to_csv(path, index=False)
    return requested


def nwp_archive_starts() -> dict[str, date]:
    try:
        from mostlyright.weather._fetchers._nwp_cycle_chunks import NWP_HISTORICAL_DEPTH
    except Exception:
        return NWP_ARCHIVE_FALLBACK_START.copy()
    starts: dict[str, date] = {}
    for model in NWP_MODELS:
        value = NWP_HISTORICAL_DEPTH.get(model)
        if isinstance(value, datetime):
            starts[model] = value.date()
        elif isinstance(value, date):
            starts[model] = value
        else:
            starts[model] = NWP_ARCHIVE_FALLBACK_START[model]
    return starts


def write_sdk_availability(
    sdk_cache_dir: str | Path,
    start_date: str = "2016-01-01",
    end_date: str | None = None,
    models: Iterable[str] | None = None,
) -> pd.DataFrame:
    out_dir = Path(sdk_cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    requested_start = date.fromisoformat(start_date[:10])
    requested_end = resolve_contract_end(end_date)
    archive = nwp_archive_starts()
    rows: list[dict[str, Any]] = []
    for model in [m.lower() for m in (models or NWP_MODELS)]:
        archive_start = archive[model]
        effective_start = max(requested_start, archive_start)
        effective_end = requested_end
        notes = "SDK NWP historical depth"
        full_range_supported = archive_start <= requested_start
        if model == "nbm":
            effective_start = max(requested_start, archive_start)
            notes = "SDK NBM historical depth; uses forecast_nwp with AWS BDP/NOMADS mirror fallback."
            full_range_supported = effective_start <= requested_start
        rows.append(
            {
                "provider": model,
                "model": model,
                "sdk_api": 'mostlyright.weather.forecast_nwp(model="%s")' % model,
                "archive_start": archive_start.isoformat(),
                "requested_start": requested_start.isoformat(),
                "requested_end": requested_end.isoformat(),
                "available_start_for_request": effective_start.isoformat(),
                "available_end_for_request": effective_end.isoformat(),
                "requested_full_range_supported": full_range_supported,
                "notes": notes,
            }
        )
    rows.append(
        {
            "provider": "actuals",
            "model": "observed_high",
            "sdk_api": "mostlyright.weather.obs.obs",
            "archive_start": pd.NA,
            "requested_start": requested_start.isoformat(),
            "requested_end": requested_end.isoformat(),
            "available_start_for_request": pd.NA,
            "available_end_for_request": pd.NA,
            "requested_full_range_supported": pd.NA,
            "notes": "Observation coverage is verified from SDK obs cache after backfill",
        }
    )
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / SDK_AVAILABILITY_FILE, index=False)
    return frame


def inspect_sdk(
    sdk_cache_dir: str | Path,
    start_date: str = "2016-01-01",
    end_date: str | None = None,
    stations: Iterable[str] | None = None,
    models: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    registry = write_station_registry(sdk_cache_dir, stations)
    availability = write_sdk_availability(sdk_cache_dir, start_date=start_date, end_date=end_date, models=models)
    return registry, availability


def backfill_sdk_actuals(
    sdk_cache_dir: str | Path,
    stations: Iterable[str] | None = None,
    start_date: str = "2016-01-01",
    end_date: str | None = None,
    chunk_days: int = 366,
    strategy: str = "exact_window",
    source: str | None = None,
    force: bool = False,
    max_chunks: int | None = None,
) -> pd.DataFrame:
    out_dir = Path(sdk_cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    station_meta = write_station_registry(out_dir, stations)
    _patch_mostlyright_station_runtime(station_meta)
    obs_fn = _load_obs()
    cache_path = out_dir / SDK_ACTUALS_FILE
    existing = _load_existing(cache_path)
    completed = _completed_actual_keys(existing) if not force else set()
    dates = date_range(start_date, resolve_contract_end(end_date))
    fresh_chunks = 0
    for station in station_meta.itertuples(index=False):
        station_id = str(station.station_id).upper()
        for chunk_start, chunk_end in chunk_dates(dates, chunk_days):
            chunk = date_range(chunk_start, chunk_end)
            if all((station_id, d) in completed for d in chunk):
                continue
            try:
                raw = obs_fn(
                    station_id,
                    chunk_start,
                    chunk_end,
                    source=source,
                    strategy=strategy,
                    as_dataframe=True,
                )
                rows = _normalize_obs_frame(raw, station, chunk_start, chunk_end)
            except Exception as exc:  # noqa: BLE001
                logging.warning("Mostly Right obs unavailable for %s %s..%s: %s", station_id, chunk_start, chunk_end, exc)
                rows = [_unavailable_actual_row(station, d, str(exc)) for d in chunk]
            existing = _append_cache(cache_path, existing, rows, ["station_id", "contract_date"])
            fresh_chunks += 1
            if max_chunks is not None and fresh_chunks >= max_chunks:
                return existing
    requested_ok = _requested_ok_actuals(existing, station_meta["station_id"], dates)
    if requested_ok == 0:
        raise RuntimeError(
            "Mostly Right SDK obs did not return any usable actual highs for the requested station/date range. "
            "No non-SDK fallback was used."
        )
    return existing


def backfill_sdk_nwp(
    sdk_cache_dir: str | Path,
    models: Iterable[str] | None = None,
    stations: Iterable[str] | None = None,
    start_date: str = "2016-01-01",
    end_date: str | None = None,
    force: bool = False,
    max_requests: int | None = None,
    max_batches: int | None = None,
    timing_mode: str = TIMING_MODE_STRICT_6AM,
    batch_stations: bool = True,
    fxx_workers: int = 1,
    include_weather_features: bool = False,
) -> pd.DataFrame:
    out_dir = Path(sdk_cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timing_mode = timing_mode.lower()
    if timing_mode not in TIMING_MODES:
        raise ValueError(f"timing_mode must be one of {TIMING_MODES}; got {timing_mode!r}")
    models = [m.lower() for m in (models or NWP_MODELS)]
    unknown = set(models) - set(NWP_MODELS)
    if unknown:
        raise ValueError(f"Unsupported SDK NWP models: {sorted(unknown)}")
    if fxx_workers < 1:
        raise ValueError("fxx_workers must be >= 1")
    station_meta = write_station_registry(out_dir, stations)
    write_sdk_availability(out_dir, start_date=start_date, end_date=end_date, models=models)
    cache_path = out_dir / SDK_NWP_FILE
    existing = _load_existing(cache_path)
    completed = _completed_nwp_keys(existing, require_weather_features=include_weather_features) if not force else set()
    dates = date_range(start_date, resolve_contract_end(end_date))
    requests = plan_nwp_requests(station_meta, dates, models, completed, timing_mode=timing_mode)
    if not requests:
        return existing
    forecast_nwp = _load_forecast_nwp(station_meta)
    client = _nwp_http_client()
    processed = 0
    batch_limit = max_batches if max_batches is not None else max_requests
    try:
        if batch_stations:
            for batch in _group_nwp_requests(requests):
                rows = _fetch_and_summarize_nwp_batch(
                    forecast_nwp,
                    batch,
                    client=client,
                    fxx_workers=fxx_workers,
                )
                existing = _append_cache(
                    cache_path,
                    existing,
                    rows,
                    ["station_id", "contract_date", "provider", "model", "timing_mode"],
                )
                processed += 1
                if batch_limit is not None and processed >= batch_limit:
                    break
        else:
            row_limit = max_requests if max_requests is not None else max_batches
            for request in requests:
                try:
                    hourly = _fetch_request_hourly(forecast_nwp, request, client=client)
                    row = _summarize_nwp_request(request, hourly)
                except Exception as exc:  # noqa: BLE001
                    logging.warning(
                        "Mostly Right NWP unavailable for %s %s %s: %s",
                        request.model,
                        request.station_id,
                        request.contract_date,
                        exc,
                    )
                    row = _unavailable_nwp_row(request, str(exc))
                existing = _append_cache(
                    cache_path,
                    existing,
                    [row],
                    ["station_id", "contract_date", "provider", "model", "timing_mode"],
                )
                processed += 1
                if row_limit is not None and processed >= row_limit:
                    break
    finally:
        if client is not None:
            client.close()
    return existing


def backfill_direct_nbm(
    cache_dir: str | Path,
    stations: Iterable[str] | None = None,
    start_date: str = "2021-01-01",
    end_date: str | None = None,
    force: bool = False,
    max_batches: int | None = None,
    timing_mode: str = TIMING_MODE_SAME_DAY_11AM,
    include_weather_features: bool = False,
) -> pd.DataFrame:
    if timing_mode != TIMING_MODE_SAME_DAY_11AM:
        raise ValueError("Direct NOAA NBM currently supports timing_mode='same_day_11am'")
    out_dir = Path(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    station_meta = write_station_registry(out_dir, stations)
    cache_path = out_dir / DIRECT_NBM_FILE
    raw_dir = out_dir / "raw_nbm"
    raw_dir.mkdir(parents=True, exist_ok=True)
    existing = _load_existing(cache_path)
    completed = (
        _completed_direct_nbm_keys(existing, require_weather_features=include_weather_features)
        if not force
        else set()
    )
    existing_ok_keys = _ok_nwp_keys(existing)
    dates = date_range(start_date, resolve_contract_end(end_date))
    requests = plan_direct_nbm_requests(station_meta, dates, completed, timing_mode=timing_mode)
    if not requests:
        return existing

    settings = {
        "nws": {
            "nbm_temperature_variable": "TMP",
            "nbm_max_forecast_hour": MODEL_MAX_FXX["nbm"],
            "nbm_max_cycle_search_hours": MODEL_SEARCH_HOURS["nbm"],
            "nbm_aws_base_url": "https://noaa-nbm-grib2-pds.s3.amazonaws.com",
            "nbm_product": "core",
            "nbm_domain_suffix": "co",
            "nbm_download_retries": 4,
            "nbm_retry_backoff_seconds": 5,
        }
    }
    processed = 0
    from ..nws_fetch import NBM_TMP, TransientNbmDownloadError, _extract_nbm_run_points
    from ..nws_fetch import _extract_nbm_run_feature_points

    for batch in _group_nwp_requests(requests):
        station_points = {
            request.station_id: {
                "lat": float(station_meta.loc[station_meta["station_id"] == request.station_id, "lat"].iloc[0]),
                "lon": float(station_meta.loc[station_meta["station_id"] == request.station_id, "lon"].iloc[0]),
            }
            for request in batch
        }
        fxx_hours = sorted({fxx for request in batch for fxx in request.fxx_hours})
        rows: list[dict[str, Any]] = []
        try:
            if include_weather_features:
                feature_values = _extract_nbm_run_feature_points(
                    station_points,
                    settings,
                    raw_dir,
                    batch[0].cycle,
                    fxx_hours,
                    force,
                )
                run_values = {
                    station_id: {
                        fxx: values.get("temp_k_2m")
                        for fxx, values in fxx_map.items()
                        if values.get("temp_k_2m") is not None
                    }
                    for station_id, fxx_map in feature_values.items()
                }
            else:
                feature_values = {}
                run_values = _extract_nbm_run_points(
                    station_points,
                    settings,
                    raw_dir,
                    batch[0].cycle,
                    fxx_hours,
                    force,
                    NBM_TMP,
                )
        except TransientNbmDownloadError:
            raise
        except Exception as exc:  # noqa: BLE001
            logging.warning("Direct NOAA NBM run unavailable for %s: %s", batch[0].cycle, exc)
            run_values = {}
            run_error = str(exc)
        else:
            run_error = ""

        for request in batch:
            station_values = run_values.get(request.station_id, {})
            temps_f = [station_values.get(fxx) for fxx in request.fxx_hours if station_values.get(fxx) is not None]
            if temps_f:
                feature_summary = (
                    _summarize_direct_nbm_feature_values(request, feature_values.get(request.station_id, {}))
                    if include_weather_features
                    else _summarize_direct_nbm_temperature_only(request, station_values)
                )
                rows.append(
                    _direct_nbm_row(
                        request,
                        float(max(temps_f)),
                        "ok",
                        "",
                        feature_summary,
                        weather_features_included=include_weather_features,
                    )
                )
            else:
                key = (request.station_id, request.contract_date, "nbm", "nbm", request.timing_mode)
                if include_weather_features and key in existing_ok_keys:
                    logging.warning(
                        "Leaving existing direct NOAA NBM row unchanged after feature extraction returned no TMP values for %s %s",
                        request.station_id,
                        request.contract_date,
                    )
                    continue
                rows.append(
                    _direct_nbm_row(
                        request,
                        pd.NA,
                        "unavailable",
                        run_error or "no NBM TMP values extracted for station/window",
                        weather_features_included=include_weather_features,
                    )
                )
        if rows:
            existing = _append_cache(
                cache_path,
                existing,
                rows,
                ["station_id", "contract_date", "provider", "model", "timing_mode"],
            )
        processed += 1
        if max_batches is not None and processed >= max_batches:
            break
    return existing


def plan_direct_nbm_requests(
    station_meta: pd.DataFrame,
    dates: list[str],
    completed: set[tuple[str, str, str, str, str]] | None = None,
    timing_mode: str = TIMING_MODE_SAME_DAY_11AM,
) -> list[NwpRequest]:
    completed = completed or set()
    requests: list[NwpRequest] = []
    for row in station_meta.itertuples(index=False):
        station_id = str(row.station_id).upper()
        timezone = str(row.timezone)
        for contract_date in dates:
            day = date.fromisoformat(contract_date)
            if day < date(2021, 1, 1):
                continue
            if (station_id, contract_date, "nbm", "nbm", timing_mode) in completed:
                continue
            as_of = forecast_as_of_for_timing(contract_date, timezone, timing_mode)
            window_start, window_end = forecast_window_for_timing(contract_date, timezone, timing_mode, as_of)
            cycle, fxx_hours = choose_latest_cycle_for_window("nbm", as_of, window_start, window_end, min_fxx=1)
            if cycle is None or not fxx_hours:
                continue
            requests.append(
                NwpRequest(
                    station_id=station_id,
                    station_name=str(row.station_name),
                    airport_name=str(row.airport_name),
                    timezone=timezone,
                    contract_date=contract_date,
                    model="nbm",
                    forecast_as_of=as_of,
                    cycle=cycle,
                    fxx_hours=fxx_hours,
                    timing_mode=timing_mode,
                    cycle_selection_policy="direct_noaa_latest_cycle_at_or_before_11am_local_no_safety_lag",
                    forecast_window_start=window_start,
                    forecast_window_end=window_end,
                )
            )
    return requests


def _direct_nbm_row(
    request: NwpRequest,
    raw_high_f: Any,
    fetch_status: str,
    reason: str,
    feature_summary: dict[str, Any] | None = None,
    weather_features_included: bool = False,
) -> dict[str, Any]:
    row = {
        "station_id": request.station_id,
        "station_name": request.station_name,
        "airport_name": request.airport_name,
        "provider": "nbm",
        "model": "nbm",
        "source_label": f"noaa_nbm_archive_tmp_{TIMING_MODE_SOURCE_LABELS[request.timing_mode]}",
        "timing_mode": request.timing_mode,
        "cycle_selection_policy": request.cycle_selection_policy,
        "contract_date": request.contract_date,
        "forecast_as_of": request.forecast_as_of.isoformat(),
        "issued_at": request.cycle.isoformat(),
        "forecast_window_start": request.forecast_window_start.isoformat(),
        "forecast_window_end": request.forecast_window_end.isoformat(),
        "horizon_hours": 0,
        "raw_forecast_high_f": raw_high_f,
        "forecast_hour_min": min(request.fxx_hours),
        "forecast_hour_max": max(request.fxx_hours),
        "data_source": "direct_noaa_nbm_archive_grib2",
        "source_file_or_url": "https://noaa-nbm-grib2-pds.s3.amazonaws.com",
        "fetch_status": fetch_status,
        "unavailable_reason": pd.NA if fetch_status == "ok" else reason,
        DIRECT_NBM_WEATHER_FEATURE_FLAG: bool(weather_features_included),
    }
    if feature_summary:
        row.update(feature_summary)
    return row


def _summarize_direct_nbm_temperature_only(request: NwpRequest, values_by_fxx: dict[int, float]) -> dict[str, Any]:
    first_fxx = min(request.fxx_hours)
    return {"forecast_temp_at_as_of_f": values_by_fxx.get(first_fxx, pd.NA)}


def _summarize_direct_nbm_feature_values(request: NwpRequest, values_by_fxx: dict[int, dict[str, float]]) -> dict[str, Any]:
    ordered = [(fxx, values_by_fxx.get(fxx, {})) for fxx in request.fxx_hours]
    first = ordered[0][1] if ordered else {}
    temp_f = [_k_scalar_to_f(v.get("temp_k_2m")) for _, v in ordered]
    dew_f = [_k_scalar_to_f(v.get("dewpoint_k_2m")) for _, v in ordered]
    wind_mph = [_ms_scalar_to_mph(v.get("wind_speed_ms_10m")) for _, v in ordered]
    gust_mph = [_ms_scalar_to_mph(v.get("wind_gust_ms")) for _, v in ordered]
    return {
        "forecast_temp_at_as_of_f": _k_scalar_to_f(first.get("temp_k_2m")),
        "dewpoint_mean_f": _clean_mean(dew_f),
        "dewpoint_at_as_of_f": _k_scalar_to_f(first.get("dewpoint_k_2m")),
        "humidity_mean": _clean_mean([v.get("relative_humidity_pct_2m") for _, v in ordered]),
        "humidity_at_as_of": first.get("relative_humidity_pct_2m", pd.NA),
        "precip_amount": _clean_sum([v.get("precip_mm_1h") for _, v in ordered]),
        "cloud_cover_mean": _clean_mean([v.get("cloud_cover_pct") for _, v in ordered]),
        "cloud_cover_max": _clean_max([v.get("cloud_cover_pct") for _, v in ordered]),
        "wind_speed_mean": _clean_mean(wind_mph),
        "wind_speed_max": _clean_max(wind_mph),
        "wind_speed_at_as_of": _ms_scalar_to_mph(first.get("wind_speed_ms_10m")),
        "wind_direction_mean": _circular_mean_deg([v.get("wind_direction_deg_10m") for _, v in ordered]),
        "wind_direction_at_as_of": first.get("wind_direction_deg_10m", pd.NA),
        "wind_gust_max": _clean_max(gust_mph),
        "visibility_mean": _clean_mean([v.get("visibility_m") for _, v in ordered]),
        "ceiling_min": _clean_min([v.get("ceiling_m") for _, v in ordered]),
    }


def plan_nwp_requests(
    station_meta: pd.DataFrame,
    dates: list[str],
    models: list[str],
    completed: set[tuple[str, str, str, str, str]] | None = None,
    timing_mode: str = TIMING_MODE_STRICT_6AM,
) -> list[NwpRequest]:
    completed = completed or set()
    archive = nwp_archive_starts()
    requests: list[NwpRequest] = []
    for row in station_meta.itertuples(index=False):
        station_id = str(row.station_id).upper()
        timezone = str(row.timezone)
        for contract_date in dates:
            day = date.fromisoformat(contract_date)
            for model in models:
                if (station_id, contract_date, model, model, timing_mode) in completed:
                    continue
                if day < archive[model]:
                    continue
                if timing_mode == TIMING_MODE_FRESH_AFTER_6AM:
                    cycle, fxx_hours, as_of, window_start, window_end = choose_fresh_after_6am_cycle(
                        model,
                        contract_date,
                        timezone,
                    )
                else:
                    as_of = forecast_as_of_for_timing(contract_date, timezone, timing_mode)
                    window_start, window_end = forecast_window_for_timing(contract_date, timezone, timing_mode, as_of)
                    cycle, fxx_hours = choose_cycle(
                        model,
                        contract_date,
                        timezone,
                        as_of,
                        timing_mode=timing_mode,
                        forecast_window_start=window_start,
                        forecast_window_end=window_end,
                    )
                if cycle is None:
                    continue
                requests.append(
                    NwpRequest(
                        station_id=station_id,
                        station_name=str(row.station_name),
                        airport_name=str(row.airport_name),
                        timezone=timezone,
                        contract_date=contract_date,
                        model=model,
                        forecast_as_of=as_of,
                        cycle=cycle,
                        fxx_hours=fxx_hours,
                        timing_mode=timing_mode,
                        cycle_selection_policy=_cycle_selection_policy(model, timing_mode),
                        forecast_window_start=window_start,
                        forecast_window_end=window_end,
                    )
                )
    return requests


def choose_cycle(
    model: str,
    contract_date: str,
    timezone: str,
    as_of_utc: datetime,
    timing_mode: str = TIMING_MODE_STRICT_6AM,
    forecast_window_start: datetime | None = None,
    forecast_window_end: datetime | None = None,
) -> tuple[datetime | None, tuple[int, ...]]:
    model = model.lower()
    if timing_mode == TIMING_MODE_FRESH_AFTER_6AM:
        cycle, fxx_hours, _, _, _ = choose_fresh_after_6am_cycle(model, contract_date, timezone)
        return (cycle, fxx_hours) if cycle is not None and fxx_hours else (None, ())
    if timing_mode == TIMING_MODE_SAME_DAY_11AM:
        if forecast_window_start is None or forecast_window_end is None:
            forecast_window_start, forecast_window_end = forecast_window_for_timing(
                contract_date,
                timezone,
                timing_mode,
                as_of_utc,
            )
        return choose_latest_cycle_for_window(
            model,
            as_of_utc,
            forecast_window_start,
            forecast_window_end,
            min_fxx=1 if model == "nbm" else 0,
        )
    if timing_mode != TIMING_MODE_STRICT_6AM:
        raise ValueError(f"Unsupported timing_mode: {timing_mode}")
    cutoff = as_of_utc.astimezone(UTC) - timedelta(minutes=MODEL_LAG_MINUTES.get(model, 120))
    cutoff = cutoff.replace(minute=0, second=0, microsecond=0)
    max_fxx = MODEL_MAX_FXX.get(model, 72)
    search_hours = MODEL_SEARCH_HOURS.get(model, 36)
    for offset in range(search_hours + 1):
        candidate = cutoff - timedelta(hours=offset)
        if model == "gfs" and candidate.hour not in GFS_CYCLES:
            continue
        fxx_hours = tuple(f for f in forecast_hours_for_local_day(candidate, contract_date, timezone) if 0 <= f <= max_fxx)
        if not fxx_hours:
            continue
        if model == "hrrr" and max(fxx_hours) > 18 and candidate.hour not in HRRR_LONG_CYCLES:
            continue
        return candidate, fxx_hours
    return None, ()


def choose_latest_cycle_for_window(
    model: str,
    as_of_utc: datetime,
    forecast_window_start: datetime,
    forecast_window_end: datetime,
    min_fxx: int = 0,
) -> tuple[datetime | None, tuple[int, ...]]:
    model = model.lower()
    allowed_hours = set(model_cycle_hours(model))
    candidate = (as_of_utc.astimezone(UTC) - timedelta(hours=min_fxx)).replace(
        minute=0,
        second=0,
        microsecond=0,
    )
    max_fxx = MODEL_MAX_FXX.get(model, 72)
    search_hours = MODEL_SEARCH_HOURS.get(model, 36)
    for offset in range(search_hours + 1):
        cycle = candidate - timedelta(hours=offset)
        if cycle.hour not in allowed_hours:
            continue
        fxx_hours = tuple(
            f
            for f in forecast_hours_for_utc_window(cycle, forecast_window_start, forecast_window_end)
            if min_fxx <= f <= max_fxx
        )
        if not fxx_hours:
            continue
        if model == "hrrr" and max(fxx_hours) > 18 and cycle.hour not in HRRR_LONG_CYCLES:
            continue
        return cycle, fxx_hours
    return None, ()


def verify_sdk_coverage(
    sdk_cache_dir: str | Path,
    stations: Iterable[str] | None = None,
    models: Iterable[str] | None = None,
    start_date: str = "2016-01-01",
    end_date: str | None = None,
    max_missing_rows: int = 200000,
    timing_mode: str = TIMING_MODE_STRICT_6AM,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = Path(sdk_cache_dir)
    station_meta = station_registry_frame(stations)
    station_ids = set(station_meta["station_id"])
    models = [m.lower() for m in (models or NWP_MODELS)]
    dates = date_range(start_date, resolve_contract_end(end_date))
    actuals = _load_existing(out_dir / SDK_ACTUALS_FILE)
    nwp = _load_existing(out_dir / SDK_NWP_FILE)
    direct_nbm = _load_existing(out_dir / DIRECT_NBM_FILE)
    if not direct_nbm.empty:
        nwp = pd.concat([frame for frame in [nwp, direct_nbm] if not frame.empty], ignore_index=True)
    archive = nwp_archive_starts()
    summary: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    actual_ok = _ok_rows(actuals)
    actual_keys = {
        (str(row.station_id).upper(), str(row.contract_date)[:10])
        for row in actual_ok.itertuples(index=False)
        if str(row.station_id).upper() in station_ids
    }
    actual_expected = {(station_id, d) for station_id in station_ids for d in dates}
    actual_missing = sorted(actual_expected - actual_keys)
    summary.append(_coverage_row("actuals", "observed_high", len(actual_expected), len(actual_keys), actuals))
    _extend_missing(missing, "actuals", "observed_high", actual_missing, max_missing_rows)

    nwp_ok = _ok_rows(nwp)
    nwp_keys = {
        (
            str(row.station_id).upper(),
            str(row.contract_date)[:10],
            str(row.provider).lower(),
            str(row.model).lower(),
            str(getattr(row, "timing_mode", TIMING_MODE_STRICT_6AM) or TIMING_MODE_STRICT_6AM).lower(),
        )
        for row in nwp_ok.itertuples(index=False)
        if str(row.station_id).upper() in station_ids
    }
    for model in models:
        eligible_dates = [d for d in dates if date.fromisoformat(d) >= archive[model]]
        expected = {(station_id, d, model, model, timing_mode) for station_id in station_ids for d in eligible_dates}
        have = {key for key in nwp_keys if key[2] == model and key[3] == model and key[4] == timing_mode}
        missing_keys = sorted(expected - have)
        summary.append(_coverage_row(model, model, len(expected), len(have & expected), nwp, timing_mode=timing_mode))
        _extend_missing(missing, model, model, missing_keys, max_missing_rows)

    summary_frame = pd.DataFrame(summary)
    missing_frame = pd.DataFrame(missing)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_frame.to_csv(out_dir / SDK_COVERAGE_SUMMARY_FILE, index=False)
    missing_frame.to_csv(out_dir / SDK_MISSING_COVERAGE_FILE, index=False)
    return summary_frame, missing_frame


def _load_obs():
    try:
        module = importlib.import_module("mostlyright.weather.obs")
        return module.obs
    except ImportError as exc:
        raise RuntimeError("Mostly Right weather obs SDK is not installed.") from exc


def _load_forecast_nwp(station_meta: pd.DataFrame):
    try:
        _ensure_ecmwflibs_available()
        _patch_mostlyright_station_runtime(station_meta)
        _patch_mostlyright_nbm_archive_urls()
        _patch_mostlyright_nwp_variables()
        _patch_mostlyright_nbm_ensemble_records()
        from mostlyright.weather.forecast_nwp import forecast_nwp
    except ImportError as exc:
        raise RuntimeError(
            "Mostly Right NWP is not installed. Install with "
            'pip install "mostlyrightmd-weather[nwp]>=1.0,<2.0".'
        ) from exc
    return forecast_nwp


def _ensure_ecmwflibs_available() -> None:
    try:
        import ecmwflibs

        root = Path(ecmwflibs.__file__).parent
        os.environ["PATH"] = f"{root};{os.environ.get('PATH', '')}"
        os.environ.setdefault("ECCODES_LIB_DIR", str(root))
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(root))
    except Exception:
        return


def _patch_mostlyright_station_runtime(station_meta: pd.DataFrame) -> None:
    try:
        from mostlyright._internal import _stations
        from mostlyright._internal._stations import StationInfo
    except Exception:
        return
    stations = getattr(_stations, "STATIONS", None)
    if not isinstance(stations, dict):
        return
    for row in station_meta.itertuples(index=False):
        station_id = str(row.station_id).upper()
        code = station_id[1:] if station_id.startswith("K") and len(station_id) == 4 else station_id
        try:
            lat = float(row.lat)
            lon = float(row.lon)
        except Exception:
            continue
        existing = stations.get(code)
        stations[code] = StationInfo(
            code=code,
            ghcnh_id=str(getattr(existing, "ghcnh_id", "") or ""),
            icao=station_id,
            name=str(row.station_name),
            tz=str(row.timezone),
            latitude=lat,
            longitude=lon,
            country=str(getattr(existing, "country", getattr(row, "country", "US")) or "US"),
        )


def _patch_mostlyright_nwp_variables() -> None:
    enable_nbm_wind = os.getenv("WEATHER_RESEARCH_NBM_ENABLE_WIND", "").strip().lower() in {"1", "true", "yes", "y", "on"}
    patched_maps: list[dict[str, Any]] = []
    try:
        module = importlib.import_module("mostlyright.weather.forecast_nwp")
    except Exception:
        module = None
    if module is not None:
        modules = module.get_variable_map.__globals__.get("_MODULES", {})
        variable_map = getattr(modules.get("nbm"), "VARIABLE_MAP", None)
        if isinstance(variable_map, dict):
            patched_maps.append(variable_map)
    try:
        nbm_grid = importlib.import_module("mostlyright.weather._fetchers._nwp_grids.nbm")
    except Exception:
        nbm_grid = None
    if nbm_grid is not None:
        variable_map = getattr(nbm_grid, "VARIABLE_MAP", None)
        if isinstance(variable_map, dict):
            patched_maps.append(variable_map)

    for variable_map in patched_maps:
        _patch_nbm_variable_map(variable_map, enable_wind=enable_nbm_wind)


def _patch_nbm_variable_map(variable_map: dict[str, Any], enable_wind: bool = False) -> None:
    # NBM wind U/V currently returns blank in our probes and adds fragile
    # GRIB fields. Keep the stable NBM feature set lean unless explicitly
    # requested for diagnostics.
    if not enable_wind:
        variable_map.pop("wind_u_ms_10m", None)
        variable_map.pop("wind_v_ms_10m", None)
    # Older NBM pressure records can crash the GRIB decoder before Python can recover.
    variable_map.pop("pressure_pa_mslp", None)


def _patch_mostlyright_nbm_archive_urls() -> None:
    try:
        archive = importlib.import_module("mostlyright.weather._fetchers._nwp_archive")
    except Exception:
        return
    urls_by_model = getattr(archive, "_MIRROR_URLS_BY_MODEL", None)
    if not isinstance(urls_by_model, dict):
        return
    nbm_urls = urls_by_model.get("nbm")
    if isinstance(nbm_urls, dict):
        nbm_urls["aws_bdp"] = "https://noaa-nbm-grib2-pds.s3.amazonaws.com"


def _patch_mostlyright_nbm_ensemble_records() -> None:
    try:
        module = importlib.import_module("mostlyright.weather.forecast_nwp")
    except Exception:
        return
    if getattr(module, "_weather_research_nbm_record_patch", False):
        return
    original = getattr(module, "filter_records", None)
    if original is None:
        return

    def filter_records_without_ensemble_stddev(records, variable_map):
        filtered = original(records, variable_map)
        return _drop_ensemble_std_dev_records(filtered)

    module.filter_records = filter_records_without_ensemble_stddev
    module._weather_research_nbm_record_patch = True


def _drop_ensemble_std_dev_records(records: list[Any]) -> list[Any]:
    grouped: dict[tuple[str, str], list[Any]] = {}
    for record in records:
        grouped.setdefault((str(record.variable), str(record.level)), []).append(record)
    clean: list[Any] = []
    for group in grouped.values():
        if len(group) == 1:
            clean.extend(group)
            continue
        non_stddev = [
            record
            for record in group
            if "ens std dev" not in str(getattr(record, "forecast_period", "")).lower()
        ]
        clean.append((non_stddev or group)[0])
    return sorted(clean, key=lambda record: int(getattr(record, "record_no", 0)))


def _normalize_obs_frame(raw: pd.DataFrame, station: Any, start_date: str, end_date: str) -> list[dict[str, Any]]:
    station_id = str(station.station_id).upper()
    frame = raw.copy() if isinstance(raw, pd.DataFrame) else pd.DataFrame(raw)
    if frame.empty:
        return [_unavailable_actual_row(station, d, "SDK obs returned no rows") for d in date_range(start_date, end_date)]
    if "date" not in frame.columns:
        frame = frame.reset_index().rename(columns={"index": "date"})
    high_col = _first_existing(frame, ["obs_high_f", "actual_high_f", "high_f", "temp_max_f", "max_temp_f"])
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in frame.itertuples(index=False):
        values = row._asdict()
        contract_date = str(values.get("date", ""))[:10]
        if not contract_date:
            continue
        seen.add(contract_date)
        actual_high = pd.to_numeric(pd.Series([values.get(high_col)]), errors="coerce").iloc[0] if high_col else pd.NA
        ok = pd.notna(actual_high)
        rows.append(
            {
                "station_id": station_id,
                "station_name": str(station.station_name),
                "airport_name": str(station.airport_name),
                "contract_date": contract_date,
                "actual_high_f": float(actual_high) if ok else pd.NA,
                "actual_source": values.get("source", pd.NA),
                "obs_count": values.get("obs_count", pd.NA),
                "obs_low_f": values.get("obs_low_f", pd.NA),
                "obs_mean_f": values.get("obs_mean_f", pd.NA),
                "data_source": "mostlyright.weather.obs.obs",
                "source_file_or_url": "mostlyright.weather.obs.obs",
                "fetch_status": "ok" if ok else "unavailable",
                "unavailable_reason": pd.NA if ok else "missing obs_high_f",
            }
        )
    for missing_date in sorted(set(date_range(start_date, end_date)) - seen):
        rows.append(_unavailable_actual_row(station, missing_date, "SDK obs returned no row for date"))
    return rows


def _unavailable_actual_row(station: Any, contract_date: str, reason: str) -> dict[str, Any]:
    return {
        "station_id": str(station.station_id).upper(),
        "station_name": str(station.station_name),
        "airport_name": str(station.airport_name),
        "contract_date": contract_date,
        "actual_high_f": pd.NA,
        "actual_source": pd.NA,
        "obs_count": pd.NA,
        "obs_low_f": pd.NA,
        "obs_mean_f": pd.NA,
        "data_source": "mostlyright.weather.obs.obs",
        "source_file_or_url": "mostlyright.weather.obs.obs",
        "fetch_status": "unavailable",
        "unavailable_reason": reason,
    }


def _cycle_selection_policy(model: str, timing_mode: str) -> str:
    if timing_mode == TIMING_MODE_FRESH_AFTER_6AM:
        return f"first_{model}_cycle_issued_at_or_after_6am_local_remaining_day"
    if timing_mode == TIMING_MODE_SAME_DAY_11AM:
        return f"latest_{model}_cycle_at_or_before_11am_local_remaining_day"
    return f"latest_{model}_cycle_before_6am_local_with_model_lag"


def _fresh_model_max_fxx(model: str, cycle: datetime) -> int:
    if model == "hrrr":
        return 48 if cycle.hour in HRRR_LONG_CYCLES else 18
    return MODEL_MAX_FXX.get(model, 72)


def _group_nwp_requests(requests: list[NwpRequest]) -> Iterable[list[NwpRequest]]:
    groups: dict[tuple[Any, ...], list[NwpRequest]] = {}
    for request in requests:
        key = (
            request.model,
            request.timing_mode,
            request.contract_date,
            request.timezone,
            request.cycle,
            request.fxx_hours,
            request.forecast_window_start,
            request.forecast_window_end,
        )
        groups.setdefault(key, []).append(request)
    for key in sorted(groups):
        yield sorted(groups[key], key=lambda request: request.station_id)


def _fetch_and_summarize_nwp_batch(
    forecast_nwp,
    requests: list[NwpRequest],
    client: Any = None,
    fxx_workers: int = 1,
) -> list[dict[str, Any]]:
    if not requests:
        return []
    try:
        hourly = _fetch_batch_hourly(forecast_nwp, requests, client=client, fxx_workers=fxx_workers)
    except Exception as exc:  # noqa: BLE001
        logging.warning(
            "Mostly Right NWP batch unavailable for %s %s %s %s stations: %s",
            requests[0].model,
            requests[0].contract_date,
            requests[0].timing_mode,
            len(requests),
            exc,
        )
        return [_unavailable_nwp_row(request, str(exc)) for request in requests]
    rows: list[dict[str, Any]] = []
    for request in requests:
        if hourly.empty or "station" not in hourly.columns:
            station_hourly = pd.DataFrame()
        else:
            station_hourly = hourly.loc[hourly["station"].astype(str).str.upper() == request.station_id].copy()
        rows.append(_summarize_nwp_request(request, station_hourly))
    return rows


def _fetch_batch_hourly(
    forecast_nwp,
    requests: list[NwpRequest],
    client: Any = None,
    fxx_workers: int = 1,
) -> pd.DataFrame:
    first = requests[0]
    station_ids = [request.station_id for request in requests]
    if _use_nwp_subprocess_fetch(first):
        return _fetch_nwp_batch_hourly_subprocess(requests)
    if fxx_workers <= 1 or len(first.fxx_hours) <= 1:
        frames = [
            _fetch_one_nwp_fxx_with_retries(forecast_nwp, station_ids, first, fxx, client=client)
            for fxx in first.fxx_hours
        ]
    else:
        # Avoid sharing one httpx client across worker threads unless the SDK explicitly guarantees it.
        max_workers = min(fxx_workers, len(first.fxx_hours))
        frames = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_fetch_one_nwp_fxx_with_retries, forecast_nwp, station_ids, first, fxx, None)
                for fxx in first.fxx_hours
            ]
            for future in as_completed(futures):
                frames.append(future.result())
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _use_nwp_subprocess_fetch(request: NwpRequest) -> bool:
    model_value = os.getenv(f"WEATHER_RESEARCH_{request.model.upper()}_SUBPROCESS")
    if model_value is not None:
        return model_value.strip().lower() not in {"0", "false", "no", "off"}
    value = os.getenv("WEATHER_RESEARCH_NWP_SUBPROCESS")
    if value is not None:
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return request.model in {"hrrr", "gfs", "nbm"}


def _use_nbm_subprocess_fetch(request: NwpRequest) -> bool:
    return _use_nwp_subprocess_fetch(request)


def _fetch_nwp_batch_hourly_subprocess(requests: list[NwpRequest]) -> pd.DataFrame:
    first = requests[0]
    payload = {
        "stations": [request.station_id for request in requests],
        "model": first.model,
        "cycle": first.cycle.isoformat(),
        "fxx_hours": list(first.fxx_hours),
    }
    max_attempts = max(1, _nwp_fxx_fetch_retries() + 1)
    timeout_seconds = _nwp_subprocess_timeout_seconds()
    last_error = ""
    for attempt in range(1, max_attempts + 1):
        with tempfile.TemporaryDirectory(prefix=f"weather_research_{first.model}_") as tmp:
            tmp_path = Path(tmp)
            payload_path = tmp_path / "payload.json"
            output_path = tmp_path / "hourly.csv"
            payload["output"] = str(output_path)
            payload_path.write_text(json.dumps(payload), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, "-c", _NWP_SUBPROCESS_CODE, str(payload_path)],
                cwd=str(_project_root()),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            if result.returncode == 0 and output_path.exists():
                frame = pd.read_csv(output_path)
                return _filter_nwp_hourly_frame(frame, first) if not frame.empty else pd.DataFrame()
            last_error = (result.stderr or result.stdout or "").strip()
            logging.warning(
                "Mostly Right %s subprocess failed for %s %s cycle=%s attempt %s/%s: %s",
                first.model.upper(),
                first.contract_date,
                first.timing_mode,
                first.cycle.isoformat(),
                attempt,
                max_attempts,
                last_error[-500:],
            )
    raise RuntimeError(f"{first.model.upper()} subprocess failed after {max_attempts} attempts: {last_error[-500:]}")


def _fetch_nbm_batch_hourly_subprocess(requests: list[NwpRequest]) -> pd.DataFrame:
    return _fetch_nwp_batch_hourly_subprocess(requests)


def _nwp_subprocess_timeout_seconds() -> int:
    value = os.getenv("WEATHER_RESEARCH_NWP_SUBPROCESS_TIMEOUT_SECONDS")
    if value is None:
        return 600
    try:
        return max(60, int(value))
    except ValueError:
        return 600


def _nbm_subprocess_timeout_seconds() -> int:
    return _nwp_subprocess_timeout_seconds()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


_NWP_SUBPROCESS_CODE = r"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.calibration.sdk_pipeline import _load_forecast_nwp, station_registry_frame

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
stations = [str(station).upper() for station in payload["stations"]]
cycle = datetime.fromisoformat(payload["cycle"])
forecast_nwp = _load_forecast_nwp(station_registry_frame(stations))
frames = []
for fxx in payload["fxx_hours"]:
    try:
        frame = forecast_nwp(stations, payload["model"], cycle=cycle, fxx=int(fxx))
    except Exception:
        station_frames = []
        for station in stations:
            try:
                station_frame = forecast_nwp([station], payload["model"], cycle=cycle, fxx=int(fxx))
            except Exception:
                continue
            if station_frame is not None and not station_frame.empty:
                station_frames.append(station_frame)
        frame = pd.concat(station_frames, ignore_index=True) if station_frames else pd.DataFrame()
    if frame is not None and not frame.empty:
        frames.append(frame)
combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
combined.to_csv(payload["output"], index=False)
"""


_NBM_SUBPROCESS_CODE = _NWP_SUBPROCESS_CODE


def _fetch_one_nwp_fxx(
    forecast_nwp,
    station_ids: list[str],
    request: NwpRequest,
    fxx: int,
    client: Any = None,
) -> pd.DataFrame:
    kwargs = {"cycle": request.cycle, "fxx": fxx}
    if client is not None:
        kwargs["client"] = client
    frame = forecast_nwp(station_ids, request.model, **kwargs)
    if frame is None or frame.empty:
        return pd.DataFrame()
    return _filter_nwp_hourly_frame(frame, request)


def _fetch_one_nwp_fxx_with_retries(
    forecast_nwp,
    station_ids: list[str],
    request: NwpRequest,
    fxx: int,
    client: Any = None,
) -> pd.DataFrame:
    max_attempts = max(1, _nwp_fxx_fetch_retries() + 1)
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return _fetch_one_nwp_fxx(forecast_nwp, station_ids, request, fxx, client=client)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < max_attempts and _is_transient_nwp_fxx_error(exc):
                logging.warning(
                    "Mostly Right NWP transient f%03d error for %s %s %s (%s/%s); retrying: %s",
                    fxx,
                    request.model,
                    request.contract_date,
                    request.timing_mode,
                    attempt,
                    max_attempts,
                    exc,
                )
                time.sleep(_nwp_fxx_retry_sleep_seconds())
                continue
            break
    logging.warning(
        "Mostly Right NWP f%03d unavailable for %s %s %s cycle=%s stations=%s: %s",
        fxx,
        request.model,
        request.contract_date,
        request.timing_mode,
        request.cycle.isoformat(),
        ",".join(station_ids),
        last_exc,
    )
    return pd.DataFrame()


def _nwp_fxx_fetch_retries() -> int:
    value = os.getenv("WEATHER_RESEARCH_NWP_FXX_RETRIES")
    if value is None:
        return NWP_FXX_FETCH_RETRIES
    try:
        return max(0, int(value))
    except ValueError:
        return NWP_FXX_FETCH_RETRIES


def _nwp_fxx_retry_sleep_seconds() -> float:
    value = os.getenv("WEATHER_RESEARCH_NWP_FXX_RETRY_SLEEP_SECONDS")
    if value is None:
        return NWP_FXX_RETRY_SLEEP_SECONDS
    try:
        return max(0.0, float(value))
    except ValueError:
        return NWP_FXX_RETRY_SLEEP_SECONDS


def _is_transient_nwp_fxx_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    transient_markers = (
        "server disconnected",
        "connection reset",
        "connection aborted",
        "timeout",
        "timed out",
        "temporarily unavailable",
        "too many requests",
        "rate limit",
        "429",
        "500",
        "502",
        "503",
        "504",
        "readerror",
        "remoteprotocolerror",
        "connecterror",
    )
    return any(marker in text for marker in transient_markers)


def _filter_nwp_hourly_frame(frame: pd.DataFrame, request: NwpRequest) -> pd.DataFrame:
    frame = frame.copy()
    frame["valid_at"] = pd.to_datetime(frame["valid_at"], errors="coerce", utc=True)
    return frame.loc[
        (frame["valid_at"] >= request.forecast_window_start)
        & (frame["valid_at"] < request.forecast_window_end)
    ]


def _fetch_request_hourly(forecast_nwp, request: NwpRequest, client: Any = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for fxx in request.fxx_hours:
        frame = _fetch_one_nwp_fxx_with_retries(
            forecast_nwp,
            [request.station_id],
            request,
            fxx,
            client=client,
        )
        if frame is None or frame.empty:
            continue
        if not frame.empty:
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _summarize_nwp_request(request: NwpRequest, hourly: pd.DataFrame) -> dict[str, Any]:
    if hourly.empty or "temp_k_2m" not in hourly:
        return _unavailable_nwp_row(request, "no hourly temperature rows returned")
    temps_f = _k_to_f(pd.to_numeric(hourly["temp_k_2m"], errors="coerce")).dropna()
    if temps_f.empty:
        return _unavailable_nwp_row(request, "no valid temp_k_2m values returned")
    dewpoint_f = _k_to_f(pd.to_numeric(hourly.get("dewpoint_k_2m"), errors="coerce")).dropna()
    wind_u = pd.to_numeric(hourly.get("wind_u_ms_10m"), errors="coerce")
    wind_v = pd.to_numeric(hourly.get("wind_v_ms_10m"), errors="coerce")
    wind_gust = pd.to_numeric(hourly.get("wind_gust_ms", pd.Series(dtype=float)), errors="coerce")
    wind_speed_mph = np.sqrt(wind_u**2 + wind_v**2) * 2.2369362921
    wind_direction_deg = _wind_direction_from_uv(wind_u, wind_v)
    ordered = hourly.sort_values("valid_at") if "valid_at" in hourly else hourly
    first = ordered.iloc[0] if not ordered.empty else pd.Series(dtype="object")
    returned_fxx = _returned_fxx_hours(request, hourly)
    missing_fxx = tuple(hour for hour in request.fxx_hours if hour not in returned_fxx)
    hour_completeness = len(returned_fxx) / len(request.fxx_hours) if request.fxx_hours else pd.NA
    return {
        "station_id": request.station_id,
        "station_name": request.station_name,
        "airport_name": request.airport_name,
        "provider": request.model,
        "model": request.model,
        "source_label": f"mostlyright_forecast_nwp_{request.model}_{TIMING_MODE_SOURCE_LABELS[request.timing_mode]}",
        "timing_mode": request.timing_mode,
        "cycle_selection_policy": request.cycle_selection_policy,
        "contract_date": request.contract_date,
        "forecast_as_of": request.forecast_as_of.isoformat(),
        "issued_at": request.cycle.isoformat(),
        "forecast_window_start": request.forecast_window_start.isoformat(),
        "forecast_window_end": request.forecast_window_end.isoformat(),
        "horizon_hours": 0,
        "raw_forecast_high_f": float(temps_f.max()),
        "forecast_hour_min": min(request.fxx_hours),
        "forecast_hour_max": max(request.fxx_hours),
        "forecast_hour_count_requested": len(request.fxx_hours),
        "forecast_hour_count_returned": len(returned_fxx),
        "forecast_hour_missing": _format_fxx_hours(missing_fxx) if missing_fxx else pd.NA,
        "forecast_hour_completeness": hour_completeness,
        "forecast_hour_fetch_status": "partial" if missing_fxx else "ok",
        "grid_dist_km_mean": _mean(hourly.get("grid_dist_km")),
        "forecast_temp_at_as_of_f": _k_scalar_to_f(first.get("temp_k_2m")),
        "cloud_cover_mean": pd.NA,
        "cloud_cover_max": pd.NA,
        "precip_amount": _sum(hourly.get("precip_mm_1h")),
        "wind_speed_mean": _series_mean(wind_speed_mph),
        "wind_speed_max": _series_max(wind_speed_mph),
        "wind_speed_at_as_of": _ms_scalar_to_mph(_uv_speed_ms(first.get("wind_u_ms_10m"), first.get("wind_v_ms_10m"))),
        "wind_direction_mean": _series_mean(wind_direction_deg),
        "wind_direction_at_as_of": _wind_direction_scalar(first.get("wind_u_ms_10m"), first.get("wind_v_ms_10m")),
        "wind_gust_max": _ms_scalar_to_mph(_series_max(wind_gust)),
        "dewpoint_mean_f": _series_mean(dewpoint_f),
        "dewpoint_at_as_of_f": _k_scalar_to_f(first.get("dewpoint_k_2m")),
        "humidity_mean": _mean(hourly.get("relative_humidity_pct_2m")),
        "humidity_at_as_of": first.get("relative_humidity_pct_2m", pd.NA),
        "pressure_mslp_mean": _mean(hourly.get("pressure_pa_mslp")),
        "pressure_surface_mean": _mean(hourly.get("pressure_pa_surface")),
        "visibility_mean": pd.NA,
        "ceiling_min": pd.NA,
        "data_source": "mostlyright.weather.forecast_nwp",
        "source_file_or_url": "mostlyright.weather.forecast_nwp",
        "fetch_status": "ok",
        "unavailable_reason": pd.NA,
        WEATHER_FEATURE_FLAG: True,
    }


def _unavailable_nwp_row(request: NwpRequest, reason: str) -> dict[str, Any]:
    return {
        "station_id": request.station_id,
        "station_name": request.station_name,
        "airport_name": request.airport_name,
        "provider": request.model,
        "model": request.model,
        "source_label": f"mostlyright_forecast_nwp_{request.model}_{TIMING_MODE_SOURCE_LABELS[request.timing_mode]}",
        "timing_mode": request.timing_mode,
        "cycle_selection_policy": request.cycle_selection_policy,
        "contract_date": request.contract_date,
        "forecast_as_of": request.forecast_as_of.isoformat(),
        "issued_at": request.cycle.isoformat(),
        "forecast_window_start": request.forecast_window_start.isoformat(),
        "forecast_window_end": request.forecast_window_end.isoformat(),
        "horizon_hours": 0,
        "raw_forecast_high_f": pd.NA,
        "forecast_hour_min": min(request.fxx_hours) if request.fxx_hours else pd.NA,
        "forecast_hour_max": max(request.fxx_hours) if request.fxx_hours else pd.NA,
        "forecast_hour_count_requested": len(request.fxx_hours),
        "forecast_hour_count_returned": 0,
        "forecast_hour_missing": _format_fxx_hours(request.fxx_hours),
        "forecast_hour_completeness": 0.0 if request.fxx_hours else pd.NA,
        "forecast_hour_fetch_status": "unavailable",
        "data_source": "mostlyright.weather.forecast_nwp",
        "source_file_or_url": "mostlyright.weather.forecast_nwp",
        "fetch_status": "unavailable",
        "unavailable_reason": reason,
        WEATHER_FEATURE_FLAG: False,
    }


def _returned_fxx_hours(request: NwpRequest, hourly: pd.DataFrame) -> tuple[int, ...]:
    if hourly.empty or "valid_at" not in hourly:
        return ()
    valid_times = pd.to_datetime(hourly["valid_at"], errors="coerce", utc=True).dropna()
    if valid_times.empty:
        return ()
    cycle = pd.Timestamp(request.cycle).tz_convert("UTC")
    hours: set[int] = set()
    for valid_at in valid_times:
        delta = valid_at - cycle
        seconds = delta.total_seconds()
        if seconds >= 0 and seconds % 3600 == 0:
            hours.add(int(seconds // 3600))
    return tuple(sorted(hours))


def _format_fxx_hours(hours: Iterable[int]) -> str:
    return ",".join(str(int(hour)) for hour in hours)


def _nwp_http_client() -> Any:
    try:
        import httpx
    except Exception:
        return None
    return httpx.Client(timeout=60)


def _load_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _append_cache(path: Path, existing: pd.DataFrame, rows: list[dict[str, Any]], keys: list[str]) -> pd.DataFrame:
    fresh = pd.DataFrame(rows)
    combined = pd.concat([existing, fresh], ignore_index=True) if not existing.empty else fresh
    for key in keys:
        if key not in combined:
            combined[key] = pd.NA
    if "timing_mode" in combined.columns:
        combined["timing_mode"] = combined["timing_mode"].fillna(TIMING_MODE_STRICT_6AM)
        combined.loc[combined["timing_mode"].astype(str).str.strip() == "", "timing_mode"] = TIMING_MODE_STRICT_6AM
    combined = combined.drop_duplicates(subset=keys, keep="last")
    sort_cols = [col for col in ["contract_date", "provider", "model", "station_id"] if col in combined]
    combined = combined.sort_values(sort_cols).reset_index(drop=True) if sort_cols else combined.reset_index(drop=True)
    combined.to_csv(path, index=False)
    return combined


def _completed_actual_keys(frame: pd.DataFrame) -> set[tuple[str, str]]:
    if frame.empty or not {"station_id", "contract_date", "actual_high_f"}.issubset(frame.columns):
        return set()
    good = _ok_rows(frame).dropna(subset=["actual_high_f"])
    return {(str(row.station_id).upper(), str(row.contract_date)[:10]) for row in good.itertuples(index=False)}


def _completed_nwp_keys(
    frame: pd.DataFrame,
    require_weather_features: bool = False,
) -> set[tuple[str, str, str, str, str]]:
    if frame.empty or not {"station_id", "contract_date", "provider", "model", "raw_forecast_high_f"}.issubset(frame.columns):
        return set()
    if require_weather_features:
        required = {"station_id", "contract_date", "provider", "model", "fetch_status"}
        if not required.issubset(frame.columns):
            return set()
        work = frame.copy()
        if "timing_mode" not in work.columns:
            work["timing_mode"] = TIMING_MODE_STRICT_6AM
        work["timing_mode"] = work["timing_mode"].fillna(TIMING_MODE_STRICT_6AM)
        if WEATHER_FEATURE_FLAG not in work.columns:
            work[WEATHER_FEATURE_FLAG] = False
        status = work["fetch_status"].astype(str).str.lower()
        raw_high_ok = pd.to_numeric(work["raw_forecast_high_f"], errors="coerce").notna()
        feature_done = work[WEATHER_FEATURE_FLAG].map(_truthy)
        good = work.loc[(status == "unavailable") | ((status == "ok") & raw_high_ok & feature_done)].copy()
    else:
        good = _ok_rows(frame).dropna(subset=["raw_forecast_high_f"])
    if "timing_mode" not in good.columns:
        good = good.copy()
        good["timing_mode"] = TIMING_MODE_STRICT_6AM
    good["timing_mode"] = good["timing_mode"].fillna(TIMING_MODE_STRICT_6AM)
    return {
        (
            str(row.station_id).upper(),
            str(row.contract_date)[:10],
            str(row.provider).lower(),
            str(row.model).lower(),
            str(row.timing_mode).lower(),
        )
        for row in good.itertuples(index=False)
    }


def _ok_nwp_keys(frame: pd.DataFrame) -> set[tuple[str, str, str, str, str]]:
    return _completed_nwp_keys(frame)


def _completed_direct_nbm_keys(
    frame: pd.DataFrame,
    require_weather_features: bool = False,
) -> set[tuple[str, str, str, str, str]]:
    if not require_weather_features:
        return _completed_nwp_keys(frame)
    required = {"station_id", "contract_date", "provider", "model", "fetch_status"}
    if frame.empty or not required.issubset(frame.columns):
        return set()
    work = frame.copy()
    if "timing_mode" not in work.columns:
        work["timing_mode"] = TIMING_MODE_SAME_DAY_11AM
    work["timing_mode"] = work["timing_mode"].fillna(TIMING_MODE_SAME_DAY_11AM)
    if DIRECT_NBM_WEATHER_FEATURE_FLAG not in work.columns:
        work[DIRECT_NBM_WEATHER_FEATURE_FLAG] = False
    status = work["fetch_status"].astype(str).str.lower()
    raw_high_ok = (
        pd.to_numeric(work["raw_forecast_high_f"], errors="coerce").notna()
        if "raw_forecast_high_f" in work.columns
        else pd.Series(False, index=work.index)
    )
    feature_done = work[DIRECT_NBM_WEATHER_FEATURE_FLAG].map(_truthy)
    complete = (status == "unavailable") | ((status == "ok") & raw_high_ok & feature_done)
    done = work.loc[complete]
    return {
        (
            str(row.station_id).upper(),
            str(row.contract_date)[:10],
            str(row.provider).lower(),
            str(row.model).lower(),
            str(row.timing_mode).lower(),
        )
        for row in done.itertuples(index=False)
    }


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _ok_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    if "fetch_status" not in frame.columns:
        return frame.copy()
    return frame.loc[frame["fetch_status"].astype(str).str.lower() == "ok"].copy()


def _requested_ok_actuals(existing: pd.DataFrame, stations: Iterable[str], dates: list[str]) -> int:
    if existing.empty:
        return 0
    station_set = {str(s).upper() for s in stations}
    date_set = set(dates)
    good = _ok_rows(existing)
    if good.empty or "actual_high_f" not in good:
        return 0
    good = good.dropna(subset=["actual_high_f"])
    return int(
        good.loc[
            good["station_id"].astype(str).str.upper().isin(station_set)
            & good["contract_date"].astype(str).str[:10].isin(date_set)
        ].shape[0]
    )


def _coverage_row(
    provider: str,
    model: str,
    expected: int,
    observed: int,
    source_frame: pd.DataFrame,
    timing_mode: str | None = None,
) -> dict[str, Any]:
    ok = _ok_rows(source_frame)
    if provider != "actuals" and not ok.empty and "provider" in ok:
        ok = ok.loc[ok["provider"].astype(str).str.lower() == provider]
        if timing_mode is not None:
            if "timing_mode" not in ok.columns:
                ok = ok.copy()
                ok["timing_mode"] = TIMING_MODE_STRICT_6AM
            ok = ok.loc[ok["timing_mode"].fillna(TIMING_MODE_STRICT_6AM).astype(str).str.lower() == timing_mode]
    dates = pd.to_datetime(ok.get("contract_date", pd.Series(dtype=str)), errors="coerce").dropna()
    return {
        "provider": provider,
        "model": model,
        "timing_mode": timing_mode if timing_mode is not None else pd.NA,
        "expected_rows": expected,
        "ok_rows": observed,
        "missing_rows": max(expected - observed, 0),
        "coverage_pct": (observed / expected * 100) if expected else 100.0,
        "first_ok_contract_date": dates.min().date().isoformat() if not dates.empty else pd.NA,
        "last_ok_contract_date": dates.max().date().isoformat() if not dates.empty else pd.NA,
    }


def _extend_missing(
    missing: list[dict[str, Any]],
    provider: str,
    model: str,
    keys: list[tuple[Any, ...]],
    max_missing_rows: int,
) -> None:
    remaining = max_missing_rows - len(missing)
    if remaining <= 0:
        return
    for key in keys[:remaining]:
        if len(key) == 2:
            station_id, contract_date = key
        else:
            station_id, contract_date = key[0], key[1]
        missing.append(
            {
                "station_id": station_id,
                "contract_date": contract_date,
                "provider": provider,
                "model": model,
            }
        )


def _first_existing(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def _k_to_f(series: pd.Series) -> pd.Series:
    return (series - 273.15) * 9 / 5 + 32


def _mean(series: pd.Series | None) -> float | Any:
    if series is None:
        return pd.NA
    values = pd.to_numeric(series, errors="coerce").dropna()
    return pd.NA if values.empty else float(values.mean())


def _sum(series: pd.Series | None) -> float | Any:
    if series is None:
        return pd.NA
    values = pd.to_numeric(series, errors="coerce").dropna()
    return pd.NA if values.empty else float(values.sum())


def _series_mean(series: pd.Series) -> float | Any:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return pd.NA if values.empty else float(values.mean())


def _series_max(series: pd.Series) -> float | Any:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return pd.NA if values.empty else float(values.max())


def _k_scalar_to_f(value: Any) -> float | Any:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").dropna()
    return pd.NA if numeric.empty else float((numeric.iloc[0] - 273.15) * 9 / 5 + 32)


def _ms_scalar_to_mph(value: Any) -> float | Any:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").dropna()
    return pd.NA if numeric.empty else float(numeric.iloc[0] * 2.2369362921)


def _uv_speed_ms(u_value: Any, v_value: Any) -> float | Any:
    u = pd.to_numeric(pd.Series([u_value]), errors="coerce")
    v = pd.to_numeric(pd.Series([v_value]), errors="coerce")
    if u.isna().iloc[0] or v.isna().iloc[0]:
        return pd.NA
    return float(np.sqrt(u.iloc[0] ** 2 + v.iloc[0] ** 2))


def _wind_direction_from_uv(u: pd.Series, v: pd.Series) -> pd.Series:
    u_num = pd.to_numeric(u, errors="coerce")
    v_num = pd.to_numeric(v, errors="coerce")
    return (270 - np.degrees(np.arctan2(v_num, u_num))) % 360


def _wind_direction_scalar(u_value: Any, v_value: Any) -> float | Any:
    u = pd.to_numeric(pd.Series([u_value]), errors="coerce")
    v = pd.to_numeric(pd.Series([v_value]), errors="coerce")
    if u.isna().iloc[0] or v.isna().iloc[0]:
        return pd.NA
    return float((270 - np.degrees(np.arctan2(v.iloc[0], u.iloc[0]))) % 360)


def _clean_values(values: Iterable[Any]) -> list[float]:
    return [float(value) for value in pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()]


def _clean_mean(values: Iterable[Any]) -> float | Any:
    clean = _clean_values(values)
    return pd.NA if not clean else float(np.mean(clean))


def _clean_sum(values: Iterable[Any]) -> float | Any:
    clean = _clean_values(values)
    return pd.NA if not clean else float(np.sum(clean))


def _clean_max(values: Iterable[Any]) -> float | Any:
    clean = _clean_values(values)
    return pd.NA if not clean else float(np.max(clean))


def _clean_min(values: Iterable[Any]) -> float | Any:
    clean = _clean_values(values)
    return pd.NA if not clean else float(np.min(clean))


def _circular_mean_deg(values: Iterable[Any]) -> float | Any:
    clean = _clean_values(values)
    if not clean:
        return pd.NA
    radians = np.radians(clean)
    return float((np.degrees(np.arctan2(np.mean(np.sin(radians)), np.mean(np.cos(radians)))) + 360) % 360)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--calibration-dir", default="data/calibration")
    parser.add_argument("--sdk-cache-dir", default=None)
    parser.add_argument("--stations", nargs="*", default=TARGET_STATIONS)
    parser.add_argument("--start-date", default="2016-01-01")
    parser.add_argument("--end-date", default="latest-complete")
    parser.add_argument("--log-level", default="INFO")


def sdk_cache_dir_from_args(args: argparse.Namespace) -> Path:
    return Path(args.sdk_cache_dir) if args.sdk_cache_dir else default_sdk_cache_dir(args.calibration_dir)
