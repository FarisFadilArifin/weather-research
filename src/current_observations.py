from __future__ import annotations

import logging
import math
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .calibration.sdk_pipeline import (
    STATION_METADATA,
    TARGET_STATIONS,
    TIMING_MODE_SAME_DAY_11AM,
    date_range,
    local_datetime_utc,
    resolve_contract_end,
)


CURRENT_OBSERVATIONS_FILE = "sdk_current_observations_11am.csv"

CACHE_KEYS = ["station_id", "contract_date", "timing_mode"]


def backfill_sdk_current_observations(
    sdk_cache_dir: str | Path,
    stations: Iterable[str] | None = None,
    start_date: str = "2021-01-01",
    end_date: str = "latest-complete",
    *,
    timing_mode: str = TIMING_MODE_SAME_DAY_11AM,
    as_of_hour_local: int = 11,
    chunk_days: int = 31,
    source: str | None = None,
    force: bool = False,
    retry_unavailable: bool = False,
    request_retries: int = 3,
    retry_sleep_seconds: float = 60.0,
    sleep_between_chunks: float = 0.0,
    max_chunks: int | None = None,
) -> pd.DataFrame:
    if timing_mode != TIMING_MODE_SAME_DAY_11AM:
        raise ValueError("Current observation features currently support timing_mode='same_day_11am' only")
    out_dir = Path(sdk_cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / CURRENT_OBSERVATIONS_FILE
    existing = _load_existing(cache_path)
    completed = set() if force else _completed_keys(existing, retry_unavailable=retry_unavailable)
    station_ids = [str(s).upper() for s in (stations or TARGET_STATIONS)]
    dates = date_range(start_date, resolve_contract_end(end_date))

    chunks_done = 0
    for station_id in station_ids:
        meta = _station_meta(station_id)
        pending_dates = [
            d
            for d in dates
            if (station_id, d, timing_mode) not in completed
        ]
        if not pending_dates:
            logging.info("Current obs %s: all requested dates already cached", station_id)
            continue
        for chunk in _chunks(pending_dates, chunk_days):
            if max_chunks is not None and chunks_done >= max_chunks:
                return existing
            chunk_start, chunk_end = chunk[0], chunk[-1]
            try:
                raw_rows = fetch_sdk_raw_observations_with_retries(
                    station_id,
                    chunk_start,
                    chunk_end,
                    source=source,
                    request_retries=request_retries,
                    retry_sleep_seconds=retry_sleep_seconds,
                )
                rows = summarize_current_observations(
                    raw_rows,
                    station_id=station_id,
                    station_name=str(meta.get("station_name", station_id)),
                    airport_name=str(meta.get("airport_name", station_id)),
                    timezone=str(meta.get("timezone", "UTC")),
                    contract_dates=chunk,
                    timing_mode=timing_mode,
                    as_of_hour_local=as_of_hour_local,
                    source_filter=source,
                )
            except Exception as exc:  # noqa: BLE001
                if is_transient_observation_error(exc):
                    logging.warning(
                        "Current obs transient failure for %s %s..%s; leaving chunk pending for resume: %s",
                        station_id,
                        chunk_start,
                        chunk_end,
                        exc,
                    )
                    if sleep_between_chunks > 0:
                        time.sleep(sleep_between_chunks)
                    continue
                logging.warning("Current obs unavailable for %s %s..%s: %s", station_id, chunk_start, chunk_end, exc)
                rows = [
                    unavailable_current_observation_row(
                        station_id=station_id,
                        station_name=str(meta.get("station_name", station_id)),
                        airport_name=str(meta.get("airport_name", station_id)),
                        timezone=str(meta.get("timezone", "UTC")),
                        contract_date=d,
                        timing_mode=timing_mode,
                        as_of_hour_local=as_of_hour_local,
                        reason=str(exc),
                    )
                    for d in chunk
                ]
            existing = _append_cache(cache_path, existing, rows)
            completed.update((station_id, d, timing_mode) for d in chunk)
            chunks_done += 1
            ok_count = sum(1 for row in rows if row.get("observed_fetch_status") == "ok")
            logging.info(
                "Current obs %s %s..%s cached %s rows (%s ok)",
                station_id,
                chunk_start,
                chunk_end,
                len(rows),
                ok_count,
            )
            if sleep_between_chunks > 0:
                time.sleep(sleep_between_chunks)
    return existing


def fetch_sdk_raw_observations_with_retries(
    station_id: str,
    start_date: str,
    end_date: str,
    *,
    source: str | None = None,
    request_retries: int = 3,
    retry_sleep_seconds: float = 60.0,
) -> list[dict[str, Any]]:
    attempts = max(int(request_retries), 0) + 1
    for attempt in range(1, attempts + 1):
        try:
            return fetch_sdk_raw_observations(station_id, start_date, end_date, source=source)
        except Exception as exc:  # noqa: BLE001
            if attempt >= attempts or not is_transient_observation_error(exc):
                raise
            sleep_for = max(float(retry_sleep_seconds), 0.0) * attempt
            logging.warning(
                "Current obs transient SDK error for %s %s..%s attempt %s/%s; sleeping %.1fs: %s",
                station_id,
                start_date,
                end_date,
                attempt,
                attempts,
                sleep_for,
                exc,
            )
            time.sleep(sleep_for)
    raise RuntimeError("unreachable")


def fetch_sdk_raw_observations(station_id: str, start_date: str, end_date: str, *, source: str | None = None) -> list[dict[str, Any]]:
    from mostlyright._exact_fetch import _exact_fetch_observations
    from mostlyright.research import _resolve_station

    info = _resolve_station(station_id)
    return _exact_fetch_observations(info, start_date, end_date, source=source)


def is_transient_observation_error(exc: Exception) -> bool:
    text = str(exc).lower()
    transient_markers = [
        "429",
        "too many requests",
        "timeout",
        "timed out",
        "connection reset",
        "temporarily unavailable",
        "503",
        "502",
        "504",
    ]
    return any(marker in text for marker in transient_markers)


def summarize_current_observations(
    raw_rows: list[dict[str, Any]],
    *,
    station_id: str,
    station_name: str,
    airport_name: str,
    timezone: str,
    contract_dates: Iterable[str],
    timing_mode: str = TIMING_MODE_SAME_DAY_11AM,
    as_of_hour_local: int = 11,
    source_filter: str | None = None,
) -> list[dict[str, Any]]:
    frame = pd.DataFrame(raw_rows)
    if frame.empty:
        return [
            unavailable_current_observation_row(
                station_id=station_id,
                station_name=station_name,
                airport_name=airport_name,
                timezone=timezone,
                contract_date=d,
                timing_mode=timing_mode,
                as_of_hour_local=as_of_hour_local,
                reason="SDK returned no raw observations",
            )
            for d in contract_dates
        ]
    frame["observed_at_utc"] = pd.to_datetime(frame.get("observed_at"), errors="coerce", utc=True)
    frame = frame.dropna(subset=["observed_at_utc"]).copy()
    if source_filter and "source" in frame:
        frame = frame.loc[frame["source"].astype(str).str.lower().eq(source_filter.lower())].copy()
    tz = ZoneInfo(timezone)
    frame["observed_at_local"] = frame["observed_at_utc"].dt.tz_convert(tz)
    out: list[dict[str, Any]] = []
    for contract_date in contract_dates:
        out.append(
            summarize_current_observation_for_date(
                frame,
                station_id=station_id,
                station_name=station_name,
                airport_name=airport_name,
                timezone=timezone,
                contract_date=contract_date,
                timing_mode=timing_mode,
                as_of_hour_local=as_of_hour_local,
            )
        )
    return out


def summarize_current_observation_for_date(
    frame: pd.DataFrame,
    *,
    station_id: str,
    station_name: str,
    airport_name: str,
    timezone: str,
    contract_date: str,
    timing_mode: str = TIMING_MODE_SAME_DAY_11AM,
    as_of_hour_local: int = 11,
) -> dict[str, Any]:
    as_of_utc = local_datetime_utc(contract_date, timezone, as_of_hour_local)
    day = date.fromisoformat(contract_date[:10])
    tz = ZoneInfo(timezone)
    day_start_utc = datetime.combine(day, datetime.min.time(), tzinfo=tz).astimezone(UTC)
    candidates = frame.loc[
        (frame["observed_at_utc"] >= pd.Timestamp(day_start_utc))
        & (frame["observed_at_utc"] <= pd.Timestamp(as_of_utc))
        & (frame["observed_at_local"].dt.date == day)
    ].copy()
    if candidates.empty:
        return unavailable_current_observation_row(
            station_id=station_id,
            station_name=station_name,
            airport_name=airport_name,
            timezone=timezone,
            contract_date=contract_date,
            timing_mode=timing_mode,
            as_of_hour_local=as_of_hour_local,
            reason="No SDK observation at or before local as-of time",
        )
    row = candidates.sort_values(["observed_at_utc", "source"], na_position="first").iloc[-1]
    observed_at_utc = pd.Timestamp(row["observed_at_utc"]).to_pydatetime()
    observed_at_local = observed_at_utc.astimezone(tz)
    age_minutes = (as_of_utc - observed_at_utc).total_seconds() / 60
    temp_f = _number(row.get("temp_f"))
    dewpoint_f = _number(row.get("dewpoint_f"))
    humidity = _relative_humidity_pct(temp_f, dewpoint_f)
    wind_speed_mph = _kt_to_mph(row.get("wind_speed_kt"))
    wind_gust_mph = _kt_to_mph(row.get("wind_gust_kt"))
    peak_wind_gust_mph = _kt_to_mph(row.get("peak_wind_gust_kt"))
    pressure_hpa, pressure_source = _pressure_hpa(row)
    return {
        "station_id": station_id,
        "station_name": station_name,
        "airport_name": airport_name,
        "contract_date": contract_date[:10],
        "timing_mode": timing_mode,
        "observed_temp_at_as_of_f": temp_f,
        "observed_dewpoint_at_as_of_f": dewpoint_f,
        "observed_humidity_at_as_of": humidity,
        "observed_wind_speed_at_as_of": wind_speed_mph,
        "observed_wind_direction_at_as_of": _number(row.get("wind_dir_degrees")),
        "observed_wind_gust_at_as_of": wind_gust_mph,
        "observed_peak_wind_gust_at_as_of": peak_wind_gust_mph,
        "observed_peak_wind_direction_at_as_of": _number(row.get("peak_wind_dir")),
        "observed_peak_wind_time_utc": _clean_text(row.get("peak_wind_time")),
        "observed_pressure_at_as_of": pressure_hpa,
        "observed_pressure_source": pressure_source,
        "observed_altimeter_inhg_at_as_of": _number(row.get("altimeter_inhg")),
        "observed_sea_level_pressure_mb_at_as_of": _number(row.get("sea_level_pressure_mb")),
        "observed_visibility_at_as_of": _number(row.get("visibility_miles")),
        "observed_ceiling_at_as_of": _ceiling_ft(row),
        "observed_cloud_cover_at_as_of": _cloud_cover_pct(row),
        "observed_weather_code_at_as_of": _clean_text(row.get("weather_codes")),
        "observed_precip_recent_at_as_of": _number(row.get("precip_1hr_inches")),
        "observed_snow_depth_at_as_of": _number(row.get("snow_depth_inches")),
        "observed_as_of_time_local": observed_at_local.isoformat(),
        "observed_as_of_time_utc": observed_at_utc.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "observed_as_of_age_minutes": float(age_minutes),
        "observed_source": _clean_text(row.get("source")),
        "observed_observation_type": _clean_text(row.get("observation_type")),
        "observed_qc_field": _clean_text(row.get("qc_field")),
        "observed_raw_metar": _clean_text(row.get("raw_metar")),
        "observed_data_source": "mostlyright._exact_fetch._exact_fetch_observations",
        "observed_fetch_status": "ok",
        "observed_unavailable_reason": pd.NA,
    }


def unavailable_current_observation_row(
    *,
    station_id: str,
    station_name: str,
    airport_name: str,
    timezone: str,
    contract_date: str,
    timing_mode: str,
    as_of_hour_local: int,
    reason: str,
) -> dict[str, Any]:
    as_of_utc = local_datetime_utc(contract_date, timezone, as_of_hour_local)
    return {
        "station_id": station_id,
        "station_name": station_name,
        "airport_name": airport_name,
        "contract_date": contract_date[:10],
        "timing_mode": timing_mode,
        "observed_temp_at_as_of_f": pd.NA,
        "observed_dewpoint_at_as_of_f": pd.NA,
        "observed_humidity_at_as_of": pd.NA,
        "observed_wind_speed_at_as_of": pd.NA,
        "observed_wind_direction_at_as_of": pd.NA,
        "observed_wind_gust_at_as_of": pd.NA,
        "observed_peak_wind_gust_at_as_of": pd.NA,
        "observed_peak_wind_direction_at_as_of": pd.NA,
        "observed_peak_wind_time_utc": pd.NA,
        "observed_pressure_at_as_of": pd.NA,
        "observed_pressure_source": pd.NA,
        "observed_altimeter_inhg_at_as_of": pd.NA,
        "observed_sea_level_pressure_mb_at_as_of": pd.NA,
        "observed_visibility_at_as_of": pd.NA,
        "observed_ceiling_at_as_of": pd.NA,
        "observed_cloud_cover_at_as_of": pd.NA,
        "observed_weather_code_at_as_of": pd.NA,
        "observed_precip_recent_at_as_of": pd.NA,
        "observed_snow_depth_at_as_of": pd.NA,
        "observed_as_of_time_local": pd.NA,
        "observed_as_of_time_utc": as_of_utc.isoformat().replace("+00:00", "Z"),
        "observed_as_of_age_minutes": pd.NA,
        "observed_source": pd.NA,
        "observed_observation_type": pd.NA,
        "observed_qc_field": pd.NA,
        "observed_raw_metar": pd.NA,
        "observed_data_source": "mostlyright._exact_fetch._exact_fetch_observations",
        "observed_fetch_status": "unavailable",
        "observed_unavailable_reason": reason,
    }


def _station_meta(station_id: str) -> dict[str, Any]:
    station_id = station_id.upper()
    if station_id not in STATION_METADATA:
        raise ValueError(f"Unknown station {station_id!r}")
    return STATION_METADATA[station_id]


def _completed_keys(frame: pd.DataFrame, *, retry_unavailable: bool) -> set[tuple[str, str, str]]:
    if frame.empty or not set(CACHE_KEYS).issubset(frame.columns):
        return set()
    subset = frame.copy()
    if retry_unavailable and "observed_fetch_status" in subset:
        subset = subset.loc[subset["observed_fetch_status"].astype(str).str.lower().eq("ok")].copy()
    return {
        (str(row.station_id).upper(), str(row.contract_date)[:10], str(row.timing_mode))
        for row in subset.itertuples(index=False)
    }


def _append_cache(path: Path, existing: pd.DataFrame, rows: list[dict[str, Any]]) -> pd.DataFrame:
    fresh = pd.DataFrame(rows)
    combined = pd.concat([existing, fresh], ignore_index=True) if not existing.empty else fresh
    for key in CACHE_KEYS:
        if key not in combined:
            combined[key] = pd.NA
    combined = combined.drop_duplicates(subset=CACHE_KEYS, keep="last")
    combined = combined.sort_values(["contract_date", "station_id", "timing_mode"]).reset_index(drop=True)
    combined.to_csv(path, index=False)
    return combined


def _load_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _chunks(values: list[str], chunk_days: int) -> list[list[str]]:
    size = max(int(chunk_days), 1)
    return [values[i : i + size] for i in range(0, len(values), size)]


def _number(value: Any) -> float | Any:
    if value is None or value is pd.NA:
        return pd.NA
    try:
        number = float(value)
    except (TypeError, ValueError):
        return pd.NA
    if math.isnan(number):
        return pd.NA
    return number


def _clean_text(value: Any) -> str | Any:
    if value is None or value is pd.NA:
        return pd.NA
    if isinstance(value, float) and math.isnan(value):
        return pd.NA
    text = str(value).strip()
    return text if text else pd.NA


def _kt_to_mph(value: Any) -> float | Any:
    number = _number(value)
    return pd.NA if pd.isna(number) else float(number) * 1.150779448


def _relative_humidity_pct(temp_f: Any, dewpoint_f: Any) -> float | Any:
    temp = _number(temp_f)
    dew = _number(dewpoint_f)
    if pd.isna(temp) or pd.isna(dew):
        return pd.NA
    temp_c = (float(temp) - 32) * 5 / 9
    dew_c = (float(dew) - 32) * 5 / 9
    es = math.exp((17.625 * temp_c) / (243.04 + temp_c))
    e = math.exp((17.625 * dew_c) / (243.04 + dew_c))
    return float(np.clip((e / es) * 100, 0, 100))


def _pressure_hpa(row: pd.Series) -> tuple[float | Any, str | Any]:
    sea_level = _number(row.get("sea_level_pressure_mb"))
    if not pd.isna(sea_level):
        return sea_level, "sea_level_pressure_mb"
    altimeter = _number(row.get("altimeter_inhg"))
    if not pd.isna(altimeter):
        return float(altimeter) * 33.8638866667, "altimeter_inhg_converted_to_hpa"
    return pd.NA, pd.NA


def _ceiling_ft(row: pd.Series) -> float | Any:
    bases: list[float] = []
    for idx in range(1, 5):
        cover = str(row.get(f"sky_cover_{idx}") or "").upper()
        base = _number(row.get(f"sky_base_{idx}_ft"))
        if cover in {"BKN", "OVC", "VV"} and not pd.isna(base):
            bases.append(float(base))
    return pd.NA if not bases else float(min(bases))


def _cloud_cover_pct(row: pd.Series) -> float | Any:
    mapping = {
        "CLR": 0.0,
        "SKC": 0.0,
        "NSC": 0.0,
        "FEW": 12.5,
        "SCT": 37.5,
        "BKN": 75.0,
        "OVC": 100.0,
        "VV": 100.0,
    }
    values: list[float] = []
    for idx in range(1, 5):
        cover = str(row.get(f"sky_cover_{idx}") or "").upper()
        if cover in mapping:
            values.append(mapping[cover])
    return pd.NA if not values else float(max(values))
