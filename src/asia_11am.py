from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, time as datetime_time, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from .current_observations import (
    summarize_current_observations,
    unavailable_current_observation_row,
)
from .direct_nwp_fetch import (
    DIRECT_NWP_FEATURES,
    _byte_ranges_for_patterns,
    _ensure_ecmwflibs_available,
    _first_data_var,
    _get_with_retries,
    _is_complete_grib2,
    _point_value_bilinear,
    direct_nwp_file_url,
    extract_direct_nwp_run_feature_points,
)
DEFAULT_DATA_ROOT = Path("data/calibration/asia_11am")
START_DATE = date(2022, 7, 3)
AS_OF_HOUR_LOCAL = 11
LIVE_DELAY_MINUTES = 10
TIMING_MODE = "asia_same_day_11am_live_safe"
GFS_FORECAST_HOURS = tuple(range(8, 21))
GEFS_TEMP_FORECAST_HOURS = (6, 9, 12, 15, 18, 21)
GEFS_TMAX_FORECAST_HOURS = (9, 12, 15, 18, 21)
GEFS_MEMBERS = ("c00", *(f"p{number:02d}" for number in range(1, 31)))
GFS_ARCHIVE = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
GEFS_ARCHIVE = "https://noaa-gefs-pds.s3.amazonaws.com"
_GEFS_CFGRIB_LOCK = Lock()
JMA_HISTORY_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
IEM_ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
IEM_FIELDS = (
    "tmpf", "dwpf", "drct", "sknt", "p01i", "alti", "mslp", "vsby",
    "gust", "skyc1", "skyc2", "skyc3", "skyc4", "skyl1", "skyl2",
    "skyl3", "skyl4", "wxcodes", "peak_wind_gust", "peak_wind_drct",
    "peak_wind_time", "metar",
)
IEM_CLEAR_WEATHER_SENTINEL = "NONE"
_METAR_PRESENT_WEATHER_TOKEN = re.compile(
    r"^(?:[+-]|VC)?(?:MI|BC|PR|DR|BL|SH|TS|FZ)?"
    r"(?:DZ|RA|SN|SG|IC|PL|GR|GS|UP|BR|FG|FU|VA|DU|SA|HZ|PO|SQ|FC|SS|DS)+$"
)

# Keep the same deterministic field contract as the established 11AM
# pipeline. Additional direct-NWP variables remain available through
# ``direct_nwp_fetch`` without making every historical pull request them.
GFS_FIELDS = (
    "temp_k_2m",
    "dewpoint_k_2m",
    "relative_humidity_pct_2m",
    "wind_u_ms_10m",
    "wind_v_ms_10m",
    "wind_gust_ms",
    "precip_mm_1h",
    "cloud_cover_pct",
)
JMA_BASE_FIELDS = (
    "temperature_2m",
    "dew_point_2m",
    "relative_humidity_2m",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
)

# Open-Meteo exposes the JMA MSM previous-day values used by the live Tokyo
# contract, but currently reports the gust variable as ``undefined``.  Keep
# requesting and normalizing it so the feature schema remains stable, while
# requiring only the variables that this endpoint actually supplies.
JMA_REQUIRED_LIVE_FIELDS = tuple(
    field for field in JMA_BASE_FIELDS if field != "wind_gusts_10m"
)


def _json_default(value: Any) -> Any:
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return value.isoformat()
    if value is pd.NA or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: Mapping[str, Any] | Sequence[Any]) -> None:
    _atomic_write_text(
        path, json.dumps(payload, indent=2, sort_keys=True, default=_json_default)
    )


def _atomic_write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if path.suffix.lower() == ".parquet":
        frame.to_parquet(temporary, index=False)
    else:
        frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _merge_normalized_month(
    path: Path,
    frame: pd.DataFrame,
    *,
    keys: Sequence[str],
) -> pd.DataFrame:
    """Merge an incremental slice into an existing normalized month."""
    parts = [frame]
    if path.exists():
        parts.insert(0, pd.read_parquet(path))
    merged = pd.concat(parts, ignore_index=True)
    present_keys = [key for key in keys if key in merged.columns]
    if present_keys:
        merged = merged.drop_duplicates(present_keys, keep="last")
        merged = merged.sort_values(present_keys)
    return merged.reset_index(drop=True)


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _request(
    url: str,
    *,
    params: Mapping[str, Any] | Sequence[tuple[str, Any]] | None = None,
    timeout: int = 90,
    attempts: int = 6,
) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=timeout,
                headers={"User-Agent": "weather-research-asia-11am/0.1"},
            )
            if response.status_code == 429 or response.status_code >= 500:
                retry_after = response.headers.get("Retry-After")
                delay = (
                    float(retry_after)
                    if retry_after and retry_after.isdigit()
                    else min(60.0, 2**attempt)
                )
                time.sleep(delay + random.random())
                continue
            response.raise_for_status()
            return response
        except requests.HTTPError as exc:
            if exc.response is not None and 400 <= exc.response.status_code < 500:
                raise
            last_error = exc
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_error = exc
        if attempt < attempts:
            time.sleep(min(60.0, 2**attempt) + random.random())
    raise RuntimeError(f"GET failed after {attempts} attempts: {url}: {last_error}") from last_error


def _content_addressed_raw_path(
    directory: Path, stem: str, content: bytes, suffix: str
) -> Path:
    checksum = _sha256_bytes(content)
    path = directory / f"{stem}_{checksum[:12]}{suffix}"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_bytes(content)
        os.replace(temporary, path)
    return path


@dataclass(frozen=True)
class AsiaCityProfile:
    city_id: str
    city_name: str
    station_id: str
    station_name: str
    latitude: float
    longitude: float
    elevation_m: float
    timezone: str
    wunderground_country: str
    wunderground_units: str = "m"
    settlement_unit: str = "C"
    as_of_hour_local: int = AS_OF_HOUR_LOCAL
    live_delay_minutes: int = LIVE_DELAY_MINUTES


CITY_PROFILES: dict[str, AsiaCityProfile] = {
    "busan": AsiaCityProfile(
        city_id="busan",
        city_name="Busan",
        station_id="RKPK",
        station_name="Gimhae International Airport",
        latitude=35.1795,
        longitude=128.9382,
        elevation_m=2.0,
        timezone="Asia/Seoul",
        wunderground_country="KR",
    ),
    "seoul": AsiaCityProfile(
        city_id="seoul",
        city_name="Seoul",
        station_id="RKSI",
        station_name="Incheon International Airport",
        latitude=37.469,
        longitude=126.451,
        elevation_m=7.0,
        timezone="Asia/Seoul",
        wunderground_country="KR",
    ),
    "tokyo": AsiaCityProfile(
        city_id="tokyo",
        city_name="Tokyo",
        station_id="RJTT",
        station_name="Tokyo Haneda Airport",
        latitude=35.553,
        longitude=139.781,
        elevation_m=5.0,
        timezone="Asia/Tokyo",
        wunderground_country="JP",
    ),
}


def resolve_profiles(cities: Iterable[str] | None = None) -> tuple[AsiaCityProfile, ...]:
    requested = [str(city).strip().lower() for city in (cities or CITY_PROFILES) if str(city).strip()]
    unknown = sorted(set(requested) - set(CITY_PROFILES))
    if unknown:
        raise ValueError(f"Unknown Asia city profile(s): {', '.join(unknown)}")
    return tuple(CITY_PROFILES[city] for city in dict.fromkeys(requested))


def gfs_day_workers(total_workers: int) -> int:
    """Map the shared worker budget to concurrent GFS target days.

    Each target day already runs four indexed-range download workers inside
    ``direct_nwp_fetch``. Cap day-level concurrency at four to avoid excessive
    simultaneous cfgrib datasets and memory pressure.
    """
    return max(1, min(4, int(total_workers) // 4))


def latest_completed_local_day(profiles: Sequence[AsiaCityProfile] | None = None) -> date:
    selected = tuple(profiles or CITY_PROFILES.values())
    return min(datetime.now(ZoneInfo(profile.timezone)).date() - timedelta(days=1) for profile in selected)


def resolve_date_bounds(
    start_date: date | str = START_DATE,
    end_date: date | str | None = None,
    *,
    profiles: Sequence[AsiaCityProfile] | None = None,
) -> tuple[date, date]:
    start = _as_date(start_date)
    end = _as_date(end_date) if end_date is not None else latest_completed_local_day(profiles)
    if start < START_DATE:
        raise ValueError(f"Asia common-provider history starts on {START_DATE}")
    if end < start:
        raise ValueError("end_date must be on or after start_date")
    return start, end


def target_dates(start_date: date, end_date: date) -> list[date]:
    return [stamp.date() for stamp in pd.date_range(start_date, end_date, freq="D")]


def month_keys(start_date: date, end_date: date) -> list[str]:
    return pd.period_range(start_date, end_date, freq="M").astype(str).tolist()


def month_bounds(month_key: str, start_date: date, end_date: date) -> tuple[date, date]:
    month_start = date.fromisoformat(f"{month_key}-01")
    next_month = (month_start.replace(day=28) + timedelta(days=4)).replace(day=1)
    return max(start_date, month_start), min(end_date, next_month - timedelta(days=1))


def forecast_timing(contract_date: date | str) -> dict[str, Any]:
    day = _as_date(contract_date)
    issue = datetime.combine(day - timedelta(days=1), datetime_time(18), tzinfo=UTC)
    as_of = datetime.combine(day, datetime_time(2), tzinfo=UTC)
    return {
        "contract_date": day.isoformat(),
        "issue_utc": issue,
        "as_of_utc": as_of,
        "window_start_utc": as_of,
        "window_end_utc": datetime.combine(day, datetime_time(14), tzinfo=UTC),
        "gfs_forecast_hours": GFS_FORECAST_HOURS,
        "gefs_temp_forecast_hours": GEFS_TEMP_FORECAST_HOURS,
        "gefs_tmax_forecast_hours": GEFS_TMAX_FORECAST_HOURS,
    }


def write_profile(data_root: Path, profiles: Sequence[AsiaCityProfile]) -> Path:
    path = data_root / "profile.json"
    _atomic_write_json(
        path,
        {
            "start_date": START_DATE,
            "timing_mode": TIMING_MODE,
            "gfs_issue_rule": "previous_calendar_day_18z",
            "cities": [asdict(profile) for profile in profiles],
        },
    )
    return path


def run_settlement_backfill(
    data_root: Path,
    profiles: Sequence[AsiaCityProfile],
    start_date: date,
    end_date: date,
    *,
    api_key: str | None = None,
    force: bool = False,
    workers: int = 4,
) -> dict[str, Any]:
    from .wunderground_history import backfill_wunderground_station_history

    output = data_root / "normalized" / "settlements" / "settlement_actual_highs.csv"
    frame = backfill_wunderground_station_history(
        output,
        stations=[profile.station_id for profile in profiles],
        station_timezones={profile.station_id: profile.timezone for profile in profiles},
        station_countries={
            profile.station_id: profile.wunderground_country for profile in profiles
        },
        station_units={profile.station_id: profile.wunderground_units for profile in profiles},
        station_slugs={profile.station_id: profile.city_id for profile in profiles},
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        api_key=api_key,
        force_refresh=force,
        workers=max(1, int(workers)),
    )
    requested = set(day.isoformat() for day in target_dates(start_date, end_date))
    results: list[dict[str, Any]] = []
    for profile in profiles:
        city = frame.loc[
            frame["station_id"].eq(profile.station_id)
            & frame["contract_date"].astype(str).isin(requested)
        ].copy()
        city["city_id"] = profile.city_id
        for month_key in month_keys(start_date, end_date):
            month_start, month_end = month_bounds(month_key, start_date, end_date)
            expected = {day.isoformat() for day in target_dates(month_start, month_end)}
            part = city.loc[city["contract_date"].astype(str).isin(expected)].copy()
            part = part.sort_values("contract_date").reset_index(drop=True)
            path = (
                data_root
                / "normalized"
                / "settlements"
                / profile.city_id
                / f"{month_key}.parquet"
            )
            part = _merge_normalized_month(path, part, keys=("contract_date",))
            _atomic_write_frame(path, part)
            ok = part["quality_flag"].astype(str).eq("ok") if not part.empty else pd.Series(dtype=bool)
            manifest = {
                "city_id": profile.city_id,
                "station_id": profile.station_id,
                "provider": "wunderground",
                "month": month_key,
                "status": "complete" if expected.issubset(set(part["contract_date"].astype(str))) else "incomplete",
                "requested_start": month_start,
                "requested_end": month_end,
                "row_count": len(part),
                "ok_count": int(ok.sum()),
                "normalized_path": str(path),
                "sha256": _sha256_file(path),
                "updated_at_utc": datetime.now(UTC),
            }
            _atomic_write_json(
                data_root
                / "manifests"
                / "settlements"
                / profile.city_id
                / f"{month_key}.json",
                manifest,
            )
            results.append(manifest)
    return {"status": _combined_status(results), "results": results, "output": str(output)}


def _iem_params(
    profile: AsiaCityProfile,
    start_date: date,
    end_date: date,
) -> list[tuple[str, Any]]:
    exclusive_end = end_date + timedelta(days=1)
    params: list[tuple[str, Any]] = [("station", profile.station_id)]
    params.extend(("data", field) for field in IEM_FIELDS)
    params.extend(
        [
            ("year1", start_date.year),
            ("month1", start_date.month),
            ("day1", start_date.day),
            ("year2", exclusive_end.year),
            ("month2", exclusive_end.month),
            ("day2", exclusive_end.day),
            ("tz", profile.timezone),
            ("format", "onlycomma"),
            ("latlon", "yes"),
            ("elev", "yes"),
            ("missing", "null"),
            ("trace", "null"),
            ("direct", "yes"),
            ("report_type", "1"),
            ("report_type", "2"),
        ]
    )
    return params


def _iem_weather_code(
    value: Any,
    *,
    raw_metar: Any,
    has_wxcodes_field: bool,
    profile: AsiaCityProfile,
) -> str | Any:
    """Keep IEM's present-weather field source-truthful.

    IEM publishes an empty ``wxcodes`` cell for a reported METAR with no
    present-weather group.  That is distinct from a missing field or a raw
    report whose weather group disagrees with the blank cell.  Only the former
    becomes the explicit categorical ``NONE`` sentinel; it never supplies a
    precipitation amount or another measurement.
    """
    if isinstance(value, str) and value.strip():
        return value.strip()
    raw = raw_metar.strip() if isinstance(raw_metar, str) else ""
    tokens = raw.split()
    valid_station_report = (
        profile.station_id in tokens
        and any(re.fullmatch(r"\d{6}Z", token) for token in tokens)
    )
    raw_has_weather = any(_METAR_PRESENT_WEATHER_TOKEN.fullmatch(token) for token in tokens)
    if has_wxcodes_field and valid_station_report and not raw_has_weather:
        return IEM_CLEAR_WEATHER_SENTINEL
    return pd.NA


def _iem_rows(content: bytes, profile: AsiaCityProfile) -> list[dict[str, Any]]:
    frame = pd.read_csv(io.BytesIO(content), na_values=["null", "M", ""], low_memory=False)
    has_wxcodes_field = "wxcodes" in frame.columns
    rows: list[dict[str, Any]] = []
    for item in frame.to_dict(orient="records"):
        valid = pd.to_datetime(item.get("valid"), errors="coerce")
        if pd.isna(valid):
            continue
        if valid.tzinfo is None:
            valid = valid.tz_localize(profile.timezone)
        raw_metar = item.get("metar")
        rows.append(
            {
                "observed_at": valid.tz_convert(UTC).isoformat(),
                "temp_f": item.get("tmpf"),
                "dewpoint_f": item.get("dwpf"),
                "wind_dir_degrees": item.get("drct"),
                "wind_speed_kt": item.get("sknt"),
                "wind_gust_kt": item.get("gust"),
                "peak_wind_gust_kt": item.get("peak_wind_gust"),
                "peak_wind_dir": item.get("peak_wind_drct"),
                "peak_wind_time": item.get("peak_wind_time"),
                "altimeter_inhg": item.get("alti"),
                "sea_level_pressure_mb": item.get("mslp"),
                "visibility_miles": item.get("vsby"),
                "weather_codes": _iem_weather_code(
                    item.get("wxcodes"),
                    raw_metar=raw_metar,
                    has_wxcodes_field=has_wxcodes_field,
                    profile=profile,
                ),
                # IEM's p01i is the observed one-hour liquid precipitation.
                # Preserve it exactly; weather codes are not a substitute amount.
                "precip_1hr_inches": item.get("p01i"),
                "sky_cover_1": item.get("skyc1"),
                "sky_cover_2": item.get("skyc2"),
                "sky_cover_3": item.get("skyc3"),
                "sky_cover_4": item.get("skyc4"),
                "sky_base_1_ft": item.get("skyl1"),
                "sky_base_2_ft": item.get("skyl2"),
                "sky_base_3_ft": item.get("skyl3"),
                "sky_base_4_ft": item.get("skyl4"),
                "raw_metar": raw_metar,
                "source": "iem_asos_global_metar",
                "observation_type": "METAR",
                "qc_field": "iem_as_is_archive",
            }
        )
    return rows


def normalize_metar_rows(
    raw_rows: list[dict[str, Any]],
    profile: AsiaCityProfile,
    dates: Sequence[date],
    *,
    source_filter: str | None,
    data_source: str,
) -> pd.DataFrame:
    summarized = summarize_current_observations(
        raw_rows,
        station_id=profile.station_id,
        station_name=profile.station_name,
        airport_name=profile.station_name,
        timezone=profile.timezone,
        contract_dates=[day.isoformat() for day in dates],
        timing_mode="same_day_11am",
        as_of_hour_local=profile.as_of_hour_local,
        source_filter=source_filter,
    )
    rows: list[dict[str, Any]] = []
    for row in summarized:
        row["city_id"] = profile.city_id
        row["timing_mode"] = TIMING_MODE
        precip = pd.to_numeric(
            pd.Series([row.get("observed_precip_recent_at_as_of")]), errors="coerce"
        ).iloc[0]
        row["observed_precip_amount_available"] = bool(pd.notna(precip))
        row["observed_data_source"] = data_source
        row["observed_temp_at_as_of_c"] = _f_to_c(row.get("observed_temp_at_as_of_f"))
        row["observed_high_temp_through_as_of_c"] = _f_to_c(
            row.get("observed_high_temp_through_as_of_f")
        )
        row["observed_dewpoint_at_as_of_c"] = _f_to_c(
            row.get("observed_dewpoint_at_as_of_f")
        )
        age = pd.to_numeric(
            pd.Series([row.get("observed_as_of_age_minutes")]), errors="coerce"
        ).iloc[0]
        if (
            row.get("observed_fetch_status") == "ok"
            and (pd.isna(age) or float(age) < 0 or float(age) > 60)
        ):
            unavailable = unavailable_current_observation_row(
                station_id=profile.station_id,
                station_name=profile.station_name,
                airport_name=profile.station_name,
                timezone=profile.timezone,
                contract_date=str(row["contract_date"]),
                timing_mode=TIMING_MODE,
                as_of_hour_local=profile.as_of_hour_local,
                reason=f"Latest METAR age exceeds 60 minutes: {age}",
            )
            unavailable.update(
                {
                    "city_id": profile.city_id,
                    "observed_precip_amount_available": False,
                    "observed_data_source": data_source,
                    "observed_temp_at_as_of_c": pd.NA,
                    "observed_high_temp_through_as_of_c": pd.NA,
                    "observed_dewpoint_at_as_of_c": pd.NA,
                }
            )
            row = unavailable
        rows.append(row)
    return pd.DataFrame(rows).sort_values("contract_date").reset_index(drop=True)


def backfill_observation_month(
    data_root: Path,
    profile: AsiaCityProfile,
    month_key: str,
    start_date: date,
    end_date: date,
    *,
    force: bool = False,
) -> dict[str, Any]:
    start, end = month_bounds(month_key, start_date, end_date)
    normalized_path = (
        data_root / "normalized" / "observations" / profile.city_id / f"{month_key}.parquet"
    )
    manifest_path = (
        data_root / "manifests" / "observations" / profile.city_id / f"{month_key}.json"
    )
    cached = _complete_manifest(manifest_path, normalized_path, force=force)
    if cached is not None:
        return cached
    response = _request(IEM_ASOS_URL, params=_iem_params(profile, start, end), timeout=180)
    content = response.content
    raw_path = _content_addressed_raw_path(
        data_root / "raw" / "iem_metar" / profile.city_id,
        f"{profile.station_id}_{month_key}",
        content,
        ".csv",
    )
    raw_rows = _iem_rows(content, profile)
    frame = normalize_metar_rows(
        raw_rows,
        profile,
        target_dates(start, end),
        source_filter="iem_asos_global_metar",
        data_source="iem_asos_global_metar_raw",
    )
    full_day_highs = _raw_metar_daily_highs(raw_rows, profile.timezone)
    frame["iem_daily_high_f"] = frame["contract_date"].map(
        {day: values["high_f"] for day, values in full_day_highs.items()}
    )
    frame["iem_daily_high_c"] = frame["contract_date"].map(
        {day: values["high_c"] for day, values in full_day_highs.items()}
    )
    frame["source_uri"] = response.url
    frame["source_checksum"] = _sha256_bytes(content)
    frame = _merge_normalized_month(normalized_path, frame, keys=("contract_date",))
    _atomic_write_frame(normalized_path, frame)
    ok_count = int(frame["observed_fetch_status"].astype(str).eq("ok").sum())
    manifest = {
        "city_id": profile.city_id,
        "station_id": profile.station_id,
        "provider": "iem",
        "month": month_key,
        "status": (
            "complete"
            if {day.isoformat() for day in target_dates(start, end)}.issubset(
                set(frame["contract_date"].astype(str))
            )
            else "incomplete"
        ),
        "requested_start": start,
        "requested_end": end,
        "row_count": len(frame),
        "ok_count": ok_count,
        "raw_path": str(raw_path),
        "normalized_path": str(normalized_path),
        "sha256": _sha256_file(normalized_path),
        "updated_at_utc": datetime.now(UTC),
    }
    _atomic_write_json(manifest_path, manifest)
    return manifest


def _forecast_points(profiles: Sequence[AsiaCityProfile]) -> dict[str, dict[str, float]]:
    return {
        profile.city_id: {"lat": profile.latitude, "lon": profile.longitude}
        for profile in profiles
    }


def _gfs_rows(
    contract_date: date,
    profiles: Sequence[AsiaCityProfile],
    values: Mapping[str, Mapping[int, Mapping[str, float]]],
) -> dict[str, pd.DataFrame]:
    timing = forecast_timing(contract_date)
    issue = timing["issue_utc"]
    result: dict[str, pd.DataFrame] = {}
    for profile in profiles:
        rows: list[dict[str, Any]] = []
        city_values = values.get(profile.city_id, {})
        for fxx in GFS_FORECAST_HOURS:
            fields = dict(city_values.get(fxx, {}))
            valid_utc = issue + timedelta(hours=fxx)
            row: dict[str, Any] = {
                "city_id": profile.city_id,
                "station_id": profile.station_id,
                "provider": "gfs",
                "lineage": "gfs_noaa_aws_previous_day_18z",
                "contract_date": contract_date.isoformat(),
                "issued_at_utc": _iso_z(issue),
                "forecast_as_of_utc": _iso_z(timing["as_of_utc"]),
                "valid_time_utc": _iso_z(valid_utc),
                "valid_time_local": valid_utc.astimezone(ZoneInfo(profile.timezone)).isoformat(),
                "forecast_hour": fxx,
                "member_id": "deterministic",
                "source_url": direct_nwp_file_url("gfs", issue, fxx),
                "fetch_status": "ok" if "temp_k_2m" in fields else "incomplete",
            }
            for field, value in fields.items():
                if field in {"temp_k_2m", "dewpoint_k_2m", "temp_k_925mb", "temp_k_850mb"}:
                    row[field.replace("_k_", "_c_")] = float(value) - 273.15
                else:
                    row[field] = value
            rows.append(row)
        result[profile.city_id] = pd.DataFrame(rows)
    return result


def backfill_gfs_day(
    data_root: Path,
    profiles: Sequence[AsiaCityProfile],
    contract_date: date,
    *,
    force: bool = False,
) -> list[dict[str, Any]]:
    pending: list[AsiaCityProfile] = []
    existing_results: list[dict[str, Any]] = []
    for profile in profiles:
        path = (
            data_root
            / "normalized"
            / "forecasts"
            / "gfs"
            / profile.city_id
            / contract_date.strftime("%Y-%m")
            / f"{contract_date}.parquet"
        )
        if path.exists() and not force:
            frame = pd.read_parquet(path)
            if (
                len(frame) == len(GFS_FORECAST_HOURS)
                and frame["fetch_status"].astype(str).eq("ok").all()
            ):
                existing_results.append(
                    {
                        "city_id": profile.city_id,
                        "contract_date": contract_date.isoformat(),
                        "status": "complete",
                        "normalized_path": str(path),
                    }
                )
                continue
        pending.append(profile)
    if not pending:
        return existing_results

    issue = forecast_timing(contract_date)["issue_utc"]
    try:
        values = extract_direct_nwp_run_feature_points(
            _forecast_points(pending),
            "gfs",
            data_root / "raw" / "nwp_subsets",
            issue,
            GFS_FORECAST_HOURS,
            force_refresh=force,
            feature_fields=list(GFS_FIELDS),
            interpolation="bilinear",
        )
        frames = _gfs_rows(contract_date, pending, values)
    except Exception as exc:  # noqa: BLE001
        frames = {
            profile.city_id: pd.DataFrame(
                [
                    {
                        "city_id": profile.city_id,
                        "station_id": profile.station_id,
                        "provider": "gfs",
                        "lineage": "gfs_noaa_aws_previous_day_18z",
                        "contract_date": contract_date.isoformat(),
                        "issued_at_utc": _iso_z(issue),
                        "forecast_as_of_utc": _iso_z(
                            forecast_timing(contract_date)["as_of_utc"]
                        ),
                        "fetch_status": "failed",
                        "unavailable_reason": str(exc),
                    }
                ]
            )
            for profile in pending
        }

    results = list(existing_results)
    for profile in pending:
        path = (
            data_root
            / "normalized"
            / "forecasts"
            / "gfs"
            / profile.city_id
            / contract_date.strftime("%Y-%m")
            / f"{contract_date}.parquet"
        )
        frame = frames[profile.city_id]
        _atomic_write_frame(path, frame)
        complete = (
            len(frame) == len(GFS_FORECAST_HOURS)
            and frame["fetch_status"].astype(str).eq("ok").all()
        )
        results.append(
            {
                "city_id": profile.city_id,
                "contract_date": contract_date.isoformat(),
                "status": "complete" if complete else "incomplete",
                "normalized_path": str(path),
            }
        )
    return results


def run_gfs_backfill(
    data_root: Path,
    profiles: Sequence[AsiaCityProfile],
    start_date: date,
    end_date: date,
    *,
    workers: int = 1,
    force: bool = False,
) -> dict[str, Any]:
    def worker(day: date) -> list[dict[str, Any]]:
        os.environ.setdefault("WEATHER_RESEARCH_DIRECT_NWP_WORKERS", "4")
        return backfill_gfs_day(data_root, profiles, day, force=force)

    day_results: list[Any] = []
    manifests: list[dict[str, Any]] = []
    # Compact, checksum, and clean each target month before downloading the
    # next one. Provider-wide cleanup can retain years of temporary GRIB
    # subsets and exhaust local disk during a historical backfill.
    for month_key in month_keys(start_date, end_date):
        month_start, month_end = month_bounds(month_key, start_date, end_date)
        day_results.extend(
            _run_threaded(
                target_dates(month_start, month_end),
                worker,
                workers=max(1, workers),
            )
        )
        manifests.extend(
            _compact_forecast_months(
                data_root,
                profiles,
                "gfs",
                month_start,
                month_end,
                expected_rows_per_day=len(GFS_FORECAST_HOURS),
            )
        )
    return {
        "status": _combined_status(manifests),
        "day_results": [
            item
            for group in day_results
            for item in (group if isinstance(group, list) else [group])
        ],
        "manifests": manifests,
    }


def gefs_file_url(issue_time: datetime, member_id: str, fxx: int) -> str:
    member = str(member_id).lower()
    if member not in GEFS_MEMBERS:
        raise ValueError(f"Unknown GEFS member: {member_id}")
    product = f"ge{member}"
    return (
        f"{GEFS_ARCHIVE}/gefs.{issue_time:%Y%m%d}/{issue_time:%H}/"
        f"atmos/pgrb2sp25/{product}.t{issue_time:%H}z.pgrb2s.0p25.f{fxx:03d}"
    )


def _download_gefs_field(
    data_root: Path,
    issue_time: datetime,
    member_id: str,
    fxx: int,
    field: str,
    *,
    force: bool,
    idx_text: str | None = None,
) -> Path:
    patterns = {
        "temp_2m_c": [":TMP:2 m above ground:"],
        "tmax_3h_c": [":TMAX:2 m above ground:"],
    }[field]
    directory = data_root / "raw" / "nwp_subsets" / "gefs"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (
        f"gefs_{member_id}_{field}_{issue_time:%Y%m%d%H}_f{fxx:03d}.grib2"
    )
    if path.exists() and not force:
        content = path.read_bytes()
        if _is_complete_grib2(content):
            return path
    url = gefs_file_url(issue_time, member_id, fxx)
    idx = (
        idx_text
        if idx_text is not None
        else _get_with_retries(f"{url}.idx", timeout=30).text
    )
    start, end = _byte_ranges_for_patterns(idx, patterns)[0]
    response = _get_with_retries(
        url,
        timeout=90,
        headers={"Range": f"bytes={start}-{end}"},
    )
    content = response.content
    if not _is_complete_grib2(content):
        raise RuntimeError(
            f"Incomplete GEFS GRIB subset for {member_id} {field} f{fxx:03d}"
        )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)
    return path


def _download_gefs_file_fields(
    data_root: Path,
    issue_time: datetime,
    member_id: str,
    fxx: int,
    fields: Sequence[str],
    *,
    force: bool,
) -> dict[str, Path | Exception]:
    """Download all requested fields from one GEFS source file with one index."""
    results: dict[str, Path | Exception] = {}
    pending: list[str] = []
    directory = data_root / "raw" / "nwp_subsets" / "gefs"
    for field in fields:
        path = directory / (
            f"gefs_{member_id}_{field}_{issue_time:%Y%m%d%H}_f{fxx:03d}.grib2"
        )
        if path.exists() and not force:
            try:
                if _is_complete_grib2(path.read_bytes()):
                    results[field] = path
                    continue
            except OSError:
                pass
        pending.append(field)
    if not pending:
        return results

    url = gefs_file_url(issue_time, member_id, fxx)
    try:
        idx_text = _get_with_retries(f"{url}.idx", timeout=30).text
    except Exception as exc:  # noqa: BLE001
        for field in pending:
            results[field] = exc
        return results

    for field in pending:
        try:
            results[field] = _download_gefs_field(
                data_root,
                issue_time,
                member_id,
                fxx,
                field,
                force=force,
                idx_text=idx_text,
            )
        except Exception as exc:  # noqa: BLE001
            results[field] = exc
    return results


def _extract_gefs_points(
    path: Path,
    profiles: Sequence[AsiaCityProfile],
) -> dict[str, float]:
    _ensure_ecmwflibs_available()
    try:
        import eccodes
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("ecCodes dependencies are required for GEFS") from exc

    def bilinear_nearest(
        points: Sequence[Mapping[str, Any]],
        latitude: float,
        longitude: float,
    ) -> float | None:
        cells = {
            (round(float(point["lat"]), 8), round(float(point["lon"]), 8)): float(
                point["value"]
            )
            for point in points
        }
        latitudes = sorted({cell[0] for cell in cells})
        longitudes = sorted({cell[1] for cell in cells})
        if len(latitudes) != 2 or len(longitudes) != 2:
            return float(points[0]["value"]) if points else None
        south, north = latitudes
        west, east = longitudes
        if north == south or east == west:
            return float(points[0]["value"]) if points else None
        required = ((south, west), (south, east), (north, west), (north, east))
        if any(cell not in cells for cell in required):
            return float(points[0]["value"]) if points else None
        x_weight = (longitude - west) / (east - west)
        y_weight = (latitude - south) / (north - south)
        return (
            cells[(south, west)] * (1 - x_weight) * (1 - y_weight)
            + cells[(south, east)] * x_weight * (1 - y_weight)
            + cells[(north, west)] * (1 - x_weight) * y_weight
            + cells[(north, east)] * x_weight * y_weight
        )

    output: dict[str, float] = {}
    handle = None
    with _GEFS_CFGRIB_LOCK:
        try:
            with path.open("rb") as stream:
                handle = eccodes.codes_grib_new_from_file(stream)
                if handle is None:
                    return output
                for profile in profiles:
                    longitude = profile.longitude % 360.0
                    nearest = eccodes.codes_grib_find_nearest(
                        handle,
                        profile.latitude,
                        longitude,
                        npoints=4,
                    )
                    value = bilinear_nearest(
                        nearest,
                        profile.latitude,
                        longitude,
                    )
                    if value is not None and np.isfinite(value):
                        output[profile.city_id] = float(value) - 273.15
        finally:
            if handle is not None:
                eccodes.codes_release(handle)
    return output


def extract_gefs_day(
    data_root: Path,
    profiles: Sequence[AsiaCityProfile],
    contract_date: date,
    *,
    workers: int = 12,
    force: bool = False,
    members: Sequence[str] = GEFS_MEMBERS,
) -> dict[str, dict[str, dict[int, dict[str, float]]]]:
    issue = forecast_timing(contract_date)["issue_utc"]
    tasks: list[tuple[str, int, tuple[str, ...]]] = []
    for member in members:
        for fxx in GEFS_TEMP_FORECAST_HOURS:
            fields = ["temp_2m_c"]
            if fxx in GEFS_TMAX_FORECAST_HOURS:
                fields.append("tmax_3h_c")
            tasks.append((member, fxx, tuple(fields)))

    paths: dict[tuple[str, int, str], Path | Exception] = {}
    with ThreadPoolExecutor(max_workers=max(1, min(int(workers), len(tasks)))) as executor:
        futures = {
            executor.submit(
                _download_gefs_file_fields,
                data_root,
                issue,
                member,
                fxx,
                fields,
                force=force,
            ): (member, fxx)
            for member, fxx, fields in tasks
        }
        for future in as_completed(futures):
            member, fxx = futures[future]
            try:
                field_results = future.result()
                for field, result in field_results.items():
                    paths[(member, fxx, field)] = result
            except Exception as exc:  # noqa: BLE001
                fields = next(
                    task_fields
                    for task_member, task_fxx, task_fields in tasks
                    if task_member == member and task_fxx == fxx
                )
                for field in fields:
                    paths[(member, fxx, field)] = exc

    values: dict[str, dict[str, dict[int, dict[str, float]]]] = {
        profile.city_id: {member: {} for member in members} for profile in profiles
    }
    # ecCodes' parser is not thread-safe in this Windows runtime. Keep decoding
    # serial within the process; run_gefs_backfill overlaps it with downloads
    # for other contract dates instead.
    for (member, fxx, field), path_or_error in paths.items():
        if isinstance(path_or_error, Exception):
            continue
        try:
            point_values = _extract_gefs_points(path_or_error, profiles)
        except Exception:
            continue
        for city_id, value in point_values.items():
            values[city_id][member].setdefault(fxx, {})[field] = value
    return values


def _gefs_rows(
    contract_date: date,
    profile: AsiaCityProfile,
    values: Mapping[str, Mapping[int, Mapping[str, float]]],
    *,
    members: Sequence[str] = GEFS_MEMBERS,
) -> pd.DataFrame:
    timing = forecast_timing(contract_date)
    issue = timing["issue_utc"]
    rows: list[dict[str, Any]] = []
    for member in members:
        member_values = values.get(member, {})
        for fxx in GEFS_TEMP_FORECAST_HOURS:
            fields = member_values.get(fxx, {})
            valid_utc = issue + timedelta(hours=fxx)
            expected_tmax = fxx in GEFS_TMAX_FORECAST_HOURS
            complete = "temp_2m_c" in fields and (
                not expected_tmax or "tmax_3h_c" in fields
            )
            rows.append(
                {
                    "city_id": profile.city_id,
                    "station_id": profile.station_id,
                    "provider": "gefs",
                    "lineage": "gefs_noaa_aws_previous_day_18z",
                    "contract_date": contract_date.isoformat(),
                    "issued_at_utc": _iso_z(issue),
                    "forecast_as_of_utc": _iso_z(timing["as_of_utc"]),
                    "valid_time_utc": _iso_z(valid_utc),
                    "valid_time_local": valid_utc.astimezone(
                        ZoneInfo(profile.timezone)
                    ).isoformat(),
                    "forecast_hour": fxx,
                    "member_id": member,
                    "temp_2m_c": fields.get("temp_2m_c", pd.NA),
                    "tmax_3h_c": fields.get("tmax_3h_c", pd.NA),
                    "tmax_interval_start_utc": (
                        _iso_z(valid_utc - timedelta(hours=3)) if expected_tmax else pd.NA
                    ),
                    "source_url": gefs_file_url(issue, member, fxx),
                    "fetch_status": "ok" if complete else "incomplete",
                }
            )
    return pd.DataFrame(rows)


def backfill_gefs_day(
    data_root: Path,
    profiles: Sequence[AsiaCityProfile],
    contract_date: date,
    *,
    workers: int = 12,
    force: bool = False,
    members: Sequence[str] = GEFS_MEMBERS,
) -> list[dict[str, Any]]:
    expected_rows = len(members) * len(GEFS_TEMP_FORECAST_HOURS)
    pending: list[AsiaCityProfile] = []
    results: list[dict[str, Any]] = []
    for profile in profiles:
        path = (
            data_root
            / "normalized"
            / "forecasts"
            / "gefs"
            / profile.city_id
            / contract_date.strftime("%Y-%m")
            / f"{contract_date}.parquet"
        )
        if path.exists() and not force:
            frame = pd.read_parquet(path)
            if len(frame) == expected_rows and frame["fetch_status"].astype(str).eq("ok").all():
                results.append(
                    {
                        "city_id": profile.city_id,
                        "contract_date": contract_date.isoformat(),
                        "status": "complete",
                        "normalized_path": str(path),
                    }
                )
                continue
        pending.append(profile)
    if not pending:
        return results

    values = extract_gefs_day(
        data_root,
        pending,
        contract_date,
        workers=workers,
        force=force,
        members=members,
    )
    for profile in pending:
        frame = _gefs_rows(
            contract_date,
            profile,
            values.get(profile.city_id, {}),
            members=members,
        )
        path = (
            data_root
            / "normalized"
            / "forecasts"
            / "gefs"
            / profile.city_id
            / contract_date.strftime("%Y-%m")
            / f"{contract_date}.parquet"
        )
        _atomic_write_frame(path, frame)
        complete = len(frame) == expected_rows and frame["fetch_status"].astype(str).eq("ok").all()
        results.append(
            {
                "city_id": profile.city_id,
                "contract_date": contract_date.isoformat(),
                "status": "complete" if complete else "incomplete",
                "normalized_path": str(path),
            }
        )
    return results


def summarize_gefs_members(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    work = frame.copy()
    work["tmax_3h_c"] = pd.to_numeric(work["tmax_3h_c"], errors="coerce")
    member = (
        work.groupby(["city_id", "station_id", "contract_date", "member_id"], as_index=False)
        .agg(
            member_daily_max_c=("tmax_3h_c", "max"),
            interval_count=("tmax_3h_c", "count"),
        )
    )
    member["fetch_status"] = np.where(
        member["interval_count"].eq(len(GEFS_TMAX_FORECAST_HOURS)),
        "ok",
        "incomplete",
    )
    ensemble = (
        member.groupby(["city_id", "station_id", "contract_date"], as_index=False)
        .agg(
            gefs_member_count=("member_daily_max_c", "count"),
            gefs_mean_high_c=("member_daily_max_c", "mean"),
            gefs_std_high_c=("member_daily_max_c", "std"),
            gefs_min_high_c=("member_daily_max_c", "min"),
            gefs_max_high_c=("member_daily_max_c", "max"),
        )
    )
    quantiles = (
        member.groupby(["city_id", "station_id", "contract_date"])["member_daily_max_c"]
        .quantile([0.1, 0.25, 0.5, 0.75, 0.9])
        .unstack()
        .rename(columns={0.1: "gefs_p10_c", 0.25: "gefs_p25_c", 0.5: "gefs_p50_c", 0.75: "gefs_p75_c", 0.9: "gefs_p90_c"})
        .reset_index()
    )
    ensemble = ensemble.merge(
        quantiles,
        on=["city_id", "station_id", "contract_date"],
        how="left",
    )
    ensemble["gefs_spread_c"] = ensemble["gefs_max_high_c"] - ensemble["gefs_min_high_c"]
    ensemble["fetch_status"] = np.where(
        ensemble["gefs_member_count"].eq(len(GEFS_MEMBERS)),
        "ok",
        "incomplete",
    )
    return member, ensemble


def run_gefs_backfill(
    data_root: Path,
    profiles: Sequence[AsiaCityProfile],
    start_date: date,
    end_date: date,
    *,
    workers: int = 12,
    force: bool = False,
    members: Sequence[str] = GEFS_MEMBERS,
) -> dict[str, Any]:
    day_results: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    summary_paths: list[str] = []
    # Keep several dates in flight so one day's network-bound range requests
    # can overlap another day's CPU-bound cfgrib decode. Split the caller's
    # worker budget across dates to avoid multiplying total concurrency.
    date_workers = max(1, min(3, int(workers) // 4))
    workers_per_date = max(1, int(workers) // date_workers)
    for month_key in month_keys(start_date, end_date):
        month_start, month_end = month_bounds(month_key, start_date, end_date)
        days = target_dates(month_start, month_end)
        with ThreadPoolExecutor(max_workers=date_workers) as executor:
            futures = {
                executor.submit(
                    backfill_gefs_day,
                    data_root,
                    profiles,
                    day,
                    workers=workers_per_date,
                    force=force,
                    members=members,
                ): day
                for day in days
            }
            for future in as_completed(futures):
                day_results.extend(future.result())
        manifests.extend(
            _compact_forecast_months(
                data_root,
                profiles,
                "gefs",
                month_start,
                month_end,
                expected_rows_per_day=len(members) * len(GEFS_TEMP_FORECAST_HOURS),
            )
        )
        for profile in profiles:
            path = (
                data_root
                / "normalized"
                / "forecasts"
                / "gefs"
                / profile.city_id
                / f"{month_key}.parquet"
            )
            if not path.exists():
                continue
            member, ensemble = summarize_gefs_members(pd.read_parquet(path))
            member_path = (
                data_root
                / "normalized"
                / "forecasts"
                / "gefs_member_daily"
                / profile.city_id
                / f"{month_key}.parquet"
            )
            ensemble_path = (
                data_root
                / "normalized"
                / "forecasts"
                / "gefs_ensemble_daily"
                / profile.city_id
                / f"{month_key}.parquet"
            )
            _atomic_write_frame(member_path, member)
            _atomic_write_frame(ensemble_path, ensemble)
            summary_paths.extend([str(member_path), str(ensemble_path)])
    return {
        "status": _combined_status(manifests),
        "day_results": day_results,
        "manifests": manifests,
        "summary_paths": summary_paths,
    }


def _jma_params(
    profile: AsiaCityProfile,
    start_date: date,
    end_date: date,
    *,
    historical: bool,
) -> dict[str, Any]:
    fields = (
        [f"{field}_previous_day1" for field in JMA_BASE_FIELDS]
        if historical
        else list(JMA_BASE_FIELDS)
    )
    params: dict[str, Any] = {
        "latitude": profile.latitude,
        "longitude": profile.longitude,
        "hourly": ",".join(fields),
        "models": "jma_msm",
        "timezone": profile.timezone,
    }
    if historical:
        params.update(
            {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            }
        )
    else:
        params["forecast_days"] = 2
    return params


def normalize_jma_payload(
    payload: Mapping[str, Any],
    profile: AsiaCityProfile,
    requested_dates: Sequence[date],
    *,
    historical: bool,
    fetched_at_utc: datetime,
    source_url: str,
    source_checksum: str,
) -> pd.DataFrame:
    hourly = payload.get("hourly", {})
    times = list(hourly.get("time", [])) if isinstance(hourly, Mapping) else []
    requested = {day.isoformat() for day in requested_dates}
    suffix = "_previous_day1" if historical else ""
    lineage = "jma_msm_previous_day1" if historical else "jma_msm_latest_at_collection"
    rows: list[dict[str, Any]] = []
    for index, raw_time in enumerate(times):
        local = pd.Timestamp(raw_time)
        if local.tzinfo is None:
            local = local.tz_localize(profile.timezone)
        else:
            local = local.tz_convert(profile.timezone)
        if local.date().isoformat() not in requested or not 11 <= local.hour <= 23:
            continue
        row: dict[str, Any] = {
            "city_id": profile.city_id,
            "station_id": profile.station_id,
            "provider": "jma_msm",
            "lineage": lineage,
            "contract_date": local.date().isoformat(),
            "issued_at_utc": pd.NA,
            "forecast_as_of_utc": _iso_z(
                datetime.combine(local.date(), datetime_time(2), tzinfo=UTC)
            ),
            "valid_time_utc": _iso_z(local.tz_convert(UTC).to_pydatetime()),
            "valid_time_local": local.isoformat(),
            "forecast_hour_local": local.hour,
            "member_id": "deterministic",
            "availability_basis": (
                "open_meteo_previous_day1_variable"
                if historical
                else "latest_available_at_collection"
            ),
            "retrieved_at_utc": _iso_z(fetched_at_utc),
            "source_url": source_url,
            "source_checksum": source_checksum,
        }
        for field in JMA_BASE_FIELDS:
            values = hourly.get(f"{field}{suffix}", [])
            row[_jma_output_column(field)] = (
                values[index] if index < len(values) and values[index] is not None else pd.NA
            )
        row["fetch_status"] = (
            "ok" if pd.notna(row.get("temp_2m_c")) else "incomplete"
        )
        rows.append(row)
    return pd.DataFrame(rows)


def backfill_jma_month(
    data_root: Path,
    profile: AsiaCityProfile,
    month_key: str,
    start_date: date,
    end_date: date,
    *,
    force: bool = False,
) -> dict[str, Any]:
    start, end = month_bounds(month_key, start_date, end_date)
    path = (
        data_root
        / "normalized"
        / "forecasts"
        / "jma_msm_previous_day1"
        / profile.city_id
        / f"{month_key}.parquet"
    )
    manifest_path = (
        data_root
        / "manifests"
        / "forecasts"
        / "jma_msm_previous_day1"
        / profile.city_id
        / f"{month_key}.json"
    )
    cached = _complete_manifest(manifest_path, path, force=force)
    if cached is not None:
        return cached
    response = _request(
        JMA_HISTORY_URL,
        params=_jma_params(profile, start, end, historical=True),
        timeout=180,
    )
    content = response.content
    raw_path = _content_addressed_raw_path(
        data_root / "raw" / "jma_msm_previous_day1" / profile.city_id,
        f"{month_key}",
        content,
        ".json",
    )
    fetched = datetime.now(UTC)
    frame = normalize_jma_payload(
        response.json(),
        profile,
        target_dates(start, end),
        historical=True,
        fetched_at_utc=fetched,
        source_url=response.url,
        source_checksum=_sha256_bytes(content),
    )
    frame = _merge_normalized_month(
        path,
        frame,
        keys=("contract_date", "valid_time_utc"),
    )
    _atomic_write_frame(path, frame)
    requested_dates = {day.isoformat() for day in target_dates(start, end)}
    requested_frame = frame.loc[frame["contract_date"].astype(str).isin(requested_dates)]
    expected_rows = len(requested_dates) * 13
    ok_count = int(frame.get("fetch_status", pd.Series(dtype=str)).astype(str).eq("ok").sum())
    manifest = {
        "city_id": profile.city_id,
        "station_id": profile.station_id,
        "provider": "jma_msm",
        "lineage": "jma_msm_previous_day1",
        "month": month_key,
        "status": "complete" if len(requested_frame) == expected_rows else "incomplete",
        "requested_start": start,
        "requested_end": end,
        "expected_rows": expected_rows,
        "row_count": len(frame),
        "ok_count": ok_count,
        "raw_path": str(raw_path),
        "normalized_path": str(path),
        "sha256": _sha256_file(path),
        "updated_at_utc": fetched,
    }
    _atomic_write_json(manifest_path, manifest)
    return manifest


def run_jma_history_backfill(
    data_root: Path,
    profiles: Sequence[AsiaCityProfile],
    start_date: date,
    end_date: date,
    *,
    workers: int = 4,
    force: bool = False,
) -> dict[str, Any]:
    jobs = [
        (profile, month)
        for profile in profiles
        for month in month_keys(start_date, end_date)
    ]

    def worker(job: tuple[AsiaCityProfile, str]) -> dict[str, Any]:
        profile, month = job
        return backfill_jma_month(
            data_root,
            profile,
            month,
            start_date,
            end_date,
            force=force,
        )

    results = _run_threaded(jobs, worker, workers=max(1, min(workers, 4)))
    return {"status": _combined_status(results), "results": results}


def collect_jma_live(
    data_root: Path,
    profile: AsiaCityProfile,
    contract_date: date,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = now or datetime.now(UTC)
    local_now = current.astimezone(ZoneInfo(profile.timezone))
    local_cutoff = datetime.combine(
        contract_date,
        datetime_time(profile.as_of_hour_local, profile.live_delay_minutes),
        tzinfo=ZoneInfo(profile.timezone),
    )
    if local_now.date() == contract_date and local_now < local_cutoff:
        raise RuntimeError(
            f"{profile.city_name} live collection starts at "
            f"{profile.as_of_hour_local:02d}:{profile.live_delay_minutes:02d} local"
        )
    # The promoted Tokyo model was trained on Open-Meteo's fixed lead-time
    # previous_day1 variables.  The ordinary JMA endpoint is a newer forecast
    # vintage and must never be substituted into this live row.
    response = _request(
        JMA_HISTORY_URL,
        params=_jma_params(profile, contract_date, contract_date, historical=True),
        timeout=120,
    )
    content = response.content
    raw_path = _content_addressed_raw_path(
        data_root / "raw" / "jma_msm_previous_day1" / profile.city_id,
        f"{contract_date}_{current:%Y%m%dT%H%M%SZ}",
        content,
        ".json",
    )
    frame = normalize_jma_payload(
        response.json(),
        profile,
        [contract_date],
        historical=True,
        fetched_at_utc=current,
        source_url=response.url,
        source_checksum=_sha256_bytes(content),
    )
    path = (
        data_root
        / "normalized"
        / "live"
        / "jma_msm_previous_day1"
        / profile.city_id
        / f"{contract_date}_{current:%Y%m%dT%H%M%SZ}.parquet"
    )
    _atomic_write_frame(path, frame)
    return {
        "city_id": profile.city_id,
        "contract_date": contract_date.isoformat(),
        "lineage": "jma_msm_previous_day1",
        "status": "complete" if len(frame) == 13 else "incomplete",
        "row_count": len(frame),
        "raw_path": str(raw_path),
        "normalized_path": str(path),
    }


def collect_live_observation(
    data_root: Path,
    profile: AsiaCityProfile,
    contract_date: date,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = now or datetime.now(UTC)
    # Tokyo training uses IEM ASOS/METAR. Live collection intentionally uses
    # that same station/population contract; do not relabel another feed as equivalent.
    response = _request(
        IEM_ASOS_URL,
        params=_iem_params(profile, contract_date, contract_date),
        timeout=180,
    )
    content = response.content
    frame = normalize_metar_rows(
        _iem_rows(content, profile),
        profile,
        [contract_date],
        source_filter="iem_asos_global_metar",
        data_source="iem_asos_global_metar_live",
    )
    source_name = "iem_asos_global_metar"
    required = (
        "observed_humidity_at_as_of",
        "observed_visibility_at_as_of",
        "observed_precip_recent_at_as_of",
    )
    weather_code = frame.get("observed_weather_code_at_as_of", pd.Series(dtype=str))
    complete = (
        not frame.empty
        and frame.get("observed_fetch_status", pd.Series(dtype=str)).astype(str).eq("ok").all()
        and all(
            name in frame
            and pd.to_numeric(frame[name], errors="coerce").notna().all()
            for name in required
        )
        and weather_code.astype(str).str.strip().ne("").all()
    )
    if not complete:
        frame["observed_fetch_status"] = "unavailable"
        frame["observed_unavailable_reason"] = "iem_required_runtime_observation_fields_missing"
    raw_path = _content_addressed_raw_path(
        data_root / "raw" / "live_observations" / profile.city_id,
        f"{contract_date}_{source_name}",
        content,
        ".json",
    )
    frame["source_uri"] = response.url
    frame["source_checksum"] = _sha256_bytes(content)
    path = (
        data_root
        / "normalized"
        / "live"
        / "observations"
        / profile.city_id
        / f"{contract_date}_{current:%Y%m%dT%H%M%SZ}.parquet"
    )
    _atomic_write_frame(path, frame)
    return {
        "city_id": profile.city_id,
        "contract_date": contract_date.isoformat(),
        "source": source_name,
        "status": (
            "complete"
            if complete
            else "incomplete"
        ),
        "raw_path": str(raw_path),
        "normalized_path": str(path),
    }


def run_observation_backfill(
    data_root: Path,
    profiles: Sequence[AsiaCityProfile],
    start_date: date,
    end_date: date,
    *,
    workers: int = 4,
    force: bool = False,
) -> dict[str, Any]:
    jobs = [
        (profile, month_key)
        for profile in profiles
        for month_key in month_keys(start_date, end_date)
    ]

    def worker(job: tuple[AsiaCityProfile, str]) -> dict[str, Any]:
        profile, month_key = job
        return backfill_observation_month(
            data_root,
            profile,
            month_key,
            start_date,
            end_date,
            force=force,
        )

    results = _run_threaded(jobs, worker, workers=max(1, min(workers, 4)))
    return {"status": _combined_status(results), "results": results}


def run_historical_pull(
    data_root: Path,
    profiles: Sequence[AsiaCityProfile],
    start_date: date,
    end_date: date,
    *,
    workers: int = 4,
    api_key: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    write_profile(data_root, profiles)
    results: dict[str, Any] = {}
    results["settlement"] = run_settlement_backfill(
        data_root,
        profiles,
        start_date,
        end_date,
        api_key=api_key,
        force=force,
        workers=workers,
    )
    results["observations"] = run_observation_backfill(
        data_root,
        profiles,
        start_date,
        end_date,
        workers=workers,
        force=force,
    )
    results["gfs"] = run_gfs_backfill(
        data_root,
        profiles,
        start_date,
        end_date,
        workers=gfs_day_workers(workers),
        force=force,
    )
    results["gefs"] = run_gefs_backfill(
        data_root,
        profiles,
        start_date,
        end_date,
        workers=max(1, workers),
        force=force,
    )
    results["jma_history"] = run_jma_history_backfill(
        data_root,
        profiles,
        start_date,
        end_date,
        workers=workers,
        force=force,
    )
    results["status"] = (
        "complete"
        if all(
            isinstance(result, Mapping) and result.get("status") == "complete"
            for key, result in results.items()
            if key != "status"
        )
        else "incomplete"
    )
    return results


def run_live(
    data_root: Path,
    profiles: Sequence[AsiaCityProfile],
    *,
    contract_date: date | None = None,
    workers: int = 12,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = now or datetime.now(UTC)
    day = contract_date or min(
        current.astimezone(ZoneInfo(profile.timezone)).date() for profile in profiles
    )
    for profile in profiles:
        cutoff = datetime.combine(
            day,
            datetime_time(profile.as_of_hour_local, profile.live_delay_minutes),
            tzinfo=ZoneInfo(profile.timezone),
        )
        if current.astimezone(ZoneInfo(profile.timezone)).date() == day and current < cutoff:
            raise RuntimeError(
                f"Live collection for {profile.city_name} starts at "
                f"{cutoff.isoformat()}"
            )
    observations = [
        collect_live_observation(data_root, profile, day, now=current)
        for profile in profiles
    ]
    gfs = backfill_gfs_day(data_root, profiles, day)
    gefs = backfill_gefs_day(data_root, profiles, day, workers=workers)
    jma = [
        collect_jma_live(data_root, profile, day, now=current) for profile in profiles
    ]
    return {
        "contract_date": day.isoformat(),
        "status": (
            "complete"
            if all(
                item.get("status") == "complete"
                for item in [*observations, *gfs, *gefs, *jma]
            )
            else "incomplete"
        ),
        "observations": observations,
        "gfs": gfs,
        "gefs": gefs,
        "jma_msm_previous_day1": jma,
    }


def audit_pipeline(
    data_root: Path,
    profiles: Sequence[AsiaCityProfile],
    start_date: date,
    end_date: date,
) -> dict[str, Any]:
    expected = {day.isoformat() for day in target_dates(start_date, end_date)}
    issues: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    comparisons: list[pd.DataFrame] = []

    for profile in profiles:
        settlements = _load_monthly_parts(
            data_root / "normalized" / "settlements" / profile.city_id
        )
        observations = _load_monthly_parts(
            data_root / "normalized" / "observations" / profile.city_id
        )
        _audit_daily_source(
            issues,
            coverage,
            profile,
            "settlement",
            settlements,
            expected,
            status_column="quality_flag",
        )
        _audit_daily_source(
            issues,
            coverage,
            profile,
            "observations",
            observations,
            expected,
            status_column="observed_fetch_status",
        )
        if not settlements.empty:
            bad = settlements.loc[
                ~settlements["station_id"].astype(str).eq(profile.station_id)
                | ~settlements["settlement_source"].astype(str).eq(
                    "wunderground_station_history"
                )
            ]
            if not bad.empty:
                issues.append(
                    {
                        "city_id": profile.city_id,
                        "scope": "settlement",
                        "issue": "non_authoritative_station_or_source",
                        "row_count": len(bad),
                    }
                )
        if not observations.empty:
            ok = observations["observed_fetch_status"].astype(str).eq("ok")
            selected = pd.to_datetime(
                observations.loc[ok, "observed_as_of_time_local"], errors="coerce"
            )
            if (selected.dt.hour > profile.as_of_hour_local).any():
                issues.append(
                    {
                        "city_id": profile.city_id,
                        "scope": "observations",
                        "issue": "post_11am_observation",
                    }
                )
        if not settlements.empty and not observations.empty:
            comparison = settlements[
                ["city_id", "station_id", "contract_date", "settlement_high_c"]
            ].merge(
                observations[
                    ["contract_date", "iem_daily_high_c"]
                ],
                on="contract_date",
                how="left",
            )
            comparison["difference_c"] = (
                pd.to_numeric(comparison["iem_daily_high_c"], errors="coerce")
                - pd.to_numeric(comparison["settlement_high_c"], errors="coerce")
            )
            comparisons.append(comparison)

        for provider, expected_rows in (
            ("gfs", len(GFS_FORECAST_HOURS)),
            ("gefs", len(GEFS_MEMBERS) * len(GEFS_TEMP_FORECAST_HOURS)),
            ("jma_msm_previous_day1", 13),
        ):
            frame = _load_monthly_parts(
                data_root / "normalized" / "forecasts" / provider / profile.city_id
            )
            dates = set(frame.get("contract_date", pd.Series(dtype=str)).astype(str))
            counts = frame.groupby("contract_date").size() if not frame.empty else pd.Series(dtype=int)
            missing = sorted(expected - dates)
            malformed = sorted(
                str(day)
                for day, count in counts.items()
                if str(day) in expected and int(count) != expected_rows
            )
            ok_rows = int(
                frame.get("fetch_status", pd.Series(dtype=str)).astype(str).eq("ok").sum()
            )
            record = {
                "city_id": profile.city_id,
                "provider": provider,
                "expected_dates": len(expected),
                "actual_dates": len(dates & expected),
                "expected_rows_per_day": expected_rows,
                "row_count": len(frame),
                "ok_rows": ok_rows,
                "missing_dates": missing,
                "malformed_dates": malformed,
            }
            coverage.append(record)
            if missing or malformed:
                issues.append(
                    {
                        "city_id": profile.city_id,
                        "scope": provider,
                        "issue": "coverage",
                        "missing_dates": missing,
                        "malformed_dates": malformed,
                    }
                )
            _audit_forecast_timing(issues, profile, provider, frame)

    comparison_frame = (
        pd.concat(comparisons, ignore_index=True)
        if comparisons
        else pd.DataFrame(
            columns=[
                "city_id",
                "station_id",
                "contract_date",
                "settlement_high_c",
                "iem_daily_high_c",
                "difference_c",
            ]
        )
    )
    mismatch = comparison_frame.loc[
        pd.to_numeric(comparison_frame.get("difference_c"), errors="coerce").abs() >= 0.5
    ]
    if not mismatch.empty:
        warnings.append(
            {
                "scope": "wunderground_iem_comparison",
                "issue": "daily_high_mismatches",
                "row_count": len(mismatch),
                "handling": "Wunderground remains authoritative",
            }
        )
    result = {
        "passed": not issues,
        "start_date": start_date,
        "end_date": end_date,
        "cities": [profile.city_id for profile in profiles],
        "issues": issues,
        "warnings": warnings,
        "updated_at_utc": datetime.now(UTC),
    }
    _atomic_write_json(data_root / "audit" / "audit_result.json", result)
    _atomic_write_frame(data_root / "audit" / "coverage.csv", pd.DataFrame(coverage))
    _atomic_write_frame(
        data_root / "audit" / "wunderground_vs_iem.csv",
        comparison_frame,
    )
    _atomic_write_frame(data_root / "audit" / "issues.csv", pd.DataFrame(issues))
    return result


def _compact_forecast_months(
    data_root: Path,
    profiles: Sequence[AsiaCityProfile],
    provider: str,
    start_date: date,
    end_date: date,
    *,
    expected_rows_per_day: int,
) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    for month_key in month_keys(start_date, end_date):
        start, end = month_bounds(month_key, start_date, end_date)
        expected_days = target_dates(start, end)
        month_complete = True
        month_manifests: list[dict[str, Any]] = []
        for profile in profiles:
            day_dir = (
                data_root
                / "normalized"
                / "forecasts"
                / provider
                / profile.city_id
                / month_key
            )
            frames = [pd.read_parquet(path) for path in sorted(day_dir.glob("*.parquet"))]
            frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            output = day_dir.parent / f"{month_key}.parquet"
            _atomic_write_frame(output, frame)
            counts = (
                frame.groupby("contract_date").size().to_dict()
                if not frame.empty and "contract_date" in frame
                else {}
            )
            ok = (
                frame.get("fetch_status", pd.Series(dtype=str))
                .astype(str)
                .eq("ok")
            )
            complete = (
                all(
                    int(counts.get(day.isoformat(), 0)) == expected_rows_per_day
                    for day in expected_days
                )
                and len(frame) > 0
                and ok.all()
            )
            month_complete = month_complete and complete
            manifest = {
                "city_id": profile.city_id,
                "station_id": profile.station_id,
                "provider": provider,
                "month": month_key,
                "status": "complete" if complete else "incomplete",
                "requested_start": start,
                "requested_end": end,
                "expected_rows_per_day": expected_rows_per_day,
                "row_count": len(frame),
                "ok_count": int(ok.sum()),
                "normalized_path": str(output),
                "sha256": _sha256_file(output),
                "updated_at_utc": datetime.now(UTC),
            }
            _atomic_write_json(
                data_root
                / "manifests"
                / "forecasts"
                / provider
                / profile.city_id
                / f"{month_key}.json",
                manifest,
            )
            manifests.append(manifest)
            month_manifests.append(manifest)
        if month_complete:
            cleanup = _cleanup_noaa_month_raw(
                data_root,
                provider,
                start,
                end,
            )
            for manifest in month_manifests:
                manifest["raw_cleanup"] = cleanup
                _atomic_write_json(
                    data_root
                    / "manifests"
                    / "forecasts"
                    / provider
                    / str(manifest["city_id"])
                    / f"{month_key}.json",
                    manifest,
                )
    return manifests


def _cleanup_noaa_month_raw(
    data_root: Path,
    provider: str,
    start_date: date,
    end_date: date,
) -> dict[str, Any]:
    directory = (data_root / "raw" / "nwp_subsets" / provider).resolve()
    if not directory.exists():
        return {"status": "complete", "deleted_files": 0, "deleted_bytes": 0}
    issues = {
        forecast_timing(day)["issue_utc"].strftime("%Y%m%d%H")
        for day in target_dates(start_date, end_date)
    }
    candidates = [
        path.resolve()
        for path in directory.glob("*.grib2")
        if any(issue in path.name for issue in issues)
    ]
    deleted_bytes = 0
    deleted_files = 0
    for path in candidates:
        if path.parent != directory:
            raise RuntimeError(f"Refusing raw cleanup outside {directory}: {path}")
        try:
            deleted_bytes += path.stat().st_size
            path.unlink()
            deleted_files += 1
        except FileNotFoundError:
            continue
    return {
        "status": "complete",
        "deleted_files": deleted_files,
        "deleted_bytes": deleted_bytes,
    }


def _audit_daily_source(
    issues: list[dict[str, Any]],
    coverage: list[dict[str, Any]],
    profile: AsiaCityProfile,
    source: str,
    frame: pd.DataFrame,
    expected: set[str],
    *,
    status_column: str,
) -> None:
    dates = set(frame.get("contract_date", pd.Series(dtype=str)).astype(str))
    missing = sorted(expected - dates)
    unexpected = sorted(dates - expected)
    duplicates = (
        frame["contract_date"].astype(str).duplicated().any()
        if not frame.empty and "contract_date" in frame
        else False
    )
    ok_count = int(
        frame.get(status_column, pd.Series(dtype=str)).astype(str).eq("ok").sum()
    )
    coverage.append(
        {
            "city_id": profile.city_id,
            "provider": source,
            "expected_dates": len(expected),
            "actual_dates": len(dates & expected),
            "row_count": len(frame),
            "ok_rows": ok_count,
            "missing_dates": missing,
            "unexpected_dates": unexpected,
        }
    )
    if missing or unexpected or duplicates:
        issues.append(
            {
                "city_id": profile.city_id,
                "scope": source,
                "issue": "daily_row_contract",
                "missing_dates": missing,
                "unexpected_dates": unexpected,
                "duplicates": bool(duplicates),
            }
        )


def _audit_forecast_timing(
    issues: list[dict[str, Any]],
    profile: AsiaCityProfile,
    provider: str,
    frame: pd.DataFrame,
) -> None:
    if frame.empty:
        return
    if provider == "jma_msm_previous_day1":
        bad_lineage = ~frame["lineage"].astype(str).eq("jma_msm_previous_day1")
        bad_basis = ~frame["availability_basis"].astype(str).eq(
            "open_meteo_previous_day1_variable"
        )
        if bad_lineage.any() or bad_basis.any():
            issues.append(
                {
                    "city_id": profile.city_id,
                    "scope": provider,
                    "issue": "historical_live_lineage_mixed",
                }
            )
        local = pd.to_datetime(frame["valid_time_local"], errors="coerce")
        if (~local.dt.hour.between(11, 23)).any():
            issues.append(
                {
                    "city_id": profile.city_id,
                    "scope": provider,
                    "issue": "invalid_local_forecast_hour",
                }
            )
        return

    for contract_date, group in frame.groupby("contract_date"):
        timing = forecast_timing(str(contract_date))
        issued = pd.to_datetime(group["issued_at_utc"], errors="coerce", utc=True)
        as_of = pd.to_datetime(group["forecast_as_of_utc"], errors="coerce", utc=True)
        if not issued.eq(pd.Timestamp(timing["issue_utc"])).all() or not as_of.eq(
            pd.Timestamp(timing["as_of_utc"])
        ).all():
            issues.append(
                {
                    "city_id": profile.city_id,
                    "scope": provider,
                    "contract_date": str(contract_date),
                    "issue": "timing_mismatch",
                }
            )
        local = pd.to_datetime(group["valid_time_local"], errors="coerce", utc=True).dt.tz_convert(
            profile.timezone
        )
        expected_day = _as_date(str(contract_date))
        bad_valid = [
            stamp
            for stamp in local
            if stamp.date() != expected_day
            and not (
                provider == "gefs"
                and stamp.date() == expected_day + timedelta(days=1)
                and stamp.hour == 0
            )
        ]
        if bad_valid:
            issues.append(
                {
                    "city_id": profile.city_id,
                    "scope": provider,
                    "contract_date": str(contract_date),
                    "issue": "valid_time_outside_local_day",
                }
            )
        if provider == "gefs" and group["member_id"].astype(str).nunique() != len(
            GEFS_MEMBERS
        ):
            issues.append(
                {
                    "city_id": profile.city_id,
                    "scope": provider,
                    "contract_date": str(contract_date),
                    "issue": "member_count",
                    "actual": int(group["member_id"].astype(str).nunique()),
                    "expected": len(GEFS_MEMBERS),
                }
            )


def _load_monthly_parts(directory: Path) -> pd.DataFrame:
    paths = sorted(directory.glob("????-??.parquet")) if directory.exists() else []
    frames = [pd.read_parquet(path) for path in paths]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _complete_manifest(
    manifest_path: Path,
    normalized_path: Path,
    *,
    force: bool,
) -> dict[str, Any] | None:
    if force or not manifest_path.exists() or not normalized_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        return None
    if manifest.get("sha256") != _sha256_file(normalized_path):
        return None
    return manifest


def _run_threaded(
    jobs: Sequence[Any],
    worker: Callable[[Any], Any],
    *,
    workers: int,
) -> list[Any]:
    if workers <= 1:
        output: list[Any] = []
        for job in jobs:
            try:
                output.append(worker(job))
            except Exception as exc:  # noqa: BLE001
                output.append({"status": "failed", "job": str(job), "error": str(exc)})
        return output
    output = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(worker, job): job for job in jobs}
        for future in as_completed(futures):
            try:
                output.append(future.result())
            except Exception as exc:  # noqa: BLE001
                output.append(
                    {
                        "status": "failed",
                        "job": str(futures[future]),
                        "error": str(exc),
                    }
                )
    return output


def _raw_metar_daily_highs(
    rows: Sequence[Mapping[str, Any]],
    timezone: str,
) -> dict[str, dict[str, float]]:
    values: dict[str, list[float]] = {}
    tz = ZoneInfo(timezone)
    for row in rows:
        observed = pd.to_datetime(row.get("observed_at"), errors="coerce", utc=True)
        temp_f = _number(row.get("temp_f"))
        if pd.isna(observed) or pd.isna(temp_f):
            continue
        day = observed.to_pydatetime().astimezone(tz).date().isoformat()
        values.setdefault(day, []).append(float(temp_f))
    return {
        day: {"high_f": max(temps), "high_c": float(_f_to_c(max(temps)))}
        for day, temps in values.items()
    }


def _jma_output_column(field: str) -> str:
    return {
        "temperature_2m": "temp_2m_c",
        "dew_point_2m": "dewpoint_2m_c",
        "relative_humidity_2m": "relative_humidity_2m_pct",
        "precipitation": "precipitation_mm",
        "cloud_cover": "cloud_cover_pct",
        "wind_speed_10m": "wind_speed_10m_kmh",
        "wind_direction_10m": "wind_direction_10m_deg",
        "wind_gusts_10m": "wind_gusts_10m_kmh",
    }[field]


def _combined_status(results: Sequence[Mapping[str, Any]]) -> str:
    return (
        "complete"
        if results and all(result.get("status") == "complete" for result in results)
        else "incomplete"
    )


def _number(value: Any) -> float | Any:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return pd.NA
    return number if math.isfinite(number) else pd.NA


def _c_to_f(value: Any) -> float | Any:
    number = _number(value)
    return float(number) * 9 / 5 + 32 if pd.notna(number) else pd.NA


def _f_to_c(value: Any) -> float | Any:
    number = _number(value)
    return (float(number) - 32) * 5 / 9 if pd.notna(number) else pd.NA


def _as_date(value: date | str) -> date:
    return value if isinstance(value, date) else date.fromisoformat(str(value)[:10])


def _iso_z(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
