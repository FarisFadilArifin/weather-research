from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import re
import time
import csv
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, time as datetime_time, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from .calibration.station_stacking import (
    PROVIDER_FORECAST_NUMERIC_COLUMNS,
    TARGET,
    TARGET_SOURCE_HKO_DAILY_MAX,
    TARGET_MODE_REMAINING_WARMUP,
    TRAINING_PROFILE_V20_ALIGNED,
    V20_EXPANDING_FOLDS,
    V20_HKO_GFS_NO_PEAK_FEATURE_VERSION,
    StationStackingConfig,
    YearSplitExperimentResult,
    _add_calendar_features,
    _add_current_observation_derived_features,
    _add_ensemble_features,
    _add_forecast_history_delta_features,
    _add_forecast_shape_features,
    _add_lagged_actual_features,
    _add_lagged_provider_error_features,
    _add_observation_forecast_delta_features,
    _add_observation_history_delta_features,
    _add_prior_month_provider_error_features,
    _add_provider_availability_features,
    _add_provider_cross_model_features,
    _add_provider_time_features,
    _modeling_frame,
    _provider_wide,
    add_v9_climatology_features,
    add_versioned_feature_engineering,
    feature_columns,
    summarize_year_split_predictions,
    tune_year_split_base_models,
    tune_year_split_stack_model,
    year_split_baseline_predictions,
    year_split_feature_importance,
    year_split_scoreboard,
    year_split_test_predictions,
)
from .current_observations import summarize_current_observations, unavailable_current_observation_row
from .direct_nwp_fetch import GFS_V16_LAYOUT_START_UTC, extract_direct_nwp_run_feature_points


TIMEZONE = "Asia/Hong_Kong"
TIMING_MODE = "hong_kong_same_day_11am_live_safe"
STATION_ID = "HKO"
OBSERVATION_STATION_ID = "HKO"
OBSERVATION_STATION_NAME = "Hong Kong Observatory Headquarters"
OBSERVATION_SOURCE_CONTRACT = "hko_open_data_archive_1min"
LEGACY_IEM_OBSERVATION_STATION_ID = "VHHH"
START_DATE = date(2021, 1, 1)
END_DATE = date(2026, 7, 20)
WARMUP_START_DATE = date(2011, 1, 1)
AS_OF_HOUR_LOCAL = 11
DECISION_DELAY_MINUTES = 10
MODEL_PROVIDERS = ("gfs",)
RESTRICTED_PROVIDERS = ("ifs", "icon")
# Backwards-compatible name used by the CLI; modeling is intentionally GFS-only.
PROVIDERS = MODEL_PROVIDERS
GFS_USABLE_START_DATE = date(2021, 3, 24)
GFS_ALLOWED_GAP_END_DATE = GFS_USABLE_START_DATE - timedelta(days=1)
FORECAST_HOURS = tuple(range(9, 22))
HKO_DAILY_EXTRACT_URL = "https://www.hko.gov.hk/cis/dailyExtract/dailyExtract_{year}.xml"
HKO_DAILY_SUMMARY_URL = (
    "https://www.weather.gov.hk/wxinfo/dailywx/yeswx/"
    "DYN_DAT_MINDS_RYES{date_yyyymmdd}.json"
)
IEM_ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
DATA_GOV_HISTORICAL_FILE_URL = "https://app.data.gov.hk/v1/historical-archive/get-file"
HKO_LATEST_TEMPERATURE_URL = (
    "https://data.weather.gov.hk/weatherAPI/hko_data/regional-weather/"
    "latest_1min_temperature.csv"
)
HKO_LATEST_MAXMIN_URL = (
    "https://data.weather.gov.hk/weatherAPI/hko_data/regional-weather/"
    "latest_since_midnight_maxmin.csv"
)
HKO_LATEST_HUMIDITY_URL = (
    "https://data.weather.gov.hk/weatherAPI/hko_data/regional-weather/"
    "latest_1min_humidity.csv"
)
GFS_ARCHIVE_URL = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
DEFAULT_DATA_ROOT = Path("data/calibration/hong_kong_11am")

HKO_POINT = {"lat": 22.3019, "lon": 114.1742, "elevation_m": 32.0}
FORECAST_POINTS = {"HKO": HKO_POINT}
FORECAST_FIELDS = [
    "temp_k_2m",
    "dewpoint_k_2m",
    "relative_humidity_pct_2m",
    "wind_u_ms_10m",
    "wind_v_ms_10m",
    "wind_gust_ms",
    "precip_mm_1h",
    "cloud_cover_pct",
]

IEM_FIELDS = (
    "tmpf",
    "dwpf",
    "drct",
    "sknt",
    "p01i",
    "alti",
    "mslp",
    "vsby",
    "gust",
    "skyc1",
    "skyc2",
    "skyc3",
    "skyc4",
    "skyl1",
    "skyl2",
    "skyl3",
    "skyl4",
    "wxcodes",
    "peak_wind_gust",
    "peak_wind_drct",
    "peak_wind_time",
    "metar",
)


@dataclass(frozen=True)
class HongKong11AMProfile:
    profile_id: str = "hong_kong_11am"
    station_id: str = STATION_ID
    observation_station_id: str = OBSERVATION_STATION_ID
    timezone: str = TIMEZONE
    as_of_hour_local: int = AS_OF_HOUR_LOCAL
    decision_delay_minutes: int = DECISION_DELAY_MINUTES
    forecast_issue_hour_utc: int = 18
    forecast_hour_min: int = min(FORECAST_HOURS)
    forecast_hour_max: int = max(FORECAST_HOURS)
    interpolation: str = "bilinear"
    start_date: str = START_DATE.isoformat()
    end_date: str = END_DATE.isoformat()
    warmup_start_date: str = WARMUP_START_DATE.isoformat()


PROFILE = HongKong11AMProfile()


def resolve_data_root(project_root: str | Path, data_root: str | Path | None = None) -> Path:
    project = Path(project_root).resolve()
    root = Path(data_root) if data_root is not None else DEFAULT_DATA_ROOT
    return root.resolve() if root.is_absolute() else (project / root).resolve()


def target_dates(start: date = START_DATE, end: date = END_DATE) -> list[date]:
    if end < start:
        return []
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]


def month_keys(start: date = START_DATE, end: date = END_DATE) -> list[str]:
    keys: list[str] = []
    cursor = start.replace(day=1)
    while cursor <= end:
        keys.append(cursor.strftime("%Y-%m"))
        cursor = (cursor.replace(day=28) + timedelta(days=4)).replace(day=1)
    return keys


def month_bounds(month_key: str, start: date = START_DATE, end: date = END_DATE) -> tuple[date, date]:
    month_start = date.fromisoformat(f"{month_key}-01")
    next_month = (month_start.replace(day=28) + timedelta(days=4)).replace(day=1)
    return max(start, month_start), min(end, next_month - timedelta(days=1))


def forecast_timing(contract_date: date | str) -> dict[str, Any]:
    day = date.fromisoformat(str(contract_date)[:10]) if not isinstance(contract_date, date) else contract_date
    as_of = datetime.combine(day, datetime_time(3), tzinfo=UTC)
    issue = datetime.combine(day - timedelta(days=1), datetime_time(18), tzinfo=UTC)
    window_end = datetime.combine(day, datetime_time(16), tzinfo=UTC)
    return {
        "contract_date": day.isoformat(),
        "issue_utc": issue,
        "as_of_utc": as_of,
        "window_start_utc": as_of,
        "window_end_utc": window_end,
        "forecast_hours": FORECAST_HOURS,
    }


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
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_json(path: Path, payload: Mapping[str, Any] | Sequence[Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=_json_default))


def _atomic_write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if path.suffix.lower() == ".parquet":
        frame.to_parquet(tmp, index=False)
    else:
        frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


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
                headers={"User-Agent": "weather-research-hong-kong/0.1"},
            )
            if response.status_code == 429 or response.status_code >= 500:
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else min(60.0, 2 ** attempt)
                time.sleep(delay + random.random())
                continue
            response.raise_for_status()
            return response
        except requests.HTTPError as exc:
            if exc.response is not None and 400 <= exc.response.status_code < 500:
                raise
            last_error = exc
            if attempt < attempts:
                time.sleep(min(60.0, 2 ** attempt) + random.random())
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(min(60.0, 2 ** attempt) + random.random())
    raise RuntimeError(f"GET failed after {attempts} attempts: {url}: {last_error}") from last_error


@contextmanager
def _host_slot(lock_root: Path, host: str, limit: int, *, stale_after_seconds: int = 900):
    """Cross-process host limiter based on atomic lock-file creation."""
    lock_root.mkdir(parents=True, exist_ok=True)
    acquired: Path | None = None
    while acquired is None:
        now = time.time()
        for slot in range(max(1, int(limit))):
            path = lock_root / f"{host}_{slot}.lock"
            if path.exists():
                try:
                    if now - path.stat().st_mtime > stale_after_seconds:
                        path.unlink()
                except FileNotFoundError:
                    pass
            try:
                descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                continue
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({"pid": os.getpid(), "acquired_at": datetime.now(UTC).isoformat()}))
            acquired = path
            break
        if acquired is None:
            time.sleep(0.25 + random.random() * 0.25)
    try:
        yield
    finally:
        try:
            acquired.unlink()
        except FileNotFoundError:
            pass


def _content_addressed_raw_path(directory: Path, stem: str, content: bytes, suffix: str) -> Path:
    checksum = _sha256_bytes(content)
    path = directory / f"{stem}_{checksum[:12]}{suffix}"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_bytes(content)
        os.replace(tmp, path)
    return path


def parse_hko_daily_extract(payload: Mapping[str, Any], year: int) -> pd.DataFrame:
    months = payload.get("stn", {}).get("data", [])
    rows: list[dict[str, Any]] = []
    for month_payload in months:
        month = int(month_payload["month"])
        for values in month_payload.get("dayData", []):
            if not values:
                continue
            try:
                day = int(str(values[0]).strip())
            except (TypeError, ValueError):
                # HKO appends rows such as "Mean/Total" after daily records.
                continue
            raw_high = values[2] if len(values) > 2 else None
            high_c = pd.to_numeric(pd.Series([raw_high]), errors="coerce").iloc[0]
            try:
                contract_date = date(year, month, day)
            except ValueError:
                continue
            rows.append(
                {
                    "station_id": STATION_ID,
                    "contract_date": contract_date.isoformat(),
                    "actual_high_c": float(high_c) if pd.notna(high_c) else pd.NA,
                    "actual_high_f": float(high_c) * 9 / 5 + 32 if pd.notna(high_c) else pd.NA,
                    "actual_source": "hko_daily_extract_absolute_daily_max",
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"HKO daily extract for {year} contained no rows")
    return frame.sort_values("contract_date").reset_index(drop=True)


def parse_hko_daily_summary(payload: Mapping[str, Any], expected_date: date) -> dict[str, Any]:
    """Parse HKO's official, provisional daily summary for a completed day."""
    summary = payload.get("DYN_DAT_MINDS_RYES", {})
    reported_date = str(summary.get("ReportTimeInfoDate", {}).get("Val_Eng", "")).strip()
    expected = expected_date.strftime("%Y%m%d")
    if reported_date != expected:
        raise ValueError(f"HKO daily-summary date mismatch: expected {expected}, got {reported_date!r}")
    raw_high = summary.get("HKOReadingsMaxTemp", {}).get("Val_Eng")
    high_c = pd.to_numeric(pd.Series([raw_high]), errors="coerce").iloc[0]
    if pd.isna(high_c):
        raise ValueError(f"HKO daily summary for {expected_date} has no maximum temperature")
    value_c = float(high_c)
    return {
        "station_id": STATION_ID,
        "contract_date": expected_date.isoformat(),
        "actual_high_c": value_c,
        "actual_high_f": value_c * 9 / 5 + 32,
        "actual_source": "hko_daily_weather_summary_provisional",
    }


def backfill_hko_labels(data_root: Path, *, end_date: date = END_DATE, force: bool = False) -> pd.DataFrame:
    raw_dir = data_root / "raw" / "hko_daily_extract"
    normalized_path = data_root / "normalized" / "hko_daily_max.parquet"
    frames: list[pd.DataFrame] = []
    raw_files: list[dict[str, Any]] = []
    for year in range(WARMUP_START_DATE.year, end_date.year + 1):
        cached = sorted(raw_dir.glob(f"daily_extract_{year}_*.json"))
        if cached and not force and year < end_date.year:
            raw_path = cached[-1]
            content = raw_path.read_bytes()
        else:
            response = _request(HKO_DAILY_EXTRACT_URL.format(year=year), timeout=60)
            content = response.content
            raw_path = _content_addressed_raw_path(raw_dir, f"daily_extract_{year}", content, ".json")
        payload = json.loads(content.decode("utf-8-sig"))
        frame = parse_hko_daily_extract(payload, year)
        frame["source_uri"] = HKO_DAILY_EXTRACT_URL.format(year=year)
        frame["source_checksum"] = _sha256_bytes(content)
        frames.append(frame)
        raw_files.append({"year": year, "path": str(raw_path), "sha256": _sha256_bytes(content)})
    labels = pd.concat(frames, ignore_index=True)
    labels = labels.loc[pd.to_datetime(labels["contract_date"]).dt.date <= end_date].copy()
    labels = labels.drop_duplicates("contract_date", keep="last").sort_values("contract_date").reset_index(drop=True)

    # Daily Extract is finalized monthly. Fill only absent target dates from
    # HKO's own archived Daily Weather Summary, published after each day and
    # explicitly marked provisional by HKO. Never synthesize a target value.
    target_dates_required = pd.date_range(START_DATE, end_date, freq="D").date
    present = set(labels["contract_date"].astype(str))
    summary_rows: list[dict[str, Any]] = []
    for missing_date in target_dates_required:
        if missing_date.isoformat() in present:
            continue
        date_key = missing_date.strftime("%Y%m%d")
        cached = sorted(raw_dir.glob(f"daily_summary_{date_key}_*.json"))
        source_uri = HKO_DAILY_SUMMARY_URL.format(date_yyyymmdd=date_key)
        if cached and not force:
            raw_path = cached[-1]
            content = raw_path.read_bytes()
        else:
            response = _request(source_uri, timeout=60)
            content = response.content
            raw_path = _content_addressed_raw_path(raw_dir, f"daily_summary_{date_key}", content, ".json")
        payload = json.loads(content.decode("utf-8-sig"))
        row = parse_hko_daily_summary(payload, missing_date)
        row["source_uri"] = source_uri
        row["source_checksum"] = _sha256_bytes(content)
        summary_rows.append(row)
        raw_files.append(
            {
                "date": missing_date.isoformat(),
                "path": str(raw_path),
                "sha256": _sha256_bytes(content),
            }
        )
    if summary_rows:
        labels = pd.concat([labels, pd.DataFrame(summary_rows)], ignore_index=True)
        labels = labels.drop_duplicates("contract_date", keep="last").sort_values("contract_date").reset_index(drop=True)

    target = labels.loc[
        (pd.to_datetime(labels["contract_date"]).dt.date >= START_DATE)
        & (pd.to_datetime(labels["contract_date"]).dt.date <= end_date)
    ]
    missing_target_count = len(target_dates_required) - target["contract_date"].nunique()
    _atomic_write_frame(normalized_path, labels)
    _atomic_write_json(
        data_root / "manifests" / "hko_labels.json",
        {
            "status": "complete" if missing_target_count == 0 else "incomplete",
            "row_count": len(labels),
            "target_row_count": int(target["contract_date"].nunique()),
            "missing_target_count": int(missing_target_count),
            "first_date": labels["contract_date"].min(),
            "last_date": labels["contract_date"].max(),
            "normalized_path": str(normalized_path),
            "raw_files": raw_files,
            "updated_at_utc": datetime.now(UTC),
        },
    )
    return labels


def _iem_params(start: date, end: date) -> list[tuple[str, Any]]:
    exclusive_end = end + timedelta(days=1)
    params: list[tuple[str, Any]] = [("station", LEGACY_IEM_OBSERVATION_STATION_ID)]
    params.extend(("data", field) for field in IEM_FIELDS)
    params.extend(
        [
            ("year1", start.year),
            ("month1", start.month),
            ("day1", start.day),
            ("year2", exclusive_end.year),
            ("month2", exclusive_end.month),
            ("day2", exclusive_end.day),
            ("tz", TIMEZONE),
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


def _iem_to_raw_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    raw_rows: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        valid = pd.to_datetime(row.get("valid"), errors="coerce")
        if pd.isna(valid):
            continue
        if valid.tzinfo is None:
            valid = valid.tz_localize(TIMEZONE)
        raw_rows.append(
            {
                "observed_at": valid.tz_convert(UTC).isoformat(),
                "temp_f": row.get("tmpf"),
                "dewpoint_f": row.get("dwpf"),
                "wind_dir_degrees": row.get("drct"),
                "wind_speed_kt": row.get("sknt"),
                "wind_gust_kt": row.get("gust"),
                "peak_wind_gust_kt": row.get("peak_wind_gust"),
                "peak_wind_dir": row.get("peak_wind_drct"),
                "peak_wind_time": row.get("peak_wind_time"),
                "altimeter_inhg": row.get("alti"),
                "sea_level_pressure_mb": row.get("mslp"),
                "visibility_miles": row.get("vsby"),
                "weather_codes": row.get("wxcodes"),
                # IEM explicitly documents non-US precipitation as unavailable.
                "precip_1hr_inches": pd.NA,
                "sky_cover_1": row.get("skyc1"),
                "sky_cover_2": row.get("skyc2"),
                "sky_cover_3": row.get("skyc3"),
                "sky_cover_4": row.get("skyc4"),
                "sky_base_1_ft": row.get("skyl1"),
                "sky_base_2_ft": row.get("skyl2"),
                "sky_base_3_ft": row.get("skyl3"),
                "sky_base_4_ft": row.get("skyl4"),
                "raw_metar": row.get("metar"),
                "source": "iem_asos_global_metar",
                "observation_type": "METAR",
                "qc_field": "iem_as_is_archive",
            }
        )
    return raw_rows


def normalize_iem_metar_csv(content: bytes, dates: Sequence[date]) -> pd.DataFrame:
    frame = pd.read_csv(io.BytesIO(content), na_values=["null", "M", ""], low_memory=False)
    raw_rows = _iem_to_raw_rows(frame)
    summarized = summarize_current_observations(
        raw_rows,
        station_id=LEGACY_IEM_OBSERVATION_STATION_ID,
        station_name="Hong Kong International Airport",
        airport_name="Hong Kong International Airport",
        timezone=TIMEZONE,
        contract_dates=[day.isoformat() for day in dates],
        timing_mode="same_day_11am",
        as_of_hour_local=AS_OF_HOUR_LOCAL,
        source_filter="iem_asos_global_metar",
    )
    rows: list[dict[str, Any]] = []
    for row in summarized:
        row["timing_mode"] = TIMING_MODE
        row["observed_precip_recent_at_as_of"] = pd.NA
        row["observed_precip_amount_available"] = False
        row["observed_data_source"] = "iem_asos_global_metar_raw"
        age = pd.to_numeric(pd.Series([row.get("observed_as_of_age_minutes")]), errors="coerce").iloc[0]
        if row.get("observed_fetch_status") == "ok" and (pd.isna(age) or float(age) < 0 or float(age) > 60):
            unavailable = unavailable_current_observation_row(
                station_id=LEGACY_IEM_OBSERVATION_STATION_ID,
                station_name="Hong Kong International Airport",
                airport_name="Hong Kong International Airport",
                timezone=TIMEZONE,
                contract_date=str(row["contract_date"]),
                timing_mode=TIMING_MODE,
                as_of_hour_local=AS_OF_HOUR_LOCAL,
                reason=f"Latest METAR age exceeds 60 minutes: {age}",
            )
            unavailable["observed_precip_amount_available"] = False
            unavailable["observed_data_source"] = "iem_asos_global_metar_raw"
            row = unavailable
        rows.append(row)
    return pd.DataFrame(rows).sort_values("contract_date").reset_index(drop=True)


def backfill_iem_month(data_root: Path, month_key: str, *, force: bool = False) -> dict[str, Any]:
    start, end = month_bounds(month_key)
    normalized_path = data_root / "normalized" / "observations" / f"{month_key}.parquet"
    manifest_path = data_root / "manifests" / "observations" / f"{month_key}.json"
    if normalized_path.exists() and manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") == "complete":
            return manifest
    with _host_slot(data_root / "locks", "iem", 2):
        response = _request(IEM_ASOS_URL, params=_iem_params(start, end), timeout=180)
    content = response.content
    raw_path = _content_addressed_raw_path(
        data_root / "raw" / "iem_metar",
        f"{LEGACY_IEM_OBSERVATION_STATION_ID}_{month_key}",
        content,
        ".csv",
    )
    dates = target_dates(start, end)
    observations = normalize_iem_metar_csv(content, dates)
    observations["source_uri"] = response.url
    observations["source_checksum"] = _sha256_bytes(content)
    _atomic_write_frame(normalized_path, observations)
    manifest = {
        "source": "iem_asos_global_metar",
        "month": month_key,
        "status": "complete",
        "row_count": len(observations),
        "ok_count": int(observations["observed_fetch_status"].astype(str).str.lower().eq("ok").sum()),
        "raw_path": str(raw_path),
        "normalized_path": str(normalized_path),
        "sha256": _sha256_file(normalized_path),
        "updated_at_utc": datetime.now(UTC),
    }
    _atomic_write_json(manifest_path, manifest)
    return manifest


def _normalized_csv_header(value: Any) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def _hko_station_rows_from_csv(content: bytes) -> list[dict[str, Any]]:
    text = content.decode("utf-8-sig", errors="replace").replace("\r\n", "\n")
    reader = csv.DictReader(io.StringIO(text))
    rows: list[dict[str, Any]] = []
    for raw in reader:
        normalized = {
            _normalized_csv_header(key): value
            for key, value in raw.items()
            if key is not None
        }
        station = str(normalized.get("automaticweatherstation", "")).strip().lower()
        if station not in {"hk observatory", "hong kong observatory"}:
            continue
        try:
            timestamp = datetime.strptime(
                str(normalized.get("datetime", "")).strip(),
                "%Y%m%d%H%M",
            )
        except ValueError:
            continue
        row: dict[str, Any] = {"observed_at_local": timestamp}
        for key, value in normalized.items():
            if key in {"datetime", "automaticweatherstation"}:
                continue
            try:
                row[key] = float(str(value).strip())
            except (TypeError, ValueError):
                row[key] = pd.NA
        rows.append(row)
    return rows


def parse_hko_historical_archive_zip(content: bytes) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        for name in archive.namelist():
            if name.endswith("/"):
                continue
            rows.extend(_hko_station_rows_from_csv(archive.read(name)))
    if not rows:
        return pd.DataFrame(columns=["observed_at_local"])
    return (
        pd.DataFrame(rows)
        .drop_duplicates("observed_at_local", keep="last")
        .sort_values("observed_at_local")
        .reset_index(drop=True)
    )


def _celsius_to_fahrenheit(value: Any) -> float | Any:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return pd.NA if pd.isna(number) else float(number) * 9.0 / 5.0 + 32.0


def _archive_value_at_or_before(
    frame: pd.DataFrame,
    cutoff: pd.Timestamp,
    column: str,
) -> float | Any:
    if frame.empty or column not in frame:
        return pd.NA
    eligible = frame.loc[
        frame["observed_at_local"].le(cutoff)
        & frame["observed_at_local"].dt.normalize().eq(cutoff.normalize())
    ]
    if eligible.empty:
        return pd.NA
    value = pd.to_numeric(eligible.sort_values("observed_at_local").iloc[-1][column], errors="coerce")
    return pd.NA if pd.isna(value) else float(value)


def normalize_hko_open_data_archives(
    temperature_zip: bytes,
    maxmin_zip: bytes,
    humidity_zip: bytes,
    dates: Sequence[date],
    *,
    max_observation_age_minutes: float = 20.0,
) -> pd.DataFrame:
    temperature = parse_hko_historical_archive_zip(temperature_zip)
    maxmin = parse_hko_historical_archive_zip(maxmin_zip)
    humidity = parse_hko_historical_archive_zip(humidity_zip)
    temperature_column = "airtemperaturedegreecelsius"
    maximum_column = "maximumairtemperaturesincemidnightdegreecelsius"
    humidity_column = "relativehumiditypercent"

    raw_rows: list[dict[str, Any]] = []
    for row in temperature.to_dict(orient="records"):
        observed_local = pd.Timestamp(row["observed_at_local"]).tz_localize(TIMEZONE)
        raw_rows.append(
            {
                "observed_at": observed_local.tz_convert(UTC).isoformat(),
                "temp_f": _celsius_to_fahrenheit(row.get(temperature_column)),
                "source": OBSERVATION_SOURCE_CONTRACT,
                "observation_type": "hko_1min_mean_air_temperature",
                "qc_field": "hko_provisional_open_data_archive",
            }
        )
    summarized = summarize_current_observations(
        raw_rows,
        station_id=OBSERVATION_STATION_ID,
        station_name=OBSERVATION_STATION_NAME,
        airport_name=OBSERVATION_STATION_NAME,
        timezone=TIMEZONE,
        contract_dates=[day.isoformat() for day in dates],
        timing_mode="same_day_11am",
        as_of_hour_local=AS_OF_HOUR_LOCAL,
        source_filter=OBSERVATION_SOURCE_CONTRACT,
    )
    rows: list[dict[str, Any]] = []
    for row in summarized:
        day = pd.Timestamp(str(row["contract_date"]))
        cutoff = day.replace(hour=AS_OF_HOUR_LOCAL)
        nine_am = day.replace(hour=9)
        high_c = _archive_value_at_or_before(maxmin, cutoff, maximum_column)
        high_9am_c = _archive_value_at_or_before(maxmin, nine_am, maximum_column)
        humidity_pct = _archive_value_at_or_before(humidity, cutoff, humidity_column)
        row["timing_mode"] = TIMING_MODE
        row["observed_high_temp_through_as_of_f"] = _celsius_to_fahrenheit(high_c)
        row["observed_humidity_at_as_of"] = humidity_pct
        row["observed_high_so_far_change_since_9am_f"] = (
            pd.NA
            if pd.isna(high_c) or pd.isna(high_9am_c)
            else float(high_c - high_9am_c) * 9.0 / 5.0
        )
        row["observed_high_so_far_change_since_11am_f"] = 0.0
        row["observed_precip_recent_at_as_of"] = pd.NA
        row["observed_precip_amount_available"] = False
        row["observed_data_source"] = OBSERVATION_SOURCE_CONTRACT
        if row.get("observed_fetch_status") == "ok" and pd.isna(high_c):
            row["observed_fetch_status"] = "unavailable"
            row["observed_unavailable_reason"] = "HKO maximum-since-midnight snapshot missing at 11 AM"
        age = pd.to_numeric(pd.Series([row.get("observed_as_of_age_minutes")]), errors="coerce").iloc[0]
        if row.get("observed_fetch_status") == "ok" and (
            pd.isna(age) or float(age) < 0 or float(age) > max_observation_age_minutes
        ):
            row["observed_fetch_status"] = "unavailable"
            row["observed_unavailable_reason"] = (
                "HKO observation age exceeds "
                f"{max_observation_age_minutes:g} minutes: {age}"
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("contract_date").reset_index(drop=True)


def _download_historical_archive(
    data_root: Path,
    resource_url: str,
    archive_date: date,
    *,
    raw_stem: str,
) -> tuple[bytes, Path, str]:
    with _host_slot(data_root / "locks", "data_gov_historical", 2):
        response = _request(
            DATA_GOV_HISTORICAL_FILE_URL,
            params={"url": resource_url, "time": archive_date.strftime("%Y%m%d")},
            timeout=180,
        )
    content = response.content
    if not content.startswith(b"PK"):
        raise RuntimeError(
            f"Historical archive response is not a ZIP file: {resource_url} {archive_date}"
        )
    raw_path = _content_addressed_raw_path(
        data_root / "raw" / "hko_open_data",
        f"{raw_stem}_{archive_date.isoformat()}",
        content,
        ".zip",
    )
    return content, raw_path, response.url


def _historical_archive_snapshot_dates(content: bytes) -> set[date]:
    dates: set[date] = set()
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        for name in archive.namelist():
            match = re.search(r"(20\d{6})-\d{4}-[^/]+\.csv$", name, flags=re.IGNORECASE)
            if match:
                dates.add(datetime.strptime(match.group(1), "%Y%m%d").date())
    return dates


def _merge_historical_archive_zips(payloads: Sequence[bytes]) -> bytes:
    buffer = io.BytesIO()
    seen: set[str] = set()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as merged:
        for payload in payloads:
            with zipfile.ZipFile(io.BytesIO(payload)) as source:
                for member in source.infolist():
                    if member.is_dir() or member.filename in seen:
                        continue
                    seen.add(member.filename)
                    merged.writestr(member.filename, source.read(member))
    return buffer.getvalue()


def _download_historical_archive_period(
    data_root: Path,
    resource_url: str,
    dates: Sequence[date],
    *,
    raw_stem: str,
) -> tuple[bytes, list[Path], list[str]]:
    requested_dates = sorted(set(dates))
    if not requested_dates:
        raise ValueError("At least one archive date is required")

    payloads: list[bytes] = []
    raw_paths: list[Path] = []
    source_uris: list[str] = []
    covered_dates: set[date] = set()

    first_payload, first_raw_path, first_uri = _download_historical_archive(
        data_root,
        resource_url,
        requested_dates[0],
        raw_stem=raw_stem,
    )
    payloads.append(first_payload)
    raw_paths.append(first_raw_path)
    source_uris.append(first_uri)
    covered_dates.update(_historical_archive_snapshot_dates(first_payload))

    # Closed months are normally delivered as one consolidated ZIP. The active
    # month is delivered as one ZIP per day, so fetch only dates not in the
    # first response and merge them into the same parser input.
    for day in requested_dates:
        if day in covered_dates:
            continue
        try:
            payload, raw_path, source_uri = _download_historical_archive(
                data_root,
                resource_url,
                day,
                raw_stem=raw_stem,
            )
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                # Some source archives omit individual days. The normalizer
                # emits an unavailable row, which the prediction-time quality
                # filter excludes from modeling without leaking future data.
                continue
            raise
        payloads.append(payload)
        raw_paths.append(raw_path)
        source_uris.append(source_uri)
        covered_dates.update(_historical_archive_snapshot_dates(payload))

    return _merge_historical_archive_zips(payloads), raw_paths, source_uris


def backfill_hko_observations_month(
    data_root: Path,
    month_key: str,
    *,
    force: bool = False,
) -> dict[str, Any]:
    start, end = month_bounds(month_key)
    normalized_path = data_root / "normalized" / "observations" / f"{month_key}.parquet"
    manifest_path = data_root / "manifests" / "observations" / f"{month_key}.json"
    if normalized_path.exists() and manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") == "complete" and manifest.get("source") == OBSERVATION_SOURCE_CONTRACT:
            return manifest

    dates = target_dates(start, end)
    temperature_zip, temperature_raw, temperature_uri = _download_historical_archive_period(
        data_root,
        HKO_LATEST_TEMPERATURE_URL,
        dates,
        raw_stem="temperature",
    )
    maxmin_zip, maxmin_raw, maxmin_uri = _download_historical_archive_period(
        data_root,
        HKO_LATEST_MAXMIN_URL,
        dates,
        raw_stem="maxmin",
    )
    humidity_zip, humidity_raw, humidity_uri = _download_historical_archive_period(
        data_root,
        HKO_LATEST_HUMIDITY_URL,
        dates,
        raw_stem="humidity",
    )
    observations = normalize_hko_open_data_archives(
        temperature_zip,
        maxmin_zip,
        humidity_zip,
        dates,
        # The official archive has one snapshot per day from June-November
        # 2021, generally between 09:53 and 10:56. These are still strictly
        # pre-decision and retain their exact age as a model feature.
        max_observation_age_minutes=75.0,
    )
    combined_checksum = _sha256_bytes(temperature_zip + maxmin_zip + humidity_zip)
    observations["source_uri"] = json.dumps(
        [*temperature_uri, *maxmin_uri, *humidity_uri],
        separators=(",", ":"),
    )
    observations["source_checksum"] = combined_checksum
    _atomic_write_frame(normalized_path, observations)
    ok_count = int(observations["observed_fetch_status"].astype(str).str.lower().eq("ok").sum())
    status = "complete" if len(observations) == len(dates) else "incomplete"
    manifest = {
        "source": OBSERVATION_SOURCE_CONTRACT,
        "month": month_key,
        "status": status,
        "row_count": len(observations),
        "ok_count": ok_count,
        "missing_or_unavailable_dates": observations.loc[
            ~observations["observed_fetch_status"].astype(str).str.lower().eq("ok"),
            "contract_date",
        ].astype(str).tolist(),
        "raw_paths": [
            str(path)
            for path in [*temperature_raw, *maxmin_raw, *humidity_raw]
        ],
        "normalized_path": str(normalized_path),
        "sha256": _sha256_file(normalized_path),
        "updated_at_utc": datetime.now(UTC),
    }
    _atomic_write_json(manifest_path, manifest)
    return manifest


def _single_csv_zip(content: bytes, name: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, content)
    return buffer.getvalue()


def collect_hko_live_observation(data_root: Path, contract_date: date) -> dict[str, Any]:
    responses: list[requests.Response] = []
    for resource_url in (
        HKO_LATEST_TEMPERATURE_URL,
        HKO_LATEST_MAXMIN_URL,
        HKO_LATEST_HUMIDITY_URL,
    ):
        with _host_slot(data_root / "locks", "hko_open_data", 2):
            responses.append(_request(resource_url, timeout=60))
    observation = normalize_hko_open_data_archives(
        _single_csv_zip(responses[0].content, "latest_1min_temperature.csv"),
        _single_csv_zip(responses[1].content, "latest_since_midnight_maxmin.csv"),
        _single_csv_zip(responses[2].content, "latest_1min_humidity.csv"),
        [contract_date],
    )
    if observation.empty or observation.loc[0, "observed_fetch_status"] != "ok":
        reason = "No prediction-time-safe HKO snapshot at or before 11 AM"
        if not observation.empty:
            reason = str(observation.loc[0, "observed_unavailable_reason"])
        raise RuntimeError(reason)

    raw_paths = [
        _content_addressed_raw_path(
            data_root / "raw" / "hko_open_data_live",
            f"{contract_date.isoformat()}_{index}",
            response.content,
            ".csv",
        )
        for index, response in enumerate(responses)
    ]
    observation["source_uri"] = json.dumps([response.url for response in responses], separators=(",", ":"))
    observation["source_checksum"] = _sha256_bytes(b"".join(response.content for response in responses))
    month_key = contract_date.strftime("%Y-%m")
    normalized_path = data_root / "normalized" / "observations" / f"{month_key}.parquet"
    existing = pd.read_parquet(normalized_path) if normalized_path.exists() else pd.DataFrame()
    if not existing.empty:
        existing = existing.loc[
            existing["contract_date"].astype(str).str[:10].ne(contract_date.isoformat())
        ]
    combined = pd.concat([existing, observation], ignore_index=True, sort=False)
    combined = combined.sort_values("contract_date").drop_duplicates("contract_date", keep="last")
    _atomic_write_frame(normalized_path, combined)
    return {
        "source": OBSERVATION_SOURCE_CONTRACT,
        "status": "complete",
        "contract_date": contract_date.isoformat(),
        "observed_at_utc": observation.loc[0, "observed_as_of_time_utc"],
        "normalized_path": str(normalized_path),
        "raw_paths": [str(path) for path in raw_paths],
    }


def _kelvin_to_f(value: Any) -> float | Any:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return pd.NA if pd.isna(number) else float(number - 273.15) * 9 / 5 + 32


def _mean(values: Iterable[Any]) -> float | Any:
    series = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    return pd.NA if series.empty else float(series.mean())


def _maximum(values: Iterable[Any]) -> float | Any:
    series = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    return pd.NA if series.empty else float(series.max())


def _circular_mean(values: Iterable[Any]) -> float | Any:
    series = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    if series.empty:
        return pd.NA
    radians = np.deg2rad(series.to_numpy(dtype=float))
    return float(np.rad2deg(np.arctan2(np.sin(radians).mean(), np.cos(radians).mean())) % 360)


def _precip_summary(values: Iterable[Any]) -> dict[str, Any]:
    clean = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna().clip(lower=0)
    if clean.empty:
        return {
            "precip_amount": pd.NA,
            "forecast_precip_total_mm": pd.NA,
            "forecast_precip_max_1h_mm": pd.NA,
            "forecast_precip_hours_count": pd.NA,
            "forecast_has_precip": pd.NA,
            "forecast_precip_intensity_code": pd.NA,
            "forecast_precip_intensity": pd.NA,
        }
    total = float(clean.sum())
    maximum = float(clean.max())
    if total <= 0:
        code, label = 0, "dry"
    elif maximum < 0.25 and total < 1.0:
        code, label = 1, "trace_or_drizzle"
    elif maximum < 2.5:
        code, label = 2, "light_rain"
    elif maximum < 7.6:
        code, label = 3, "moderate_rain"
    else:
        code, label = 4, "heavy_rain"
    return {
        "precip_amount": total,
        "forecast_precip_total_mm": total,
        "forecast_precip_max_1h_mm": maximum,
        "forecast_precip_hours_count": int(clean.gt(0).sum()),
        "forecast_has_precip": int(total > 0),
        "forecast_precip_intensity_code": code,
        "forecast_precip_intensity": label,
    }


def summarize_gfs_values(contract_date: date, values: Mapping[str, Mapping[int, Mapping[str, float]]]) -> dict[str, Any]:
    timing = forecast_timing(contract_date)
    hko = values.get("HKO", {})
    ordered_hko = [hko.get(hour, {}) for hour in FORECAST_HOURS]
    temp_f = [_kelvin_to_f(fields.get("temp_k_2m")) for fields in ordered_hko]
    dewpoint_f = [_kelvin_to_f(fields.get("dewpoint_k_2m")) for fields in ordered_hko]
    wind_speed_mph = [
        float(fields["wind_speed_ms_10m"]) * 2.2369362921
        if "wind_speed_ms_10m" in fields
        else pd.NA
        for fields in ordered_hko
    ]
    wind_gust_mph = [
        float(fields["wind_gust_ms"]) * 2.2369362921 if "wind_gust_ms" in fields else pd.NA
        for fields in ordered_hko
    ]
    precip = _precip_summary(fields.get("precip_mm_1h") for fields in ordered_hko)
    high = _maximum(temp_f)
    returned = [hour for hour in FORECAST_HOURS if pd.notna(_kelvin_to_f(hko.get(hour, {}).get("temp_k_2m")))]
    return {
        "station_id": STATION_ID,
        "station_name": "Hong Kong Observatory Headquarters",
        "airport_name": OBSERVATION_STATION_NAME,
        "provider": "gfs",
        "model": "gfs_operational_0p25",
        "source_label": "noaa_gfs_operational_exact_18z",
        "timing_mode": TIMING_MODE,
        "cycle_selection_policy": "fixed_previous_utc_day_18z",
        "contract_date": contract_date.isoformat(),
        "forecast_as_of": timing["as_of_utc"].isoformat().replace("+00:00", "Z"),
        "issued_at": timing["issue_utc"].isoformat().replace("+00:00", "Z"),
        "forecast_window_start": timing["window_start_utc"].isoformat().replace("+00:00", "Z"),
        "forecast_window_end": timing["window_end_utc"].isoformat().replace("+00:00", "Z"),
        "horizon_hours": 0,
        "raw_forecast_high_f": high,
        "forecast_hour_min": min(FORECAST_HOURS),
        "forecast_hour_max": max(FORECAST_HOURS),
        "forecast_hour_count_requested": len(FORECAST_HOURS),
        "forecast_hour_count_returned": len(returned),
        "forecast_hour_missing": ",".join(str(hour) for hour in FORECAST_HOURS if hour not in returned) or pd.NA,
        "forecast_hour_completeness": len(returned) / len(FORECAST_HOURS),
        "grid_dist_km_mean": pd.NA,
        "forecast_temp_at_as_of_f": _kelvin_to_f(hko.get(9, {}).get("temp_k_2m")),
        "dewpoint_mean_f": _mean(dewpoint_f),
        "dewpoint_at_as_of_f": _kelvin_to_f(hko.get(9, {}).get("dewpoint_k_2m")),
        "humidity_mean": _mean(fields.get("relative_humidity_pct_2m") for fields in ordered_hko),
        "humidity_at_as_of": hko.get(9, {}).get("relative_humidity_pct_2m", pd.NA),
        **precip,
        "cloud_cover_mean": _mean(fields.get("cloud_cover_pct") for fields in ordered_hko),
        "cloud_cover_max": _maximum(fields.get("cloud_cover_pct") for fields in ordered_hko),
        "wind_speed_mean": _mean(wind_speed_mph),
        "wind_speed_max": _maximum(wind_speed_mph),
        "wind_speed_at_as_of": wind_speed_mph[0] if wind_speed_mph else pd.NA,
        "wind_direction_mean": _circular_mean(fields.get("wind_direction_deg_10m") for fields in ordered_hko),
        "wind_direction_at_as_of": hko.get(9, {}).get("wind_direction_deg_10m", pd.NA),
        "wind_gust_max": _maximum(wind_gust_mph),
        "pressure_mslp_mean": pd.NA,
        "pressure_surface_mean": pd.NA,
        "visibility_mean": pd.NA,
        "ceiling_min": pd.NA,
        "data_source": "noaa_gfs_bdp_pds_range_requests",
        "source_file_or_url": GFS_ARCHIVE_URL,
        "archive_backend": "noaa_aws",
        "interpolation": "bilinear_regular_0p25",
        "fetch_status": "ok" if len(returned) == len(FORECAST_HOURS) and pd.notna(high) else "partial",
        "unavailable_reason": pd.NA if len(returned) == len(FORECAST_HOURS) else "Missing one or more hourly temperatures",
    }


def _unavailable_forecast_row(provider: str, contract_date: date, status: str, reason: str) -> dict[str, Any]:
    timing = forecast_timing(contract_date)
    return {
        "station_id": STATION_ID,
        "station_name": "Hong Kong Observatory Headquarters",
        "airport_name": OBSERVATION_STATION_NAME,
        "provider": provider,
        "model": f"{provider}_operational",
        "source_label": f"{provider}_operational_exact_18z",
        "timing_mode": TIMING_MODE,
        "cycle_selection_policy": "fixed_previous_utc_day_18z",
        "contract_date": contract_date.isoformat(),
        "forecast_as_of": timing["as_of_utc"].isoformat().replace("+00:00", "Z"),
        "issued_at": timing["issue_utc"].isoformat().replace("+00:00", "Z"),
        "forecast_window_start": timing["window_start_utc"].isoformat().replace("+00:00", "Z"),
        "forecast_window_end": timing["window_end_utc"].isoformat().replace("+00:00", "Z"),
        "forecast_hour_min": min(FORECAST_HOURS),
        "forecast_hour_max": max(FORECAST_HOURS),
        "fetch_status": status,
        "unavailable_reason": reason,
    }


def backfill_gfs_day(data_root: Path, contract_date: date, *, force: bool = False) -> dict[str, Any]:
    month_key = contract_date.strftime("%Y-%m")
    day_path = data_root / "normalized" / "forecasts" / "gfs" / month_key / f"{contract_date}.parquet"
    if day_path.exists() and not force:
        existing = pd.read_parquet(day_path)
        return existing.iloc[0].to_dict()
    timing = forecast_timing(contract_date)
    issue = timing["issue_utc"]
    if issue < GFS_V16_LAYOUT_START_UTC:
        row = _unavailable_forecast_row(
            "gfs",
            contract_date,
            "access_blocked",
            "Pre-v16 exact GFS requires the free NCAR GDEX archive/subset credential; no proxy was used",
        )
        row["archive_backend"] = "ncar_gdex_required"
    else:
        # Monthly shards default to one active request, while an explicit
        # override allows fast, targeted repair of an incomplete day.
        os.environ.setdefault("WEATHER_RESEARCH_DIRECT_NWP_WORKERS", "1")
        raw_dir = data_root / "raw" / "nwp_subsets"
        try:
            values = extract_direct_nwp_run_feature_points(
                FORECAST_POINTS,
                "gfs",
                raw_dir,
                issue,
                FORECAST_HOURS,
                force_refresh=force,
                feature_fields=FORECAST_FIELDS,
                interpolation="bilinear",
            )
            row = summarize_gfs_values(contract_date, values)
        except Exception as exc:  # noqa: BLE001
            row = _unavailable_forecast_row("gfs", contract_date, "failed", str(exc))
            row["archive_backend"] = "noaa_aws"
    _atomic_write_frame(day_path, pd.DataFrame([row]))
    return row


_GFS_RAW_FILE_RE = re.compile(
    r"^\.?gfs_.+_(?P<issue>\d{10})_f\d{3}\.grib2(?:\.\d+\.tmp)?$"
)


def cleanup_gfs_month_raw(data_root: Path, month_key: str) -> dict[str, Any]:
    """Remove only raw GFS subsets owned by a completed target-month shard."""
    start, end = month_bounds(month_key)
    issue_keys = {
        forecast_timing(day)["issue_utc"].strftime("%Y%m%d%H")
        for day in target_dates(start, end)
    }
    raw_dir = (data_root / "raw" / "nwp_subsets" / "gfs").resolve()
    if not raw_dir.exists():
        return {"status": "complete", "deleted_files": 0, "deleted_bytes": 0}

    candidates: list[Path] = []
    for path in raw_dir.iterdir():
        if not path.is_file():
            continue
        match = _GFS_RAW_FILE_RE.match(path.name)
        if match and match.group("issue") in issue_keys:
            resolved = path.resolve()
            if resolved.parent != raw_dir:
                raise RuntimeError(f"Refusing GFS cleanup outside raw directory: {resolved}")
            candidates.append(resolved)

    deleted_bytes = 0
    for path in candidates:
        try:
            deleted_bytes += path.stat().st_size
            path.unlink()
        except FileNotFoundError:
            # Another recovery attempt may have already removed the same file.
            continue
    return {
        "status": "complete",
        "deleted_files": len(candidates),
        "deleted_bytes": deleted_bytes,
    }


def backfill_gfs_month(data_root: Path, month_key: str, *, force: bool = False) -> dict[str, Any]:
    start, end = month_bounds(month_key)
    month_path = data_root / "normalized" / "forecasts" / "gfs" / f"{month_key}.parquet"
    manifest_path = data_root / "manifests" / "forecasts" / "gfs" / f"{month_key}.json"
    if month_path.exists() and manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") == "complete":
            return manifest
    rows = [backfill_gfs_day(data_root, day, force=force) for day in target_dates(start, end)]
    frame = pd.DataFrame(rows).sort_values("contract_date").reset_index(drop=True)
    _atomic_write_frame(month_path, frame)
    counts = frame["fetch_status"].astype(str).value_counts().to_dict()
    manifest = {
        "provider": "gfs",
        "month": month_key,
        "status": "cleanup_pending",
        "row_count": len(frame),
        "status_counts": counts,
        "normalized_path": str(month_path),
        "sha256": _sha256_file(month_path),
        "raw_retention_policy": "delete_after_month_normalized_and_checksummed",
        "updated_at_utc": datetime.now(UTC),
    }
    _atomic_write_json(manifest_path, manifest)
    cleanup = cleanup_gfs_month_raw(data_root, month_key)
    manifest["status"] = "complete"
    manifest["raw_cleanup"] = cleanup
    manifest["updated_at_utc"] = datetime.now(UTC)
    _atomic_write_json(manifest_path, manifest)
    return manifest


def _free_month_worker(data_root_text: str, month_key: str, force: bool) -> dict[str, Any]:
    data_root = Path(data_root_text)
    os.environ["WEATHER_RESEARCH_DIRECT_NWP_WORKERS"] = "1"
    result: dict[str, Any] = {"month": month_key}
    try:
        result["observations"] = backfill_hko_observations_month(data_root, month_key, force=force)
    except Exception as exc:  # noqa: BLE001
        result["observations"] = {"status": "failed", "error": str(exc)}
    try:
        result["gfs"] = backfill_gfs_month(data_root, month_key, force=force)
    except Exception as exc:  # noqa: BLE001
        result["gfs"] = {"status": "failed", "error": str(exc)}
    return result


def _hko_observation_month_worker(data_root_text: str, month_key: str, force: bool) -> dict[str, Any]:
    data_root = Path(data_root_text)
    try:
        return backfill_hko_observations_month(data_root, month_key, force=force)
    except Exception as exc:  # noqa: BLE001
        return {"month": month_key, "status": "failed", "error": str(exc)}


def _gfs_month_worker(data_root_text: str, month_key: str, force: bool) -> dict[str, Any]:
    data_root = Path(data_root_text)
    os.environ["WEATHER_RESEARCH_DIRECT_NWP_WORKERS"] = "4"
    try:
        return backfill_gfs_month(data_root, month_key, force=force)
    except Exception as exc:  # noqa: BLE001
        return {"month": month_key, "status": "failed", "error": str(exc)}


def _run_monthly_backfill(
    data_root: Path,
    *,
    worker: Any,
    progress_name: str,
    workers: int,
    force: bool,
) -> list[dict[str, Any]]:
    data_root.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(data_root / "profile.json", asdict(PROFILE))
    months = month_keys()
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = {
            executor.submit(worker, str(data_root), month, force): month
            for month in months
        }
        for future in as_completed(futures):
            month = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:  # noqa: BLE001
                results.append({"month": month, "status": "failed", "error": str(exc)})
            _atomic_write_json(
                data_root / "manifests" / progress_name,
                {
                    "status": "running" if len(results) < len(months) else "complete",
                    "workers": workers,
                    "completed_months": len(results),
                    "total_months": len(months),
                    "results": sorted(results, key=lambda item: str(item.get("month"))),
                    "updated_at_utc": datetime.now(UTC),
                },
            )
    return sorted(results, key=lambda item: str(item.get("month")))


def run_hko_observation_backfill(
    data_root: Path,
    *,
    workers: int = 12,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Backfill official HKO Headquarters observations without touching forecasts."""
    return _run_monthly_backfill(
        data_root,
        worker=_hko_observation_month_worker,
        progress_name="hko_observation_backfill_progress.json",
        workers=workers,
        force=force,
    )


def run_gfs_backfill(
    data_root: Path,
    *,
    workers: int = 12,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Backfill HKO-point GFS forecasts without touching observations."""
    return _run_monthly_backfill(
        data_root,
        worker=_gfs_month_worker,
        progress_name="gfs_backfill_progress.json",
        workers=workers,
        force=force,
    )


def run_free_backfill(data_root: Path, *, workers: int = 12, force: bool = False) -> list[dict[str, Any]]:
    data_root.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(data_root / "profile.json", asdict(PROFILE))
    backfill_hko_labels(data_root, force=force)
    months = month_keys()
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = {
            executor.submit(_free_month_worker, str(data_root), month, force): month
            for month in months
        }
        for future in as_completed(futures):
            month = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:  # noqa: BLE001
                results.append({"month": month, "status": "failed", "error": str(exc)})
            _atomic_write_json(
                data_root / "manifests" / "free_backfill_progress.json",
                {
                    "status": "running" if len(results) < len(months) else "complete",
                    "workers": workers,
                    "completed_months": len(results),
                    "total_months": len(months),
                    "results": sorted(results, key=lambda item: str(item.get("month"))),
                    "updated_at_utc": datetime.now(UTC),
                },
            )
    return sorted(results, key=lambda item: str(item.get("month")))


def write_quote_packets(data_root: Path) -> dict[str, Path]:
    quote_dir = data_root / "access" / "quote_requests"
    quote_dir.mkdir(parents=True, exist_ok=True)
    ecmwf_request = {
        "class": "od",
        "dataset": "operational archive",
        "stream": "oper",
        "type": "fc",
        "date": f"{START_DATE.isoformat()}/to/{END_DATE.isoformat()}",
        "time": "18:00:00",
        "step": "9/to/21/by/1",
        "levtype": "sfc",
        "param": "2t/2d/tcc/tp/10u/10v/10fg",
        "grid": "0.25/0.25",
        "area": "22.75/113.50/21.75/114.50",
        "format": "grib2",
        "purpose": "personal research; Hong Kong point forecast calibration; no redistribution",
    }
    dwd_request = {
        "model": "ICON global operational",
        "dates": [START_DATE.isoformat(), END_DATE.isoformat()],
        "cycles_utc": [18],
        "forecast_hours": [min(FORECAST_HOURS), max(FORECAST_HOURS)],
        "forecast_hour_increment": 1,
        "variables": ["T_2M", "TD_2M", "RELHUM_2M", "TOT_PREC", "CLCT", "U_10M", "V_10M", "VMAX_10M"],
        "grid": "regular latitude/longitude 0.25 degree",
        "area": {"north": 22.75, "west": 113.5, "south": 21.75, "east": 114.5},
        "purpose": "personal research; exact historical/live operational-run alignment",
        "restriction": "Do not substitute ICON-DREAM or another reanalysis",
    }
    ecmwf_json = quote_dir / "ecmwf_mars_request.json"
    dwd_json = quote_dir / "dwd_icon_archive_request.json"
    ecmwf_text = quote_dir / "ecmwf_personal_research_request.txt"
    dwd_text = quote_dir / "dwd_personal_research_request.txt"
    _atomic_write_json(ecmwf_json, ecmwf_request)
    _atomic_write_json(dwd_json, dwd_request)
    _atomic_write_text(
        ecmwf_text,
        "Please confirm personal-research access or a fee waiver for the attached narrowly scoped MARS request. "
        "No order or payment is authorised. We require the exact historical operational deterministic IFS runs, "
        "not ERA5, reanalysis, or reconstructed forecasts.\n",
    )
    _atomic_write_text(
        dwd_text,
        "Please confirm whether exact archived operational ICON-global 18 UTC forecasts are available for "
        "2021-01-01 through 2026-07-20 for the attached variables and Hong Kong cutout. No purchase is authorised. "
        "ICON-DREAM/reanalysis is not an acceptable substitute.\n",
    )
    _atomic_write_json(
        data_root / "access" / "provider_access.json",
        {
            "spend_authorized": False,
            "usage_class": "personal_research",
            "ifs": {"status": "quote_required", "request": str(ecmwf_json)},
            "icon": {"status": "quote_required", "request": str(dwd_json)},
            "gfs_legacy": {"status": "free_access_credential_required", "archive": "NCAR GDEX d084001"},
            "updated_at_utc": datetime.now(UTC),
        },
    )
    return {"ecmwf_json": ecmwf_json, "ecmwf_text": ecmwf_text, "dwd_json": dwd_json, "dwd_text": dwd_text}


def _validate_imported_provider_frame(frame: pd.DataFrame, provider: str) -> pd.DataFrame:
    required = {
        "contract_date",
        "provider",
        "issued_at",
        "forecast_as_of",
        "forecast_hour_min",
        "forecast_hour_max",
        "raw_forecast_high_f",
        "forecast_temp_at_as_of_f",
        "fetch_status",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{provider} normalized import missing columns: {', '.join(missing)}")
    out = frame.copy()
    out["provider"] = out["provider"].astype(str).str.lower()
    if not out["provider"].eq(provider).all():
        raise ValueError(f"{provider} normalized import contains another provider")
    out["contract_date"] = out["contract_date"].astype(str).str[:10]
    for row in out.itertuples(index=False):
        expected = forecast_timing(str(row.contract_date))
        issued = pd.Timestamp(row.issued_at).to_pydatetime().astimezone(UTC)
        as_of = pd.Timestamp(row.forecast_as_of).to_pydatetime().astimezone(UTC)
        if issued != expected["issue_utc"] or as_of != expected["as_of_utc"]:
            raise ValueError(f"{provider} timing mismatch on {row.contract_date}")
        if int(row.forecast_hour_min) != 9 or int(row.forecast_hour_max) != 21:
            raise ValueError(f"{provider} forecast-hour mismatch on {row.contract_date}")
    return out.sort_values("contract_date").drop_duplicates("contract_date", keep="last").reset_index(drop=True)


def run_restricted_import(
    data_root: Path,
    providers: Sequence[str] = RESTRICTED_PROVIDERS,
    *,
    force: bool = False,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for provider in providers:
        provider = provider.lower()
        incoming = data_root / "incoming" / provider
        files = sorted([*incoming.glob("*.parquet"), *incoming.glob("*.csv")]) if incoming.exists() else []
        if not files:
            results[provider] = {"status": "access_blocked", "reason": f"No normalized exact {provider} files in {incoming}"}
            continue
        frames = [pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path, low_memory=False) for path in files]
        frame = _validate_imported_provider_frame(pd.concat(frames, ignore_index=True, sort=False), provider)
        for month, part in frame.groupby(frame["contract_date"].str[:7]):
            output = data_root / "normalized" / "forecasts" / provider / f"{month}.parquet"
            if output.exists() and not force:
                continue
            _atomic_write_frame(output, part.reset_index(drop=True))
        results[provider] = {"status": "complete", "row_count": len(frame), "files": [str(path) for path in files]}
    _atomic_write_json(data_root / "manifests" / "restricted_import.json", {"results": results, "updated_at_utc": datetime.now(UTC)})
    return results


def _load_parquet_parts(directory: Path) -> pd.DataFrame:
    files = sorted(directory.glob("*.parquet")) if directory.exists() else []
    frames = [pd.read_parquet(path) for path in files]
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _load_forecast_provider(data_root: Path, provider: str) -> pd.DataFrame:
    directory = data_root / "normalized" / "forecasts" / provider
    files = sorted(directory.glob("*.parquet")) if directory.exists() else []
    files = [path for path in files if path.parent == directory]
    frames = [pd.read_parquet(path) for path in files]
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True, sort=False)
    frame["contract_date"] = frame["contract_date"].astype(str).str[:10]
    frame = frame.loc[frame["fetch_status"].astype(str).str.lower().eq("ok")].copy()
    return frame.drop_duplicates("contract_date", keep="last").sort_values("contract_date").reset_index(drop=True)


def build_rolling_climatology(labels: pd.DataFrame) -> pd.DataFrame:
    work = labels.copy()
    work["date"] = pd.to_datetime(work["contract_date"], errors="coerce")
    work["year"] = work["date"].dt.year
    work["month_day"] = work["date"].dt.strftime("%m-%d")
    rows: list[dict[str, Any]] = []
    for row in work.itertuples(index=False):
        history = work.loc[
            work["month_day"].eq(row.month_day)
            & work["year"].between(int(row.year) - 10, int(row.year) - 1)
        ]
        values = pd.to_numeric(history["actual_high_f"], errors="coerce").dropna()
        rows.append(
            {
                "station_code": STATION_ID,
                "target_year": int(row.year),
                "month_day": row.month_day,
                "climatology_high_10y_f": float(values.mean()) if not values.empty else pd.NA,
                "climatology_high_10y_std_f": float(values.std()) if len(values) >= 2 else pd.NA,
                "climatology_high_10y_count": int(len(values)),
                "climatology_source_start_year": int(history["year"].min()) if not history.empty else pd.NA,
                "climatology_source_end_year": int(history["year"].max()) if not history.empty else pd.NA,
            }
        )
    return pd.DataFrame(rows).drop_duplicates(["station_code", "target_year", "month_day"], keep="last")


def build_features(data_root: Path, *, providers: Sequence[str] = MODEL_PROVIDERS) -> pd.DataFrame:
    labels_path = data_root / "normalized" / "hko_daily_max.parquet"
    if not labels_path.exists():
        raise FileNotFoundError("HKO labels are missing; run free-backfill first")
    labels = pd.read_parquet(labels_path).sort_values("contract_date").reset_index(drop=True)
    observations = _load_parquet_parts(data_root / "normalized" / "observations")
    if not observations.empty:
        observations["contract_date"] = observations["contract_date"].astype(str).str[:10]
        observations = observations.drop_duplicates("contract_date", keep="last")
    wide = labels.copy()
    wide[TARGET] = pd.to_numeric(wide["actual_high_f"], errors="coerce")
    wide["target_source"] = TARGET_SOURCE_HKO_DAILY_MAX
    wide["actual_data_quality_flag"] = "ok"
    wide["actual_raw_observation_count"] = pd.NA
    if not observations.empty:
        wide = wide.merge(observations, on="contract_date", how="left", suffixes=("", "_observation"))
    providers_tuple = tuple(str(provider).lower() for provider in providers)
    for provider in providers_tuple:
        forecast = _load_forecast_provider(data_root, provider)
        provider_wide = _provider_wide(forecast, provider, numeric_columns=PROVIDER_FORECAST_NUMERIC_COLUMNS)
        wide = wide.merge(provider_wide, on="contract_date", how="left")
    wide["station_id"] = STATION_ID
    wide["station_code"] = STATION_ID
    wide["station_name"] = "Hong Kong Observatory Headquarters"
    wide["airport_name"] = OBSERVATION_STATION_NAME
    wide["city_label"] = "Hong Kong"
    wide["timezone"] = TIMEZONE
    wide["country"] = "HK"
    wide["lat"] = HKO_POINT["lat"]
    wide["lon"] = HKO_POINT["lon"]
    wide = wide.sort_values("contract_date").reset_index(drop=True)
    wide = _add_calendar_features(wide)
    wide = _add_current_observation_derived_features(wide)
    wide = _add_provider_availability_features(wide, providers_tuple)
    wide = _add_provider_time_features(wide, providers_tuple, TIMEZONE)
    wide = _add_ensemble_features(wide, providers_tuple)
    wide = _add_forecast_shape_features(wide, providers_tuple)
    wide = _add_provider_cross_model_features(wide, providers_tuple)
    wide = _add_lagged_actual_features(wide)
    wide = _add_lagged_provider_error_features(wide, providers_tuple)
    wide = _add_prior_month_provider_error_features(wide, providers_tuple)
    wide = _add_forecast_history_delta_features(wide, providers_tuple)
    wide = _add_observation_history_delta_features(wide)
    wide = _add_observation_forecast_delta_features(wide, providers_tuple)
    wide = add_versioned_feature_engineering(
        wide,
        feature_version=V20_HKO_GFS_NO_PEAK_FEATURE_VERSION,
        providers=providers_tuple,
    )
    climatology = build_rolling_climatology(labels)
    climatology_path = data_root / "normalized" / "hko_rolling_10y_daily_high_normals.csv"
    _atomic_write_frame(climatology_path, climatology)
    wide = add_v9_climatology_features(
        wide,
        station_id=STATION_ID,
        climatology_normals_path=climatology_path,
    )
    target_mask = pd.to_datetime(wide["contract_date"]).dt.date >= START_DATE
    target_frame = wide.loc[target_mask & (pd.to_datetime(wide["contract_date"]).dt.date <= END_DATE)].copy()
    output = data_root / "features" / "HKO_features.parquet"
    _atomic_write_frame(output, target_frame)
    config = StationStackingConfig(
        station_id=STATION_ID,
        providers=providers_tuple,
        timing_mode=TIMING_MODE,
        feature_version=V20_HKO_GFS_NO_PEAK_FEATURE_VERSION,
        training_profile=TRAINING_PROFILE_V20_ALIGNED,
        target_mode=TARGET_MODE_REMAINING_WARMUP,
        target_source=TARGET_SOURCE_HKO_DAILY_MAX,
    )
    categorical, numeric = feature_columns(target_frame, config)
    excluded_metadata = {"actual_high_c", "source_checksum", "source_uri", "observed_precip_amount_available"}
    numeric = [column for column in numeric if column not in excluded_metadata]
    inventory = pd.DataFrame(
        [{"feature": column, "kind": "categorical"} for column in categorical]
        + [{"feature": column, "kind": "numeric"} for column in numeric]
    )
    inventory["coverage_pct"] = inventory["feature"].map(target_frame.notna().mean().mul(100).to_dict())
    _atomic_write_frame(data_root / "features" / "HKO_feature_columns.csv", inventory)
    return target_frame


def _required_provider_dates(provider: str) -> set[str]:
    provider_name = str(provider).lower()
    start = GFS_USABLE_START_DATE if provider_name == "gfs" else START_DATE
    return {day.isoformat() for day in target_dates(start, END_DATE)}


def provider_modeling_coverage(data_root: Path, provider: str) -> dict[str, Any]:
    provider_name = str(provider).lower()
    frame = _load_forecast_provider(data_root, provider_name)
    actual_dates = set(frame.get("contract_date", pd.Series(dtype="string")).astype(str).str[:10])
    required_dates = _required_provider_dates(provider_name)
    target_date_set = {day.isoformat() for day in target_dates()}
    missing_dates = sorted(required_dates - actual_dates)
    unexpected_dates = sorted(actual_dates - target_date_set)
    return {
        "provider": provider_name,
        "usable_start_date": (
            GFS_USABLE_START_DATE.isoformat() if provider_name == "gfs" else START_DATE.isoformat()
        ),
        "usable_end_date": END_DATE.isoformat(),
        "ok_rows": len(frame),
        "unique_ok_dates": len(actual_dates),
        "required_usable_rows": len(required_dates),
        "allowed_early_gap_rows": (
            (GFS_ALLOWED_GAP_END_DATE - START_DATE).days + 1 if provider_name == "gfs" else 0
        ),
        "missing_usable_dates": missing_dates,
        "unexpected_dates": unexpected_dates,
        "modeling_ready": not missing_dates and not unexpected_dates,
    }


def audit_pipeline(data_root: Path, *, providers: Sequence[str] = MODEL_PROVIDERS) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    expected_dates = {day.isoformat() for day in target_dates()}
    labels_path = data_root / "normalized" / "hko_daily_max.parquet"
    labels = pd.read_parquet(labels_path) if labels_path.exists() else pd.DataFrame()
    target_labels = labels.loc[labels.get("contract_date", pd.Series(dtype=str)).astype(str).isin(expected_dates)].copy() if not labels.empty else labels
    if len(target_labels) != len(expected_dates) or set(target_labels.get("contract_date", [])) != expected_dates:
        issues.append({"scope": "labels", "issue": "target_date_coverage", "expected": len(expected_dates), "actual": len(target_labels)})
    if not target_labels.empty and target_labels["contract_date"].duplicated().any():
        issues.append({"scope": "labels", "issue": "duplicate_dates"})

    observations = _load_parquet_parts(data_root / "normalized" / "observations")
    if not observations.empty:
        observation_dates = set(observations["contract_date"].astype(str).str[:10])
        missing_observation_dates = sorted(expected_dates - observation_dates)
        unexpected_observation_dates = sorted(observation_dates - expected_dates)
        if missing_observation_dates or unexpected_observation_dates:
            issues.append(
                {
                    "scope": "observations",
                    "issue": "date_coverage",
                    "missing_dates": missing_observation_dates,
                    "unexpected_dates": unexpected_observation_dates,
                }
            )
        if observations["contract_date"].astype(str).str[:10].duplicated().any():
            issues.append({"scope": "observations", "issue": "duplicate_dates"})
        source = observations.get("observed_data_source", pd.Series(index=observations.index, dtype=str))
        if not source.astype(str).eq(OBSERVATION_SOURCE_CONTRACT).all():
            issues.append({"scope": "observations", "issue": "unexpected_source"})
        ok = observations["observed_fetch_status"].astype(str).str.lower().eq("ok")
        if not ok.all():
            warnings.append(
                {
                    "scope": "observations",
                    "issue": "unavailable_rows",
                    "handling": "excluded_by_prediction_time_quality_filter",
                    "dates": observations.loc[~ok, "contract_date"].astype(str).tolist(),
                }
            )
        observed_at = pd.to_datetime(observations.loc[ok, "observed_as_of_time_utc"], errors="coerce", utc=True)
        contract = pd.to_datetime(observations.loc[ok, "contract_date"], errors="coerce", utc=True) + pd.Timedelta(hours=3)
        if (observed_at > contract).any():
            issues.append({"scope": "observations", "issue": "post_11am_observation"})
        if pd.to_numeric(observations.get("observed_precip_recent_at_as_of"), errors="coerce").notna().any():
            issues.append({"scope": "observations", "issue": "non_us_precip_amount_not_null"})

    coverage: list[dict[str, Any]] = []
    for provider in providers:
        frame = _load_forecast_provider(data_root, str(provider))
        provider_coverage = provider_modeling_coverage(data_root, str(provider))
        coverage.append(provider_coverage)
        if not provider_coverage["modeling_ready"]:
            issues.append(
                {
                    "scope": str(provider),
                    "issue": "modeling_coverage",
                    "missing_usable_dates": provider_coverage["missing_usable_dates"],
                    "unexpected_dates": provider_coverage["unexpected_dates"],
                }
            )
        for row in frame.itertuples(index=False):
            expected = forecast_timing(str(row.contract_date))
            issued = pd.Timestamp(row.issued_at).to_pydatetime().astimezone(UTC)
            as_of = pd.Timestamp(row.forecast_as_of).to_pydatetime().astimezone(UTC)
            if issued != expected["issue_utc"] or as_of != expected["as_of_utc"]:
                issues.append({"scope": provider, "contract_date": row.contract_date, "issue": "timing_mismatch"})
            if int(row.forecast_hour_min) != 9 or int(row.forecast_hour_max) != 21:
                issues.append({"scope": provider, "contract_date": row.contract_date, "issue": "forecast_hour_mismatch"})

    result = {
        "passed": not issues,
        "expected_target_rows": 2027,
        "label_rows": len(target_labels),
        "observation_rows": len(observations),
        "forecast_coverage": coverage,
        "issues": issues,
        "warnings": warnings,
        "updated_at_utc": datetime.now(UTC),
    }
    _atomic_write_json(data_root / "audit" / "audit_result.json", result)
    _atomic_write_frame(data_root / "audit" / "issues.csv", pd.DataFrame(issues))
    _atomic_write_frame(data_root / "audit" / "forecast_coverage.csv", pd.DataFrame(coverage))
    return result


def _complete_providers(data_root: Path, providers: Sequence[str]) -> tuple[str, ...]:
    complete: list[str] = []
    for provider in providers:
        if provider_modeling_coverage(data_root, str(provider))["modeling_ready"]:
            complete.append(str(provider).lower())
    return tuple(complete)


def fahrenheit_to_celsius(values: Any) -> Any:
    numeric = pd.to_numeric(values, errors="coerce")
    return (numeric - 32.0) * 5.0 / 9.0


def round_half_up_celsius(values: Any) -> Any:
    numeric = pd.to_numeric(values, errors="coerce")
    if np.isscalar(numeric):
        if pd.isna(numeric):
            return pd.NA
        return int(math.floor(float(numeric) + 0.5) if float(numeric) >= 0 else math.ceil(float(numeric) - 0.5))
    rounded = np.where(numeric >= 0, np.floor(numeric + 0.5), np.ceil(numeric - 0.5))
    return pd.Series(pd.array(rounded, dtype="Int64"), index=getattr(values, "index", None))


def floor_celsius_bucket(values: Any) -> Any:
    numeric = pd.to_numeric(values, errors="coerce")
    if np.isscalar(numeric):
        return pd.NA if pd.isna(numeric) else int(math.floor(float(numeric)))
    floored = np.floor(numeric)
    return pd.Series(pd.array(floored, dtype="Int64"), index=getattr(values, "index", None))


def add_celsius_prediction_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if TARGET in out:
        out["actual_high_c"] = fahrenheit_to_celsius(out[TARGET])
    if "predicted_high_f" in out:
        out["predicted_high_c"] = fahrenheit_to_celsius(out["predicted_high_f"])
    if "error_f" in out:
        out["error_c"] = pd.to_numeric(out["error_f"], errors="coerce") * 5.0 / 9.0
    if "absolute_error_f" in out:
        out["absolute_error_c"] = pd.to_numeric(out["absolute_error_f"], errors="coerce") * 5.0 / 9.0
    if {"actual_high_c", "predicted_high_c"}.issubset(out.columns):
        out["actual_bucket_c"] = floor_celsius_bucket(out["actual_high_c"])
        out["predicted_bucket_c"] = floor_celsius_bucket(out["predicted_high_c"])
        out["bucket_error_c"] = out["predicted_bucket_c"] - out["actual_bucket_c"]
        out["exact_bucket_c"] = out["bucket_error_c"].eq(0)
        out["within_1c_bucket"] = out["bucket_error_c"].abs().le(1)
        out["two_bucket_lower_c"] = out["predicted_bucket_c"] - 1
        out["two_bucket_upper_c"] = out["predicted_bucket_c"]
        out["two_bucket_hit_c"] = (
            out["actual_bucket_c"].ge(out["two_bucket_lower_c"])
            & out["actual_bucket_c"].le(out["two_bucket_upper_c"])
        )
    return out


def add_celsius_metric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in list(out.columns):
        if not column.endswith("_f") or not any(token in column for token in ("mae", "rmse", "bias", "error")):
            continue
        out[f"{column[:-2]}_c"] = pd.to_numeric(out[column], errors="coerce") * 5.0 / 9.0
    return out


def celsius_bucket_metrics(
    predictions: pd.DataFrame,
    *,
    group_columns: Sequence[str] = ("method",),
) -> pd.DataFrame:
    frame = add_celsius_prediction_columns(predictions)
    groups = [column for column in group_columns if column in frame]
    required = {*groups, "actual_bucket_c", "predicted_bucket_c"}
    output_columns = [
        *groups,
        "bucket_unit",
        "bucket_width_c",
        "rounding_rule",
        "count",
        "exact_bucket_accuracy",
        "exact_bucket_accuracy_pct",
        "within_1c_accuracy",
        "within_1c_accuracy_pct",
        "two_bucket_accuracy",
        "two_bucket_accuracy_pct",
        "bucket_mae_c",
    ]
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame(columns=output_columns)
    clean = frame.dropna(subset=["actual_bucket_c", "predicted_bucket_c"]).copy()
    if clean.empty:
        return pd.DataFrame(columns=output_columns)
    metrics = (
        clean.groupby(groups, as_index=False)
        .agg(
            count=("bucket_error_c", "size"),
            exact_bucket_accuracy=("exact_bucket_c", "mean"),
            within_1c_accuracy=("within_1c_bucket", "mean"),
            two_bucket_accuracy=("two_bucket_hit_c", "mean"),
            bucket_mae_c=("bucket_error_c", lambda values: float(values.abs().mean())),
        )
        .sort_values(
            ["exact_bucket_accuracy", "two_bucket_accuracy", "bucket_mae_c"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )
    metrics.insert(len(groups), "bucket_unit", "celsius")
    metrics.insert(len(groups) + 1, "bucket_width_c", 1.0)
    metrics.insert(len(groups) + 2, "rounding_rule", "floor_integer_celsius")
    metrics["exact_bucket_accuracy_pct"] = metrics["exact_bucket_accuracy"] * 100.0
    metrics["within_1c_accuracy_pct"] = metrics["within_1c_accuracy"] * 100.0
    metrics["two_bucket_accuracy_pct"] = metrics["two_bucket_accuracy"] * 100.0
    return metrics[output_columns]


def apply_celsius_bucket_metrics_to_summary(
    summary: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    out = summary.drop(columns=["bucket_log_loss", "bucket_accuracy_pct"], errors="ignore").copy()
    bucket_metrics = celsius_bucket_metrics(
        predictions,
        group_columns=("evaluation_scope", "method"),
    )
    if bucket_metrics.empty:
        return out
    keep = [
        "evaluation_scope",
        "method",
        "bucket_unit",
        "bucket_width_c",
        "rounding_rule",
        "exact_bucket_accuracy_pct",
        "within_1c_accuracy_pct",
        "two_bucket_accuracy_pct",
        "bucket_mae_c",
    ]
    return out.merge(bucket_metrics[keep], on=["evaluation_scope", "method"], how="left")


def hong_kong_stacking_config(
    project_root: str | Path,
    data_root: Path,
    *,
    fast_mode: bool = False,
    optuna_trials: int | None = None,
) -> StationStackingConfig:
    trials = int(optuna_trials) if optuna_trials is not None else (8 if fast_mode else 30)
    return StationStackingConfig(
        station_id=STATION_ID,
        project_root=project_root,
        providers=MODEL_PROVIDERS,
        timing_mode=TIMING_MODE,
        feature_version=V20_HKO_GFS_NO_PEAK_FEATURE_VERSION,
        training_profile=TRAINING_PROFILE_V20_ALIGNED,
        target_mode=TARGET_MODE_REMAINING_WARMUP,
        target_source=TARGET_SOURCE_HKO_DAILY_MAX,
        fast_mode=fast_mode,
        optuna_trials=trials,
        stack_optuna_trials=trials,
        optuna_startup_trials=15,
        stack_optuna_startup_trials=15,
        optuna_metric="mae_f",
        optuna_verbose=True,
        hyperparameter_space="wide",
        base_model_methods=("xgboost", "lightgbm", "catboost"),
        stack_enabled=True,
        year_split_folds=V20_EXPANDING_FOLDS,
        year_split_validation_weights={2022: 1.0, 2023: 1.0, 2024: 1.0, 2025: 1.0},
        year_split_test_train_years=(2021, 2025),
        year_split_test_year=2026,
        max_feature_missing_fraction=0.03,
        output_dir=data_root / "models" / "v20_hko_no_peak",
        observation_target_same_station=True,
        observation_source=OBSERVATION_SOURCE_CONTRACT,
    )


def run_hong_kong_year_split_experiment(
    data_root: Path,
    *,
    project_root: str | Path | None = None,
    providers: Sequence[str] = MODEL_PROVIDERS,
    fast_mode: bool = False,
    optuna_trials: int | None = None,
) -> YearSplitExperimentResult:
    requested = tuple(str(provider).lower() for provider in providers)
    if requested != MODEL_PROVIDERS:
        raise RuntimeError(
            f"The {V20_HKO_GFS_NO_PEAK_FEATURE_VERSION} roster is fixed at {MODEL_PROVIDERS}; requested={requested}"
        )
    complete = _complete_providers(data_root, requested)
    if complete != requested:
        coverage = [provider_modeling_coverage(data_root, provider) for provider in requested]
        raise RuntimeError(f"GFS usable-date coverage is incomplete: {coverage}")

    root = Path(project_root).resolve() if project_root is not None else Path(__file__).resolve().parents[1]
    config = hong_kong_stacking_config(
        root,
        data_root,
        fast_mode=fast_mode,
        optuna_trials=optuna_trials,
    )
    features = build_features(data_root, providers=complete)
    modeling_frame, categorical, numeric = _modeling_frame(features, config)
    categorical = [column for column in categorical if column != "actual_high_c"]
    numeric = [column for column in numeric if column != "actual_high_c"]
    if modeling_frame.empty:
        raise RuntimeError("No modeling rows remain after GFS, observation, target, and quality gates")

    folds = config.effective_year_split_folds
    baseline_validation = year_split_baseline_predictions(modeling_frame, config, folds)
    baseline_validation = baseline_validation.loc[baseline_validation["method"].eq("gfs_raw")].copy()
    tuning, validation, selected = tune_year_split_base_models(
        modeling_frame,
        config,
        categorical,
        numeric,
        folds,
    )
    tuning = tuning.drop(columns=["bucket_log_loss"], errors="ignore")
    selected = selected.drop(
        columns=["mean_validation_bucket_log_loss", "worst_validation_bucket_log_loss"],
        errors="ignore",
    )
    validation_all = pd.concat([baseline_validation, validation], ignore_index=True, sort=False)
    test = year_split_test_predictions(
        modeling_frame,
        config,
        categorical,
        numeric,
        selected,
        train_years=config.effective_year_split_test_train_years,
        test_year=config.effective_year_split_test_year,
    )
    test = test.loc[~test["method"].isin(["provider_mean", "provider_median"])].copy()
    stack_test, stack_tuning = tune_year_split_stack_model(
        validation_all,
        test,
        config,
        test_year=config.effective_year_split_test_year,
    )
    stack_tuning = stack_tuning.drop(columns=["bucket_log_loss"], errors="ignore")
    if not stack_test.empty:
        test = pd.concat([test, stack_test], ignore_index=True, sort=False)

    all_predictions = pd.concat([validation_all, test], ignore_index=True, sort=False)
    metrics = apply_celsius_bucket_metrics_to_summary(
        summarize_year_split_predictions(validation_all, test),
        all_predictions,
    )
    scoreboard = year_split_scoreboard(validation_all, test)
    bracket_predictions = add_celsius_prediction_columns(test)
    bracket_metrics = celsius_bucket_metrics(test)
    importance = year_split_feature_importance(
        modeling_frame,
        config,
        categorical,
        numeric,
        selected,
        train_years=config.effective_year_split_test_train_years,
        test_year=config.effective_year_split_test_year,
    )
    feature_columns_frame = pd.DataFrame(
        [{"feature": column, "kind": "categorical"} for column in categorical]
        + [{"feature": column, "kind": "numeric"} for column in numeric]
    )

    output_dir = config.resolved_output_dir()
    paths = {
        "features": output_dir / "HKO_features.csv",
        "year_split_tuning": output_dir / "HKO_year_split_tuning.csv",
        "year_split_validation_predictions": output_dir / "HKO_year_split_validation_predictions.csv",
        "year_split_test_predictions": output_dir / "HKO_year_split_test_predictions.csv",
        "year_split_metrics": output_dir / "HKO_year_split_metrics.csv",
        "year_split_selected_hyperparameters": output_dir / "HKO_year_split_selected_hyperparameters.csv",
        "year_split_feature_importance": output_dir / "HKO_year_split_feature_importance.csv",
        "year_split_stack_tuning": output_dir / "HKO_year_split_stack_tuning.csv",
        "year_split_scoreboard": output_dir / "HKO_year_split_scoreboard.csv",
        "year_split_bracket_predictions": output_dir / "HKO_year_split_bracket_predictions.csv",
        "year_split_bracket_metrics": output_dir / "HKO_year_split_bracket_metrics.csv",
        "feature_columns": output_dir / "HKO_feature_columns.csv",
    }
    for key, frame in {
        "features": features,
        "year_split_tuning": tuning,
        "year_split_validation_predictions": validation_all,
        "year_split_test_predictions": test,
        "year_split_metrics": metrics,
        "year_split_selected_hyperparameters": selected,
        "year_split_feature_importance": importance,
        "year_split_stack_tuning": stack_tuning,
        "year_split_scoreboard": scoreboard,
        "year_split_bracket_predictions": bracket_predictions,
        "year_split_bracket_metrics": bracket_metrics,
        "feature_columns": feature_columns_frame,
    }.items():
        _atomic_write_frame(paths[key], frame)

    dual_validation = add_celsius_prediction_columns(validation_all)
    dual_test = add_celsius_prediction_columns(test)
    _atomic_write_frame(output_dir / "HKO_year_split_validation_predictions_dual_units.csv", dual_validation)
    _atomic_write_frame(output_dir / "HKO_year_split_test_predictions_dual_units.csv", dual_test)
    _atomic_write_frame(output_dir / "HKO_year_split_metrics_dual_units.csv", add_celsius_metric_columns(metrics))
    _atomic_write_frame(output_dir / "HKO_year_split_scoreboard_dual_units.csv", add_celsius_metric_columns(scoreboard))
    _atomic_write_frame(output_dir / "HKO_2026_celsius_bucket_predictions.csv", bracket_predictions)
    _atomic_write_frame(output_dir / "HKO_2026_celsius_bucket_metrics.csv", bracket_metrics)

    return YearSplitExperimentResult(
        station_id=STATION_ID,
        features=features,
        tuning_results=tuning,
        validation_predictions=validation_all,
        test_predictions=test,
        metrics=metrics,
        stack_tuning_results=stack_tuning,
        scoreboard=scoreboard,
        bracket_predictions=bracket_predictions,
        bracket_metrics=bracket_metrics,
        feature_columns=feature_columns_frame,
        selected_hyperparameters=selected,
        feature_importance=importance,
        output_paths=paths,
    )


def run_training(
    data_root: Path,
    *,
    providers: Sequence[str] = MODEL_PROVIDERS,
    fast_mode: bool = False,
    optuna_trials: int | None = None,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    result = run_hong_kong_year_split_experiment(
        data_root,
        project_root=root,
        providers=providers,
        fast_mode=fast_mode,
        optuna_trials=optuna_trials,
    )
    config = hong_kong_stacking_config(root, data_root, fast_mode=fast_mode, optuna_trials=optuna_trials)
    from .export_station_stacking_v2_models import export_station_model_weights

    exported = export_station_model_weights(
        project_root=root,
        station_id=STATION_ID,
        artifact_dir=config.resolved_output_dir(),
        model_version="station_high_regressor_v20_hko_gfs_no_peak_stack",
        timing_mode=TIMING_MODE,
        providers=MODEL_PROVIDERS,
        feature_version=V20_HKO_GFS_NO_PEAK_FEATURE_VERSION,
        training_profile=TRAINING_PROFILE_V20_ALIGNED,
        optuna_metric="mae_f",
        target_mode=TARGET_MODE_REMAINING_WARMUP,
        target_source=TARGET_SOURCE_HKO_DAILY_MAX,
        base_model_methods=("xgboost", "lightgbm", "catboost"),
        stack_enabled=True,
        source_pipeline="notebooks/experiments/station_stacking_v20_hko_no_peak",
        max_feature_missing_fraction=0.03,
        bucket_contract="floor_1c",
        observation_target_same_station=True,
        observation_source=OBSERVATION_SOURCE_CONTRACT,
    )
    manifest = {
        "status": "complete",
        "providers": MODEL_PROVIDERS,
        "feature_version": V20_HKO_GFS_NO_PEAK_FEATURE_VERSION,
        "target_source": TARGET_SOURCE_HKO_DAILY_MAX,
        "observation_target_same_station": True,
        "observation_source": OBSERVATION_SOURCE_CONTRACT,
        "eligibility_contract": "prediction_time_safe_same_station",
        "holdout": {"start": "2026-01-01", "end": END_DATE.isoformat()},
        "holdout_recorded_before_refit": True,
        "scoreboard": str(result.output_paths["year_split_scoreboard"]),
        "bundle_path": str(exported.bundle_path),
        "manifest_path": str(exported.manifest_path),
        "updated_at_utc": datetime.now(UTC),
    }
    _atomic_write_json(config.resolved_output_dir() / "training_manifest.json", manifest)
    return manifest


def live_collection_date(now: datetime | None = None) -> date:
    current = now or datetime.now(ZoneInfo(TIMEZONE))
    return current.astimezone(ZoneInfo(TIMEZONE)).date()


def run_live(data_root: Path, contract_date: date | None = None) -> dict[str, Any]:
    day = contract_date or live_collection_date()
    today = live_collection_date()
    observation = (
        collect_hko_live_observation(data_root, day)
        if day == today
        else backfill_hko_observations_month(data_root, day.strftime("%Y-%m"), force=True)
    )
    gfs = backfill_gfs_day(data_root, day, force=True)
    return {"contract_date": day.isoformat(), "observation": observation, "gfs": gfs, "restricted": "Run restricted-backfill when licensed feeds are configured"}
