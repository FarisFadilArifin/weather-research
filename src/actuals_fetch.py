from __future__ import annotations

import io
import json
import logging
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests


ACTUAL_COLUMNS = [
    "station_code",
    "station_name",
    "airport_name",
    "date_local",
    "actual_high_f",
    "actual_high_time_local",
    "actual_high_time_utc",
    "actual_low_f",
    "source",
    "data_quality_flag",
    "raw_observation_count",
]


def local_day_window(date_local: str | date, timezone: str) -> tuple[datetime, datetime, datetime, datetime]:
    day = date.fromisoformat(date_local) if isinstance(date_local, str) else date_local
    tz = ZoneInfo(timezone)
    start_local = datetime.combine(day, time.min, tzinfo=tz)
    next_local = start_local + timedelta(days=1)
    end_local = next_local - timedelta(microseconds=1)
    return start_local, end_local, start_local.astimezone(UTC), next_local.astimezone(UTC)


def fetch_actual_highs(
    station_map: pd.DataFrame,
    station_registry: pd.DataFrame,
    raw_dir: str | Path,
    force_refresh: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    if station_map.empty or station_registry.empty:
        return pd.DataFrame(columns=ACTUAL_COLUMNS)

    merged = station_map.merge(station_registry, on="station_code", suffixes=("_map", ""), how="left")
    unique = merged.dropna(subset=["station_code", "target_date_local"]).drop_duplicates(
        subset=["station_code", "target_date_local"]
    )
    for station, group in unique.groupby("station_code", dropna=False):
        if not str(station).startswith("K"):
            rows.extend(_fetch_actual_rows_one_by_one(group, raw_dir, force_refresh))
            continue
        rows.extend(_fetch_us_station_actual_rows_batched(group, raw_dir, force_refresh))
    frame = pd.DataFrame(rows)
    for column in ACTUAL_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    return frame[ACTUAL_COLUMNS]


def _fetch_actual_rows_one_by_one(group: pd.DataFrame, raw_dir: Path, force_refresh: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in group.iterrows():
        actual = _fetch_single_actual_row(row, raw_dir, force_refresh)
        if actual:
            rows.append(actual)
    return rows


def _fetch_us_station_actual_rows_batched(group: pd.DataFrame, raw_dir: Path, force_refresh: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    review = group["needs_manual_review"] if "needs_manual_review" in group else pd.Series(False, index=group.index)
    clean = group.loc[~review.fillna(False).astype(bool)].copy()
    if clean.empty:
        return rows
    timezone = clean.iloc[0].get("timezone") or clean.iloc[0].get("timezone_map")
    if not timezone or pd.isna(timezone):
        return _fetch_actual_rows_one_by_one(clean, raw_dir, force_refresh)
    station = str(clean.iloc[0]["station_code"])
    clean["target_date_obj"] = pd.to_datetime(clean["target_date_local"], errors="coerce").dt.date
    clean = clean.dropna(subset=["target_date_obj"])
    rows_by_date = {str(row["target_date_local"]): row for _, row in clean.iterrows()}
    completed: set[str] = set()

    for chunk_start, chunk_end in _date_chunks(sorted(clean["target_date_obj"].unique()), chunk_days=31):
        try:
            obs = fetch_iem_asos_1min_range(station, chunk_start.isoformat(), chunk_end.isoformat(), timezone, raw_dir, force_refresh)
            if obs.empty:
                continue
            obs_dates = obs["valid_local"].dt.date
            for day in sorted(d for d in clean["target_date_obj"].unique() if chunk_start <= d <= chunk_end):
                target_date = day.isoformat()
                day_obs = obs.loc[obs_dates == day].copy()
                if day_obs.empty:
                    continue
                day_obs.attrs["source"] = obs.attrs.get("source", "iem_asos_1min")
                try:
                    rows.append(_observations_to_actual_row(day_obs, rows_by_date[target_date], target_date, timezone))
                    completed.add(target_date)
                except Exception as exc:  # noqa: BLE001
                    logging.info("Batched actuals parse failed for %s %s: %s", station, target_date, exc)
        except Exception as exc:  # noqa: BLE001
            logging.info("Batched IEM 1-minute unavailable for %s %s-%s: %s", station, chunk_start, chunk_end, exc)

    remaining_days = sorted(day for day in clean["target_date_obj"].unique() if day.isoformat() not in completed)
    for chunk_start, chunk_end in _date_chunks(remaining_days, chunk_days=366):
        try:
            obs = fetch_iem_asos_hourly_range(station, chunk_start.isoformat(), chunk_end.isoformat(), timezone, raw_dir, force_refresh)
            if obs.empty:
                continue
            obs_dates = obs["valid_local"].dt.date
            for day in sorted(d for d in remaining_days if chunk_start <= d <= chunk_end):
                target_date = day.isoformat()
                day_obs = obs.loc[obs_dates == day].copy()
                if day_obs.empty:
                    continue
                day_obs.attrs["source"] = obs.attrs.get("source", "iem_asos_hourly")
                try:
                    rows.append(_observations_to_actual_row(day_obs, rows_by_date[target_date], target_date, timezone))
                    completed.add(target_date)
                except Exception as exc:  # noqa: BLE001
                    logging.info("Batched hourly actuals parse failed for %s %s: %s", station, target_date, exc)
        except Exception as exc:  # noqa: BLE001
            logging.info("Batched IEM hourly unavailable for %s %s-%s: %s", station, chunk_start, chunk_end, exc)

    for _, row in clean.iterrows():
        target_date = str(row["target_date_local"])
        if target_date not in completed:
            actual = _fetch_single_actual_row(row, raw_dir, force_refresh)
            if actual:
                rows.append(actual)
    return rows


def _fetch_single_actual_row(row: pd.Series, raw_dir: Path, force_refresh: bool) -> dict[str, Any]:
    if bool(row.get("needs_manual_review")):
        return {}
    station = str(row["station_code"])
    target_date = str(row["target_date_local"])
    timezone = row.get("timezone") or row.get("timezone_map")
    if not timezone or pd.isna(timezone):
        return {}
    try:
        obs = fetch_station_observations(station, target_date, timezone, raw_dir, force_refresh=force_refresh)
        return _observations_to_actual_row(obs, row, target_date, timezone)
    except Exception as exc:  # noqa: BLE001 - keep the pipeline running and expose quality flags.
        logging.warning("Actuals unavailable for %s %s: %s", station, target_date, exc)
        return {
            "station_code": station,
            "station_name": row.get("station_name"),
            "airport_name": row.get("airport_name"),
            "date_local": target_date,
            "actual_high_f": pd.NA,
            "actual_high_time_local": pd.NA,
            "actual_high_time_utc": pd.NA,
            "actual_low_f": pd.NA,
            "source": "unavailable",
            "data_quality_flag": str(exc),
            "raw_observation_count": 0,
        }


def fetch_station_observations(
    station_code: str,
    target_date_local: str,
    timezone: str,
    raw_dir: Path,
    force_refresh: bool = False,
) -> pd.DataFrame:
    if station_code.startswith("K"):
        try:
            frame = fetch_iem_asos_1min(station_code, target_date_local, timezone, raw_dir, force_refresh=force_refresh)
            if not frame.empty:
                frame.attrs["source"] = "iem_asos_1min"
                return frame
        except Exception as exc:  # noqa: BLE001
            logging.info("IEM 1-minute unavailable for %s %s: %s", station_code, target_date_local, exc)
        frame = fetch_iem_asos_hourly(station_code, target_date_local, timezone, raw_dir, force_refresh=force_refresh)
        if not frame.empty:
            frame.attrs["source"] = "iem_asos_hourly"
            return frame
    raise RuntimeError("No verified official actual-temperature source implemented for this station")


def fetch_iem_asos_1min(
    station_code: str,
    target_date_local: str,
    timezone: str,
    raw_dir: Path,
    force_refresh: bool = False,
) -> pd.DataFrame:
    start_local, _, start_utc, end_utc = local_day_window(target_date_local, timezone)
    station = station_code[1:] if station_code.startswith("K") else station_code
    params = {
        "station": station,
        "vars": "tmpf",
        "sts": start_utc.strftime("%Y-%m-%dT%H:%MZ"),
        "ets": end_utc.strftime("%Y-%m-%dT%H:%MZ"),
        "sample": "1min",
        "what": "download",
        "tz": "UTC",
        "delim": "comma",
    }
    cache = raw_dir / f"iem_asos_1min_{station_code}_{target_date_local}.csv"
    text = _get_text("https://mesonet.agron.iastate.edu/cgi-bin/request/asos1min.py", params, cache, force_refresh)
    return _parse_iem_csv(text, start_local.tzinfo)


def fetch_iem_asos_1min_range(
    station_code: str,
    start_date_local: str,
    end_date_local: str,
    timezone: str,
    raw_dir: Path,
    force_refresh: bool = False,
) -> pd.DataFrame:
    start_local, _, start_utc, _ = local_day_window(start_date_local, timezone)
    _, _, _, end_utc = local_day_window(end_date_local, timezone)
    station = station_code[1:] if station_code.startswith("K") else station_code
    params = {
        "station": station,
        "vars": "tmpf",
        "sts": start_utc.strftime("%Y-%m-%dT%H:%MZ"),
        "ets": end_utc.strftime("%Y-%m-%dT%H:%MZ"),
        "sample": "1min",
        "what": "download",
        "tz": "UTC",
        "delim": "comma",
    }
    cache = raw_dir / f"iem_asos_1min_{station_code}_{start_date_local}_{end_date_local}.csv"
    text = _get_text("https://mesonet.agron.iastate.edu/cgi-bin/request/asos1min.py", params, cache, force_refresh)
    frame = _parse_iem_csv(text, start_local.tzinfo)
    frame.attrs["source"] = "iem_asos_1min"
    return frame


def fetch_iem_asos_hourly(
    station_code: str,
    target_date_local: str,
    timezone: str,
    raw_dir: Path,
    force_refresh: bool = False,
) -> pd.DataFrame:
    start_local, _, start_utc, end_utc = local_day_window(target_date_local, timezone)
    station = station_code
    params = {
        "station": station,
        "data": "tmpf",
        "year1": start_utc.year,
        "month1": start_utc.month,
        "day1": start_utc.day,
        "hour1": start_utc.hour,
        "minute1": start_utc.minute,
        "year2": end_utc.year,
        "month2": end_utc.month,
        "day2": end_utc.day,
        "hour2": end_utc.hour,
        "minute2": end_utc.minute,
        "tz": "Etc/UTC",
        "format": "onlycomma",
        "latlon": "no",
        "elev": "no",
        "missing": "M",
        "trace": "T",
        "direct": "no",
        "report_type": "3",
    }
    cache = raw_dir / f"iem_asos_hourly_{station_code}_{target_date_local}.csv"
    text = _get_text("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py", params, cache, force_refresh)
    return _parse_iem_csv(text, start_local.tzinfo)


def fetch_iem_asos_hourly_range(
    station_code: str,
    start_date_local: str,
    end_date_local: str,
    timezone: str,
    raw_dir: Path,
    force_refresh: bool = False,
) -> pd.DataFrame:
    start_local, _, start_utc, _ = local_day_window(start_date_local, timezone)
    _, _, _, end_utc = local_day_window(end_date_local, timezone)
    station = station_code
    params = {
        "station": station,
        "data": "tmpf",
        "year1": start_utc.year,
        "month1": start_utc.month,
        "day1": start_utc.day,
        "hour1": start_utc.hour,
        "minute1": start_utc.minute,
        "year2": end_utc.year,
        "month2": end_utc.month,
        "day2": end_utc.day,
        "hour2": end_utc.hour,
        "minute2": end_utc.minute,
        "tz": "Etc/UTC",
        "format": "onlycomma",
        "latlon": "no",
        "elev": "no",
        "missing": "M",
        "trace": "T",
        "direct": "no",
        "report_type": "3",
    }
    cache = raw_dir / f"iem_asos_hourly_{station_code}_{start_date_local}_{end_date_local}.csv"
    text = _get_text("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py", params, cache, force_refresh)
    frame = _parse_iem_csv(text, start_local.tzinfo)
    frame.attrs["source"] = "iem_asos_hourly"
    return frame


def _get_text(url: str, params: dict[str, Any], cache: Path, force_refresh: bool) -> str:
    if cache.exists() and not force_refresh:
        return cache.read_text(encoding="utf-8")
    response = requests.get(url, params=params, timeout=60, headers={"User-Agent": "weather-research/0.1"})
    response.raise_for_status()
    text = response.text
    cache.write_text(text, encoding="utf-8")
    return text


def _parse_iem_csv(text: str, local_tz: ZoneInfo | None) -> pd.DataFrame:
    if not text.strip() or text.lstrip().startswith("ERROR"):
        return pd.DataFrame()
    frame = pd.read_csv(io.StringIO(text), comment="#")
    if frame.empty:
        return frame
    valid_cols = {c.lower(): c for c in frame.columns}
    time_col = valid_cols.get("valid") or valid_cols.get("utc_valid")
    if time_col is None:
        time_col = next((c for c in frame.columns if c.lower().startswith("valid")), None)
    if time_col is None:
        return pd.DataFrame()
    temp_col = valid_cols.get("tmpf") or valid_cols.get("tmpc")
    if temp_col is None:
        return pd.DataFrame()
    frame["valid_utc"] = pd.to_datetime(frame[time_col], utc=True, errors="coerce")
    frame["tmpf"] = pd.to_numeric(frame[temp_col], errors="coerce")
    if temp_col.lower() == "tmpc":
        frame["tmpf"] = frame["tmpf"] * 9 / 5 + 32
    frame = frame.dropna(subset=["valid_utc", "tmpf"])
    if local_tz:
        frame["valid_local"] = frame["valid_utc"].dt.tz_convert(local_tz)
    return frame[["valid_utc", "valid_local", "tmpf"]]


def _observations_to_actual_row(obs: pd.DataFrame, meta: pd.Series, target_date: str, timezone: str) -> dict[str, Any]:
    if obs.empty:
        raise RuntimeError("No observations returned")
    day = date.fromisoformat(target_date)
    local_dates = obs["valid_local"].dt.date
    obs = obs.loc[local_dates == day].copy()
    if obs.empty:
        raise RuntimeError("No observations in local target-day window")
    high_idx = obs["tmpf"].idxmax()
    low_idx = obs["tmpf"].idxmin()
    high_time_utc = obs.loc[high_idx, "valid_utc"]
    high_time_local = obs.loc[high_idx, "valid_local"]
    count = int(obs["tmpf"].count())
    expected = 18 if obs.attrs.get("source") == "iem_asos_hourly" else 24 * 60 * 0.75
    quality = "ok" if count >= expected else "sparse_observations"
    return {
        "station_code": meta.get("station_code"),
        "station_name": meta.get("station_name"),
        "airport_name": meta.get("airport_name"),
        "date_local": target_date,
        "actual_high_f": round(float(obs.loc[high_idx, "tmpf"]), 2),
        "actual_high_time_local": high_time_local.isoformat(),
        "actual_high_time_utc": high_time_utc.isoformat(),
        "actual_low_f": round(float(obs.loc[low_idx, "tmpf"]), 2),
        "source": obs.attrs.get("source", "unknown"),
        "data_quality_flag": quality,
        "raw_observation_count": count,
    }


def _date_chunks(dates: list[date], chunk_days: int) -> list[tuple[date, date]]:
    if not dates:
        return []
    chunks: list[tuple[date, date]] = []
    current = min(dates)
    final = max(dates)
    while current <= final:
        chunk_end = min(current + timedelta(days=chunk_days - 1), final)
        chunks.append((current, chunk_end))
        current = chunk_end + timedelta(days=1)
    return chunks


def write_actual_highs(
    station_map: pd.DataFrame,
    station_registry: pd.DataFrame,
    processed_dir: str | Path,
    raw_dir: str | Path,
    force_refresh: bool = False,
) -> pd.DataFrame:
    processed = Path(processed_dir)
    processed.mkdir(parents=True, exist_ok=True)
    frame = fetch_actual_highs(station_map, station_registry, raw_dir, force_refresh=force_refresh)
    frame.to_csv(processed / "actual_highs.csv", index=False)
    return frame
