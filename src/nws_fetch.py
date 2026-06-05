from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests import Response
from requests import exceptions as request_exceptions
import numpy as np

from .openmeteo_fetch import FORECAST_COLUMNS, _base_forecast_row, target_cutoff_utc
from .hrrr_fetch import _forecast_hours_for_local_day


NBM_TMP = "TMP"
NBM_TMAX = "TMAX"
NBM_PROVIDER = "nbm"
NBM_MODEL_TMAX = "nbm_tmax"
NBM_MODEL_TMP = "nbm"
TRANSIENT_NBM_STATUS_CODES = {429, 500, 502, 503, 504}
NBM_FEATURE_FIELDS: dict[str, dict[str, Any]] = {
    "temp_k_2m": {"variable": "TMP", "level": "2 m above ground", "names": ["t2m", "t", "unknown"]},
    "dewpoint_k_2m": {"variable": "DPT", "level": "2 m above ground", "names": ["d2m", "dpt", "unknown"]},
    "relative_humidity_pct_2m": {"variable": "RH", "level": "2 m above ground", "names": ["r2", "r", "unknown"]},
    "precip_mm_1h": {"variable": "APCP", "level": "surface", "names": ["tp", "unknown"]},
    "cloud_cover_pct": {"variable": "TCDC", "level": "surface", "names": ["tcc", "unknown"]},
    "wind_speed_ms_10m": {"variable": "WIND", "level": "10 m above ground", "names": ["si10", "wind", "unknown"]},
    "wind_direction_deg_10m": {"variable": "WDIR", "level": "10 m above ground", "names": ["wdir10", "wdir", "unknown"]},
    "wind_gust_ms": {"variable": "GUST", "level": "10 m above ground", "names": ["gust", "fg10", "unknown"]},
    "ceiling_m": {"variable": "CEIL", "level": "cloud ceiling", "names": ["ceil", "unknown"]},
    "visibility_m": {"variable": "VIS", "level": "surface", "names": ["vis", "unknown"]},
}


class TransientNbmDownloadError(RuntimeError):
    """Raised when NBM downloading should stop and resume later."""


def fetch_nws_snapshots(
    station_map: pd.DataFrame,
    settings: dict[str, Any],
    raw_dir: str | Path,
    horizons: list[int],
    force_refresh: bool = False,
    skip_keys: set[tuple[str, str, int]] | None = None,
    on_group_rows: Callable[[list[dict[str, Any]]], None] | None = None,
) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    if station_map.empty:
        return pd.DataFrame(columns=FORECAST_COLUMNS)
    lag = int(settings.get("providers", {}).get("publication_lag_minutes", {}).get("nws", 30))
    max_fxx = int(settings.get("nws", {}).get("nbm_max_forecast_hour", 72))
    search_hours = int(settings.get("nws", {}).get("nbm_max_cycle_search_hours", 12))
    variable = str(settings.get("nws", {}).get("nbm_temperature_variable", NBM_TMAX)).upper()
    eligible = station_map.dropna(subset=["station_code", "target_date_local"]).copy()
    eligible = eligible.loc[eligible["target_date_local"].astype(str).str.len() > 0]
    requests_to_fill: list[dict[str, Any]] = []
    skip_keys = skip_keys or set()
    for _, station in eligible.drop_duplicates(subset=["station_code", "target_date_local"]).iterrows():
        if bool(station.get("needs_manual_review")) or station.get("country") != "US":
            continue
        for horizon in horizons:
            target_date = str(station.get("target_date_local"))
            key = (str(station.get("station_code")).upper(), target_date, int(horizon))
            if key in skip_keys:
                continue
            request = _plan_nbm_snapshot_request(
                station,
                settings,
                int(horizon),
                lag,
                max_fxx,
                search_hours,
                variable,
            )
            if request["issue_utc"] is None:
                rows.append(
                    _base_forecast_row(
                        station,
                        NBM_PROVIDER,
                        _nbm_model_label(variable),
                        request["cutoff_utc"],
                        request["issue_local"],
                        request["target_date"],
                        int(horizon),
                        {},
                        request["unavailable_reason"],
                    )
                )
            else:
                requests_to_fill.append(request)
    rows.extend(_fill_nbm_snapshot_requests(requests_to_fill, settings, raw_dir, force_refresh, on_group_rows=on_group_rows))
    frame = pd.DataFrame(rows)
    for column in FORECAST_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    return frame[FORECAST_COLUMNS]


def _nbm_snapshot_row(
    station: pd.Series,
    settings: dict[str, Any],
    raw_dir: Path,
    horizon: int,
    lag_minutes: int,
    force_refresh: bool,
) -> dict[str, Any]:
    max_fxx = int(settings.get("nws", {}).get("nbm_max_forecast_hour", 72))
    search_hours = int(settings.get("nws", {}).get("nbm_max_cycle_search_hours", 12))
    variable = str(settings.get("nws", {}).get("nbm_temperature_variable", NBM_TMAX)).upper()
    request = _plan_nbm_snapshot_request(station, settings, horizon, lag_minutes, max_fxx, search_hours, variable)
    target_date = request["target_date"]
    issue_utc = request["issue_utc"]
    fxx_hours = request["fxx_hours"]
    cutoff_utc = request["cutoff_utc"]
    issue_local = request["issue_local"]
    if issue_utc is None or not fxx_hours:
        return _base_forecast_row(
            station,
            NBM_PROVIDER,
            _nbm_model_label(variable),
            cutoff_utc,
            issue_local,
            target_date,
            horizon,
            {},
            f"unavailable: no NBM cycle found before cutoff {cutoff_utc.isoformat()} covering target day within f{max_fxx}",
        )
    try:
        summary = _extract_nbm_point_summary(station, settings, raw_dir, issue_utc, fxx_hours, force_refresh, variable)
        source = f"nbm {variable.lower()} {issue_utc:%Y-%m-%dT%HZ} f{min(fxx_hours):03d}-f{max(fxx_hours):03d}"
    except TransientNbmDownloadError:
        raise
    except Exception as exc:  # noqa: BLE001
        logging.warning("NBM unavailable for %s %s h%s: %s", station["station_code"], target_date, horizon, exc)
        summary, source = {}, f"unavailable: {exc}"
    return _base_forecast_row(
        station,
        NBM_PROVIDER,
        _nbm_model_label(variable),
        issue_utc,
        issue_local,
        target_date,
        horizon,
        summary,
        source,
    )


def _plan_nbm_snapshot_request(
    station: pd.Series,
    settings: dict[str, Any],
    horizon: int,
    lag_minutes: int,
    max_fxx: int,
    search_hours: int,
    variable: str,
) -> dict[str, Any]:
    target_date = str(station.get("target_date_local"))
    cutoff_local, cutoff_utc = target_cutoff_utc(target_date, station.get("timezone"), horizon, lag_minutes)
    if variable == NBM_TMAX:
        issue_utc, fxx_hours = _choose_nbm_tmax_issue_time(cutoff_utc, target_date, settings, max_fxx, search_hours)
        unavailable = (
            f"unavailable: no NBM TMAX cycle found before cutoff {cutoff_utc.isoformat()} "
            f"covering target-day daytime max within f{max_fxx}"
        )
    else:
        variable = NBM_TMP
        issue_utc, fxx_hours = _choose_nbm_hourly_issue_time(
            cutoff_utc,
            target_date,
            station.get("timezone"),
            settings,
            max_fxx,
            search_hours,
        )
        unavailable = (
            f"unavailable: no NBM hourly cycle found before cutoff {cutoff_utc.isoformat()} "
            f"covering target day within f{max_fxx}"
        )
    issue_local = cutoff_local if issue_utc is None else issue_utc.astimezone(cutoff_local.tzinfo)
    return {
        "station": station,
        "station_code": station.get("station_code"),
        "target_date": target_date,
        "horizon": horizon,
        "cutoff_utc": cutoff_utc,
        "issue_utc": issue_utc,
        "issue_local": issue_local,
        "fxx_hours": fxx_hours,
        "variable": variable,
        "unavailable_reason": unavailable,
    }


def _choose_nbm_tmax_issue_time(
    cutoff_utc: datetime,
    target_date: str,
    settings: dict[str, Any],
    max_fxx: int,
    search_hours: int,
) -> tuple[datetime | None, list[int]]:
    cutoff_utc = cutoff_utc.replace(minute=0, second=0, microsecond=0)
    # NBM TMAX is a 12-hour max ending at 00 UTC. For CONUS airport highs this
    # captures the target date's daytime window using one field instead of 24
    # hourly temperature fields.
    target_tmax_end_utc = datetime.fromisoformat(target_date).replace(tzinfo=UTC) + timedelta(days=1)
    base_url = settings.get("nws", {}).get("nbm_aws_base_url", "https://noaa-nbm-grib2-pds.s3.amazonaws.com")
    product = settings.get("nws", {}).get("nbm_product", "core")
    suffix = settings.get("nws", {}).get("nbm_domain_suffix", "co")
    for offset in range(search_hours + 1):
        candidate = cutoff_utc - timedelta(hours=offset)
        if not _nbm_cycle_allowed(candidate, settings):
            continue
        fxx = int((target_tmax_end_utc - candidate).total_seconds() // 3600)
        if fxx <= 0 or fxx > max_fxx:
            continue
        return candidate, [fxx]
    return None, []


def _choose_nbm_hourly_issue_time(
    cutoff_utc: datetime,
    target_date: str,
    timezone: str,
    settings: dict[str, Any],
    max_fxx: int,
    search_hours: int,
) -> tuple[datetime | None, list[int]]:
    cutoff_utc = cutoff_utc.replace(minute=0, second=0, microsecond=0)
    base_url = settings.get("nws", {}).get("nbm_aws_base_url", "https://noaa-nbm-grib2-pds.s3.amazonaws.com")
    product = settings.get("nws", {}).get("nbm_product", "core")
    suffix = settings.get("nws", {}).get("nbm_domain_suffix", "co")
    for offset in range(search_hours + 1):
        candidate = cutoff_utc - timedelta(hours=offset)
        if not _nbm_cycle_allowed(candidate, settings):
            continue
        fxx_hours = _forecast_hours_for_local_day(candidate, target_date, timezone)
        if not fxx_hours or max(fxx_hours) > max_fxx:
            continue
        if _nbm_inventory_available(base_url, candidate, min(fxx_hours), product, suffix, variable=NBM_TMP) and _nbm_inventory_available(
            base_url, candidate, max(fxx_hours), product, suffix, variable=NBM_TMP
        ):
            return candidate, fxx_hours
    return None, []


def _nbm_cycle_allowed(issue_utc: datetime, settings: dict[str, Any]) -> bool:
    allowed = settings.get("nws", {}).get("nbm_allowed_cycle_hours")
    if not allowed:
        return True
    return issue_utc.hour in {int(hour) for hour in allowed}


def nbm_file_url(base_url: str, issue_time: datetime, fxx: int, product: str = "core", suffix: str = "co") -> str:
    filename = f"blend.t{issue_time:%H}z.{product}.f{fxx:03d}.{suffix}.grib2"
    return f"{base_url.rstrip('/')}/blend.{issue_time:%Y%m%d}/{issue_time:%H}/{product}/{filename}"


def nbm_file_url_candidates(
    base_url: str,
    issue_time: datetime,
    fxx: int,
    product: str = "core",
    suffix: str = "co",
) -> list[str]:
    filename = f"blend.t{issue_time:%H}z.{product}.f{fxx:03d}.{suffix}.grib2"
    root = f"{base_url.rstrip('/')}/blend.{issue_time:%Y%m%d}/{issue_time:%H}"
    return [
        f"{root}/{product}/{filename}",
        f"{root}/grib2/{product}/{filename}",
        f"{root}/grib2/{filename}",
    ]


def _nbm_inventory_available(
    base_url: str,
    issue_time: datetime,
    fxx: int,
    product: str,
    suffix: str,
    variable: str = NBM_TMP,
) -> bool:
    for url in nbm_file_url_candidates(base_url, issue_time, fxx, product=product, suffix=suffix):
        try:
            response = requests.get(f"{url}.idx", timeout=15, headers={"User-Agent": "weather-research/0.1"})
            if response.ok and _nbm_byte_ranges(response.text, variable=variable):
                return True
        except Exception:
            continue
    return False


def _fill_nbm_snapshot_requests(
    requests_to_fill: list[dict[str, Any]],
    settings: dict[str, Any],
    raw_dir: Path,
    force_refresh: bool,
    on_group_rows: Callable[[list[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    if not requests_to_fill:
        return []
    rows: list[dict[str, Any]] = []
    station_meta: dict[str, dict[str, float]] = {}
    for request in requests_to_fill:
        station = request["station"]
        station_meta[str(request["station_code"])] = {"lat": float(station["lat"]), "lon": float(station["lon"])}

    for (issue_utc, variable), group in _group_nbm_requests(requests_to_fill).items():
        fxx_hours = sorted({fxx for request in group for fxx in request["fxx_hours"]})
        group_station_codes = sorted({str(request["station_code"]) for request in group})
        group_station_meta = {code: station_meta[code] for code in group_station_codes}
        try:
            run_values = _extract_nbm_run_points(
                group_station_meta,
                settings,
                raw_dir,
                issue_utc,
                fxx_hours,
                force_refresh,
                variable,
            )
        except TransientNbmDownloadError:
            raise
        except Exception as exc:  # noqa: BLE001
            logging.warning("NBM run unavailable for %s %s: %s", issue_utc, variable, exc)
            run_values = {}
            run_error = f"unavailable: {exc}"
        else:
            run_error = None

        group_rows: list[dict[str, Any]] = []
        for request in group:
            station_code = str(request["station_code"])
            values = [
                run_values.get(station_code, {}).get(fxx)
                for fxx in request["fxx_hours"]
                if run_values.get(station_code, {}).get(fxx) is not None
            ]
            summary = _summarize_nbm_temperature_values(values, variable)
            source = (
                f"nbm {variable.lower()} {issue_utc:%Y-%m-%dT%HZ} "
                f"f{min(request['fxx_hours']):03d}-f{max(request['fxx_hours']):03d}"
                if values
                else run_error or "unavailable: no NBM temperatures extracted for station/window"
            )
            group_rows.append(
                _base_forecast_row(
                    request["station"],
                    NBM_PROVIDER,
                    _nbm_model_label(variable),
                    issue_utc,
                    request["issue_local"],
                    request["target_date"],
                    request["horizon"],
                    summary,
                    source,
                )
            )
        rows.extend(group_rows)
        if on_group_rows is not None:
            on_group_rows(group_rows)
    return rows


def _group_nbm_requests(requests_to_fill: list[dict[str, Any]]) -> dict[tuple[datetime, str], list[dict[str, Any]]]:
    grouped: dict[tuple[datetime, str], list[dict[str, Any]]] = {}
    for request in requests_to_fill:
        grouped.setdefault((request["issue_utc"], request["variable"]), []).append(request)
    return grouped


def _extract_nbm_point_summary(
    station: pd.Series,
    settings: dict[str, Any],
    raw_dir: Path,
    issue_utc: datetime,
    fxx_hours: list[int],
    force_refresh: bool,
    variable: str = NBM_TMP,
) -> dict[str, Any]:
    values = _extract_nbm_run_points(
        {str(station["station_code"]): {"lat": float(station["lat"]), "lon": float(station["lon"])}},
        settings,
        raw_dir,
        issue_utc,
        fxx_hours,
        force_refresh,
        variable,
    ).get(str(station["station_code"]), {})
    return _summarize_nbm_temperature_values([values.get(fxx) for fxx in fxx_hours if values.get(fxx) is not None], variable)


def _extract_nbm_run_points(
    stations: dict[str, dict[str, float]],
    settings: dict[str, Any],
    raw_dir: Path,
    issue_utc: datetime,
    fxx_hours: list[int],
    force_refresh: bool,
    variable: str = NBM_TMP,
) -> dict[str, dict[int, float]]:
    _ensure_ecmwflibs_available()
    try:
        import xarray as xr
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("xarray/cfgrib dependencies are required for NBM extraction") from exc

    values: dict[str, dict[int, float]] = {station_code: {} for station_code in stations}
    grid_indexes: dict[str, tuple[int, int]] | None = None
    base_url = settings.get("nws", {}).get("nbm_aws_base_url", "https://noaa-nbm-grib2-pds.s3.amazonaws.com")
    product = settings.get("nws", {}).get("nbm_product", "core")
    suffix = settings.get("nws", {}).get("nbm_domain_suffix", "co")
    for fxx in fxx_hours:
        try:
            grib = _download_nbm_subset(base_url, issue_utc, fxx, raw_dir, product, suffix, force_refresh, variable, settings)
            with xr.open_dataset(grib, engine="cfgrib", backend_kwargs={"indexpath": f"{grib}.cfidx"}) as ds:
                temp_name = _first_present(ds, ["tmax", "t2m", "t", "unknown"] if variable == NBM_TMAX else ["t2m", "t", "unknown"])
                if not temp_name:
                    continue
                if grid_indexes is None:
                    grid_indexes = _nearest_grid_indexes(ds, stations)
                for station_code, station in stations.items():
                    y, x = grid_indexes[station_code]
                    point = ds.isel(y=y, x=x)
                    value = float(point[temp_name].values)
                    values[station_code][fxx] = _kelvin_to_f(value) if value > 170 else value
        except TransientNbmDownloadError:
            raise
        except Exception as exc:  # noqa: BLE001
            logging.warning("Skipping NBM %s %s f%03d: %s", variable, issue_utc, fxx, exc)
            continue
    return values


def _extract_nbm_run_feature_points(
    stations: dict[str, dict[str, float]],
    settings: dict[str, Any],
    raw_dir: Path,
    issue_utc: datetime,
    fxx_hours: list[int],
    force_refresh: bool,
    feature_fields: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[int, dict[str, float]]]:
    _ensure_ecmwflibs_available()
    try:
        import xarray as xr
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("xarray/cfgrib dependencies are required for NBM extraction") from exc

    fields = feature_fields or NBM_FEATURE_FIELDS
    values: dict[str, dict[int, dict[str, float]]] = {
        station_code: {fxx: {} for fxx in fxx_hours}
        for station_code in stations
    }
    grid_indexes: dict[str, tuple[int, int]] | None = None
    base_url = settings.get("nws", {}).get("nbm_aws_base_url", "https://noaa-nbm-grib2-pds.s3.amazonaws.com")
    product = settings.get("nws", {}).get("nbm_product", "core")
    suffix = settings.get("nws", {}).get("nbm_domain_suffix", "co")
    for fxx in fxx_hours:
        for column, spec in fields.items():
            try:
                grib = _download_nbm_subset(
                    base_url,
                    issue_utc,
                    fxx,
                    raw_dir,
                    product,
                    suffix,
                    force_refresh,
                    str(spec["variable"]),
                    settings,
                    level=str(spec["level"]),
                )
                with xr.open_dataset(grib, engine="cfgrib", backend_kwargs={"indexpath": f"{grib}.cfidx"}) as ds:
                    value_name = _first_present(ds, list(spec.get("names", []))) or _first_data_var(ds)
                    if not value_name:
                        continue
                    if grid_indexes is None:
                        grid_indexes = _nearest_grid_indexes(ds, stations)
                    for station_code in stations:
                        y, x = grid_indexes[station_code]
                        values[station_code][fxx][column] = float(ds.isel(y=y, x=x)[value_name].values)
            except TransientNbmDownloadError:
                raise
            except Exception as exc:  # noqa: BLE001
                logging.warning("Skipping NBM feature %s %s f%03d: %s", column, issue_utc, fxx, exc)
                continue
    return values


def _nearest_grid_indexes(dataset: Any, stations: dict[str, dict[str, float]]) -> dict[str, tuple[int, int]]:
    if "latitude" not in dataset or "longitude" not in dataset:
        return {station_code: (0, 0) for station_code in stations}
    latitudes = np.asarray(dataset.latitude.values)
    longitudes = np.asarray(dataset.longitude.values)
    width = latitudes.shape[-1]
    indexes: dict[str, tuple[int, int]] = {}
    for station_code, station in stations.items():
        longitude = float(station["lon"]) % 360
        distance = (latitudes - float(station["lat"])) ** 2 + (longitudes - longitude) ** 2
        y, x = divmod(int(np.nanargmin(distance)), width)
        indexes[station_code] = (y, x)
    return indexes


def _summarize_nbm_temperature_values(temps_f: list[float], variable: str) -> dict[str, Any]:
    if variable == NBM_TMAX:
        return {
            "forecast_high_f": max(temps_f) if temps_f else None,
            "forecast_low_f": None,
            "forecast_hourly_max_f": max(temps_f) if temps_f else None,
        }
    return {
        "forecast_high_f": max(temps_f) if temps_f else None,
        "forecast_low_f": min(temps_f) if temps_f else None,
        "forecast_hourly_max_f": max(temps_f) if temps_f else None,
    }


def _nbm_model_label(variable: str) -> str:
    return NBM_MODEL_TMAX if variable.upper() == NBM_TMAX else NBM_MODEL_TMP


def _download_nbm_subset(
    base_url: str,
    issue_time: datetime,
    fxx: int,
    raw_dir: Path,
    product: str,
    suffix: str,
    force_refresh: bool,
    variable: str = NBM_TMP,
    settings: dict[str, Any] | None = None,
    level: str | None = None,
) -> Path:
    variable = variable.upper()
    level_label = "" if level is None else "_" + "".join(ch.lower() if ch.isalnum() else "_" for ch in level).strip("_")
    local = raw_dir / f"nbm_{variable.lower()}{level_label}_{issue_time:%Y%m%d%H}_f{fxx:03d}_{suffix}.grib2"
    if local.exists() and local.stat().st_size > 0 and not force_refresh:
        return local
    if local.exists() and local.stat().st_size == 0:
        local.unlink()

    last_error: Exception | None = None
    for url in nbm_file_url_candidates(base_url, issue_time, fxx, product=product, suffix=suffix):
        try:
            idx_response = _nbm_get_with_retries(
                f"{url}.idx",
                settings or {},
                timeout=30,
                headers={"User-Agent": "weather-research/0.1"},
            )
            idx_response.raise_for_status()
            ranges = _nbm_byte_ranges(idx_response.text, variable=variable, level=level)
            with local.open("wb") as handle:
                for start, end in ranges:
                    headers = {"Range": f"bytes={start}-{end}", "User-Agent": "weather-research/0.1"}
                    response = _nbm_get_with_retries(url, settings or {}, timeout=90, headers=headers)
                    response.raise_for_status()
                    handle.write(response.content)
            return local
        except TransientNbmDownloadError:
            if local.exists() and local.stat().st_size == 0:
                local.unlink()
            raise
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if local.exists() and local.stat().st_size == 0:
                local.unlink()
    raise RuntimeError(f"Could not download NBM subset for {issue_time:%Y-%m-%d %HZ} f{fxx:03d}: {last_error}") from last_error


def _nbm_get_with_retries(
    url: str,
    settings: dict[str, Any],
    timeout: int,
    headers: dict[str, str],
) -> Response:
    nws_settings = settings.get("nws", {})
    attempts = max(1, int(nws_settings.get("nbm_download_retries", 4)))
    backoff = max(0.0, float(nws_settings.get("nbm_retry_backoff_seconds", 10)))
    abort_on_network_error = bool(nws_settings.get("nbm_abort_on_network_error", True))
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(url, timeout=timeout, headers=headers)
            if response.status_code in TRANSIENT_NBM_STATUS_CODES:
                response.raise_for_status()
            return response
        except (request_exceptions.ConnectionError, request_exceptions.Timeout) as exc:
            last_error = exc
        except request_exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status not in TRANSIENT_NBM_STATUS_CODES:
                raise
            last_error = exc
        if attempt < attempts:
            sleep_seconds = backoff * attempt
            logging.warning(
                "Transient NBM download failure; retrying %s/%s in %.1fs: %s",
                attempt + 1,
                attempts,
                sleep_seconds,
                last_error,
            )
            if sleep_seconds:
                time.sleep(sleep_seconds)
    message = f"Transient NBM network failure for {url} after {attempts} attempts: {last_error}"
    if abort_on_network_error:
        raise TransientNbmDownloadError(
            f"{message}. Stop here and rerun later; cached NBM files will be reused."
        ) from last_error
    if last_error is not None:
        raise last_error
    raise TransientNbmDownloadError(message)


def _nbm_byte_ranges(idx_text: str, variable: str = NBM_TMP, level: str | None = None) -> list[tuple[int, int]]:
    lines = [line for line in idx_text.splitlines() if line.strip()]
    starts: list[tuple[int, str]] = []
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) >= 2 and parts[1].isdigit():
            starts.append((int(parts[1]), line))
    ranges: list[tuple[int, int]] = []
    variable = variable.upper()
    if variable == NBM_TMAX:
        needle = ":TMAX:2 m above ground:"
    elif level is not None:
        needle = f":{variable}:{level}:"
    else:
        needle = f":{variable}:2 m above ground:" if variable in {NBM_TMP, "DPT", "RH"} else f":{variable}:"
    for i, (start, line) in enumerate(starts):
        lowered = line.lower()
        if needle in line and "ens std dev" not in lowered and "prob " not in lowered and "probability forecast" not in lowered:
            end = starts[i + 1][0] - 1 if i + 1 < len(starts) else start + 2_000_000
            ranges.append((start, end))
    if not ranges:
        raise RuntimeError("Requested NBM variables were not found in .idx inventory")
    return ranges


def _first_present(dataset: Any, names: list[str]) -> str | None:
    for name in names:
        if name in dataset:
            return name
    return None


def _first_data_var(dataset: Any) -> str | None:
    data_vars = list(getattr(dataset, "data_vars", []))
    return data_vars[0] if data_vars else None


def _kelvin_to_f(value: float) -> float:
    return (value - 273.15) * 9 / 5 + 32


def _ensure_ecmwflibs_available() -> None:
    try:
        import ecmwflibs

        root = Path(ecmwflibs.__file__).parent
        os.environ["PATH"] = f"{root};{os.environ.get('PATH', '')}"
        os.environ.setdefault("ECCODES_LIB_DIR", str(root))
        os.add_dll_directory(str(root))
    except Exception:
        return


def _fetch_current_nws_grid_summary(
    station: pd.Series,
    settings: dict[str, Any],
    raw_dir: Path,
    target_date: str,
    force_refresh: bool,
) -> tuple[dict[str, Any], str]:
    ua = settings.get("nws", {}).get("user_agent", "weather-research/0.1")
    points_url = settings.get("nws", {}).get("points_url", "https://api.weather.gov/points/{lat},{lon}").format(
        lat=station["lat"], lon=station["lon"]
    )
    cache = raw_dir / f"nws_grid_{station['station_code']}_{target_date}.json"
    if cache.exists() and not force_refresh:
        payload = json.loads(cache.read_text(encoding="utf-8"))
        source = str(cache)
    else:
        points = requests.get(points_url, timeout=30, headers={"User-Agent": ua, "Accept": "application/geo+json"})
        points.raise_for_status()
        grid_url = points.json()["properties"]["forecastGridData"]
        grid = requests.get(grid_url, timeout=30, headers={"User-Agent": ua, "Accept": "application/geo+json"})
        grid.raise_for_status()
        payload = grid.json()
        cache.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        source = grid_url
    temps = _values_for_date(payload, "temperature", target_date)
    return {
        "forecast_high_f": max(temps) if temps else None,
        "forecast_low_f": min(temps) if temps else None,
        "forecast_hourly_max_f": max(temps) if temps else None,
    }, source


def _values_for_date(payload: dict[str, Any], key: str, target_date: str) -> list[float]:
    values = payload.get("properties", {}).get(key, {}).get("values", [])
    out: list[float] = []
    for item in values:
        valid = str(item.get("validTime", "")).split("/")[0]
        if valid.startswith(target_date) and item.get("value") is not None:
            value = float(item["value"])
            unit = payload.get("properties", {}).get(key, {}).get("uom", "")
            if "degC" in unit:
                value = value * 9 / 5 + 32
            out.append(value)
    return out


def write_nws_snapshots(
    station_map: pd.DataFrame,
    settings: dict[str, Any],
    processed_dir: str | Path,
    raw_dir: str | Path,
    horizons: list[int],
    force_refresh: bool = False,
) -> pd.DataFrame:
    processed = Path(processed_dir)
    processed.mkdir(parents=True, exist_ok=True)
    output_path = processed / "nws_forecast_snapshots.csv"
    existing_good = _load_reusable_nws_forecasts(output_path, force_refresh)
    skip_keys = _forecast_keys(existing_good)
    if skip_keys:
        logging.info("Reusing %s completed NWS/NBM forecast rows from %s", len(skip_keys), output_path)

    checkpoint_rows: list[dict[str, Any]] = []

    def checkpoint(group_rows: list[dict[str, Any]]) -> None:
        checkpoint_rows.extend(group_rows)
        checkpoint_frame = _combine_nws_forecasts(existing_good, pd.DataFrame(checkpoint_rows))
        checkpoint_frame.to_csv(output_path, index=False)

    try:
        fresh = fetch_nws_snapshots(
            station_map,
            settings,
            raw_dir,
            horizons,
            force_refresh=force_refresh,
            skip_keys=skip_keys,
            on_group_rows=checkpoint,
        )
    except TransientNbmDownloadError:
        if checkpoint_rows:
            checkpoint_frame = _combine_nws_forecasts(existing_good, pd.DataFrame(checkpoint_rows))
            checkpoint_frame.to_csv(output_path, index=False)
        raise

    frame = _combine_nws_forecasts(existing_good, fresh)
    frame.to_csv(output_path, index=False)
    return frame


def _load_reusable_nws_forecasts(path: Path, force_refresh: bool) -> pd.DataFrame:
    if force_refresh or not path.exists():
        return pd.DataFrame(columns=FORECAST_COLUMNS)
    try:
        frame = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        logging.warning("Could not load existing NWS snapshot cache %s: %s", path, exc)
        return pd.DataFrame(columns=FORECAST_COLUMNS)
    for column in FORECAST_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    provider = frame["provider"].astype(str).str.lower() if "provider" in frame else pd.Series("nws", index=frame.index)
    model = frame["model"].astype(str).str.lower() if "model" in frame else pd.Series("nbm", index=frame.index)
    good = frame.loc[
        provider.isin(["nws", NBM_PROVIDER])
        & model.str.contains("nbm", case=False, na=False)
        & frame["forecast_high_f"].notna()
    ].copy()
    if not good.empty:
        source = (
            good["source_file_or_url"].fillna("").astype(str)
            if "source_file_or_url" in good
            else pd.Series("", index=good.index)
        )
        good["provider"] = NBM_PROVIDER
        good["model"] = source.map(lambda value: NBM_MODEL_TMAX if "tmax" in value.lower() else NBM_MODEL_TMP)
    return _combine_nws_forecasts(pd.DataFrame(columns=FORECAST_COLUMNS), good)


def _forecast_keys(frame: pd.DataFrame) -> set[tuple[str, str, int]]:
    keys: set[tuple[str, str, int]] = set()
    if frame.empty:
        return keys
    for _, row in frame.iterrows():
        try:
            keys.add((str(row["station_code"]).upper(), str(row["target_date_local"]), int(row["forecast_horizon_hours"])))
        except Exception:
            continue
    return keys


def _combine_nws_forecasts(existing: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    frames = [frame for frame in [existing, fresh] if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame(columns=FORECAST_COLUMNS)
    combined = pd.concat(frames, ignore_index=True)
    for column in FORECAST_COLUMNS:
        if column not in combined:
            combined[column] = pd.NA
    combined["_has_forecast"] = combined["forecast_high_f"].notna()
    combined = combined.sort_values(["station_code", "target_date_local", "forecast_horizon_hours", "_has_forecast"])
    combined = combined.drop_duplicates(
        subset=["station_code", "target_date_local", "forecast_horizon_hours"],
        keep="last",
    )
    combined = combined.drop(columns=["_has_forecast"])
    return combined[FORECAST_COLUMNS]
