from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


_DIRECT_NWP_IDX_CACHE: dict[str, str] = {}
TRANSIENT_DIRECT_NWP_STATUS_CODES = {429, 500, 502, 503, 504}
GFS_V16_LAYOUT_START_UTC = datetime(2021, 3, 23, tzinfo=UTC)


class TransientDirectNwpDownloadError(RuntimeError):
    """Raised when direct NWP downloading should pause and retry later."""


DIRECT_NWP_FEATURES: dict[str, dict[str, Any]] = {
    "temp_k_2m": {
        "patterns": [":TMP:2 m above ground:"],
        "names": ["t2m", "t", "unknown"],
    },
    "dewpoint_k_2m": {
        "patterns": [":DPT:2 m above ground:"],
        "names": ["d2m", "dpt", "unknown"],
    },
    "relative_humidity_pct_2m": {
        "patterns": [":RH:2 m above ground:"],
        "names": ["r2", "r", "unknown"],
    },
    "wind_u_ms_10m": {
        "patterns": [":UGRD:10 m above ground:"],
        "names": ["u10", "u", "unknown"],
    },
    "wind_v_ms_10m": {
        "patterns": [":VGRD:10 m above ground:"],
        "names": ["v10", "v", "unknown"],
    },
    "wind_gust_ms": {
        "patterns": [":GUST:surface:", ":GUST:10 m above ground:"],
        "names": ["gust", "fg10", "unknown"],
    },
    "precip_mm_1h": {
        "patterns": [":APCP:surface:"],
        "names": ["tp", "unknown"],
    },
    "cloud_cover_pct": {
        "patterns": [":TCDC:entire atmosphere:", ":TCDC:surface:"],
        "names": ["tcc", "unknown"],
    },
    "ceiling_m": {
        "patterns": [":HGT:cloud ceiling:", ":CEIL:cloud ceiling:"],
        "names": ["gh", "ceil", "unknown"],
    },
    "visibility_m": {
        "patterns": [":VIS:surface:"],
        "names": ["vis", "unknown"],
    },
    "shortwave_radiation_w_m2": {
        "patterns": [":DSWRF:surface:"],
        "names": ["dswrf", "unknown"],
    },
    "boundary_layer_cloud_cover_pct": {
        "patterns": [":TCDC:boundary layer cloud layer:"],
        "names": ["tcc", "unknown"],
    },
    "low_cloud_cover_pct": {
        "patterns": [":LCDC:low cloud layer:"],
        "names": ["lcc", "unknown"],
    },
    "mid_cloud_cover_pct": {
        "patterns": [":MCDC:middle cloud layer:"],
        "names": ["mcc", "unknown"],
    },
    "high_cloud_cover_pct": {
        "patterns": [":HCDC:high cloud layer:"],
        "names": ["hcc", "unknown"],
    },
    "pbl_height_m": {
        "patterns": [":HPBL:surface:"],
        "names": ["hpbl", "unknown"],
    },
    "temp_k_925mb": {
        "patterns": [":TMP:925 mb:"],
        "names": ["t", "unknown"],
    },
    "temp_k_850mb": {
        "patterns": [":TMP:850 mb:"],
        "names": ["t", "unknown"],
    },
    "pwat_kg_m2": {
        "patterns": [":PWAT:entire atmosphere"],
        "names": ["pwat", "unknown"],
    },
    "cape_j_kg_surface": {
        "patterns": [":CAPE:surface:"],
        "names": ["cape", "unknown"],
    },
    "cin_j_kg_surface": {
        "patterns": [":CIN:surface:"],
        "names": ["cin", "unknown"],
    },
}


def direct_nwp_file_url(model: str, issue_time: datetime, fxx: int) -> str:
    model = model.lower()
    if model == "hrrr":
        return (
            "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/"
            f"hrrr.{issue_time:%Y%m%d}/conus/hrrr.t{issue_time:%H}z.wrfsfcf{fxx:02d}.grib2"
        )
    if model == "gfs":
        layout = "atmos/" if _as_utc(issue_time) >= GFS_V16_LAYOUT_START_UTC else ""
        return (
            "https://noaa-gfs-bdp-pds.s3.amazonaws.com/"
            f"gfs.{issue_time:%Y%m%d}/{issue_time:%H}/{layout}"
            f"gfs.t{issue_time:%H}z.pgrb2.0p25.f{fxx:03d}"
        )
    if model == "rap":
        return (
            "https://noaa-rap-pds.s3.amazonaws.com/"
            f"rap.{issue_time:%Y%m%d}/rap.t{issue_time:%H}z.awip32f{fxx:02d}.grib2"
        )
    raise ValueError(f"Unsupported direct NWP model: {model}")


def extract_direct_nwp_run_feature_points(
    stations: dict[str, dict[str, float]],
    model: str,
    raw_dir: str | Path,
    issue_utc: datetime,
    fxx_hours: list[int] | tuple[int, ...],
    force_refresh: bool = False,
    feature_fields: list[str] | None = None,
) -> dict[str, dict[int, dict[str, float]]]:
    _ensure_ecmwflibs_available()
    try:
        import xarray as xr
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("xarray/cfgrib dependencies are required for direct NWP extraction") from exc

    model = model.lower()
    raw_path = Path(raw_dir) / model
    raw_path.mkdir(parents=True, exist_ok=True)
    fields = feature_fields or list(DIRECT_NWP_FEATURES)
    requested_fxx = sorted({int(value) for value in fxx_hours})
    extraction_fxx = _available_direct_nwp_fxx_hours(model, issue_utc, requested_fxx)
    values: dict[str, dict[int, dict[str, float]]] = {station_code: {} for station_code in stations}
    grid_indexers: dict[str, dict[str, int]] | None = None
    prefetched = _prefetch_direct_nwp_subsets(
        model,
        issue_utc,
        extraction_fxx,
        raw_path,
        fields,
        force_refresh=force_refresh,
    )

    for fxx in extraction_fxx:
        hourly_by_station: dict[str, dict[str, float]] = {station_code: {} for station_code in stations}
        for field in fields:
            spec = DIRECT_NWP_FEATURES[field]
            try:
                grib = prefetched[(int(fxx), field)]
                if isinstance(grib, Exception):
                    raise grib
                with xr.open_dataset(grib, engine="cfgrib", backend_kwargs={"indexpath": ""}) as ds:
                    var_name = _first_present(ds, spec["names"]) or _first_data_var(ds)
                    if var_name is None:
                        continue
                    if grid_indexers is None:
                        grid_indexers = _nearest_point_indexers(ds, stations)
                    for station_code, station in stations.items():
                        indexer = grid_indexers.get(station_code) if grid_indexers else None
                        value = (
                            _point_value_from_indexer(ds, var_name, indexer)
                            if indexer is not None
                            else _point_value(ds, var_name, station["lat"], station["lon"])
                        )
                        if value is not None and np.isfinite(value):
                            hourly_by_station[station_code][field] = float(value)
            except TransientDirectNwpDownloadError:
                raise
            except Exception as exc:  # noqa: BLE001
                logging.warning("Skipping direct %s %s f%03d %s: %s", model.upper(), issue_utc, fxx, field, exc)
                continue
        for station_code, fields_for_hour in hourly_by_station.items():
            if "wind_u_ms_10m" in fields_for_hour and "wind_v_ms_10m" in fields_for_hour:
                u = fields_for_hour["wind_u_ms_10m"]
                v = fields_for_hour["wind_v_ms_10m"]
                fields_for_hour["wind_speed_ms_10m"] = float(np.sqrt(u * u + v * v))
                fields_for_hour["wind_direction_deg_10m"] = float((270 - np.degrees(np.arctan2(v, u))) % 360)
            if fields_for_hour:
                values[station_code][int(fxx)] = fields_for_hour
    if model == "gfs" and _as_utc(issue_utc) < GFS_V16_LAYOUT_START_UTC:
        return _interpolate_legacy_gfs_values(values, requested_fxx)
    if model == "gfs":
        return _incrementalize_gfs_precip(values, requested_fxx)
    return values


def _prefetch_direct_nwp_subsets(
    model: str,
    issue_utc: datetime,
    fxx_hours: list[int],
    raw_path: Path,
    fields: list[str],
    *,
    force_refresh: bool,
) -> dict[tuple[int, str], Path | Exception]:
    workers = max(1, int(os.getenv("WEATHER_RESEARCH_DIRECT_NWP_WORKERS", "8")))
    # Fetch each inventory once before field workers fan out. This prevents
    # seven concurrent requests for the same .idx file on a cold cache.
    with ThreadPoolExecutor(max_workers=min(workers, max(1, len(fxx_hours)))) as executor:
        inventory_futures = {
            executor.submit(_direct_nwp_idx_text, direct_nwp_file_url(model, issue_utc, fxx)): fxx
            for fxx in fxx_hours
        }
        for future in as_completed(inventory_futures):
            future.result()

    results: dict[tuple[int, str], Path | Exception] = {}
    tasks = [(fxx, field) for fxx in fxx_hours for field in fields]
    with ThreadPoolExecutor(max_workers=min(workers, max(1, len(tasks)))) as executor:
        futures = {
            executor.submit(
                _download_direct_nwp_variable_subset,
                model,
                issue_utc,
                fxx,
                raw_path,
                field,
                DIRECT_NWP_FEATURES[field]["patterns"],
                force_refresh,
            ): (fxx, field)
            for fxx, field in tasks
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception as exc:  # noqa: BLE001
                results[key] = exc
    return results


def _available_direct_nwp_fxx_hours(model: str, issue_utc: datetime, requested_fxx: list[int]) -> list[int]:
    if not requested_fxx:
        return []
    if model.lower() != "gfs":
        return requested_fxx
    if _as_utc(issue_utc) >= GFS_V16_LAYOUT_START_UTC:
        baseline = max(0, min(requested_fxx) - 1)
        return sorted(set([baseline, *requested_fxx]))
    lower = max(0, ((min(requested_fxx) - 1) // 3) * 3)
    upper = ((max(requested_fxx) + 2) // 3) * 3
    return list(range(lower, upper + 1, 3))


def _interpolate_legacy_gfs_values(
    values: dict[str, dict[int, dict[str, float]]],
    requested_fxx: list[int],
) -> dict[str, dict[int, dict[str, float]]]:
    """Interpolate pre-v16 three-hour GFS fields onto the requested hourly grid.

    APCP is cumulative in this archive. Linear interpolation of the cumulative
    curve preserves the period total; downstream differencing produces an
    explicitly approximate hourly distribution. Wind speed/direction are
    recomputed from interpolated U/V rather than interpolated circular angles.
    """
    output: dict[str, dict[int, dict[str, float]]] = {}
    derived_wind = {"wind_speed_ms_10m", "wind_direction_deg_10m"}
    for station_code, by_hour in values.items():
        available_fields = sorted(
            {field for fields in by_hour.values() for field in fields}
            - derived_wind
            - {"precip_mm_1h"}
        )
        station_output: dict[int, dict[str, float]] = {hour: {} for hour in requested_fxx}
        for field in available_fields:
            samples = pd.Series(
                {hour: fields[field] for hour, fields in by_hour.items() if field in fields},
                dtype=float,
            ).sort_index()
            if samples.empty:
                continue
            index = sorted(set(samples.index.astype(int)) | set(requested_fxx))
            interpolated = samples.reindex(index).interpolate(method="index", limit_area="inside")
            for hour in requested_fxx:
                value = interpolated.get(hour)
                if value is not None and np.isfinite(value):
                    station_output[hour][field] = float(value)
        for hour, fields in station_output.items():
            if "wind_u_ms_10m" in fields and "wind_v_ms_10m" in fields:
                u = fields["wind_u_ms_10m"]
                v = fields["wind_v_ms_10m"]
                fields["wind_speed_ms_10m"] = float(np.hypot(u, v))
                fields["wind_direction_deg_10m"] = float((270 - np.degrees(np.arctan2(v, u))) % 360)
        precip_samples = pd.Series(
            {hour: fields["precip_mm_1h"] for hour, fields in by_hour.items() if "precip_mm_1h" in fields},
            dtype=float,
        ).sort_index()
        if not precip_samples.empty:
            baseline_hour = max(0, min(requested_fxx) - 1)
            index = sorted(set(precip_samples.index.astype(int)) | set(requested_fxx) | {baseline_hour})
            cumulative = precip_samples.reindex(index).interpolate(method="index", limit_area="inside")
            for hour in requested_fxx:
                current = cumulative.get(hour)
                previous = cumulative.get(max(0, hour - 1)) if hour > 0 else 0.0
                if current is not None and previous is not None and np.isfinite(current) and np.isfinite(previous):
                    station_output[hour]["precip_mm_1h"] = float(max(0.0, current - previous))
                    station_output[hour]["_precip_is_incremental"] = 1.0
        output[station_code] = {hour: fields for hour, fields in station_output.items() if fields}
    return output


def _incrementalize_gfs_precip(
    values: dict[str, dict[int, dict[str, float]]],
    requested_fxx: list[int],
) -> dict[str, dict[int, dict[str, float]]]:
    output: dict[str, dict[int, dict[str, float]]] = {}
    requested = set(requested_fxx)
    for station_code, by_hour in values.items():
        station_output = {hour: dict(fields) for hour, fields in by_hour.items() if hour in requested}
        samples = sorted(
            (hour, float(fields["precip_mm_1h"]))
            for hour, fields in by_hour.items()
            if "precip_mm_1h" in fields and np.isfinite(fields["precip_mm_1h"])
        )
        decreases = sum(1 for (_, previous), (_, current) in zip(samples, samples[1:], strict=False) if current + 0.01 < previous)
        cumulative = len(samples) >= 2 and decreases <= max(1, len(samples) // 4)
        sample_map = dict(samples)
        for hour in requested_fxx:
            if hour not in station_output or hour not in sample_map:
                continue
            value = sample_map[hour]
            if cumulative:
                previous_hours = [candidate for candidate in sample_map if candidate < hour]
                previous = sample_map[max(previous_hours)] if previous_hours else 0.0
                value = max(0.0, value - previous)
            station_output[hour]["precip_mm_1h"] = float(value)
            station_output[hour]["_precip_is_incremental"] = 1.0
        output[station_code] = station_output
    return output


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _nearest_point_indexers(ds: Any, stations: dict[str, dict[str, float]]) -> dict[str, dict[str, int]]:
    if "latitude" in ds and "longitude" in ds:
        latv = ds["latitude"]
        lonv = ds["longitude"]
    elif "lat" in ds and "lon" in ds:
        latv = ds["lat"]
        lonv = ds["lon"]
    else:
        return {}

    out: dict[str, dict[str, int]] = {}
    if getattr(latv, "ndim", 0) == 1 and getattr(lonv, "ndim", 0) == 1:
        lat_dim = latv.dims[0]
        lon_dim = lonv.dims[0]
        lat_values = np.asarray(latv.values)
        lon_values = np.asarray(lonv.values)
        for station_code, station in stations.items():
            longitude = float(station["lon"]) % 360
            out[station_code] = {
                lat_dim: int(np.abs(lat_values - float(station["lat"])).argmin()),
                lon_dim: int(np.abs(lon_values - longitude).argmin()),
            }
        return out

    lat_values = np.asarray(latv.values)
    lon_values = np.asarray(lonv.values)
    dims = latv.dims
    for station_code, station in stations.items():
        longitude = float(station["lon"]) % 360
        dist = (lat_values - float(station["lat"])) ** 2 + (lon_values - longitude) ** 2
        y, x = np.unravel_index(int(np.nanargmin(dist)), dist.shape)
        out[station_code] = {dims[-2]: int(y), dims[-1]: int(x)}
    return out


def _point_value_from_indexer(ds: Any, var_name: str, indexer: dict[str, int]) -> float | None:
    return _scalar_value(ds[var_name].isel(indexer))


def _download_direct_nwp_variable_subset(
    model: str,
    issue_time: datetime,
    fxx: int,
    raw_dir: Path,
    field: str,
    patterns: list[str],
    force_refresh: bool = False,
) -> Path:
    local = raw_dir / f"{model}_{field}_{issue_time:%Y%m%d%H}_f{fxx:03d}.grib2"
    local.parent.mkdir(parents=True, exist_ok=True)
    if local.exists() and local.stat().st_size > 0 and not force_refresh:
        return local
    if local.exists():
        local.unlink()

    url = direct_nwp_file_url(model, issue_time, fxx)
    idx_text = _direct_nwp_idx_text(url)
    ranges = _byte_ranges_for_patterns(idx_text, patterns)
    tmp = local.with_name(f".{local.name}.{os.getpid()}.tmp")
    if tmp.exists():
        tmp.unlink()
    try:
        with tmp.open("wb") as handle:
            for start, end in ranges:
                headers = {"Range": f"bytes={start}-{end}", "User-Agent": "weather-research/0.1"}
                response = _get_with_retries(url, headers=headers, timeout=90)
                handle.write(response.content)
        os.replace(tmp, local)
    finally:
        if tmp.exists():
            tmp.unlink()
    return local


def _byte_ranges_for_patterns(idx_text: str, patterns: list[str]) -> list[tuple[int, int]]:
    lines = [line for line in idx_text.splitlines() if line.strip()]
    starts: list[tuple[int, str]] = []
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) >= 2 and parts[1].isdigit():
            starts.append((int(parts[1]), line))
    ranges: list[tuple[int, int]] = []
    for i, (start, line) in enumerate(starts):
        lowered = line.lower()
        if "ens std dev" in lowered or "prob " in lowered or "probability forecast" in lowered:
            continue
        if any(pattern in line for pattern in patterns):
            end = starts[i + 1][0] - 1 if i + 1 < len(starts) else start + 2_000_000
            ranges.append((start, end))
            break
    if not ranges:
        raise RuntimeError(f"Requested variables were not found in inventory: {patterns}")
    return ranges


def _point_value(ds: Any, var_name: str, lat: float, lon: float) -> float | None:
    if "latitude" in ds and "longitude" in ds:
        latv = ds["latitude"]
        lonv = ds["longitude"]
    elif "lat" in ds and "lon" in ds:
        latv = ds["lat"]
        lonv = ds["lon"]
    else:
        return _scalar_value(ds[var_name])

    longitude = lon % 360
    if getattr(latv, "ndim", 0) == 1 and getattr(lonv, "ndim", 0) == 1:
        lat_dim = latv.dims[0]
        lon_dim = lonv.dims[0]
        lat_idx = int(np.abs(latv.values - lat).argmin())
        lon_idx = int(np.abs(lonv.values - longitude).argmin())
        return _scalar_value(ds[var_name].isel({lat_dim: lat_idx, lon_dim: lon_idx}))

    dist = (latv - lat) ** 2 + (lonv - longitude) ** 2
    flat_idx = int(dist.values.argmin())
    y, x = np.unravel_index(flat_idx, dist.shape)
    dims = latv.dims
    indexer = {dims[-2]: int(y), dims[-1]: int(x)}
    return _scalar_value(ds[var_name].isel(indexer))


def _scalar_value(value: Any) -> float | None:
    arr = np.asarray(getattr(value, "values", value)).astype(float)
    if arr.size == 0:
        return None
    scalar = float(arr.reshape(-1)[0])
    return scalar if np.isfinite(scalar) else None


def _first_present(dataset: Any, names: list[str]) -> str | None:
    for name in names:
        if name in dataset:
            return name
    return None


def _first_data_var(dataset: Any) -> str | None:
    data_vars = list(getattr(dataset, "data_vars", []))
    return data_vars[0] if data_vars else None


def _get_text_with_retries(url: str, timeout: int) -> str:
    response = _get_with_retries(url, timeout=timeout)
    return response.text


def _direct_nwp_idx_text(grib_url: str) -> str:
    idx_url = f"{grib_url}.idx"
    cached = _DIRECT_NWP_IDX_CACHE.get(idx_url)
    if cached is not None:
        return cached
    text = _get_text_with_retries(idx_url, timeout=30)
    _DIRECT_NWP_IDX_CACHE[idx_url] = text
    return text


def _get_with_retries(url: str, timeout: int, headers: dict[str, str] | None = None) -> requests.Response:
    last_exc: Exception | None = None
    last_was_transient = False
    attempts = max(1, int(os.getenv("WEATHER_RESEARCH_DIRECT_NWP_RETRIES", "6")))
    backoff = max(0.0, float(os.getenv("WEATHER_RESEARCH_DIRECT_NWP_BACKOFF_SECONDS", "2")))
    request_headers = {"User-Agent": "weather-research/0.1"}
    if headers:
        request_headers.update(headers)
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(url, timeout=timeout, headers=request_headers)
            response.raise_for_status()
            return response
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            last_exc = exc
            last_was_transient = True
        except requests.exceptions.HTTPError as exc:
            last_exc = exc
            status = exc.response.status_code if exc.response is not None else None
            last_was_transient = status in TRANSIENT_DIRECT_NWP_STATUS_CODES
            if not last_was_transient:
                break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            last_was_transient = False
            break
        if attempt < attempts and backoff:
            time.sleep(backoff * attempt)
    message = f"GET failed for {url}: {last_exc}"
    if last_was_transient:
        raise TransientDirectNwpDownloadError(message) from last_exc
    raise RuntimeError(message) from last_exc


def _ensure_ecmwflibs_available() -> None:
    try:
        import ecmwflibs

        root = Path(ecmwflibs.__file__).parent
        os.environ["PATH"] = f"{root};{os.environ.get('PATH', '')}"
        os.environ.setdefault("ECCODES_LIB_DIR", str(root))
        os.add_dll_directory(str(root))
    except Exception:
        return
