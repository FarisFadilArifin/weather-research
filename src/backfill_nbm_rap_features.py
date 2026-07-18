from __future__ import annotations

import argparse
import logging
import math
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .calibration.sdk_pipeline import (
    LIVE_SAFE_DECISION_DELAY_MINUTES,
    STATION_METADATA,
    TIMING_MODE_SAME_DAY_11AM_LIVE_SAFE,
    _circular_mean_deg,
    _clean_max,
    _clean_mean,
    _clean_min,
    _k_scalar_to_f,
    _ms_scalar_to_mph,
    choose_direct_nbm_13z_live_safe_cycle,
    date_range,
    forecast_as_of_for_timing,
    forecast_hours_for_utc_window,
    forecast_window_for_timing,
    resolve_contract_end,
)
from .direct_nwp_fetch import TransientDirectNwpDownloadError, extract_direct_nwp_run_feature_points
from .nws_fetch import TransientNbmDownloadError, _extract_nbm_run_feature_points


OUTPUT_FILE = "nbm_rap_features.csv"
DEFAULT_STATIONS = ("KATL", "KDAL", "KMIA")
DEFAULT_LOCAL_HOURS = tuple(range(11, 19))
PHYSICS_LAG_MINUTES = 75
PHYSICS_SEARCH_HOURS = 18
RAP_LONG_CYCLES = {3, 9, 15, 21}
RAP_MAX_FXX_DEFAULT = 21
RAP_MAX_FXX_LONG = 51
HRRR_MAX_FXX_DEFAULT = 18

NBM_FEATURE_FIELDS: dict[str, dict[str, Any]] = {
    "temp_k_2m": {"variable": "TMP", "level": "2 m above ground", "names": ["t2m", "t", "unknown"]},
}

RAP_FEATURE_FIELDS = [
    "temp_k_2m",
    "dewpoint_k_2m",
    "wind_u_ms_10m",
    "wind_v_ms_10m",
    "shortwave_radiation_w_m2",
    "boundary_layer_cloud_cover_pct",
    "low_cloud_cover_pct",
    "mid_cloud_cover_pct",
    "high_cloud_cover_pct",
    "pbl_height_m",
    "temp_k_925mb",
    "temp_k_850mb",
    "pwat_kg_m2",
    "cape_j_kg_surface",
    "cin_j_kg_surface",
]

ONSHORE_FROM_DEG = {
    "KMIA": 90.0,
    "KLAX": 250.0,
    "KSEA": 270.0,
    "KHOU": 150.0,
}


@dataclass(frozen=True)
class FeatureRequest:
    station_id: str
    contract_date: str
    timezone: str
    lat: float
    lon: float
    source: str
    cycle: datetime
    fxx_hours: tuple[int, ...]
    forecast_as_of: datetime
    forecast_window_start: datetime
    forecast_window_end: datetime
    cycle_selection_policy: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill live-safe NBM hourly curve and RAP physics features for station-stacking research."
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("data/calibration/nbm_rap_features"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/nbm_rap_features"))
    parser.add_argument("--stations", nargs="*", default=list(DEFAULT_STATIONS))
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default="latest")
    parser.add_argument(
        "--local-hours",
        nargs="*",
        type=int,
        default=list(DEFAULT_LOCAL_HOURS),
        help="Local valid hours to extract from each pre-cutoff forecast cycle.",
    )
    parser.add_argument("--blocks", nargs="*", choices=["nbm", "rap"], default=["nbm", "rap"])
    parser.add_argument(
        "--physics-model",
        choices=["hrrr", "rap"],
        default=os.getenv("WEATHER_RESEARCH_PHYSICS_MODEL", "hrrr").lower(),
        help="Direct NWP source for the physics block. HRRR is the default because this local ecCodes build cannot decode RAP JPEG GRIBs.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-days", type=int)
    parser.add_argument(
        "--discard-raw",
        action="store_true",
        help="Delete temporary GRIB/cfgrib files under --raw-dir after each completed contract date.",
    )
    parser.add_argument(
        "--transient-retry-sleep-seconds",
        type=int,
        default=int(os.getenv("WEATHER_RESEARCH_TRANSIENT_RETRY_SLEEP_SECONDS", "300")),
        help="Seconds to sleep before retrying the same date after a transient network failure.",
    )
    parser.add_argument(
        "--transient-max-retries",
        type=int,
        default=None,
        help="Maximum transient retries per contract date. Omit for unlimited retries.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    frame = backfill_nbm_rap_features(
        cache_dir=args.cache_dir,
        raw_dir=args.raw_dir,
        stations=args.stations,
        start_date=args.start_date,
        end_date=args.end_date,
        local_hours=tuple(sorted(set(args.local_hours))),
        blocks=tuple(args.blocks),
        physics_model=str(args.physics_model).lower(),
        force=args.force,
        max_days=args.max_days,
        discard_raw=bool(args.discard_raw),
        transient_retry_sleep_seconds=int(args.transient_retry_sleep_seconds),
        transient_max_retries=args.transient_max_retries,
    )
    logging.info("NBM/RAP feature cache rows: %s", len(frame))


def backfill_nbm_rap_features(
    *,
    cache_dir: str | Path,
    raw_dir: str | Path,
    stations: Iterable[str] | None,
    start_date: str,
    end_date: str | None,
    local_hours: tuple[int, ...] = DEFAULT_LOCAL_HOURS,
    blocks: tuple[str, ...] = ("nbm", "rap"),
    physics_model: str = "hrrr",
    force: bool = False,
    max_days: int | None = None,
    discard_raw: bool = False,
    transient_retry_sleep_seconds: int = 300,
    transient_max_retries: int | None = None,
) -> pd.DataFrame:
    cache_path = Path(cache_dir) / OUTPUT_FILE
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    existing = _read_existing(cache_path)
    completed = set() if force else _completed_keys(existing, blocks)
    wanted_stations = _normalize_stations(stations)
    dates = date_range(start_date, resolve_contract_end(end_date))
    processed_days = 0

    for contract_date in dates:
        day_stations = [station for station in wanted_stations if (station, contract_date) not in completed]
        if not day_stations:
            continue
        transient_attempt = 0
        while True:
            rows = _base_rows(day_stations, contract_date, local_hours)
            try:
                if "nbm" in blocks:
                    _fill_nbm_rows(rows, raw_dir=Path(raw_dir), local_hours=local_hours)
                if "rap" in blocks:
                    _fill_rap_rows(rows, raw_dir=Path(raw_dir), local_hours=local_hours, physics_model=physics_model)
                break
            except (TransientNbmDownloadError, TransientDirectNwpDownloadError) as exc:
                transient_attempt += 1
                if transient_max_retries is not None and transient_attempt > transient_max_retries:
                    raise
                logging.warning(
                    "Transient network failure for %s attempt %s; retrying in %ss: %s",
                    contract_date,
                    transient_attempt,
                    transient_retry_sleep_seconds,
                    exc,
                )
                if transient_retry_sleep_seconds > 0:
                    time.sleep(transient_retry_sleep_seconds)
        fresh = list(rows.values())
        existing = _append_rows(cache_path, existing, fresh)
        if discard_raw:
            _discard_temporary_raw_files(Path(raw_dir))
        processed_days += 1
        logging.info("Wrote %s rows for %s (%s/%s days processed)", len(fresh), contract_date, processed_days, len(dates))
        if max_days is not None and processed_days >= max_days:
            break
    return existing


def _normalize_stations(stations: Iterable[str] | None) -> list[str]:
    wanted = [str(station).upper() for station in (stations or DEFAULT_STATIONS)]
    missing = [station for station in wanted if station not in STATION_METADATA]
    if missing:
        raise ValueError(f"Missing station metadata for: {', '.join(missing)}")
    return list(dict.fromkeys(wanted))


def _base_rows(stations: list[str], contract_date: str, local_hours: tuple[int, ...]) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for station_id in stations:
        meta = STATION_METADATA[station_id]
        rows[(station_id, contract_date)] = {
            "station_id": station_id,
            "contract_date": contract_date,
            "timezone": meta["timezone"],
            "lat": meta["lat"],
            "lon": meta["lon"],
            "timing_mode": TIMING_MODE_SAME_DAY_11AM_LIVE_SAFE,
            "local_hours": ",".join(str(hour) for hour in local_hours),
            "row_status": "ok",
        }
    return rows


def _fill_nbm_rows(
    rows: dict[tuple[str, str], dict[str, Any]],
    *,
    raw_dir: Path,
    local_hours: tuple[int, ...],
) -> None:
    requests: list[FeatureRequest] = []
    for (station_id, contract_date), row in rows.items():
        meta = STATION_METADATA[station_id]
        cycle, _, as_of, window_start, window_end = choose_direct_nbm_13z_live_safe_cycle(contract_date, meta["timezone"])
        if cycle is None:
            row.update(_unavailable_status("nbm_core", "no 13Z NBM cycle available by live-safe cutoff"))
            continue
        fxx_hours = _fxx_for_local_hours(cycle, contract_date, meta["timezone"], local_hours)
        if not fxx_hours:
            row.update(_unavailable_status("nbm_core", "no requested local hours covered by NBM cycle"))
            continue
        requests.append(
            FeatureRequest(
                station_id=station_id,
                contract_date=contract_date,
                timezone=str(meta["timezone"]),
                lat=float(meta["lat"]),
                lon=float(meta["lon"]),
                source="nbm_core",
                cycle=cycle,
                fxx_hours=fxx_hours,
                forecast_as_of=as_of,
                forecast_window_start=window_start,
                forecast_window_end=window_end,
                cycle_selection_policy="direct_noaa_nbm_13z_cycle_available_by_1115_local_with_120min_buffer",
            )
        )
    for cycle, group in _group_requests_by_cycle(requests).items():
        station_points = {request.station_id: {"lat": request.lat, "lon": request.lon} for request in group}
        fxx_hours = sorted({hour for request in group for hour in request.fxx_hours})
        try:
            values = _extract_nbm_run_feature_points(
                station_points,
                _nbm_settings(),
                raw_dir / "nbm_core",
                cycle,
                fxx_hours,
                False,
                feature_fields=NBM_FEATURE_FIELDS,
            )
        except TransientNbmDownloadError:
            raise
        except Exception as exc:  # noqa: BLE001
            logging.warning("NBM core unavailable for %s: %s", cycle.isoformat(), exc)
            for request in group:
                rows[(request.station_id, request.contract_date)].update(_unavailable_status("nbm_core", str(exc)))
            continue
        for request in group:
            summary = _summarize_nbm_curve(request, values.get(request.station_id, {}), local_hours)
            rows[(request.station_id, request.contract_date)].update(summary)


def _fill_rap_rows(
    rows: dict[tuple[str, str], dict[str, Any]],
    *,
    raw_dir: Path,
    local_hours: tuple[int, ...],
    physics_model: str,
) -> None:
    physics_model = physics_model.lower()
    if physics_model not in {"hrrr", "rap"}:
        raise ValueError(f"physics_model must be 'hrrr' or 'rap'; got {physics_model!r}")
    requests: list[FeatureRequest] = []
    for (station_id, contract_date), row in rows.items():
        meta = STATION_METADATA[station_id]
        cycle, fxx_hours, as_of, window_start, window_end = _choose_rap_live_safe_cycle(
            contract_date,
            str(meta["timezone"]),
            local_hours,
            physics_model,
        )
        if cycle is None or not fxx_hours:
            row.update(_unavailable_status("rap", f"no {physics_model.upper()} cycle available by live-safe cutoff"))
            continue
        requests.append(
            FeatureRequest(
                station_id=station_id,
                contract_date=contract_date,
                timezone=str(meta["timezone"]),
                lat=float(meta["lat"]),
                lon=float(meta["lon"]),
                source=physics_model,
                cycle=cycle,
                fxx_hours=fxx_hours,
                forecast_as_of=as_of,
                forecast_window_start=window_start,
                forecast_window_end=window_end,
                cycle_selection_policy=f"latest_{physics_model}_cycle_available_by_1115_local_with_75min_lag_requested_hours",
            )
        )
    for cycle, group in _group_requests_by_cycle(requests).items():
        station_points = {request.station_id: {"lat": request.lat, "lon": request.lon} for request in group}
        for request in group:
            station_points.update(_pseudo_points_for_station(request))
        fxx_hours = sorted({hour for request in group for hour in request.fxx_hours})
        try:
            values = _extract_physics_run_feature_points(
                station_points=station_points,
                physics_model=physics_model,
                raw_dir=raw_dir / "physics",
                cycle=cycle,
                fxx_hours=fxx_hours,
                requests=group,
            )
        except TransientDirectNwpDownloadError:
            raise
        except Exception as exc:  # noqa: BLE001
            logging.warning("%s physics unavailable for %s: %s", physics_model.upper(), cycle.isoformat(), exc)
            for request in group:
                rows[(request.station_id, request.contract_date)].update(_unavailable_status("rap", str(exc)))
            continue
        for request in group:
            summary = _summarize_rap_physics(request, values, local_hours, physics_model=physics_model)
            rows[(request.station_id, request.contract_date)].update(summary)


def _extract_physics_run_feature_points(
    *,
    station_points: dict[str, dict[str, float]],
    physics_model: str,
    raw_dir: Path,
    cycle: datetime,
    fxx_hours: list[int],
    requests: list[FeatureRequest],
) -> dict[str, dict[int, dict[str, float]]]:
    values: dict[str, dict[int, dict[str, float]]] = {station_code: {} for station_code in station_points}
    fields_by_fxx = _physics_fields_by_fxx(fxx_hours, requests)
    grouped_fxx: dict[tuple[str, ...], list[int]] = {}
    for fxx, fields in fields_by_fxx.items():
        if fields:
            grouped_fxx.setdefault(tuple(sorted(fields)), []).append(fxx)
    for fields, grouped_hours in grouped_fxx.items():
        partial = extract_direct_nwp_run_feature_points(
            station_points,
            model=physics_model,
            raw_dir=raw_dir,
            issue_utc=cycle,
            fxx_hours=grouped_hours,
            force_refresh=False,
            feature_fields=list(fields),
        )
        for station_code, station_values in partial.items():
            for fxx, field_values in station_values.items():
                values.setdefault(station_code, {}).setdefault(int(fxx), {}).update(field_values)
    return values


def _physics_fields_by_fxx(
    fxx_hours: list[int],
    requests: list[FeatureRequest],
) -> dict[int, set[str]]:
    has_kdal = any(request.station_id == "KDAL" for request in requests)
    fields_by_fxx: dict[int, set[str]] = {int(fxx): set() for fxx in fxx_hours}
    for request in requests:
        for fxx in request.fxx_hours:
            if int(fxx) not in fields_by_fxx:
                continue
            valid_local = (request.cycle + timedelta(hours=int(fxx))).astimezone(ZoneInfo(request.timezone))
            if valid_local.date().isoformat() != request.contract_date:
                continue
            fields_by_fxx[int(fxx)].update(_physics_fields_for_local_hour(int(valid_local.hour), has_kdal=has_kdal))
    return fields_by_fxx


def _physics_fields_for_local_hour(hour: int, *, has_kdal: bool) -> set[str]:
    fields = {"temp_k_2m"}
    if 11 <= hour <= 17:
        fields.update({"wind_u_ms_10m", "wind_v_ms_10m"})
    if 12 <= hour <= 17:
        fields.update(
            {
                "shortwave_radiation_w_m2",
                "boundary_layer_cloud_cover_pct",
                "low_cloud_cover_pct",
                "mid_cloud_cover_pct",
                "high_cloud_cover_pct",
                "pbl_height_m",
            }
        )
    if has_kdal and 11 <= hour <= 15:
        fields.add("dewpoint_k_2m")
    if hour == 12:
        fields.add("pwat_kg_m2")
    if hour == 15:
        fields.update({"temp_k_925mb", "cape_j_kg_surface", "cin_j_kg_surface"})
    if hour == 18:
        fields.add("temp_k_850mb")
    return fields


def _summarize_nbm_curve(
    request: FeatureRequest,
    values_by_fxx: dict[int, dict[str, float]],
    local_hours: tuple[int, ...],
) -> dict[str, Any]:
    temps_by_hour: dict[int, float] = {}
    for hour, fxx, fields in _local_hour_values(request, values_by_fxx):
        if hour not in local_hours:
            continue
        temp = _k_scalar_to_f(fields.get("temp_k_2m"))
        if pd.notna(temp):
            temps_by_hour[hour] = float(temp)
    out: dict[str, Any] = {
        "nbm_core_fetch_status": _status_from_count(len(temps_by_hour), len(local_hours)),
        "nbm_core_unavailable_reason": pd.NA if temps_by_hour else "no NBM hourly temperatures extracted",
        "nbm_issued_at": request.cycle.isoformat(),
        "nbm_forecast_as_of": request.forecast_as_of.isoformat(),
        "nbm_forecast_hour_min": min(request.fxx_hours),
        "nbm_forecast_hour_max": max(request.fxx_hours),
        "nbm_hour_count_requested": len(local_hours),
        "nbm_hour_count_returned": len(temps_by_hour),
        "nbm_cycle_selection_policy": request.cycle_selection_policy,
    }
    for hour in local_hours:
        out[f"nbm_t{hour:02d}l_f"] = temps_by_hour.get(hour, pd.NA)
    if temps_by_hour:
        max_hour, max_temp = max(temps_by_hour.items(), key=lambda item: item[1])
        out["nbm_max_post11_f"] = max_temp
        out["nbm_hour_of_max_local"] = max_hour
        out["nbm_slope_11_14_f"] = _delta(temps_by_hour, 11, 14)
        out["nbm_slope_14_17_f"] = _delta(temps_by_hour, 14, 17)
        out["nbm_cooling_onset_hour_local"] = _cooling_onset_hour(temps_by_hour)
    return out


def _summarize_rap_physics(
    request: FeatureRequest,
    values_by_station: dict[str, dict[int, dict[str, float]]],
    local_hours: tuple[int, ...],
    physics_model: str,
) -> dict[str, Any]:
    values_by_fxx = values_by_station.get(request.station_id, {})
    by_hour = {hour: fields for hour, _, fields in _local_hour_values(request, values_by_fxx) if hour in local_hours}
    out: dict[str, Any] = {
        "rap_fetch_status": _status_from_count(len(by_hour), len(local_hours)),
        "rap_unavailable_reason": pd.NA if by_hour else "no RAP hourly fields extracted",
        "rap_source_model": physics_model,
        "physics_source_model": physics_model,
        "physics_fetch_status": _status_from_count(len(by_hour), len(local_hours)),
        "physics_unavailable_reason": pd.NA if by_hour else f"no {physics_model.upper()} hourly fields extracted",
        "rap_issued_at": request.cycle.isoformat(),
        "rap_forecast_as_of": request.forecast_as_of.isoformat(),
        "rap_forecast_hour_min": min(request.fxx_hours),
        "rap_forecast_hour_max": max(request.fxx_hours),
        "rap_hour_count_requested": len(local_hours),
        "rap_hour_count_returned": len(by_hour),
        "rap_cycle_selection_policy": request.cycle_selection_policy,
    }
    if not by_hour:
        return out

    for hour in local_hours:
        fields = by_hour.get(hour, {})
        out[f"rap_t{hour:02d}l_f"] = _k_scalar_to_f(fields.get("temp_k_2m"))
    hours_12_17 = [hour for hour in range(12, 18) if hour in by_hour]
    out["rap_dswrf_12_17_sum"] = _clean_sum([by_hour[h].get("shortwave_radiation_w_m2") for h in hours_12_17])
    out["rap_lcdc_12_17_mean"] = _clean_mean([by_hour[h].get("low_cloud_cover_pct") for h in hours_12_17])
    out["rap_mcdc_12_17_mean"] = _clean_mean([by_hour[h].get("mid_cloud_cover_pct") for h in hours_12_17])
    out["rap_hcdc_12_17_mean"] = _clean_mean([by_hour[h].get("high_cloud_cover_pct") for h in hours_12_17])
    out["rap_boundary_layer_cloud_12_17_mean"] = _clean_mean(
        [by_hour[h].get("boundary_layer_cloud_cover_pct") for h in hours_12_17]
    )
    out["rap_hpbl_max_12_17"] = _clean_max([by_hour[h].get("pbl_height_m") for h in hours_12_17])
    out["rap_hpbl_growth_12_15"] = _field_delta(by_hour, "pbl_height_m", 12, 15)
    out["rap_t925_15l_f"] = _k_scalar_to_f(by_hour.get(15, {}).get("temp_k_925mb"))
    out["rap_t850_18l_f"] = _k_scalar_to_f(by_hour.get(18, {}).get("temp_k_850mb"))
    out["rap_t925_minus_rap_t11_f"] = _numeric_delta(out.get("rap_t925_15l_f"), out.get("rap_t11l_f"))
    out["rap_mixed_down_margin_f"] = out["rap_t925_minus_rap_t11_f"]
    out["rap_pwat_12l"] = by_hour.get(12, {}).get("pwat_kg_m2", pd.NA)
    out["rap_cape_15l"] = by_hour.get(15, {}).get("cape_j_kg_surface", pd.NA)
    out["rap_cin_15l"] = by_hour.get(15, {}).get("cin_j_kg_surface", pd.NA)
    out["rap_cin_abs_15l"] = _abs_or_na(out["rap_cin_15l"])
    out["rap_wind_speed_11l_mph"] = _wind_speed_mph(by_hour.get(11, {}))
    out["rap_wind_direction_11l_deg"] = by_hour.get(11, {}).get("wind_direction_deg_10m", pd.NA)
    out["rap_wind_direction_12_17_mean_deg"] = _circular_mean_deg(
        [by_hour[h].get("wind_direction_deg_10m") for h in hours_12_17]
    )
    out["rap_deep_mixing_flag"] = _deep_mixing_flag(out)
    out.update(_station_specific_features(request, values_by_station, by_hour, out))
    return out


def _station_specific_features(
    request: FeatureRequest,
    values_by_station: dict[str, dict[int, dict[str, float]]],
    by_hour: dict[int, dict[str, float]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    station = request.station_id
    out: dict[str, Any] = {}
    if station in ONSHORE_FROM_DEG:
        target = ONSHORE_FROM_DEG[station]
        out[f"{station.lower()}_rap_onshore_component_11l_mph"] = _wind_component_for_hour(by_hour, 11, target)
        components = [_wind_component_for_hour(by_hour, hour, target) for hour in (13, 14, 15)]
        out[f"{station.lower()}_rap_onshore_component_13_15_mph"] = _clean_mean(components)
    if station == "KMIA":
        onshore = out.get("kmia_rap_onshore_component_13_15_mph")
        pwat = summary.get("rap_pwat_12l")
        cin_abs = summary.get("rap_cin_abs_15l")
        out["kmia_sea_breeze_index"] = _sea_breeze_index(onshore, pwat, cin_abs)
    if station == "KATL":
        ne_component = _clean_mean([_wind_component_for_hour(by_hour, hour, 45.0) for hour in (11, 12, 13, 14, 15)])
        out["katl_ne_wind_component_11_15_mph"] = ne_component
        out["katl_cad_like_flag"] = _cad_like_flag(summary, ne_component)
    if station == "KDAL":
        west = values_by_station.get("KDAL_W", {})
        east = values_by_station.get("KDAL_E", {})
        out["kdal_dewpoint_gradient_west_east_f"] = _dewpoint_gradient(request, west, east)
        out["kdal_dryline_proximity_score"] = _dryline_score(out["kdal_dewpoint_gradient_west_east_f"], summary)
    return out


def _choose_rap_live_safe_cycle(
    contract_date: str,
    timezone: str,
    local_hours: tuple[int, ...],
    physics_model: str = "rap",
) -> tuple[datetime | None, tuple[int, ...], datetime, datetime, datetime]:
    forecast_as_of = forecast_as_of_for_timing(contract_date, timezone, TIMING_MODE_SAME_DAY_11AM_LIVE_SAFE)
    window_start, window_end = forecast_window_for_timing(
        contract_date,
        timezone,
        TIMING_MODE_SAME_DAY_11AM_LIVE_SAFE,
        forecast_as_of,
    )
    decision_time = forecast_as_of + timedelta(minutes=LIVE_SAFE_DECISION_DELAY_MINUTES)
    cutoff = (decision_time - timedelta(minutes=PHYSICS_LAG_MINUTES)).replace(minute=0, second=0, microsecond=0)
    for offset in range(PHYSICS_SEARCH_HOURS + 1):
        cycle = cutoff - timedelta(hours=offset)
        fxx_hours = _fxx_for_local_hours(cycle, contract_date, timezone, local_hours)
        if not fxx_hours:
            continue
        if physics_model == "hrrr":
            max_fxx = HRRR_MAX_FXX_DEFAULT
        else:
            max_fxx = RAP_MAX_FXX_LONG if cycle.hour in RAP_LONG_CYCLES else RAP_MAX_FXX_DEFAULT
        fxx_hours = tuple(fxx for fxx in fxx_hours if 0 <= fxx <= max_fxx)
        if fxx_hours:
            return cycle, fxx_hours, forecast_as_of, window_start, window_end
    return None, (), forecast_as_of, window_start, window_end


def _fxx_for_local_hours(
    cycle: datetime,
    contract_date: str,
    timezone: str,
    local_hours: tuple[int, ...],
) -> tuple[int, ...]:
    tz = ZoneInfo(timezone)
    day = date.fromisoformat(contract_date[:10])
    cycle_utc = cycle.astimezone(UTC)
    fxx: list[int] = []
    for hour in local_hours:
        valid_local = datetime.combine(day, datetime.min.time(), tzinfo=tz).replace(hour=int(hour))
        valid_utc = valid_local.astimezone(UTC)
        delta = valid_utc - cycle_utc
        seconds = delta.total_seconds()
        if seconds >= 0 and seconds % 3600 == 0:
            fxx.append(int(seconds // 3600))
    return tuple(sorted(set(fxx)))


def _local_hour_values(
    request: FeatureRequest,
    values_by_fxx: dict[int, dict[str, float]],
) -> list[tuple[int, int, dict[str, float]]]:
    tz = ZoneInfo(request.timezone)
    rows: list[tuple[int, int, dict[str, float]]] = []
    for fxx, fields in values_by_fxx.items():
        valid_local = (request.cycle + timedelta(hours=int(fxx))).astimezone(tz)
        if valid_local.date().isoformat() != request.contract_date:
            continue
        rows.append((int(valid_local.hour), int(fxx), fields))
    return sorted(rows)


def _nbm_settings() -> dict[str, Any]:
    return {
        "nws": {
            "nbm_temperature_variable": "TMP",
            "nbm_max_forecast_hour": 72,
            "nbm_max_cycle_search_hours": 12,
            "nbm_aws_base_url": "https://noaa-nbm-grib2-pds.s3.amazonaws.com",
            "nbm_product": "core",
            "nbm_domain_suffix": "co",
            "nbm_download_retries": 4,
            "nbm_retry_backoff_seconds": 5,
            "nbm_prefetch_workers": max(1, int(os.getenv("WEATHER_RESEARCH_NBM_PREFETCH_WORKERS", "4"))),
        }
    }


def _pseudo_points_for_station(request: FeatureRequest) -> dict[str, dict[str, float]]:
    if request.station_id != "KDAL":
        return {}
    return {
        "KDAL_W": {"lat": request.lat, "lon": request.lon - 1.5},
        "KDAL_E": {"lat": request.lat, "lon": request.lon + 1.5},
    }


def _group_requests_by_cycle(requests: list[FeatureRequest]) -> dict[datetime, list[FeatureRequest]]:
    groups: dict[datetime, list[FeatureRequest]] = {}
    for request in requests:
        groups.setdefault(request.cycle, []).append(request)
    return groups


def _status_from_count(returned: int, requested: int) -> str:
    if returned <= 0:
        return "unavailable"
    if returned < requested:
        return "partial"
    return "ok"


def _unavailable_status(prefix: str, reason: str) -> dict[str, Any]:
    return {
        f"{prefix}_fetch_status": "unavailable",
        f"{prefix}_unavailable_reason": reason,
        "row_status": "partial",
    }


def _delta(values: dict[int, float], left_hour: int, right_hour: int) -> float | Any:
    left = values.get(left_hour)
    right = values.get(right_hour)
    return _numeric_delta(right, left)


def _numeric_delta(right: Any, left: Any) -> float | Any:
    try:
        if pd.isna(left) or pd.isna(right):
            return pd.NA
        return float(right) - float(left)
    except (TypeError, ValueError):
        return pd.NA


def _field_delta(by_hour: dict[int, dict[str, float]], field: str, left_hour: int, right_hour: int) -> float | Any:
    return _numeric_delta(by_hour.get(right_hour, {}).get(field), by_hour.get(left_hour, {}).get(field))


def _cooling_onset_hour(temps_by_hour: dict[int, float]) -> int | Any:
    hours = sorted(temps_by_hour)
    for left, right in zip(hours, hours[1:], strict=False):
        if temps_by_hour[right] <= temps_by_hour[left] - 0.25:
            return right
    return pd.NA


def _clean_sum(values: Iterable[Any]) -> float | Any:
    clean = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    return float(clean.sum()) if not clean.empty else pd.NA


def _abs_or_na(value: Any) -> float | Any:
    try:
        if pd.isna(value):
            return pd.NA
        return abs(float(value))
    except (TypeError, ValueError):
        return pd.NA


def _wind_speed_mph(fields: dict[str, Any]) -> float | Any:
    speed = fields.get("wind_speed_ms_10m")
    if speed is not None and pd.notna(speed):
        return _ms_scalar_to_mph(speed)
    u = fields.get("wind_u_ms_10m")
    v = fields.get("wind_v_ms_10m")
    try:
        if pd.isna(u) or pd.isna(v):
            return pd.NA
        return float(math.sqrt(float(u) * float(u) + float(v) * float(v)) * 2.2369362921)
    except (TypeError, ValueError):
        return pd.NA


def _wind_component_for_hour(by_hour: dict[int, dict[str, float]], hour: int, target_from_deg: float) -> float | Any:
    fields = by_hour.get(hour, {})
    speed = _wind_speed_mph(fields)
    direction = fields.get("wind_direction_deg_10m")
    try:
        if pd.isna(speed) or pd.isna(direction):
            return pd.NA
        diff = math.radians(_angle_diff(float(direction), float(target_from_deg)))
        return float(speed) * math.cos(diff)
    except (TypeError, ValueError):
        return pd.NA


def _angle_diff(left: float, right: float) -> float:
    return ((left - right + 180.0) % 360.0) - 180.0


def _deep_mixing_flag(summary: dict[str, Any]) -> bool | Any:
    hpbl = summary.get("rap_hpbl_max_12_17")
    dswrf = summary.get("rap_dswrf_12_17_sum")
    low_cloud = summary.get("rap_lcdc_12_17_mean")
    try:
        if pd.isna(hpbl) or pd.isna(dswrf) or pd.isna(low_cloud):
            return pd.NA
        return bool(float(hpbl) >= 1500.0 and float(dswrf) >= 2500.0 and float(low_cloud) <= 40.0)
    except (TypeError, ValueError):
        return pd.NA


def _cad_like_flag(summary: dict[str, Any], ne_component: Any) -> bool | Any:
    low_cloud = summary.get("rap_lcdc_12_17_mean")
    hpbl_growth = summary.get("rap_hpbl_growth_12_15")
    try:
        if pd.isna(low_cloud) or pd.isna(hpbl_growth) or pd.isna(ne_component):
            return pd.NA
        return bool(float(low_cloud) >= 60.0 and float(hpbl_growth) <= 500.0 and float(ne_component) >= 3.0)
    except (TypeError, ValueError):
        return pd.NA


def _sea_breeze_index(onshore: Any, pwat: Any, cin_abs: Any) -> float | Any:
    try:
        if pd.isna(onshore):
            return pd.NA
        score = float(onshore)
        if pd.notna(pwat):
            score += min(float(pwat), 60.0) / 20.0
        if pd.notna(cin_abs):
            score -= min(float(cin_abs), 250.0) / 125.0
        return float(score)
    except (TypeError, ValueError):
        return pd.NA


def _dewpoint_gradient(
    request: FeatureRequest,
    west_values: dict[int, dict[str, float]],
    east_values: dict[int, dict[str, float]],
) -> float | Any:
    west_by_hour = {
        hour: _k_scalar_to_f(fields.get("dewpoint_k_2m"))
        for hour, _, fields in _local_hour_values(request, west_values)
        if 11 <= hour <= 15
    }
    east_by_hour = {
        hour: _k_scalar_to_f(fields.get("dewpoint_k_2m"))
        for hour, _, fields in _local_hour_values(request, east_values)
        if 11 <= hour <= 15
    }
    diffs = []
    for hour in sorted(set(west_by_hour) & set(east_by_hour)):
        diff = _numeric_delta(east_by_hour[hour], west_by_hour[hour])
        if pd.notna(diff):
            diffs.append(diff)
    return _clean_mean(diffs)


def _dryline_score(dewpoint_gradient: Any, summary: dict[str, Any]) -> float | Any:
    try:
        if pd.isna(dewpoint_gradient):
            return pd.NA
        score = max(0.0, float(dewpoint_gradient))
        if summary.get("rap_deep_mixing_flag") is True:
            score += 2.0
        cin_abs = summary.get("rap_cin_abs_15l")
        if pd.notna(cin_abs):
            score += min(float(cin_abs), 250.0) / 125.0
        return float(score)
    except (TypeError, ValueError):
        return pd.NA


def _read_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _completed_keys(frame: pd.DataFrame, blocks: tuple[str, ...]) -> set[tuple[str, str]]:
    if frame.empty or not {"station_id", "contract_date"}.issubset(frame.columns):
        return set()
    out: set[tuple[str, str]] = set()
    required_status_columns: list[str] = []
    if "nbm" in blocks:
        required_status_columns.append("nbm_core_fetch_status")
    if "rap" in blocks:
        required_status_columns.append("rap_fetch_status")
    missing_status_columns = [column for column in required_status_columns if column not in frame.columns]
    if missing_status_columns:
        return set()
    columns = ["station_id", "contract_date", *required_status_columns]
    for row in frame[columns].dropna(subset=["station_id", "contract_date"]).itertuples(index=False):
        if all(str(getattr(row, column)).lower() == "ok" for column in required_status_columns):
            out.add((str(row.station_id).upper(), str(row.contract_date)[:10]))
    return out


def _append_rows(path: Path, existing: pd.DataFrame, rows: list[dict[str, Any]]) -> pd.DataFrame:
    fresh = pd.DataFrame(rows)
    combined = pd.concat([existing, fresh], ignore_index=True) if not existing.empty else fresh
    combined["station_id"] = combined["station_id"].astype(str).str.upper()
    combined["contract_date"] = combined["contract_date"].astype(str).str[:10]
    combined = combined.drop_duplicates(subset=["station_id", "contract_date"], keep="last")
    combined = combined.sort_values(["station_id", "contract_date"]).reset_index(drop=True)
    _write_csv_atomic(combined, path)
    return combined


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        frame.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _discard_temporary_raw_files(raw_dir: Path) -> None:
    raw_dir = raw_dir.resolve()
    if not raw_dir.exists():
        return
    removable_suffixes = {".grib2", ".cfidx", ".idx"}
    for path in raw_dir.rglob("*"):
        if path.is_file() and any(str(path).endswith(suffix) for suffix in removable_suffixes):
            try:
                path.unlink()
            except OSError as exc:
                logging.warning("Could not discard temporary raw file %s: %s", path, exc)


if __name__ == "__main__":
    main()
