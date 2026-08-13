from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .calibration.sdk_pipeline import (
    PRECIP_FEATURE_FLAG,
    SDK_NWP_FILE,
    TIMING_MODE_SAME_DAY_11AM_LIVE_SAFE,
    WEATHER_FEATURE_FLAG,
    _append_cache,
    _completed_nwp_keys,
    _group_nwp_requests,
    _summarize_direct_nbm_feature_values,
    add_common_args,
    date_range,
    plan_nwp_requests,
    resolve_contract_end,
    sdk_cache_dir_from_args,
    station_registry_frame,
    write_station_registry,
)
from .direct_nwp_fetch import extract_direct_nwp_run_feature_points

DEFAULT_DIRECT_FEATURE_FIELDS = [
    "temp_k_2m",
    "dewpoint_k_2m",
    "relative_humidity_pct_2m",
    "wind_u_ms_10m",
    "wind_v_ms_10m",
    "wind_gust_ms",
    "precip_mm_1h",
    "cloud_cover_pct",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill direct NOAA GFS/HRRR 11 AM live-safe weather features")
    add_common_args(parser)
    parser.add_argument("--models", nargs="*", default=["hrrr", "gfs"], choices=["hrrr", "gfs"])
    parser.add_argument("--timing-mode", default=TIMING_MODE_SAME_DAY_11AM_LIVE_SAFE, choices=[TIMING_MODE_SAME_DAY_11AM_LIVE_SAFE])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--raw-dir", default="data/raw/direct_nwp")
    parser.add_argument(
        "--feature-fields",
        nargs="*",
        default=DEFAULT_DIRECT_FEATURE_FIELDS,
        help="Direct GRIB fields to extract. Defaults to rain/heat core fields; add ceiling_m visibility_m for slower second pass.",
    )
    return parser.parse_args()


def backfill_direct_nwp(
    cache_dir: str | Path,
    stations: list[str] | None,
    start_date: str,
    end_date: str | None,
    models: list[str],
    timing_mode: str = TIMING_MODE_SAME_DAY_11AM_LIVE_SAFE,
    force: bool = False,
    max_batches: int | None = None,
    raw_dir: str | Path = "data/raw/direct_nwp",
    feature_fields: list[str] | None = None,
) -> pd.DataFrame:
    out_dir = Path(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    station_meta = write_station_registry(out_dir, stations)
    cache_path = out_dir / SDK_NWP_FILE
    existing = pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame()
    completed = set() if force else _completed_nwp_keys(existing, require_weather_features=True)
    requests = plan_nwp_requests(
        station_meta,
        date_range(start_date, resolve_contract_end(end_date)),
        [model.lower() for model in models],
        completed=completed,
        timing_mode=timing_mode,
    )
    if not requests:
        return existing

    processed_batches = 0
    station_lookup = station_meta.set_index("station_id").to_dict(orient="index")
    for batch in _group_nwp_requests(requests):
        model = batch[0].model.lower()
        stations_for_batch = {
            request.station_id: {
                "lat": float(station_lookup[request.station_id]["lat"]),
                "lon": float(station_lookup[request.station_id]["lon"]),
            }
            for request in batch
        }
        fxx_hours = sorted({fxx for request in batch for fxx in request.fxx_hours})
        try:
            values = extract_direct_nwp_run_feature_points(
                stations_for_batch,
                model=model,
                raw_dir=raw_dir,
                issue_utc=batch[0].cycle,
                fxx_hours=fxx_hours,
                feature_fields=feature_fields or DEFAULT_DIRECT_FEATURE_FIELDS,
            )
        except Exception as exc:  # noqa: BLE001
            logging.warning("Direct %s run unavailable for %s: %s", model.upper(), batch[0].cycle.isoformat(), exc)
            rows = [_unavailable_direct_nwp_row(request, str(exc)) for request in batch]
        else:
            rows = [_direct_nwp_row(request, values.get(request.station_id, {})) for request in batch]

        existing = _append_cache(
            cache_path,
            existing,
            rows,
            keys=["station_id", "contract_date", "provider", "model", "timing_mode"],
        )
        processed_batches += 1
        if max_batches is not None and processed_batches >= max_batches:
            break
    return existing


def _direct_nwp_row(request: Any, values_by_fxx: dict[int, dict[str, float]]) -> dict[str, Any]:
    values_by_fxx = _normalize_direct_precip(values_by_fxx)
    temp_values = [
        fields.get("temp_k_2m")
        for fxx, fields in values_by_fxx.items()
        if fxx in set(request.fxx_hours) and fields.get("temp_k_2m") is not None
    ]
    if not temp_values:
        return _unavailable_direct_nwp_row(request, "no direct temperature values extracted")
    temp_f = [(float(value) - 273.15) * 9 / 5 + 32 for value in temp_values]
    feature_summary = _summarize_direct_nbm_feature_values(request, values_by_fxx)
    return {
        "station_id": request.station_id,
        "station_name": request.station_name,
        "airport_name": request.airport_name,
        "provider": request.model,
        "model": request.model,
        "source_label": f"direct_noaa_{request.model}_grib2_{request.timing_mode}",
        "timing_mode": request.timing_mode,
        "cycle_selection_policy": request.cycle_selection_policy,
        "contract_date": request.contract_date,
        "forecast_as_of": request.forecast_as_of.isoformat(),
        "issued_at": request.cycle.isoformat(),
        "forecast_window_start": request.forecast_window_start.isoformat(),
        "forecast_window_end": request.forecast_window_end.isoformat(),
        "horizon_hours": 0,
        "raw_forecast_high_f": max(temp_f),
        "forecast_hour_min": min(request.fxx_hours),
        "forecast_hour_max": max(request.fxx_hours),
        "forecast_hour_count_requested": len(request.fxx_hours),
        "forecast_hour_count_returned": len(values_by_fxx),
        "forecast_hour_missing": pd.NA,
        "forecast_hour_completeness": len(values_by_fxx) / len(request.fxx_hours) if request.fxx_hours else pd.NA,
        "forecast_hour_fetch_status": "partial" if len(values_by_fxx) < len(request.fxx_hours) else "ok",
        **feature_summary,
        "data_source": f"direct_noaa_{request.model}_grib2",
        "source_file_or_url": _source_url_label(request.model),
        "fetch_status": "ok",
        "unavailable_reason": pd.NA,
        WEATHER_FEATURE_FLAG: True,
        PRECIP_FEATURE_FLAG: True,
    }


def _normalize_direct_precip(values_by_fxx: dict[int, dict[str, float]]) -> dict[int, dict[str, float]]:
    out = {int(fxx): dict(fields) for fxx, fields in values_by_fxx.items()}
    ordered = sorted(
        (fxx, fields.get("precip_mm_1h"))
        for fxx, fields in out.items()
        if fields.get("precip_mm_1h") is not None
    )
    clean = [(fxx, float(value)) for fxx, value in ordered if pd.notna(value)]
    if len(clean) < 2:
        return out
    decreases = sum(1 for (_, prev), (_, cur) in zip(clean, clean[1:], strict=False) if cur + 0.01 < prev)
    if decreases > max(1, len(clean) // 4):
        return out
    previous = 0.0
    for fxx, cumulative in clean:
        increment = max(0.0, cumulative - previous)
        out[fxx]["precip_mm_1h"] = increment
        previous = cumulative
    return out


def _unavailable_direct_nwp_row(request: Any, reason: str) -> dict[str, Any]:
    return {
        "station_id": request.station_id,
        "station_name": request.station_name,
        "airport_name": request.airport_name,
        "provider": request.model,
        "model": request.model,
        "source_label": f"direct_noaa_{request.model}_grib2_{request.timing_mode}",
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
        "data_source": f"direct_noaa_{request.model}_grib2",
        "source_file_or_url": _source_url_label(request.model),
        "fetch_status": "unavailable",
        "unavailable_reason": reason,
        WEATHER_FEATURE_FLAG: False,
        PRECIP_FEATURE_FLAG: False,
    }


def _source_url_label(model: str) -> str:
    if model == "hrrr":
        return "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
    if model == "gfs":
        return "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
    return "direct_noaa_grib2"


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")
    frame = backfill_direct_nwp(
        cache_dir=sdk_cache_dir_from_args(args),
        stations=args.stations,
        start_date=args.start_date,
        end_date=args.end_date,
        models=args.models,
        timing_mode=args.timing_mode,
        force=args.force,
        max_batches=args.max_batches,
        raw_dir=args.raw_dir,
        feature_fields=args.feature_fields,
    )
    logging.info("Direct NOAA GFS/HRRR cache rows: %s", len(frame))


if __name__ == "__main__":
    main()
