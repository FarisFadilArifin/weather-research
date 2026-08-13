from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.calibration.sdk_pipeline import (  # noqa: E402
    TIMING_MODE_SAME_DAY_11AM_LIVE_SAFE,
    HRRR_LONG_CYCLES,
    LIVE_SAFE_MODEL_LAG_MINUTES,
    MODEL_MAX_FXX,
    MODEL_SEARCH_HOURS,
    _group_nwp_requests,
    date_range,
    forecast_as_of_for_timing,
    forecast_hours_for_utc_window,
    local_day_utc_bounds,
    model_cycle_hours,
    plan_direct_nbm_requests,
    plan_nwp_requests,
    station_registry_frame,
)
from src.calibration.v11_settlement_enrichment import (  # noqa: E402
    FEATURE_VERSION,
    FORECAST_RAW_FIELDS,
    PROVIDERS,
    STATIONS,
    enrichment_cache_root,
    hourly_partition_path,
    normalize_hourly_forecast,
    summarize_hourly_forecast,
    write_contract_manifest,
)
from src.direct_nwp_fetch import DIRECT_NWP_FEATURES, extract_direct_nwp_run_feature_points  # noqa: E402
from src.nws_fetch import NBM_FEATURE_FIELDS, _extract_nbm_run_feature_points  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resumable hourly backfill for V11 Settlement enrichment")
    parser.add_argument("--stations", default=",".join(STATIONS))
    parser.add_argument("--providers", default=",".join(PROVIDERS))
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default=date.today().isoformat())
    parser.add_argument("--cache-root", type=Path, default=enrichment_cache_root(REPO_ROOT))
    parser.add_argument("--summary-file", type=Path, help="Optional isolated daily-summary CSV for parallel workers")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(message)s")
    stations = tuple(value.strip().upper() for value in args.stations.split(",") if value.strip())
    providers = tuple(value.strip().lower() for value in args.providers.split(",") if value.strip())
    invalid = set(providers) - set(PROVIDERS)
    if invalid:
        raise ValueError(f"Unsupported providers: {sorted(invalid)}")
    args.cache_root.mkdir(parents=True, exist_ok=True)
    write_contract_manifest(args.cache_root / "feature_contract.json")
    summary_rows: list[dict[str, object]] = []
    for provider in providers:
        summary_rows.extend(
            backfill_provider(
                provider=provider,
                stations=stations,
                start_date=args.start_date,
                end_date=args.end_date,
                cache_root=args.cache_root,
                force=args.force,
                max_batches=args.max_batches,
            )
        )
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary_path = args.summary_file or (args.cache_root / "forecast_daily_enriched.csv")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        if summary_path.exists() and not args.force:
            old = pd.read_csv(summary_path)
            summary = pd.concat([old, summary], ignore_index=True)
        summary = summary.drop_duplicates(["station_id", "contract_date", "provider"], keep="last")
        summary.sort_values(["station_id", "contract_date", "provider"]).to_csv(summary_path, index=False)
    logging.info("%s: wrote %s daily summaries", FEATURE_VERSION, len(summary))


def backfill_provider(
    *,
    provider: str,
    stations: tuple[str, ...],
    start_date: str,
    end_date: str,
    cache_root: Path,
    force: bool,
    max_batches: int | None,
) -> list[dict[str, object]]:
    station_meta = station_registry_frame(stations)
    dates = date_range(start_date, end_date)
    if provider == "nbm":
        requests = plan_direct_nbm_requests(station_meta, dates, timing_mode=TIMING_MODE_SAME_DAY_11AM_LIVE_SAFE)
    else:
        requests = _exact_11am_requests(plan_nwp_requests(
            station_meta,
            dates,
            [provider],
            timing_mode=TIMING_MODE_SAME_DAY_11AM_LIVE_SAFE,
        ))
    completed_requests = [
        request
        for request in requests
        if not force and hourly_partition_path(cache_root, provider, request.station_id, request.contract_date).exists()
    ]
    pending = [
        request
        for request in requests
        if force or not hourly_partition_path(cache_root, provider, request.station_id, request.contract_date).exists()
    ]
    lookup = station_meta.set_index("station_id").to_dict(orient="index")
    rows: list[dict[str, object]] = []
    for request in completed_requests:
        hourly = pd.read_csv(hourly_partition_path(cache_root, provider, request.station_id, request.contract_date))
        summary = summarize_hourly_forecast(hourly)
        rows.append(
            {
                "station_id": request.station_id,
                "contract_date": request.contract_date,
                "provider": provider,
                "issued_at": request.cycle.isoformat(),
                "forecast_as_of": request.forecast_as_of.isoformat(),
                "forecast_window_start": request.forecast_window_start.isoformat(),
                "forecast_window_end": request.forecast_window_end.isoformat(),
                "fetch_status": "ok" if summary else "unavailable",
                **summary,
            }
        )
    processed = 0
    for batch in _group_nwp_requests(pending):
        points = {
            request.station_id: {"lat": float(lookup[request.station_id]["lat"]), "lon": float(lookup[request.station_id]["lon"])}
            for request in batch
        }
        fxx = sorted({hour for request in batch for hour in request.fxx_hours})
        try:
            if provider == "nbm":
                values = _extract_nbm_run_feature_points(
                    points,
                    _nbm_settings(),
                    cache_root / "downloads" / "nbm",
                    batch[0].cycle,
                    fxx,
                    force,
                    feature_fields={name: spec for name, spec in NBM_FEATURE_FIELDS.items() if name in FORECAST_RAW_FIELDS},
                )
            else:
                values = extract_direct_nwp_run_feature_points(
                    points,
                    model=provider,
                    raw_dir=cache_root / "downloads",
                    issue_utc=batch[0].cycle,
                    fxx_hours=fxx,
                    force_refresh=force,
                    feature_fields=[field for field in FORECAST_RAW_FIELDS if field in DIRECT_NWP_FEATURES],
                )
        except Exception as exc:  # noqa: BLE001
            logging.warning("%s %s failed: %s", provider.upper(), batch[0].cycle.isoformat(), exc)
            values = {}
        for request in batch:
            hourly = normalize_hourly_forecast(
                values.get(request.station_id, {}),
                provider=provider,
                station_id=request.station_id,
                contract_date=request.contract_date,
                issue_utc=request.cycle,
            )
            target = hourly_partition_path(cache_root, provider, request.station_id, request.contract_date)
            if not hourly.empty:
                target.parent.mkdir(parents=True, exist_ok=True)
                hourly.to_csv(target, index=False)
            summary = summarize_hourly_forecast(hourly)
            rows.append(
                {
                    "station_id": request.station_id,
                    "contract_date": request.contract_date,
                    "provider": provider,
                    "issued_at": request.cycle.isoformat(),
                    "forecast_as_of": request.forecast_as_of.isoformat(),
                    "forecast_window_start": request.forecast_window_start.isoformat(),
                    "forecast_window_end": request.forecast_window_end.isoformat(),
                    "fetch_status": "ok" if summary else "unavailable",
                    **summary,
                }
            )
        processed += 1
        logging.info("%s batch %s complete (%s request rows)", provider.upper(), processed, len(batch))
        if max_batches is not None and processed >= max_batches:
            break
    return rows


def _nbm_settings() -> dict[str, object]:
    return {
        "nws": {
            "nbm_aws_base_url": "https://noaa-nbm-grib2-pds.s3.amazonaws.com",
            "nbm_product": "core",
            "nbm_domain_suffix": "co",
            "nbm_download_retries": 4,
            "nbm_retry_backoff_seconds": 5,
            "nbm_prefetch_workers": 1,
        }
    }


def _exact_11am_requests(requests: list[object]) -> list[object]:
    """Move the existing 11:15 live-safe planner to the required 11:00 cutoff."""
    output = []
    for request in requests:
        model = request.model.lower()
        as_of = forecast_as_of_for_timing(request.contract_date, request.timezone, TIMING_MODE_SAME_DAY_11AM_LIVE_SAFE)
        _, midnight = local_day_utc_bounds(request.contract_date, request.timezone)
        lag = LIVE_SAFE_MODEL_LAG_MINUTES[model]
        cutoff = (as_of - timedelta(minutes=lag)).replace(minute=0, second=0, microsecond=0)
        allowed = set(model_cycle_hours(model))
        chosen = None
        hours: tuple[int, ...] = ()
        for offset in range(MODEL_SEARCH_HOURS[model] + 1):
            cycle = cutoff - timedelta(hours=offset)
            if cycle.hour not in allowed:
                continue
            candidate = tuple(
                hour
                for hour in forecast_hours_for_utc_window(cycle, as_of, midnight)
                if 0 <= hour <= MODEL_MAX_FXX[model]
            )
            if not candidate or (model == "hrrr" and max(candidate) > 18 and cycle.hour not in HRRR_LONG_CYCLES):
                continue
            chosen, hours = cycle, candidate
            break
        if chosen is not None:
            output.append(
                replace(
                    request,
                    cycle=chosen,
                    fxx_hours=hours,
                    forecast_as_of=as_of,
                    forecast_window_start=as_of,
                    forecast_window_end=midnight,
                    cycle_selection_policy=f"latest_{model}_cycle_available_by_1100_local_with_{lag}min_lag_remaining_day",
                )
            )
    return output


if __name__ == "__main__":
    main()
