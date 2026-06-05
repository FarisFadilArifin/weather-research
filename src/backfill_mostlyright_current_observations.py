from __future__ import annotations

import argparse
import logging

from .calibration.sdk_pipeline import TARGET_STATIONS, TIMING_MODE_SAME_DAY_11AM
from .current_observations import backfill_sdk_current_observations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill SDK-only 11 AM current observation features")
    parser.add_argument("--calibration-dir", default="data/calibration")
    parser.add_argument("--sdk-cache-dir", default=None)
    parser.add_argument("--stations", nargs="*", default=TARGET_STATIONS)
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default="latest-complete")
    parser.add_argument("--timing-mode", default=TIMING_MODE_SAME_DAY_11AM, choices=[TIMING_MODE_SAME_DAY_11AM])
    parser.add_argument("--as-of-hour-local", type=int, default=11)
    parser.add_argument("--chunk-days", type=int, default=31)
    parser.add_argument("--source", choices=["iem", "ghcnh", "awc"], default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retry-unavailable", action="store_true")
    parser.add_argument("--request-retries", type=int, default=3)
    parser.add_argument("--retry-sleep-seconds", type=float, default=60.0)
    parser.add_argument("--sleep-between-chunks", type=float, default=0.0)
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")
    sdk_cache_dir = args.sdk_cache_dir or f"{args.calibration_dir}/sdk_current_observations"
    frame = backfill_sdk_current_observations(
        sdk_cache_dir=sdk_cache_dir,
        stations=args.stations,
        start_date=args.start_date,
        end_date=args.end_date,
        timing_mode=args.timing_mode,
        as_of_hour_local=args.as_of_hour_local,
        chunk_days=args.chunk_days,
        source=args.source,
        force=args.force,
        retry_unavailable=args.retry_unavailable,
        request_retries=args.request_retries,
        retry_sleep_seconds=args.retry_sleep_seconds,
        sleep_between_chunks=args.sleep_between_chunks,
        max_chunks=args.max_chunks,
    )
    logging.info("SDK current observation cache rows: %s", len(frame))


if __name__ == "__main__":
    main()
