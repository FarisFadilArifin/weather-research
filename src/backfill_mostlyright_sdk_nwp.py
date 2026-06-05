from __future__ import annotations

import argparse
import logging

from .calibration.sdk_pipeline import TIMING_MODES, add_common_args, backfill_sdk_nwp, sdk_cache_dir_from_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill SDK-only Mostly Right forecast_nwp 0h daily highs")
    add_common_args(parser)
    parser.add_argument("--models", nargs="*", default=["hrrr", "gfs", "nbm"], choices=["hrrr", "gfs", "nbm"])
    parser.add_argument("--timing-mode", default="strict_6am", choices=TIMING_MODES)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--no-batch-stations", action="store_true")
    parser.add_argument(
        "--include-weather-features",
        action="store_true",
        help=(
            "Revisit completed OK rows that were written before weather feature enrichment was tracked. "
            "Unavailable rows remain checkpointed."
        ),
    )
    parser.add_argument(
        "--fxx-workers",
        type=int,
        default=1,
        help="Fetch this many forecast hours concurrently inside each station batch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")
    frame = backfill_sdk_nwp(
        sdk_cache_dir=sdk_cache_dir_from_args(args),
        models=args.models,
        stations=args.stations,
        start_date=args.start_date,
        end_date=args.end_date,
        force=args.force,
        max_requests=args.max_requests,
        max_batches=args.max_batches,
        timing_mode=args.timing_mode,
        batch_stations=not args.no_batch_stations,
        fxx_workers=args.fxx_workers,
        include_weather_features=args.include_weather_features,
    )
    logging.info("SDK NWP cache rows: %s", len(frame))


if __name__ == "__main__":
    main()
