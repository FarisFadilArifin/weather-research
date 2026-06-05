from __future__ import annotations

import argparse
import logging

from .calibration.sdk_pipeline import add_common_args, backfill_sdk_actuals, sdk_cache_dir_from_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill SDK-only observed daily highs through Mostly Right obs()")
    add_common_args(parser)
    parser.add_argument("--chunk-days", type=int, default=366)
    parser.add_argument("--strategy", default="exact_window", choices=["auto", "exact_window", "warm_cache"])
    parser.add_argument("--source", default=None, choices=["iem", "ghcnh", "awc"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-chunks", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")
    frame = backfill_sdk_actuals(
        sdk_cache_dir=sdk_cache_dir_from_args(args),
        stations=args.stations,
        start_date=args.start_date,
        end_date=args.end_date,
        chunk_days=args.chunk_days,
        strategy=args.strategy,
        source=args.source,
        force=args.force,
        max_chunks=args.max_chunks,
    )
    logging.info("SDK actual-high cache rows: %s", len(frame))


if __name__ == "__main__":
    main()
