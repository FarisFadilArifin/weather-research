from __future__ import annotations

import argparse
import logging

from .calibration.sdk_pipeline import (
    TIMING_MODE_SAME_DAY_11AM,
    add_common_args,
    backfill_direct_nbm,
    sdk_cache_dir_from_args,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill direct NOAA NBM 11 AM remaining-day daily highs")
    add_common_args(parser)
    parser.set_defaults(start_date="2021-01-01")
    parser.add_argument("--timing-mode", default=TIMING_MODE_SAME_DAY_11AM, choices=[TIMING_MODE_SAME_DAY_11AM])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument(
        "--include-weather-features",
        action="store_true",
        help=(
            "Fetch or enrich extra NBM raw fields such as cloud, precip, dewpoint, humidity, wind, "
            "ceiling, and visibility. Existing core rows are revisited until weather_features_included=true."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")
    frame = backfill_direct_nbm(
        cache_dir=sdk_cache_dir_from_args(args),
        stations=args.stations,
        start_date=args.start_date,
        end_date=args.end_date,
        force=args.force,
        max_batches=args.max_batches,
        timing_mode=args.timing_mode,
        include_weather_features=args.include_weather_features,
    )
    logging.info("Direct NOAA NBM cache rows: %s", len(frame))


if __name__ == "__main__":
    main()
