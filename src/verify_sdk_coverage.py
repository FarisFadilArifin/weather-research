from __future__ import annotations

import argparse
import logging

from .calibration.sdk_pipeline import TIMING_MODES, add_common_args, sdk_cache_dir_from_args, verify_sdk_coverage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify SDK-only cache coverage for calibration")
    add_common_args(parser)
    parser.add_argument("--models", nargs="*", default=["hrrr", "gfs", "nbm"], choices=["hrrr", "gfs", "nbm"])
    parser.add_argument("--timing-mode", default="strict_6am", choices=TIMING_MODES)
    parser.add_argument("--max-missing-rows", type=int, default=200000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")
    summary, missing = verify_sdk_coverage(
        sdk_cache_dir=sdk_cache_dir_from_args(args),
        stations=args.stations,
        models=args.models,
        start_date=args.start_date,
        end_date=args.end_date,
        max_missing_rows=args.max_missing_rows,
        timing_mode=args.timing_mode,
    )
    logging.info("Wrote SDK coverage summary: %s rows", len(summary))
    logging.info("Wrote SDK missing-coverage rows: %s", len(missing))


if __name__ == "__main__":
    main()
