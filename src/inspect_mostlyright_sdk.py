from __future__ import annotations

import argparse
import logging

from .calibration.sdk_pipeline import add_common_args, inspect_sdk, sdk_cache_dir_from_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Mostly Right SDK station and archive metadata")
    add_common_args(parser)
    parser.add_argument("--models", nargs="*", default=["hrrr", "gfs", "nbm"], choices=["hrrr", "gfs", "nbm"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")
    registry, availability = inspect_sdk(
        sdk_cache_dir=sdk_cache_dir_from_args(args),
        start_date=args.start_date,
        end_date=args.end_date,
        stations=args.stations,
        models=args.models,
    )
    logging.info("Wrote SDK station registry: %s rows", len(registry))
    logging.info("Wrote SDK archive availability: %s rows", len(availability))


if __name__ == "__main__":
    main()
