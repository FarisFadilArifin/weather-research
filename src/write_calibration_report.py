from __future__ import annotations

import argparse
import logging

from .calibration.report import write_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write the weather calibration research report")
    parser.add_argument("--calibration-dir", default="data/calibration")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")
    path = write_report(args.calibration_dir)
    logging.info("Wrote calibration report: %s", path)


if __name__ == "__main__":
    main()
