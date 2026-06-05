from __future__ import annotations

import argparse
import logging

from .calibration.dataset import build_calibration_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build normalized weather calibration samples")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--calibration-dir", default=None)
    parser.add_argument("--source-mode", default="legacy", choices=["legacy", "sdk"])
    parser.add_argument("--sdk-cache-dir", default=None)
    parser.add_argument("--providers", nargs="*")
    parser.add_argument("--timing-modes", nargs="*")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")
    frame = build_calibration_samples(
        project_root=args.project_root,
        calibration_dir=args.calibration_dir,
        include_providers=args.providers,
        include_timing_modes=args.timing_modes,
        source_mode=args.source_mode,
        sdk_cache_dir=args.sdk_cache_dir,
    )
    logging.info("Wrote calibration samples: %s rows", len(frame))


if __name__ == "__main__":
    main()
