from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from .calibration.sdk_pipeline import SDK_NWP_FILE, TIMING_MODE_STRICT_6AM, default_sdk_cache_dir


def merge_sdk_nwp_shards(
    shard_dirs: list[str | Path],
    calibration_dir: str | Path = "data/calibration",
    output_sdk_cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    out_dir = Path(output_sdk_cache_dir) if output_sdk_cache_dir else default_sdk_cache_dir(calibration_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / SDK_NWP_FILE
    frames: list[pd.DataFrame] = []
    if output_path.exists():
        frames.append(pd.read_csv(output_path))
    for shard_dir in shard_dirs:
        path = Path(shard_dir) / SDK_NWP_FILE
        if not path.exists():
            logging.warning("Skipping missing shard cache: %s", path)
            continue
        frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError("No SDK NWP cache files found to merge")
    combined = pd.concat(frames, ignore_index=True)
    if "timing_mode" not in combined.columns:
        combined["timing_mode"] = TIMING_MODE_STRICT_6AM
    combined["timing_mode"] = combined["timing_mode"].fillna(TIMING_MODE_STRICT_6AM)
    keys = ["station_id", "contract_date", "provider", "model", "timing_mode"]
    for key in keys:
        if key not in combined.columns:
            raise ValueError(f"Cannot merge SDK NWP shards; missing required column {key!r}")
    combined = combined.drop_duplicates(subset=keys, keep="last")
    sort_cols = [col for col in ["provider", "model", "timing_mode", "contract_date", "station_id"] if col in combined]
    combined = combined.sort_values(sort_cols).reset_index(drop=True)
    combined.to_csv(output_path, index=False)
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge separate SDK NWP shard caches into the main SDK cache")
    parser.add_argument("--calibration-dir", default="data/calibration")
    parser.add_argument("--output-sdk-cache-dir", default=None)
    parser.add_argument("--shard-dirs", nargs="+", required=True)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")
    frame = merge_sdk_nwp_shards(
        shard_dirs=args.shard_dirs,
        calibration_dir=args.calibration_dir,
        output_sdk_cache_dir=args.output_sdk_cache_dir,
    )
    logging.info("Merged SDK NWP cache rows: %s", len(frame))


if __name__ == "__main__":
    main()
