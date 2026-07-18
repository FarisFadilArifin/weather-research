from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


KEY_COLUMNS = ["station_id", "contract_date", "provider", "model", "timing_mode"]
NWP_CACHE_FILE = "sdk_nwp_0h_cache.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge sharded 9AM SDK NWP caches into the canonical cache.")
    parser.add_argument("--canonical-cache-dir", required=True)
    parser.add_argument("--shards-root", required=True)
    return parser.parse_args()


def read_cache(path: Path, source_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["_merge_source"] = source_name
    return frame


def main() -> None:
    args = parse_args()
    canonical_dir = Path(args.canonical_cache_dir)
    shards_root = Path(args.shards_root)
    canonical_path = canonical_dir / NWP_CACHE_FILE

    frames: list[pd.DataFrame] = []
    if canonical_path.exists():
        frames.append(read_cache(canonical_path, "canonical_existing"))

    for shard_path in sorted(shards_root.glob(f"*/{NWP_CACHE_FILE}")):
        frames.append(read_cache(shard_path, shard_path.parent.name))

    if not frames:
        raise SystemExit("No SDK NWP cache files found to merge.")

    merged = pd.concat(frames, ignore_index=True, sort=False)
    missing_keys = [column for column in KEY_COLUMNS if column not in merged.columns]
    if missing_keys:
        raise SystemExit(f"Cannot merge: missing key columns {missing_keys}")

    merged["_source_rank"] = merged["_merge_source"].ne("canonical_existing").astype(int)
    merged = merged.sort_values(KEY_COLUMNS + ["_source_rank", "_merge_source"])
    merged = merged.drop_duplicates(KEY_COLUMNS, keep="last")
    merged = merged.drop(columns=["_merge_source", "_source_rank"])
    merged = merged.sort_values(["model", "contract_date", "station_id", "provider"]).reset_index(drop=True)

    canonical_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = canonical_path.with_suffix(".csv.tmp")
    merged.to_csv(tmp_path, index=False)
    tmp_path.replace(canonical_path)

    print(f"merged_rows={len(merged)}")
    print(merged.groupby(["provider", "model", "fetch_status"]).size().reset_index(name="rows").to_string(index=False))


if __name__ == "__main__":
    main()
