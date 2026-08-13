from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.hong_kong_11am import (  # noqa: E402
    MODEL_PROVIDERS,
    RESTRICTED_PROVIDERS,
    audit_pipeline,
    build_features,
    resolve_data_root,
    run_free_backfill,
    run_gfs_backfill,
    run_hko_observation_backfill,
    run_live,
    run_restricted_import,
    run_training,
    write_quote_packets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hong Kong 11AM HKO v20 GFS-only no-peak pipeline")
    parser.add_argument(
        "stage",
        choices=(
            "quote",
            "free-backfill",
            "observations-backfill",
            "gfs-backfill",
            "restricted-backfill",
            "features",
            "train",
            "audit",
            "live",
        ),
    )
    parser.add_argument("--project-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--providers", default=",".join(MODEL_PROVIDERS))
    parser.add_argument("--restricted-providers", default=",".join(RESTRICTED_PROVIDERS))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--optuna-trials", type=int, default=None)
    parser.add_argument("--contract-date", type=date.fromisoformat, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = resolve_data_root(args.project_root, args.data_root)
    providers = tuple(item.strip().lower() for item in args.providers.split(",") if item.strip())
    restricted_providers = tuple(
        item.strip().lower() for item in args.restricted_providers.split(",") if item.strip()
    )
    if args.stage == "quote":
        result = {key: str(value) for key, value in write_quote_packets(data_root).items()}
    elif args.stage == "free-backfill":
        result = run_free_backfill(data_root, workers=args.workers, force=args.force)
    elif args.stage == "observations-backfill":
        result = run_hko_observation_backfill(data_root, workers=args.workers, force=args.force)
    elif args.stage == "gfs-backfill":
        result = run_gfs_backfill(data_root, workers=args.workers, force=args.force)
    elif args.stage == "restricted-backfill":
        result = run_restricted_import(data_root, providers=restricted_providers, force=args.force)
    elif args.stage == "features":
        frame = build_features(data_root, providers=providers)
        result = {"status": "complete", "rows": len(frame), "output": str(data_root / "features/HKO_features.parquet")}
    elif args.stage == "train":
        result = run_training(data_root, providers=providers, fast_mode=args.fast, optuna_trials=args.optuna_trials)
    elif args.stage == "audit":
        result = audit_pipeline(data_root, providers=providers)
    else:
        result = run_live(data_root, contract_date=args.contract_date)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
