from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.asia_11am import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    START_DATE,
    audit_pipeline,
    gfs_day_workers,
    resolve_date_bounds,
    resolve_profiles,
    run_gefs_backfill,
    run_gfs_backfill,
    run_historical_pull,
    run_jma_history_backfill,
    run_live,
    run_observation_backfill,
    run_settlement_backfill,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resumable Seoul/Tokyo 11AM GFS + GEFS + JMA MSM pull pipeline"
    )
    parser.add_argument(
        "stage",
        choices=(
            "settlement",
            "observations",
            "gfs",
            "gefs",
            "jma-history",
            "live",
            "pull",
            "audit",
        ),
    )
    parser.add_argument("--cities", default="seoul,tokyo")
    parser.add_argument("--start-date", type=date.fromisoformat, default=START_DATE)
    parser.add_argument("--end-date", type=date.fromisoformat, default=None)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--contract-date", type=date.fromisoformat, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles = resolve_profiles(args.cities.split(","))
    if args.stage == "live":
        result = run_live(
            args.data_root,
            profiles,
            contract_date=args.contract_date,
            workers=max(1, args.workers),
            now=datetime.now(UTC),
        )
    else:
        start_date, end_date = resolve_date_bounds(
            args.start_date,
            args.end_date,
            profiles=profiles,
        )
        args.data_root.mkdir(parents=True, exist_ok=True)
        if args.stage == "settlement":
            result = run_settlement_backfill(
                args.data_root,
                profiles,
                start_date,
                end_date,
                api_key=args.api_key,
                force=args.force,
            )
        elif args.stage == "observations":
            result = run_observation_backfill(
                args.data_root,
                profiles,
                start_date,
                end_date,
                workers=max(1, args.workers),
                force=args.force,
            )
        elif args.stage == "gfs":
            result = run_gfs_backfill(
                args.data_root,
                profiles,
                start_date,
                end_date,
                workers=gfs_day_workers(args.workers),
                force=args.force,
            )
        elif args.stage == "gefs":
            result = run_gefs_backfill(
                args.data_root,
                profiles,
                start_date,
                end_date,
                workers=max(1, args.workers),
                force=args.force,
            )
        elif args.stage == "jma-history":
            result = run_jma_history_backfill(
                args.data_root,
                profiles,
                start_date,
                end_date,
                workers=max(1, args.workers),
                force=args.force,
            )
        elif args.stage == "pull":
            result = run_historical_pull(
                args.data_root,
                profiles,
                start_date,
                end_date,
                workers=max(1, args.workers),
                api_key=args.api_key,
                force=args.force,
            )
        else:
            result = audit_pipeline(
                args.data_root,
                profiles,
                start_date,
                end_date,
            )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
