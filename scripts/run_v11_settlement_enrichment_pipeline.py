from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = REPO_ROOT / "data/calibration/v11_settlement_enriched_v1"
PROVIDERS = ("gfs", "hrrr", "nbm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run data enrichment separately from V11 model training")
    parser.add_argument("--stations", default="KATL,KDAL")
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default=date.today().isoformat())
    parser.add_argument("--parity-days", type=int, default=30)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--direct-workers-per-process", type=int, default=2)
    parser.add_argument("--skip-forecast", action="store_true")
    parser.add_argument("--skip-observations", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    common = ["--stations", args.stations, "--start-date", args.start_date, "--end-date", args.end_date]
    if not args.skip_forecast:
        _run_parallel_forecast_backfill(args)
    if not args.skip_observations:
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts/backfill_v11_settlement_observations.py"),
            *common,
            "--parity-days",
            str(args.parity_days),
        ]
        if args.force:
            command.append("--force")
        subprocess.run(command, cwd=REPO_ROOT, check=True)
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/prepare_v11_settlement_enriched_features.py"), "--stations", args.stations],
        cwd=REPO_ROOT,
        check=True,
    )
    print("Enrichment is complete. The training notebook can now run without fetching or transforming data.")


def _run_parallel_forecast_backfill(args: argparse.Namespace) -> None:
    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)
    summary_root = CACHE_ROOT / "daily_summary_shards"
    log_root = REPO_ROOT / "logs/v11_settlement_enrichment_v1/workers"
    summary_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    tasks: list[tuple[str, int, date, date]] = []
    # Interleave providers so network and decoder workloads remain balanced.
    for year in range(start.year, end.year + 1):
        task_start = max(start, date(year, 1, 1))
        task_end = min(end, date(year, 12, 31))
        for provider in PROVIDERS:
            tasks.append((provider, year, task_start, task_end))
    workers = min(max(1, int(args.workers)), len(tasks))
    print(f"Starting {len(tasks)} provider-year shards with {workers} concurrent workers", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _run_forecast_worker,
                provider,
                year,
                task_start,
                task_end,
                args,
                summary_root,
                log_root,
            ): (provider, year)
            for provider, year, task_start, task_end in tasks
        }
        completed = 0
        for future in as_completed(futures):
            provider, year = futures[future]
            future.result()
            completed += 1
            print(f"Completed provider-year worker {completed}/{len(tasks)}: {provider} {year}", flush=True)
    _merge_forecast_summaries(summary_root, CACHE_ROOT / "forecast_daily_enriched.csv")


def _run_forecast_worker(
    provider: str,
    year: int,
    start: date,
    end: date,
    args: argparse.Namespace,
    summary_root: Path,
    log_root: Path,
) -> None:
    summary = summary_root / f"{provider}_{year}.csv"
    log = log_root / f"{provider}_{year}.log"
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts/backfill_v11_settlement_enrichment.py"),
        "--stations", args.stations,
        "--providers", provider,
        "--start-date", start.isoformat(),
        "--end-date", end.isoformat(),
        "--summary-file", str(summary),
    ]
    if args.force:
        command.append("--force")
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\nSTART {' '.join(command)}\n")
        handle.flush()
        environment = os.environ.copy()
        environment["WEATHER_RESEARCH_DIRECT_NWP_WORKERS"] = str(
            max(1, int(args.direct_workers_per_process))
        )
        subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
            env=environment,
        )


def _merge_forecast_summaries(summary_root: Path, output: Path) -> None:
    paths = sorted(summary_root.glob("*.csv"))
    frames = [pd.read_csv(path) for path in paths if path.stat().st_size > 0]
    if not frames:
        raise RuntimeError("Parallel forecast workers produced no daily summaries")
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(["station_id", "contract_date", "provider"], keep="last")
    merged = merged.sort_values(["station_id", "contract_date", "provider"])
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)
    print(f"Merged {len(merged)} daily forecast summaries into {output}", flush=True)


if __name__ == "__main__":
    main()
