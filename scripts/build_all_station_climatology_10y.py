from __future__ import annotations

import argparse
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.actuals_fetch import fetch_iem_asos_hourly_range  # noqa: E402


CLIMATOLOGY_COLUMNS = [
    "climatology_high_10y_f",
    "climatology_high_10y_std_f",
    "climatology_high_10y_count",
    "climatology_source_start_year",
    "climatology_source_end_year",
    "provider_mean_minus_climatology_10y_f",
    "observed_temp_minus_climatology_10y_f",
    "observed_high_so_far_minus_climatology_10y_f",
    "actual_minus_climatology_10y_f_DIAGNOSTIC_ONLY",
]


@dataclass(frozen=True)
class StationYearTask:
    station_code: str
    timezone: str
    year: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch hourly ASOS history, build leakage-safe rolling 10-year daily max-temp "
            "climatology, and join it into station stacking v8 feature files."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=REPO_ROOT / "data/calibration/station_stacking_v8")
    parser.add_argument("--registry", type=Path, default=REPO_ROOT / "data/processed/station_registry.csv")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs/climatology_all_stations")
    parser.add_argument("--history-start-year", type=int, default=2011)
    parser.add_argument("--history-end-year", type=int, default=2025)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--stations", default="", help="Comma-separated station list. Default: all registry stations.")
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument(
        "--copy-raw-from",
        type=Path,
        default=REPO_ROOT / "outputs/climatology_small_sample/raw",
        help="Optional raw cache directory to seed the new cache from before fetching.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.history_end_year < args.history_start_year:
        raise ValueError("--history-end-year must be >= --history-start-year")

    output_root = args.output_root
    raw_dir = output_root / "raw"
    joined_dir = output_root / "station_stacking_v8_with_climatology"
    output_root.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    joined_dir.mkdir(parents=True, exist_ok=True)

    if args.copy_raw_from.exists():
        copied = seed_raw_cache(args.copy_raw_from, raw_dir)
        print(f"Seeded raw cache with {copied} existing files from {args.copy_raw_from}", flush=True)

    registry = pd.read_csv(args.registry)
    registry["station_code"] = registry["station_code"].astype(str)
    station_codes = choose_stations(registry, args.stations)
    station_meta = registry.loc[registry["station_code"].isin(station_codes), ["station_code", "timezone"]].copy()
    if station_meta.empty:
        raise RuntimeError("No matching stations found in registry")

    feature_files = sorted(args.input_dir.glob("*_features.csv"))
    target_years = infer_target_years(feature_files)
    print(
        f"Stations: {', '.join(station_meta['station_code'])} | "
        f"history={args.history_start_year}-{args.history_end_year} | "
        f"target_years={min(target_years)}-{max(target_years)} | "
        f"workers={args.max_workers}",
        flush=True,
    )

    tasks = [
        StationYearTask(row.station_code, row.timezone, year)
        for row in station_meta.itertuples(index=False)
        for year in range(args.history_start_year, args.history_end_year + 1)
    ]
    daily_history, pull_status = fetch_daily_history(tasks, raw_dir, args.force_refresh, args.max_workers)
    status_path = output_root / "climatology_pull_status.csv"
    pull_status.to_csv(status_path, index=False)
    print(f"Wrote pull status: {status_path}", flush=True)

    if daily_history.empty:
        raise RuntimeError("No daily history was fetched")
    daily_history = daily_history.sort_values(["station_code", "date_local"]).reset_index(drop=True)
    daily_path = output_root / f"station_hourly_daily_high_history_{args.history_start_year}_{args.history_end_year}.csv"
    daily_history.to_csv(daily_path, index=False)
    print(f"Wrote daily history: {daily_path} ({len(daily_history):,} rows)", flush=True)

    normals = build_rolling_normals(daily_history, target_years)
    normals_path = output_root / "station_rolling_10y_daily_high_normals.csv"
    normals.to_csv(normals_path, index=False)
    print(f"Wrote rolling normals: {normals_path} ({len(normals):,} rows)", flush=True)

    coverage = join_feature_files(feature_files, normals, joined_dir)
    coverage_path = output_root / "climatology_feature_join_coverage.csv"
    coverage.to_csv(coverage_path, index=False)
    print(f"Wrote joined feature coverage: {coverage_path}", flush=True)

    history_coverage = summarize_history_coverage(
        daily_history,
        station_meta["station_code"].tolist(),
        args.history_start_year,
        args.history_end_year,
    )
    history_coverage_path = output_root / "climatology_history_coverage.csv"
    history_coverage.to_csv(history_coverage_path, index=False)
    print(f"Wrote history coverage: {history_coverage_path}", flush=True)
    print("Done.", flush=True)


def seed_raw_cache(source_dir: Path, raw_dir: Path) -> int:
    copied = 0
    for src in source_dir.glob("iem_asos_hourly_*.csv"):
        dst = raw_dir / src.name
        if dst.exists():
            continue
        shutil.copy2(src, dst)
        copied += 1
    return copied


def choose_stations(registry: pd.DataFrame, station_arg: str) -> list[str]:
    if station_arg.strip():
        requested = [station.strip().upper() for station in station_arg.split(",") if station.strip()]
        known = set(registry["station_code"])
        missing = sorted(set(requested) - known)
        if missing:
            raise RuntimeError(f"Stations not found in registry: {', '.join(missing)}")
        return requested
    return registry["station_code"].dropna().astype(str).tolist()


def infer_target_years(feature_files: list[Path]) -> list[int]:
    years: set[int] = set()
    for path in feature_files:
        frame = pd.read_csv(path, usecols=["contract_date"])
        dates = pd.to_datetime(frame["contract_date"], errors="coerce")
        years.update(int(year) for year in dates.dt.year.dropna().unique())
    if not years:
        raise RuntimeError("Could not infer target years from feature files")
    return sorted(years)


def fetch_daily_history(
    tasks: list[StationYearTask],
    raw_dir: Path,
    force_refresh: bool,
    max_workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_frames: list[pd.DataFrame] = []
    status_rows: list[dict[str, Any]] = []
    worker_count = max(1, max_workers)
    start = time.monotonic()
    completed = 0
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(fetch_station_year_daily, task, raw_dir, force_refresh): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            completed += 1
            try:
                daily, status = future.result()
            except Exception as exc:  # noqa: BLE001
                daily = pd.DataFrame()
                status = {
                    "station_code": task.station_code,
                    "source_year": task.year,
                    "ok": False,
                    "rows": 0,
                    "daily_rows": 0,
                    "message": repr(exc),
                }
            if not daily.empty:
                daily_frames.append(daily)
            status_rows.append(status)
            elapsed = time.monotonic() - start
            print(
                f"[{completed:>3}/{len(tasks)}] {task.station_code} {task.year}: "
                f"{'ok' if status['ok'] else 'failed'} | daily_rows={status['daily_rows']} | "
                f"elapsed={elapsed:0.1f}s",
                flush=True,
            )
    daily_history = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    pull_status = pd.DataFrame(status_rows).sort_values(["station_code", "source_year"]).reset_index(drop=True)
    return daily_history, pull_status


def fetch_station_year_daily(task: StationYearTask, raw_dir: Path, force_refresh: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    start_date = f"{task.year}-01-01"
    end_date = f"{task.year}-12-31"
    last_error: Exception | None = None
    for attempt in range(1, 6):
        try:
            obs = fetch_iem_asos_hourly_range(
                task.station_code,
                start_date,
                end_date,
                task.timezone,
                raw_dir,
                force_refresh=force_refresh,
            )
            daily = observations_to_daily_highs(obs, task)
            return daily, {
                "station_code": task.station_code,
                "source_year": task.year,
                "ok": True,
                "rows": int(len(obs)),
                "daily_rows": int(len(daily)),
                "message": "",
            }
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < 5:
                time.sleep(retry_sleep_seconds(exc, attempt))
    return pd.DataFrame(), {
        "station_code": task.station_code,
        "source_year": task.year,
        "ok": False,
        "rows": 0,
        "daily_rows": 0,
        "message": repr(last_error),
    }


def retry_sleep_seconds(exc: Exception, attempt: int) -> int:
    if isinstance(exc, requests.HTTPError) and exc.response is not None and exc.response.status_code == 429:
        return min(90, 15 * attempt)
    return min(30, 2**attempt)


def observations_to_daily_highs(obs: pd.DataFrame, task: StationYearTask) -> pd.DataFrame:
    if obs.empty:
        return pd.DataFrame(
            columns=["date_local", "actual_high_f", "obs_count", "station_code", "source_year", "month_day"]
        )
    frame = obs.copy()
    frame["date_local"] = frame["valid_local"].dt.date
    start = date(task.year, 1, 1)
    end = date(task.year, 12, 31)
    frame = frame.loc[(frame["date_local"] >= start) & (frame["date_local"] <= end)]
    if frame.empty:
        return pd.DataFrame(
            columns=["date_local", "actual_high_f", "obs_count", "station_code", "source_year", "month_day"]
        )
    daily = (
        frame.groupby("date_local", as_index=False)
        .agg(actual_high_f=("tmpf", "max"), obs_count=("tmpf", "count"))
        .sort_values("date_local")
    )
    daily["actual_high_f"] = daily["actual_high_f"].round(2)
    daily["date_local"] = pd.to_datetime(daily["date_local"]).dt.strftime("%Y-%m-%d")
    daily["station_code"] = task.station_code
    daily["source_year"] = task.year
    daily["month_day"] = pd.to_datetime(daily["date_local"]).dt.strftime("%m-%d")
    return daily[["date_local", "actual_high_f", "obs_count", "station_code", "source_year", "month_day"]]


def build_rolling_normals(daily_history: pd.DataFrame, target_years: list[int]) -> pd.DataFrame:
    history = daily_history.copy()
    history["actual_high_f"] = pd.to_numeric(history["actual_high_f"], errors="coerce")
    history["source_year"] = pd.to_numeric(history["source_year"], errors="coerce").astype("Int64")
    history = history.dropna(subset=["actual_high_f", "source_year", "month_day", "station_code"])
    records: list[dict[str, Any]] = []
    for (station_code, month_day), group in history.groupby(["station_code", "month_day"], sort=True):
        for target_year in target_years:
            source_start = target_year - 10
            source_end = target_year - 1
            window = group.loc[
                (group["source_year"] >= source_start)
                & (group["source_year"] <= source_end)
                & (group["source_year"] < target_year)
            ]
            count = int(window["actual_high_f"].count())
            records.append(
                {
                    "station_code": station_code,
                    "target_year": target_year,
                    "month_day": month_day,
                    "climatology_high_10y_f": round(float(window["actual_high_f"].mean()), 3) if count else pd.NA,
                    "climatology_high_10y_std_f": round(float(window["actual_high_f"].std()), 3) if count > 1 else pd.NA,
                    "climatology_high_10y_count": count,
                    "climatology_source_start_year": source_start,
                    "climatology_source_end_year": source_end,
                }
            )
    return pd.DataFrame(records).sort_values(["station_code", "target_year", "month_day"]).reset_index(drop=True)


def join_feature_files(feature_files: list[Path], normals: pd.DataFrame, joined_dir: Path) -> pd.DataFrame:
    coverage_rows: list[dict[str, Any]] = []
    normal_keys = normals.copy()
    for path in feature_files:
        station_code = path.name.removesuffix("_features.csv")
        features = pd.read_csv(path)
        for column in CLIMATOLOGY_COLUMNS:
            if column in features.columns:
                features = features.drop(columns=[column])
        features = features.copy()
        contract_dates = pd.to_datetime(features["contract_date"], errors="coerce")
        derived_columns: dict[str, Any] = {"target_year": contract_dates.dt.year}
        if "month_day" not in features.columns:
            derived_columns["month_day"] = contract_dates.dt.strftime("%m-%d")
        if "station_code" not in features.columns:
            derived_columns["station_code"] = station_code
        else:
            features["station_code"] = features["station_code"].fillna(station_code).astype(str)
        features = pd.concat([features, pd.DataFrame(derived_columns, index=features.index)], axis=1)

        joined = features.merge(normal_keys, on=["station_code", "target_year", "month_day"], how="left")
        joined = add_climatology_deltas(joined)
        joined = joined.drop(columns=["target_year"])
        output_path = joined_dir / path.name
        joined.to_csv(output_path, index=False)

        has_climo = joined["climatology_high_10y_f"].notna()
        coverage_rows.append(
            {
                "station_code": station_code,
                "input_file": str(path),
                "output_file": str(output_path),
                "rows": int(len(joined)),
                "rows_with_climatology": int(has_climo.sum()),
                "coverage_pct": round(float(has_climo.mean() * 100), 3) if len(joined) else 0.0,
                "min_contract_date": str(joined["contract_date"].min()) if "contract_date" in joined else "",
                "max_contract_date": str(joined["contract_date"].max()) if "contract_date" in joined else "",
                "min_climatology_count": int(joined.loc[has_climo, "climatology_high_10y_count"].min()) if has_climo.any() else 0,
                "median_climatology_count": float(joined.loc[has_climo, "climatology_high_10y_count"].median())
                if has_climo.any()
                else 0.0,
            }
        )
        print(f"Joined {station_code}: {has_climo.sum():,}/{len(joined):,} rows -> {output_path}", flush=True)
    return pd.DataFrame(coverage_rows).sort_values("station_code").reset_index(drop=True)


def add_climatology_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    climo = pd.to_numeric(result["climatology_high_10y_f"], errors="coerce")
    subtract_sources = {
        "provider_mean_minus_climatology_10y_f": "provider_mean_high_f",
        "observed_temp_minus_climatology_10y_f": "observed_temp_at_as_of_f",
        "observed_high_so_far_minus_climatology_10y_f": "observed_high_temp_through_as_of_f",
        "actual_minus_climatology_10y_f_DIAGNOSTIC_ONLY": "actual_high_f",
    }
    for output_col, source_col in subtract_sources.items():
        if source_col in result.columns:
            result[output_col] = pd.to_numeric(result[source_col], errors="coerce") - climo
        else:
            result[output_col] = pd.NA
    return result


def summarize_history_coverage(
    daily_history: pd.DataFrame,
    station_codes: list[str],
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    start = date(start_year, 1, 1)
    end = date(end_year, 12, 31)
    expected_days = (end - start).days + 1
    rows: list[dict[str, Any]] = []
    for station_code in station_codes:
        station_history = daily_history.loc[daily_history["station_code"] == station_code]
        rows.append(
            {
                "station_code": station_code,
                "expected_days": expected_days,
                "daily_rows": int(len(station_history)),
                "coverage_pct": round(float(len(station_history) / expected_days * 100), 3),
                "min_date_local": str(station_history["date_local"].min()) if not station_history.empty else "",
                "max_date_local": str(station_history["date_local"].max()) if not station_history.empty else "",
                "median_obs_count": float(station_history["obs_count"].median()) if not station_history.empty else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("station_code").reset_index(drop=True)


if __name__ == "__main__":
    main()
