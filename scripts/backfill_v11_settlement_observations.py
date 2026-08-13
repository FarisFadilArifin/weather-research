from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.calibration.sdk_pipeline import station_registry_frame  # noqa: E402
from src.calibration.v11_settlement_enrichment import (  # noqa: E402
    OBSERVED_BASE_FIELDS,
    OBSERVED_ENRICHED_FIELDS,
    STATIONS,
    enrichment_cache_root,
    observation_partition_path,
    parity_report,
    summarize_observation_day,
)
from src.current_observations import fetch_sdk_raw_observations_with_retries  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill enriched IEM observations and recent IEM/AWC parity sample")
    parser.add_argument("--stations", default=",".join(STATIONS))
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default=date.today().isoformat())
    parser.add_argument("--parity-days", type=int, default=30)
    parser.add_argument("--cache-root", type=Path, default=enrichment_cache_root(REPO_ROOT))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(message)s")
    stations = tuple(value.strip().upper() for value in args.stations.split(",") if value.strip())
    metadata = station_registry_frame(stations).set_index("station_id")
    summaries: list[dict[str, object]] = []
    for station in stations:
        timezone = str(metadata.loc[station, "timezone"])
        for year in range(pd.Timestamp(args.start_date).year, pd.Timestamp(args.end_date).year + 1):
            start = max(date.fromisoformat(args.start_date), date(year, 1, 1))
            end = min(date.fromisoformat(args.end_date), date(year, 12, 31))
            if start > end:
                continue
            raw = load_or_fetch_year(args.cache_root, "iem", station, start, end, force=args.force)
            for day in pd.date_range(start, end, freq="D"):
                values = summarize_observation_day(raw, contract_date=day.strftime("%Y-%m-%d"), timezone=timezone)
                summaries.append({"station_id": station, "contract_date": day.strftime("%Y-%m-%d"), "observed_source": "iem", **values})
    summary = pd.DataFrame(summaries)
    summary_path = args.cache_root / "observation_daily_enriched.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    write_recent_parity(args.cache_root, metadata, stations, args.parity_days, force=args.force)
    logging.info("Wrote %s observation summaries to %s", len(summary), summary_path)


def load_or_fetch_year(root: Path, source: str, station: str, start: date, end: date, *, force: bool) -> pd.DataFrame:
    target = observation_partition_path(root, source, station, start.year)
    if target.exists() and not force:
        return pd.read_csv(target)
    rows = fetch_sdk_raw_observations_with_retries(station, start.isoformat(), end.isoformat(), source=source)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(target, index=False)
    return frame


def write_recent_parity(root: Path, metadata: pd.DataFrame, stations: tuple[str, ...], parity_days: int, *, force: bool) -> None:
    end = min(date.today(), date(2026, 12, 31))
    start = end - timedelta(days=max(1, parity_days) - 1)
    source_summaries: dict[str, list[dict[str, object]]] = {"iem": [], "awc": []}
    for source in source_summaries:
        for station in stations:
            raw = fetch_sdk_raw_observations_with_retries(station, start.isoformat(), end.isoformat(), source=source)
            raw_frame = pd.DataFrame(raw)
            target = root / "parity" / "raw" / source / f"{station}_{start}_{end}.csv"
            if not raw_frame.empty:
                target.parent.mkdir(parents=True, exist_ok=True)
                raw_frame.to_csv(target, index=False)
            timezone = str(metadata.loc[station, "timezone"])
            for day in pd.date_range(start, end):
                values = summarize_observation_day(raw_frame, contract_date=day.strftime("%Y-%m-%d"), timezone=timezone)
                source_summaries[source].append({"station_id": station, "contract_date": day.strftime("%Y-%m-%d"), **values})
    fields = [*OBSERVED_BASE_FIELDS, *OBSERVED_ENRICHED_FIELDS, "observed_as_of_time_utc"]
    report = parity_report(
        pd.DataFrame(source_summaries["iem"]),
        pd.DataFrame(source_summaries["awc"]),
        fields,
        tolerance={
            "observed_temp_at_as_of_f": 0.2,
            "observed_high_temp_through_as_of_f": 0.2,
            "observed_dewpoint_at_as_of_f": 0.2,
            "observed_humidity_at_as_of": 1.0,
            "observed_pressure_at_as_of": 0.2,
            "observed_visibility_at_as_of": 0.2,
            "observed_cloud_cover_at_as_of": 1.0,
            "observed_wind_speed_at_as_of": 0.2,
            "observed_wind_u_at_as_of_mph": 0.3,
            "observed_wind_v_at_as_of_mph": 0.3,
            "observed_cloud_category": 0.0,
        },
    )
    target = root / "parity" / "iem_awc_parity_report.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(target, index=False)
    units = pd.DataFrame(
        [
            {"feature": field, "normalized_unit": _normalized_unit(field), "iem_decoder": "mostlyright_exact_fetch", "awc_decoder": "mostlyright_exact_fetch"}
            for field in fields
        ]
    )
    units.to_csv(root / "parity" / "iem_awc_unit_contract.csv", index=False)


def _normalized_unit(field: str) -> str:
    if field.endswith("_f") or "_f_" in field:
        return "degF"
    if "pressure" in field:
        return "hPa"
    if "visibility" in field:
        return "statute_mile"
    if "wind" in field and "direction" not in field:
        return "mph"
    if "humidity" in field or "cloud" in field:
        return "percent_or_category"
    if "minutes" in field:
        return "minute"
    if field.endswith("_utc"):
        return "UTC_ISO8601"
    return "unitless_or_category"


if __name__ == "__main__":
    main()
