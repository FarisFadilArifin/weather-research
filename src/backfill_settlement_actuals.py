from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from .settlement_actuals import (
    backfill_weather_company_pws_history,
    backfill_wunderground_station_history,
    default_polymarket_bounds_output_path,
    default_output_path,
    infer_polymarket_settlement_bounds,
    import_manual_settlement_csv,
    write_missing_settlement_template,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build settlement actual-high labels for v12 training")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output", default=None, help="Defaults to data/processed/settlement_actual_highs.csv")
    parser.add_argument("--import-csv", help="Manual or exported settlement CSV to merge into the canonical table")
    parser.add_argument("--default-source", default="manual_polymarket")
    parser.add_argument("--from-polymarket-raw", action="store_true", help="Infer settlement buckets from cached Polymarket event JSON")
    parser.add_argument("--raw-polymarket-dir", default="data/raw/polymarket")
    parser.add_argument("--bounds-output", default=None, help="Defaults to data/processed/polymarket_implied_settlement_bounds.csv")
    parser.add_argument("--no-merge-exact", action="store_true", help="Do not merge exact inferred buckets into settlement_actual_highs.csv")
    parser.add_argument("--weather-company-api", action="store_true", help="Fetch official PWS daily history via API")
    parser.add_argument(
        "--wunderground-history",
        action="store_true",
        help="Fetch exact airport-station history used by Wunderground settlement pages",
    )
    parser.add_argument("--write-missing-template", action="store_true", help="Write a manual settlement-label template")
    parser.add_argument("--template-output", default=None, help="Output CSV for --write-missing-template")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-key-env", default="WEATHER_COMPANY_API_KEY")
    parser.add_argument("--api-query-param", default="apiKey")
    parser.add_argument("--stations", nargs="*", help="Station IDs to fetch, e.g. KATL KLGA")
    parser.add_argument("--start-date", help="YYYY-MM-DD")
    parser.add_argument("--end-date", help="YYYY-MM-DD")
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")
    output = args.output or default_output_path(args.processed_dir)

    if args.import_csv:
        frame = import_manual_settlement_csv(args.import_csv, output, default_source=args.default_source)
        logging.info("Merged %s settlement rows into %s", len(frame), output)

    if args.from_polymarket_raw:
        bounds_output = args.bounds_output or default_polymarket_bounds_output_path(args.processed_dir)
        bounds, exact = infer_polymarket_settlement_bounds(
            args.raw_polymarket_dir,
            bounds_output,
            exact_output_path=None if args.no_merge_exact else output,
        )
        logging.info(
            "Inferred %s Polymarket settlement buckets (%s exact labels) into %s",
            len(bounds),
            int((bounds["inference_quality"].astype(str) == "exact").sum()) if not bounds.empty else 0,
            bounds_output,
        )
        if not args.no_merge_exact:
            logging.info("Merged exact inferred settlement labels into %s (%s canonical rows)", output, len(exact))

    if args.weather_company_api:
        if not args.stations or not args.start_date or not args.end_date:
            raise SystemExit("--weather-company-api requires --stations, --start-date, and --end-date")
        frame = backfill_weather_company_pws_history(
            output,
            stations=args.stations,
            start_date=args.start_date,
            end_date=args.end_date,
            api_key=args.api_key,
            api_key_env=args.api_key_env,
            api_query_param=args.api_query_param,
            sleep_seconds=args.sleep_seconds,
            force_refresh=args.force_refresh,
        )
        logging.info("Wrote %s settlement rows to %s", len(frame), output)

    if args.wunderground_history:
        if not args.stations or not args.start_date or not args.end_date:
            raise SystemExit("--wunderground-history requires --stations, --start-date, and --end-date")
        registry_path = Path(args.processed_dir) / "station_registry.csv"
        if not registry_path.exists():
            raise SystemExit(f"Missing station registry: {registry_path}")
        registry = pd.read_csv(registry_path)
        station_column = "station_code" if "station_code" in registry else "station_id"
        station_timezones = {
            str(row[station_column]).upper(): str(row["timezone"])
            for _, row in registry.iterrows()
            if pd.notna(row.get(station_column)) and pd.notna(row.get("timezone"))
        }
        station_countries = {
            str(row[station_column]).upper(): str(row["country"]).upper()
            for _, row in registry.iterrows()
            if pd.notna(row.get(station_column)) and pd.notna(row.get("country"))
        }
        station_units = {
            station: ("e" if station_countries.get(station, "US") == "US" else "m")
            for station in station_timezones
        }
        frame = backfill_wunderground_station_history(
            output,
            stations=args.stations,
            station_timezones=station_timezones,
            station_countries=station_countries,
            station_units=station_units,
            start_date=args.start_date,
            end_date=args.end_date,
            api_key=args.api_key,
            api_key_env=args.api_key_env,
            sleep_seconds=args.sleep_seconds,
            force_refresh=args.force_refresh,
        )
        logging.info("Wrote %s exact Wunderground station-history rows to %s", len(frame), output)

    if args.write_missing_template:
        if not args.stations or not args.start_date or not args.end_date:
            raise SystemExit("--write-missing-template requires --stations, --start-date, and --end-date")
        template_output = args.template_output or f"{args.processed_dir}/settlement_actual_highs_missing_template.csv"
        template = write_missing_settlement_template(
            template_output,
            settlement_path=output,
            stations=args.stations,
            start_date=args.start_date,
            end_date=args.end_date,
            default_source=args.default_source,
        )
        logging.info("Wrote %s missing settlement-label rows to %s", len(template), template_output)

    if not any(
        [
            args.import_csv,
            args.from_polymarket_raw,
            args.weather_company_api,
            args.wunderground_history,
            args.write_missing_template,
        ]
    ):
        raise SystemExit(
            "Choose --import-csv, --from-polymarket-raw, --weather-company-api, "
            "--wunderground-history, and/or --write-missing-template"
        )


if __name__ == "__main__":
    main()
