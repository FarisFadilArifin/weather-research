from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from . import analysis, hrrr_fetch, join_data, metrics, nws_fetch, openmeteo_fetch, plots
from .actuals_fetch import ACTUAL_COLUMNS, write_actual_highs
from .hrrr_fetch import FORECAST_COLUMNS
from .polymarket_fetch import MARKET_COLUMNS, discover_weather_events, load_settings, write_polymarket_markets
from .station_registry import REGISTRY_COLUMNS, STATION_MAP_COLUMNS, build_station_registry, enrich_station, write_station_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Polymarket weather forecast-error research pipeline")
    parser.add_argument(
        "--research-mode",
        choices=["polymarket", "station-calendar"],
        default="polymarket",
        help="Use Polymarket market discovery or direct station x calendar-date backtesting.",
    )
    parser.add_argument("--start-date", help="Earliest target/end date to include, YYYY-MM-DD")
    parser.add_argument("--end-date", help="Latest target/end date to include, YYYY-MM-DD")
    parser.add_argument("--markets-lookback-days", type=int, default=90)
    parser.add_argument("--stations", nargs="*", help="Limit to station codes after station discovery")
    parser.add_argument("--providers", nargs="*", default=None, choices=["hrrr", "openmeteo", "nws", "actuals"])
    parser.add_argument("--horizons", nargs="*", type=int, default=None)
    parser.add_argument("--include-active-markets", action="store_true")
    parser.add_argument("--include-resolved-markets", action="store_true", default=True)
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--settings", default="config/settings.yaml")
    parser.add_argument("--manual-overrides", default="config/manual_station_overrides.yaml")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")
    settings = load_settings(args.settings)
    paths = _resolve_paths(settings, args.output_dir)
    _ensure_dirs(paths)

    providers = args.providers or settings["providers"]["default"]
    horizons = args.horizons or [int(h) for h in settings["providers"]["horizons_hours"]]

    station_filter = {station.upper() for station in args.stations} if args.stations else None
    if args.research_mode == "station-calendar":
        if not station_filter:
            raise SystemExit("--stations is required with --research-mode station-calendar")
        if not args.start_date or not args.end_date:
            raise SystemExit("--start-date and --end-date are required with --research-mode station-calendar")
        markets, station_map, registry = write_station_calendar_outputs(
            stations=sorted(station_filter),
            start_date=args.start_date,
            end_date=args.end_date,
            processed_dir=paths["processed"],
            raw_actuals_dir=paths["raw_actuals"],
            force_refresh=args.force_refresh,
        )
        logging.info(
            "Built station-calendar research rows for %s stations from %s to %s",
            len(station_filter),
            args.start_date,
            args.end_date,
        )
    else:
        events = discover_weather_events(
            settings,
            raw_dir=paths["raw_polymarket"],
            markets_lookback_days=args.markets_lookback_days,
            include_active_markets=args.include_active_markets,
            include_resolved_markets=args.include_resolved_markets,
            start_date=args.start_date,
            end_date=args.end_date,
            force_refresh=args.force_refresh,
        )
        markets = write_polymarket_markets(events, paths["processed"])

        if station_filter:
            markets = markets.loc[markets["parsed_station_code"].astype(str).str.upper().isin(station_filter)].copy()
            markets = markets.loc[
                markets["target_date_local"].notna() & (markets["target_date_local"].astype(str).str.len() > 0)
            ].copy()
            markets = _filter_target_dates(markets, args.start_date, args.end_date)
            markets.to_csv(paths["processed"] / "polymarket_markets.csv", index=False)
            logging.info("Filtered Polymarket markets to stations: %s", ", ".join(sorted(station_filter)))

        station_map, registry = write_station_outputs(
            markets,
            processed_dir=paths["processed"],
            raw_actuals_dir=paths["raw_actuals"],
            overrides_path=args.manual_overrides,
            force_refresh=args.force_refresh,
        )
        if station_filter:
            station_map = station_map.loc[station_map["station_code"].astype(str).str.upper().isin(station_filter)]
            registry = registry.loc[registry["station_code"].astype(str).str.upper().isin(station_filter)]
            station_map.to_csv(paths["processed"] / "polymarket_station_map.csv", index=False)
            registry.to_csv(paths["processed"] / "station_registry.csv", index=False)

    actuals = _empty_actuals()
    if "actuals" in providers:
        cached_actuals = _load_cached_actuals_if_complete(paths["processed"] / "actual_highs.csv", station_map, args.force_refresh)
        if cached_actuals is not None:
            actuals = cached_actuals
            logging.info("Reusing complete cached actual highs: %s", paths["processed"] / "actual_highs.csv")
        else:
            actuals = write_actual_highs(
                station_map,
                registry,
                processed_dir=paths["processed"],
                raw_dir=paths["raw_actuals"],
                force_refresh=args.force_refresh,
            )
    else:
        actuals.to_csv(paths["processed"] / "actual_highs.csv", index=False)

    forecast_frames: list[pd.DataFrame] = []
    if "hrrr" in providers:
        forecast_frames.append(
            hrrr_fetch.write_hrrr_snapshots(
                station_map,
                settings,
                paths["processed"],
                paths["raw_hrrr"],
                horizons,
                force_refresh=args.force_refresh,
            )
        )
    else:
        _empty_forecasts().to_csv(paths["processed"] / "hrrr_forecast_snapshots.csv", index=False)
    if "openmeteo" in providers:
        forecast_frames.append(
            openmeteo_fetch.write_openmeteo_snapshots(
                station_map,
                settings,
                paths["processed"],
                paths["raw_openmeteo"],
                horizons,
                force_refresh=args.force_refresh,
            )
        )
    else:
        _empty_forecasts().to_csv(paths["processed"] / "openmeteo_forecast_snapshots.csv", index=False)
    if "nws" in providers:
        forecast_frames.append(
            nws_fetch.write_nws_snapshots(
                station_map,
                settings,
                paths["processed"],
                paths["raw_nws"],
                horizons,
                force_refresh=args.force_refresh,
            )
        )
    else:
        _empty_forecasts().to_csv(paths["processed"] / "nws_forecast_snapshots.csv", index=False)

    forecasts = join_data.combine_forecasts(forecast_frames)
    model_errors = join_data.write_model_errors(
        markets,
        station_map,
        actuals,
        forecasts,
        paths["processed"],
        include_active_training=args.include_active_markets,
    )
    metrics.write_metric_outputs(model_errors, paths["outputs"])
    trading = analysis.write_trading_table(model_errors, settings, paths["outputs"])
    plots.save_required_plots(model_errors, trading, paths["plots"])
    analysis.write_research_summary(station_map, model_errors, trading, paths["outputs"])
    logging.info("Pipeline complete. Processed data: %s Outputs: %s", paths["processed"], paths["outputs"])


def _resolve_paths(settings: dict, output_dir: str | None) -> dict[str, Path]:
    project = settings["project"]
    data_dir = Path(project.get("data_dir", "data"))
    outputs = Path(output_dir) if output_dir else Path(project.get("output_dir", "data/outputs"))
    return {
        "data": data_dir,
        "processed": Path(project.get("processed_dir", "data/processed")),
        "outputs": outputs,
        "plots": outputs / "plots",
        "raw_polymarket": Path(project.get("raw_dir", "data/raw")) / "polymarket",
        "raw_hrrr": Path(project.get("raw_dir", "data/raw")) / "hrrr",
        "raw_openmeteo": Path(project.get("raw_dir", "data/raw")) / "openmeteo",
        "raw_actuals": Path(project.get("raw_dir", "data/raw")) / "actuals",
        "raw_nws": Path(project.get("raw_dir", "data/raw")) / "nws",
    }


def _ensure_dirs(paths: dict[str, Path]) -> None:
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)


def _empty_actuals() -> pd.DataFrame:
    return pd.DataFrame(columns=ACTUAL_COLUMNS)


def _empty_forecasts() -> pd.DataFrame:
    return pd.DataFrame(columns=FORECAST_COLUMNS)


def _load_cached_actuals_if_complete(path: Path, station_map: pd.DataFrame, force_refresh: bool) -> pd.DataFrame | None:
    if force_refresh or not path.exists() or station_map.empty:
        return None
    try:
        actuals = pd.read_csv(path)
    except Exception:
        return None
    required = station_map.dropna(subset=["station_code", "target_date_local"])[["station_code", "target_date_local"]].copy()
    required["station_code"] = required["station_code"].astype(str)
    required["target_date_local"] = required["target_date_local"].astype(str)
    required = required.drop_duplicates()
    if required.empty:
        return actuals
    if not {"station_code", "date_local"}.issubset(actuals.columns):
        return None
    present = actuals[["station_code", "date_local"]].dropna().copy()
    present["station_code"] = present["station_code"].astype(str)
    present["date_local"] = present["date_local"].astype(str)
    covered = required.merge(
        present.drop_duplicates(),
        left_on=["station_code", "target_date_local"],
        right_on=["station_code", "date_local"],
        how="left",
        indicator=True,
    )
    if bool((covered["_merge"] == "left_only").any()):
        return None
    for column in ACTUAL_COLUMNS:
        if column not in actuals:
            actuals[column] = pd.NA
    return actuals[ACTUAL_COLUMNS]


def write_station_calendar_outputs(
    stations: list[str],
    start_date: str,
    end_date: str,
    processed_dir: str | Path,
    raw_actuals_dir: str | Path,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    processed = Path(processed_dir)
    processed.mkdir(parents=True, exist_ok=True)
    days = _date_range(start_date, end_date)
    market_rows: list[dict] = []
    station_rows: list[dict] = []

    for station_code in stations:
        station_code = station_code.upper()
        info = enrich_station(station_code, raw_actuals_dir, force_refresh=force_refresh)
        info.setdefault("station_code", station_code)
        needs_review = not all(info.get(key) for key in ("station_code", "lat", "lon", "timezone", "country"))
        confidence = 1.0 if not needs_review else 0.0
        station_name = info.get("station_name") or station_code
        airport_name = info.get("airport_name") or station_name

        for day in days:
            day_text = day.isoformat()
            slug = f"station-calendar-{station_code.lower()}-{day_text}"
            market_rows.append(
                {
                    "event_id": f"station-calendar-{station_code.lower()}",
                    "market_id": slug,
                    "slug": slug,
                    "title": f"Daily high temperature at {station_code} on {day_text}",
                    "description": "Station-calendar backtest row; no Polymarket market involved.",
                    "rules": "Direct station-calendar research input using the requested airport station code.",
                    "resolution_source": "station-calendar direct station input",
                    "market_start_time_utc": pd.NA,
                    "market_end_time_utc": pd.NA,
                    "target_date_local": day_text,
                    "city_text_from_title": station_code,
                    "parsed_station_name": station_name,
                    "parsed_station_code": station_code,
                    "parsed_airport_name": airport_name,
                    "parsed_country": info.get("country"),
                    "temperature_unit": "F",
                    "outcome_buckets": pd.NA,
                    "final_outcome": pd.NA,
                    "is_resolved": True,
                    "is_active": False,
                    "parse_confidence": confidence,
                    "needs_manual_review": needs_review,
                }
            )
            station_rows.append(
                {
                    "polymarket_city": station_code,
                    "market_slug": slug,
                    "target_date_local": day_text,
                    "station_code": station_code,
                    "station_name": station_name,
                    "airport_name": airport_name,
                    "lat": info.get("lat"),
                    "lon": info.get("lon"),
                    "timezone": info.get("timezone"),
                    "country": info.get("country"),
                    "resolution_source_text": "station-calendar direct station input; no Polymarket market involved",
                    "first_seen_market_date": pd.NA,
                    "last_seen_market_date": pd.NA,
                    "mapping_confidence": confidence,
                    "needs_manual_review": needs_review,
                }
            )

    markets = _frame_with_columns(market_rows, MARKET_COLUMNS)
    station_map = _frame_with_columns(station_rows, STATION_MAP_COLUMNS)
    registry = build_station_registry(station_map)
    if not registry.empty:
        registry["is_active_polymarket_station"] = False

    markets.to_csv(processed / "polymarket_markets.csv", index=False)
    station_map.to_csv(processed / "polymarket_station_map.csv", index=False)
    registry.to_csv(processed / "station_registry.csv", index=False)
    station_map.to_csv(processed / "station_calendar_station_map.csv", index=False)
    return markets, station_map, registry


def _date_range(start_date: str, end_date: str) -> list[date]:
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    if end < start:
        raise SystemExit("--end-date must be on or after --start-date")
    days: list[date] = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)
    return days


def _frame_with_columns(rows: list[dict], columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame:
            frame[column] = pd.NA
    return frame[columns]


def _filter_target_dates(markets: pd.DataFrame, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    out = markets.copy()
    dates = pd.to_datetime(out["target_date_local"], errors="coerce")
    mask = dates.notna()
    if start_date:
        mask &= dates >= pd.Timestamp(start_date)
    if end_date:
        mask &= dates <= pd.Timestamp(end_date)
    return out.loc[mask].copy()


if __name__ == "__main__":
    main()
