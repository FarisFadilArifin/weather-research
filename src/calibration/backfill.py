from __future__ import annotations

import argparse
import importlib
import logging
import os
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .time_rules import forecast_as_of_utc, forecast_hours_for_local_day, local_day_utc_bounds


MODEL_LAG_MINUTES = {"hrrr": 75, "gfs": 240}
MODEL_MAX_FXX = {"hrrr": 48, "gfs": 120}
MODEL_SEARCH_HOURS = {"hrrr": 18, "gfs": 24}
GFS_CYCLES = {0, 6, 12, 18}
HRRR_LONG_CYCLES = {0, 6, 12, 18}


@dataclass(frozen=True)
class BackfillRequest:
    station_id: str
    station_name: str | None
    airport_name: str | None
    timezone: str
    contract_date: str
    model: str
    forecast_as_of: datetime
    cycle: datetime
    fxx_hours: tuple[int, ...]


def run_backfill(
    project_root: str | Path = ".",
    calibration_dir: str | Path | None = None,
    models: list[str] | None = None,
    stations: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    force: bool = False,
) -> pd.DataFrame:
    root = Path(project_root)
    out_dir = Path(calibration_dir) if calibration_dir is not None else root / "data" / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "mostlyright_nwp_0h_cache.csv"
    models = [m.lower() for m in (models or ["hrrr", "gfs"])]
    station_meta = _load_station_meta(root, stations)
    dates = _resolve_dates(root, start_date, end_date)
    existing = _load_existing_cache(cache_path)
    completed = _completed_keys(existing) if not force else set()
    requests = _plan_requests(station_meta, dates, models, completed)
    if not requests:
        return existing

    forecast_nwp = _load_forecast_nwp(station_meta)
    rows: list[dict[str, Any]] = []
    client = _nwp_http_client()
    try:
        for request in requests:
            try:
                hourly = _fetch_request_hourly(forecast_nwp, request, client=client)
                row = _summarize_request(request, hourly)
            except Exception as exc:  # noqa: BLE001
                logging.warning(
                    "Mostly Right NWP unavailable for %s %s %s: %s",
                    request.model,
                    request.station_id,
                    request.contract_date,
                    exc,
                )
                row = _unavailable_row(request, str(exc))
            rows.append(row)
            existing = _append_cache(cache_path, existing, rows)
            rows = []
    finally:
        if client is not None:
            client.close()
    return existing


def choose_cycle(model: str, contract_date: str, timezone: str, as_of_utc: datetime) -> tuple[datetime | None, tuple[int, ...]]:
    model = model.lower()
    cutoff = as_of_utc.astimezone(UTC) - timedelta(minutes=MODEL_LAG_MINUTES.get(model, 90))
    cutoff = cutoff.replace(minute=0, second=0, microsecond=0)
    max_fxx = MODEL_MAX_FXX.get(model, 120)
    search_hours = MODEL_SEARCH_HOURS.get(model, 24)
    for offset in range(search_hours + 1):
        candidate = cutoff - timedelta(hours=offset)
        if model == "gfs" and candidate.hour not in GFS_CYCLES:
            continue
        fxx_hours = tuple(f for f in forecast_hours_for_local_day(candidate, contract_date, timezone) if 0 <= f <= max_fxx)
        if not fxx_hours:
            continue
        if model == "hrrr" and max(fxx_hours) > 18 and candidate.hour not in HRRR_LONG_CYCLES:
            continue
        return candidate, fxx_hours
    return None, ()


def _load_forecast_nwp(station_meta: pd.DataFrame):
    try:
        _ensure_ecmwflibs_available()
        _patch_mostlyright_nwp_runtime(station_meta)
        from mostlyright.weather.forecast_nwp import forecast_nwp
    except ImportError as exc:
        raise RuntimeError(
            "Mostly Right NWP is not installed. Install with "
            "pip install \"mostlyrightmd-weather[nwp]>=1.0,<2.0\"."
        ) from exc
    return forecast_nwp


def _ensure_ecmwflibs_available() -> None:
    try:
        import ecmwflibs

        root = Path(ecmwflibs.__file__).parent
        os.environ["PATH"] = f"{root};{os.environ.get('PATH', '')}"
        os.environ.setdefault("ECCODES_LIB_DIR", str(root))
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(root))
    except Exception:
        return


def _patch_mostlyright_nwp_runtime(station_meta: pd.DataFrame) -> None:
    module = importlib.import_module("mostlyright.weather.forecast_nwp")
    # Mostly Right v0.1 can see multiple accumulated-precip windows for HRRR/GFS.
    # This calibration only requires temperature for daily highs, so keep the
    # optional precip feature empty instead of failing the whole backfill.
    modules = module.get_variable_map.__globals__.get("_MODULES", {})
    for model in ("hrrr", "gfs"):
        variable_map = getattr(modules.get(model), "VARIABLE_MAP", None)
        if isinstance(variable_map, dict):
            variable_map.pop("precip_mm_1h", None)

    stations = module._resolve_stations.__globals__.get("STATIONS", {})
    if not isinstance(stations, dict) or station_meta.empty:
        return
    try:
        from mostlyright._internal._stations import StationInfo
    except Exception:
        return
    for row in station_meta.itertuples(index=False):
        station_id = str(row.station_id).upper()
        if not station_id.startswith("K") or len(station_id) != 4:
            continue
        try:
            lat = float(row.lat)
            lon = float(row.lon)
        except Exception:
            continue
        code = station_id[1:]
        if any(getattr(info, "icao", None) == station_id for info in stations.values()):
            continue
        stations[code] = StationInfo(
            code=code,
            ghcnh_id="",
            icao=station_id,
            name=str(getattr(row, "station_name", station_id) or station_id),
            tz=str(row.timezone),
            latitude=lat,
            longitude=lon,
            country=str(getattr(row, "country", "US") or "US"),
        )


def _load_station_meta(root: Path, stations: list[str] | None) -> pd.DataFrame:
    path = root / "data" / "processed" / "station_registry.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing station registry: {path}")
    frame = pd.read_csv(path)
    frame = frame.rename(columns={"station_code": "station_id"}).copy()
    frame["station_id"] = frame["station_id"].astype(str).str.upper()
    if stations:
        wanted = {s.upper() for s in stations}
        frame = frame.loc[frame["station_id"].isin(wanted)].copy()
    return frame.dropna(subset=["station_id", "timezone"])


def _resolve_dates(root: Path, start_date: str | None, end_date: str | None) -> list[str]:
    actuals = pd.read_csv(root / "data" / "processed" / "actual_highs.csv")
    dates = pd.to_datetime(actuals["date_local"], errors="coerce").dropna().dt.date
    start = date.fromisoformat(start_date) if start_date else dates.min()
    end = date.fromisoformat(end_date) if end_date else dates.max()
    return [d.isoformat() for d in sorted(set(dates)) if start <= d <= end]


def _load_existing_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _completed_keys(frame: pd.DataFrame) -> set[tuple[str, str, str]]:
    if frame.empty or not {"station_id", "contract_date", "provider"}.issubset(frame.columns):
        return set()
    good = frame.loc[frame.get("raw_forecast_high_f").notna() if "raw_forecast_high_f" in frame else pd.Series(False, index=frame.index)]
    return {
        (str(row.station_id).upper(), str(row.contract_date)[:10], str(row.provider).lower())
        for row in good.itertuples(index=False)
    }


def _plan_requests(
    station_meta: pd.DataFrame,
    dates: list[str],
    models: list[str],
    completed: set[tuple[str, str, str]],
) -> list[BackfillRequest]:
    requests: list[BackfillRequest] = []
    for row in station_meta.itertuples(index=False):
        station_id = str(row.station_id).upper()
        timezone = str(row.timezone)
        for contract_date in dates:
            as_of = forecast_as_of_utc(contract_date, timezone)
            for model in models:
                if (station_id, contract_date, model) in completed:
                    continue
                cycle, fxx_hours = choose_cycle(model, contract_date, timezone, as_of)
                if cycle is None:
                    continue
                requests.append(
                    BackfillRequest(
                        station_id=station_id,
                        station_name=getattr(row, "station_name", None),
                        airport_name=getattr(row, "airport_name", None),
                        timezone=timezone,
                        contract_date=contract_date,
                        model=model,
                        forecast_as_of=as_of,
                        cycle=cycle,
                        fxx_hours=fxx_hours,
                    )
                )
    return requests


def _fetch_request_hourly(forecast_nwp, request: BackfillRequest, client: Any = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    start_utc, end_utc = local_day_utc_bounds(request.contract_date, request.timezone)
    for fxx in request.fxx_hours:
        kwargs = {"cycle": request.cycle, "fxx": fxx}
        if client is not None:
            kwargs["client"] = client
        frame = forecast_nwp(request.station_id, request.model, **kwargs)
        if frame is None or frame.empty:
            continue
        frame = frame.copy()
        frame["valid_at"] = pd.to_datetime(frame["valid_at"], errors="coerce", utc=True)
        frame = frame.loc[(frame["valid_at"] >= start_utc) & (frame["valid_at"] < end_utc)]
        if not frame.empty:
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _summarize_request(request: BackfillRequest, hourly: pd.DataFrame) -> dict[str, Any]:
    if hourly.empty or "temp_k_2m" not in hourly:
        return _unavailable_row(request, "no hourly temperature rows returned")
    temps_f = _k_to_f(pd.to_numeric(hourly["temp_k_2m"], errors="coerce")).dropna()
    if temps_f.empty:
        return _unavailable_row(request, "no valid temp_k_2m values returned")
    dewpoint_f = _k_to_f(pd.to_numeric(hourly.get("dewpoint_k_2m"), errors="coerce")).dropna()
    wind_u = pd.to_numeric(hourly.get("wind_u_ms_10m"), errors="coerce")
    wind_v = pd.to_numeric(hourly.get("wind_v_ms_10m"), errors="coerce")
    wind_speed_mph = np.sqrt(wind_u**2 + wind_v**2) * 2.2369362921
    return {
        "station_id": request.station_id,
        "station_name": request.station_name,
        "airport_name": request.airport_name,
        "provider": request.model,
        "model": request.model,
        "source_label": f"mostlyright_forecast_nwp_{request.model}",
        "contract_date": request.contract_date,
        "forecast_as_of": request.forecast_as_of.isoformat(),
        "issued_at": request.cycle.isoformat(),
        "horizon_hours": 0,
        "raw_forecast_high_f": float(temps_f.max()),
        "forecast_hour_min": min(request.fxx_hours),
        "forecast_hour_max": max(request.fxx_hours),
        "grid_dist_km_mean": _mean(hourly.get("grid_dist_km")),
        "cloud_cover_mean": pd.NA,
        "cloud_cover_max": pd.NA,
        "precip_amount": _sum(hourly.get("precip_mm_1h")),
        "wind_speed_mean": _series_mean(wind_speed_mph),
        "wind_speed_max": _series_max(wind_speed_mph),
        "wind_direction_mean": pd.NA,
        "dewpoint_mean_f": _series_mean(dewpoint_f),
        "humidity_mean": _mean(hourly.get("relative_humidity_pct_2m")),
        "data_source": "mostlyright_weather_forecast_nwp",
        "source_file_or_url": "mostlyright.weather.forecast_nwp",
        "fetch_status": "ok",
        "unavailable_reason": pd.NA,
    }


def _unavailable_row(request: BackfillRequest, reason: str) -> dict[str, Any]:
    return {
        "station_id": request.station_id,
        "station_name": request.station_name,
        "airport_name": request.airport_name,
        "provider": request.model,
        "model": request.model,
        "source_label": f"mostlyright_forecast_nwp_{request.model}",
        "contract_date": request.contract_date,
        "forecast_as_of": request.forecast_as_of.isoformat(),
        "issued_at": request.cycle.isoformat(),
        "horizon_hours": 0,
        "raw_forecast_high_f": pd.NA,
        "data_source": "mostlyright_weather_forecast_nwp",
        "source_file_or_url": "mostlyright.weather.forecast_nwp",
        "fetch_status": "unavailable",
        "unavailable_reason": reason,
    }


def _append_cache(path: Path, existing: pd.DataFrame, rows: list[dict[str, Any]]) -> pd.DataFrame:
    fresh = pd.DataFrame(rows)
    combined = pd.concat([existing, fresh], ignore_index=True) if not existing.empty else fresh
    combined = combined.drop_duplicates(subset=["station_id", "contract_date", "provider"], keep="last")
    combined = combined.sort_values(["contract_date", "provider", "station_id"]).reset_index(drop=True)
    combined.to_csv(path, index=False)
    return combined


def _nwp_http_client() -> Any:
    try:
        import httpx
    except Exception:
        return None
    return httpx.Client(timeout=60)


def _k_to_f(series: pd.Series) -> pd.Series:
    return (series - 273.15) * 9 / 5 + 32


def _mean(series: pd.Series | None) -> float | Any:
    if series is None:
        return pd.NA
    values = pd.to_numeric(series, errors="coerce").dropna()
    return pd.NA if values.empty else float(values.mean())


def _sum(series: pd.Series | None) -> float | Any:
    if series is None:
        return pd.NA
    values = pd.to_numeric(series, errors="coerce").dropna()
    return pd.NA if values.empty else float(values.sum())


def _series_mean(series: pd.Series) -> float | Any:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return pd.NA if values.empty else float(values.mean())


def _series_max(series: pd.Series) -> float | Any:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return pd.NA if values.empty else float(values.max())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill Mostly Right GFS/HRRR 0h daily-high calibration rows")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--calibration-dir", default=None)
    parser.add_argument("--models", nargs="*", default=["hrrr", "gfs"], choices=["hrrr", "gfs"])
    parser.add_argument("--stations", nargs="*")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")
    frame = run_backfill(
        project_root=args.project_root,
        calibration_dir=args.calibration_dir,
        models=args.models,
        stations=args.stations,
        start_date=args.start_date,
        end_date=args.end_date,
        force=args.force,
    )
    logging.info("Mostly Right backfill cache rows: %s", len(frame))


if __name__ == "__main__":
    main()
