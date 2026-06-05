from __future__ import annotations

import logging
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .openmeteo_fetch import FORECAST_COLUMNS, _base_forecast_row, target_cutoff_utc


HRRR_VARIABLES = {
    "temperature": r":TMP:2 m above ground:",
}


def hrrr_file_url(base_url: str, issue_time: datetime, fxx: int, domain: str = "conus", product: str = "wrfsfc") -> str:
    return (
        f"{base_url.rstrip('/')}/hrrr.{issue_time:%Y%m%d}/{domain}/"
        f"hrrr.t{issue_time:%H}z.{product}f{fxx:02d}.grib2"
    )


def fetch_hrrr_snapshots(
    station_map: pd.DataFrame,
    settings: dict[str, Any],
    raw_dir: str | Path,
    horizons: list[int],
    force_refresh: bool = False,
) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    if station_map.empty:
        return pd.DataFrame(columns=FORECAST_COLUMNS)
    lag = int(settings.get("providers", {}).get("publication_lag_minutes", {}).get("hrrr", 75))
    max_fxx = int(settings.get("hrrr", {}).get("max_forecast_hour", 48))
    search_hours = int(settings.get("hrrr", {}).get("max_cycle_search_hours", 18))
    min_coverage = int(settings.get("hrrr", {}).get("min_target_day_coverage_hours", 18))

    eligible = station_map.dropna(subset=["station_code", "target_date_local"]).copy()
    eligible = eligible.loc[eligible["target_date_local"].astype(str).str.len() > 0]
    requests_to_fill: list[dict[str, Any]] = []
    for _, station in eligible.drop_duplicates(subset=["station_code", "target_date_local"]).iterrows():
        if bool(station.get("needs_manual_review")) or station.get("country") != "US":
            continue
        for horizon in horizons:
            request = _plan_hrrr_snapshot_request(
                station,
                settings,
                int(horizon),
                lag,
                max_fxx,
                search_hours,
                min_coverage,
            )
            if request["issue_utc"] is None:
                rows.append(
                    _base_forecast_row(
                        station,
                        "hrrr",
                        "hrrr",
                        request["cutoff_utc"],
                        request["issue_local"],
                        request["target_date"],
                        int(horizon),
                        {},
                        request["unavailable_reason"],
                    )
                )
            else:
                requests_to_fill.append(request)
    rows.extend(_fill_hrrr_snapshot_requests(requests_to_fill, settings, raw_dir, force_refresh))
    frame = pd.DataFrame(rows)
    for column in FORECAST_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    return frame[FORECAST_COLUMNS]


def _plan_hrrr_snapshot_request(
    station: pd.Series,
    settings: dict[str, Any],
    horizon: int,
    lag_minutes: int,
    max_fxx: int,
    search_hours: int,
    min_coverage: int,
) -> dict[str, Any]:
    target_date = str(station.get("target_date_local"))
    cutoff_local, cutoff_utc = target_cutoff_utc(target_date, station.get("timezone"), horizon, lag_minutes)
    issue_utc, fxx_hours = _choose_hrrr_issue_time(
        cutoff_utc,
        target_date,
        station.get("timezone"),
        max_fxx,
        search_hours,
        min_coverage,
    )
    issue_local = cutoff_local if issue_utc is None else issue_utc.astimezone(cutoff_local.tzinfo)
    return {
        "station": station,
        "station_code": station.get("station_code"),
        "target_date": target_date,
        "horizon": horizon,
        "cutoff_utc": cutoff_utc,
        "issue_utc": issue_utc,
        "issue_local": issue_local,
        "fxx_hours": fxx_hours,
        "unavailable_reason": (
            f"unavailable: no HRRR cycle found before cutoff {cutoff_utc.isoformat()} "
            f"covering at least {min_coverage} target-day hours within f{max_fxx}"
        ),
    }


def _fill_hrrr_snapshot_requests(
    requests_to_fill: list[dict[str, Any]],
    settings: dict[str, Any],
    raw_dir: Path,
    force_refresh: bool,
) -> list[dict[str, Any]]:
    if not requests_to_fill:
        return []
    rows: list[dict[str, Any]] = []
    station_meta: dict[str, dict[str, float]] = {}
    for request in requests_to_fill:
        station = request["station"]
        station_meta[str(request["station_code"])] = {"lat": float(station["lat"]), "lon": float(station["lon"])}

    for issue_utc, group in _group_requests_by_issue(requests_to_fill).items():
        fxx_hours = sorted({fxx for request in group for fxx in request["fxx_hours"]})
        group_station_codes = sorted({str(request["station_code"]) for request in group})
        group_station_meta = {code: station_meta[code] for code in group_station_codes}
        try:
            run_values = _extract_hrrr_run_points(
                group_station_meta,
                settings,
                raw_dir,
                issue_utc,
                fxx_hours,
                force_refresh,
            )
        except Exception as exc:  # noqa: BLE001
            logging.warning("HRRR run unavailable for %s: %s", issue_utc, exc)
            run_values = {}
            run_error = f"unavailable: {exc}"
        else:
            run_error = None

        for request in group:
            station_code = str(request["station_code"])
            values = [
                run_values.get(station_code, {}).get(fxx)
                for fxx in request["fxx_hours"]
                if run_values.get(station_code, {}).get(fxx) is not None
            ]
            summary = _summarize_temperature_values(values)
            source = (
                f"hrrr {issue_utc:%Y-%m-%dT%HZ} f{min(request['fxx_hours']):02d}-f{max(request['fxx_hours']):02d}"
                if values
                else run_error or "unavailable: no HRRR temperatures extracted for station/window"
            )
            rows.append(
                _base_forecast_row(
                    request["station"],
                    "hrrr",
                    "hrrr",
                    issue_utc,
                    request["issue_local"],
                    request["target_date"],
                    request["horizon"],
                    summary,
                    source,
                )
            )
    return rows


def _group_requests_by_issue(requests_to_fill: list[dict[str, Any]]) -> dict[datetime, list[dict[str, Any]]]:
    grouped: dict[datetime, list[dict[str, Any]]] = {}
    for request in requests_to_fill:
        grouped.setdefault(request["issue_utc"], []).append(request)
    return grouped


def _forecast_hours_for_local_day(issue_utc: datetime, target_date: str, timezone: str) -> list[int]:
    from zoneinfo import ZoneInfo

    tz = ZoneInfo(timezone)
    day_start = datetime.fromisoformat(target_date).replace(tzinfo=tz).astimezone(UTC)
    hours = []
    for offset in range(24):
        valid = day_start + timedelta(hours=offset)
        fxx = int((valid - issue_utc).total_seconds() // 3600)
        if fxx >= 0:
            hours.append(fxx)
    return hours


def _choose_hrrr_issue_time(
    cutoff_utc: datetime,
    target_date: str,
    timezone: str,
    max_fxx: int,
    search_hours: int,
    min_coverage_hours: int = 18,
) -> tuple[datetime | None, list[int]]:
    cutoff_utc = cutoff_utc.replace(minute=0, second=0, microsecond=0)
    for offset in range(search_hours + 1):
        candidate = cutoff_utc - timedelta(hours=offset)
        fxx_hours = _forecast_hours_for_local_day(candidate, target_date, timezone)
        fxx_hours = [fxx for fxx in fxx_hours if 0 <= fxx <= max_fxx]
        if len(fxx_hours) < min_coverage_hours:
            continue
        # HRRR archive usually carries f00-f18 hourly for all cycles, with longer
        # f00-f48 runs on 00/06/12/18 UTC. Pick a cycle that can contain the full
        # target-day window before downloading lots of files.
        if max(fxx_hours) <= 18 or candidate.hour in {0, 6, 12, 18}:
            return candidate, fxx_hours
    return None, []


def _extract_hrrr_point_summary(
    station: pd.Series,
    settings: dict[str, Any],
    raw_dir: Path,
    issue_utc: datetime,
    fxx_hours: list[int],
    force_refresh: bool,
) -> dict[str, Any]:
    values = _extract_hrrr_run_points(
        {str(station["station_code"]): {"lat": float(station["lat"]), "lon": float(station["lon"])}},
        settings,
        raw_dir,
        issue_utc,
        fxx_hours,
        force_refresh,
    ).get(str(station["station_code"]), {})
    return _summarize_temperature_values([values.get(fxx) for fxx in fxx_hours if values.get(fxx) is not None])


def _extract_hrrr_run_points(
    stations: dict[str, dict[str, float]],
    settings: dict[str, Any],
    raw_dir: Path,
    issue_utc: datetime,
    fxx_hours: list[int],
    force_refresh: bool,
) -> dict[str, dict[int, float]]:
    _ensure_ecmwflibs_available()
    try:
        import xarray as xr
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("xarray/cfgrib dependencies are required for HRRR extraction") from exc

    values: dict[str, dict[int, float]] = {station_code: {} for station_code in stations}
    base_url = settings.get("hrrr", {}).get("aws_base_url", "https://noaa-hrrr-bdp-pds.s3.amazonaws.com")
    product = settings.get("hrrr", {}).get("product", "wrfsfc")
    domain = settings.get("hrrr", {}).get("domain", "conus")

    for fxx in fxx_hours:
        try:
            grib = _download_hrrr_subset(base_url, issue_utc, fxx, raw_dir, domain, product, force_refresh)
            with xr.open_dataset(grib, engine="cfgrib", backend_kwargs={"indexpath": ""}) as ds:
                if "t2m" not in ds:
                    continue
                for station_code, station in stations.items():
                    point = _nearest_point(ds, station["lat"], station["lon"])
                    values[station_code][fxx] = float((point["t2m"].values - 273.15) * 9 / 5 + 32)
        except Exception as exc:  # noqa: BLE001
            logging.warning("Skipping HRRR %s f%02d: %s", issue_utc, fxx, exc)
            continue
    return values


def _summarize_temperature_values(temps_f: list[float]) -> dict[str, Any]:
    return {
        "forecast_high_f": max(temps_f) if temps_f else None,
        "forecast_low_f": min(temps_f) if temps_f else None,
        "forecast_hourly_max_f": max(temps_f) if temps_f else None,
    }


def _download_hrrr_subset(
    base_url: str,
    issue_time: datetime,
    fxx: int,
    raw_dir: Path,
    domain: str,
    product: str,
    force_refresh: bool,
) -> Path:
    local = raw_dir / f"hrrr_{issue_time:%Y%m%d%H}_f{fxx:02d}.grib2"
    if local.exists() and local.stat().st_size > 0 and not force_refresh:
        return local
    if local.exists() and local.stat().st_size == 0:
        local.unlink()
    url = hrrr_file_url(base_url, issue_time, fxx, domain=domain, product=product)
    idx_response = requests.get(f"{url}.idx", timeout=30, headers={"User-Agent": "weather-research/0.1"})
    idx_response.raise_for_status()
    ranges = _byte_ranges_for_variables(idx_response.text, list(HRRR_VARIABLES.values()))
    with local.open("wb") as handle:
        for start, end in ranges:
            headers = {"Range": f"bytes={start}-{end}", "User-Agent": "weather-research/0.1"}
            response = requests.get(url, headers=headers, timeout=90)
            response.raise_for_status()
            handle.write(response.content)
    return local


def _byte_ranges_for_variables(idx_text: str, variable_patterns: list[str]) -> list[tuple[int, int]]:
    lines = [line for line in idx_text.splitlines() if line.strip()]
    starts: list[tuple[int, str]] = []
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) >= 2 and parts[1].isdigit():
            starts.append((int(parts[1]), line))
    ranges: list[tuple[int, int]] = []
    for i, (start, line) in enumerate(starts):
        if any(pattern.strip(":") in line for pattern in variable_patterns):
            end = starts[i + 1][0] - 1 if i + 1 < len(starts) else start + 2_000_000
            ranges.append((start, end))
    if not ranges:
        raise RuntimeError("Requested HRRR variables were not found in .idx inventory")
    return ranges


def _nearest_point(ds: Any, lat: float, lon: float) -> Any:
    longitude = lon % 360
    if "latitude" in ds and "longitude" in ds:
        dist = (ds.latitude - lat) ** 2 + (ds.longitude - longitude) ** 2
        y, x = divmod(int(dist.argmin()), dist.shape[-1])
        return ds.isel(y=y, x=x)
    return ds


def _ensure_ecmwflibs_available() -> None:
    try:
        import ecmwflibs

        root = Path(ecmwflibs.__file__).parent
        os.environ["PATH"] = f"{root};{os.environ.get('PATH', '')}"
        os.environ.setdefault("ECCODES_LIB_DIR", str(root))
        os.add_dll_directory(str(root))
    except Exception:
        return


def write_hrrr_snapshots(
    station_map: pd.DataFrame,
    settings: dict[str, Any],
    processed_dir: str | Path,
    raw_dir: str | Path,
    horizons: list[int],
    force_refresh: bool = False,
) -> pd.DataFrame:
    processed = Path(processed_dir)
    processed.mkdir(parents=True, exist_ok=True)
    frame = fetch_hrrr_snapshots(station_map, settings, raw_dir, horizons, force_refresh=force_refresh)
    frame.to_csv(processed / "hrrr_forecast_snapshots.csv", index=False)
    return frame
