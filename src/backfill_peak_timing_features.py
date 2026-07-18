from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .backfill_nbm_rap_features import (
    FeatureRequest,
    _append_rows,
    _base_rows,
    _choose_rap_live_safe_cycle,
    _discard_temporary_raw_files,
    _fill_nbm_rows,
    _group_requests_by_cycle,
    _local_hour_values,
    _read_existing,
    _status_from_count,
    _unavailable_status,
)
from .calibration.sdk_pipeline import STATION_METADATA, date_range, resolve_contract_end
from .direct_nwp_fetch import TransientDirectNwpDownloadError, extract_direct_nwp_run_feature_points
from .nws_fetch import TransientNbmDownloadError


OUTPUT_FILE = "peak_timing_features.csv"
SCHEMA_VERSION = 1
DEFAULT_STATIONS = ("KATL", "KDAL")
DEFAULT_LOCAL_HOURS = tuple(range(11, 19))
PRECIP_ONSET_THRESHOLD_MM = 0.1
HRRR_FIELDS = (
    "temp_k_2m",
    "shortwave_radiation_w_m2",
    "precip_mm_1h",
    "cloud_cover_pct",
    "boundary_layer_cloud_cover_pct",
    "low_cloud_cover_pct",
    "mid_cloud_cover_pct",
    "high_cloud_cover_pct",
)
HOURLY_FIELD_COLUMNS = {
    "shortwave_radiation_w_m2": "dswrf",
    "precip_mm_1h": "precip",
    "cloud_cover_pct": "tcc",
    "boundary_layer_cloud_cover_pct": "blcc",
    "low_cloud_cover_pct": "lcc",
    "mid_cloud_cover_pct": "mcc",
    "high_cloud_cover_pct": "hcc",
}
CLOUD_FIELDS = {
    "tcc": "cloud_cover_pct",
    "blcc": "boundary_layer_cloud_cover_pct",
    "lcc": "low_cloud_cover_pct",
    "mcc": "mid_cloud_cover_pct",
    "hcc": "high_cloud_cover_pct",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill point-in-time-safe KATL/KDAL HRRR peak-timing features."
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("data/calibration/peak_timing_features"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/peak_timing_features"))
    parser.add_argument(
        "--nbm-seed-root",
        type=Path,
        default=Path("data/calibration/nbm_rap_features_shards_priority_20260702_full"),
        help="Existing v18 shard root used to seed NBM curve columns before fetching missing dates.",
    )
    parser.add_argument("--stations", nargs="*", default=list(DEFAULT_STATIONS))
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--end-date", default="latest")
    parser.add_argument("--local-hours", nargs="*", type=int, default=list(DEFAULT_LOCAL_HOURS))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-days", type=int)
    parser.add_argument("--discard-raw", action="store_true")
    parser.add_argument("--transient-retry-sleep-seconds", type=int, default=300)
    parser.add_argument("--transient-max-retries", type=int)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    frame = backfill_peak_timing_features(
        cache_dir=args.cache_dir,
        raw_dir=args.raw_dir,
        nbm_seed_root=args.nbm_seed_root,
        stations=args.stations,
        start_date=args.start_date,
        end_date=args.end_date,
        local_hours=tuple(sorted(set(args.local_hours))),
        force=bool(args.force),
        max_days=args.max_days,
        discard_raw=bool(args.discard_raw),
        transient_retry_sleep_seconds=int(args.transient_retry_sleep_seconds),
        transient_max_retries=args.transient_max_retries,
    )
    logging.info("Peak-timing feature cache rows: %s", len(frame))


def backfill_peak_timing_features(
    *,
    cache_dir: str | Path,
    raw_dir: str | Path,
    nbm_seed_root: str | Path | None = None,
    stations: Iterable[str] | None,
    start_date: str,
    end_date: str | None,
    local_hours: tuple[int, ...] = DEFAULT_LOCAL_HOURS,
    force: bool = False,
    max_days: int | None = None,
    discard_raw: bool = False,
    transient_retry_sleep_seconds: int = 300,
    transient_max_retries: int | None = None,
) -> pd.DataFrame:
    wanted_stations = _normalize_stations(stations)
    _validate_local_hours(local_hours)
    cache_path = Path(cache_dir) / OUTPUT_FILE
    raw_path = Path(raw_dir)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.mkdir(parents=True, exist_ok=True)
    existing = _read_existing(cache_path)
    nbm_seed = _load_nbm_seed(nbm_seed_root)
    completed = set() if force else _completed_keys(existing)
    processed_days = 0

    for contract_date in date_range(start_date, resolve_contract_end(end_date)):
        day_stations = [station for station in wanted_stations if (station, contract_date) not in completed]
        if not day_stations:
            continue
        attempt = 0
        while True:
            rows = _base_rows(day_stations, contract_date, local_hours)
            for row in rows.values():
                row["schema_version"] = SCHEMA_VERSION
                row["feature_profile"] = "peak_timing_v1"
                row["precip_onset_threshold_mm"] = PRECIP_ONSET_THRESHOLD_MM
            try:
                _seed_nbm_rows(rows, nbm_seed)
                missing_nbm = {
                    key: row
                    for key, row in rows.items()
                    if str(row.get("nbm_core_fetch_status", "")).lower() != "ok"
                }
                if missing_nbm:
                    _fill_nbm_rows(missing_nbm, raw_dir=raw_path, local_hours=local_hours)
                _fill_hrrr_rows(rows, raw_dir=raw_path, local_hours=local_hours)
                break
            except (TransientNbmDownloadError, TransientDirectNwpDownloadError) as exc:
                attempt += 1
                if transient_max_retries is not None and attempt > transient_max_retries:
                    raise
                logging.warning(
                    "Transient network failure for %s attempt %s; retrying in %ss: %s",
                    contract_date,
                    attempt,
                    transient_retry_sleep_seconds,
                    exc,
                )
                if transient_retry_sleep_seconds > 0:
                    time.sleep(transient_retry_sleep_seconds)
        existing = _append_rows_with_lock_retry(cache_path, existing, list(rows.values()))
        if discard_raw:
            _discard_temporary_raw_files(raw_path)
        processed_days += 1
        logging.info("Wrote %s rows for %s", len(rows), contract_date)
        if max_days is not None and processed_days >= max_days:
            break
    return existing


def _append_rows_with_lock_retry(
    path: Path,
    existing: pd.DataFrame,
    rows: list[dict[str, Any]],
    *,
    max_attempts: int = 12,
    retry_sleep_seconds: float = 0.25,
) -> pd.DataFrame:
    for attempt in range(1, max_attempts + 1):
        try:
            return _append_rows(path, existing, rows)
        except PermissionError:
            if attempt >= max_attempts:
                raise
            logging.warning(
                "CSV replace temporarily locked for %s (attempt %s/%s); retrying",
                path,
                attempt,
                max_attempts,
            )
            time.sleep(retry_sleep_seconds * attempt)
    raise RuntimeError("unreachable")


def _load_nbm_seed(root: str | Path | None) -> pd.DataFrame:
    if root is None:
        return pd.DataFrame()
    path = Path(root)
    if not path.exists():
        logging.info("NBM seed root does not exist; missing NBM rows will be fetched: %s", path)
        return pd.DataFrame()
    files = sorted(path.rglob("nbm_rap_features.csv"))
    frames = [pd.read_csv(file, low_memory=False) for file in files]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    required = {"station_id", "contract_date", "nbm_core_fetch_status"}
    if not required.issubset(combined.columns):
        return pd.DataFrame()
    combined["station_id"] = combined["station_id"].astype(str).str.upper()
    combined["contract_date"] = combined["contract_date"].astype(str).str[:10]
    combined = combined.loc[combined["nbm_core_fetch_status"].astype(str).str.lower().eq("ok")]
    return combined.drop_duplicates(["station_id", "contract_date"], keep="last").set_index(
        ["station_id", "contract_date"]
    )


def _seed_nbm_rows(
    rows: dict[tuple[str, str], dict[str, Any]],
    seed: pd.DataFrame,
) -> None:
    if seed.empty:
        return
    nbm_columns = [column for column in seed.columns if column.startswith("nbm_")]
    for key, row in rows.items():
        if key not in seed.index:
            continue
        source = seed.loc[key]
        if isinstance(source, pd.DataFrame):
            source = source.iloc[-1]
        for column in nbm_columns:
            row[column] = source.get(column, pd.NA)


def _normalize_stations(stations: Iterable[str] | None) -> list[str]:
    values = [str(value).strip().upper() for value in (stations or DEFAULT_STATIONS) if str(value).strip()]
    unsupported = sorted(set(values) - set(DEFAULT_STATIONS))
    if unsupported:
        raise ValueError(f"peak_timing_v1 is limited to {DEFAULT_STATIONS}; got {unsupported}")
    return list(dict.fromkeys(values))


def _validate_local_hours(local_hours: tuple[int, ...]) -> None:
    if tuple(local_hours) != DEFAULT_LOCAL_HOURS:
        raise ValueError(f"peak_timing_v1 requires local hours {DEFAULT_LOCAL_HOURS}; got {local_hours}")


def _fill_hrrr_rows(
    rows: dict[tuple[str, str], dict[str, Any]],
    *,
    raw_dir: Path,
    local_hours: tuple[int, ...],
) -> None:
    requests: list[FeatureRequest] = []
    for (station_id, contract_date), row in rows.items():
        meta = STATION_METADATA[station_id]
        cycle, fxx_hours, as_of, window_start, window_end = _choose_rap_live_safe_cycle(
            contract_date,
            str(meta["timezone"]),
            local_hours,
            "hrrr",
        )
        if cycle is None or not fxx_hours:
            row.update(_unavailable_status("hrrr", "no HRRR cycle available by live-safe cutoff"))
            continue
        requests.append(
            FeatureRequest(
                station_id=station_id,
                contract_date=contract_date,
                timezone=str(meta["timezone"]),
                lat=float(meta["lat"]),
                lon=float(meta["lon"]),
                source="hrrr",
                cycle=cycle,
                fxx_hours=fxx_hours,
                forecast_as_of=as_of,
                forecast_window_start=window_start,
                forecast_window_end=window_end,
                cycle_selection_policy="latest_hrrr_cycle_available_by_1115_local_with_75min_lag_requested_hours",
            )
        )

    for cycle, group in _group_requests_by_cycle(requests).items():
        stations = {request.station_id: {"lat": request.lat, "lon": request.lon} for request in group}
        fxx_hours = sorted({fxx for request in group for fxx in request.fxx_hours})
        try:
            values = extract_direct_nwp_run_feature_points(
                stations,
                model="hrrr",
                raw_dir=raw_dir / "hrrr_peak_timing",
                issue_utc=cycle,
                fxx_hours=fxx_hours,
                force_refresh=False,
                feature_fields=list(HRRR_FIELDS),
            )
        except TransientDirectNwpDownloadError:
            raise
        except Exception as exc:  # noqa: BLE001
            logging.warning("HRRR peak-timing fields unavailable for %s: %s", cycle.isoformat(), exc)
            for request in group:
                rows[(request.station_id, request.contract_date)].update(_unavailable_status("hrrr", str(exc)))
            continue
        for request in group:
            summary = summarize_hrrr_peak_timing(
                request,
                values.get(request.station_id, {}),
                local_hours,
            )
            row = rows[(request.station_id, request.contract_date)]
            row.update(summary)
            row.update(_features_relative_to_nbm_peak(row, summary, local_hours))


def summarize_hrrr_peak_timing(
    request: FeatureRequest,
    values_by_fxx: dict[int, dict[str, float]],
    local_hours: tuple[int, ...] = DEFAULT_LOCAL_HOURS,
) -> dict[str, Any]:
    by_hour = {
        hour: fields
        for hour, _, fields in _local_hour_values(request, values_by_fxx)
        if hour in local_hours
    }
    out: dict[str, Any] = {
        "hrrr_source_model": "hrrr",
        "hrrr_issued_at": request.cycle.isoformat(),
        "hrrr_forecast_as_of": request.forecast_as_of.isoformat(),
        "hrrr_cycle_selection_policy": request.cycle_selection_policy,
        "hrrr_hour_count_requested": len(local_hours),
        "hrrr_hour_count_returned": len(by_hour),
    }
    _write_hourly_values(out, by_hour, local_hours)

    required_count = len(local_hours) * len(HRRR_FIELDS)
    returned_count = sum(
        1
        for hour in local_hours
        for field in HRRR_FIELDS
        if pd.notna(by_hour.get(hour, {}).get(field))
    )
    out["hrrr_required_value_count"] = required_count
    out["hrrr_required_value_count_returned"] = returned_count
    out["hrrr_missing_required_value_count"] = required_count - returned_count
    out["hrrr_profile_complete"] = int(returned_count == required_count)
    out["hrrr_fetch_status"] = _status_from_count(returned_count, required_count)
    out["hrrr_unavailable_reason"] = pd.NA if returned_count else "no HRRR peak-timing fields extracted"

    temps = {
        hour: _kelvin_to_f(fields.get("temp_k_2m"))
        for hour, fields in by_hour.items()
        if pd.notna(_kelvin_to_f(fields.get("temp_k_2m")))
    }
    if temps:
        peak_hour = min(hour for hour, value in temps.items() if value == max(temps.values()))
        out["hrrr_max_post11_f"] = temps[peak_hour]
        out["hrrr_hour_of_max_local"] = peak_hour
        out["hrrr_peak_at_window_end"] = int(peak_hour == max(local_hours))
        out["hrrr_slope_11_14_f"] = _delta(temps, 11, 14)
        out["hrrr_slope_14_to_peak_f"] = _delta(temps, 14, peak_hour)
        out.update(_peak_relative_features(by_hour, peak_hour, "hrrr_peak", local_hours))

    onset = _precip_onset_hour(by_hour, local_hours)
    precip_complete = all(pd.notna(by_hour.get(hour, {}).get("precip_mm_1h")) for hour in local_hours)
    out["hrrr_precip_profile_complete"] = int(precip_complete)
    out["hrrr_precip_onset_hour_local"] = onset
    out["hrrr_no_precip_11_18"] = int(precip_complete and pd.isna(onset)) if precip_complete else pd.NA
    if pd.notna(onset) and pd.notna(out.get("hrrr_hour_of_max_local")):
        out["hrrr_precip_onset_minus_hrrr_peak_hours"] = int(onset) - int(out["hrrr_hour_of_max_local"])
    return out


def _write_hourly_values(
    out: dict[str, Any],
    by_hour: dict[int, dict[str, float]],
    local_hours: tuple[int, ...],
) -> None:
    for hour in local_hours:
        fields = by_hour.get(hour, {})
        out[f"hrrr_t{hour:02d}l_f"] = _kelvin_to_f(fields.get("temp_k_2m"))
        for field, short_name in HOURLY_FIELD_COLUMNS.items():
            suffix = "w_m2" if field == "shortwave_radiation_w_m2" else "mm" if field == "precip_mm_1h" else "pct"
            out[f"hrrr_{short_name}_{hour:02d}l_{suffix}"] = fields.get(field, pd.NA)


def _features_relative_to_nbm_peak(
    row: dict[str, Any],
    hrrr_summary: dict[str, Any],
    local_hours: tuple[int, ...],
) -> dict[str, Any]:
    peak = row.get("nbm_hour_of_max_local")
    if pd.isna(peak):
        return {}
    by_hour = _hourly_columns_to_fields(hrrr_summary, local_hours)
    out = _peak_relative_features(by_hour, int(peak), "nbm_peak", local_hours)
    onset = hrrr_summary.get("hrrr_precip_onset_hour_local")
    if pd.notna(onset):
        out["hrrr_precip_onset_minus_nbm_peak_hours"] = int(onset) - int(peak)
    return out


def _hourly_columns_to_fields(summary: dict[str, Any], local_hours: tuple[int, ...]) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for hour in local_hours:
        fields: dict[str, float] = {}
        for field, short_name in HOURLY_FIELD_COLUMNS.items():
            suffix = "w_m2" if field == "shortwave_radiation_w_m2" else "mm" if field == "precip_mm_1h" else "pct"
            value = summary.get(f"hrrr_{short_name}_{hour:02d}l_{suffix}")
            if pd.notna(value):
                fields[field] = float(value)
        out[hour] = fields
    return out


def _peak_relative_features(
    by_hour: dict[int, dict[str, float]],
    peak_hour: int,
    peak_label: str,
    local_hours: tuple[int, ...],
) -> dict[str, Any]:
    hours = [hour for hour in local_hours if hour <= peak_hour]
    if not hours:
        return {}
    out: dict[str, Any] = {}
    solar = _complete_values(by_hour, hours, "shortwave_radiation_w_m2")
    precip = _complete_values(by_hour, hours, "precip_mm_1h")
    out[f"hrrr_solar_energy_11_to_{peak_label}_wh_m2"] = sum(solar) if solar is not None else pd.NA
    out[f"hrrr_precip_total_11_to_{peak_label}_mm"] = sum(precip) if precip is not None else pd.NA
    out[f"hrrr_precip_wet_hours_11_to_{peak_label}"] = (
        sum(value >= PRECIP_ONSET_THRESHOLD_MM for value in precip) if precip is not None else pd.NA
    )
    for short_name, field in CLOUD_FIELDS.items():
        values = _complete_values(by_hour, hours, field)
        out[f"hrrr_{short_name}_11_to_{peak_label}_mean_pct"] = (
            sum(values) / len(values) if values is not None else pd.NA
        )
        out[f"hrrr_{short_name}_11_to_{peak_label}_max_pct"] = max(values) if values is not None else pd.NA
    return out


def _complete_values(
    by_hour: dict[int, dict[str, float]],
    hours: list[int],
    field: str,
) -> list[float] | None:
    values = [by_hour.get(hour, {}).get(field) for hour in hours]
    if any(pd.isna(value) for value in values):
        return None
    return [float(value) for value in values]


def _precip_onset_hour(
    by_hour: dict[int, dict[str, float]],
    local_hours: tuple[int, ...],
) -> int | Any:
    for hour in local_hours:
        value = by_hour.get(hour, {}).get("precip_mm_1h")
        if pd.notna(value) and float(value) >= PRECIP_ONSET_THRESHOLD_MM:
            return hour
    return pd.NA


def _kelvin_to_f(value: Any) -> float | Any:
    if pd.isna(value):
        return pd.NA
    return (float(value) - 273.15) * 9.0 / 5.0 + 32.0


def _delta(values: dict[int, float], left: int, right: int) -> float | Any:
    if left not in values or right not in values:
        return pd.NA
    return float(values[right]) - float(values[left])


def _completed_keys(frame: pd.DataFrame) -> set[tuple[str, str]]:
    required = {"station_id", "contract_date", "schema_version", "nbm_core_fetch_status", "hrrr_fetch_status"}
    if frame.empty or not required.issubset(frame.columns):
        return set()
    rows = frame.loc[
        pd.to_numeric(frame["schema_version"], errors="coerce").eq(SCHEMA_VERSION)
        & frame["nbm_core_fetch_status"].astype(str).str.lower().eq("ok")
        & frame["hrrr_fetch_status"].astype(str).str.lower().eq("ok")
    ]
    return {
        (str(row.station_id).upper(), str(row.contract_date)[:10])
        for row in rows[["station_id", "contract_date"]].itertuples(index=False)
    }


if __name__ == "__main__":
    main()
