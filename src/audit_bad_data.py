from __future__ import annotations

import argparse
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
import re
from typing import Any

import pandas as pd


TARGET_STATIONS = ("KATL", "KAUS", "KORD", "KDAL", "KHOU", "KLAX", "KMIA", "KLGA", "KSEA")
FORECAST_FILES = ("sdk_nwp_0h_cache.csv", "direct_nbm_0h_cache.csv", "mostlyright_nwp_0h_cache.csv")
CURRENT_OBS_FILE = "sdk_current_observations_11am.csv"
STATION_STACKING_VERSION = "station_stacking_v5"
OPENMETEO_REQUIRED_HOURLY = (
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "cloud_cover",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
)
RAW_ACTUAL_RE = re.compile(
    r"iem_asos_(?P<cadence>1min|hourly)_(?P<station>K[A-Z0-9]{3})_"
    r"(?P<start>\d{4}-\d{2}-\d{2})(?:_(?P<end>\d{4}-\d{2}-\d{2}))?$"
)


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _series(frame: pd.DataFrame, column: str, default: Any = pd.NA) -> pd.Series:
    if column in frame:
        return frame[column]
    return pd.Series(default, index=frame.index)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(_series(frame, column), errors="coerce")


def _date_year(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_datetime(_series(frame, column).astype("string").str[:10], errors="coerce").dt.year


def _station(frame: pd.DataFrame) -> pd.Series:
    if "station_id" in frame:
        return frame["station_id"].astype("string").str.upper()
    if "station_code" in frame:
        return frame["station_code"].astype("string").str.upper()
    return pd.Series(pd.NA, index=frame.index, dtype="string")


def _provider(frame: pd.DataFrame) -> pd.Series:
    if "provider" in frame:
        return frame["provider"].astype("string").str.lower()
    if "model" in frame:
        return frame["model"].astype("string").str.lower()
    return pd.Series(pd.NA, index=frame.index, dtype="string")


def _bool_mask(mask: Any, frame: pd.DataFrame) -> pd.Series:
    if isinstance(mask, pd.Series):
        return mask.fillna(False).astype(bool)
    return pd.Series(bool(mask), index=frame.index)


def _summary_row(
    *,
    category: str,
    scope: str,
    source_path: str,
    rule: str,
    rows: int,
    bad_rows: int,
    station: str = "ALL",
    provider: str = "ALL",
    year: str = "ALL",
    severity: str = "warning",
    note: str = "",
) -> dict[str, Any]:
    return {
        "category": category,
        "scope": scope,
        "source_path": source_path,
        "station": station,
        "provider": provider,
        "year": year,
        "rule": rule,
        "severity": severity,
        "rows": int(rows),
        "bad_rows": int(bad_rows),
        "bad_pct": (float(bad_rows) / float(rows) * 100.0) if rows else 0.0,
        "note": note,
    }


def _record_rule(
    summary_rows: list[dict[str, Any]],
    example_rows: list[dict[str, Any]],
    *,
    frame: pd.DataFrame,
    category: str,
    source_path: str,
    rule: str,
    mask: Any,
    severity: str = "warning",
    note: str = "",
    group_cols: Iterable[str] = ("station", "provider", "year"),
    example_cols: Iterable[str] = (),
    examples_per_rule: int = 25,
) -> None:
    mask = _bool_mask(mask, frame)
    rows = len(frame)
    bad_rows = int(mask.sum())
    summary_rows.append(
        _summary_row(
            category=category,
            scope="overall",
            source_path=source_path,
            rule=rule,
            rows=rows,
            bad_rows=bad_rows,
            severity=severity,
            note=note,
        )
    )
    if bad_rows == 0 or rows == 0:
        return

    groupable = frame.copy()
    groupable["_bad"] = mask
    groups = [column for column in group_cols if column in groupable]
    if groups:
        grouped = groupable.groupby(groups, dropna=False)["_bad"].agg(rows="size", bad_rows="sum").reset_index()
        grouped = grouped.loc[grouped["bad_rows"].gt(0)]
        for _, row in grouped.iterrows():
            summary_rows.append(
                _summary_row(
                    category=category,
                    scope="group",
                    source_path=source_path,
                    rule=rule,
                    station=str(row["station"]) if "station" in grouped else "ALL",
                    provider=str(row["provider"]) if "provider" in grouped else "ALL",
                    year=str(int(row["year"])) if "year" in grouped and pd.notna(row["year"]) else "ALL",
                    rows=int(row["rows"]),
                    bad_rows=int(row["bad_rows"]),
                    severity=severity,
                    note=note,
                )
            )

    keep_cols = [column for column in example_cols if column in frame]
    examples = frame.loc[mask, keep_cols].head(examples_per_rule)
    for _, row in examples.iterrows():
        data = {column: row.get(column) for column in keep_cols}
        example_rows.append(
            {
                "category": category,
                "source_path": source_path,
                "rule": rule,
                "severity": severity,
                **data,
            }
        )


def _add_common_fields(frame: pd.DataFrame, *, date_col: str | None = None) -> pd.DataFrame:
    out = frame.copy()
    out["station"] = _station(out)
    out["provider"] = _provider(out)
    if date_col and date_col in out:
        out["date"] = out[date_col].astype("string").str[:10]
        out["year"] = _date_year(out, date_col)
    else:
        out["date"] = pd.NA
        out["year"] = pd.NA
    return out


def audit_processed_actuals(
    root: Path,
    summary_rows: list[dict[str, Any]],
    example_rows: list[dict[str, Any]],
    examples_per_rule: int,
) -> None:
    path = root / "data" / "processed" / "actual_highs.csv"
    if not path.exists():
        return
    frame = _add_common_fields(_read_csv(path), date_col="date_local")
    source_path = _rel(path, root)
    high = _numeric(frame, "actual_high_f")
    low = _numeric(frame, "actual_low_f")
    raw_count = _numeric(frame, "raw_observation_count")
    quality = _series(frame, "data_quality_flag").astype("string").str.lower()
    example_cols = [
        "station_code",
        "date_local",
        "actual_high_f",
        "actual_low_f",
        "actual_high_time_local",
        "source",
        "data_quality_flag",
        "raw_observation_count",
    ]

    _record_rule(
        summary_rows,
        example_rows,
        frame=frame,
        category="actual_processed",
        source_path=source_path,
        rule="actual_missing_high",
        mask=high.isna(),
        severity="critical",
        example_cols=example_cols,
        examples_per_rule=examples_per_rule,
    )
    _record_rule(
        summary_rows,
        example_rows,
        frame=frame,
        category="actual_processed",
        source_path=source_path,
        rule="actual_sparse_observations",
        mask=quality.ne("ok"),
        severity="critical",
        note="Daily high label is derived from incomplete observation coverage.",
        example_cols=example_cols,
        examples_per_rule=examples_per_rule,
    )
    _record_rule(
        summary_rows,
        example_rows,
        frame=frame,
        category="actual_processed",
        source_path=source_path,
        rule="actual_raw_count_extremely_low",
        mask=raw_count.notna() & raw_count.lt(18),
        severity="critical",
        note="Fewer than 18 temperature observations for the local day.",
        example_cols=example_cols,
        examples_per_rule=examples_per_rule,
    )
    _record_rule(
        summary_rows,
        example_rows,
        frame=frame,
        category="actual_processed",
        source_path=source_path,
        rule="actual_high_out_of_range",
        mask=high.notna() & (high.lt(-80) | high.gt(140)),
        severity="critical",
        example_cols=example_cols,
        examples_per_rule=examples_per_rule,
    )
    _record_rule(
        summary_rows,
        example_rows,
        frame=frame,
        category="actual_processed",
        source_path=source_path,
        rule="actual_low_above_high",
        mask=high.notna() & low.notna() & low.gt(high),
        severity="critical",
        example_cols=example_cols,
        examples_per_rule=examples_per_rule,
    )
    duplicated = frame.duplicated(["station", "date"], keep=False) if {"station", "date"}.issubset(frame.columns) else False
    _record_rule(
        summary_rows,
        example_rows,
        frame=frame,
        category="actual_processed",
        source_path=source_path,
        rule="actual_duplicate_station_date",
        mask=duplicated,
        severity="warning",
        example_cols=example_cols,
        examples_per_rule=examples_per_rule,
    )


def audit_sdk_actuals(
    root: Path,
    summary_rows: list[dict[str, Any]],
    example_rows: list[dict[str, Any]],
    examples_per_rule: int,
) -> None:
    for path in sorted((root / "data" / "calibration").glob("**/sdk_actual_highs.csv")):
        frame = _add_common_fields(_read_csv(path), date_col="contract_date")
        source_path = _rel(path, root)
        high = _numeric(frame, "actual_high_f")
        obs_count = _numeric(frame, "obs_count")
        fetch_status = _series(frame, "fetch_status").astype("string").str.lower().fillna("ok")
        example_cols = ["station_id", "contract_date", "actual_high_f", "actual_source", "obs_count", "fetch_status"]
        _record_rule(
            summary_rows,
            example_rows,
            frame=frame,
            category="actual_sdk",
            source_path=source_path,
            rule="sdk_actual_fetch_not_ok",
            mask=fetch_status.ne("ok"),
            severity="critical",
            example_cols=example_cols,
            examples_per_rule=examples_per_rule,
        )
        _record_rule(
            summary_rows,
            example_rows,
            frame=frame,
            category="actual_sdk",
            source_path=source_path,
            rule="sdk_actual_missing_high",
            mask=high.isna(),
            severity="critical",
            example_cols=example_cols,
            examples_per_rule=examples_per_rule,
        )
        _record_rule(
            summary_rows,
            example_rows,
            frame=frame,
            category="actual_sdk",
            source_path=source_path,
            rule="sdk_actual_low_obs_count",
            mask=obs_count.notna() & obs_count.lt(18),
            severity="warning",
            example_cols=example_cols,
            examples_per_rule=examples_per_rule,
        )


def audit_raw_actuals(
    root: Path,
    summary_rows: list[dict[str, Any]],
    example_rows: list[dict[str, Any]],
    examples_per_rule: int,
) -> None:
    raw_dir = root / "data" / "raw" / "actuals"
    if not raw_dir.exists():
        return
    rows: list[dict[str, Any]] = []
    for path in sorted(raw_dir.glob("*.csv")):
        match = RAW_ACTUAL_RE.match(path.stem)
        cadence = match.group("cadence") if match else pd.NA
        station = match.group("station") if match else pd.NA
        start_date = match.group("start") if match else pd.NA
        end_date = match.group("end") if match else pd.NA
        single_day = bool(match and not end_date)
        row: dict[str, Any] = {
            "source_path": _rel(path, root),
            "station": station,
            "provider": "iem",
            "date": start_date,
            "year": pd.to_datetime(str(start_date)[:10], errors="coerce").year,
            "cadence": cadence,
            "file_size": path.stat().st_size,
            "row_count": 0,
            "valid_temp_count": 0,
            "invalid_csv": False,
            "missing_time_or_temp_column": False,
            "no_valid_temperature_rows": False,
            "temperature_out_of_range": False,
            "single_day_extremely_low_count": False,
            "single_day_sparse_count": False,
            "note": "",
        }
        try:
            frame = pd.read_csv(path, comment="#", low_memory=False)
        except Exception as exc:  # noqa: BLE001
            row["invalid_csv"] = True
            row["note"] = str(exc)
            rows.append(row)
            continue
        row["row_count"] = len(frame)
        columns = {str(column).lower(): column for column in frame.columns}
        time_col = columns.get("valid") or columns.get("utc_valid")
        temp_col = columns.get("tmpf") or columns.get("tmpc")
        if time_col is None or temp_col is None:
            row["missing_time_or_temp_column"] = True
            row["note"] = ",".join(str(column) for column in frame.columns[:10])
            rows.append(row)
            continue
        valid_time = pd.to_datetime(frame[time_col], errors="coerce", utc=True)
        temp = pd.to_numeric(frame[temp_col], errors="coerce")
        if str(temp_col).lower() == "tmpc":
            temp = temp * 9 / 5 + 32
        valid_temp = temp[valid_time.notna() & temp.notna()]
        row["valid_temp_count"] = int(valid_temp.count())
        row["no_valid_temperature_rows"] = row["valid_temp_count"] == 0
        if not valid_temp.empty:
            row["temperature_out_of_range"] = bool((valid_temp.lt(-80) | valid_temp.gt(140)).any())
        if single_day:
            if cadence == "hourly":
                row["single_day_extremely_low_count"] = row["valid_temp_count"] < 6
                row["single_day_sparse_count"] = row["valid_temp_count"] < 18
            else:
                row["single_day_extremely_low_count"] = row["valid_temp_count"] < 18
                row["single_day_sparse_count"] = row["valid_temp_count"] < 1080
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return
    example_cols = ["source_path", "station", "date", "cadence", "file_size", "row_count", "valid_temp_count", "note"]
    raw_rules = {
        "raw_actual_invalid_csv": ("invalid_csv", "critical"),
        "raw_actual_missing_time_or_temp_column": ("missing_time_or_temp_column", "critical"),
        "raw_actual_no_valid_temperature_rows": ("no_valid_temperature_rows", "critical"),
        "raw_actual_temperature_out_of_range": ("temperature_out_of_range", "critical"),
        "raw_actual_single_day_extremely_low_count": ("single_day_extremely_low_count", "critical"),
        "raw_actual_single_day_sparse_count": ("single_day_sparse_count", "warning"),
    }
    for rule, (column, severity) in raw_rules.items():
        _record_rule(
            summary_rows,
            example_rows,
            frame=frame,
            category="actual_raw_iem",
            source_path="data/raw/actuals",
            rule=rule,
            mask=frame[column],
            severity=severity,
            example_cols=example_cols,
            examples_per_rule=examples_per_rule,
        )


def audit_current_observations(
    root: Path,
    summary_rows: list[dict[str, Any]],
    example_rows: list[dict[str, Any]],
    examples_per_rule: int,
) -> None:
    for path in sorted((root / "data" / "calibration").glob(f"**/{CURRENT_OBS_FILE}")):
        frame = _add_common_fields(_read_csv(path), date_col="contract_date")
        source_path = _rel(path, root)
        fetch_status = _series(frame, "observed_fetch_status").astype("string").str.lower()
        temp = _numeric(frame, "observed_temp_at_as_of_f")
        high_so_far = _numeric(frame, "observed_high_temp_through_as_of_f")
        dewpoint = _numeric(frame, "observed_dewpoint_at_as_of_f")
        humidity = _numeric(frame, "observed_humidity_at_as_of")
        wind_speed = _numeric(frame, "observed_wind_speed_at_as_of")
        wind_gust = _numeric(frame, "observed_wind_gust_at_as_of")
        direction = _numeric(frame, "observed_wind_direction_at_as_of")
        pressure = _numeric(frame, "observed_pressure_at_as_of")
        visibility = _numeric(frame, "observed_visibility_at_as_of")
        cloud_cover = _numeric(frame, "observed_cloud_cover_at_as_of")
        precip_recent = _numeric(frame, "observed_precip_recent_at_as_of")
        age = _numeric(frame, "observed_as_of_age_minutes")
        as_of_text = _series(frame, "observed_as_of_time_local").astype("string")
        local_clock = as_of_text.str.extract(r"T(?P<hour>\d{2}):(?P<minute>\d{2})")
        local_hour = pd.to_numeric(local_clock["hour"], errors="coerce")
        local_minute = pd.to_numeric(local_clock["minute"], errors="coerce")
        local_minutes = local_hour * 60 + local_minute
        has_local_clock = local_hour.notna() & local_minute.notna()
        in_decision_window = local_minutes.between(10 * 60 + 50, 11 * 60 + 10)
        example_cols = [
            "station_id",
            "contract_date",
            "timing_mode",
            "observed_temp_at_as_of_f",
            "observed_high_temp_through_as_of_f",
            "observed_as_of_time_local",
            "observed_fetch_status",
            "observed_unavailable_reason",
            "observed_raw_metar",
        ]
        _record_rule(
            summary_rows,
            example_rows,
            frame=frame,
            category="current_observation",
            source_path=source_path,
            rule="observed_fetch_not_ok",
            mask=fetch_status.ne("ok"),
            severity="critical",
            example_cols=example_cols,
            examples_per_rule=examples_per_rule,
        )
        _record_rule(
            summary_rows,
            example_rows,
            frame=frame,
            category="current_observation",
            source_path=source_path,
            rule="observed_ok_missing_temp",
            mask=fetch_status.eq("ok") & temp.isna(),
            severity="critical",
            example_cols=example_cols,
            examples_per_rule=examples_per_rule,
        )
        _record_rule(
            summary_rows,
            example_rows,
            frame=frame,
            category="current_observation",
            source_path=source_path,
            rule="observed_high_so_far_below_temp",
            mask=temp.notna() & high_so_far.notna() & high_so_far.lt(temp),
            severity="critical",
            example_cols=example_cols,
            examples_per_rule=examples_per_rule,
        )
        _record_rule(
            summary_rows,
            example_rows,
            frame=frame,
            category="current_observation",
            source_path=source_path,
            rule="observed_outside_1050_1110_window",
            mask=fetch_status.eq("ok") & (~has_local_clock | ~in_decision_window),
            severity="warning",
            note="Deployment contract expects observations between 10:50 and 11:10 local.",
            example_cols=example_cols,
            examples_per_rule=examples_per_rule,
        )
        _record_rule(
            summary_rows,
            example_rows,
            frame=frame,
            category="current_observation",
            source_path=source_path,
            rule="observed_stale_age_gt_20_minutes",
            mask=fetch_status.eq("ok") & age.notna() & age.gt(20),
            severity="warning",
            example_cols=example_cols,
            examples_per_rule=examples_per_rule,
        )
        numeric_rules = {
            "observed_temp_out_of_range": temp.notna() & (temp.lt(-80) | temp.gt(140)),
            "observed_dewpoint_out_of_range": dewpoint.notna() & (dewpoint.lt(-100) | dewpoint.gt(100)),
            "observed_humidity_out_of_range": humidity.notna() & (humidity.lt(0) | humidity.gt(100)),
            "observed_wind_speed_out_of_range": wind_speed.notna() & (wind_speed.lt(0) | wind_speed.gt(150)),
            "observed_wind_gust_out_of_range": wind_gust.notna() & (wind_gust.lt(0) | wind_gust.gt(180)),
            "observed_wind_direction_out_of_range": direction.notna() & (direction.lt(0) | direction.gt(360)),
            "observed_pressure_out_of_range": pressure.notna() & (pressure.lt(850) | pressure.gt(1100)),
            "observed_visibility_out_of_range": visibility.notna() & (visibility.lt(0) | visibility.gt(100)),
            "observed_cloud_cover_out_of_range": cloud_cover.notna() & (cloud_cover.lt(0) | cloud_cover.gt(100)),
            "observed_precip_recent_negative": precip_recent.notna() & precip_recent.lt(0),
        }
        for rule, mask in numeric_rules.items():
            _record_rule(
                summary_rows,
                example_rows,
                frame=frame,
                category="current_observation",
                source_path=source_path,
                rule=rule,
                mask=mask,
                severity="critical",
                example_cols=example_cols,
                examples_per_rule=examples_per_rule,
            )

        duplicate_key = ["station", "date", "timing_mode"]
        duplicated = frame.duplicated(duplicate_key, keep=False) if set(duplicate_key).issubset(frame.columns) else False
        _record_rule(
            summary_rows,
            example_rows,
            frame=frame,
            category="current_observation",
            source_path=source_path,
            rule="observed_duplicate_station_date_timing",
            mask=duplicated,
            severity="warning",
            example_cols=example_cols,
            examples_per_rule=examples_per_rule,
        )


def audit_forecasts(
    root: Path,
    summary_rows: list[dict[str, Any]],
    example_rows: list[dict[str, Any]],
    examples_per_rule: int,
) -> None:
    calibration_dir = root / "data" / "calibration"
    paths: list[Path] = []
    for filename in FORECAST_FILES:
        paths.extend(calibration_dir.glob(f"**/{filename}"))
    for path in sorted(set(paths)):
        frame = _add_common_fields(_read_csv(path), date_col="contract_date")
        source_path = _rel(path, root)
        fetch_status = _series(frame, "fetch_status").astype("string").str.lower().fillna("ok")
        provider = _provider(frame)
        model = _series(frame, "model").astype("string").str.lower()
        high = _numeric(frame, "raw_forecast_high_f")
        dewpoint = _numeric(frame, "dewpoint_mean_f")
        humidity = _numeric(frame, "humidity_mean")
        wind_speed_mean = _numeric(frame, "wind_speed_mean")
        wind_speed_max = _numeric(frame, "wind_speed_max")
        wind_gust = _numeric(frame, "wind_gust_max")
        wind_direction = _numeric(frame, "wind_direction_mean")
        precip_amount = _numeric(frame, "precip_amount")
        precip_total = _numeric(frame, "forecast_precip_total_mm")
        precip_max_1h = _numeric(frame, "forecast_precip_max_1h_mm")
        precip_hours = _numeric(frame, "forecast_precip_hours_count")
        has_precip = _numeric(frame, "forecast_has_precip")
        intensity = _numeric(frame, "forecast_precip_intensity_code")
        horizon = _numeric(frame, "horizon_hours")
        fmin = _numeric(frame, "forecast_hour_min")
        fmax = _numeric(frame, "forecast_hour_max")
        grid_dist = _numeric(frame, "grid_dist_km_mean")
        issued = pd.to_datetime(_series(frame, "issued_at"), errors="coerce", utc=True)
        as_of = pd.to_datetime(_series(frame, "forecast_as_of"), errors="coerce", utc=True)
        window_start = pd.to_datetime(_series(frame, "forecast_window_start"), errors="coerce", utc=True)
        window_end = pd.to_datetime(_series(frame, "forecast_window_end"), errors="coerce", utc=True)
        example_cols = [
            "station_id",
            "provider",
            "model",
            "timing_mode",
            "contract_date",
            "forecast_as_of",
            "issued_at",
            "forecast_window_start",
            "forecast_window_end",
            "raw_forecast_high_f",
            "fetch_status",
            "unavailable_reason",
            "source_cache_dir",
        ]
        _record_rule(
            summary_rows,
            example_rows,
            frame=frame,
            category="forecast",
            source_path=source_path,
            rule="forecast_fetch_not_ok",
            mask=fetch_status.ne("ok"),
            severity="critical",
            example_cols=example_cols,
            examples_per_rule=examples_per_rule,
        )
        _record_rule(
            summary_rows,
            example_rows,
            frame=frame,
            category="forecast",
            source_path=source_path,
            rule="forecast_ok_missing_high",
            mask=fetch_status.eq("ok") & high.isna(),
            severity="critical",
            example_cols=example_cols,
            examples_per_rule=examples_per_rule,
        )
        _record_rule(
            summary_rows,
            example_rows,
            frame=frame,
            category="forecast",
            source_path=source_path,
            rule="forecast_high_out_of_range",
            mask=high.notna() & (high.lt(-80) | high.gt(140)),
            severity="critical",
            example_cols=example_cols,
            examples_per_rule=examples_per_rule,
        )
        _record_rule(
            summary_rows,
            example_rows,
            frame=frame,
            category="forecast",
            source_path=source_path,
            rule="forecast_provider_model_mismatch",
            mask=provider.notna() & model.notna() & provider.ne(model),
            severity="warning",
            example_cols=example_cols,
            examples_per_rule=examples_per_rule,
        )
        time_rules = {
            "forecast_issued_after_as_of": issued.notna() & as_of.notna() & issued.gt(as_of),
            "forecast_window_end_before_start": window_start.notna() & window_end.notna() & window_end.le(window_start),
            "forecast_window_start_before_as_of": window_start.notna() & as_of.notna() & window_start.lt(as_of - pd.Timedelta(minutes=1)),
        }
        for rule, mask in time_rules.items():
            _record_rule(
                summary_rows,
                example_rows,
                frame=frame,
                category="forecast",
                source_path=source_path,
                rule=rule,
                mask=mask,
                severity="warning",
                example_cols=example_cols,
                examples_per_rule=examples_per_rule,
            )
        numeric_rules = {
            "forecast_horizon_out_of_range": horizon.notna() & (horizon.lt(0) | horizon.gt(240)),
            "forecast_hour_min_gt_max": fmin.notna() & fmax.notna() & fmin.gt(fmax),
            "forecast_grid_distance_out_of_range": grid_dist.notna() & (grid_dist.lt(0) | grid_dist.gt(100)),
            "forecast_dewpoint_out_of_range": dewpoint.notna() & (dewpoint.lt(-100) | dewpoint.gt(100)),
            "forecast_humidity_out_of_range": humidity.notna() & (humidity.lt(0) | humidity.gt(100)),
            "forecast_wind_speed_mean_out_of_range": wind_speed_mean.notna()
            & (wind_speed_mean.lt(0) | wind_speed_mean.gt(150)),
            "forecast_wind_speed_max_out_of_range": wind_speed_max.notna()
            & (wind_speed_max.lt(0) | wind_speed_max.gt(150)),
            "forecast_wind_gust_out_of_range": wind_gust.notna() & (wind_gust.lt(0) | wind_gust.gt(180)),
            "forecast_wind_direction_out_of_range": wind_direction.notna()
            & (wind_direction.lt(0) | wind_direction.gt(360)),
            "forecast_precip_amount_negative": precip_amount.notna() & precip_amount.lt(0),
            "forecast_precip_total_negative": precip_total.notna() & precip_total.lt(0),
            "forecast_precip_max_1h_negative": precip_max_1h.notna() & precip_max_1h.lt(0),
            "forecast_precip_hours_out_of_range": precip_hours.notna() & (precip_hours.lt(0) | precip_hours.gt(24)),
            "forecast_has_precip_not_binary": has_precip.notna() & ~has_precip.isin([0, 1]),
            "forecast_precip_intensity_out_of_range": intensity.notna() & (intensity.lt(0) | intensity.gt(4)),
        }
        for rule, mask in numeric_rules.items():
            _record_rule(
                summary_rows,
                example_rows,
                frame=frame,
                category="forecast",
                source_path=source_path,
                rule=rule,
                mask=mask,
                severity="critical",
                example_cols=example_cols,
                examples_per_rule=examples_per_rule,
            )
        duplicate_key = ["station", "provider", "date", "timing_mode"]
        duplicated = frame.duplicated(duplicate_key, keep=False) if set(duplicate_key).issubset(frame.columns) else False
        _record_rule(
            summary_rows,
            example_rows,
            frame=frame,
            category="forecast",
            source_path=source_path,
            rule="forecast_duplicate_station_provider_date_timing",
            mask=duplicated,
            severity="warning",
            example_cols=example_cols,
            examples_per_rule=examples_per_rule,
        )


def audit_processed_forecast_snapshots(
    root: Path,
    summary_rows: list[dict[str, Any]],
    example_rows: list[dict[str, Any]],
    examples_per_rule: int,
) -> None:
    processed_dir = root / "data" / "processed"
    for path in sorted(processed_dir.glob("*forecast_snapshots.csv")):
        frame = _read_csv(path)
        source_path = _rel(path, root)
        if frame.empty:
            summary_rows.append(
                _summary_row(
                    category="forecast_processed",
                    scope="overall",
                    source_path=source_path,
                    rule="processed_forecast_empty_file",
                    rows=1,
                    bad_rows=1,
                    severity="warning",
                    note="Processed forecast snapshot file has headers but no rows.",
                )
            )
            continue
        frame = _add_common_fields(frame, date_col="target_date_local")
        high = _numeric(frame, "forecast_high_f")
        low = _numeric(frame, "forecast_low_f")
        hourly_max = _numeric(frame, "forecast_hourly_max_f")
        horizon = _numeric(frame, "forecast_horizon_hours")
        cloud_mean = _numeric(frame, "cloud_cover_mean")
        cloud_max = _numeric(frame, "cloud_cover_max")
        precip = _numeric(frame, "precip_amount")
        wind_mean = _numeric(frame, "wind_speed_mean")
        wind_max = _numeric(frame, "wind_speed_max")
        wind_dir = _numeric(frame, "wind_direction_mean")
        dewpoint = _numeric(frame, "dewpoint_mean_f")
        humidity = _numeric(frame, "humidity_mean")
        example_cols = [
            "station_code",
            "provider",
            "model",
            "target_date_local",
            "forecast_horizon_hours",
            "forecast_high_f",
            "forecast_low_f",
            "forecast_hourly_max_f",
            "source_file_or_url",
        ]
        rules = {
            "processed_forecast_missing_high": high.isna(),
            "processed_forecast_high_out_of_range": high.notna() & (high.lt(-80) | high.gt(140)),
            "processed_forecast_low_above_high": high.notna() & low.notna() & low.gt(high),
            "processed_forecast_hourly_max_mismatch": high.notna()
            & hourly_max.notna()
            & (high - hourly_max).abs().gt(0.25),
            "processed_forecast_horizon_out_of_range": horizon.notna() & (horizon.lt(0) | horizon.gt(240)),
            "processed_forecast_cloud_mean_out_of_range": cloud_mean.notna() & (cloud_mean.lt(0) | cloud_mean.gt(100)),
            "processed_forecast_cloud_max_out_of_range": cloud_max.notna() & (cloud_max.lt(0) | cloud_max.gt(100)),
            "processed_forecast_precip_negative": precip.notna() & precip.lt(0),
            "processed_forecast_wind_mean_out_of_range": wind_mean.notna() & (wind_mean.lt(0) | wind_mean.gt(150)),
            "processed_forecast_wind_max_out_of_range": wind_max.notna() & (wind_max.lt(0) | wind_max.gt(150)),
            "processed_forecast_wind_direction_out_of_range": wind_dir.notna() & (wind_dir.lt(0) | wind_dir.gt(360)),
            "processed_forecast_dewpoint_out_of_range": dewpoint.notna() & (dewpoint.lt(-100) | dewpoint.gt(100)),
            "processed_forecast_humidity_out_of_range": humidity.notna() & (humidity.lt(0) | humidity.gt(100)),
        }
        for rule, mask in rules.items():
            _record_rule(
                summary_rows,
                example_rows,
                frame=frame,
                category="forecast_processed",
                source_path=source_path,
                rule=rule,
                mask=mask,
                severity="critical",
                example_cols=example_cols,
                examples_per_rule=examples_per_rule,
            )
        duplicate_key = ["station", "provider", "date", "forecast_horizon_hours"]
        duplicated = frame.duplicated(duplicate_key, keep=False) if set(duplicate_key).issubset(frame.columns) else False
        _record_rule(
            summary_rows,
            example_rows,
            frame=frame,
            category="forecast_processed",
            source_path=source_path,
            rule="processed_forecast_duplicate_station_provider_date_horizon",
            mask=duplicated,
            severity="warning",
            example_cols=example_cols,
            examples_per_rule=examples_per_rule,
        )


def audit_raw_openmeteo(
    root: Path,
    summary_rows: list[dict[str, Any]],
    example_rows: list[dict[str, Any]],
    examples_per_rule: int,
) -> None:
    raw_dir = root / "data" / "raw" / "openmeteo"
    if not raw_dir.exists():
        return
    rows: list[dict[str, Any]] = []
    for path in sorted(raw_dir.glob("openmeteo_*.json")):
        parts = path.stem.split("_")
        station = parts[1] if len(parts) >= 4 else pd.NA
        target_date = parts[2] if len(parts) >= 4 else pd.NA
        horizon = parts[3] if len(parts) >= 4 else pd.NA
        row: dict[str, Any] = {
            "source_path": _rel(path, root),
            "station": str(station).upper() if pd.notna(station) else pd.NA,
            "provider": "openmeteo",
            "date": target_date,
            "year": pd.to_datetime(str(target_date)[:10], errors="coerce").year,
            "horizon": horizon,
            "file_size": path.stat().st_size,
            "invalid_json": False,
            "payload_error": False,
            "missing_hourly": False,
            "missing_hourly_time": False,
            "hourly_count_not_24": False,
            "hourly_length_mismatch": False,
            "missing_required_hourly_variable": False,
            "temperature_out_of_range": False,
            "humidity_out_of_range": False,
            "cloud_cover_out_of_range": False,
            "precipitation_negative": False,
            "wind_speed_out_of_range": False,
            "wind_direction_out_of_range": False,
            "missing_variables": "",
        }
        try:
            import json

            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            row["invalid_json"] = True
            row["missing_variables"] = str(exc)
            rows.append(row)
            continue
        row["payload_error"] = bool(payload.get("error"))
        hourly = payload.get("hourly")
        if not isinstance(hourly, dict):
            row["missing_hourly"] = True
            rows.append(row)
            continue
        times = hourly.get("time")
        if not isinstance(times, list):
            row["missing_hourly_time"] = True
        else:
            row["hourly_count_not_24"] = len(times) != 24
        missing_vars = [name for name in OPENMETEO_REQUIRED_HOURLY if name not in hourly]
        row["missing_required_hourly_variable"] = bool(missing_vars)
        row["missing_variables"] = ",".join(missing_vars)
        expected_len = len(times) if isinstance(times, list) else None
        for name in OPENMETEO_REQUIRED_HOURLY:
            values = hourly.get(name)
            if not isinstance(values, list):
                continue
            if expected_len is not None and len(values) != expected_len:
                row["hourly_length_mismatch"] = True
            series = pd.to_numeric(pd.Series(values), errors="coerce")
            if name == "temperature_2m" and series.notna().any():
                row["temperature_out_of_range"] = bool((series.lt(-80) | series.gt(140)).any())
            elif name == "relative_humidity_2m" and series.notna().any():
                row["humidity_out_of_range"] = bool((series.lt(0) | series.gt(100)).any())
            elif name == "cloud_cover" and series.notna().any():
                row["cloud_cover_out_of_range"] = bool((series.lt(0) | series.gt(100)).any())
            elif name == "precipitation" and series.notna().any():
                row["precipitation_negative"] = bool(series.lt(0).any())
            elif name == "wind_speed_10m" and series.notna().any():
                row["wind_speed_out_of_range"] = bool((series.lt(0) | series.gt(150)).any())
            elif name == "wind_direction_10m" and series.notna().any():
                row["wind_direction_out_of_range"] = bool((series.lt(0) | series.gt(360)).any())
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return
    example_cols = ["source_path", "station", "date", "horizon", "file_size", "missing_variables"]
    raw_rules = {
        "raw_openmeteo_invalid_json": "invalid_json",
        "raw_openmeteo_payload_error": "payload_error",
        "raw_openmeteo_missing_hourly": "missing_hourly",
        "raw_openmeteo_missing_hourly_time": "missing_hourly_time",
        "raw_openmeteo_hourly_count_not_24": "hourly_count_not_24",
        "raw_openmeteo_hourly_length_mismatch": "hourly_length_mismatch",
        "raw_openmeteo_missing_required_hourly_variable": "missing_required_hourly_variable",
        "raw_openmeteo_temperature_out_of_range": "temperature_out_of_range",
        "raw_openmeteo_humidity_out_of_range": "humidity_out_of_range",
        "raw_openmeteo_cloud_cover_out_of_range": "cloud_cover_out_of_range",
        "raw_openmeteo_precipitation_negative": "precipitation_negative",
        "raw_openmeteo_wind_speed_out_of_range": "wind_speed_out_of_range",
        "raw_openmeteo_wind_direction_out_of_range": "wind_direction_out_of_range",
    }
    for rule, column in raw_rules.items():
        _record_rule(
            summary_rows,
            example_rows,
            frame=frame,
            category="forecast_raw_openmeteo",
            source_path="data/raw/openmeteo",
            rule=rule,
            mask=frame[column],
            severity="critical",
            example_cols=example_cols,
            examples_per_rule=examples_per_rule,
        )


def audit_raw_grib_files(
    root: Path,
    summary_rows: list[dict[str, Any]],
    example_rows: list[dict[str, Any]],
    examples_per_rule: int,
) -> None:
    rows: list[dict[str, Any]] = []
    for raw_subdir in ("hrrr", "nws"):
        raw_dir = root / "data" / "raw" / raw_subdir
        if not raw_dir.exists():
            continue
        for path in sorted(raw_dir.glob("*.grib2")):
            rows.append(
                {
                    "source_path": _rel(path, root),
                    "station": pd.NA,
                    "provider": raw_subdir,
                    "year": pd.NA,
                    "file_size": path.stat().st_size,
                    "too_small": path.stat().st_size < 1000,
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return
    _record_rule(
        summary_rows,
        example_rows,
        frame=frame,
        category="forecast_raw_grib",
        source_path="data/raw/{hrrr,nws}",
        rule="raw_grib_file_too_small",
        mask=frame["too_small"],
        severity="critical",
        example_cols=["source_path", "provider", "file_size"],
        examples_per_rule=examples_per_rule,
    )


def audit_joined_station_features(
    root: Path,
    summary_rows: list[dict[str, Any]],
    example_rows: list[dict[str, Any]],
    examples_per_rule: int,
) -> None:
    stacking_dir = root / "data" / "calibration" / STATION_STACKING_VERSION
    if not stacking_dir.exists():
        return
    actuals_path = root / "data" / "processed" / "actual_highs.csv"
    actual_quality = pd.DataFrame()
    if actuals_path.exists():
        actual_quality = _read_csv(actuals_path)
        actual_quality["station"] = _station(actual_quality)
        actual_quality["date"] = actual_quality["date_local"].astype("string").str[:10]
        actual_quality = actual_quality[["station", "date", "data_quality_flag", "raw_observation_count"]]
    for path in sorted(stacking_dir.glob("*_features.csv")):
        if "feature_columns" in path.name:
            continue
        station_id = path.name.split("_")[0]
        columns = set(pd.read_csv(path, nrows=0).columns)
        needed = {
            "contract_date",
            "actual_high_f",
            "observed_temp_at_as_of_f",
            "observed_high_temp_through_as_of_f",
            "gfs_high_f",
            "hrrr_high_f",
        }
        read_cols = [column for column in needed if column in columns]
        if not {"contract_date", "actual_high_f"}.issubset(read_cols):
            continue
        frame = _read_csv(path)[read_cols]
        frame["station"] = station_id
        frame["provider"] = "ALL"
        frame["date"] = frame["contract_date"].astype("string").str[:10]
        frame["year"] = _date_year(frame, "contract_date")
        if not actual_quality.empty:
            frame = frame.merge(actual_quality, on=["station", "date"], how="left")
        source_path = _rel(path, root)
        actual_high = _numeric(frame, "actual_high_f")
        observed_temp = _numeric(frame, "observed_temp_at_as_of_f")
        high_so_far = _numeric(frame, "observed_high_temp_through_as_of_f")
        example_cols = [
            "station",
            "contract_date",
            "actual_high_f",
            "observed_temp_at_as_of_f",
            "observed_high_temp_through_as_of_f",
            "gfs_high_f",
            "hrrr_high_f",
            "data_quality_flag",
            "raw_observation_count",
        ]
        _record_rule(
            summary_rows,
            example_rows,
            frame=frame,
            category="joined_station_features",
            source_path=source_path,
            rule="joined_actual_below_observed_high_so_far",
            mask=actual_high.notna() & high_so_far.notna() & actual_high.lt(high_so_far),
            severity="critical",
            note="Final daily high cannot be below the current-observation high-so-far.",
            group_cols=("station", "year"),
            example_cols=example_cols,
            examples_per_rule=examples_per_rule,
        )
        _record_rule(
            summary_rows,
            example_rows,
            frame=frame,
            category="joined_station_features",
            source_path=source_path,
            rule="joined_observed_temp_above_actual_high",
            mask=actual_high.notna() & observed_temp.notna() & observed_temp.gt(actual_high),
            severity="critical",
            note="11 AM observed temperature cannot exceed final daily high.",
            group_cols=("station", "year"),
            example_cols=example_cols,
            examples_per_rule=examples_per_rule,
        )
        for provider in ("gfs", "hrrr"):
            high_col = f"{provider}_high_f"
            if high_col not in frame:
                continue
            provider_high = _numeric(frame, high_col)
            _record_rule(
                summary_rows,
                example_rows,
                frame=frame.assign(provider=provider),
                category="joined_station_features",
                source_path=source_path,
                rule=f"joined_{provider}_forecast_below_observed_high_so_far",
                mask=provider_high.notna() & high_so_far.notna() & provider_high.lt(high_so_far),
                severity="warning",
                note="Not always corrupt, but risky if raw forecast high is interpreted as final-day high.",
                group_cols=("station", "provider", "year"),
                example_cols=example_cols,
                examples_per_rule=examples_per_rule,
            )


def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 20) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.loc[:, columns].head(max_rows).copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.2f}")
        else:
            display[column] = display[column].astype(str)
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in display.to_numpy()]
    return "\n".join([header, sep, *rows])


def write_report(summary: pd.DataFrame, examples: pd.DataFrame, output_dir: Path) -> Path:
    report_path = output_dir / "bad_data_audit_report.md"
    overall = summary.loc[summary["scope"].eq("overall")].copy()
    nonzero = overall.loc[overall["bad_rows"].gt(0)].copy()
    nonzero = nonzero.sort_values(["severity", "bad_rows"], ascending=[True, False])

    category_rollup = (
        nonzero.groupby("category", dropna=False)
        .agg(rules_with_bad_rows=("rule", "nunique"), bad_rows=("bad_rows", "sum"))
        .reset_index()
        .sort_values("bad_rows", ascending=False)
    )
    top_rules = nonzero[["category", "rule", "severity", "bad_rows", "bad_pct", "source_path"]].head(30)

    joined = summary.loc[
        summary["scope"].eq("group") & summary["category"].eq("joined_station_features"),
        ["station", "rule", "year", "rows", "bad_rows", "bad_pct"],
    ].sort_values(["rule", "bad_rows"], ascending=[True, False])

    lines = [
        "# Bad Data Audit",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "This audit scans processed actual labels, raw IEM actual files, SDK actual caches, current-observation caches, forecast caches, legacy processed forecast snapshots, raw forecast payload/files, and station-stacking joined feature files.",
        "",
        "## Category Rollup",
        "",
        _markdown_table(category_rollup, ["category", "rules_with_bad_rows", "bad_rows"]),
        "",
        "## Top Nonzero Rules",
        "",
        _markdown_table(top_rules, ["category", "rule", "severity", "bad_rows", "bad_pct", "source_path"], max_rows=30),
        "",
        "## Joined Feature Sanity Checks",
        "",
        _markdown_table(joined, ["station", "rule", "year", "rows", "bad_rows", "bad_pct"], max_rows=60),
        "",
        "## Output Files",
        "",
        "- `bad_data_summary.csv`: all rule counts, including overall and grouped rows.",
        "- `bad_data_examples.csv`: sample offending rows per rule.",
        "- `bad_data_audit_report.md`: this report.",
    ]
    if not examples.empty:
        lines.extend(
            [
                "",
                "## Example Rules Captured",
                "",
                _markdown_table(
                    examples[["category", "rule", "severity", "source_path"]].drop_duplicates().head(40),
                    ["category", "rule", "severity", "source_path"],
                    max_rows=40,
                ),
            ]
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def run_audit(project_root: str | Path = ".", output_dir: str | Path = "outputs/bad_data_audit", examples_per_rule: int = 25) -> dict[str, Path]:
    root = Path(project_root).resolve()
    out_dir = (root / output_dir).resolve() if not Path(output_dir).is_absolute() else Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    example_rows: list[dict[str, Any]] = []

    audit_processed_actuals(root, summary_rows, example_rows, examples_per_rule)
    audit_sdk_actuals(root, summary_rows, example_rows, examples_per_rule)
    audit_raw_actuals(root, summary_rows, example_rows, examples_per_rule)
    audit_current_observations(root, summary_rows, example_rows, examples_per_rule)
    audit_forecasts(root, summary_rows, example_rows, examples_per_rule)
    audit_processed_forecast_snapshots(root, summary_rows, example_rows, examples_per_rule)
    audit_raw_openmeteo(root, summary_rows, example_rows, examples_per_rule)
    audit_raw_grib_files(root, summary_rows, example_rows, examples_per_rule)
    audit_joined_station_features(root, summary_rows, example_rows, examples_per_rule)

    summary = pd.DataFrame(summary_rows)
    examples = pd.DataFrame(example_rows)
    summary_path = out_dir / "bad_data_summary.csv"
    examples_path = out_dir / "bad_data_examples.csv"
    summary.to_csv(summary_path, index=False)
    examples.to_csv(examples_path, index=False)
    report_path = write_report(summary, examples, out_dir)
    return {"summary": summary_path, "examples": examples_path, "report": report_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit weather research actual, observation, forecast, and joined data quality.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-dir", default="outputs/bad_data_audit")
    parser.add_argument("--examples-per-rule", type=int, default=25)
    args = parser.parse_args()
    paths = run_audit(args.project_root, args.output_dir, args.examples_per_rule)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
