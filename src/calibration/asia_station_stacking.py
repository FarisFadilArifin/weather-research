from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data_quality import add_strict_quality_flags
from .station_stacking import (
    HIGH_COLUMNS,
    OBSERVED_NUMERIC_COLUMNS,
    OBSERVED_TEXT_COLUMNS,
    V20_ASIA_NO_PEAK_FEATURE_VERSION,
    _add_calendar_features,
    _add_current_observation_derived_features,
    _add_forecast_history_delta_features,
    _add_forecast_shape_features,
    _add_lagged_actual_features,
    _add_lagged_provider_error_features,
    _add_observation_forecast_delta_features,
    _add_observation_history_delta_features,
    _add_prior_month_provider_error_features,
    _add_provider_availability_features,
    _add_provider_cross_model_features,
    _add_provider_time_features,
    _add_ensemble_features,
    add_versioned_feature_engineering,
)


ASIA_PROVIDERS = ("gfs", "gefs", "jma_msm")
ASIA_START_YEAR = 2022
ASIA_TEST_YEAR = 2026
ASIA_TIMING_MODE = "asia_same_day_11am_live_safe"
CITY_METADATA: dict[str, dict[str, Any]] = {
    "tokyo": {
        "station_id": "RJTT",
        "city_label": "Tokyo",
        "station_name": "Tokyo Haneda Airport",
        "timezone": "Asia/Tokyo",
        "country": "JP",
        "lat": 35.553,
        "lon": 139.781,
    },
    "seoul": {
        "station_id": "RKSI",
        "city_label": "Seoul",
        "station_name": "Incheon International Airport",
        "timezone": "Asia/Seoul",
        "country": "KR",
        "lat": 37.469,
        "lon": 126.451,
    },
}


def asia_expanding_folds() -> tuple[Any, ...]:
    """Return expanding folds supported by the July-2022 Asia history."""
    from .station_stacking import YearSplitFold

    return (
        YearSplitFold("fold_2022_to_2023", 2022, 2022, 2023),
        YearSplitFold("fold_2022_2023_to_2024", 2022, 2023, 2024),
        YearSplitFold("fold_2022_2024_to_2025", 2022, 2024, 2025),
    )


def _monthly_city_parts(
    root: Path,
    section: str,
    city_id: str,
    *,
    deduplicate_dates: bool = True,
) -> pd.DataFrame:
    directory = root / "normalized" / section / city_id
    paths = sorted(directory.glob("????-??.parquet"))
    frames = [pd.read_parquet(path) for path in paths]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if deduplicate_dates and "contract_date" in out:
        out["contract_date"] = out["contract_date"].astype("string").str[:10]
        out = out.sort_values("contract_date").drop_duplicates("contract_date", keep="last")
    return out.reset_index(drop=True)


def _f_to_fahrenheit(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce") * 9.0 / 5.0 + 32.0


def _provider_summary(
    frame: pd.DataFrame,
    *,
    provider: str,
    temperature_column: str,
    forecast_hour_column: str,
    as_of_hour: int,
    dewpoint_column: str | None = None,
    humidity_column: str | None = None,
    wind_speed_column: str | None = None,
    wind_direction_column: str | None = None,
    wind_gust_column: str | None = None,
    precipitation_column: str | None = None,
    cloud_column: str | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["contract_date", HIGH_COLUMNS[provider]])

    work = frame.copy()
    work["contract_date"] = work["contract_date"].astype("string").str[:10]
    work[temperature_column] = pd.to_numeric(work[temperature_column], errors="coerce")
    work["_temp_f"] = _f_to_fahrenheit(work[temperature_column])
    work[forecast_hour_column] = pd.to_numeric(work[forecast_hour_column], errors="coerce")
    work["_as_of_temp_f"] = work["_temp_f"].where(work[forecast_hour_column].eq(as_of_hour))
    work["_as_of_temp_f"] = work["_as_of_temp_f"].groupby(work["contract_date"]).transform(
        lambda values: values.dropna().iloc[0] if values.notna().any() else np.nan
    )

    aggregations: dict[str, tuple[str, str]] = {
        HIGH_COLUMNS[provider]: ("_temp_f", "max"),
        f"{provider}_forecast_temp_at_as_of_f": ("_as_of_temp_f", "first"),
        f"{provider}_forecast_hour_min": (forecast_hour_column, "min"),
        f"{provider}_forecast_hour_max": (forecast_hour_column, "max"),
    }
    optional = {
        f"{provider}_dewpoint_mean_f": (dewpoint_column, "mean", True),
        f"{provider}_humidity_mean": (humidity_column, "mean", False),
        f"{provider}_wind_speed_mean": (wind_speed_column, "mean", False),
        f"{provider}_wind_speed_max": (wind_speed_column, "max", False),
        f"{provider}_wind_direction_mean": (wind_direction_column, "mean", False),
        f"{provider}_wind_gust_max": (wind_gust_column, "max", False),
        f"{provider}_forecast_precip_total_mm": (precipitation_column, "sum", False),
        f"{provider}_forecast_precip_max_1h_mm": (precipitation_column, "max", False),
        f"{provider}_forecast_precip_hours_count": (precipitation_column, lambda values: values.gt(0).sum(), False),
        f"{provider}_cloud_cover_mean": (cloud_column, "mean", False),
        f"{provider}_cloud_cover_max": (cloud_column, "max", False),
    }
    for output, (column, function, *is_dewpoint) in optional.items():
        if column is None or column not in work:
            continue
        if is_dewpoint:
            work[column] = _f_to_fahrenheit(work[column])
        else:
            work[column] = pd.to_numeric(work[column], errors="coerce")
        aggregations[output] = (column, function)

    grouped = work.groupby("contract_date", as_index=False).agg(**aggregations)
    grouped[f"{provider}_forecast_window_hours"] = (
        grouped[f"{provider}_forecast_hour_max"] - grouped[f"{provider}_forecast_hour_min"] + 1
    )
    if f"{provider}_forecast_precip_total_mm" in grouped:
        grouped[f"{provider}_forecast_has_precip"] = (
            grouped[f"{provider}_forecast_precip_total_mm"].fillna(0).gt(0)
        ).astype(int)
    metadata_columns = {
        f"{provider}_model": "lineage",
        f"{provider}_source_label": "source_url",
        f"{provider}_forecast_as_of": "forecast_as_of_utc",
        f"{provider}_issued_at": "issued_at_utc",
        f"{provider}_source_file_or_url": "source_url",
        f"{provider}_data_source": "lineage",
        f"{provider}_fetch_status": "fetch_status",
    }
    first = work.sort_values("contract_date").groupby("contract_date", as_index=False).first()
    for output, source in metadata_columns.items():
        if source in first:
            grouped = grouped.merge(first[["contract_date", source]].rename(columns={source: output}), on="contract_date", how="left")
    return grouped


def _gefs_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["contract_date", HIGH_COLUMNS["gefs"]])
    out = frame.copy()
    out["contract_date"] = out["contract_date"].astype("string").str[:10]
    out["gefs_p50_high_f"] = _f_to_fahrenheit(out["gefs_p50_c"])
    out["gefs_mean_high_f"] = _f_to_fahrenheit(out["gefs_mean_high_c"])
    out["gefs_std_high_f"] = _f_to_fahrenheit(out["gefs_std_high_c"]) - 32.0
    out["gefs_spread_high_f"] = _f_to_fahrenheit(out["gefs_spread_c"]) - 32.0
    out = out.rename(columns={"gefs_p50_high_f": HIGH_COLUMNS["gefs"]})
    out["gefs_forecast_temp_at_as_of_f"] = out["gefs_mean_high_f"]
    out["gefs_forecast_hour_min"] = 11
    out["gefs_forecast_hour_max"] = 23
    out["gefs_forecast_window_hours"] = 13
    out["gefs_fetch_status"] = out.get("fetch_status", "ok")
    keep = [
        "contract_date",
        HIGH_COLUMNS["gefs"],
        "gefs_mean_high_f",
        "gefs_std_high_f",
        "gefs_spread_high_f",
        "gefs_forecast_temp_at_as_of_f",
        "gefs_forecast_hour_min",
        "gefs_forecast_hour_max",
        "gefs_forecast_window_hours",
        "gefs_fetch_status",
    ]
    return out[keep].drop_duplicates("contract_date", keep="last")


def build_asia_station_wide_dataset(
    data_root: str | Path,
    city_id: str,
    *,
    feature_version: str = V20_ASIA_NO_PEAK_FEATURE_VERSION,
    providers: tuple[str, ...] = ASIA_PROVIDERS,
) -> pd.DataFrame:
    city = str(city_id).strip().lower()
    if city not in CITY_METADATA:
        raise ValueError(f"Unknown Asia city: {city_id!r}")
    metadata = CITY_METADATA[city]
    expected_station = metadata["station_id"]
    selected = tuple(providers)
    unknown = sorted(set(selected) - set(ASIA_PROVIDERS))
    if unknown:
        raise ValueError(f"Unsupported Asia providers: {unknown}")
    root = Path(data_root)

    settlements = _monthly_city_parts(root, "settlements", city)
    observations = _monthly_city_parts(root, "observations", city)
    if settlements.empty or observations.empty:
        raise FileNotFoundError(f"Missing settlement or observation parquet data for {city}")
    settlements = settlements.loc[settlements["station_id"].astype(str).str.upper().eq(expected_station)].copy()
    observations = observations.loc[observations["station_id"].astype(str).str.upper().eq(expected_station)].copy()
    for column in [*OBSERVED_NUMERIC_COLUMNS, *OBSERVED_TEXT_COLUMNS]:
        if column not in observations:
            observations[column] = pd.NA
    settlements["actual_high_f"] = pd.to_numeric(settlements["settlement_high_f"], errors="coerce")
    settlements["settlement_high_f"] = settlements["actual_high_f"]
    native_celsius_columns: list[str] = []
    if "settlement_high_c" in settlements:
        settlements["settlement_high_c"] = pd.to_numeric(
            settlements["settlement_high_c"], errors="coerce"
        )
        settlements["actual_high_c"] = settlements["settlement_high_c"]
        settlements["actual_high_c_source"] = "settlement_high_c"
        native_celsius_columns = [
            "actual_high_c",
            "settlement_high_c",
            "actual_high_c_source",
        ]
    settlements["target_source"] = "wunderground_only"
    settlements["actual_source"] = settlements.get("settlement_source", "wunderground_station_history")
    settlements["actual_data_quality_flag"] = settlements.get("quality_flag", "ok")
    settlements["actual_raw_observation_count"] = pd.NA
    actuals = settlements[
        [
            "contract_date",
            "station_id",
            "actual_high_f",
            "settlement_high_f",
            *native_celsius_columns,
            "settlement_source",
            "quality_flag",
            "target_source",
            "actual_source",
            "actual_data_quality_flag",
            "actual_raw_observation_count",
        ]
    ].drop_duplicates("contract_date", keep="last")

    wide = actuals.merge(observations, on=["contract_date", "station_id"], how="left", suffixes=("", "_observation"))
    wide["station_name"] = metadata["station_name"]
    wide["airport_name"] = metadata["station_name"]
    wide["city_label"] = metadata["city_label"]
    wide["timezone"] = metadata["timezone"]
    wide["country"] = metadata["country"]
    wide["lat"] = metadata["lat"]
    wide["lon"] = metadata["lon"]
    wide["feature_version"] = feature_version
    wide["timing_mode"] = ASIA_TIMING_MODE

    provider_frames: dict[str, pd.DataFrame] = {}
    if "gfs" in selected:
        provider_frames["gfs"] = _provider_summary(
            _monthly_city_parts(root, "forecasts/gfs", city, deduplicate_dates=False),
            provider="gfs",
            temperature_column="temp_c_2m",
            forecast_hour_column="forecast_hour",
            as_of_hour=8,
            dewpoint_column="dewpoint_c_2m",
            humidity_column="relative_humidity_pct_2m",
            wind_speed_column="wind_speed_ms_10m",
            wind_direction_column="wind_direction_deg_10m",
            wind_gust_column="wind_gust_ms",
            precipitation_column="precip_mm_1h",
            cloud_column="cloud_cover_pct",
        )
    if "jma_msm" in selected:
        provider_frames["jma_msm"] = _provider_summary(
            _monthly_city_parts(root, "forecasts/jma_msm_previous_day1", city, deduplicate_dates=False),
            provider="jma_msm",
            temperature_column="temp_2m_c",
            forecast_hour_column="forecast_hour_local",
            as_of_hour=11,
            dewpoint_column="dewpoint_2m_c",
            humidity_column="relative_humidity_2m_pct",
            wind_speed_column="wind_speed_10m_kmh",
            wind_direction_column="wind_direction_10m_deg",
            wind_gust_column="wind_gusts_10m_kmh",
            precipitation_column="precipitation_mm",
            cloud_column="cloud_cover_pct",
        )
    if "gefs" in selected:
        provider_frames["gefs"] = _gefs_summary(_monthly_city_parts(root, "forecasts/gefs_ensemble_daily", city))
    for provider, provider_frame in provider_frames.items():
        wide = wide.merge(provider_frame, on="contract_date", how="left")

    wide = wide.sort_values("contract_date").reset_index(drop=True)
    wide = _add_calendar_features(wide)
    wide = _add_current_observation_derived_features(wide)
    wide = _add_provider_availability_features(wide, selected)
    wide = _add_provider_time_features(wide, selected, metadata["timezone"])
    wide = _add_ensemble_features(wide, selected)
    wide = _add_forecast_shape_features(wide, selected)
    wide = _add_provider_cross_model_features(wide, selected)
    wide = _add_lagged_actual_features(wide)
    wide = _add_lagged_provider_error_features(wide, selected)
    wide = _add_prior_month_provider_error_features(wide, selected)
    wide = _add_forecast_history_delta_features(wide, selected)
    wide = _add_observation_history_delta_features(wide)
    wide = _add_observation_forecast_delta_features(wide, selected)
    wide = add_versioned_feature_engineering(wide, feature_version=feature_version, providers=selected)
    return add_strict_quality_flags(wide, providers=selected)


def provider_readiness(data_root: str | Path, city_id: str, providers: tuple[str, ...] = ASIA_PROVIDERS) -> pd.DataFrame:
    root = Path(data_root)
    rows: list[dict[str, Any]] = []
    for provider in providers:
        section = {
            "gfs": "forecasts/gfs",
            "gefs": "forecasts/gefs_ensemble_daily",
            "jma_msm": "forecasts/jma_msm_previous_day1",
        }[provider]
        frame = _monthly_city_parts(
            root,
            section,
            city_id,
            deduplicate_dates=provider not in {"gfs", "jma_msm"},
        )
        status = frame.get("fetch_status", pd.Series("ok", index=frame.index)).astype("string").str.lower()
        rows.append(
            {
                "city_id": city_id,
                "provider": provider,
                "row_count": int(len(frame)),
                "ok_count": int(status.eq("ok").sum()),
                "first_contract_date": frame["contract_date"].min() if not frame.empty else pd.NA,
                "last_contract_date": frame["contract_date"].max() if not frame.empty else pd.NA,
                "ready": bool(not frame.empty and status.eq("ok").any()),
            }
        )
    return pd.DataFrame(rows)
