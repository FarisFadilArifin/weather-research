from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

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
    "busan": {
        "station_id": "RKPK",
        "city_label": "Busan",
        "station_name": "Gimhae International Airport",
        "timezone": "Asia/Seoul",
        "country": "KR",
        "lat": 35.1795,
        "lon": 128.9382,
    },
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


_OPTIONAL_FIELD_BOUNDS: dict[str, tuple[float, float]] = {
    "temperature_c": (-100.0, 70.0),
    "percentage": (0.0, 100.0),
    "wind_speed": (0.0, 500.0),
    "direction_degrees": (0.0, 360.0),
    "precipitation_mm": (0.0, 500.0),
}


def _normalized_optional_provider_field(
    values: pd.Series,
    *,
    provider: str,
    column: str,
    dimension: str,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    try:
        minimum, maximum = _OPTIONAL_FIELD_BOUNDS[dimension]
    except KeyError as exc:
        raise ValueError(f"unsupported optional-field dimension: {dimension}") from exc
    invalid = numeric.notna() & ((numeric < minimum) | (numeric > maximum))
    if invalid.any():
        raise ValueError(
            "provider_optional_field_out_of_bounds:"
            f"{provider}:{column}:{dimension}"
        )
    if dimension == "temperature_c":
        return _f_to_fahrenheit(numeric)
    return numeric


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
        f"{provider}_dewpoint_mean_f": (dewpoint_column, "mean", "temperature_c"),
        f"{provider}_humidity_mean": (humidity_column, "mean", "percentage"),
        f"{provider}_wind_speed_mean": (wind_speed_column, "mean", "wind_speed"),
        f"{provider}_wind_speed_max": (wind_speed_column, "max", "wind_speed"),
        f"{provider}_wind_direction_mean": (wind_direction_column, "mean", "direction_degrees"),
        f"{provider}_wind_gust_max": (wind_gust_column, "max", "wind_speed"),
        f"{provider}_forecast_precip_total_mm": (precipitation_column, "sum", "precipitation_mm"),
        f"{provider}_forecast_precip_max_1h_mm": (precipitation_column, "max", "precipitation_mm"),
        f"{provider}_forecast_precip_hours_count": (precipitation_column, lambda values: values.gt(0).sum(), "precipitation_mm"),
        f"{provider}_cloud_cover_mean": (cloud_column, "mean", "percentage"),
        f"{provider}_cloud_cover_max": (cloud_column, "max", "percentage"),
    }
    normalized_columns: dict[tuple[str, str], str] = {}
    for output, (column, function, dimension) in optional.items():
        if column is None or column not in work:
            continue
        key = (column, dimension)
        normalized_column = normalized_columns.get(key)
        if normalized_column is None:
            normalized_column = f"_optional_{len(normalized_columns)}"
            work[normalized_column] = _normalized_optional_provider_field(
                work[column],
                provider=provider,
                column=column,
                dimension=dimension,
            )
            normalized_columns[key] = normalized_column
        aggregations[output] = (normalized_column, function)

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
        settlements["settlement_high_c_source"] = settlements.get(
            "settlement_source", "wunderground_station_history"
        )
        settlements["actual_high_c_settlement_source"] = settlements[
            "settlement_high_c_source"
        ]
        native_celsius_columns = [
            "actual_high_c",
            "settlement_high_c",
            "actual_high_c_source",
            "actual_high_c_settlement_source",
            "settlement_high_c_source",
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


def _latest_live_part(directory: Path, contract_date: date) -> Path:
    paths = sorted(directory.glob(f"{contract_date.isoformat()}_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"missing_live_input:{directory}:{contract_date}")
    return paths[-1]


def _composite_sha256(paths: list[Path]) -> str:
    if not paths:
        raise ValueError("source_checksum_missing")
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _finite_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> bool:
    return all(
        column in frame
        and pd.to_numeric(frame[column], errors="coerce").notna().all()
        for column in columns
    )


def _jma_live_frame_is_complete(
    frame: pd.DataFrame, required_value_columns: tuple[str, ...]
) -> bool:
    """Validate the point-in-time-safe JMA fields needed for live inference."""
    hours = set(
        pd.to_numeric(frame.get("forecast_hour_local"), errors="coerce")
        .dropna()
        .astype(int)
    )
    return (
        len(frame) == 13
        and hours == set(range(11, 24))
        and frame.get("lineage", pd.Series(dtype=str))
        .astype(str)
        .eq("jma_msm_previous_day1")
        .all()
        and frame.get("availability_basis", pd.Series(dtype=str))
        .astype(str)
        .eq("open_meteo_previous_day1_variable")
        .all()
        and frame.get("fetch_status", pd.Series(dtype=str)).astype(str).eq("ok").all()
        and _finite_columns(frame, required_value_columns)
    )


IEM_ASOS_METAR_PROVIDER = "iem_asos_global_metar"
IEM_ASOS_METAR_LIVE_DATA_SOURCE = f"{IEM_ASOS_METAR_PROVIDER}_live"
IEM_REQUIRED_LIVE_OBSERVATION_FIELDS = (
    "observed_temp_at_as_of_f",
    "observed_high_temp_through_as_of_f",
    "observed_humidity_at_as_of",
    "observed_precip_recent_at_as_of",
    "observed_visibility_at_as_of",
)


def _validate_live_iem_observation(
    observation: pd.DataFrame,
    *,
    station_id: str,
    cutoff_utc: datetime,
) -> tuple[pd.Series, pd.Timestamp]:
    """Validate the same IEM RJTT METAR population used by Tokyo training."""
    if len(observation) != 1:
        raise ValueError("metar_row_count")
    observed = observation.iloc[0]
    if str(observed.get("station_id") or "").upper() != station_id:
        raise ValueError("metar_station_mismatch")
    if str(observed.get("observed_source") or "") != IEM_ASOS_METAR_PROVIDER:
        raise ValueError("metar_source_mismatch")
    if str(observed.get("observed_data_source") or "") != IEM_ASOS_METAR_LIVE_DATA_SOURCE:
        raise ValueError("metar_data_source_mismatch")
    if str(observed.get("observed_fetch_status") or "") != "ok":
        raise ValueError("metar_unavailable")
    observed_at = pd.to_datetime(observed.get("observed_as_of_time_local"), utc=True)
    if pd.isna(observed_at):
        raise ValueError("metar_observation_timestamp_invalid")
    age_minutes = (pd.Timestamp(cutoff_utc) - observed_at).total_seconds() / 60.0
    if observed_at > pd.Timestamp(cutoff_utc):
        raise ValueError("metar_post_cutoff")
    if age_minutes < 0 or age_minutes > 60:
        raise ValueError("metar_too_old")
    if not _finite_columns(observation, IEM_REQUIRED_LIVE_OBSERVATION_FIELDS):
        raise ValueError("metar_required_value_missing")
    if not str(observed.get("observed_weather_code_at_as_of") or "").strip():
        raise ValueError("metar_weather_code_missing")
    if not str(observed.get("source_uri") or "") or not str(observed.get("source_checksum") or "").lower().strip():
        raise ValueError("source_checksum_missing")
    source_checksum = str(observed.get("source_checksum") or "").strip()
    if len(source_checksum) != 64 or any(
        character not in "0123456789abcdef" for character in source_checksum.lower()
    ):
        raise ValueError("source_checksum_missing")
    return observed, observed_at


def build_asia_live_feature_row(
    data_root: str | Path,
    city_id: str,
    contract_date: date,
    *,
    generated_at: datetime,
    feature_version: str = V20_ASIA_NO_PEAK_FEATURE_VERSION,
    providers: tuple[str, ...] = ASIA_PROVIDERS,
) -> pd.DataFrame:
    """Build one target-free live row and retain its alignment proof in attrs."""
    from ..asia_11am import (
        GEFS_MEMBERS,
        GEFS_TEMP_FORECAST_HOURS,
        GEFS_TMAX_FORECAST_HOURS,
        GFS_FIELDS,
        GFS_FORECAST_HOURS,
        JMA_REQUIRED_LIVE_FIELDS,
        _jma_output_column,
        forecast_timing,
        summarize_gefs_members,
    )
    from ..direct_nwp_fetch import _available_direct_nwp_fxx_hours

    city = str(city_id).strip().lower()
    if city not in CITY_METADATA:
        raise ValueError(f"Unknown Asia city: {city_id!r}")
    metadata = CITY_METADATA[city]
    station_id = str(metadata["station_id"])
    timezone = str(metadata["timezone"])
    selected = tuple(providers)
    if set(selected) != set(ASIA_PROVIDERS):
        raise ValueError("live_provider_contract_mismatch")
    root = Path(data_root)
    current = generated_at.astimezone(UTC)
    cutoff_local = datetime.combine(contract_date, time(11), tzinfo=ZoneInfo(timezone))
    cutoff_utc = cutoff_local.astimezone(UTC)
    not_before_utc = cutoff_utc + timedelta(minutes=10)
    if current < not_before_utc:
        raise ValueError("collection_before_not_before")
    timing = forecast_timing(contract_date)

    observation_path = _latest_live_part(
        root / "normalized" / "live" / "observations" / city, contract_date
    )
    observation = pd.read_parquet(observation_path)
    observed, observed_at = _validate_live_iem_observation(
        observation,
        station_id=station_id,
        cutoff_utc=cutoff_utc,
    )

    gfs_path = (
        root / "normalized" / "forecasts" / "gfs" / city
        / contract_date.strftime("%Y-%m") / f"{contract_date}.parquet"
    )
    gfs = pd.read_parquet(gfs_path)
    expected_issue = pd.Timestamp(timing["issue_utc"])
    expected_as_of = pd.Timestamp(timing["as_of_utc"])
    gfs_issue = pd.to_datetime(gfs.get("issued_at_utc"), errors="coerce", utc=True)
    gfs_as_of = pd.to_datetime(gfs.get("forecast_as_of_utc"), errors="coerce", utc=True)
    if (
        len(gfs) != len(GFS_FORECAST_HOURS)
        or set(pd.to_numeric(gfs.get("forecast_hour"), errors="coerce").dropna().astype(int))
        != set(GFS_FORECAST_HOURS)
        or not gfs_issue.eq(expected_issue).all()
        or not gfs_as_of.eq(expected_as_of).all()
        or not gfs.get("fetch_status", pd.Series(dtype=str)).astype(str).eq("ok").all()
    ):
        raise ValueError("gfs_wrong_cycle_or_incomplete_hours")
    gfs_required = tuple(
        field.replace("_k_", "_c_") if "_k_" in field else field for field in GFS_FIELDS
    )
    if not _finite_columns(gfs, gfs_required):
        raise ValueError("gfs_required_value_missing")

    gefs_path = (
        root / "normalized" / "forecasts" / "gefs" / city
        / contract_date.strftime("%Y-%m") / f"{contract_date}.parquet"
    )
    gefs = pd.read_parquet(gefs_path)
    gefs_issue = pd.to_datetime(gefs.get("issued_at_utc"), errors="coerce", utc=True)
    member_hours = set(
        zip(
            gefs.get("member_id", pd.Series(dtype=str)).astype(str),
            pd.to_numeric(gefs.get("forecast_hour"), errors="coerce").fillna(-1).astype(int),
        )
    )
    expected_member_hours = set(
        (member, hour) for member in GEFS_MEMBERS for hour in GEFS_TEMP_FORECAST_HOURS
    )
    if (
        len(gefs) != len(expected_member_hours)
        or member_hours != expected_member_hours
        or not gefs_issue.eq(expected_issue).all()
        or not gefs.get("fetch_status", pd.Series(dtype=str)).astype(str).eq("ok").all()
        or not _finite_columns(gefs, ("temp_2m_c",))
    ):
        raise ValueError("gefs_wrong_cycle_or_missing_member")
    required_tmax = gefs.loc[
        pd.to_numeric(gefs["forecast_hour"], errors="coerce").isin(GEFS_TMAX_FORECAST_HOURS)
    ]
    if len(required_tmax) != len(GEFS_MEMBERS) * len(GEFS_TMAX_FORECAST_HOURS) or not _finite_columns(
        required_tmax, ("tmax_3h_c",)
    ):
        raise ValueError("gefs_incomplete_tmax_intervals")
    _members, gefs_daily = summarize_gefs_members(gefs)
    if len(gefs_daily) != 1 or int(gefs_daily.iloc[0]["gefs_member_count"]) != len(GEFS_MEMBERS):
        raise ValueError("gefs_missing_member")

    jma_path = _latest_live_part(
        root / "normalized" / "live" / "jma_msm_previous_day1" / city,
        contract_date,
    )
    jma = pd.read_parquet(jma_path)
    jma_required_columns = tuple(
        _jma_output_column(field) for field in JMA_REQUIRED_LIVE_FIELDS
    )
    if not _jma_live_frame_is_complete(jma, jma_required_columns):
        raise ValueError("jma_wrong_lineage_or_incomplete_hours")
    if not jma.get("source_url", pd.Series(dtype=str)).astype(str).str.len().gt(0).all() or not jma.get(
        "source_checksum", pd.Series(dtype=str)
    ).astype(str).str.fullmatch(r"[0-9a-fA-F]{64}").all():
        raise ValueError("source_checksum_missing")

    wide = observation.copy()
    wide["station_name"] = metadata["station_name"]
    wide["airport_name"] = metadata["station_name"]
    wide["city_label"] = metadata["city_label"]
    wide["timezone"] = timezone
    wide["country"] = metadata["country"]
    wide["lat"] = metadata["lat"]
    wide["lon"] = metadata["lon"]
    wide["feature_version"] = feature_version
    wide["timing_mode"] = ASIA_TIMING_MODE
    provider_frames = {
        "gfs": _provider_summary(
            gfs,
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
        ),
        "gefs": _gefs_summary(gefs_daily),
        "jma_msm": _provider_summary(
            jma,
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
        ),
    }
    for provider in selected:
        wide = wide.merge(provider_frames[provider], on="contract_date", how="left")
    wide = _add_calendar_features(wide)
    wide = _add_current_observation_derived_features(wide)
    wide = _add_provider_availability_features(wide, selected)
    wide = _add_provider_time_features(wide, selected, timezone)
    wide = _add_ensemble_features(wide, selected)
    wide = _add_forecast_shape_features(wide, selected)
    wide = _add_provider_cross_model_features(wide, selected)
    wide = _add_observation_forecast_delta_features(wide, selected)

    issue_key = timing["issue_utc"].strftime("%Y%m%d%H")
    gfs_source_hours = _available_direct_nwp_fxx_hours(
        "gfs", timing["issue_utc"], list(GFS_FORECAST_HOURS)
    )
    gfs_raw = [
        root
        / "raw"
        / "nwp_subsets"
        / "gfs"
        / f"gfs_{field}_{issue_key}_f{hour:03d}.grib2"
        for hour in gfs_source_hours
        for field in GFS_FIELDS
    ]
    gefs_raw = [
        root
        / "raw"
        / "nwp_subsets"
        / "gefs"
        / f"gefs_{member}_temp_2m_c_{issue_key}_f{hour:03d}.grib2"
        for member in GEFS_MEMBERS
        for hour in GEFS_TEMP_FORECAST_HOURS
    ] + [
        root
        / "raw"
        / "nwp_subsets"
        / "gefs"
        / f"gefs_{member}_tmax_3h_c_{issue_key}_f{hour:03d}.grib2"
        for member in GEFS_MEMBERS
        for hour in GEFS_TMAX_FORECAST_HOURS
    ]
    if not all(path.is_file() for path in [*gfs_raw, *gefs_raw]):
        raise ValueError("source_checksum_missing")
    wide.attrs["alignment"] = {
        "alignmentStatus": "aligned",
        "stationId": station_id,
        "contractDate": contract_date.isoformat(),
        "timezone": timezone,
        "featureCutoffLocal": cutoff_local.isoformat(),
        "featureCutoffUtc": cutoff_utc.isoformat().replace("+00:00", "Z"),
        "collectionNotBeforeUtc": not_before_utc.isoformat().replace("+00:00", "Z"),
        "gfsCycleUtc": timing["issue_utc"].isoformat().replace("+00:00", "Z"),
        "gefsCycleUtc": timing["issue_utc"].isoformat().replace("+00:00", "Z"),
        "jmaLineage": "jma_msm_previous_day1",
        "jmaAvailabilityBasis": "open_meteo_previous_day1_variable",
        "metarObservedAtUtc": observed_at.isoformat().replace("+00:00", "Z"),
        "metarSource": IEM_ASOS_METAR_PROVIDER,
        "timingMode": ASIA_TIMING_MODE,
        "sources": {
            "gfs": {
                "retrievedAtUtc": current.isoformat().replace("+00:00", "Z"),
                "sourceUrls": sorted(set(gfs["source_url"].astype(str))),
                "sourceChecksum": _composite_sha256(gfs_raw),
            },
            "gefs": {
                "retrievedAtUtc": current.isoformat().replace("+00:00", "Z"),
                "sourceUrls": sorted(set(gefs["source_url"].astype(str))),
                "sourceChecksum": _composite_sha256(gefs_raw),
            },
            "jma_msm": {
                "retrievedAtUtc": str(jma.iloc[0]["retrieved_at_utc"]),
                "sourceUrls": sorted(set(jma["source_url"].astype(str))),
                "sourceChecksum": str(jma.iloc[0]["source_checksum"]),
            },
            "metar": {
                "retrievedAtUtc": current.isoformat().replace("+00:00", "Z"),
                "sourceUrls": [str(observed["source_uri"])],
                "sourceChecksum": str(observed["source_checksum"]),
            },
        },
    }
    return wide


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
