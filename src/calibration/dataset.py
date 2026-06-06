from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .sdk_pipeline import DIRECT_NBM_FILE, SDK_ACTUALS_FILE, SDK_NWP_FILE, SDK_STATION_REGISTRY_FILE, default_sdk_cache_dir
from .time_rules import forecast_as_of_utc


DEFAULT_SDK_PROVIDERS = ("hrrr", "gfs")
DEFAULT_SDK_START_DATE = "2021-01-01"

CURRENT_OBSERVATION_FILE = "sdk_current_observations_11am.csv"
CURRENT_OBSERVATION_CACHE_PATTERN = "sdk_current_obs_*/sdk_current_observations_11am.csv"

SAFE_FORECAST_NUMERIC_COLUMNS = [
    "dewpoint_mean_f",
    "humidity_mean",
    "wind_speed_mean",
    "wind_speed_max",
]

SAFE_OBSERVED_NUMERIC_COLUMNS = [
    "observed_temp_at_as_of_f",
    "observed_dewpoint_at_as_of_f",
    "observed_humidity_at_as_of",
    "observed_wind_speed_at_as_of",
    "observed_pressure_at_as_of",
    "observed_visibility_at_as_of",
    "observed_as_of_age_minutes",
]

CALIBRATION_COLUMNS = [
    "station_id",
    "station_name",
    "airport_name",
    "provider",
    "model",
    "source_label",
    "timing_mode",
    "contract_date",
    "forecast_as_of",
    "issued_at",
    "horizon_hours",
    "raw_forecast_high_f",
    "actual_high_f",
    "calibration_bias_f",
    "absolute_error_before",
    "squared_error_before",
    "month",
    "day_of_year",
    "day_of_year_sin",
    "day_of_year_cos",
    "day_of_week",
    "forecast_issue_hour_utc",
    "forecast_cycle_hour",
    "forecast_lead_hours",
    "dewpoint_mean_f",
    "humidity_mean",
    "wind_speed_mean",
    "wind_speed_max",
    *SAFE_OBSERVED_NUMERIC_COLUMNS,
    "provider_disagreement_f",
    "rain_regime",
    "cloud_regime",
    "rolling_provider_bias_7d",
    "rolling_provider_bias_30d",
    "rolling_provider_bias_90d",
    "expanding_provider_bias",
    "past_provider_sample_count",
    "station_month_past_actual_mean_f",
    "station_month_past_actual_std_f",
    "raw_forecast_station_month_z",
    "extreme_heat_flag",
    "extreme_cold_flag",
    "data_source",
    "source_file_or_url",
]

WEATHER_NUMERIC_COLUMNS = [
    "dewpoint_mean_f",
    "humidity_mean",
    "wind_speed_mean",
    "wind_speed_max",
]


def build_calibration_samples(
    project_root: str | Path = ".",
    calibration_dir: str | Path | None = None,
    include_providers: Iterable[str] | None = None,
    include_timing_modes: Iterable[str] | None = None,
    source_mode: str = "legacy",
    sdk_cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    root = Path(project_root)
    out_dir = Path(calibration_dir) if calibration_dir is not None else root / "data" / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    providers = {p.lower() for p in include_providers} if include_providers else None
    timing_modes = {p.lower() for p in include_timing_modes} if include_timing_modes else None
    source_mode = source_mode.lower()

    if source_mode == "sdk":
        sdk_dir = Path(sdk_cache_dir) if sdk_cache_dir is not None else default_sdk_cache_dir(out_dir)
        if providers is None:
            providers = set(DEFAULT_SDK_PROVIDERS)
        actuals = _load_sdk_actuals(sdk_dir)
        station_meta = _load_sdk_station_meta(root, sdk_dir)
        forecast_frames = [_load_sdk_nwp_cache(sdk_dir, station_meta)]
        current_observations = _load_current_observations(sdk_dir)
        if _include_direct_nbm_cache():
            forecast_frames.append(_load_direct_nbm_cache(sdk_dir, station_meta))
    elif source_mode == "legacy":
        actuals = _load_actuals(root)
        station_meta = _load_station_meta(root)
        current_observations = pd.DataFrame()
        forecast_frames = [
            _load_local_nbm(root, station_meta),
            _load_mostlyright_nwp_cache(out_dir, station_meta),
            _load_exact_weathergov_nws(root, out_dir, station_meta),
        ]
    else:
        raise ValueError("source_mode must be 'legacy' or 'sdk'")
    forecasts = pd.concat([frame for frame in forecast_frames if not frame.empty], ignore_index=True)
    if providers is not None and not forecasts.empty:
        forecasts = forecasts.loc[forecasts["provider"].astype(str).str.lower().isin(providers)].copy()
    if source_mode == "sdk" and not forecasts.empty:
        forecasts = forecasts.loc[forecasts["contract_date"].astype(str).str[:10] >= DEFAULT_SDK_START_DATE].copy()
    if timing_modes is not None and not forecasts.empty:
        if "timing_mode" not in forecasts.columns:
            forecasts["timing_mode"] = "strict_6am"
        forecasts["timing_mode"] = forecasts["timing_mode"].fillna("strict_6am")
        forecasts = forecasts.loc[forecasts["timing_mode"].astype(str).str.lower().isin(timing_modes)].copy()
    if forecasts.empty:
        samples = pd.DataFrame(columns=CALIBRATION_COLUMNS)
        samples.to_csv(out_dir / "calibration_samples.csv", index=False)
        return samples

    forecasts = forecasts.dropna(subset=["station_id", "contract_date", "raw_forecast_high_f"]).copy()
    forecasts["contract_date"] = forecasts["contract_date"].astype(str).str[:10]
    forecasts["provider"] = forecasts["provider"].astype(str).str.lower()
    forecasts["model"] = forecasts["model"].astype(str).str.lower()
    _assert_provider_lineage(forecasts)
    forecasts = _prefer_sdk_nbm_over_direct(forecasts)

    joined = forecasts.merge(
        actuals[["station_id", "contract_date", "actual_high_f"]],
        on=["station_id", "contract_date"],
        how="inner",
    )
    if not current_observations.empty:
        joined = joined.merge(current_observations, on=["station_id", "contract_date"], how="left")
    joined = joined.dropna(subset=["actual_high_f", "raw_forecast_high_f"]).copy()
    joined["calibration_bias_f"] = joined["actual_high_f"] - joined["raw_forecast_high_f"]
    joined["absolute_error_before"] = joined["calibration_bias_f"].abs()
    joined["squared_error_before"] = joined["calibration_bias_f"] ** 2
    joined = _add_time_features(joined)
    joined = _add_provider_disagreement(joined)
    joined = _add_regime_features(joined)
    joined = _add_past_bias_features(joined)
    joined = _add_station_month_climatology_features(joined)
    joined = joined.sort_values(["contract_date", "station_id", "provider", "model"]).reset_index(drop=True)

    for column in CALIBRATION_COLUMNS:
        if column not in joined:
            joined[column] = pd.NA
    samples = joined[CALIBRATION_COLUMNS]
    samples.to_csv(out_dir / "calibration_samples.csv", index=False)
    return samples


def _load_actuals(root: Path) -> pd.DataFrame:
    path = root / "data" / "processed" / "actual_highs.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing actual highs: {path}")
    actuals = pd.read_csv(path)
    required = {"station_code", "date_local", "actual_high_f"}
    missing = required - set(actuals.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    out = actuals.rename(columns={"station_code": "station_id", "date_local": "contract_date"}).copy()
    out["station_id"] = out["station_id"].astype(str).str.upper()
    out["contract_date"] = out["contract_date"].astype(str).str[:10]
    out["actual_high_f"] = pd.to_numeric(out["actual_high_f"], errors="coerce")
    return out


def _load_sdk_actuals(sdk_dir: Path) -> pd.DataFrame:
    path = _sdk_metadata_path(sdk_dir, SDK_ACTUALS_FILE)
    if not path.exists():
        raise FileNotFoundError(f"Missing SDK actual highs: {path}")
    actuals = pd.read_csv(path)
    required = {"station_id", "contract_date", "actual_high_f"}
    missing = required - set(actuals.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    out = actuals.copy()
    if "fetch_status" in out.columns:
        out = out.loc[out["fetch_status"].astype(str).str.lower() == "ok"].copy()
    out["station_id"] = out["station_id"].astype(str).str.upper()
    out["contract_date"] = out["contract_date"].astype(str).str[:10]
    out["actual_high_f"] = pd.to_numeric(out["actual_high_f"], errors="coerce")
    return out.dropna(subset=["actual_high_f"])


def _load_current_observations(sdk_dir: Path) -> pd.DataFrame:
    paths = _cache_paths(sdk_dir, CURRENT_OBSERVATION_FILE, CURRENT_OBSERVATION_CACHE_PATTERN)
    if not paths:
        return pd.DataFrame(columns=["station_id", "contract_date", *SAFE_OBSERVED_NUMERIC_COLUMNS])
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path, low_memory=False)
        if frame.empty:
            continue
        required = ["station_id", "contract_date", "timing_mode", "observed_fetch_status", *SAFE_OBSERVED_NUMERIC_COLUMNS]
        for column in required:
            if column not in frame:
                frame[column] = pd.NA
        frame = frame[required].copy()
        frame["source_cache_dir"] = path.parent.name
        frame["source_cache_mtime"] = path.stat().st_mtime
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["station_id", "contract_date", *SAFE_OBSERVED_NUMERIC_COLUMNS])
    out = pd.concat(frames, ignore_index=True)
    out["station_id"] = out["station_id"].astype(str).str.upper()
    out["contract_date"] = out["contract_date"].astype(str).str[:10]
    out["timing_mode"] = out["timing_mode"].astype(str).str.lower()
    out["observed_fetch_status"] = out["observed_fetch_status"].astype(str).str.lower()
    out = out.loc[
        out["timing_mode"].eq("same_day_11am")
        & out["observed_fetch_status"].eq("ok")
    ].copy()
    if out.empty:
        return pd.DataFrame(columns=["station_id", "contract_date", *SAFE_OBSERVED_NUMERIC_COLUMNS])
    for column in SAFE_OBSERVED_NUMERIC_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.sort_values(
        ["station_id", "contract_date", "source_cache_mtime", "source_cache_dir"],
        ascending=[True, True, False, True],
    )
    out = out.drop_duplicates(["station_id", "contract_date"], keep="first")
    return out[["station_id", "contract_date", *SAFE_OBSERVED_NUMERIC_COLUMNS]].reset_index(drop=True)


def _load_station_meta(root: Path) -> pd.DataFrame:
    path = root / "data" / "processed" / "station_registry.csv"
    if not path.exists():
        return pd.DataFrame(columns=["station_id", "station_name", "airport_name", "timezone"])
    frame = pd.read_csv(path)
    frame = frame.rename(columns={"station_code": "station_id"}).copy()
    frame["station_id"] = frame["station_id"].astype(str).str.upper()
    return frame


def _load_sdk_station_meta(root: Path, sdk_dir: Path) -> pd.DataFrame:
    path = _sdk_metadata_path(sdk_dir, SDK_STATION_REGISTRY_FILE)
    if path.exists():
        frame = pd.read_csv(path)
    else:
        frame = _load_station_meta(root)
    if "station_code" in frame.columns and "station_id" not in frame.columns:
        frame = frame.rename(columns={"station_code": "station_id"})
    required = {"station_id", "timezone"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    frame = frame.copy()
    frame["station_id"] = frame["station_id"].astype(str).str.upper()
    return frame


def _sdk_metadata_path(sdk_dir: Path, filename: str) -> Path:
    direct = sdk_dir / filename
    if direct.exists():
        return direct
    nested = sdk_dir / "sdk" / filename
    if nested.exists():
        return nested
    return direct


def _load_local_nbm(root: Path, station_meta: pd.DataFrame) -> pd.DataFrame:
    path = root / "data" / "processed" / "nws_forecast_snapshots.csv"
    if not path.exists():
        return _empty_forecasts()
    frame = pd.read_csv(path)
    if frame.empty:
        return _empty_forecasts()
    required = {"station_code", "target_date_local", "forecast_horizon_hours", "forecast_high_f"}
    if not required.issubset(frame.columns):
        return _empty_forecasts()
    out = frame.loc[pd.to_numeric(frame["forecast_horizon_hours"], errors="coerce") == 0].copy()
    out = out.dropna(subset=["forecast_high_f"])
    if out.empty:
        return _empty_forecasts()
    out = out.rename(
        columns={
            "station_code": "station_id",
            "target_date_local": "contract_date",
            "issue_time_utc": "issued_at",
            "forecast_high_f": "raw_forecast_high_f",
        }
    )
    out["station_id"] = out["station_id"].astype(str).str.upper()
    out["contract_date"] = out["contract_date"].astype(str).str[:10]
    out["provider"] = "nbm"
    out["model"] = "nbm_tmax"
    out["source_label"] = "noaa_nbm_archive_tmax"
    out["horizon_hours"] = 0
    out["data_source"] = "local_processed_nbm_from_legacy_nws_file"
    out = _apply_forecast_as_of(out, station_meta)
    return _select_forecast_columns(out)


def _load_mostlyright_nwp_cache(calibration_dir: Path, station_meta: pd.DataFrame) -> pd.DataFrame:
    paths = [
        calibration_dir / "mostlyright_nwp_0h_cache.csv",
        calibration_dir / "mostlyright_nwp_daily_0h.csv",
    ]
    frames = [pd.read_csv(path) for path in paths if path.exists()]
    if not frames:
        return _empty_forecasts()
    frame = pd.concat(frames, ignore_index=True)
    if frame.empty:
        return _empty_forecasts()
    rename = {
        "station": "station_id",
        "target_date_local": "contract_date",
        "forecast_high_f": "raw_forecast_high_f",
        "issue_time_utc": "issued_at",
    }
    frame = frame.rename(columns={k: v for k, v in rename.items() if k in frame.columns}).copy()
    required = {"station_id", "provider", "model", "contract_date", "raw_forecast_high_f"}
    if not required.issubset(frame.columns):
        return _empty_forecasts()
    out = frame.dropna(subset=["raw_forecast_high_f"]).copy()
    out["station_id"] = out["station_id"].astype(str).str.upper()
    out["contract_date"] = out["contract_date"].astype(str).str[:10]
    out["provider"] = out["provider"].astype(str).str.lower()
    out["model"] = out["model"].astype(str).str.lower()
    out["horizon_hours"] = 0
    out["source_label"] = out.get("source_label", "mostlyright_nwp")
    out["data_source"] = out.get("data_source", "mostlyright_weather_forecast_nwp")
    out = _apply_forecast_as_of(out, station_meta)
    return _select_forecast_columns(out)


def _load_sdk_nwp_cache(sdk_dir: Path, station_meta: pd.DataFrame) -> pd.DataFrame:
    paths = _cache_paths(sdk_dir, SDK_NWP_FILE, "sdk_11am_*/sdk_nwp_0h_cache.csv")
    if not paths:
        return _empty_forecasts()
    frame = pd.concat([pd.read_csv(path, low_memory=False).assign(source_cache_dir=path.parent.name) for path in paths], ignore_index=True)
    frame = _drop_probe_cache_rows(frame)
    if frame.empty:
        return _empty_forecasts()
    required = {"station_id", "provider", "model", "contract_date", "raw_forecast_high_f"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{paths[0]} missing required columns: {sorted(missing)}")
    out = frame.copy()
    if "fetch_status" in out.columns:
        out = out.loc[out["fetch_status"].astype(str).str.lower() == "ok"].copy()
    out = out.dropna(subset=["raw_forecast_high_f"])
    if out.empty:
        return _empty_forecasts()
    out["station_id"] = out["station_id"].astype(str).str.upper()
    out["contract_date"] = out["contract_date"].astype(str).str[:10]
    out["provider"] = out["provider"].astype(str).str.lower()
    out["model"] = out["model"].astype(str).str.lower()
    if (out["provider"] == "nws").any():
        raise ValueError("SDK-only calibration input must not contain provider='nws' rows")
    if "horizon_hours" not in out.columns:
        out["horizon_hours"] = 0
    out["horizon_hours"] = pd.to_numeric(out["horizon_hours"], errors="coerce").fillna(0).astype(int)
    out["source_label"] = out.get("source_label", "mostlyright_forecast_nwp")
    out["data_source"] = out.get("data_source", "mostlyright.weather.forecast_nwp")
    out = _apply_forecast_as_of(out, station_meta)
    return _select_forecast_columns(out)


def _load_direct_nbm_cache(sdk_dir: Path, station_meta: pd.DataFrame) -> pd.DataFrame:
    paths = _cache_paths(sdk_dir, DIRECT_NBM_FILE, "direct_nbm_*/direct_nbm_0h_cache.csv")
    if not paths:
        return _empty_forecasts()
    frame = pd.concat([pd.read_csv(path, low_memory=False).assign(source_cache_dir=path.parent.name) for path in paths], ignore_index=True)
    frame = _drop_probe_cache_rows(frame)
    if frame.empty:
        return _empty_forecasts()
    required = {"station_id", "provider", "model", "contract_date", "raw_forecast_high_f"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{paths[0]} missing required columns: {sorted(missing)}")
    out = frame.copy()
    if "fetch_status" in out.columns:
        out = out.loc[out["fetch_status"].astype(str).str.lower() == "ok"].copy()
    out = out.dropna(subset=["raw_forecast_high_f"])
    if out.empty:
        return _empty_forecasts()
    out["station_id"] = out["station_id"].astype(str).str.upper()
    out["contract_date"] = out["contract_date"].astype(str).str[:10]
    out["provider"] = "nbm"
    out["model"] = out["model"].astype(str).str.lower()
    if "horizon_hours" not in out.columns:
        out["horizon_hours"] = 0
    out["horizon_hours"] = pd.to_numeric(out["horizon_hours"], errors="coerce").fillna(0).astype(int)
    out["source_label"] = out.get("source_label", "noaa_nbm_archive_tmp")
    out["data_source"] = out.get("data_source", "direct_noaa_nbm_archive_grib2")
    out = _apply_forecast_as_of(out, station_meta)
    return _select_forecast_columns(out)


def _cache_paths(sdk_dir: Path, filename: str, shard_pattern: str) -> list[Path]:
    paths: set[Path] = set()
    direct = sdk_dir / filename
    if direct.exists():
        paths.add(direct)
    paths.update(path for path in sdk_dir.glob(shard_pattern) if path.exists())
    return sorted(paths)


def _drop_probe_cache_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if "source_cache_dir" not in frame.columns:
        return frame
    source = frame["source_cache_dir"].astype(str).str.lower()
    probe_markers = ("smoke", "retry", "gated", "fixed", "overlap_probe")
    return frame.loc[~source.str.contains("|".join(probe_markers), regex=True)].copy()


def _include_direct_nbm_cache() -> bool:
    value = os.getenv("WEATHER_RESEARCH_INCLUDE_DIRECT_NBM", "")
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _prefer_sdk_nbm_over_direct(forecasts: pd.DataFrame) -> pd.DataFrame:
    if forecasts.empty:
        return forecasts
    out = forecasts.copy()
    if "source_cache_dir" not in out.columns:
        out["source_cache_dir"] = ""
    if "timing_mode" not in out.columns:
        out["timing_mode"] = "strict_6am"
    data_source = out.get("data_source", pd.Series("", index=out.index)).astype(str).str.lower()
    source_cache = out["source_cache_dir"].astype(str).str.lower()
    is_nbm = out["provider"].astype(str).str.lower().eq("nbm")
    is_direct = is_nbm & (data_source.str.contains("direct_noaa") | source_cache.str.contains("direct_nbm"))
    out["_source_rank"] = 1
    out.loc[is_direct, "_source_rank"] = 2
    out.loc[is_nbm & ~is_direct, "_source_rank"] = 0
    out = out.sort_values(
        ["station_id", "provider", "model", "timing_mode", "contract_date", "_source_rank"],
        ascending=[True, True, True, True, True, True],
    )
    out = out.drop_duplicates(["station_id", "provider", "model", "timing_mode", "contract_date"], keep="first")
    return out.drop(columns=["_source_rank"]).reset_index(drop=True)


def _load_exact_weathergov_nws(root: Path, calibration_dir: Path, station_meta: pd.DataFrame) -> pd.DataFrame:
    candidate_paths = [
        calibration_dir / "weathergov_nws_forecast_snapshots.csv",
        calibration_dir / "nws_weathergov_forecast_snapshots.csv",
        root / "data" / "processed" / "weathergov_forecast_snapshots.csv",
    ]
    frames = [pd.read_csv(path) for path in candidate_paths if path.exists()]
    if not frames:
        return _empty_forecasts()
    frame = pd.concat(frames, ignore_index=True)
    rename = {
        "station_code": "station_id",
        "target_date_local": "contract_date",
        "forecast_high_f": "raw_forecast_high_f",
        "issue_time_utc": "issued_at",
    }
    frame = frame.rename(columns={k: v for k, v in rename.items() if k in frame.columns}).copy()
    required = {"station_id", "contract_date", "raw_forecast_high_f", "issued_at"}
    if not required.issubset(frame.columns):
        return _empty_forecasts()
    out = frame.dropna(subset=["raw_forecast_high_f"]).copy()
    out["station_id"] = out["station_id"].astype(str).str.upper()
    out["contract_date"] = out["contract_date"].astype(str).str[:10]
    out["provider"] = "nws"
    out["model"] = out.get("model", "weather_gov_grid")
    out["source_label"] = "weather_gov_captured_forecast"
    out["horizon_hours"] = 0
    out["data_source"] = "exact_weather_gov_local_capture"
    out = _apply_forecast_as_of(out, station_meta)
    return _select_forecast_columns(out)


def _apply_forecast_as_of(frame: pd.DataFrame, station_meta: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "timezone" not in out.columns:
        out = out.merge(station_meta[["station_id", "timezone"]].drop_duplicates(), on="station_id", how="left")
    computed = [
        forecast_as_of_utc(contract_date, timezone).isoformat()
        for contract_date, timezone in zip(out["contract_date"], out["timezone"], strict=False)
    ]
    if "forecast_as_of" not in out.columns:
        out["forecast_as_of"] = computed
    else:
        out["forecast_as_of"] = out["forecast_as_of"].fillna(pd.Series(computed, index=out.index))
        out.loc[out["forecast_as_of"].astype(str).str.strip() == "", "forecast_as_of"] = pd.Series(computed, index=out.index)
    return out


def _select_forecast_columns(frame: pd.DataFrame) -> pd.DataFrame:
    for column in [
        "station_id",
        "station_name",
        "airport_name",
        "provider",
        "model",
        "source_label",
        "contract_date",
        "forecast_as_of",
        "issued_at",
        "horizon_hours",
        "raw_forecast_high_f",
        "data_source",
        "source_file_or_url",
        *WEATHER_NUMERIC_COLUMNS,
        "timing_mode",
    ]:
        if column not in frame:
            frame[column] = pd.NA
    frame["timing_mode"] = frame["timing_mode"].fillna("strict_6am")
    for column in ["raw_forecast_high_f", *WEATHER_NUMERIC_COLUMNS]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame[
        [
            "station_id",
            "station_name",
            "airport_name",
            "provider",
            "model",
            "source_label",
            "timing_mode",
            "contract_date",
            "forecast_as_of",
            "issued_at",
            "horizon_hours",
            "raw_forecast_high_f",
            "data_source",
            "source_file_or_url",
            *WEATHER_NUMERIC_COLUMNS,
        ]
    ]


def _empty_forecasts() -> pd.DataFrame:
    return pd.DataFrame(columns=["station_id", "provider", "model", "timing_mode", "contract_date", "raw_forecast_high_f"])


def _assert_provider_lineage(forecasts: pd.DataFrame) -> None:
    bad_nws = forecasts.loc[
        (forecasts["provider"].astype(str).str.lower() == "nws")
        & forecasts["model"].astype(str).str.contains("nbm", case=False, na=False)
    ]
    if not bad_nws.empty:
        raise ValueError("NBM rows must use provider='nbm'; exact provider='nws' is reserved for weather.gov captures")


def _add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    dates = pd.to_datetime(out["contract_date"], errors="coerce")
    out["month"] = dates.dt.month
    out["day_of_year"] = dates.dt.dayofyear
    out["day_of_year_sin"] = np.sin(2 * math.pi * out["day_of_year"] / 366)
    out["day_of_year_cos"] = np.cos(2 * math.pi * out["day_of_year"] / 366)
    out["day_of_week"] = dates.dt.day_name()
    issued = pd.to_datetime(out["issued_at"], errors="coerce", utc=True)
    as_of = pd.to_datetime(out["forecast_as_of"], errors="coerce", utc=True)
    out["forecast_issue_hour_utc"] = issued.dt.hour
    out["forecast_cycle_hour"] = issued.dt.hour
    out["forecast_lead_hours"] = (as_of - issued).dt.total_seconds() / 3600
    return out


def _add_provider_disagreement(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    grouped = out.groupby(["station_id", "contract_date", "forecast_as_of"], dropna=False)["raw_forecast_high_f"]
    out["provider_disagreement_f"] = grouped.transform(lambda s: s.max(skipna=True) - s.min(skipna=True))
    return out


def _add_regime_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "precip_amount" in out.columns:
        precip = pd.to_numeric(out["precip_amount"], errors="coerce")
        out["rain_regime"] = np.select(
            [precip.fillna(0) <= 0.01, precip.fillna(0) <= 0.10, precip.fillna(0) <= 0.50],
            ["dry", "light", "moderate"],
            default="heavy",
        )
        out.loc[precip.isna(), "rain_regime"] = pd.NA
    else:
        out["rain_regime"] = pd.NA
    if "cloud_cover_mean" in out.columns:
        clouds = pd.to_numeric(out["cloud_cover_mean"], errors="coerce")
        out["cloud_regime"] = pd.cut(
            clouds,
            bins=[-0.1, 25, 50, 75, 100],
            labels=["clear", "partly_cloudy", "mostly_cloudy", "overcast"],
        ).astype("object")
        out.loc[clouds.isna(), "cloud_regime"] = pd.NA
    else:
        out["cloud_regime"] = pd.NA
    return out


def _add_past_bias_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.sort_values(["station_id", "provider", "model", "contract_date"]).copy()
    group_cols = ["station_id", "provider", "model"]
    grouped = out.groupby(group_cols, dropna=False)["calibration_bias_f"]
    out["rolling_provider_bias_7d"] = grouped.transform(lambda s: s.shift(1).rolling(7, min_periods=3).mean())
    out["rolling_provider_bias_30d"] = grouped.transform(lambda s: s.shift(1).rolling(30, min_periods=5).mean())
    out["rolling_provider_bias_90d"] = grouped.transform(lambda s: s.shift(1).rolling(90, min_periods=10).mean())
    out["expanding_provider_bias"] = grouped.transform(lambda s: s.shift(1).expanding(min_periods=5).mean())
    out["past_provider_sample_count"] = grouped.cumcount()
    return out


def _add_station_month_climatology_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.sort_values(["station_id", "month", "contract_date", "provider", "model"]).copy()
    daily_actuals = (
        out[["station_id", "month", "contract_date", "actual_high_f"]]
        .drop_duplicates(["station_id", "month", "contract_date"])
        .sort_values(["station_id", "month", "contract_date"])
        .copy()
    )
    grouped = daily_actuals.groupby(["station_id", "month"], dropna=False)["actual_high_f"]
    daily_actuals["station_month_past_actual_mean_f"] = grouped.transform(
        lambda s: s.shift(1).expanding(min_periods=5).mean()
    )
    daily_actuals["station_month_past_actual_std_f"] = grouped.transform(
        lambda s: s.shift(1).expanding(min_periods=10).std()
    )
    out = out.merge(
        daily_actuals[
            [
                "station_id",
                "month",
                "contract_date",
                "station_month_past_actual_mean_f",
                "station_month_past_actual_std_f",
            ]
        ],
        on=["station_id", "month", "contract_date"],
        how="left",
    )
    std = out["station_month_past_actual_std_f"].replace(0, np.nan)
    out["raw_forecast_station_month_z"] = (out["raw_forecast_high_f"] - out["station_month_past_actual_mean_f"]) / std
    out["extreme_heat_flag"] = out["raw_forecast_station_month_z"] >= 1.5
    out["extreme_cold_flag"] = out["raw_forecast_station_month_z"] <= -1.5
    out.loc[out["raw_forecast_station_month_z"].isna(), ["extreme_heat_flag", "extreme_cold_flag"]] = False
    return out
