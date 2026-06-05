from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


TARGET_STATIONS = ("KATL", "KAUS", "KORD", "KDAL", "KHOU", "KLAX", "KMIA", "KLGA", "KSEA")
TARGET_PROVIDERS = ("gfs", "hrrr", "nbm")
TARGET = "actual_high_f"

FORECAST_CACHE_PATTERNS = (
    ("sdk_nwp_0h_cache.csv", "sdk_11am_*/sdk_nwp_0h_cache.csv"),
    ("direct_nbm_0h_cache.csv", "direct_nbm_*/direct_nbm_0h_cache.csv"),
)

CURRENT_OBSERVATION_CACHE_PATTERN = "sdk_current_obs_*/sdk_current_observations_11am.csv"

FORECAST_COLUMNS = [
    "station_id",
    "station_name",
    "airport_name",
    "provider",
    "model",
    "source_label",
    "timing_mode",
    "cycle_selection_policy",
    "contract_date",
    "forecast_as_of",
    "issued_at",
    "forecast_window_start",
    "forecast_window_end",
    "horizon_hours",
    "raw_forecast_high_f",
    "forecast_hour_min",
    "forecast_hour_max",
    "grid_dist_km_mean",
    "precip_amount",
    "wind_speed_mean",
    "wind_speed_max",
    "wind_direction_mean",
    "wind_gust_max",
    "dewpoint_mean_f",
    "humidity_mean",
    "data_source",
    "source_file_or_url",
    "fetch_status",
    "unavailable_reason",
]

PROVIDER_NUMERIC_COLUMNS = [
    "horizon_hours",
    "raw_forecast_high_f",
    "forecast_hour_min",
    "forecast_hour_max",
    "grid_dist_km_mean",
    "precip_amount",
    "wind_speed_mean",
    "wind_speed_max",
    "wind_direction_mean",
    "wind_gust_max",
    "dewpoint_mean_f",
    "humidity_mean",
]

PROVIDER_TEXT_COLUMNS = [
    "model",
    "source_label",
    "cycle_selection_policy",
    "forecast_as_of",
    "issued_at",
    "forecast_window_start",
    "forecast_window_end",
    "data_source",
    "source_file_or_url",
    "source_cache_dir",
]

OBSERVED_NUMERIC_COLUMNS = [
    "observed_temp_at_as_of_f",
    "observed_dewpoint_at_as_of_f",
    "observed_humidity_at_as_of",
    "observed_wind_speed_at_as_of",
    "observed_wind_direction_at_as_of",
    "observed_wind_gust_at_as_of",
    "observed_peak_wind_gust_at_as_of",
    "observed_peak_wind_direction_at_as_of",
    "observed_pressure_at_as_of",
    "observed_altimeter_inhg_at_as_of",
    "observed_sea_level_pressure_mb_at_as_of",
    "observed_visibility_at_as_of",
    "observed_ceiling_at_as_of",
    "observed_cloud_cover_at_as_of",
    "observed_precip_recent_at_as_of",
    "observed_snow_depth_at_as_of",
    "observed_as_of_age_minutes",
]

OBSERVED_TEXT_COLUMNS = [
    "observed_pressure_source",
    "observed_weather_code_at_as_of",
    "observed_as_of_time_local",
    "observed_as_of_time_utc",
    "observed_source",
    "observed_observation_type",
    "observed_qc_field",
    "observed_raw_metar",
    "observed_data_source",
    "observed_fetch_status",
    "observed_unavailable_reason",
]

OBSERVED_CATEGORICAL_FEATURES = [
    "observed_pressure_source",
    "observed_weather_code_at_as_of",
    "observed_observation_type",
    "observed_fetch_status",
]

HIGH_COLUMNS = {provider: f"{provider}_high_f" for provider in TARGET_PROVIDERS}
BASELINE_METHODS = [
    "gfs_raw",
    "hrrr_raw",
    "nbm_raw",
    "provider_mean",
    "provider_median",
    "best_raw_provider",
]
BASE_MODEL_METHODS = ["xgboost", "lightgbm", "catboost"]
STACK_METHOD = "ridge_stack"


@dataclass(frozen=True)
class StationStackingConfig:
    station_id: str
    project_root: str | Path = "."
    timing_mode: str = "same_day_11am"
    providers: tuple[str, ...] = TARGET_PROVIDERS
    min_train_rows: int = 180
    refit_days: int = 30
    min_meta_train_rows: int = 60
    random_state: int = 42
    fast_mode: bool = False
    fast_max_validation_blocks: int = 3
    output_dir: str | Path | None = None

    def resolved_project_root(self) -> Path:
        return Path(self.project_root).resolve()

    def resolved_output_dir(self) -> Path:
        if self.output_dir is not None:
            return Path(self.output_dir).resolve()
        return self.resolved_project_root() / "data" / "calibration" / "station_stacking"

    @property
    def effective_min_train_rows(self) -> int:
        return min(self.min_train_rows, 30) if self.fast_mode else self.min_train_rows

    @property
    def effective_min_meta_train_rows(self) -> int:
        return min(self.min_meta_train_rows, 15) if self.fast_mode else self.min_meta_train_rows

    @property
    def effective_refit_days(self) -> int:
        return min(self.refit_days, 14) if self.fast_mode else self.refit_days


@dataclass
class StationStackingResult:
    station_id: str
    features: pd.DataFrame
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    feature_columns: pd.DataFrame
    output_paths: dict[str, Path]


def provider_availability(
    project_root: str | Path = ".",
    timing_mode: str = "same_day_11am",
    providers: tuple[str, ...] = TARGET_PROVIDERS,
) -> pd.DataFrame:
    forecasts = load_same_day_provider_forecasts(project_root, timing_mode=timing_mode, providers=providers)
    if forecasts.empty:
        return pd.DataFrame(
            columns=["station_id", "provider", "row_count", "first_contract_date", "last_contract_date"]
        )
    grouped = forecasts.groupby(["station_id", "provider"], dropna=False)["contract_date"].agg(
        row_count="count",
        first_contract_date="min",
        last_contract_date="max",
    )
    return grouped.reset_index().sort_values(["station_id", "provider"]).reset_index(drop=True)


def load_current_observation_features(
    project_root: str | Path = ".",
    station_id: str | None = None,
    timing_mode: str = "same_day_11am",
) -> pd.DataFrame:
    root = Path(project_root)
    calibration_dir = root / "data" / "calibration"
    cache_paths = sorted(calibration_dir.glob(CURRENT_OBSERVATION_CACHE_PATTERN))
    frames: list[pd.DataFrame] = []
    required = ["station_id", "contract_date", "timing_mode", *OBSERVED_NUMERIC_COLUMNS, *OBSERVED_TEXT_COLUMNS]
    for path in cache_paths:
        try:
            frame = pd.read_csv(path, low_memory=False)
        except pd.errors.EmptyDataError:
            continue
        if frame.empty:
            continue
        for column in required:
            if column not in frame:
                frame[column] = pd.NA
        frame = frame[required].copy()
        frame["source_cache_dir"] = path.parent.name
        frame["source_cache_mtime"] = path.stat().st_mtime
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["contract_date", *OBSERVED_NUMERIC_COLUMNS, *OBSERVED_TEXT_COLUMNS])

    out = pd.concat(frames, ignore_index=True)
    out["station_id"] = out["station_id"].astype("string").str.upper()
    out["timing_mode"] = out["timing_mode"].astype("string")
    out["contract_date"] = out["contract_date"].astype("string").str[:10]
    out["observed_fetch_status"] = out["observed_fetch_status"].astype("string").str.lower()
    if station_id is not None:
        out = out.loc[out["station_id"].eq(station_id.upper())].copy()
    out = out.loc[out["timing_mode"].eq(timing_mode)].copy()
    if out.empty:
        return pd.DataFrame(columns=["contract_date", *OBSERVED_NUMERIC_COLUMNS, *OBSERVED_TEXT_COLUMNS])
    for column in OBSERVED_NUMERIC_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.sort_values(
        ["station_id", "contract_date", "source_cache_mtime", "source_cache_dir"],
        ascending=[True, True, False, True],
    )
    out = out.drop_duplicates(["station_id", "contract_date"], keep="first")
    keep = ["contract_date", *OBSERVED_NUMERIC_COLUMNS, *OBSERVED_TEXT_COLUMNS]
    return out[keep].sort_values("contract_date").reset_index(drop=True)


def load_same_day_provider_forecasts(
    project_root: str | Path = ".",
    timing_mode: str = "same_day_11am",
    providers: tuple[str, ...] = TARGET_PROVIDERS,
) -> pd.DataFrame:
    root = Path(project_root)
    frames: list[pd.DataFrame] = []
    calibration_dir = root / "data" / "calibration"
    cache_paths = sorted(
        {
            path
            for _, pattern in FORECAST_CACHE_PATTERNS
            for path in calibration_dir.glob(pattern)
            if _include_forecast_cache_path(path)
        }
    )
    for path in cache_paths:
        try:
            frame = pd.read_csv(path, low_memory=False)
        except pd.errors.EmptyDataError:
            continue
        if frame.empty:
            continue
        for column in FORECAST_COLUMNS:
            if column not in frame:
                frame[column] = pd.NA
        frame = frame[FORECAST_COLUMNS].copy()
        frame["source_cache_dir"] = path.parent.name
        frame["source_cache_mtime"] = path.stat().st_mtime
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=[*FORECAST_COLUMNS, "source_cache_dir", "source_cache_mtime"])

    out = pd.concat(frames, ignore_index=True)
    out["station_id"] = out["station_id"].astype("string").str.upper()
    out["provider"] = out["provider"].astype("string").str.lower()
    out["timing_mode"] = out["timing_mode"].astype("string")
    out["contract_date"] = out["contract_date"].astype("string").str[:10]
    out["fetch_status"] = out["fetch_status"].astype("string").str.lower().fillna("ok")
    out = out.loc[
        out["provider"].isin(providers)
        & out["timing_mode"].eq(timing_mode)
        & out["fetch_status"].eq("ok")
    ].copy()
    if out.empty:
        return out
    for column in PROVIDER_NUMERIC_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["station_id", "provider", "contract_date", "raw_forecast_high_f"])
    if out.empty:
        return out
    out["_source_quality_rank"] = out["source_cache_dir"].map(_source_quality_rank)
    out = out.sort_values(
        [
            "station_id",
            "provider",
            "contract_date",
            "_source_quality_rank",
            "source_cache_mtime",
            "source_cache_dir",
        ],
        ascending=[True, True, True, True, False, True],
    )
    out = out.drop_duplicates(["station_id", "provider", "contract_date"], keep="first")
    out = out.drop(columns=["_source_quality_rank"]).reset_index(drop=True)
    return out


def build_station_wide_dataset(
    project_root: str | Path = ".",
    station_id: str = "KATL",
    timing_mode: str = "same_day_11am",
    providers: tuple[str, ...] = TARGET_PROVIDERS,
) -> pd.DataFrame:
    root = Path(project_root)
    station_id = station_id.upper()
    actuals = _load_station_actuals(root, station_id)
    current_observations = load_current_observation_features(root, station_id, timing_mode=timing_mode)
    station_meta = _load_station_meta(root, station_id)
    forecasts = load_same_day_provider_forecasts(root, timing_mode=timing_mode, providers=providers)
    forecasts = forecasts.loc[forecasts["station_id"].eq(station_id)].copy()

    wide = actuals.copy()
    if not current_observations.empty:
        wide = wide.merge(current_observations, on="contract_date", how="left")
    for provider in providers:
        provider_frame = forecasts.loc[forecasts["provider"].eq(provider)].copy()
        provider_wide = _provider_wide(provider_frame, provider)
        wide = wide.merge(provider_wide, on="contract_date", how="left")

    for key, value in station_meta.items():
        wide[key] = value

    wide = wide.sort_values("contract_date").reset_index(drop=True)
    wide = _add_calendar_features(wide)
    wide = _add_current_observation_derived_features(wide)
    wide = _add_provider_availability_features(wide, providers)
    wide = _add_provider_time_features(wide, providers, str(station_meta.get("timezone", "UTC")))
    wide = _add_ensemble_features(wide, providers)
    wide = _add_forecast_shape_features(wide, providers)
    wide = _add_lagged_actual_features(wide)
    wide = _add_lagged_provider_error_features(wide, providers)
    wide = _add_forecast_history_delta_features(wide, providers)
    return wide


def raw_baseline_predictions(frame: pd.DataFrame, config: StationStackingConfig) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    base = frame.dropna(subset=[TARGET]).copy()
    for provider in config.providers:
        column = HIGH_COLUMNS[provider]
        if column not in base:
            continue
        pred = base.loc[base[column].notna(), ["contract_date", TARGET, column]].copy()
        if pred.empty:
            continue
        pred["method"] = f"{provider}_raw"
        pred["predicted_high_f"] = pred[column]
        pred["evaluation_scope"] = "provider_available_dates"
        rows.append(_prediction_columns(pred))

    for method, column in [("provider_mean", "provider_mean_high_f"), ("provider_median", "provider_median_high_f")]:
        if column not in base:
            continue
        pred = base.loc[base[column].notna(), ["contract_date", TARGET, column]].copy()
        if pred.empty:
            continue
        pred["method"] = method
        pred["predicted_high_f"] = pred[column]
        pred["evaluation_scope"] = "provider_available_dates"
        rows.append(_prediction_columns(pred))

    best = _walk_forward_best_raw_provider(base, config)
    if not best.empty:
        rows.append(best)
    return pd.concat(rows, ignore_index=True) if rows else _empty_predictions()


def walk_forward_base_model_predictions(frame: pd.DataFrame, config: StationStackingConfig) -> pd.DataFrame:
    modeling_frame, categorical, numeric = _modeling_frame(frame, config)
    if modeling_frame.empty or len(modeling_frame) <= config.effective_min_train_rows:
        return _empty_predictions()
    models = _build_base_model_pipelines(config, categorical, numeric)
    feature_cols = categorical + numeric
    rows: list[pd.DataFrame] = []
    dates = sorted(modeling_frame["contract_date"].unique())
    completed_blocks = 0
    for start_idx in range(0, len(dates), config.effective_refit_days):
        block_start = dates[start_idx]
        block_dates = dates[start_idx : start_idx + config.effective_refit_days]
        train = modeling_frame.loc[modeling_frame["contract_date"] < block_start].copy()
        valid = modeling_frame.loc[modeling_frame["contract_date"].isin(block_dates)].copy()
        if len(train) < config.effective_min_train_rows or valid.empty:
            continue
        for method, estimator in models.items():
            estimator.fit(train[feature_cols], train[TARGET])
            pred = valid[["contract_date", TARGET]].copy()
            pred["method"] = method
            pred["predicted_high_f"] = estimator.predict(valid[feature_cols])
            pred["evaluation_scope"] = "walk_forward_model"
            rows.append(_prediction_columns(pred))
        completed_blocks += 1
        if config.fast_mode and completed_blocks >= config.fast_max_validation_blocks:
            break
    return pd.concat(rows, ignore_index=True) if rows else _empty_predictions()


def walk_forward_stack_predictions(
    base_predictions: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    config: StationStackingConfig,
) -> pd.DataFrame:
    if base_predictions.empty or baseline_predictions.empty:
        return _empty_predictions()
    stack_source = _stack_source_frame(base_predictions, baseline_predictions)
    if stack_source.empty or len(stack_source) <= config.effective_min_meta_train_rows:
        return _empty_predictions()

    from sklearn.linear_model import RidgeCV

    stack_features = [f"{method}_predicted_high_f" for method in [*BASE_MODEL_METHODS, *BASELINE_METHODS]]
    if any(column not in stack_source for column in stack_features):
        return _empty_predictions()
    stack_source = stack_source.dropna(subset=stack_features + [TARGET]).sort_values("contract_date").reset_index(drop=True)
    rows: list[pd.DataFrame] = []
    dates = sorted(stack_source["contract_date"].unique())
    completed_blocks = 0
    for start_idx in range(0, len(dates), config.effective_refit_days):
        block_start = dates[start_idx]
        block_dates = dates[start_idx : start_idx + config.effective_refit_days]
        train = stack_source.loc[stack_source["contract_date"] < block_start].copy()
        valid = stack_source.loc[stack_source["contract_date"].isin(block_dates)].copy()
        if len(train) < config.effective_min_meta_train_rows or valid.empty:
            continue
        model = RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0, 100.0))
        model.fit(train[stack_features], train[TARGET])
        pred = valid[["contract_date", TARGET]].copy()
        pred["method"] = STACK_METHOD
        pred["predicted_high_f"] = model.predict(valid[stack_features])
        pred["evaluation_scope"] = "walk_forward_stack"
        rows.append(_prediction_columns(pred))
        completed_blocks += 1
        if config.fast_mode and completed_blocks >= config.fast_max_validation_blocks:
            break
    return pd.concat(rows, ignore_index=True) if rows else _empty_predictions()


def summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(
            columns=[
                "evaluation_scope",
                "method",
                "count",
                "mae_f",
                "rmse_f",
                "bias_f",
                "within_1f_pct",
                "within_2f_pct",
                "within_3f_pct",
                "first_contract_date",
                "last_contract_date",
            ]
        )
    metrics = (
        predictions.groupby(["evaluation_scope", "method"], dropna=False)
        .apply(_metric_row, include_groups=False)
        .reset_index()
    )
    common = _common_date_metrics(predictions)
    if not common.empty:
        metrics = pd.concat([metrics, common], ignore_index=True)
    return metrics.sort_values(["evaluation_scope", "mae_f", "method"]).reset_index(drop=True)


def run_station_stacking_experiment(config: StationStackingConfig) -> StationStackingResult:
    features = build_station_wide_dataset(
        config.resolved_project_root(),
        station_id=config.station_id,
        timing_mode=config.timing_mode,
        providers=config.providers,
    )
    baseline_predictions = raw_baseline_predictions(features, config)
    model_predictions = walk_forward_base_model_predictions(features, config)
    stack_predictions = walk_forward_stack_predictions(model_predictions, baseline_predictions, config)
    predictions = pd.concat(
        [frame for frame in [baseline_predictions, model_predictions, stack_predictions] if not frame.empty],
        ignore_index=True,
    ) if not baseline_predictions.empty or not model_predictions.empty or not stack_predictions.empty else _empty_predictions()
    metrics = summarize_predictions(predictions)
    categorical, numeric = feature_columns(features, config)
    feature_columns_frame = pd.DataFrame(
        [{"feature": feature, "kind": "categorical"} for feature in categorical]
        + [{"feature": feature, "kind": "numeric"} for feature in numeric]
    )

    output_dir = config.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    station = config.station_id.upper()
    paths = {
        "features": output_dir / f"{station}_features.csv",
        "predictions": output_dir / f"{station}_predictions.csv",
        "metrics": output_dir / f"{station}_metrics.csv",
        "feature_columns": output_dir / f"{station}_feature_columns.csv",
    }
    features.to_csv(paths["features"], index=False)
    predictions.to_csv(paths["predictions"], index=False)
    metrics.to_csv(paths["metrics"], index=False)
    feature_columns_frame.to_csv(paths["feature_columns"], index=False)
    return StationStackingResult(
        station_id=station,
        features=features,
        predictions=predictions,
        metrics=metrics,
        feature_columns=feature_columns_frame,
        output_paths=paths,
    )


def feature_columns(frame: pd.DataFrame, config: StationStackingConfig) -> tuple[list[str], list[str]]:
    categorical = [column for column in ["day_of_week", *OBSERVED_CATEGORICAL_FEATURES] if column in frame]
    excluded = {
        TARGET,
        "contract_date",
        "station_id",
        "station_name",
        "airport_name",
        "city_label",
        "timezone",
        "country",
        "all_provider_highs_available",
        "observed_as_of_time_local",
        "observed_as_of_time_utc",
        "observed_source",
        "observed_qc_field",
        "observed_raw_metar",
        "observed_data_source",
        "observed_unavailable_reason",
    }
    excluded.update(column for column in frame.columns if column.endswith("_source_file_or_url"))
    excluded.update(column for column in frame.columns if column.endswith("_source_cache_dir"))
    excluded.update(column for column in frame.columns if column.endswith("_data_source"))
    excluded.update(column for column in frame.columns if column.endswith("_source_label"))
    excluded.update(column for column in frame.columns if column.endswith("_model"))
    excluded.update(column for column in frame.columns if column.endswith("_cycle_selection_policy"))
    excluded.update(column for column in frame.columns if column.endswith("_forecast_as_of"))
    excluded.update(column for column in frame.columns if column.endswith("_issued_at"))
    excluded.update(column for column in frame.columns if column.endswith("_forecast_window_start"))
    excluded.update(column for column in frame.columns if column.endswith("_forecast_window_end"))

    numeric: list[str] = []
    for column in frame.columns:
        if column in excluded or column in categorical:
            continue
        series = frame[column]
        if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
            if pd.to_numeric(series, errors="coerce").notna().any():
                numeric.append(column)
    return categorical, numeric


def _source_quality_rank(source_cache_dir: str) -> int:
    name = str(source_cache_dir).lower()
    if "smoke" in name:
        return 5
    if "retry" in name or "gated" in name:
        return 4
    if "fixed" in name:
        return 4
    if "direct_nbm" in name:
        return 2
    if "sdk_11am_nbm" in name:
        return 0
    return 1


def _include_forecast_cache_path(path: Path) -> bool:
    name = path.parent.name.lower()
    if "direct_nbm" in name:
        value = os.getenv("WEATHER_RESEARCH_INCLUDE_DIRECT_NBM", "")
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return True


def _load_station_actuals(root: Path, station_id: str) -> pd.DataFrame:
    path = root / "data" / "processed" / "actual_highs.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing actual highs file: {path}")
    actuals = pd.read_csv(path)
    required = {"station_code", "date_local", "actual_high_f"}
    missing = required - set(actuals.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    out = actuals.loc[actuals["station_code"].astype(str).str.upper().eq(station_id)].copy()
    out = out.rename(columns={"date_local": "contract_date"})
    out["contract_date"] = out["contract_date"].astype(str).str[:10]
    out[TARGET] = pd.to_numeric(out[TARGET], errors="coerce")
    return out[["contract_date", TARGET]].dropna(subset=["contract_date"]).sort_values("contract_date").reset_index(drop=True)


def _load_station_meta(root: Path, station_id: str) -> dict[str, Any]:
    path = root / "data" / "processed" / "station_registry.csv"
    if not path.exists():
        return {"station_id": station_id, "timezone": "UTC"}
    frame = pd.read_csv(path)
    code_col = "station_code" if "station_code" in frame.columns else "station_id"
    row = frame.loc[frame[code_col].astype(str).str.upper().eq(station_id)].head(1)
    if row.empty:
        return {"station_id": station_id, "timezone": "UTC"}
    values = row.iloc[0].to_dict()
    values["station_id"] = station_id
    for column in ["lat", "lon"]:
        if column in values:
            values[column] = pd.to_numeric(values[column], errors="coerce")
    return values


def _provider_wide(frame: pd.DataFrame, provider: str) -> pd.DataFrame:
    output_columns = _provider_wide_columns(provider)
    if frame.empty:
        return pd.DataFrame(columns=["contract_date", *output_columns])
    keep = ["contract_date", *PROVIDER_NUMERIC_COLUMNS, *PROVIDER_TEXT_COLUMNS]
    for column in keep:
        if column not in frame:
            frame[column] = pd.NA
    out = frame[keep].drop_duplicates("contract_date", keep="first").copy()
    rename: dict[str, str] = {
        "raw_forecast_high_f": HIGH_COLUMNS[provider],
    }
    for column in PROVIDER_NUMERIC_COLUMNS:
        if column != "raw_forecast_high_f":
            rename[column] = f"{provider}_{column}"
    for column in PROVIDER_TEXT_COLUMNS:
        rename[column] = f"{provider}_{column}"
    out = out.rename(columns=rename)
    for column in output_columns:
        if column not in out:
            out[column] = pd.NA
    return out[["contract_date", *output_columns]]


def _provider_wide_columns(provider: str) -> list[str]:
    columns = [HIGH_COLUMNS[provider]]
    columns.extend(f"{provider}_{column}" for column in PROVIDER_NUMERIC_COLUMNS if column != "raw_forecast_high_f")
    columns.extend(f"{provider}_{column}" for column in PROVIDER_TEXT_COLUMNS)
    return columns


def _add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    dates = pd.to_datetime(out["contract_date"], errors="coerce")
    out["year"] = dates.dt.year
    out["month"] = dates.dt.month
    out["day_of_year"] = dates.dt.dayofyear
    out["day_of_year_sin"] = np.sin(2 * math.pi * out["day_of_year"] / 366)
    out["day_of_year_cos"] = np.cos(2 * math.pi * out["day_of_year"] / 366)
    out["day_of_week"] = dates.dt.day_name()
    out["is_weekend"] = dates.dt.dayofweek >= 5
    return out


def _add_current_observation_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "observed_temp_at_as_of_f" not in out:
        return out
    temp = pd.to_numeric(out.get("observed_temp_at_as_of_f"), errors="coerce")
    dewpoint = pd.to_numeric(out.get("observed_dewpoint_at_as_of_f"), errors="coerce")
    humidity = pd.to_numeric(out.get("observed_humidity_at_as_of"), errors="coerce")
    wind_speed = pd.to_numeric(out.get("observed_wind_speed_at_as_of"), errors="coerce")
    wind_direction = pd.to_numeric(out.get("observed_wind_direction_at_as_of"), errors="coerce")
    visibility = pd.to_numeric(out.get("observed_visibility_at_as_of"), errors="coerce")
    precip_recent = pd.to_numeric(out.get("observed_precip_recent_at_as_of"), errors="coerce")
    weather_code = out.get("observed_weather_code_at_as_of")
    weather_text = weather_code.astype("string").str.upper() if weather_code is not None else pd.Series(pd.NA, index=out.index)

    out["observed_dewpoint_depression_f"] = temp - dewpoint
    out["observed_heat_index_at_as_of_f"] = _heat_index_f(temp, humidity)
    out["observed_wind_chill_at_as_of_f"] = _wind_chill_f(temp, wind_speed)
    radians = 2 * math.pi * wind_direction / 360
    out["observed_wind_dir_sin"] = np.sin(radians)
    out["observed_wind_dir_cos"] = np.cos(radians)
    out["observed_is_raining_at_as_of"] = (
        weather_text.str.contains(r"\b(?:RA|DZ|SH|TSRA|FZRA)\b", regex=True, na=False)
        | precip_recent.fillna(0).gt(0)
    )
    out["observed_is_fog_or_mist_at_as_of"] = (
        weather_text.str.contains(r"\b(?:FG|BR|HZ)\b", regex=True, na=False)
        | visibility.le(3)
    )
    out["observed_is_thunder_at_as_of"] = weather_text.str.contains(r"\bTS\b|TSRA|VCTS", regex=True, na=False)
    return out


def _heat_index_f(temp_f: pd.Series, humidity_pct: pd.Series) -> pd.Series:
    temp = pd.to_numeric(temp_f, errors="coerce")
    humidity = pd.to_numeric(humidity_pct, errors="coerce")
    heat_index = (
        -42.379
        + 2.04901523 * temp
        + 10.14333127 * humidity
        - 0.22475541 * temp * humidity
        - 0.00683783 * temp**2
        - 0.05481717 * humidity**2
        + 0.00122874 * temp**2 * humidity
        + 0.00085282 * temp * humidity**2
        - 0.00000199 * temp**2 * humidity**2
    )
    return heat_index.where(temp.ge(80) & humidity.ge(40))


def _wind_chill_f(temp_f: pd.Series, wind_speed_mph: pd.Series) -> pd.Series:
    temp = pd.to_numeric(temp_f, errors="coerce")
    wind_speed = pd.to_numeric(wind_speed_mph, errors="coerce")
    wind_chill = 35.74 + 0.6215 * temp - 35.75 * wind_speed**0.16 + 0.4275 * temp * wind_speed**0.16
    return wind_chill.where(temp.le(50) & wind_speed.gt(3))


def _add_provider_availability_features(frame: pd.DataFrame, providers: tuple[str, ...]) -> pd.DataFrame:
    out = frame.copy()
    high_cols = [HIGH_COLUMNS[provider] for provider in providers]
    for provider, column in HIGH_COLUMNS.items():
        if provider not in providers:
            continue
        if column not in out:
            out[column] = np.nan
        out[f"{provider}_available"] = out[column].notna()
        out[f"{provider}_missing"] = out[column].isna()
    out["provider_count_available"] = out[high_cols].notna().sum(axis=1)
    out["all_provider_highs_available"] = out[high_cols].notna().all(axis=1)
    return out


def _add_provider_time_features(frame: pd.DataFrame, providers: tuple[str, ...], timezone: str) -> pd.DataFrame:
    out = frame.copy()
    tz = ZoneInfo(timezone) if timezone else ZoneInfo("UTC")
    for provider in providers:
        issued_col = f"{provider}_issued_at"
        as_of_col = f"{provider}_forecast_as_of"
        if issued_col not in out:
            continue
        issued = pd.to_datetime(out[issued_col], errors="coerce", utc=True)
        as_of = pd.to_datetime(out.get(as_of_col), errors="coerce", utc=True)
        out[f"{provider}_issue_hour_utc"] = issued.dt.hour
        out[f"{provider}_issue_hour_local"] = issued.dt.tz_convert(tz).dt.hour
        out[f"{provider}_as_of_hour_local"] = as_of.dt.tz_convert(tz).dt.hour
        out[f"{provider}_forecast_lead_hours"] = (as_of - issued).dt.total_seconds() / 3600
    return out


def _add_ensemble_features(frame: pd.DataFrame, providers: tuple[str, ...]) -> pd.DataFrame:
    out = frame.copy()
    high_cols = [HIGH_COLUMNS[provider] for provider in providers]
    highs = out[high_cols]
    out["provider_mean_high_f"] = highs.mean(axis=1)
    out["provider_median_high_f"] = highs.median(axis=1)
    out["provider_min_high_f"] = highs.min(axis=1)
    out["provider_max_high_f"] = highs.max(axis=1)
    out["provider_spread_high_f"] = out["provider_max_high_f"] - out["provider_min_high_f"]
    out["provider_std_high_f"] = highs.std(axis=1)
    ranks = highs.rank(axis=1, method="average", ascending=True)
    for provider, column in HIGH_COLUMNS.items():
        if provider not in providers:
            continue
        out[f"{provider}_minus_provider_mean_high_f"] = out[column] - out["provider_mean_high_f"]
        out[f"{provider}_rank_high"] = ranks[column]
        out[f"{provider}_is_warmest"] = out[column].eq(out["provider_max_high_f"])
        out[f"{provider}_is_coldest"] = out[column].eq(out["provider_min_high_f"])
    return out


def _add_forecast_shape_features(frame: pd.DataFrame, providers: tuple[str, ...]) -> pd.DataFrame:
    out = frame.copy()
    for provider in providers:
        hour_min = out.get(f"{provider}_forecast_hour_min")
        hour_max = out.get(f"{provider}_forecast_hour_max")
        if hour_min is not None and hour_max is not None:
            out[f"{provider}_forecast_window_hours"] = hour_max - hour_min + 1
        if f"{provider}_cloud_cover_max" in out and f"{provider}_cloud_cover_mean" in out:
            out[f"{provider}_cloud_variability"] = out[f"{provider}_cloud_cover_max"] - out[f"{provider}_cloud_cover_mean"]
        if f"{provider}_wind_speed_max" in out and f"{provider}_wind_speed_mean" in out:
            out[f"{provider}_wind_gustiness"] = out[f"{provider}_wind_speed_max"] - out[f"{provider}_wind_speed_mean"]
        if HIGH_COLUMNS[provider] in out and f"{provider}_dewpoint_mean_f" in out:
            out[f"{provider}_dewpoint_depression_f"] = out[HIGH_COLUMNS[provider]] - out[f"{provider}_dewpoint_mean_f"]
        if f"{provider}_wind_direction_mean" in out:
            radians = 2 * math.pi * pd.to_numeric(out[f"{provider}_wind_direction_mean"], errors="coerce") / 360
            out[f"{provider}_wind_dir_sin"] = np.sin(radians)
            out[f"{provider}_wind_dir_cos"] = np.cos(radians)
    return out


def _add_lagged_actual_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    actual = pd.to_numeric(out[TARGET], errors="coerce")
    shifted = actual.shift(1)
    out["actual_high_lag_1d"] = shifted
    out["actual_high_lag_2d"] = actual.shift(2)
    out["actual_high_lag_3d"] = actual.shift(3)
    out["actual_high_trend_1d"] = out["actual_high_lag_1d"] - out["actual_high_lag_2d"]
    out["actual_high_trend_3d"] = out["actual_high_lag_1d"] - actual.shift(4)
    for window in (3, 7, 14, 30):
        out[f"actual_high_roll_{window}d_mean"] = shifted.rolling(window, min_periods=1).mean()
        out[f"actual_high_roll_{window}d_std"] = shifted.rolling(window, min_periods=2).std()
        out[f"actual_high_roll_{window}d_min"] = shifted.rolling(window, min_periods=1).min()
        out[f"actual_high_roll_{window}d_max"] = shifted.rolling(window, min_periods=1).max()
    return out


def _add_lagged_provider_error_features(frame: pd.DataFrame, providers: tuple[str, ...]) -> pd.DataFrame:
    out = frame.copy()
    actual = pd.to_numeric(out[TARGET], errors="coerce")
    for provider in providers:
        high_col = HIGH_COLUMNS[provider]
        if high_col not in out:
            continue
        error = actual - pd.to_numeric(out[high_col], errors="coerce")
        shifted_error = error.shift(1)
        shifted_abs_error = error.abs().shift(1)
        out[f"{provider}_error_lag_1d_f"] = shifted_error
        out[f"{provider}_abs_error_lag_1d_f"] = shifted_abs_error
        out[f"{provider}_rolling_bias_7d_f"] = shifted_error.rolling(7, min_periods=2).mean()
        out[f"{provider}_rolling_bias_30d_f"] = shifted_error.rolling(30, min_periods=5).mean()
        out[f"{provider}_rolling_mae_30d_f"] = shifted_abs_error.rolling(30, min_periods=5).mean()
    return out


def _add_forecast_history_delta_features(frame: pd.DataFrame, providers: tuple[str, ...]) -> pd.DataFrame:
    out = frame.copy()
    for provider in providers:
        high_col = HIGH_COLUMNS[provider]
        if high_col not in out:
            continue
        out[f"{provider}_minus_actual_high_lag_1d_f"] = out[high_col] - out["actual_high_lag_1d"]
        out[f"{provider}_minus_actual_high_roll_7d_mean_f"] = out[high_col] - out["actual_high_roll_7d_mean"]
        out[f"{provider}_minus_actual_high_roll_30d_mean_f"] = out[high_col] - out["actual_high_roll_30d_mean"]
    return out


def _prediction_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame[["contract_date", "method", "evaluation_scope", TARGET, "predicted_high_f"]].copy()
    out["error_f"] = out[TARGET] - out["predicted_high_f"]
    out["absolute_error_f"] = out["error_f"].abs()
    return out


def _empty_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["contract_date", "method", "evaluation_scope", TARGET, "predicted_high_f", "error_f", "absolute_error_f"]
    )


def _walk_forward_best_raw_provider(frame: pd.DataFrame, config: StationStackingConfig) -> pd.DataFrame:
    high_cols = [HIGH_COLUMNS[provider] for provider in config.providers]
    required = [TARGET, *high_cols]
    complete = frame.dropna(subset=required).sort_values("contract_date").reset_index(drop=True)
    if len(complete) <= config.effective_min_train_rows:
        return _empty_predictions()
    rows: list[pd.DataFrame] = []
    dates = sorted(complete["contract_date"].unique())
    completed_blocks = 0
    for start_idx in range(0, len(dates), config.effective_refit_days):
        block_start = dates[start_idx]
        block_dates = dates[start_idx : start_idx + config.effective_refit_days]
        train = complete.loc[complete["contract_date"] < block_start]
        valid = complete.loc[complete["contract_date"].isin(block_dates)].copy()
        if len(train) < config.effective_min_train_rows or valid.empty:
            continue
        mae_by_provider = {
            provider: float((train[TARGET] - train[HIGH_COLUMNS[provider]]).abs().mean())
            for provider in config.providers
        }
        best_provider = min(mae_by_provider, key=mae_by_provider.get)
        pred = valid[["contract_date", TARGET]].copy()
        pred["method"] = "best_raw_provider"
        pred["predicted_high_f"] = valid[HIGH_COLUMNS[best_provider]]
        pred["evaluation_scope"] = "complete_provider_walk_forward"
        rows.append(_prediction_columns(pred))
        completed_blocks += 1
        if config.fast_mode and completed_blocks >= config.fast_max_validation_blocks:
            break
    return pd.concat(rows, ignore_index=True) if rows else _empty_predictions()


def _modeling_frame(frame: pd.DataFrame, config: StationStackingConfig) -> tuple[pd.DataFrame, list[str], list[str]]:
    categorical, numeric = feature_columns(frame, config)
    required = [TARGET, *[HIGH_COLUMNS[provider] for provider in config.providers]]
    clean = frame.dropna(subset=required).loc[frame["all_provider_highs_available"].fillna(False)].copy()
    clean = clean.sort_values("contract_date").reset_index(drop=True)
    numeric = [column for column in numeric if column in clean and clean[column].notna().any()]
    categorical = [column for column in categorical if column in clean]
    return clean, categorical, numeric


def _build_base_model_pipelines(
    config: StationStackingConfig,
    categorical: list[str],
    numeric: list[str],
) -> dict[str, Any]:
    try:
        from catboost import CatBoostRegressor
        from lightgbm import LGBMRegressor
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError(
            "Station stacking notebooks need xgboost, lightgbm, and catboost. "
            "Install them with: python -m pip install -r requirements.txt"
        ) from exc

    n_estimators = 80 if config.fast_mode else 450
    cat_iterations = 80 if config.fast_mode else 450
    def preprocessor() -> ColumnTransformer:
        return ColumnTransformer(
            transformers=[
                (
                    "cat",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                        ]
                    ),
                    categorical,
                ),
                ("num", SimpleImputer(strategy="median"), numeric),
            ],
            remainder="drop",
        )

    return {
        "xgboost": Pipeline(
            [
                ("prep", preprocessor()),
                (
                    "model",
                    XGBRegressor(
                        objective="reg:squarederror",
                        n_estimators=n_estimators,
                        learning_rate=0.04,
                        max_depth=3,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        random_state=config.random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "lightgbm": Pipeline(
            [
                ("prep", preprocessor()),
                (
                    "model",
                    LGBMRegressor(
                        n_estimators=n_estimators,
                        learning_rate=0.04,
                        num_leaves=31,
                        min_child_samples=20,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        random_state=config.random_state,
                        n_jobs=-1,
                        verbose=-1,
                    ),
                ),
            ]
        ),
        "catboost": Pipeline(
            [
                ("prep", preprocessor()),
                (
                    "model",
                    CatBoostRegressor(
                        iterations=cat_iterations,
                        learning_rate=0.04,
                        depth=6,
                        loss_function="RMSE",
                        random_seed=config.random_state,
                        verbose=False,
                        allow_writing_files=False,
                    ),
                ),
            ]
        ),
    }


def _stack_source_frame(base_predictions: pd.DataFrame, baseline_predictions: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([base_predictions, baseline_predictions], ignore_index=True)
    combined = combined.loc[combined["method"].isin([*BASE_MODEL_METHODS, *BASELINE_METHODS])].copy()
    if combined.empty:
        return pd.DataFrame()
    pivot = combined.pivot_table(
        index="contract_date",
        columns="method",
        values="predicted_high_f",
        aggfunc="first",
    )
    pivot.columns = [f"{column}_predicted_high_f" for column in pivot.columns]
    actuals = combined.groupby("contract_date", dropna=False)[TARGET].first()
    out = pivot.join(actuals).reset_index()
    return out


def _metric_row(group: pd.DataFrame) -> pd.Series:
    error = pd.to_numeric(group["error_f"], errors="coerce")
    abs_error = error.abs()
    return pd.Series(
        {
            "count": int(error.notna().sum()),
            "mae_f": float(abs_error.mean()),
            "rmse_f": float(np.sqrt((error**2).mean())),
            "bias_f": float(error.mean()),
            "within_1f_pct": float((abs_error <= 1).mean() * 100),
            "within_2f_pct": float((abs_error <= 2).mean() * 100),
            "within_3f_pct": float((abs_error <= 3).mean() * 100),
            "first_contract_date": group["contract_date"].min(),
            "last_contract_date": group["contract_date"].max(),
        }
    )


def _common_date_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    method_dates = predictions.groupby("method")["contract_date"].apply(set)
    if method_dates.empty:
        return pd.DataFrame()
    common_dates = set.intersection(*method_dates.tolist())
    if not common_dates:
        return pd.DataFrame()
    common = predictions.loc[predictions["contract_date"].isin(common_dates)].copy()
    common["evaluation_scope"] = "common_dates_all_methods"
    return (
        common.groupby(["evaluation_scope", "method"], dropna=False)
        .apply(_metric_row, include_groups=False)
        .reset_index()
    )
