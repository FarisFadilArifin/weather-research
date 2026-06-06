from __future__ import annotations

import math
import os
from importlib.util import find_spec
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


TARGET_STATIONS = ("KATL", "KAUS", "KORD", "KDAL", "KHOU", "KLAX", "KMIA", "KLGA", "KSEA")
TARGET_PROVIDERS = ("gfs", "hrrr")
OPTIONAL_PROVIDERS = ("nbm",)
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
HIGH_COLUMNS.update({provider: f"{provider}_high_f" for provider in OPTIONAL_PROVIDERS})
BASELINE_METHODS = [
    "gfs_raw",
    "hrrr_raw",
    "provider_mean",
    "provider_median",
    "best_raw_provider",
]
BASE_MODEL_METHODS = ["xgboost", "lightgbm", "catboost"]
STACK_METHOD = "ridge_stack"
YEAR_SPLIT_SCOREBOARD_METHODS = (*BASE_MODEL_METHODS, STACK_METHOD, "hrrr_raw", "gfs_raw")
YEAR_SPLIT_VALIDATION_WEIGHTS = {2024: 0.35, 2025: 0.65}
STACK_FEATURE_SETS = {
    "models_only": tuple(BASE_MODEL_METHODS),
    "models_plus_raw": (*BASE_MODEL_METHODS, "hrrr_raw", "gfs_raw"),
}
REQUIRED_MODEL_PACKAGES = {
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "catboost": "catboost",
    "optuna": "optuna",
}


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
    optuna_trials: int | None = None
    stack_optuna_trials: int | None = None
    optuna_verbose: bool = False
    feature_importance_repeats: int | None = None
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

    @property
    def effective_optuna_trials(self) -> int:
        if self.optuna_trials is not None:
            return max(1, int(self.optuna_trials))
        return 8 if self.fast_mode else 50

    @property
    def effective_stack_optuna_trials(self) -> int:
        if self.stack_optuna_trials is not None:
            return max(1, int(self.stack_optuna_trials))
        return 8 if self.fast_mode else min(self.effective_optuna_trials, 50)

    @property
    def effective_feature_importance_repeats(self) -> int:
        if self.feature_importance_repeats is not None:
            return max(1, int(self.feature_importance_repeats))
        return 3 if self.fast_mode else 10


@dataclass
class StationStackingResult:
    station_id: str
    features: pd.DataFrame
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    feature_columns: pd.DataFrame
    output_paths: dict[str, Path]


@dataclass(frozen=True)
class YearSplitFold:
    name: str
    train_start_year: int
    train_end_year: int
    validation_year: int


@dataclass
class YearSplitExperimentResult:
    station_id: str
    features: pd.DataFrame
    tuning_results: pd.DataFrame
    validation_predictions: pd.DataFrame
    test_predictions: pd.DataFrame
    metrics: pd.DataFrame
    stack_tuning_results: pd.DataFrame
    scoreboard: pd.DataFrame
    bracket_predictions: pd.DataFrame
    bracket_metrics: pd.DataFrame
    feature_columns: pd.DataFrame
    selected_hyperparameters: pd.DataFrame
    feature_importance: pd.DataFrame
    output_paths: dict[str, Path]


YEAR_SPLIT_FOLDS = (
    YearSplitFold("fold_2021_2023_to_2024", 2021, 2023, 2024),
    YearSplitFold("fold_2022_2024_to_2025", 2022, 2024, 2025),
)
YEAR_SPLIT_TEST_TRAIN_YEARS = (2021, 2025)
YEAR_SPLIT_TEST_YEAR = 2026


def missing_model_dependencies() -> list[str]:
    return sorted(package for package, module in REQUIRED_MODEL_PACKAGES.items() if find_spec(module) is None)


def require_model_dependencies() -> None:
    missing = missing_model_dependencies()
    if missing:
        missing_list = ", ".join(missing)
        raise ImportError(
            "Station stacking ML requires xgboost, lightgbm, and catboost. "
            f"Missing: {missing_list}. Install them with: python -m pip install -r requirements.txt"
        )


def missing_expected_model_methods(metrics: pd.DataFrame) -> list[str]:
    if metrics.empty or "method" not in metrics:
        return [*BASE_MODEL_METHODS, STACK_METHOD]
    methods = set(metrics["method"].dropna().astype(str))
    return [method for method in [*BASE_MODEL_METHODS, STACK_METHOD] if method not in methods]


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
    wide = _add_provider_cross_model_features(wide, providers)
    wide = _add_lagged_actual_features(wide)
    wide = _add_lagged_provider_error_features(wide, providers)
    wide = _add_prior_month_provider_error_features(wide, providers)
    wide = _add_forecast_history_delta_features(wide, providers)
    wide = _add_observation_history_delta_features(wide)
    wide = _add_observation_forecast_delta_features(wide, providers)
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

    baseline_methods = [f"{provider}_raw" for provider in config.providers]
    baseline_methods.extend(method for method in BASELINE_METHODS if not method.endswith("_raw"))
    stack_features = [f"{method}_predicted_high_f" for method in [*BASE_MODEL_METHODS, *baseline_methods]]
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


def run_station_year_split_experiment(config: StationStackingConfig) -> YearSplitExperimentResult:
    features = build_station_wide_dataset(
        config.resolved_project_root(),
        station_id=config.station_id,
        timing_mode=config.timing_mode,
        providers=config.providers,
    )
    modeling_frame, categorical, numeric = _modeling_frame(features, config)
    feature_columns_frame = pd.DataFrame(
        [{"feature": feature, "kind": "categorical"} for feature in categorical]
        + [{"feature": feature, "kind": "numeric"} for feature in numeric]
    )

    baseline_validation = year_split_baseline_predictions(modeling_frame, config, YEAR_SPLIT_FOLDS)
    tuning_results, validation_predictions, selected = tune_year_split_base_models(
        modeling_frame,
        config,
        categorical,
        numeric,
        YEAR_SPLIT_FOLDS,
    )
    test_predictions = year_split_test_predictions(
        modeling_frame,
        config,
        categorical,
        numeric,
        selected,
        train_years=YEAR_SPLIT_TEST_TRAIN_YEARS,
        test_year=YEAR_SPLIT_TEST_YEAR,
    )
    test_stack_predictions, stack_tuning_results = tune_year_split_stack_model(
        validation_predictions=pd.concat(
            [frame for frame in [baseline_validation, validation_predictions] if not frame.empty],
            ignore_index=True,
        )
        if not baseline_validation.empty or not validation_predictions.empty
        else _empty_year_split_predictions(),
        test_predictions=test_predictions,
        config=config,
        test_year=YEAR_SPLIT_TEST_YEAR,
    )
    if not test_stack_predictions.empty:
        test_predictions = pd.concat([test_predictions, test_stack_predictions], ignore_index=True)
    feature_importance = year_split_feature_importance(
        modeling_frame,
        config,
        categorical,
        numeric,
        selected,
        train_years=YEAR_SPLIT_TEST_TRAIN_YEARS,
        test_year=YEAR_SPLIT_TEST_YEAR,
    )
    validation_predictions = pd.concat(
        [frame for frame in [baseline_validation, validation_predictions] if not frame.empty],
        ignore_index=True,
    ) if not baseline_validation.empty or not validation_predictions.empty else _empty_year_split_predictions()
    metrics = summarize_year_split_predictions(validation_predictions, test_predictions)
    scoreboard = year_split_scoreboard(validation_predictions, test_predictions)
    bracket_predictions = year_split_bracket_predictions(test_predictions, test_year=YEAR_SPLIT_TEST_YEAR)
    bracket_metrics = year_split_bracket_metrics(bracket_predictions)

    output_dir = config.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    station = config.station_id.upper()
    paths = {
        "features": output_dir / f"{station}_features.csv",
        "year_split_tuning": output_dir / f"{station}_year_split_tuning.csv",
        "year_split_validation_predictions": output_dir / f"{station}_year_split_validation_predictions.csv",
        "year_split_test_predictions": output_dir / f"{station}_year_split_test_predictions.csv",
        "year_split_metrics": output_dir / f"{station}_year_split_metrics.csv",
        "year_split_selected_hyperparameters": output_dir / f"{station}_year_split_selected_hyperparameters.csv",
        "year_split_feature_importance": output_dir / f"{station}_year_split_feature_importance.csv",
        "year_split_stack_tuning": output_dir / f"{station}_year_split_stack_tuning.csv",
        "year_split_scoreboard": output_dir / f"{station}_year_split_scoreboard.csv",
        "year_split_bracket_predictions": output_dir / f"{station}_year_split_bracket_predictions.csv",
        "year_split_bracket_metrics": output_dir / f"{station}_year_split_bracket_metrics.csv",
        "feature_columns": output_dir / f"{station}_feature_columns.csv",
    }
    features.to_csv(paths["features"], index=False)
    tuning_results.to_csv(paths["year_split_tuning"], index=False)
    validation_predictions.to_csv(paths["year_split_validation_predictions"], index=False)
    test_predictions.to_csv(paths["year_split_test_predictions"], index=False)
    metrics.to_csv(paths["year_split_metrics"], index=False)
    selected.to_csv(paths["year_split_selected_hyperparameters"], index=False)
    feature_importance.to_csv(paths["year_split_feature_importance"], index=False)
    stack_tuning_results.to_csv(paths["year_split_stack_tuning"], index=False)
    scoreboard.to_csv(paths["year_split_scoreboard"], index=False)
    bracket_predictions.to_csv(paths["year_split_bracket_predictions"], index=False)
    bracket_metrics.to_csv(paths["year_split_bracket_metrics"], index=False)
    feature_columns_frame.to_csv(paths["feature_columns"], index=False)
    return YearSplitExperimentResult(
        station_id=station,
        features=features,
        tuning_results=tuning_results,
        validation_predictions=validation_predictions,
        test_predictions=test_predictions,
        metrics=metrics,
        stack_tuning_results=stack_tuning_results,
        scoreboard=scoreboard,
        bracket_predictions=bracket_predictions,
        bracket_metrics=bracket_metrics,
        feature_columns=feature_columns_frame,
        selected_hyperparameters=selected,
        feature_importance=feature_importance,
        output_paths=paths,
    )


def year_split_baseline_predictions(
    frame: pd.DataFrame,
    config: StationStackingConfig,
    folds: tuple[YearSplitFold, ...] = YEAR_SPLIT_FOLDS,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    if frame.empty:
        return _empty_year_split_predictions()
    year = pd.to_numeric(frame.get("year"), errors="coerce")
    for fold in folds:
        train = frame.loc[year.between(fold.train_start_year, fold.train_end_year)].copy()
        valid = frame.loc[year.eq(fold.validation_year)].copy()
        if train.empty or valid.empty:
            continue
        for provider in config.providers:
            column = HIGH_COLUMNS[provider]
            if column in valid:
                pred = valid.loc[valid[column].notna(), ["contract_date", TARGET, column]].copy()
                if pred.empty:
                    continue
                pred["method"] = f"{provider}_raw"
                pred["predicted_high_f"] = pred[column]
                pred["evaluation_scope"] = "year_split_validation"
                pred["fold"] = fold.name
                rows.append(_year_split_prediction_columns(pred))
    return pd.concat(rows, ignore_index=True) if rows else _empty_year_split_predictions()


def tune_year_split_base_models(
    frame: pd.DataFrame,
    config: StationStackingConfig,
    categorical: list[str],
    numeric: list[str],
    folds: tuple[YearSplitFold, ...] = YEAR_SPLIT_FOLDS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return pd.DataFrame(), _empty_year_split_predictions(), pd.DataFrame()
    rows: list[dict[str, Any]] = []
    year = pd.to_numeric(frame.get("year"), errors="coerce")
    for method in BASE_MODEL_METHODS:
        study = _create_optuna_study(config, method)

        def objective(trial) -> float:
            params = _suggest_hyperparameters(method, trial, config)
            param_key = f"trial_{trial.number}"
            fold_scores: list[tuple[YearSplitFold, float]] = []
            for fold in folds:
                train = frame.loc[year.between(fold.train_start_year, fold.train_end_year)].copy()
                valid = frame.loc[year.eq(fold.validation_year)].copy()
                if train.empty or valid.empty:
                    continue
                try:
                    predicted, fit_metadata = _fit_predict_base_model(
                        config=config,
                        categorical=categorical,
                        numeric=numeric,
                        method=method,
                        params=params,
                        train=train,
                        valid=valid,
                        early_stopping=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    rows.append(
                        {
                            "method": method,
                            "trial_number": trial.number,
                            "param_key": param_key,
                            "fold": fold.name,
                            "fold_weight": _year_split_fold_weight(fold),
                            "mae_f": pd.NA,
                            "rmse_f": pd.NA,
                            "count": 0,
                            "status": "failed",
                            "error": str(exc),
                            **{f"param_{key}": value for key, value in params.items()},
                        }
                    )
                    raise
                pred = valid[["contract_date", TARGET]].copy()
                pred["method"] = method
                pred["param_key"] = param_key
                pred["predicted_high_f"] = predicted
                pred["evaluation_scope"] = "year_split_validation"
                pred["fold"] = fold.name
                metrics = _metric_row(_prediction_columns(pred))
                mae = float(metrics["mae_f"])
                rmse = float(metrics["rmse_f"])
                fold_scores.append((fold, rmse))
                rows.append(
                    {
                        "method": method,
                        "trial_number": trial.number,
                        "param_key": param_key,
                        "fold": fold.name,
                        "fold_weight": _year_split_fold_weight(fold),
                        "mae_f": mae,
                        "rmse_f": rmse,
                        "count": int(metrics["count"]),
                        "fit_numeric_features": fit_metadata["numeric_features"],
                        "fit_categorical_features": fit_metadata["categorical_features"],
                        "best_iteration": fit_metadata["best_iteration"],
                        "status": "ok",
                        "error": "",
                        **{f"param_{key}": value for key, value in params.items()},
                    }
                )
                current_score = _weighted_fold_score(fold_scores)
                if hasattr(trial, "report"):
                    trial.report(current_score, step=len(fold_scores))
                if hasattr(trial, "should_prune") and trial.should_prune():
                    raise _trial_pruned_exception()
            if not fold_scores:
                return float("inf")
            return _weighted_fold_score(fold_scores)

        study.optimize(objective, n_trials=config.effective_optuna_trials, show_progress_bar=False, catch=(Exception,))
    tuning = pd.DataFrame(rows)
    selected = _selected_hyperparameters(tuning)
    validation_predictions = _validation_predictions_for_selected_params(frame, config, categorical, numeric, folds, selected)
    return tuning, validation_predictions, selected


def year_split_test_predictions(
    frame: pd.DataFrame,
    config: StationStackingConfig,
    categorical: list[str],
    numeric: list[str],
    selected_hyperparameters: pd.DataFrame,
    train_years: tuple[int, int] = YEAR_SPLIT_TEST_TRAIN_YEARS,
    test_year: int = YEAR_SPLIT_TEST_YEAR,
) -> pd.DataFrame:
    if frame.empty:
        return _empty_year_split_predictions()
    year = pd.to_numeric(frame.get("year"), errors="coerce")
    train = frame.loc[year.between(train_years[0], train_years[1])].copy()
    test = frame.loc[year.eq(test_year)].copy()
    rows: list[pd.DataFrame] = []
    if train.empty or test.empty:
        return _empty_year_split_predictions()
    for provider in config.providers:
        column = HIGH_COLUMNS[provider]
        if column in test:
            pred = test.loc[test[column].notna(), ["contract_date", TARGET, column]].copy()
            if not pred.empty:
                pred["method"] = f"{provider}_raw"
                pred["predicted_high_f"] = pred[column]
                pred["evaluation_scope"] = "year_split_test"
                pred["fold"] = f"train_{train_years[0]}_{train_years[1]}_test_{test_year}"
                rows.append(_year_split_prediction_columns(pred))
    selected = selected_hyperparameters.copy()
    for _, row in selected.iterrows():
        method = str(row["method"])
        params = _params_from_selected_row(row)
        try:
            predicted, _ = _fit_predict_base_model(
                config=config,
                categorical=categorical,
                numeric=numeric,
                method=method,
                params=params,
                train=train,
                valid=test,
                early_stopping=False,
            )
        except Exception:
            continue
        pred = test[["contract_date", TARGET]].copy()
        pred["method"] = method
        pred["param_key"] = str(row["param_key"])
        pred["predicted_high_f"] = predicted
        pred["evaluation_scope"] = "year_split_test"
        pred["fold"] = f"train_{train_years[0]}_{train_years[1]}_test_{test_year}"
        rows.append(_year_split_prediction_columns(pred))
    return pd.concat(rows, ignore_index=True) if rows else _empty_year_split_predictions()


def tune_year_split_stack_model(
    validation_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    config: StationStackingConfig,
    test_year: int = YEAR_SPLIT_TEST_YEAR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tuning_columns = [
        "method",
        "trial_number",
        "param_key",
        "feature_set",
        "alpha",
        "fit_intercept",
        "mae_f",
        "rmse_f",
        "count",
        "status",
        "error",
    ]
    if validation_predictions.empty or test_predictions.empty:
        return _empty_year_split_predictions(), pd.DataFrame(columns=tuning_columns)
    stack_methods = list(STACK_FEATURE_SETS["models_plus_raw"])
    train_source = _year_split_stack_source_frame(validation_predictions, stack_methods)
    test_source = _year_split_stack_source_frame(test_predictions, stack_methods)
    if train_source.empty or test_source.empty:
        return _empty_year_split_predictions(), pd.DataFrame(columns=tuning_columns)
    if len(train_source) < config.effective_min_meta_train_rows:
        return _empty_year_split_predictions(), pd.DataFrame(columns=tuning_columns)

    from sklearn.linear_model import Ridge

    meta_train, meta_valid = _stack_meta_train_valid_split(train_source)
    rows: list[dict[str, Any]] = []
    if meta_train.empty or meta_valid.empty:
        return _empty_year_split_predictions(), pd.DataFrame(columns=tuning_columns)
    study = _create_stack_optuna_study(config)

    def objective(trial) -> float:
        feature_set = trial.suggest_categorical("feature_set", list(STACK_FEATURE_SETS))
        alpha = trial.suggest_float("alpha", 1e-4, 1e3, log=True)
        fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])
        stack_features = _stack_features_for_set(feature_set)
        train = meta_train.dropna(subset=[*stack_features, TARGET]).copy()
        valid = meta_valid.dropna(subset=[*stack_features, TARGET]).copy()
        param_key = f"stack_trial_{trial.number}"
        if train.empty or valid.empty:
            rows.append(
                {
                    "method": STACK_METHOD,
                    "trial_number": trial.number,
                    "param_key": param_key,
                    "feature_set": feature_set,
                    "alpha": alpha,
                    "fit_intercept": fit_intercept,
                    "mae_f": pd.NA,
                    "rmse_f": pd.NA,
                    "count": 0,
                    "status": "failed",
                    "error": "Missing complete stack train/validation rows.",
                }
            )
            return float("inf")
        try:
            model = Ridge(alpha=alpha, fit_intercept=fit_intercept)
            model.fit(train[stack_features], train[TARGET])
            predicted = model.predict(valid[stack_features])
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "method": STACK_METHOD,
                    "trial_number": trial.number,
                    "param_key": param_key,
                    "feature_set": feature_set,
                    "alpha": alpha,
                    "fit_intercept": fit_intercept,
                    "mae_f": pd.NA,
                    "rmse_f": pd.NA,
                    "count": 0,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            raise
        pred = valid[["contract_date", TARGET]].copy()
        pred["method"] = STACK_METHOD
        pred["predicted_high_f"] = predicted
        pred["evaluation_scope"] = "year_split_stack_validation"
        metrics = _metric_row(_prediction_columns(pred))
        rows.append(
            {
                "method": STACK_METHOD,
                "trial_number": trial.number,
                "param_key": param_key,
                "feature_set": feature_set,
                "alpha": alpha,
                "fit_intercept": fit_intercept,
                "mae_f": float(metrics["mae_f"]),
                "rmse_f": float(metrics["rmse_f"]),
                "count": int(metrics["count"]),
                "status": "ok",
                "error": "",
            }
        )
        return float(metrics["rmse_f"])

    study.optimize(objective, n_trials=config.effective_stack_optuna_trials, show_progress_bar=False, catch=(Exception,))
    tuning = pd.DataFrame(rows, columns=tuning_columns)
    ok = tuning.loc[tuning["status"].eq("ok")].copy()
    if ok.empty:
        return _empty_year_split_predictions(), tuning
    selected = ok.sort_values(["rmse_f", "param_key"]).iloc[0]
    stack_features = _stack_features_for_set(str(selected["feature_set"]))
    train = train_source.dropna(subset=[*stack_features, TARGET]).copy()
    test = test_source.dropna(subset=[*stack_features, TARGET]).copy()
    if len(train) < config.effective_min_meta_train_rows or test.empty:
        return _empty_year_split_predictions(), tuning
    try:
        model = Ridge(alpha=float(selected["alpha"]), fit_intercept=bool(selected["fit_intercept"]))
        model.fit(train[stack_features], train[TARGET])
        predicted = model.predict(test[stack_features])
    except Exception:
        return _empty_year_split_predictions(), tuning
    pred = test[["contract_date", TARGET]].copy()
    pred["method"] = STACK_METHOD
    pred["param_key"] = str(selected["param_key"])
    pred["predicted_high_f"] = predicted
    pred["evaluation_scope"] = "year_split_test"
    pred["fold"] = f"ridge_meta_train_2024_2025_test_{test_year}"
    return _year_split_prediction_columns(pred), tuning


def year_split_stack_test_predictions(
    validation_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    config: StationStackingConfig,
    test_year: int = YEAR_SPLIT_TEST_YEAR,
) -> pd.DataFrame:
    predictions, _ = tune_year_split_stack_model(validation_predictions, test_predictions, config, test_year)
    return predictions


def year_split_scoreboard(validation_predictions: pd.DataFrame, test_predictions: pd.DataFrame) -> pd.DataFrame:
    columns = ["period", "method", "count", "mae_f", "rmse_f"]
    frames: list[pd.DataFrame] = []
    for period, predictions in [
        ("validation_2024_2025", validation_predictions),
        (f"test_{YEAR_SPLIT_TEST_YEAR}", test_predictions),
    ]:
        if predictions.empty:
            continue
        frame = predictions.loc[predictions["method"].isin(YEAR_SPLIT_SCOREBOARD_METHODS)].copy()
        if frame.empty:
            continue
        metrics = frame.groupby("method", dropna=False).apply(_metric_row, include_groups=False).reset_index()
        metrics["period"] = period
        frames.append(metrics[columns])
    if not frames:
        return pd.DataFrame(columns=columns)
    return _sort_year_split_visible_methods(pd.concat(frames, ignore_index=True), include_period=True)[columns]


def year_split_bracket_predictions(
    test_predictions: pd.DataFrame,
    test_year: int = YEAR_SPLIT_TEST_YEAR,
) -> pd.DataFrame:
    columns = [
        "contract_date",
        "method",
        "actual_high_f",
        "predicted_high_f",
        "error_f",
        "absolute_error_f",
        "actual_rounded_high_f",
        "predicted_rounded_high_f",
        "actual_bracket",
        "predicted_bracket",
        "bracket_hit",
    ]
    if test_predictions.empty:
        return pd.DataFrame(columns=columns)
    frame = test_predictions.loc[
        test_predictions["evaluation_scope"].eq("year_split_test")
        & test_predictions["method"].isin(YEAR_SPLIT_SCOREBOARD_METHODS)
    ].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame["actual_rounded_high_f"] = _round_half_up_series(frame[TARGET])
    frame["predicted_rounded_high_f"] = _round_half_up_series(frame["predicted_high_f"])
    frame["actual_bracket"] = frame["actual_rounded_high_f"].map(_temperature_bracket_from_rounded)
    frame["predicted_bracket"] = frame["predicted_rounded_high_f"].map(_temperature_bracket_from_rounded)
    missing_bracket = frame["actual_bracket"].isna() | frame["predicted_bracket"].isna()
    frame["bracket_hit"] = frame["actual_bracket"].eq(frame["predicted_bracket"]).mask(missing_bracket).astype("boolean")
    return _sort_year_split_visible_methods(frame[columns]).reset_index(drop=True)


def year_split_bracket_metrics(bracket_predictions: pd.DataFrame) -> pd.DataFrame:
    columns = ["method", "count", "mae_f", "rmse_f", "bracket_accuracy_pct"]
    if bracket_predictions.empty:
        return pd.DataFrame(columns=columns)
    metrics = bracket_predictions.groupby("method", dropna=False).apply(_metric_row, include_groups=False).reset_index()
    bracket_accuracy = (
        bracket_predictions.groupby("method", dropna=False)["bracket_hit"]
        .mean()
        .mul(100)
        .rename("bracket_accuracy_pct")
        .reset_index()
    )
    metrics = metrics.merge(bracket_accuracy, on="method", how="left")
    return _sort_year_split_visible_methods(metrics[columns]).reset_index(drop=True)


def polymarket_temperature_bracket(value: Any) -> str | None:
    rounded = round_temperature_half_up(value)
    if rounded is None:
        return None
    lower = rounded if rounded % 2 == 0 else rounded - 1
    return f"{lower}-{lower + 1}"


def round_temperature_half_up(value: Any) -> int | None:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return None
    return int(math.floor(float(number) + 0.5))


def year_split_feature_importance(
    frame: pd.DataFrame,
    config: StationStackingConfig,
    categorical: list[str],
    numeric: list[str],
    selected_hyperparameters: pd.DataFrame,
    train_years: tuple[int, int] = YEAR_SPLIT_TEST_TRAIN_YEARS,
    test_year: int = YEAR_SPLIT_TEST_YEAR,
) -> pd.DataFrame:
    columns = [
        "method",
        "param_key",
        "feature",
        "importance_mean_mae_f",
        "importance_std_mae_f",
        "n_repeats",
        "train_start_year",
        "train_end_year",
        "test_year",
        "train_rows",
        "test_rows",
    ]
    if frame.empty or selected_hyperparameters.empty:
        return pd.DataFrame(columns=columns)

    from sklearn.inspection import permutation_importance

    year = pd.to_numeric(frame.get("year"), errors="coerce")
    train = frame.loc[year.between(train_years[0], train_years[1])].copy()
    test = frame.loc[year.eq(test_year)].copy()
    if train.empty or test.empty:
        return pd.DataFrame(columns=columns)

    fit_categorical, fit_numeric = _fit_feature_columns(train, categorical, numeric)
    feature_names = [*fit_categorical, *fit_numeric]
    if not feature_names:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for _, row in selected_hyperparameters.iterrows():
        method = str(row["method"])
        params = _params_from_selected_row(row)
        estimator = _build_base_model_pipeline(config, fit_categorical, fit_numeric, method, params)
        try:
            estimator.fit(train[feature_names], train[TARGET])
            importance = permutation_importance(
                estimator,
                test[feature_names],
                test[TARGET],
                scoring="neg_mean_absolute_error",
                n_repeats=config.effective_feature_importance_repeats,
                random_state=config.random_state,
                n_jobs=1,
            )
        except Exception:
            continue
        for index, feature in enumerate(feature_names):
            rows.append(
                {
                    "method": method,
                    "param_key": str(row.get("param_key", "")),
                    "feature": feature,
                    "importance_mean_mae_f": float(importance.importances_mean[index]),
                    "importance_std_mae_f": float(importance.importances_std[index]),
                    "n_repeats": config.effective_feature_importance_repeats,
                    "train_start_year": train_years[0],
                    "train_end_year": train_years[1],
                    "test_year": test_year,
                    "train_rows": len(train),
                    "test_rows": len(test),
                }
            )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["importance_mean_mae_f", "importance_std_mae_f"],
        ascending=[False, False],
        ignore_index=True,
    )


def summarize_year_split_predictions(validation_predictions: pd.DataFrame, test_predictions: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for predictions in [validation_predictions, test_predictions]:
        if predictions.empty:
            continue
        frame = predictions.groupby(["evaluation_scope", "method"], dropna=False).apply(_metric_row, include_groups=False).reset_index()
        frames.append(frame)
    if not frames:
        return pd.DataFrame(
            columns=["evaluation_scope", "method", "count", "mae_f", "rmse_f", "bias_f", "within_1f_pct", "within_2f_pct", "within_3f_pct"]
        )
    return pd.concat(frames, ignore_index=True).sort_values(["evaluation_scope", "rmse_f", "method"]).reset_index(drop=True)


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


def _add_provider_cross_model_features(frame: pd.DataFrame, providers: tuple[str, ...]) -> pd.DataFrame:
    out = frame.copy()
    if len(providers) < 2:
        return out
    provider_pairs = [(left, right) for index, left in enumerate(providers) for right in providers[index + 1 :]]
    feature_map = {
        "high_f": lambda provider: HIGH_COLUMNS[provider],
        "dewpoint_mean_f": lambda provider: f"{provider}_dewpoint_mean_f",
        "humidity_mean": lambda provider: f"{provider}_humidity_mean",
        "wind_speed_mean": lambda provider: f"{provider}_wind_speed_mean",
        "wind_speed_max": lambda provider: f"{provider}_wind_speed_max",
        "wind_gust_max": lambda provider: f"{provider}_wind_gust_max",
        "precip_amount": lambda provider: f"{provider}_precip_amount",
        "grid_dist_km_mean": lambda provider: f"{provider}_grid_dist_km_mean",
    }
    for left, right in provider_pairs:
        prefix = f"{left}_{right}"
        for feature_name, column_for in feature_map.items():
            left_col = column_for(left)
            right_col = column_for(right)
            if left_col not in out or right_col not in out:
                continue
            left_values = pd.to_numeric(out[left_col], errors="coerce")
            right_values = pd.to_numeric(out[right_col], errors="coerce")
            out[f"{prefix}_{feature_name}_diff_f"] = left_values - right_values
            out[f"{prefix}_{feature_name}_abs_diff_f"] = (left_values - right_values).abs()
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
        out[f"{provider}_rolling_bias_14d_f"] = shifted_error.rolling(14, min_periods=3).mean()
        out[f"{provider}_rolling_bias_30d_f"] = shifted_error.rolling(30, min_periods=5).mean()
        out[f"{provider}_rolling_mae_7d_f"] = shifted_abs_error.rolling(7, min_periods=2).mean()
        out[f"{provider}_rolling_mae_14d_f"] = shifted_abs_error.rolling(14, min_periods=3).mean()
        out[f"{provider}_rolling_mae_30d_f"] = shifted_abs_error.rolling(30, min_periods=5).mean()
        out[f"{provider}_high_plus_rolling_bias_7d_f"] = pd.to_numeric(out[high_col], errors="coerce") + out[f"{provider}_rolling_bias_7d_f"]
        out[f"{provider}_high_plus_rolling_bias_14d_f"] = pd.to_numeric(out[high_col], errors="coerce") + out[f"{provider}_rolling_bias_14d_f"]
        out[f"{provider}_high_plus_rolling_bias_30d_f"] = pd.to_numeric(out[high_col], errors="coerce") + out[f"{provider}_rolling_bias_30d_f"]
    return out


def _add_prior_month_provider_error_features(frame: pd.DataFrame, providers: tuple[str, ...]) -> pd.DataFrame:
    out = frame.copy()
    actual = pd.to_numeric(out[TARGET], errors="coerce")
    month = pd.to_numeric(out.get("month"), errors="coerce")
    for provider in providers:
        high_col = HIGH_COLUMNS[provider]
        if high_col not in out:
            continue
        error = actual - pd.to_numeric(out[high_col], errors="coerce")
        abs_error = error.abs()
        prior_month_bias = error.groupby(month, dropna=False).transform(
            lambda series: series.shift(1).expanding(min_periods=2).mean()
        )
        prior_month_mae = abs_error.groupby(month, dropna=False).transform(
            lambda series: series.shift(1).expanding(min_periods=2).mean()
        )
        prior_month_count = error.groupby(month, dropna=False).transform(
            lambda series: series.shift(1).expanding(min_periods=1).count()
        )
        out[f"{provider}_prior_month_bias_f"] = prior_month_bias
        out[f"{provider}_prior_month_mae_f"] = prior_month_mae
        out[f"{provider}_prior_month_error_count"] = prior_month_count
        out[f"{provider}_high_plus_prior_month_bias_f"] = pd.to_numeric(out[high_col], errors="coerce") + prior_month_bias
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


def _add_observation_history_delta_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "observed_temp_at_as_of_f" not in out:
        return out
    observed_temp = pd.to_numeric(out["observed_temp_at_as_of_f"], errors="coerce")
    out["observed_temp_minus_actual_high_lag_1d_f"] = observed_temp - out.get("actual_high_lag_1d")
    out["observed_temp_minus_actual_high_roll_7d_mean_f"] = observed_temp - out.get("actual_high_roll_7d_mean")
    out["observed_temp_minus_actual_high_roll_30d_mean_f"] = observed_temp - out.get("actual_high_roll_30d_mean")
    if "observed_dewpoint_at_as_of_f" in out:
        dewpoint = pd.to_numeric(out["observed_dewpoint_at_as_of_f"], errors="coerce")
        out["observed_dewpoint_minus_actual_high_roll_7d_mean_f"] = dewpoint - out.get("actual_high_roll_7d_mean")
    return out


def _add_observation_forecast_delta_features(frame: pd.DataFrame, providers: tuple[str, ...]) -> pd.DataFrame:
    out = frame.copy()
    observed_temp = pd.to_numeric(out.get("observed_temp_at_as_of_f"), errors="coerce")
    observed_dewpoint = pd.to_numeric(out.get("observed_dewpoint_at_as_of_f"), errors="coerce")
    observed_humidity = pd.to_numeric(out.get("observed_humidity_at_as_of"), errors="coerce")
    observed_wind = pd.to_numeric(out.get("observed_wind_speed_at_as_of"), errors="coerce")
    observed_pressure = pd.to_numeric(out.get("observed_pressure_at_as_of"), errors="coerce")
    for provider in providers:
        high_col = HIGH_COLUMNS[provider]
        if high_col in out:
            out[f"{provider}_high_minus_observed_temp_f"] = pd.to_numeric(out[high_col], errors="coerce") - observed_temp
        if f"{provider}_dewpoint_mean_f" in out:
            out[f"{provider}_dewpoint_minus_observed_dewpoint_f"] = (
                pd.to_numeric(out[f"{provider}_dewpoint_mean_f"], errors="coerce") - observed_dewpoint
            )
        if f"{provider}_humidity_mean" in out:
            out[f"{provider}_humidity_minus_observed_humidity"] = (
                pd.to_numeric(out[f"{provider}_humidity_mean"], errors="coerce") - observed_humidity
            )
        if f"{provider}_wind_speed_mean" in out:
            out[f"{provider}_wind_speed_minus_observed_wind_speed"] = (
                pd.to_numeric(out[f"{provider}_wind_speed_mean"], errors="coerce") - observed_wind
            )
        if f"{provider}_pressure_mslp_mean" in out:
            out[f"{provider}_pressure_minus_observed_pressure"] = (
                pd.to_numeric(out[f"{provider}_pressure_mslp_mean"], errors="coerce") - observed_pressure
            )
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
    return {
        method: _build_base_model_pipeline(config, categorical, numeric, method, params={})
        for method in BASE_MODEL_METHODS
    }


def _build_base_model_pipeline(
    config: StationStackingConfig,
    categorical: list[str],
    numeric: list[str],
    method: str,
    params: dict[str, Any],
):
    require_model_dependencies()
    try:
        from sklearn.pipeline import Pipeline
    except ImportError as exc:
        raise ImportError("Station stacking notebooks need scikit-learn and the gradient boosting packages.") from exc

    return Pipeline(
        [
            ("prep", _build_preprocessor(categorical, numeric)),
            ("model", _build_base_model_estimator(config, method, params)),
        ]
    )


def _build_preprocessor(categorical: list[str], numeric: list[str]):
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
    except ImportError as exc:
        raise ImportError("Station stacking notebooks need scikit-learn.") from exc

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


def _build_base_model_estimator(
    config: StationStackingConfig,
    method: str,
    params: dict[str, Any],
    early_stopping_rounds: int | None = None,
):
    require_model_dependencies()
    try:
        from catboost import CatBoostRegressor
        from lightgbm import LGBMRegressor
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError("Station stacking notebooks need the gradient boosting packages.") from exc

    n_estimators = int(params.get("n_estimators", 120 if config.fast_mode else 900))
    cat_iterations = int(params.get("iterations", 120 if config.fast_mode else 900))
    if method == "xgboost":
        estimator_params = {
            "objective": "reg:squarederror",
            "n_estimators": n_estimators,
            "learning_rate": float(params.get("learning_rate", 0.035)),
            "max_depth": int(params.get("max_depth", 3)),
            "min_child_weight": float(params.get("min_child_weight", 1.0)),
            "gamma": float(params.get("gamma", 0.0)),
            "subsample": float(params.get("subsample", 0.9)),
            "colsample_bytree": float(params.get("colsample_bytree", 0.9)),
            "reg_alpha": float(params.get("reg_alpha", 0.0)),
            "reg_lambda": float(params.get("reg_lambda", 1.0)),
            "random_state": config.random_state,
            "n_jobs": -1,
            "eval_metric": "rmse",
        }
        if early_stopping_rounds is not None:
            estimator_params["early_stopping_rounds"] = early_stopping_rounds
        return XGBRegressor(**estimator_params)
    if method == "lightgbm":
        return LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=float(params.get("learning_rate", 0.035)),
            num_leaves=int(params.get("num_leaves", 31)),
            max_depth=int(params.get("max_depth", -1)),
            min_child_samples=int(params.get("min_child_samples", 20)),
            min_split_gain=float(params.get("min_split_gain", 0.0)),
            bagging_fraction=float(params.get("bagging_fraction", params.get("subsample", 0.9))),
            bagging_freq=int(params.get("bagging_freq", 1)),
            feature_fraction=float(params.get("feature_fraction", params.get("colsample_bytree", 0.9))),
            lambda_l1=float(params.get("lambda_l1", params.get("reg_alpha", 0.0))),
            lambda_l2=float(params.get("lambda_l2", params.get("reg_lambda", 0.0))),
            random_state=config.random_state,
            n_jobs=-1,
            verbose=-1,
        )
    if method == "catboost":
        return CatBoostRegressor(
            iterations=cat_iterations,
            learning_rate=float(params.get("learning_rate", 0.035)),
            depth=int(params.get("depth", 6)),
            l2_leaf_reg=float(params.get("l2_leaf_reg", 3.0)),
            random_strength=float(params.get("random_strength", 1.0)),
            bagging_temperature=float(params.get("bagging_temperature", 1.0)),
            border_count=int(params.get("border_count", 128)),
            rsm=float(params.get("rsm", 1.0)),
            bootstrap_type="Bayesian",
            loss_function="RMSE",
            random_seed=config.random_state,
            verbose=False,
            allow_writing_files=False,
        )
    raise ValueError(f"Unknown base model method: {method}")


def _fit_predict_base_model(
    config: StationStackingConfig,
    categorical: list[str],
    numeric: list[str],
    method: str,
    params: dict[str, Any],
    train: pd.DataFrame,
    valid: pd.DataFrame,
    early_stopping: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    fit_categorical, fit_numeric = _fit_feature_columns(train, categorical, numeric)
    feature_names = [*fit_categorical, *fit_numeric]
    if not feature_names:
        raise ValueError("No non-empty calibration features are available for this fit.")

    preprocessor = _build_preprocessor(fit_categorical, fit_numeric)
    x_train = preprocessor.fit_transform(train[feature_names])
    x_valid = preprocessor.transform(valid[feature_names])
    y_train = pd.to_numeric(train[TARGET], errors="coerce")
    y_valid = pd.to_numeric(valid[TARGET], errors="coerce")
    early_stopping_rounds = _early_stopping_rounds(config) if early_stopping else None
    estimator = _build_base_model_estimator(config, method, params, early_stopping_rounds=early_stopping_rounds)
    _fit_base_estimator(
        estimator=estimator,
        method=method,
        x_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
        y_valid=y_valid,
        early_stopping_rounds=early_stopping_rounds,
    )
    metadata = {
        "numeric_features": ",".join(fit_numeric),
        "categorical_features": ",".join(fit_categorical),
        "best_iteration": _best_iteration(estimator),
    }
    return np.asarray(estimator.predict(x_valid), dtype=float), metadata


def _fit_feature_columns(train: pd.DataFrame, categorical: list[str], numeric: list[str]) -> tuple[list[str], list[str]]:
    fit_categorical = [column for column in categorical if column in train]
    fit_numeric = [column for column in numeric if column in train and pd.to_numeric(train[column], errors="coerce").notna().any()]
    return fit_categorical, fit_numeric


def _fit_base_estimator(
    estimator: Any,
    method: str,
    x_train: Any,
    y_train: pd.Series,
    x_valid: Any,
    y_valid: pd.Series,
    early_stopping_rounds: int | None,
) -> None:
    if early_stopping_rounds is None:
        estimator.fit(x_train, y_train)
        return
    try:
        if method == "xgboost":
            estimator.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
            return
        if method == "lightgbm":
            import lightgbm as lgb

            estimator.fit(
                x_train,
                y_train,
                eval_set=[(x_valid, y_valid)],
                eval_metric="rmse",
                callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
            )
            return
        if method == "catboost":
            estimator.fit(
                x_train,
                y_train,
                eval_set=(x_valid, y_valid),
                early_stopping_rounds=early_stopping_rounds,
                use_best_model=True,
                verbose=False,
            )
            return
    except TypeError:
        pass
    estimator.fit(x_train, y_train)


def _early_stopping_rounds(config: StationStackingConfig) -> int:
    return 20 if config.fast_mode else 50


def _best_iteration(estimator: Any) -> int | None:
    for attr in ("best_iteration", "best_iteration_"):
        value = getattr(estimator, attr, None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    if hasattr(estimator, "get_best_iteration"):
        try:
            value = estimator.get_best_iteration()
            return None if value is None else int(value)
        except (TypeError, ValueError):
            return None
    return None


def _year_split_fold_weight(fold: YearSplitFold) -> float:
    return float(YEAR_SPLIT_VALIDATION_WEIGHTS.get(fold.validation_year, 1.0))


def _weighted_fold_score(fold_scores: list[tuple[YearSplitFold, float]]) -> float:
    if not fold_scores:
        return float("inf")
    weights = np.asarray([_year_split_fold_weight(fold) for fold, _ in fold_scores], dtype=float)
    scores = np.asarray([score for _, score in fold_scores], dtype=float)
    if float(weights.sum()) <= 0:
        return float(np.mean(scores))
    return float(np.average(scores, weights=weights))


def _trial_pruned_exception() -> Exception:
    import optuna

    return optuna.TrialPruned()


def _create_stack_optuna_study(config: StationStackingConfig):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.INFO if config.optuna_verbose else optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=config.random_state + 1000)
    return optuna.create_study(direction="minimize", sampler=sampler)


def _stack_meta_train_valid_split(stack_source: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if stack_source.empty:
        return pd.DataFrame(), pd.DataFrame()
    years = pd.to_datetime(stack_source["contract_date"], errors="coerce").dt.year
    train = stack_source.loc[years.eq(2024)].copy()
    valid = stack_source.loc[years.eq(2025)].copy()
    if not train.empty and not valid.empty:
        return train, valid
    ordered = stack_source.sort_values("contract_date").reset_index(drop=True)
    split_at = max(1, int(len(ordered) * 0.5))
    return ordered.iloc[:split_at].copy(), ordered.iloc[split_at:].copy()


def _stack_features_for_set(feature_set: str) -> list[str]:
    methods = STACK_FEATURE_SETS.get(feature_set)
    if methods is None:
        raise ValueError(f"Unknown stack feature set: {feature_set}")
    return [f"{method}_predicted_high_f" for method in methods]


def _create_optuna_study(config: StationStackingConfig, method: str):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.INFO if config.optuna_verbose else optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=config.random_state + BASE_MODEL_METHODS.index(method))
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    return optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)


def _suggest_hyperparameters(method: str, trial, config: StationStackingConfig) -> dict[str, Any]:
    if method == "xgboost":
        max_estimators = 250 if config.fast_mode else 2000
        return {
            "n_estimators": trial.suggest_int("n_estimators", 80, max_estimators),
            "learning_rate": trial.suggest_float("learning_rate", 0.003, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 1, 8),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.1, 20.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
        }
    if method == "lightgbm":
        max_estimators = 250 if config.fast_mode else 2000
        return {
            "n_estimators": trial.suggest_int("n_estimators", 80, max_estimators),
            "learning_rate": trial.suggest_float("learning_rate", 0.003, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 4, 256),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 150),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 2.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 50.0, log=True),
        }
    if method == "catboost":
        max_iterations = 250 if config.fast_mode else 2000
        return {
            "iterations": trial.suggest_int("iterations", 80, max_iterations),
            "learning_rate": trial.suggest_float("learning_rate", 0.003, 0.15, log=True),
            "depth": trial.suggest_int("depth", 2, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 50.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 10.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "rsm": trial.suggest_float("rsm", 0.5, 1.0),
        }
    raise ValueError(f"Unknown base model method: {method}")


def _selected_hyperparameters(tuning: pd.DataFrame) -> pd.DataFrame:
    if tuning.empty:
        return pd.DataFrame(columns=["method", "param_key", "mean_validation_rmse_f"])
    ok = tuning.loc[tuning["status"].eq("ok")].copy()
    if ok.empty:
        return pd.DataFrame(columns=["method", "param_key", "mean_validation_rmse_f"])
    if "fold" in ok:
        fold_counts = ok.groupby(["method", "param_key"], dropna=False)["fold"].nunique()
        complete_fold_count = int(fold_counts.max()) if not fold_counts.empty else 0
        if complete_fold_count > 1:
            complete_keys = fold_counts.loc[fold_counts.eq(complete_fold_count)].index
            ok = ok.set_index(["method", "param_key"]).loc[complete_keys].reset_index()
    if "fold_weight" not in ok:
        ok["fold_weight"] = 1.0

    def weighted_metrics(group: pd.DataFrame) -> pd.Series:
        weights = pd.to_numeric(group["fold_weight"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
        if weights.sum() <= 0:
            weights = np.ones(len(group), dtype=float)
        return pd.Series(
            {
                "mean_validation_rmse_f": float(np.average(pd.to_numeric(group["rmse_f"], errors="coerce"), weights=weights)),
                "mean_validation_mae_f": float(np.average(pd.to_numeric(group["mae_f"], errors="coerce"), weights=weights)),
            }
        )

    grouped = ok.groupby(["method", "param_key"], dropna=False).apply(weighted_metrics, include_groups=False).reset_index()
    grouped = grouped.sort_values(["method", "mean_validation_rmse_f", "param_key"])
    selected = grouped.groupby("method", dropna=False).head(1).reset_index(drop=True)
    param_columns = [column for column in tuning.columns if column.startswith("param_") and column != "param_key"]
    params = ok[["method", "param_key", *param_columns]].drop_duplicates(["method", "param_key"])
    return selected.merge(params, on=["method", "param_key"], how="left")


def _params_from_selected_row(row: pd.Series) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for key, value in row.items():
        if not str(key).startswith("param_") or pd.isna(value):
            continue
        params[str(key).removeprefix("param_")] = value
    return params


def _filter_predictions_to_selected_params(predictions: list[pd.DataFrame], selected: pd.DataFrame) -> pd.DataFrame:
    if not predictions or selected.empty:
        return _empty_year_split_predictions()
    selected_keys = set(zip(selected["method"].astype(str), selected["param_key"].astype(str), strict=False))
    frames = []
    for frame in predictions:
        if frame.empty or "param_key" not in frame:
            continue
        keys = list(zip(frame["method"].astype(str), frame["param_key"].astype(str), strict=False))
        mask = [key in selected_keys for key in keys]
        if any(mask):
            frames.append(frame.loc[mask].copy())
    return pd.concat(frames, ignore_index=True) if frames else _empty_year_split_predictions()


def _validation_predictions_for_selected_params(
    frame: pd.DataFrame,
    config: StationStackingConfig,
    categorical: list[str],
    numeric: list[str],
    folds: tuple[YearSplitFold, ...],
    selected: pd.DataFrame,
) -> pd.DataFrame:
    if frame.empty or selected.empty:
        return _empty_year_split_predictions()
    year = pd.to_numeric(frame.get("year"), errors="coerce")
    rows: list[pd.DataFrame] = []
    for _, selected_row in selected.iterrows():
        method = str(selected_row["method"])
        params = _params_from_selected_row(selected_row)
        for fold in folds:
            train = frame.loc[year.between(fold.train_start_year, fold.train_end_year)].copy()
            valid = frame.loc[year.eq(fold.validation_year)].copy()
            if train.empty or valid.empty:
                continue
            try:
                predicted, _ = _fit_predict_base_model(
                    config=config,
                    categorical=categorical,
                    numeric=numeric,
                    method=method,
                    params=params,
                    train=train,
                    valid=valid,
                    early_stopping=False,
                )
            except Exception:
                continue
            pred = valid[["contract_date", TARGET]].copy()
            pred["method"] = method
            pred["param_key"] = str(selected_row["param_key"])
            pred["predicted_high_f"] = predicted
            pred["evaluation_scope"] = "year_split_validation"
            pred["fold"] = fold.name
            rows.append(_year_split_prediction_columns(pred))
    return pd.concat(rows, ignore_index=True) if rows else _empty_year_split_predictions()


def _year_split_best_raw_provider(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    config: StationStackingConfig,
    fold: str,
    evaluation_scope: str,
) -> pd.DataFrame:
    if train.empty or valid.empty:
        return _empty_year_split_predictions()
    mae_by_provider = {
        provider: float((train[TARGET] - pd.to_numeric(train[HIGH_COLUMNS[provider]], errors="coerce")).abs().mean())
        for provider in config.providers
        if HIGH_COLUMNS[provider] in train
    }
    if not mae_by_provider:
        return _empty_year_split_predictions()
    best_provider = min(mae_by_provider, key=mae_by_provider.get)
    column = HIGH_COLUMNS[best_provider]
    pred = valid.loc[valid[column].notna(), ["contract_date", TARGET, column]].copy()
    if pred.empty:
        return _empty_year_split_predictions()
    pred["method"] = "best_raw_provider"
    pred["predicted_high_f"] = pred[column]
    pred["evaluation_scope"] = evaluation_scope
    pred["fold"] = fold
    return _year_split_prediction_columns(pred)


def _year_split_prediction_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = _prediction_columns(frame)
    out["fold"] = frame.get("fold", pd.Series(pd.NA, index=frame.index)).to_numpy()
    out["param_key"] = frame.get("param_key", pd.Series(pd.NA, index=frame.index)).to_numpy()
    return out[
        [
            "contract_date",
            "fold",
            "method",
            "param_key",
            "evaluation_scope",
            TARGET,
            "predicted_high_f",
            "error_f",
            "absolute_error_f",
        ]
    ]


def _empty_year_split_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "contract_date",
            "fold",
            "method",
            "param_key",
            "evaluation_scope",
            TARGET,
            "predicted_high_f",
            "error_f",
            "absolute_error_f",
        ]
    )


def _year_split_stack_source_frame(predictions: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    combined = predictions.loc[predictions["method"].isin(methods)].copy()
    if combined.empty:
        return pd.DataFrame()
    pivot = combined.pivot_table(
        index="contract_date",
        columns="method",
        values="predicted_high_f",
        aggfunc="first",
    )
    pivot.columns = [f"{column}_predicted_high_f" for column in pivot.columns]
    required = [f"{method}_predicted_high_f" for method in methods]
    if any(column not in pivot for column in required):
        return pd.DataFrame()
    actuals = combined.groupby("contract_date", dropna=False)[TARGET].first()
    return pivot.join(actuals).reset_index()


def _round_half_up_series(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return np.floor(values + 0.5).astype("Int64")


def _temperature_bracket_from_rounded(value: Any) -> Any:
    if pd.isna(value):
        return pd.NA
    rounded = int(value)
    lower = rounded if rounded % 2 == 0 else rounded - 1
    return f"{lower}-{lower + 1}"


def _sort_year_split_visible_methods(frame: pd.DataFrame, include_period: bool = False) -> pd.DataFrame:
    out = frame.copy()
    method_order = {method: index for index, method in enumerate(YEAR_SPLIT_SCOREBOARD_METHODS)}
    sort_columns = []
    if include_period and "period" in out:
        period_order = {"validation_2024_2025": 0, f"test_{YEAR_SPLIT_TEST_YEAR}": 1}
        out["_period_order"] = out["period"].map(period_order).fillna(len(period_order))
        sort_columns.append("_period_order")
    out["_method_order"] = out["method"].map(method_order).fillna(len(method_order))
    sort_columns.append("_method_order")
    if "contract_date" in out:
        sort_columns.append("contract_date")
    out = out.sort_values(sort_columns).drop(columns=[column for column in ["_period_order", "_method_order"] if column in out])
    return out.reset_index(drop=True)


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
