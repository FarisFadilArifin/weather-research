from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.calibration.station_stacking import (
    BASE_MODEL_METHODS,
    OBSERVED_HIGH_SO_FAR_COLUMN,
    REMAINING_WARMUP_TARGET,
    STACK_FEATURE_SETS,
    STACK_METHOD,
    TARGET,
    TARGET_MODE_DIRECT_HIGH,
    TARGET_STATIONS,
    TRAINING_PROFILE_LEGACY,
    StationStackingConfig,
    _build_base_model_pipeline,
    _fit_feature_columns,
    _model_target_column,
    _model_target_values,
    _modeling_frame,
    _params_from_selected_row,
    _select_stack_tuning_candidate,
    _stack_features_for_set,
    _year_split_fold_weight,
    _year_split_stack_source_frame,
)


MODEL_VERSION = "station_high_regressor_v2"
DEFAULT_ARTIFACT_DIR = Path("data") / "calibration" / "station_stacking_v2"
DEFAULT_MODEL_DIR_NAME = "model_weights"
DEFAULT_TIMING_MODE = "same_day_11am"
DEFAULT_PROVIDERS = ("gfs", "hrrr")
DEFAULT_FEATURE_VERSION = "base"
DEFAULT_TRAINING_PROFILE = TRAINING_PROFILE_LEGACY
DEFAULT_OPTUNA_METRIC = "rmse_f"
DEFAULT_TARGET_MODE = TARGET_MODE_DIRECT_HIGH
DEFAULT_BASE_MODEL_METHODS = tuple(BASE_MODEL_METHODS)
DEFAULT_SOURCE_PIPELINE = "notebooks/station_stacking_v2"


@dataclass(frozen=True)
class ExportedModelWeights:
    station_id: str
    bundle_path: Path
    manifest_path: Path


def export_station_model_weights(
    project_root: str | Path = ".",
    station_id: str = "KSEA",
    artifact_dir: str | Path | None = None,
    model_dir: str | Path | None = None,
    train_years: tuple[int, int] | None = None,
    model_version: str = MODEL_VERSION,
    timing_mode: str = DEFAULT_TIMING_MODE,
    providers: tuple[str, ...] = DEFAULT_PROVIDERS,
    feature_version: str = DEFAULT_FEATURE_VERSION,
    training_profile: str = DEFAULT_TRAINING_PROFILE,
    optuna_metric: str = DEFAULT_OPTUNA_METRIC,
    target_mode: str = DEFAULT_TARGET_MODE,
    target_source: str = "iem_hourly",
    base_model_methods: tuple[str, ...] = DEFAULT_BASE_MODEL_METHODS,
    stack_enabled: bool = True,
    source_pipeline: str = DEFAULT_SOURCE_PIPELINE,
    feature_pipeline: str | None = None,
    selected_guarded_cap_f: float | None = None,
    baseline_comparison: dict[str, Any] | None = None,
) -> ExportedModelWeights:
    root = Path(project_root).resolve()
    station = station_id.upper()
    artifacts = _resolve_under_root(root, DEFAULT_ARTIFACT_DIR if artifact_dir is None else Path(artifact_dir))
    output_dir = artifacts / DEFAULT_MODEL_DIR_NAME if model_dir is None else _resolve_under_root(root, Path(model_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    features = _read_required_csv(artifacts / f"{station}_features.csv")
    selected = _read_required_csv(artifacts / f"{station}_year_split_selected_hyperparameters.csv")
    validation_predictions = _read_required_csv(artifacts / f"{station}_year_split_validation_predictions.csv")
    stack_tuning = (
        _read_required_csv(artifacts / f"{station}_year_split_stack_tuning.csv")
        if stack_enabled
        else _read_optional_csv(artifacts / f"{station}_year_split_stack_tuning.csv")
    )
    methods = _validated_base_model_methods(base_model_methods)

    config = StationStackingConfig(
        station_id=station,
        project_root=root,
        timing_mode=timing_mode,
        providers=providers,
        output_dir=artifacts,
        feature_version=feature_version,
        training_profile=training_profile,
        optuna_metric=optuna_metric,
        target_mode=target_mode,
        target_source=target_source,
        base_model_methods=methods,
        stack_enabled=stack_enabled,
    )
    modeling_frame, categorical, numeric = _modeling_frame(features, config)
    if modeling_frame.empty:
        raise ValueError(f"No usable modeling rows found for {station}.")

    year = pd.to_numeric(modeling_frame.get("year"), errors="coerce")
    if train_years is None:
        train = modeling_frame.copy()
        training_mode = "production_refit_all_available_actuals"
        train_start_year = int(year.min())
        train_end_year = int(year.max())
    else:
        train_start_year, train_end_year = train_years
        train = modeling_frame.loc[year.between(train_start_year, train_end_year)].copy()
        training_mode = f"refit_train_{train_start_year}_{train_end_year}"
    if train.empty:
        raise ValueError(f"No training rows available for {station} in requested year range.")

    fit_categorical, fit_numeric = _fit_feature_columns(train, categorical, numeric)
    feature_names = [*fit_categorical, *fit_numeric]
    if not feature_names:
        raise ValueError(f"No non-empty feature columns available for {station}.")

    base_models: dict[str, Any] = {}
    base_model_manifests: list[dict[str, Any]] = []
    selected_by_method = {
        str(row["method"]): row
        for _, row in selected.iterrows()
        if str(row.get("method", "")) in methods
    }
    missing_methods = [method for method in methods if method not in selected_by_method]
    if missing_methods:
        raise ValueError(f"{station} selected hyperparameters missing methods: {missing_methods}")

    for method in methods:
        row = selected_by_method[method]
        params = _params_from_selected_row(row)
        estimator = _build_base_model_pipeline(config, fit_categorical, fit_numeric, method, params)
        estimator.fit(train[feature_names], _model_target_values(train, config))
        base_models[method] = estimator
        base_model_manifests.append(
            {
                "method": method,
                "param_key": str(row.get("param_key", "")),
                "mean_validation_rmse_f": _jsonable(row.get("mean_validation_rmse_f")),
                "mean_validation_mae_f": _jsonable(row.get("mean_validation_mae_f")),
                "mean_validation_bucket_log_loss": _jsonable(row.get("mean_validation_bucket_log_loss")),
                "params": _jsonable(params),
            }
        )

    if stack_enabled:
        stack_model, stack_manifest = _fit_stack_model(
            validation_predictions,
            stack_tuning,
            metric_col=config.effective_optuna_metric,
            base_model_methods=methods,
            providers=config.providers,
            training_profile=config.effective_training_profile,
        )
        final_model_method = STACK_METHOD
    else:
        stack_model, stack_manifest = None, _disabled_stack_manifest()
        final_model_method = methods[0]
    residual_calibrator = (
        _ridge_residual_calibrator(validation_predictions, stack_model, stack_manifest["features"], methods)
        if stack_enabled
        else _empty_residual_calibrator()
    )
    bucket_probability_policy = _bucket_probability_policy(residual_calibrator)

    bundle = {
        "schema_version": 1,
        "model_version": model_version,
        "station_id": station,
        "target": TARGET,
        "target_mode": config.effective_target_mode,
        "target_source": config.effective_target_source,
        "model_target": _model_target_column(config),
        "observed_high_so_far_column": OBSERVED_HIGH_SO_FAR_COLUMN,
        "training_mode": training_mode,
        "base_model_methods": tuple(methods),
        "stack_enabled": bool(stack_enabled),
        "final_model_method": final_model_method,
        "base_models": base_models,
        "stack_model": stack_model,
        "stack_features": stack_manifest["features"],
        "categorical_features": fit_categorical,
        "numeric_features": fit_numeric,
        "feature_names": feature_names,
        "providers": tuple(config.providers),
        "timing_mode": config.timing_mode,
        "feature_version": config.effective_feature_version,
        "training_profile": config.effective_training_profile,
        "optuna_metric": config.effective_optuna_metric,
        "bucket_probability_policy": bucket_probability_policy,
        "residual_calibrator": residual_calibrator,
    }

    bundle_path = output_dir / f"{station}_{model_version}.joblib"
    manifest_path = output_dir / f"{station}_{model_version}.json"
    _dump_joblib(bundle, bundle_path)
    bundle_sha256 = _sha256_file(bundle_path)
    resolved_feature_pipeline = feature_pipeline or _feature_pipeline_name(
        config.effective_feature_version
    )

    manifest = {
        "schema_version": 1,
        "artifact_type": "station_high_regression_model_weights",
        "model_version": model_version,
        "station_id": station,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_pipeline": source_pipeline,
        "source_artifact_dir": str(artifacts),
        "bundle_path": str(bundle_path),
        "artifact_integrity": {
            "bundle_sha256": bundle_sha256,
        },
        "source_identity": _git_identity(root),
        "package_runtime_compatibility": {
            "runtime_contract": "requirements-ml-runtime.txt",
            "python": ">=3.11",
            "feature_pipeline": resolved_feature_pipeline,
            "package_versions": _runtime_package_versions(),
        },
        "model_contract": {
            "timing_mode": config.timing_mode,
            "providers": list(config.providers),
            "feature_version": config.effective_feature_version,
            "training_profile": config.effective_training_profile,
            "optuna_metric": config.effective_optuna_metric,
            "target_mode": config.effective_target_mode,
            "target_source": config.effective_target_source,
            "model_target": _model_target_column(config),
            "base_model_methods": list(methods),
            "stack_enabled": bool(stack_enabled),
            "final_model_method": final_model_method,
        },
        "training": {
            "mode": training_mode,
            "train_start_year": train_start_year,
            "train_end_year": train_end_year,
            "train_rows": int(len(train)),
            "first_contract_date": str(train["contract_date"].min()),
            "last_contract_date": str(train["contract_date"].max()),
            "target": TARGET,
            "model_target": _model_target_column(config),
            "target_source": config.effective_target_source,
        },
        "training_validation": {
            "training_profile": config.effective_training_profile,
            "base_fold_policy": (
                "four_equal_weight_expanding_folds_2022_2025"
                if config.effective_training_profile != TRAINING_PROFILE_LEGACY
                else "configured_legacy_folds"
            ),
            "base_folds": [
                {
                    "name": fold.name,
                    "train_start_year": fold.train_start_year,
                    "train_end_year": fold.train_end_year,
                    "validation_year": fold.validation_year,
                    "weight": _year_split_fold_weight(fold, config),
                }
                for fold in config.effective_year_split_folds
            ],
            "stack_validation_mode": (
                "three_expanding_meta_folds_2023_2025"
                if config.effective_training_profile != TRAINING_PROFILE_LEGACY
                else "single_latest_meta_year"
            ),
            "selected_ridge_trial": stack_manifest.get("param_key"),
            "selected_ridge_trial_number": stack_manifest.get("trial_number"),
            "stack_mean_selection_metric": stack_manifest.get("validation_mean_selection_metric"),
            "stack_worst_selection_metric": stack_manifest.get("validation_worst_selection_metric"),
            "stack_fold_count": stack_manifest.get("validation_fold_count", 0),
            "stack_selection_rule": stack_manifest.get("selection_rule"),
        },
        "features": {
            "categorical": fit_categorical,
            "numeric": fit_numeric,
            "all": feature_names,
        },
        "base_models": base_model_manifests,
        "stack_model": stack_manifest,
        "bucket_probability_policy": bucket_probability_policy,
        "residual_calibrator": residual_calibrator,
        "inference": {
            "primary_output": "predictedHighF",
            "secondary_output": "bucketProbabilities",
            "final_model_method": final_model_method,
            "base_prediction_inputs": [f"{method}_predicted_high_f" for method in methods],
            "base_model_raw_output": _model_target_column(config),
            "remaining_warmup_target": REMAINING_WARMUP_TARGET,
            "observed_high_so_far_column": OBSERVED_HIGH_SO_FAR_COLUMN,
            "base_prediction_transform": _base_prediction_transform(config),
            "provider_high_inputs": [f"{provider}_high_f" for provider in config.providers],
            "stack_raw_forecast_inputs": [
                feature.removesuffix("_predicted_high_f")
                for feature in stack_manifest["features"]
                if feature.endswith("_raw_predicted_high_f")
            ],
            "point_in_time_rule": _point_in_time_rule(config),
        },
        "v12_guarded_blend": {
            "selected_cap_f": selected_guarded_cap_f,
            "baseline_comparison": baseline_comparison or {},
            "candidate_only_until_handoff_acceptance": True,
        },
    }
    manifest_path.write_text(json.dumps(_jsonable(manifest), indent=2) + "\n", encoding="utf-8")
    return ExportedModelWeights(station_id=station, bundle_path=bundle_path, manifest_path=manifest_path)


def export_all_station_model_weights(
    project_root: str | Path = ".",
    stations: tuple[str, ...] = TARGET_STATIONS,
    artifact_dir: str | Path | None = None,
    model_dir: str | Path | None = None,
    train_years: tuple[int, int] | None = None,
    model_version: str = MODEL_VERSION,
    timing_mode: str = DEFAULT_TIMING_MODE,
    providers: tuple[str, ...] = DEFAULT_PROVIDERS,
    feature_version: str = DEFAULT_FEATURE_VERSION,
    training_profile: str = DEFAULT_TRAINING_PROFILE,
    optuna_metric: str = DEFAULT_OPTUNA_METRIC,
    target_mode: str = DEFAULT_TARGET_MODE,
    target_source: str = "iem_hourly",
    base_model_methods: tuple[str, ...] = DEFAULT_BASE_MODEL_METHODS,
    stack_enabled: bool = True,
    source_pipeline: str = DEFAULT_SOURCE_PIPELINE,
    feature_pipeline: str | None = None,
    selected_guarded_cap_f: float | None = None,
    baseline_comparison: dict[str, Any] | None = None,
) -> list[ExportedModelWeights]:
    exports = [
        export_station_model_weights(
            project_root=project_root,
            station_id=station,
            artifact_dir=artifact_dir,
            model_dir=model_dir,
            train_years=train_years,
            model_version=model_version,
            timing_mode=timing_mode,
            providers=providers,
            feature_version=feature_version,
            training_profile=training_profile,
            optuna_metric=optuna_metric,
            target_mode=target_mode,
            target_source=target_source,
            base_model_methods=base_model_methods,
            stack_enabled=stack_enabled,
            source_pipeline=source_pipeline,
            feature_pipeline=feature_pipeline,
            selected_guarded_cap_f=selected_guarded_cap_f,
            baseline_comparison=baseline_comparison,
        )
        for station in stations
    ]
    _write_export_index(exports, model_version=model_version)
    return exports


def _fit_stack_model(
    validation_predictions: pd.DataFrame,
    stack_tuning: pd.DataFrame,
    metric_col: str = DEFAULT_OPTUNA_METRIC,
    base_model_methods: tuple[str, ...] = DEFAULT_BASE_MODEL_METHODS,
    providers: tuple[str, ...] = DEFAULT_PROVIDERS,
    training_profile: str = DEFAULT_TRAINING_PROFILE,
) -> tuple[Any, dict[str, Any]]:
    profile_config = StationStackingConfig(station_id="EXPORT", training_profile=training_profile)
    selected, selected_rows, selection_summary = _select_stack_tuning_candidate(
        stack_tuning,
        metric_col,
        aggregate_folds=profile_config.effective_training_profile != TRAINING_PROFILE_LEGACY,
    )
    feature_set = str(selected["feature_set"])
    available_methods = set(validation_predictions.get("method", pd.Series(dtype=str)).astype(str))
    ordered_providers = tuple(
        provider
        for provider in dict.fromkeys(("hrrr", "gfs", *providers))
        if f"{provider}_raw" in available_methods
    )
    stack_features = _stack_features_for_set(feature_set, base_model_methods, ordered_providers)
    stack_methods = [feature.removesuffix("_predicted_high_f") for feature in stack_features]
    train_source = _year_split_stack_source_frame(validation_predictions, stack_methods)
    train = train_source.dropna(subset=[*stack_features, TARGET]).copy()
    if train.empty:
        raise ValueError("No complete rows are available for fitting the ridge stack.")

    from sklearn.linear_model import Ridge

    model = Ridge(alpha=float(selected["alpha"]), fit_intercept=_coerce_bool(selected["fit_intercept"]))
    model.fit(train[stack_features], train[TARGET])
    manifest = {
        "method": STACK_METHOD,
        "param_key": str(selected["param_key"]),
        "trial_number": _jsonable(selected.get("trial_number")),
        "feature_set": feature_set,
        "features": stack_features,
        "alpha": float(selected["alpha"]),
        "fit_intercept": _coerce_bool(selected["fit_intercept"]),
        "validation_rmse_f": _selected_metric_mean(selected_rows, "rmse_f"),
        "validation_mae_f": _selected_metric_mean(selected_rows, "mae_f"),
        "validation_bucket_log_loss": _selected_metric_mean(selected_rows, "bucket_log_loss"),
        "validation_fold_count": selection_summary["fold_count"],
        "validation_mean_selection_metric": selection_summary["mean_metric"],
        "validation_worst_selection_metric": selection_summary["worst_metric"],
        "selection_rule": selection_summary["selection_rule"],
        "selection_metric": metric_col,
        "meta_train_rows": int(len(train)),
        "meta_train_first_contract_date": str(train["contract_date"].min()),
        "meta_train_last_contract_date": str(train["contract_date"].max()),
    }
    return model, manifest


def _selected_metric_mean(selected_rows: pd.DataFrame, metric_col: str) -> Any:
    if metric_col not in selected_rows:
        return None
    values = pd.to_numeric(selected_rows[metric_col], errors="coerce")
    return _jsonable(values.mean())


def _disabled_stack_manifest() -> dict[str, Any]:
    return {
        "method": None,
        "param_key": "",
        "feature_set": None,
        "features": [],
        "selection_metric": None,
        "meta_train_rows": 0,
    }


def _ridge_residual_calibrator(
    validation_predictions: pd.DataFrame,
    stack_model: Any,
    stack_features: list[str],
    base_model_methods: tuple[str, ...],
) -> dict[str, Any]:
    if validation_predictions.empty or stack_model is None or not stack_features:
        return _empty_residual_calibrator()
    stack_methods = [*base_model_methods, "hrrr_raw", "gfs_raw"]
    source = _year_split_stack_source_frame(validation_predictions, stack_methods)
    if source.empty or any(column not in source for column in stack_features):
        return _empty_residual_calibrator()
    source = source.dropna(subset=[*stack_features, TARGET]).copy()
    if source.empty:
        return _empty_residual_calibrator()
    predicted = pd.to_numeric(pd.Series(stack_model.predict(source[stack_features]), index=source.index), errors="coerce")
    actual = pd.to_numeric(source[TARGET], errors="coerce")
    residual = (actual - predicted).dropna()
    if residual.empty:
        return _empty_residual_calibrator()
    error_std = float(residual.std(ddof=0)) if len(residual) >= 2 else float("nan")
    if not np.isfinite(error_std) or error_std <= 0:
        mae = float(residual.abs().mean()) if residual.notna().any() else 1.0
        error_std = max(0.75, mae if np.isfinite(mae) and mae > 0 else 1.0)
    return {
        "method": STACK_METHOD,
        "source": "validation_predictions",
        "error_mean_f": float(residual.mean()),
        "error_std_f": max(0.25, error_std),
        "row_count": int(len(residual)),
        "first_contract_date": str(source.loc[residual.index, "contract_date"].min()),
        "last_contract_date": str(source.loc[residual.index, "contract_date"].max()),
    }


def _empty_residual_calibrator() -> dict[str, Any]:
    return {
        "method": STACK_METHOD,
        "source": "unavailable",
        "error_mean_f": 0.0,
        "error_std_f": 2.0,
        "row_count": 0,
        "first_contract_date": None,
        "last_contract_date": None,
    }


def _bucket_probability_policy(residual_calibrator: dict[str, Any]) -> dict[str, Any]:
    return {
        "enabled": True,
        "method": "normal_residual_interval_probability",
        "bucket_rounding": "polymarket_half_up_2f",
        "continuity_correction_f": 0.5,
        "residual_calibrator_method": residual_calibrator.get("method", STACK_METHOD),
        "live_output": "bucketProbabilities",
    }


def _read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required station-stacking artifact: {path}")
    frame = pd.read_csv(path, low_memory=False)
    if frame.empty:
        raise ValueError(f"Required station-stacking artifact is empty: {path}")
    return frame


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _feature_pipeline_name(feature_version: str) -> str:
    if feature_version == "v20_peak_timing":
        return "station_stacking_v20_peak_timing"
    if feature_version == "v11_settlement_fix_temp":
        return "station_stacking_v11_settlement_fix"
    if feature_version.startswith("v11"):
        return "station_stacking_v11"
    return f"station_stacking_{feature_version}"


def _runtime_package_versions() -> dict[str, str | None]:
    packages = {
        "catboost": "catboost",
        "joblib": "joblib",
        "lightgbm": "lightgbm",
        "numpy": "numpy",
        "pandas": "pandas",
        "scikit-learn": "scikit-learn",
        "xgboost": "xgboost",
    }
    out: dict[str, str | None] = {}
    for label, distribution in packages.items():
        try:
            out[label] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            out[label] = None
    return out


def _git_identity(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(root), "status", "--porcelain"], text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit = None
        dirty = None
    return {"git_commit": commit, "git_dirty": dirty}


def _validated_base_model_methods(methods: tuple[str, ...]) -> tuple[str, ...]:
    validated: list[str] = []
    for raw_method in methods:
        method = str(raw_method).strip().lower()
        if not method:
            continue
        if method not in BASE_MODEL_METHODS:
            raise ValueError(f"base_model_methods must be drawn from: {', '.join(BASE_MODEL_METHODS)}")
        if method not in validated:
            validated.append(method)
    if not validated:
        raise ValueError("base_model_methods must include at least one supported model method")
    return tuple(validated)


def _point_in_time_rule(config: StationStackingConfig) -> str:
    providers = "/".join(str(provider).upper() for provider in config.providers)
    if config.effective_feature_version in {"v7", "v8"}:
        return (
            f"{providers} {config.timing_mode}; direct NBM included when provider is nbm; "
            "current-observation features and morning trends must be at or before 11:00 AM local."
        )
    return (
        f"{providers} {config.timing_mode} plus current observation features from "
        f"feature_version={config.effective_feature_version} feature table."
    )


def _base_prediction_transform(config: StationStackingConfig) -> str:
    if config.effective_target_mode == TARGET_MODE_DIRECT_HIGH:
        return "identity"
    return f"predicted_high_f=max({OBSERVED_HIGH_SO_FAR_COLUMN}, {OBSERVED_HIGH_SO_FAR_COLUMN}+model_output)"


def _dump_joblib(value: Any, path: Path) -> None:
    try:
        import joblib
    except ImportError as exc:
        raise ImportError("Exporting station-stacking weights requires joblib.") from exc

    joblib.dump(value, path)


def _write_export_index(exports: list[ExportedModelWeights], model_version: str = MODEL_VERSION) -> None:
    if not exports:
        return
    output_dir = exports[0].bundle_path.parent
    rows = [
        {
            "station_id": item.station_id,
            "bundle_path": str(item.bundle_path),
            "manifest_path": str(item.manifest_path),
        }
        for item in exports
    ]
    pd.DataFrame(rows).to_csv(output_dir / f"{model_version}_index.csv", index=False)


def _resolve_under_root(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return bool(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if pd.isna(value):
        return None
    return value


def _parse_train_years(value: str | None) -> tuple[int, int] | None:
    if value is None or value.strip().lower() in {"", "all", "all_available"}:
        return None
    parts = value.split("-", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Use YEAR-YEAR, for example 2021-2025, or all_available.")
    return int(parts[0]), int(parts[1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Export station-stacking model weights.")
    parser.add_argument("--project-root", default=".", help="Project root. Defaults to current directory.")
    parser.add_argument(
        "--stations",
        nargs="+",
        default=list(TARGET_STATIONS),
        help="Station IDs to export. Defaults to all target stations.",
    )
    parser.add_argument(
        "--artifact-dir",
        default=None,
        help="Directory containing station-stacking CSV artifacts.",
    )
    parser.add_argument("--model-dir", default=None, help="Output directory for model weights.")
    parser.add_argument("--model-version", default=MODEL_VERSION, help="Model version string for exported files.")
    parser.add_argument("--timing-mode", default=DEFAULT_TIMING_MODE, help="Timing mode recorded in the bundle.")
    parser.add_argument(
        "--providers",
        nargs="+",
        default=list(DEFAULT_PROVIDERS),
        help="Forecast providers required by this bundle.",
    )
    parser.add_argument("--feature-version", default=DEFAULT_FEATURE_VERSION, help="Feature version recorded in the bundle.")
    parser.add_argument("--training-profile", default=DEFAULT_TRAINING_PROFILE, help="Training/validation profile recorded in the bundle.")
    parser.add_argument("--optuna-metric", default=DEFAULT_OPTUNA_METRIC, help="Metric used to select stack tuning rows.")
    parser.add_argument("--target-mode", default=DEFAULT_TARGET_MODE, help="Base-model target mode recorded in the bundle.")
    parser.add_argument("--target-source", default="iem_hourly", help="Target source recorded in the bundle.")
    parser.add_argument(
        "--base-model-methods",
        nargs="+",
        default=list(DEFAULT_BASE_MODEL_METHODS),
        help="Base model methods to export.",
    )
    parser.add_argument("--disable-stack", action="store_true", help="Export base model only without ridge stack.")
    parser.add_argument("--source-pipeline", default=DEFAULT_SOURCE_PIPELINE, help="Notebook or pipeline source label.")
    parser.add_argument(
        "--train-years",
        default="all_available",
        help="Training year range such as 2021-2025, or all_available for all completed rows.",
    )
    args = parser.parse_args()
    exports = export_all_station_model_weights(
        project_root=args.project_root,
        stations=tuple(station.upper() for station in args.stations),
        artifact_dir=args.artifact_dir,
        model_dir=args.model_dir,
        train_years=_parse_train_years(args.train_years),
        model_version=args.model_version,
        timing_mode=args.timing_mode,
        providers=tuple(provider.lower() for provider in args.providers),
        feature_version=args.feature_version,
        training_profile=args.training_profile,
        optuna_metric=args.optuna_metric,
        target_mode=args.target_mode,
        target_source=args.target_source,
        base_model_methods=tuple(args.base_model_methods),
        stack_enabled=not args.disable_stack,
        source_pipeline=args.source_pipeline,
    )
    for item in exports:
        print(f"{item.station_id}: {item.bundle_path}")


if __name__ == "__main__":
    main()
