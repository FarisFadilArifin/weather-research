from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.calibration.station_stacking import (
    BASE_MODEL_METHODS,
    STACK_FEATURE_SETS,
    STACK_METHOD,
    TARGET,
    TARGET_STATIONS,
    YEAR_SPLIT_TEST_TRAIN_YEARS,
    StationStackingConfig,
    _build_base_model_pipeline,
    _fit_feature_columns,
    _modeling_frame,
    _params_from_selected_row,
    _stack_features_for_set,
    _year_split_stack_source_frame,
)


MODEL_VERSION = "station_high_regressor_v2"
DEFAULT_ARTIFACT_DIR = Path("data") / "calibration" / "station_stacking_v2"
DEFAULT_MODEL_DIR_NAME = "model_weights"


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
) -> ExportedModelWeights:
    root = Path(project_root).resolve()
    station = station_id.upper()
    artifacts = _resolve_under_root(root, DEFAULT_ARTIFACT_DIR if artifact_dir is None else Path(artifact_dir))
    output_dir = artifacts / DEFAULT_MODEL_DIR_NAME if model_dir is None else _resolve_under_root(root, Path(model_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    features = _read_required_csv(artifacts / f"{station}_features.csv")
    selected = _read_required_csv(artifacts / f"{station}_year_split_selected_hyperparameters.csv")
    validation_predictions = _read_required_csv(artifacts / f"{station}_year_split_validation_predictions.csv")
    stack_tuning = _read_required_csv(artifacts / f"{station}_year_split_stack_tuning.csv")

    config = StationStackingConfig(station_id=station, project_root=root, output_dir=artifacts)
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
        if str(row.get("method", "")) in BASE_MODEL_METHODS
    }
    missing_methods = [method for method in BASE_MODEL_METHODS if method not in selected_by_method]
    if missing_methods:
        raise ValueError(f"{station} selected hyperparameters missing methods: {missing_methods}")

    for method in BASE_MODEL_METHODS:
        row = selected_by_method[method]
        params = _params_from_selected_row(row)
        estimator = _build_base_model_pipeline(config, fit_categorical, fit_numeric, method, params)
        estimator.fit(train[feature_names], train[TARGET])
        base_models[method] = estimator
        base_model_manifests.append(
            {
                "method": method,
                "param_key": str(row.get("param_key", "")),
                "mean_validation_rmse_f": _jsonable(row.get("mean_validation_rmse_f")),
                "mean_validation_mae_f": _jsonable(row.get("mean_validation_mae_f")),
                "params": _jsonable(params),
            }
        )

    stack_model, stack_manifest = _fit_stack_model(validation_predictions, stack_tuning)

    bundle = {
        "schema_version": 1,
        "model_version": model_version,
        "station_id": station,
        "target": TARGET,
        "training_mode": training_mode,
        "base_models": base_models,
        "stack_model": stack_model,
        "stack_features": stack_manifest["features"],
        "categorical_features": fit_categorical,
        "numeric_features": fit_numeric,
        "feature_names": feature_names,
        "providers": tuple(config.providers),
    }

    bundle_path = output_dir / f"{station}_{model_version}.joblib"
    manifest_path = output_dir / f"{station}_{model_version}.json"
    _dump_joblib(bundle, bundle_path)

    manifest = {
        "schema_version": 1,
        "artifact_type": "station_high_regression_model_weights",
        "model_version": model_version,
        "station_id": station,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_pipeline": "notebooks/station_stacking_v2",
        "source_artifact_dir": str(artifacts),
        "bundle_path": str(bundle_path),
        "training": {
            "mode": training_mode,
            "train_start_year": train_start_year,
            "train_end_year": train_end_year,
            "train_rows": int(len(train)),
            "first_contract_date": str(train["contract_date"].min()),
            "last_contract_date": str(train["contract_date"].max()),
            "target": TARGET,
        },
        "features": {
            "categorical": fit_categorical,
            "numeric": fit_numeric,
            "all": feature_names,
        },
        "base_models": base_model_manifests,
        "stack_model": stack_manifest,
        "inference": {
            "primary_output": "predictedHighF",
            "base_prediction_inputs": [f"{method}_predicted_high_f" for method in BASE_MODEL_METHODS],
            "raw_forecast_inputs": ["hrrr_high_f", "gfs_high_f"],
            "point_in_time_rule": "HRRR/GFS same-day 11 AM local plus current observation features from notebook v2 feature table.",
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
) -> list[ExportedModelWeights]:
    exports = [
        export_station_model_weights(
            project_root=project_root,
            station_id=station,
            artifact_dir=artifact_dir,
            model_dir=model_dir,
            train_years=train_years,
            model_version=model_version,
        )
        for station in stations
    ]
    _write_export_index(exports, model_version=model_version)
    return exports


def _fit_stack_model(validation_predictions: pd.DataFrame, stack_tuning: pd.DataFrame) -> tuple[Any, dict[str, Any]]:
    ok = stack_tuning.loc[stack_tuning["status"].astype(str).str.lower().eq("ok")].copy()
    if ok.empty:
        raise ValueError("No successful ridge stack tuning rows are available.")
    selected = ok.sort_values(["rmse_f", "param_key"]).iloc[0]
    stack_methods = list(STACK_FEATURE_SETS["models_plus_raw"])
    train_source = _year_split_stack_source_frame(validation_predictions, stack_methods)
    feature_set = str(selected["feature_set"])
    stack_features = _stack_features_for_set(feature_set)
    train = train_source.dropna(subset=[*stack_features, TARGET]).copy()
    if train.empty:
        raise ValueError("No complete rows are available for fitting the ridge stack.")

    from sklearn.linear_model import Ridge

    model = Ridge(alpha=float(selected["alpha"]), fit_intercept=_coerce_bool(selected["fit_intercept"]))
    model.fit(train[stack_features], train[TARGET])
    manifest = {
        "method": STACK_METHOD,
        "param_key": str(selected["param_key"]),
        "feature_set": feature_set,
        "features": stack_features,
        "alpha": float(selected["alpha"]),
        "fit_intercept": _coerce_bool(selected["fit_intercept"]),
        "validation_rmse_f": _jsonable(selected.get("rmse_f")),
        "validation_mae_f": _jsonable(selected.get("mae_f")),
        "meta_train_rows": int(len(train)),
        "meta_train_first_contract_date": str(train["contract_date"].min()),
        "meta_train_last_contract_date": str(train["contract_date"].max()),
    }
    return model, manifest


def _read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required v2 artifact: {path}")
    frame = pd.read_csv(path, low_memory=False)
    if frame.empty:
        raise ValueError(f"Required v2 artifact is empty: {path}")
    return frame


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
    parser = argparse.ArgumentParser(description="Export notebook v2 station-stacking model weights.")
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
        help="Directory containing station_stacking_v2 CSV artifacts.",
    )
    parser.add_argument("--model-dir", default=None, help="Output directory for model weights.")
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
    )
    for item in exports:
        print(f"{item.station_id}: {item.bundle_path}")


if __name__ == "__main__":
    main()
