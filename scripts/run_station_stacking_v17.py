from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.calibration.station_stacking import (  # noqa: E402
    TARGET_SOURCE_IEM_HOURLY,
    TARGET_STATIONS,
    YEAR_SPLIT_EXPANDING_FOLDS,
    StationStackingConfig,
    run_station_year_split_experiment,
)
from src.export_station_stacking_v2_models import export_station_model_weights  # noqa: E402


TIMING_MODE = "same_day_11am_live_safe"
PROVIDERS = ("gfs", "hrrr", "nbm")
FEATURE_VERSION = "v17_importance_015"
TARGET_MODE = "remaining_warmup"
BASE_MODEL_METHODS = ("xgboost", "lightgbm", "catboost")
OPTUNA_METRIC = "mae_f"
ARM_NAME = "importance_015"
MODEL_VERSION = "station_high_regressor_v17_importance_015_stack"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/calibration/station_stacking_v17"
DEFAULT_V15_OUTPUT_DIR = REPO_ROOT / "data/calibration/station_stacking_v15"
DEFAULT_V16_OUTPUT_DIR = REPO_ROOT / "data/calibration/station_stacking_v16"
V15_REFERENCE_ARMS = {
    "v15_base": "base",
    "v15_forecast_temp_at_as_of": "forecast_temp_at_as_of",
    "v15_precip_cloud": "precip_cloud",
}
V16_REFERENCE_ARMS = {
    "v16_fused": "fused",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run station-stacking v17 importance-pruned remaining-warmup stack.")
    parser.add_argument("--stations", default=",".join(TARGET_STATIONS), help="Comma-separated station codes.")
    parser.add_argument("--optuna-trials", type=int, default=30)
    parser.add_argument("--startup-trials", type=int, default=15)
    parser.add_argument("--stack-optuna-trials", type=int, default=30)
    parser.add_argument("--stack-startup-trials", type=int, default=15)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--v15-output-dir", type=Path, default=DEFAULT_V15_OUTPUT_DIR)
    parser.add_argument("--v16-output-dir", type=Path, default=DEFAULT_V16_OUTPUT_DIR)
    parser.add_argument(
        "--climatology-normals",
        type=Path,
        default=REPO_ROOT / "data/calibration/station_stacking_v9/station_rolling_10y_daily_high_normals.csv",
    )
    parser.add_argument("--target-source", default=TARGET_SOURCE_IEM_HOURLY)
    parser.add_argument("--fast-mode", action="store_true", help="Use shortened fast-mode validation blocks.")
    parser.add_argument("--quiet-optuna", action="store_true", help="Reduce Optuna logging.")
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Run training only and do not export joblib model weights.",
    )
    parser.add_argument(
        "--skip-reference-comparison",
        action="store_true",
        help="Do not write optional comparisons against existing v15/v16 outputs.",
    )
    return parser.parse_args()


def _parse_stations(value: str) -> list[str]:
    return [station.strip().upper() for station in str(value).split(",") if station.strip()]


def _v17_output_dir(output_dir: Path) -> Path:
    return output_dir / ARM_NAME


def _read_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    predictions = pd.read_csv(path)
    needed = {"contract_date", "method", "actual_high_f", "predicted_high_f"}
    missing = needed - set(predictions.columns)
    if missing:
        raise ValueError(f"{path} missing required comparison columns: {sorted(missing)}")
    predictions = predictions.copy()
    predictions["contract_date"] = predictions["contract_date"].astype(str).str[:10]
    return predictions


def _mae(error: pd.Series) -> float:
    return float(error.abs().mean())


def _rmse(error: pd.Series) -> float:
    return float(error.pow(2).mean() ** 0.5)


def _comparison_rows(
    *,
    station: str,
    v17_predictions: pd.DataFrame,
    reference_label: str,
    reference_predictions: pd.DataFrame,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if v17_predictions.empty or reference_predictions.empty:
        return rows
    common_methods = sorted(set(v17_predictions["method"]) & set(reference_predictions["method"]))
    for method in common_methods:
        reference_method = reference_predictions.loc[
            reference_predictions["method"].eq(method),
            ["contract_date", "actual_high_f", "predicted_high_f"],
        ].rename(
            columns={
                "actual_high_f": "reference_actual_high_f",
                "predicted_high_f": "reference_predicted_high_f",
            }
        )
        v17_method = v17_predictions.loc[
            v17_predictions["method"].eq(method),
            ["contract_date", "actual_high_f", "predicted_high_f"],
        ].rename(
            columns={
                "actual_high_f": "v17_actual_high_f",
                "predicted_high_f": "v17_predicted_high_f",
            }
        )
        merged = reference_method.merge(v17_method, on="contract_date", how="inner")
        if merged.empty:
            continue
        actual = pd.to_numeric(merged["reference_actual_high_f"], errors="coerce")
        reference_predicted = pd.to_numeric(merged["reference_predicted_high_f"], errors="coerce")
        v17_predicted = pd.to_numeric(merged["v17_predicted_high_f"], errors="coerce")
        reference_error = actual - reference_predicted
        v17_error = actual - v17_predicted
        reference_abs = reference_error.abs()
        v17_abs = v17_error.abs()
        reference_mae = _mae(reference_error)
        v17_mae = _mae(v17_error)
        reference_rmse = _rmse(reference_error)
        v17_rmse = _rmse(v17_error)
        mismatch = pd.to_numeric(merged["v17_actual_high_f"], errors="coerce").ne(actual).sum()
        rows.append(
            {
                "station_id": station,
                "reference": reference_label,
                "comparison": FEATURE_VERSION,
                "feature_version": FEATURE_VERSION,
                "model_version": MODEL_VERSION,
                "method": method,
                "common_date_count": int(len(merged)),
                "reference_mae_f": reference_mae,
                "v17_mae_f": v17_mae,
                "delta_mae_f": v17_mae - reference_mae,
                "reference_rmse_f": reference_rmse,
                "v17_rmse_f": v17_rmse,
                "delta_rmse_f": v17_rmse - reference_rmse,
                "v17_better_days": int(v17_abs.lt(reference_abs).sum()),
                "reference_better_days": int(reference_abs.lt(v17_abs).sum()),
                "tied_days": int(v17_abs.eq(reference_abs).sum()),
                "actual_mismatch_count": int(mismatch),
                "first_common_date": str(merged["contract_date"].min()),
                "last_common_date": str(merged["contract_date"].max()),
            }
        )
    return rows


def write_reference_comparisons(output_dir: Path, v15_output_dir: Path, v16_output_dir: Path, station: str) -> None:
    v17_path = _v17_output_dir(output_dir) / f"{station}_year_split_test_predictions.csv"
    v17_predictions = _read_predictions(v17_path)
    rows: list[dict[str, object]] = []
    reference_dirs = {
        **{label: v15_output_dir / dirname for label, dirname in V15_REFERENCE_ARMS.items()},
        **{label: v16_output_dir / dirname for label, dirname in V16_REFERENCE_ARMS.items()},
    }
    for label, reference_dir in reference_dirs.items():
        reference_path = reference_dir / f"{station}_year_split_test_predictions.csv"
        reference_predictions = _read_predictions(reference_path)
        if reference_predictions.empty:
            continue
        rows.extend(
            _comparison_rows(
                station=station,
                v17_predictions=v17_predictions,
                reference_label=label,
                reference_predictions=reference_predictions,
            )
        )
    comparison_path = output_dir / f"{station}_v17_importance_015_vs_references_common_date_comparison.csv"
    pd.DataFrame(rows).to_csv(comparison_path, index=False)
    print(f"Wrote {station} v17 reference comparison: {comparison_path}", flush=True)


def main() -> None:
    args = parse_args()
    os.environ["WEATHER_RESEARCH_INCLUDE_DIRECT_NBM"] = "1"
    stations = _parse_stations(args.stations)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arm_dir = _v17_output_dir(args.output_dir)
    arm_dir.mkdir(parents=True, exist_ok=True)
    for station in stations:
        config = StationStackingConfig(
            station_id=station,
            project_root=REPO_ROOT,
            timing_mode=TIMING_MODE,
            providers=PROVIDERS,
            fast_mode=args.fast_mode,
            optuna_trials=args.optuna_trials,
            stack_optuna_trials=args.stack_optuna_trials,
            optuna_startup_trials=args.startup_trials,
            stack_optuna_startup_trials=args.stack_startup_trials,
            optuna_metric=OPTUNA_METRIC,
            optuna_verbose=not args.quiet_optuna,
            feature_version=FEATURE_VERSION,
            target_mode=TARGET_MODE,
            target_source=args.target_source,
            hyperparameter_space="wide",
            base_model_methods=BASE_MODEL_METHODS,
            stack_enabled=True,
            year_split_folds=YEAR_SPLIT_EXPANDING_FOLDS,
            year_split_test_train_years=(2021, 2025),
            year_split_test_year=2026,
            output_dir=arm_dir,
            climatology_normals_path=args.climatology_normals,
        )
        print(f"Running {station} v17 importance_015: storage={config.resolved_optuna_storage_uri()}", flush=True)
        result = run_station_year_split_experiment(config)
        print(result.scoreboard.to_string(index=False), flush=True)
        if not args.skip_export:
            exported = export_station_model_weights(
                project_root=REPO_ROOT,
                station_id=station,
                artifact_dir=arm_dir,
                model_version=MODEL_VERSION,
                timing_mode=TIMING_MODE,
                providers=PROVIDERS,
                feature_version=FEATURE_VERSION,
                optuna_metric=OPTUNA_METRIC,
                target_mode=TARGET_MODE,
                target_source=args.target_source,
                base_model_methods=BASE_MODEL_METHODS,
                stack_enabled=True,
                source_pipeline="scripts/run_station_stacking_v17.py",
            )
            print(f"Exported {station} v17 importance_015: {exported.bundle_path}", flush=True)
        if not args.skip_reference_comparison:
            write_reference_comparisons(args.output_dir, args.v15_output_dir, args.v16_output_dir, station)


if __name__ == "__main__":
    main()
