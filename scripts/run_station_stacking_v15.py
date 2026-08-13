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
TARGET_MODE = "remaining_warmup"
BASE_MODEL_METHODS = ("xgboost", "lightgbm", "catboost")
OPTUNA_METRIC = "mae_f"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/calibration/station_stacking_v15"
ARM_CONFIGS = {
    "base": {
        "feature_version": "v15_base",
        "model_version": "station_high_regressor_v15_base_v11_current_stack",
    },
    "forecast_temp_at_as_of": {
        "feature_version": "v15_forecast_temp_at_as_of",
        "model_version": "station_high_regressor_v15_forecast_temp_at_as_of_stack",
    },
    "precip_cloud": {
        "feature_version": "v15_precip_cloud",
        "model_version": "station_high_regressor_v15_precip_cloud_stack",
    },
}
DEFAULT_ARMS = tuple(ARM_CONFIGS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run station-stacking v15 weather-ablation remaining-warmup stacks.")
    parser.add_argument("--stations", default=",".join(TARGET_STATIONS), help="Comma-separated station codes.")
    parser.add_argument(
        "--arms",
        default=",".join(DEFAULT_ARMS),
        help="Comma-separated v15 arms: base, forecast_temp_at_as_of, precip_cloud.",
    )
    parser.add_argument("--optuna-trials", type=int, default=30)
    parser.add_argument("--startup-trials", type=int, default=15)
    parser.add_argument("--stack-optuna-trials", type=int, default=30)
    parser.add_argument("--stack-startup-trials", type=int, default=15)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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
    args = parser.parse_args()
    args.arms = _parse_arms(args.arms, parser)
    return args


def _parse_arms(value: str, parser: argparse.ArgumentParser) -> list[str]:
    arms = [arm.strip() for arm in str(value).split(",") if arm.strip()]
    unknown = [arm for arm in arms if arm not in ARM_CONFIGS]
    if unknown:
        parser.error(f"unknown --arms values: {', '.join(unknown)}")
    if not arms:
        parser.error("--arms must include at least one arm")
    if "base" not in arms:
        parser.error("--arms must include base so comparisons can be computed versus v15_base")
    return list(dict.fromkeys(arms))


def _parse_stations(value: str) -> list[str]:
    return [station.strip().upper() for station in str(value).split(",") if station.strip()]


def _arm_output_dir(output_dir: Path, arm: str) -> Path:
    return output_dir / arm


def _read_arm_metrics(output_dir: Path, station: str, arm: str) -> pd.DataFrame:
    path = _arm_output_dir(output_dir, arm) / f"{station}_year_split_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    metrics = pd.read_csv(path)
    if "evaluation_scope" in metrics:
        metrics = metrics.loc[metrics["evaluation_scope"].eq("year_split_test")].copy()
    metrics.insert(0, "model_version", ARM_CONFIGS[arm]["model_version"])
    metrics.insert(0, "feature_version", ARM_CONFIGS[arm]["feature_version"])
    metrics.insert(0, "arm", arm)
    metrics.insert(0, "station_id", station)
    return metrics


def _read_arm_test_predictions(output_dir: Path, station: str, arm: str) -> pd.DataFrame:
    path = _arm_output_dir(output_dir, arm) / f"{station}_year_split_test_predictions.csv"
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


def _comparison_rows(output_dir: Path, station: str, arms: list[str]) -> list[dict[str, object]]:
    base_predictions = _read_arm_test_predictions(output_dir, station, "base")
    if base_predictions.empty:
        return []
    rows: list[dict[str, object]] = []
    for arm in arms:
        if arm == "base":
            continue
        arm_predictions = _read_arm_test_predictions(output_dir, station, arm)
        if arm_predictions.empty:
            continue
        common_methods = sorted(set(base_predictions["method"]) & set(arm_predictions["method"]))
        for method in common_methods:
            base_method = base_predictions.loc[
                base_predictions["method"].eq(method),
                ["contract_date", "actual_high_f", "predicted_high_f"],
            ].rename(
                columns={
                    "actual_high_f": "base_actual_high_f",
                    "predicted_high_f": "base_predicted_high_f",
                }
            )
            arm_method = arm_predictions.loc[
                arm_predictions["method"].eq(method),
                ["contract_date", "actual_high_f", "predicted_high_f"],
            ].rename(
                columns={
                    "actual_high_f": "arm_actual_high_f",
                    "predicted_high_f": "arm_predicted_high_f",
                }
            )
            merged = base_method.merge(arm_method, on="contract_date", how="inner")
            if merged.empty:
                continue
            actual = pd.to_numeric(merged["base_actual_high_f"], errors="coerce")
            base_predicted = pd.to_numeric(merged["base_predicted_high_f"], errors="coerce")
            arm_predicted = pd.to_numeric(merged["arm_predicted_high_f"], errors="coerce")
            base_error = actual - base_predicted
            arm_error = actual - arm_predicted
            base_abs = base_error.abs()
            arm_abs = arm_error.abs()
            base_mae = _mae(base_error)
            arm_mae = _mae(arm_error)
            base_rmse = _rmse(base_error)
            arm_rmse = _rmse(arm_error)
            mismatch = pd.to_numeric(merged["arm_actual_high_f"], errors="coerce").ne(actual).sum()
            rows.append(
                {
                    "station_id": station,
                    "baseline_arm": "base",
                    "comparison_arm": arm,
                    "feature_version": ARM_CONFIGS[arm]["feature_version"],
                    "model_version": ARM_CONFIGS[arm]["model_version"],
                    "method": method,
                    "common_date_count": int(len(merged)),
                    "base_mae_f": base_mae,
                    "arm_mae_f": arm_mae,
                    "delta_mae_f": arm_mae - base_mae,
                    "base_rmse_f": base_rmse,
                    "arm_rmse_f": arm_rmse,
                    "delta_rmse_f": arm_rmse - base_rmse,
                    "arm_better_days": int(arm_abs.lt(base_abs).sum()),
                    "base_better_days": int(base_abs.lt(arm_abs).sum()),
                    "tied_days": int(arm_abs.eq(base_abs).sum()),
                    "actual_mismatch_count": int(mismatch),
                    "first_common_date": str(merged["contract_date"].min()),
                    "last_common_date": str(merged["contract_date"].max()),
                }
            )
    return rows


def write_station_comparisons(output_dir: Path, station: str, arms: list[str]) -> None:
    metrics_frames = [_read_arm_metrics(output_dir, station, arm) for arm in arms]
    metrics_frames = [frame for frame in metrics_frames if not frame.empty]
    metrics_path = output_dir / f"{station}_v15_arm_test_metrics.csv"
    if metrics_frames:
        pd.concat(metrics_frames, ignore_index=True).to_csv(metrics_path, index=False)
    else:
        pd.DataFrame().to_csv(metrics_path, index=False)

    comparison_path = output_dir / f"{station}_v15_arm_common_date_comparison.csv"
    pd.DataFrame(_comparison_rows(output_dir, station, arms)).to_csv(comparison_path, index=False)
    print(f"Wrote {station} v15 comparisons: {metrics_path} and {comparison_path}", flush=True)


def main() -> None:
    args = parse_args()
    os.environ["WEATHER_RESEARCH_INCLUDE_DIRECT_NBM"] = "1"
    stations = _parse_stations(args.stations)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for station in stations:
        for arm in args.arms:
            arm_config = ARM_CONFIGS[arm]
            arm_dir = _arm_output_dir(args.output_dir, arm)
            arm_dir.mkdir(parents=True, exist_ok=True)
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
                feature_version=str(arm_config["feature_version"]),
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
            print(
                f"Running {station} v15 {arm}: feature_version={arm_config['feature_version']} "
                f"storage={config.resolved_optuna_storage_uri()}",
                flush=True,
            )
            result = run_station_year_split_experiment(config)
            print(result.scoreboard.to_string(index=False), flush=True)
            if not args.skip_export:
                exported = export_station_model_weights(
                    project_root=REPO_ROOT,
                    station_id=station,
                    artifact_dir=arm_dir,
                    model_version=str(arm_config["model_version"]),
                    timing_mode=TIMING_MODE,
                    providers=PROVIDERS,
                    feature_version=str(arm_config["feature_version"]),
                    optuna_metric=OPTUNA_METRIC,
                    target_mode=TARGET_MODE,
                    target_source=args.target_source,
                    base_model_methods=BASE_MODEL_METHODS,
                    stack_enabled=True,
                    source_pipeline="scripts/run_station_stacking_v15.py",
                )
                print(f"Exported {station} v15 {arm}: {exported.bundle_path}", flush=True)
        write_station_comparisons(args.output_dir, station, args.arms)


if __name__ == "__main__":
    main()
