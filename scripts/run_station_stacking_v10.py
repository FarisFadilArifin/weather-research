from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.calibration.station_stacking import (  # noqa: E402
    TARGET_STATIONS,
    YEAR_SPLIT_EXPANDING_FOLDS,
    StationStackingConfig,
    run_station_year_split_experiment,
)
from src.export_station_stacking_v2_models import export_station_model_weights  # noqa: E402


MODEL_VERSION = "station_high_regressor_v10_catboost_huber"
TIMING_MODE = "same_day_11am_live_safe"
PROVIDERS = ("gfs", "hrrr", "nbm")
FEATURE_VERSION = "v10"
TARGET_MODE = "remaining_warmup"
BASE_MODEL_METHODS = ("catboost",)
OPTUNA_METRIC = "mae_f"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run station-stacking v10 CatBoost Huber year-split training.")
    parser.add_argument("--stations", default=",".join(TARGET_STATIONS), help="Comma-separated station codes.")
    parser.add_argument("--optuna-trials", type=int, default=30)
    parser.add_argument("--startup-trials", type=int, default=15)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data/calibration/station_stacking_v10")
    parser.add_argument(
        "--climatology-normals",
        type=Path,
        default=REPO_ROOT / "data/calibration/station_stacking_v9/station_rolling_10y_daily_high_normals.csv",
    )
    parser.add_argument("--fast-mode", action="store_true", help="Use shortened fast-mode validation blocks.")
    parser.add_argument("--quiet-optuna", action="store_true", help="Reduce Optuna logging.")
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Run training only and do not export joblib model weights.",
    )
    parser.add_argument(
        "--model-version",
        default=MODEL_VERSION,
        help="Model version string for exported joblib and manifest files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["WEATHER_RESEARCH_INCLUDE_DIRECT_NBM"] = "1"
    stations = [station.strip().upper() for station in args.stations.split(",") if station.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for station in stations:
        config = StationStackingConfig(
            station_id=station,
            project_root=REPO_ROOT,
            timing_mode=TIMING_MODE,
            providers=PROVIDERS,
            fast_mode=args.fast_mode,
            optuna_trials=args.optuna_trials,
            stack_optuna_trials=0,
            optuna_startup_trials=args.startup_trials,
            stack_optuna_startup_trials=0,
            optuna_metric=OPTUNA_METRIC,
            optuna_verbose=not args.quiet_optuna,
            feature_version=FEATURE_VERSION,
            target_mode=TARGET_MODE,
            hyperparameter_space="wide",
            base_model_methods=BASE_MODEL_METHODS,
            stack_enabled=False,
            year_split_folds=YEAR_SPLIT_EXPANDING_FOLDS,
            year_split_test_train_years=(2021, 2025),
            year_split_test_year=2026,
            output_dir=args.output_dir,
            climatology_normals_path=args.climatology_normals,
        )
        print(f"Running {station}: storage={config.resolved_optuna_storage_uri()}", flush=True)
        result = run_station_year_split_experiment(config)
        print(result.scoreboard.to_string(index=False), flush=True)
        if not args.skip_export:
            exported = export_station_model_weights(
                project_root=REPO_ROOT,
                station_id=station,
                artifact_dir=args.output_dir,
                model_version=args.model_version,
                timing_mode=TIMING_MODE,
                providers=PROVIDERS,
                feature_version=FEATURE_VERSION,
                optuna_metric=OPTUNA_METRIC,
                target_mode=TARGET_MODE,
                base_model_methods=BASE_MODEL_METHODS,
                stack_enabled=False,
                source_pipeline="scripts/run_station_stacking_v10.py",
            )
            print(f"Exported {station}: {exported.bundle_path}", flush=True)


if __name__ == "__main__":
    main()
