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
    TARGET_SOURCE_SETTLEMENT_FIRST,
    StationStackingConfig,
    YearSplitFold,
    run_station_year_split_experiment,
)
from src.export_station_stacking_v2_models import export_station_model_weights  # noqa: E402


MODEL_VERSION = "station_high_regressor_v11_settlement_expanding_4fold_ridge_stack"
TIMING_MODE = "same_day_11am_live_safe"
PROVIDERS = ("gfs", "hrrr", "nbm")
FEATURE_VERSION = "v11"
TARGET_MODE = "remaining_warmup"
BASE_MODEL_METHODS = ("xgboost", "lightgbm", "catboost")
OPTUNA_METRIC = "mae_f"
DEFAULT_STATIONS = ("KATL", "KDAL")

EXPANDING_4FOLDS = (
    YearSplitFold("fold_2021_to_2022", 2021, 2021, 2022),
    YearSplitFold("fold_2021_2022_to_2023", 2021, 2022, 2023),
    YearSplitFold("fold_2021_2023_to_2024", 2021, 2023, 2024),
    YearSplitFold("fold_2021_2024_to_2025", 2021, 2024, 2025),
)
EQUAL_FOLD_WEIGHTS = {2022: 1.0, 2023: 1.0, 2024: 1.0, 2025: 1.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v11 settlement with four expanding-year validation folds.")
    parser.add_argument("--stations", default=",".join(DEFAULT_STATIONS))
    parser.add_argument("--optuna-trials", type=int, default=30)
    parser.add_argument("--startup-trials", type=int, default=15)
    parser.add_argument("--stack-optuna-trials", type=int, default=30)
    parser.add_argument("--stack-startup-trials", type=int, default=15)
    parser.add_argument("--catboost-max-iterations", type=int, default=1200)
    parser.add_argument("--catboost-max-depth", type=int, default=8)
    parser.add_argument("--catboost-min-learning-rate", type=float, default=0.005)
    parser.add_argument("--catboost-max-border-count", type=int, default=128)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data/calibration/station_stacking_v11_settlement_expanding_4fold",
    )
    parser.add_argument(
        "--climatology-normals",
        type=Path,
        default=REPO_ROOT / "data/calibration/station_stacking_v9/station_rolling_10y_daily_high_normals.csv",
    )
    parser.add_argument("--fast-mode", action="store_true")
    parser.add_argument("--quiet-optuna", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stations = tuple(station.strip().upper() for station in args.stations.split(",") if station.strip())
    if not stations:
        raise ValueError("At least one station is required.")

    os.environ["WEATHER_RESEARCH_INCLUDE_DIRECT_NBM"] = "1"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_rows: list[dict[str, object]] = []

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
            target_source=TARGET_SOURCE_SETTLEMENT_FIRST,
            hyperparameter_space="wide",
            catboost_max_iterations=args.catboost_max_iterations,
            catboost_max_depth=args.catboost_max_depth,
            catboost_min_learning_rate=args.catboost_min_learning_rate,
            catboost_max_border_count=args.catboost_max_border_count,
            base_model_methods=BASE_MODEL_METHODS,
            stack_enabled=True,
            year_split_folds=EXPANDING_4FOLDS,
            year_split_validation_weights=EQUAL_FOLD_WEIGHTS,
            year_split_test_train_years=(2021, 2025),
            year_split_test_year=2026,
            output_dir=args.output_dir,
            climatology_normals_path=args.climatology_normals,
        )
        print(f"Running expanding-4fold/{station}: storage={config.resolved_optuna_storage_uri()}", flush=True)
        result = run_station_year_split_experiment(config)
        print(result.scoreboard.to_string(index=False), flush=True)
        comparison_rows.extend(_comparison_rows(result.scoreboard, result.bracket_metrics, station))

        if not args.skip_export:
            exported = export_station_model_weights(
                project_root=REPO_ROOT,
                station_id=station,
                artifact_dir=args.output_dir,
                model_version=MODEL_VERSION,
                timing_mode=TIMING_MODE,
                providers=PROVIDERS,
                feature_version=FEATURE_VERSION,
                optuna_metric=OPTUNA_METRIC,
                target_mode=TARGET_MODE,
                target_source=TARGET_SOURCE_SETTLEMENT_FIRST,
                base_model_methods=BASE_MODEL_METHODS,
                stack_enabled=True,
                source_pipeline="scripts/run_station_stacking_v11_settlement_expanding_4fold.py",
            )
            print(f"Exported {station}: {exported.bundle_path}", flush=True)

    pd.DataFrame(comparison_rows).to_csv(args.output_dir / "expanding_4fold_comparison.csv", index=False)


def _comparison_rows(scoreboard: pd.DataFrame, bracket_metrics: pd.DataFrame, station: str) -> list[dict[str, object]]:
    ridge = scoreboard.loc[
        scoreboard["period"].eq("test_2026") & scoreboard["method"].eq("ridge_stack")
    ]
    bracket = bracket_metrics.loc[bracket_metrics["method"].eq("ridge_stack")]
    if ridge.empty:
        return []
    row = ridge.iloc[0]
    return [
        {
            "station_id": station,
            "fold_scheme": "expanding_4fold_equal_weight",
            "validation_years": "2022,2023,2024,2025",
            "test_year": 2026,
            "count": int(row["count"]),
            "mae_f": float(row["mae_f"]),
            "rmse_f": float(row["rmse_f"]),
            "bracket_accuracy_pct": (
                float(bracket.iloc[0]["bracket_accuracy_pct"]) if not bracket.empty else pd.NA
            ),
            "model_version": MODEL_VERSION,
        }
    ]


if __name__ == "__main__":
    main()
