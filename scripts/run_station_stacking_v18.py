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
    TARGET_SOURCE_WUNDERGROUND_ONLY,
    TARGET_STATIONS,
    YEAR_SPLIT_EXPANDING_FOLDS,
    StationStackingConfig,
    feature_columns,
    run_station_year_split_experiment,
    summarize_v18_nbm_rap_readiness,
)
from src.export_station_stacking_v2_models import export_station_model_weights  # noqa: E402


MODEL_VERSION = "station_high_regressor_v18_nbm_hrrr_physics_settlement_stack"
TIMING_MODE = "same_day_11am_live_safe"
PROVIDERS = ("gfs", "hrrr", "nbm")
FEATURE_VERSION = "v18"
TARGET_MODE = "remaining_warmup"
BASE_MODEL_METHODS = ("xgboost", "lightgbm", "catboost")
OPTUNA_METRIC = "mae_f"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/calibration/station_stacking_v18"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run station-stacking v18 Wunderground-only bucket-aware physics stack.")
    parser.add_argument("--stations", default=",".join(TARGET_STATIONS), help="Comma-separated station codes.")
    parser.add_argument("--optuna-trials", type=int, default=100)
    parser.add_argument("--startup-trials", type=int, default=40)
    parser.add_argument("--stack-optuna-trials", type=int, default=100)
    parser.add_argument("--stack-startup-trials", type=int, default=40)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--climatology-normals",
        type=Path,
        default=REPO_ROOT / "data/calibration/station_stacking_v9/station_rolling_10y_daily_high_normals.csv",
    )
    parser.add_argument("--target-source", default=TARGET_SOURCE_WUNDERGROUND_ONLY)
    parser.add_argument("--fast-mode", action="store_true", help="Use shortened fast-mode validation blocks.")
    parser.add_argument("--quiet-optuna", action="store_true", help="Reduce Optuna logging.")
    parser.add_argument("--readiness-only", action="store_true", help="Write v18 readiness reports and exit.")
    parser.add_argument("--skip-export", action="store_true", help="Run training only and do not export joblib model weights.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["WEATHER_RESEARCH_INCLUDE_DIRECT_NBM"] = "1"
    stations = [station.strip().upper() for station in str(args.stations).split(",") if station.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_readiness_reports(args.output_dir, stations)
    if args.readiness_only:
        return
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
            hyperparameter_space="wide_plus",
            base_model_methods=BASE_MODEL_METHODS,
            stack_enabled=True,
            year_split_folds=YEAR_SPLIT_EXPANDING_FOLDS,
            year_split_test_train_years=(2021, 2025),
            year_split_test_year=2026,
            output_dir=args.output_dir,
            climatology_normals_path=args.climatology_normals,
        )
        print(f"Running {station} v18: storage={config.resolved_optuna_storage_uri()}", flush=True)
        result = run_station_year_split_experiment(config)
        print(result.scoreboard.to_string(index=False), flush=True)
        _write_feature_coverage(args.output_dir, result.features, config)
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
                target_source=args.target_source,
                base_model_methods=BASE_MODEL_METHODS,
                stack_enabled=True,
                source_pipeline="scripts/run_station_stacking_v18.py",
            )
            print(f"Exported {station} v18: {exported.bundle_path}", flush=True)


def _write_readiness_reports(output_dir: Path, stations: list[str]) -> None:
    settlement = _settlement_readiness(stations)
    settlement.to_csv(output_dir / "v18_wunderground_settlement_readiness.csv", index=False)
    shard = summarize_v18_nbm_rap_readiness(REPO_ROOT, stations=stations)
    shard.to_csv(output_dir / "v18_nbm_rap_shard_readiness.csv", index=False)
    print(f"Wrote v18 readiness reports to {output_dir}", flush=True)


def _settlement_readiness(stations: list[str]) -> pd.DataFrame:
    path = REPO_ROOT / "data/processed/settlement_actual_highs.csv"
    columns = ["station_id", "rows", "first_contract_date", "last_contract_date", "ok_wunderground_rows"]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path)
    rows: list[dict[str, object]] = []
    for station in stations:
        group = frame.loc[frame["station_id"].astype("string").str.upper().eq(station)].copy()
        ok = (
            group["settlement_source"].astype("string").str.strip().str.lower().eq("wunderground_station_history")
            & group["quality_flag"].astype("string").str.strip().str.lower().eq("ok")
            & pd.to_numeric(group["settlement_high_f"], errors="coerce").notna()
        )
        ok_group = group.loc[ok]
        rows.append(
            {
                "station_id": station,
                "rows": int(len(group)),
                "first_contract_date": str(ok_group["contract_date"].min()) if not ok_group.empty else pd.NA,
                "last_contract_date": str(ok_group["contract_date"].max()) if not ok_group.empty else pd.NA,
                "ok_wunderground_rows": int(len(ok_group)),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _write_feature_coverage(output_dir: Path, features: pd.DataFrame, config: StationStackingConfig) -> None:
    categorical, numeric = feature_columns(features, config)
    train_start, train_end = config.effective_year_split_test_train_years
    if "year" in features:
        years = pd.to_numeric(features["year"], errors="coerce")
        train = features.loc[years.between(train_start, train_end)].copy()
    else:
        train = features.copy()
    rows = []
    for kind, columns in [("categorical", categorical), ("numeric", numeric)]:
        for column in columns:
            non_null = train[column].notna() if column in train else pd.Series(False, index=train.index)
            rows.append(
                {
                    "station_id": config.station_id.upper(),
                    "feature": column,
                    "kind": kind,
                    "train_rows": int(len(train)),
                    "non_null_train_rows": int(non_null.sum()),
                    "non_null_train_pct": float(non_null.mean() * 100.0) if len(train) else 0.0,
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / f"{config.station_id.upper()}_v18_selected_feature_coverage.csv", index=False)


if __name__ == "__main__":
    main()
