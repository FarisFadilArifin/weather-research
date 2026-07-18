from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.calibration.station_stacking import (  # noqa: E402
    TARGET_SOURCE_WUNDERGROUND_ONLY,
    YEAR_SPLIT_EXPANDING_FOLDS,
    StationStackingConfig,
    feature_columns,
    run_station_year_split_experiment,
    summarize_v18_nbm_rap_readiness,
)
from src.export_station_stacking_v2_models import export_station_model_weights  # noqa: E402


TARGET_STATIONS = ("KATL", "KDAL", "KMIA")
TIMING_MODE = "same_day_11am_live_safe"
PROVIDERS = ("gfs", "hrrr", "nbm")
TARGET_MODE = "remaining_warmup"
BASE_MODEL_METHODS = ("xgboost", "lightgbm", "catboost")
OPTUNA_METRIC = "mae_f"


@dataclass(frozen=True)
class Variant:
    name: str
    feature_version: str
    model_version: str

    @property
    def output_dir(self) -> Path:
        return REPO_ROOT / "data" / "calibration" / f"station_stacking_{self.name}"


VARIANTS = {
    "nbm": Variant(
        name="v18_1_nbm",
        feature_version="v18_1_nbm",
        model_version="station_high_regressor_v18_1_nbm_settlement_stack",
    ),
    "rap": Variant(
        name="v18_1_rap",
        feature_version="v18_1_rap",
        model_version="station_high_regressor_v18_1_rap_physics_settlement_stack",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v18.1 Wunderground-only NBM or RAP feature ablation.")
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--stations", default=",".join(TARGET_STATIONS), help="Comma-separated station codes.")
    parser.add_argument("--optuna-trials", type=int, default=100)
    parser.add_argument("--startup-trials", type=int, default=40)
    parser.add_argument("--stack-optuna-trials", type=int, default=100)
    parser.add_argument("--stack-startup-trials", type=int, default=40)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--climatology-normals",
        type=Path,
        default=REPO_ROOT / "data/calibration/station_stacking_v9/station_rolling_10y_daily_high_normals.csv",
    )
    parser.add_argument("--fast-mode", action="store_true", help="Use shortened fast-mode validation blocks.")
    parser.add_argument("--quiet-optuna", action="store_true", help="Reduce Optuna logging.")
    parser.add_argument("--readiness-only", action="store_true", help="Write readiness reports and exit.")
    parser.add_argument("--skip-export", action="store_true", help="Run training only and do not export joblib model weights.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variant = VARIANTS[args.variant]
    output_dir = args.output_dir or variant.output_dir
    stations = [station.strip().upper() for station in str(args.stations).split(",") if station.strip()]
    unsupported = sorted(set(stations) - set(TARGET_STATIONS))
    if unsupported:
        raise ValueError(f"v18.1 shard ablations are limited to {TARGET_STATIONS}; got {unsupported}")
    os.environ["WEATHER_RESEARCH_INCLUDE_DIRECT_NBM"] = "1"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_readiness_reports(output_dir, variant, stations)
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
            feature_version=variant.feature_version,
            target_mode=TARGET_MODE,
            target_source=TARGET_SOURCE_WUNDERGROUND_ONLY,
            hyperparameter_space="wide_plus",
            base_model_methods=BASE_MODEL_METHODS,
            stack_enabled=True,
            year_split_folds=YEAR_SPLIT_EXPANDING_FOLDS,
            year_split_test_train_years=(2021, 2025),
            year_split_test_year=2026,
            output_dir=output_dir,
            climatology_normals_path=args.climatology_normals,
        )
        print(f"Running {station} {variant.name}: storage={config.resolved_optuna_storage_uri()}", flush=True)
        result = run_station_year_split_experiment(config)
        print(result.scoreboard.to_string(index=False), flush=True)
        _write_feature_coverage(output_dir, result.features, config, variant)
        if not args.skip_export:
            exported = export_station_model_weights(
                project_root=REPO_ROOT,
                station_id=station,
                artifact_dir=output_dir,
                model_version=variant.model_version,
                timing_mode=TIMING_MODE,
                providers=PROVIDERS,
                feature_version=variant.feature_version,
                optuna_metric=OPTUNA_METRIC,
                target_mode=TARGET_MODE,
                target_source=TARGET_SOURCE_WUNDERGROUND_ONLY,
                base_model_methods=BASE_MODEL_METHODS,
                stack_enabled=True,
                source_pipeline="scripts/run_station_stacking_v18_1.py",
            )
            print(f"Exported {station} {variant.name}: {exported.bundle_path}", flush=True)


def _write_readiness_reports(output_dir: Path, variant: Variant, stations: list[str]) -> None:
    settlement = _settlement_readiness(stations)
    settlement.to_csv(output_dir / f"{variant.name}_wunderground_settlement_readiness.csv", index=False)
    shard = summarize_v18_nbm_rap_readiness(REPO_ROOT, stations=stations)
    shard.to_csv(output_dir / f"{variant.name}_shard_readiness.csv", index=False)
    print(f"Wrote {variant.name} readiness reports to {output_dir}", flush=True)


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


def _write_feature_coverage(output_dir: Path, features: pd.DataFrame, config: StationStackingConfig, variant: Variant) -> None:
    categorical, numeric = feature_columns(features, config)
    train_start, train_end = config.effective_year_split_test_train_years
    years = pd.to_numeric(features.get("year"), errors="coerce")
    train = features.loc[years.between(train_start, train_end)].copy()
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
    pd.DataFrame(rows).to_csv(output_dir / f"{config.station_id.upper()}_{variant.name}_selected_feature_coverage.csv", index=False)


if __name__ == "__main__":
    main()
