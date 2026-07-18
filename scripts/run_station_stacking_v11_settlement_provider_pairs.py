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
    TARGET_SOURCE_SETTLEMENT_FIRST,
    YEAR_SPLIT_EXPANDING_FOLDS,
    StationStackingConfig,
    run_station_year_split_experiment,
)
from src.export_station_stacking_v2_models import export_station_model_weights  # noqa: E402


TIMING_MODE = "same_day_11am_live_safe"
FEATURE_VERSION = "v11"
TARGET_MODE = "remaining_warmup"
BASE_MODEL_METHODS = ("xgboost", "lightgbm", "catboost")
OPTUNA_METRIC = "mae_f"
DEFAULT_STATIONS = ("KATL", "KDAL")


@dataclass(frozen=True)
class ProviderPair:
    name: str
    providers: tuple[str, str]

    @property
    def model_version(self) -> str:
        return f"station_high_regressor_v11_settlement_{self.name}_ridge_stack"


PROVIDER_PAIRS = {
    "gfs_hrrr": ProviderPair("gfs_hrrr", ("gfs", "hrrr")),
    "gfs_nbm": ProviderPair("gfs_nbm", ("gfs", "nbm")),
    "nbm_hrrr": ProviderPair("nbm_hrrr", ("nbm", "hrrr")),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train v11 settlement models using each requested two-provider pair."
    )
    parser.add_argument(
        "--variants",
        default=",".join(PROVIDER_PAIRS),
        help=f"Comma-separated variants from: {','.join(PROVIDER_PAIRS)}",
    )
    parser.add_argument("--stations", default=",".join(DEFAULT_STATIONS), help="Comma-separated station IDs.")
    parser.add_argument("--optuna-trials", type=int, default=30)
    parser.add_argument("--startup-trials", type=int, default=15)
    parser.add_argument("--stack-optuna-trials", type=int, default=30)
    parser.add_argument("--stack-startup-trials", type=int, default=15)
    parser.add_argument("--catboost-max-iterations", type=int, default=1200)
    parser.add_argument("--catboost-max-depth", type=int, default=8)
    parser.add_argument("--catboost-min-learning-rate", type=float, default=0.005)
    parser.add_argument("--catboost-max-border-count", type=int, default=128)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "data/calibration/station_stacking_v11_settlement_provider_pairs",
    )
    parser.add_argument(
        "--climatology-normals",
        type=Path,
        default=REPO_ROOT / "data/calibration/station_stacking_v9/station_rolling_10y_daily_high_normals.csv",
    )
    parser.add_argument("--fast-mode", action="store_true", help="Use shortened fast-mode validation blocks.")
    parser.add_argument("--quiet-optuna", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = _selected_variants(args.variants)
    stations = tuple(station.strip().upper() for station in args.stations.split(",") if station.strip())
    if not stations:
        raise ValueError("At least one station is required.")

    os.environ["WEATHER_RESEARCH_INCLUDE_DIRECT_NBM"] = "1"
    args.output_root.mkdir(parents=True, exist_ok=True)
    comparison_rows: list[dict[str, object]] = []

    for variant in variants:
        output_dir = args.output_root / variant.name
        output_dir.mkdir(parents=True, exist_ok=True)
        for station in stations:
            config = StationStackingConfig(
                station_id=station,
                project_root=REPO_ROOT,
                timing_mode=TIMING_MODE,
                providers=variant.providers,
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
                year_split_folds=YEAR_SPLIT_EXPANDING_FOLDS,
                year_split_test_train_years=(2021, 2025),
                year_split_test_year=2026,
                output_dir=output_dir,
                climatology_normals_path=args.climatology_normals,
            )
            print(
                f"Running {variant.name}/{station}: providers={variant.providers} "
                f"storage={config.resolved_optuna_storage_uri()}",
                flush=True,
            )
            result = run_station_year_split_experiment(config)
            print(result.scoreboard.to_string(index=False), flush=True)
            comparison_rows.extend(_comparison_rows(result.scoreboard, variant, station))

            if not args.skip_export:
                exported = export_station_model_weights(
                    project_root=REPO_ROOT,
                    station_id=station,
                    artifact_dir=output_dir,
                    model_version=variant.model_version,
                    timing_mode=TIMING_MODE,
                    providers=variant.providers,
                    feature_version=FEATURE_VERSION,
                    optuna_metric=OPTUNA_METRIC,
                    target_mode=TARGET_MODE,
                    target_source=TARGET_SOURCE_SETTLEMENT_FIRST,
                    base_model_methods=BASE_MODEL_METHODS,
                    stack_enabled=True,
                    source_pipeline="scripts/run_station_stacking_v11_settlement_provider_pairs.py",
                )
                print(f"Exported {variant.name}/{station}: {exported.bundle_path}", flush=True)

    comparison = pd.DataFrame(comparison_rows)
    comparison_path = args.output_root / "provider_pair_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    print(f"Wrote comparison: {comparison_path}", flush=True)


def _selected_variants(value: str) -> tuple[ProviderPair, ...]:
    names = tuple(name.strip().lower() for name in value.split(",") if name.strip())
    unknown = sorted(set(names) - set(PROVIDER_PAIRS))
    if unknown:
        raise ValueError(f"Unknown provider-pair variants: {unknown}")
    if not names:
        raise ValueError("At least one provider-pair variant is required.")
    return tuple(PROVIDER_PAIRS[name] for name in names)


def _comparison_rows(scoreboard: pd.DataFrame, variant: ProviderPair, station: str) -> list[dict[str, object]]:
    if scoreboard.empty:
        return []
    rows = scoreboard.loc[scoreboard["method"].eq("ridge_stack")].copy()
    return [
        {
            "variant": variant.name,
            "providers": "+".join(variant.providers),
            "station_id": station,
            "period": row["period"],
            "method": row["method"],
            "count": row["count"],
            "mae_f": row["mae_f"],
            "rmse_f": row["rmse_f"],
            "model_version": variant.model_version,
        }
        for _, row in rows.iterrows()
    ]


if __name__ == "__main__":
    main()
