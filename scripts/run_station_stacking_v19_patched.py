from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.calibration.station_stacking import (  # noqa: E402
    TARGET_SOURCE_SETTLEMENT_FIRST,
    StationStackingConfig,
    YearSplitFold,
    _modeling_frame,
    run_station_year_split_experiment,
)
from src.calibration.v19_bucket import (  # noqa: E402
    bucket_decision_metrics,
    crossfit_ridge_predictions,
    empirical_modal_bucket_decisions,
    feature_missingness_audit,
    ordinal_blend_bucket_decisions,
    ordinal_blend_metrics,
    paired_bootstrap_accuracy_gain,
    paired_bootstrap_bucket_gain,
)


STATIONS = ("KATL", "KDAL")
OPTUNA_TRIALS = 30
OPTUNA_STARTUP_TRIALS = 15
STACK_OPTUNA_TRIALS = 30
STACK_OPTUNA_STARTUP_TRIALS = 15
MISSINGNESS_LIMIT = 0.03
MONTHLY_SHRINKAGE = 60.0
BLEND_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
OUTPUT_ROOT = REPO_ROOT / "data" / "calibration" / "station_stacking_v19_patched"
LEGACY_V19_ROOT = REPO_ROOT / "data" / "calibration" / "station_stacking_v19"
FOLDS = (
    YearSplitFold("train_2021_valid_2022", 2021, 2021, 2022),
    YearSplitFold("train_2021_2022_valid_2023", 2021, 2022, 2023),
    YearSplitFold("train_2021_2023_valid_2024", 2021, 2023, 2024),
    YearSplitFold("train_2021_2024_valid_2025", 2021, 2024, 2025),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run patched V19-B/C sequentially with clean Optuna studies.")
    parser.add_argument("--stations", default=",".join(STATIONS))
    parser.add_argument("--fast-mode", action="store_true")
    parser.add_argument("--quiet-optuna", action="store_true")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--seed-studies-from",
        type=Path,
        default=LEGACY_V19_ROOT,
        help="Clone only terminal trials from this V19 root into the fresh study database.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["WEATHER_RESEARCH_INCLUDE_DIRECT_NBM"] = "1"
    stations = tuple(value.strip().upper() for value in args.stations.split(",") if value.strip())
    args.output_root.mkdir(parents=True, exist_ok=True)
    for station in stations:
        run_station(station, args)


def run_station(station: str, args: argparse.Namespace) -> None:
    station_root = args.output_root / station
    dense_output = station_root / "dense_backbone"
    b_output = station_root / "v19_b"
    c_output = station_root / "v19_c"
    for path in (dense_output, b_output, c_output):
        path.mkdir(parents=True, exist_ok=True)
    _seed_clean_optuna_studies(
        source_path=args.seed_studies_from / station / "dense_backbone" / f"{station}_optuna.sqlite3",
        destination_path=dense_output / f"{station}_optuna.sqlite3",
    )
    config = StationStackingConfig(
        station_id=station,
        project_root=REPO_ROOT,
        timing_mode="same_day_11am_live_safe",
        providers=("gfs", "hrrr", "nbm"),
        fast_mode=args.fast_mode,
        optuna_trials=OPTUNA_TRIALS,
        optuna_startup_trials=OPTUNA_STARTUP_TRIALS,
        stack_optuna_trials=STACK_OPTUNA_TRIALS,
        stack_optuna_startup_trials=STACK_OPTUNA_STARTUP_TRIALS,
        optuna_metric="mae_f",
        optuna_verbose=not args.quiet_optuna,
        feature_version="v11",
        target_mode="remaining_warmup",
        target_source=TARGET_SOURCE_SETTLEMENT_FIRST,
        hyperparameter_space="wide",
        base_model_methods=("xgboost", "lightgbm", "catboost"),
        stack_enabled=True,
        year_split_folds=FOLDS,
        year_split_validation_weights={2022: 1.0, 2023: 1.0, 2024: 1.0, 2025: 1.0},
        year_split_test_train_years=(2021, 2025),
        year_split_test_year=2026,
        max_feature_missing_fraction=MISSINGNESS_LIMIT,
        output_dir=dense_output,
    )
    print(f"Running patched V19 dense backbone sequentially for {station}", flush=True)
    result = run_station_year_split_experiment(config)

    modeling_frame, categorical, numeric = _modeling_frame(result.features, config)
    missingness = feature_missingness_audit(
        modeling_frame,
        categorical,
        numeric,
        train_years=(2021, 2025),
        max_missing_fraction=MISSINGNESS_LIMIT,
    )
    missingness.to_csv(station_root / "feature_missingness_audit.csv", index=False)

    residuals = crossfit_ridge_predictions(
        result.validation_predictions,
        base_model_methods=tuple(config.effective_base_model_methods),
        providers=tuple(config.providers),
        min_train_rows=config.effective_min_meta_train_rows,
    )
    residuals.to_csv(station_root / "nested_crossfit_ridge_residuals.csv", index=False)

    b_decisions = empirical_modal_bucket_decisions(
        result.test_predictions,
        residuals,
        monthly_shrinkage=MONTHLY_SHRINKAGE,
    )
    b_metrics = bucket_decision_metrics(b_decisions)
    b_gain = paired_bootstrap_bucket_gain(b_decisions)
    b_decisions.to_csv(b_output / "2026_empirical_modal_decisions.csv", index=False)
    b_metrics.to_csv(b_output / "2026_metrics.csv", index=False)
    b_gain.to_frame("value").to_csv(b_output / "paired_bootstrap_gain.csv")

    c_decisions, blend_tuning, metadata = ordinal_blend_bucket_decisions(
        result.validation_predictions,
        result.test_predictions,
        residuals,
        base_model_methods=tuple(config.effective_base_model_methods),
        providers=tuple(config.providers),
        monthly_shrinkage=MONTHLY_SHRINKAGE,
        blend_weights=BLEND_WEIGHTS,
        min_train_rows=config.effective_min_meta_train_rows,
    )
    c_metrics = ordinal_blend_metrics(c_decisions)
    c_gain = paired_bootstrap_accuracy_gain(
        c_decisions["blended_bucket_hit"],
        c_decisions["empirical_bucket_hit"],
    )
    c_decisions.to_csv(c_output / "2026_ordinal_blend_decisions.csv", index=False)
    c_metrics.to_csv(c_output / "2026_metrics.csv", index=False)
    blend_tuning.to_csv(c_output / "ordinal_blend_tuning.csv", index=False)
    c_gain.to_frame("value").to_csv(c_output / "paired_bootstrap_blend_vs_empirical.csv")
    (c_output / "ordinal_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(b_metrics.to_string(index=False), flush=True)
    print(c_metrics.to_string(index=False), flush=True)
    print(f"Completed patched V19 for {station}", flush=True)


def _seed_clean_optuna_studies(source_path: Path, destination_path: Path) -> None:
    if destination_path.exists() or not source_path.exists():
        return
    import optuna

    source_storage = f"sqlite:///{source_path.resolve().as_posix()}"
    destination_storage = f"sqlite:///{destination_path.resolve().as_posix()}"
    summaries = optuna.study.get_all_study_summaries(storage=source_storage)
    for summary in summaries:
        source = optuna.load_study(study_name=summary.study_name, storage=source_storage)
        if len(source.directions) == 1:
            destination = optuna.create_study(
                study_name=summary.study_name,
                storage=destination_storage,
                direction=source.direction,
            )
        else:
            destination = optuna.create_study(
                study_name=summary.study_name,
                storage=destination_storage,
                directions=source.directions,
            )
        terminal_trials = [trial for trial in source.get_trials(deepcopy=True) if trial.state.is_finished()]
        destination.add_trials(terminal_trials)
        print(
            f"Seeded clean study {summary.study_name}: {len(terminal_trials)} terminal trials; "
            f"discarded {len(source.trials) - len(terminal_trials)} non-terminal trials",
            flush=True,
        )


if __name__ == "__main__":
    main()
