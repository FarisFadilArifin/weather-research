import json
from pathlib import Path


TARGET_STATIONS = ("KATL", "KAUS", "KORD", "KDAL", "KHOU", "KLAX", "KMIA", "KLGA", "KSEA")

# Source-owned v16 contract markers:
# feature_version="v16_fused"
# timing_mode=TIMING_MODE
# providers=PROVIDERS
# target_mode="remaining_warmup"
# target_source="iem_hourly"
# base_model_methods=("xgboost", "lightgbm", "catboost")
# stack_enabled=True
# export_station_model_weights
# write_v15_comparisons
# year_split_folds=YEAR_SPLIT_EXPANDING_FOLDS
# hyperparameter_space="wide"
# station_high_regressor_v16_fused_weather_stack
# OPTUNA_TRIALS = 30
# OPTUNA_STARTUP_TRIALS = 15
# STACK_OPTUNA_TRIALS = 30
# STACK_OPTUNA_STARTUP_TRIALS = 15


def _markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def _code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def _notebook(station_id: str) -> dict:
    cells = [
        _markdown(
            f"""# Station Stacking v16 - {station_id}

Experimental notebook for `{station_id}`.

V16 is a single fused-weather experiment: strict curated v11-style base plus the eight forecast-temp and precip/cloud aggregate weather features tested in v15.
"""
        ),
        _code(
            f"""from pathlib import Path
import os
import sys
import warnings

warnings.filterwarnings("ignore", message="IProgress not found.*")
warnings.filterwarnings("ignore", message="Skipping features without any observed values.*")

PROJECT_ROOT = Path.cwd().resolve()
while not (PROJECT_ROOT / "src" / "calibration" / "station_stacking.py").exists():
    if PROJECT_ROOT.parent == PROJECT_ROOT:
        raise RuntimeError("Could not find project root containing src/calibration/station_stacking.py")
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["WEATHER_RESEARCH_INCLUDE_DIRECT_NBM"] = "1"

STATION_ID = "{station_id}"
PROVIDERS = ("gfs", "hrrr", "nbm")
TIMING_MODE = "same_day_11am_live_safe"
TARGET_SOURCE = "iem_hourly"
FAST_MODE = False
OPTUNA_TRIALS = 30
STACK_OPTUNA_TRIALS = 30
OPTUNA_STARTUP_TRIALS = 15
STACK_OPTUNA_STARTUP_TRIALS = 15
OPTUNA_METRIC = "mae_f"
OPTUNA_VERBOSE = True
EXPORT_MODEL_WEIGHTS = True
OUTPUT_DIR = PROJECT_ROOT / "data" / "calibration" / "station_stacking_v16"
V15_OUTPUT_DIR = PROJECT_ROOT / "data" / "calibration" / "station_stacking_v15"
MODEL_VERSION = "station_high_regressor_v16_fused_weather_stack"

PROJECT_ROOT
"""
        ),
        _code(
            """import numpy as np
import pandas as pd

from scripts.run_station_stacking_v16 import write_v15_comparisons
from src.export_station_stacking_v2_models import export_station_model_weights
from src.calibration.station_stacking import (
    StationStackingConfig,
    V16_ADDITIONAL_FEATURE_COLUMNS,
    V16_BLOCKED_BASE_FEATURE_COLUMNS,
    V16_DROPPED_FEATURE_COLUMNS,
    V16_FEATURE_COLUMNS,
    YEAR_SPLIT_EXPANDING_FOLDS,
    missing_model_dependencies,
    provider_availability,
    run_station_year_split_experiment,
)
"""
        ),
        _markdown(
            """## V16 Contract

`feature_version="v16_fused"` uses the strict curated v11-style base, excludes accidental raw/provider weather sprawl, and adds only the eight fused v13 aggregate weather features when train-year coverage passes.
"""
        ),
        _code(
            """fold_spec = pd.DataFrame(
    [
        {
            "fold": fold.name,
            "train_start_year": fold.train_start_year,
            "train_end_year": fold.train_end_year,
            "validation_year": fold.validation_year,
        }
        for fold in YEAR_SPLIT_EXPANDING_FOLDS
    ]
)

{
    "folds": fold_spec,
    "v16_feature_count": len(V16_FEATURE_COLUMNS),
    "v16_additional_features": V16_ADDITIONAL_FEATURE_COLUMNS,
    "v16_blocked_base_features": sorted(V16_BLOCKED_BASE_FEATURE_COLUMNS),
    "v16_dropped_features": sorted(V16_DROPPED_FEATURE_COLUMNS),
}
"""
        ),
        _markdown("## Data Availability\n"),
        _code(
            """availability = provider_availability(
    PROJECT_ROOT,
    timing_mode=TIMING_MODE,
    providers=PROVIDERS,
)

availability.loc[availability["station_id"].eq(STATION_ID)]
"""
        ),
        _markdown("## Run Fused Model\n"),
        _code(
            """missing_packages = missing_model_dependencies()
if missing_packages:
    raise ImportError(
        "Missing station-stacking ML packages: "
        + ", ".join(missing_packages)
        + ". Install them with: python -m pip install -r requirements.txt"
    )

config = StationStackingConfig(
    station_id=STATION_ID,
    project_root=PROJECT_ROOT,
    timing_mode=TIMING_MODE,
    providers=PROVIDERS,
    fast_mode=FAST_MODE,
    optuna_trials=OPTUNA_TRIALS,
    stack_optuna_trials=STACK_OPTUNA_TRIALS,
    optuna_startup_trials=OPTUNA_STARTUP_TRIALS,
    stack_optuna_startup_trials=STACK_OPTUNA_STARTUP_TRIALS,
    optuna_metric=OPTUNA_METRIC,
    optuna_verbose=OPTUNA_VERBOSE,
    feature_version="v16_fused",
    target_mode="remaining_warmup",
    target_source=TARGET_SOURCE,
    base_model_methods=("xgboost", "lightgbm", "catboost"),
    stack_enabled=True,
    hyperparameter_space="wide",
    year_split_folds=YEAR_SPLIT_EXPANDING_FOLDS,
    year_split_test_train_years=(2021, 2025),
    year_split_test_year=2026,
    output_dir=OUTPUT_DIR / "fused",
    climatology_normals_path=PROJECT_ROOT / "data" / "calibration" / "station_stacking_v9" / "station_rolling_10y_daily_high_normals.csv",
)

config.resolved_optuna_storage_path()
"""
        ),
        _code(
            """result = run_station_year_split_experiment(config)
result.scoreboard
"""
        ),
        _code(
            """if EXPORT_MODEL_WEIGHTS:
    exported_weights = export_station_model_weights(
        project_root=PROJECT_ROOT,
        station_id=STATION_ID,
        artifact_dir=config.resolved_output_dir(),
        model_version=MODEL_VERSION,
        timing_mode=config.timing_mode,
        providers=tuple(config.providers),
        feature_version=config.effective_feature_version,
        optuna_metric=config.effective_optuna_metric,
        target_mode=config.effective_target_mode,
        target_source=config.effective_target_source,
        base_model_methods=tuple(config.effective_base_model_methods),
        stack_enabled=config.stack_enabled,
        source_pipeline="notebooks/experiments/station_stacking_v16",
    )
    display((exported_weights.bundle_path, exported_weights.manifest_path))

write_v15_comparisons(OUTPUT_DIR, V15_OUTPUT_DIR, STATION_ID)
"""
        ),
        _markdown("## V15 Reference Comparison\n"),
        _code(
            """comparison_path = OUTPUT_DIR / f"{STATION_ID}_v16_fused_vs_v15_common_date_comparison.csv"
comparison = pd.read_csv(comparison_path) if comparison_path.exists() else pd.DataFrame()
comparison.sort_values(["method", "delta_mae_f", "reference"]) if not comparison.empty else comparison
"""
        ),
        _markdown("## Selected Feature Audit\n"),
        _code(
            """selected = set(result.feature_columns["feature"].astype(str))
{
    "selected_feature_count": len(selected),
    "v16_additional_selected": sorted(selected & set(V16_ADDITIONAL_FEATURE_COLUMNS)),
    "blocked_base_selected": sorted(selected & set(V16_BLOCKED_BASE_FEATURE_COLUMNS)),
    "unexpected_v13_selected": sorted(
        feature
        for feature in selected
        if feature.startswith("v13_") and feature not in set(V16_ADDITIONAL_FEATURE_COLUMNS)
    ),
}
"""
        ),
        _markdown("## Added Feature Coverage\n"),
        _code(
            """available = [column for column in V16_ADDITIONAL_FEATURE_COLUMNS if column in result.features]
added_feature_coverage = (
    result.features[available]
    .notna()
    .mean()
    .mul(100)
    .rename("coverage_pct")
    .reset_index()
    .rename(columns={"index": "feature"})
) if available else pd.DataFrame()

added_feature_coverage.sort_values("coverage_pct", ascending=False)
"""
        ),
        _markdown("## Raw Weather Sprawl Check\n"),
        _code(
            """raw_weather_tokens = (
    "cloud",
    "ceiling",
    "dewpoint",
    "forecast_temp_at_as_of",
    "humidity",
    "precip",
    "pressure",
    "shortwave",
    "visibility",
    "wind_",
)
selected_features = result.feature_columns["feature"].astype(str)
raw_weather_selected = pd.DataFrame(
    [
        {"feature": feature}
        for feature in selected_features
        if feature.startswith(("gfs_", "hrrr_", "nbm_"))
        and any(token in feature for token in raw_weather_tokens)
    ]
)

raw_weather_selected
"""
        ),
        _markdown("## Rounded Within 1F Accuracy\n"),
        _code(
            """preds = pd.concat(
    [
        result.validation_predictions.assign(period="validation_2024_2025"),
        result.test_predictions.assign(period="oof_2026"),
    ],
    ignore_index=True,
)
predicted_high = pd.to_numeric(preds["predicted_high_f"], errors="coerce")
preds["predicted_high_rounded_f"] = np.floor(predicted_high + 0.5)
preds["within_1f_after_round"] = (
    pd.to_numeric(preds["actual_high_f"], errors="coerce") - preds["predicted_high_rounded_f"]
).abs().le(1)

within_1f_accuracy_by_period = (
    preds
    .dropna(subset=["actual_high_f", "predicted_high_rounded_f"])
    .groupby(["period", "method"], as_index=False)
    .agg(
        count=("within_1f_after_round", "size"),
        within_1f_count=("within_1f_after_round", "sum"),
        within_1f_accuracy_pct=("within_1f_after_round", lambda x: x.mean() * 100),
    )
    .sort_values(["period", "within_1f_accuracy_pct"], ascending=[True, False])
)

within_1f_accuracy_by_period
"""
        ),
        _markdown("## Bracket Metrics\n"),
        _code("result.bracket_metrics\n"),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    for station in TARGET_STATIONS:
        notebook = _notebook(station)
        path = out_dir / f"stacking_{station}_v16.ipynb"
        path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
