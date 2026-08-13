import json
from pathlib import Path


TARGET_STATIONS = ("KATL", "KAUS", "KORD", "KDAL", "KHOU", "KLAX", "KMIA", "KLGA", "KSEA")

# Source-owned v15 contract markers:
# feature_version="v15_base"
# feature_version="v15_forecast_temp_at_as_of"
# feature_version="v15_precip_cloud"
# timing_mode=TIMING_MODE
# providers=PROVIDERS
# target_mode="remaining_warmup"
# target_source="iem_hourly"
# base_model_methods=("xgboost", "lightgbm", "catboost")
# stack_enabled=True
# export_station_model_weights
# write_station_comparisons
# year_split_folds=YEAR_SPLIT_EXPANDING_FOLDS
# hyperparameter_space="wide"
# station_high_regressor_v15_base_v11_current_stack
# station_high_regressor_v15_forecast_temp_at_as_of_stack
# station_high_regressor_v15_precip_cloud_stack
# OPTUNA_TRIALS = 30
# OPTUNA_STARTUP_TRIALS = 15
# STACK_OPTUNA_TRIALS = 30
# STACK_OPTUNA_STARTUP_TRIALS = 15


def _markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


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
            f"""# Station Stacking v15 - {station_id}

Experimental notebook for `{station_id}`.

V15 is a three-arm weather ablation family. It reruns a fresh v11-current baseline as `v15_base`, then tests `forecast_temp_at_as_of` additions and `precip_cloud` additions in separate output folders under `data/calibration/station_stacking_v15`.
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
TREND_COLUMNS = [
    "observed_temp_change_last_1h_f",
    "observed_temp_change_last_3h_f",
    "observed_morning_warmup_rate_f_per_hour",
    "observed_high_so_far_change_since_9am_f",
]
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
OUTPUT_DIR = PROJECT_ROOT / "data" / "calibration" / "station_stacking_v15"
ARMS = [
    {{
        "arm": "base",
        "feature_version": "v15_base",
        "model_version": "station_high_regressor_v15_base_v11_current_stack",
    }},
    {{
        "arm": "forecast_temp_at_as_of",
        "feature_version": "v15_forecast_temp_at_as_of",
        "model_version": "station_high_regressor_v15_forecast_temp_at_as_of_stack",
    }},
    {{
        "arm": "precip_cloud",
        "feature_version": "v15_precip_cloud",
        "model_version": "station_high_regressor_v15_precip_cloud_stack",
    }},
]

PROJECT_ROOT
"""
        ),
        _code(
            """import numpy as np
import pandas as pd

from scripts.run_station_stacking_v15 import write_station_comparisons
from src.export_station_stacking_v2_models import export_station_model_weights
from src.calibration.station_stacking import (
    StationStackingConfig,
    V15_ADDITIONAL_FEATURE_COLUMNS,
    V15_BASE_FEATURE_COLUMNS,
    V15_DROPPED_FEATURE_COLUMNS,
    V15_FORECAST_TEMP_AT_AS_OF_FEATURE_COLUMNS,
    V15_PRECIP_CLOUD_FEATURE_COLUMNS,
    YEAR_SPLIT_EXPANDING_FOLDS,
    missing_model_dependencies,
    provider_availability,
    run_station_year_split_experiment,
)
"""
        ),
        _markdown(
            """## V15 Contract

`v15_base` is a fresh v11-current baseline. The added-feature arms always keep the full v11 base, block raw provider weather fields and provider-diff sprawl, and only select their explicit aggregate allowlists after train-year coverage passes.
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

arm_spec = pd.DataFrame(ARMS)
fold_spec, arm_spec
"""
        ),
        _code(
            """{
    "base_columns": V15_BASE_FEATURE_COLUMNS,
    "forecast_temp_at_as_of_additions": V15_FORECAST_TEMP_AT_AS_OF_FEATURE_COLUMNS,
    "precip_cloud_additions": V15_PRECIP_CLOUD_FEATURE_COLUMNS,
    "dropped_columns": sorted(V15_DROPPED_FEATURE_COLUMNS),
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
        _markdown("## Run Three Arms\n"),
        _code(
            """missing_packages = missing_model_dependencies()
if missing_packages:
    raise ImportError(
        "Missing station-stacking ML packages: "
        + ", ".join(missing_packages)
        + ". Install them with: python -m pip install -r requirements.txt"
    )

results = {}
for arm in ARMS:
    arm_dir = OUTPUT_DIR / arm["arm"]
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
        feature_version=arm["feature_version"],
        target_mode="remaining_warmup",
        target_source=TARGET_SOURCE,
        base_model_methods=("xgboost", "lightgbm", "catboost"),
        stack_enabled=True,
        hyperparameter_space="wide",
        year_split_folds=YEAR_SPLIT_EXPANDING_FOLDS,
        year_split_test_train_years=(2021, 2025),
        year_split_test_year=2026,
        output_dir=arm_dir,
        climatology_normals_path=PROJECT_ROOT / "data" / "calibration" / "station_stacking_v9" / "station_rolling_10y_daily_high_normals.csv",
    )
    print(f"Running {STATION_ID} {arm['arm']}: {config.resolved_optuna_storage_uri()}")
    result = run_station_year_split_experiment(config)
    results[arm["arm"]] = result
    display(result.scoreboard.assign(arm=arm["arm"], feature_version=arm["feature_version"]))
    if EXPORT_MODEL_WEIGHTS:
        exported_weights = export_station_model_weights(
            project_root=PROJECT_ROOT,
            station_id=STATION_ID,
            artifact_dir=config.resolved_output_dir(),
            model_version=arm["model_version"],
            timing_mode=config.timing_mode,
            providers=tuple(config.providers),
            feature_version=config.effective_feature_version,
            optuna_metric=config.effective_optuna_metric,
            target_mode=config.effective_target_mode,
            target_source=config.effective_target_source,
            base_model_methods=tuple(config.effective_base_model_methods),
            stack_enabled=config.stack_enabled,
            source_pipeline="notebooks/experiments/station_stacking_v15",
        )
        print(f"Exported {arm['arm']}: {exported_weights.bundle_path}")

write_station_comparisons(OUTPUT_DIR, STATION_ID, [arm["arm"] for arm in ARMS])
"""
        ),
        _markdown("## Arm Scoreboards\n"),
        _code(
            """scoreboards = pd.concat(
    [result.scoreboard.assign(arm=arm) for arm, result in results.items()],
    ignore_index=True,
) if results else pd.DataFrame()

scoreboards.sort_values(["period", "mae_f", "arm", "method"])
"""
        ),
        _markdown("## Common-Date Comparison Versus Base\n"),
        _code(
            """metrics_path = OUTPUT_DIR / f"{STATION_ID}_v15_arm_test_metrics.csv"
comparison_path = OUTPUT_DIR / f"{STATION_ID}_v15_arm_common_date_comparison.csv"

arm_test_metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
common_date_comparison = pd.read_csv(comparison_path) if comparison_path.exists() else pd.DataFrame()

common_date_comparison.sort_values(["method", "delta_mae_f", "comparison_arm"]) if not common_date_comparison.empty else common_date_comparison
"""
        ),
        _markdown("## Added Feature Selection Audit\n"),
        _code(
            """selected_rows = []
for arm, result in results.items():
    selected = set(result.feature_columns["feature"])
    selected_rows.append(
        {
            "arm": arm,
            "selected_feature_count": len(selected),
            "v15_added_features_selected": sorted(selected & set(V15_ADDITIONAL_FEATURE_COLUMNS)),
            "forecast_temp_features_selected": sorted(selected & set(V15_FORECAST_TEMP_AT_AS_OF_FEATURE_COLUMNS)),
            "precip_cloud_features_selected": sorted(selected & set(V15_PRECIP_CLOUD_FEATURE_COLUMNS)),
        }
    )

pd.DataFrame(selected_rows)
"""
        ),
        _markdown("## Added Feature Coverage\n"),
        _code(
            """coverage_frames = []
for arm, result in results.items():
    available = [column for column in V15_ADDITIONAL_FEATURE_COLUMNS if column in result.features]
    if not available:
        continue
    coverage = (
        result.features[available]
        .notna()
        .mean()
        .mul(100)
        .rename("coverage_pct")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    coverage["arm"] = arm
    coverage_frames.append(coverage)

added_feature_coverage = pd.concat(coverage_frames, ignore_index=True) if coverage_frames else pd.DataFrame()
added_feature_coverage.sort_values(["arm", "coverage_pct"], ascending=[True, False])
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
raw_weather_selected = []
for arm, result in results.items():
    selected = result.feature_columns["feature"].astype(str)
    raw_weather_selected.extend(
        {
            "arm": arm,
            "feature": feature,
        }
        for feature in selected
        if feature.startswith(("gfs_", "hrrr_", "nbm_"))
        and any(token in feature for token in raw_weather_tokens)
    )

pd.DataFrame(raw_weather_selected)
"""
        ),
        _markdown("## Rounded Within 1F Accuracy\n"),
        _code(
            """rounded_frames = []
for arm, result in results.items():
    preds = pd.concat(
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
    rounded = (
        preds
        .dropna(subset=["actual_high_f", "predicted_high_rounded_f"])
        .groupby(["period", "method"], as_index=False)
        .agg(
            count=("within_1f_after_round", "size"),
            within_1f_count=("within_1f_after_round", "sum"),
            within_1f_accuracy_pct=("within_1f_after_round", lambda x: x.mean() * 100),
        )
    )
    rounded["arm"] = arm
    rounded_frames.append(rounded)

within_1f_accuracy_by_period = pd.concat(rounded_frames, ignore_index=True) if rounded_frames else pd.DataFrame()
within_1f_accuracy_by_period.sort_values(["period", "method", "within_1f_accuracy_pct"], ascending=[True, True, False])
"""
        ),
        _markdown("## Bracket Metrics\n"),
        _code(
            """bracket_frames = []
for arm, result in results.items():
    bracket = result.bracket_metrics.copy()
    bracket["arm"] = arm
    bracket_frames.append(bracket)

pd.concat(bracket_frames, ignore_index=True) if bracket_frames else pd.DataFrame()
"""
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    for station in TARGET_STATIONS:
        notebook = _notebook(station)
        path = out_dir / f"stacking_{station}_v15.ipynb"
        path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
