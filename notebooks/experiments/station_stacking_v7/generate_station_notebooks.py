import json
from pathlib import Path


TARGET_STATIONS = ("KATL", "KAUS", "KORD", "KDAL", "KHOU", "KLAX", "KMIA", "KLGA", "KSEA")
TREND_COLUMNS = [
    "observed_temp_change_last_1h_f",
    "observed_temp_change_last_3h_f",
    "observed_morning_warmup_rate_f_per_hour",
    "observed_high_so_far_change_since_9am_f",
]


def _cell(cell_type: str, source: str) -> dict:
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def _notebook(station_id: str) -> dict:
    return {
        "cells": [
            _cell(
                "markdown",
                f"""# Station Stacking v7 - {station_id}

Current experimental notebook for `{station_id}`.

This version trains on live-safe same-day 11 AM GFS/HRRR timing, direct 13Z NBM raw-high data, source-owned v6 trend feature inputs, expanding year folds, and durable Optuna SQLite storage. Artifacts are written to `data/calibration/station_stacking_v7`.
""",
            ),
            _cell(
                "code",
                """from pathlib import Path
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

STATION_ID = "__STATION_ID__"
PROVIDERS = ("gfs", "hrrr", "nbm")
TIMING_MODE = "same_day_11am_live_safe"
FAST_MODE = False
OPTUNA_TRIALS = 50
STACK_OPTUNA_TRIALS = 50
OPTUNA_STARTUP_TRIALS = 20
STACK_OPTUNA_STARTUP_TRIALS = 20
OPTUNA_METRIC = "mae_f"
OPTUNA_VERBOSE = True
PROJECT_ROOT
""".replace("__STATION_ID__", station_id),
            ),
            _cell(
                "code",
                """import numpy as np
import pandas as pd

from src.calibration.station_stacking import (
    StationStackingConfig,
    V7_FEATURE_COLUMNS,
    YEAR_SPLIT_EXPANDING_FOLDS,
    missing_model_dependencies,
    provider_availability,
    run_station_year_split_experiment,
)
""",
            ),
            _cell(
                "markdown",
                """## V7 Contract

`feature_version="v7"` applies the source-owned v5 feature block and keeps the 11 AM observation trend columns. `timing_mode="same_day_11am_live_safe"` selects forecast cycles that would have been available by the bot decision time, while the current-observation trend cache falls back to the existing 11 AM observation timing.
""",
            ),
            _cell(
                "code",
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

fold_spec
""",
            ),
            _cell(
                "code",
                """TREND_COLUMNS = [
    "observed_temp_change_last_1h_f",
    "observed_temp_change_last_3h_f",
    "observed_morning_warmup_rate_f_per_hour",
    "observed_high_so_far_change_since_9am_f",
]

V7_FEATURE_COLUMNS
""",
            ),
            _cell(
                "markdown",
                """## Data Availability
""",
            ),
            _cell(
                "code",
                """availability = provider_availability(
    PROJECT_ROOT,
    timing_mode=TIMING_MODE,
    providers=PROVIDERS,
)

availability.loc[availability["station_id"].eq(STATION_ID)]
""",
            ),
            _cell(
                "markdown",
                """## Model Scores
""",
            ),
            _cell(
                "code",
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
    feature_version="v7",
    hyperparameter_space="wide",
    year_split_folds=YEAR_SPLIT_EXPANDING_FOLDS,
    year_split_test_train_years=(2021, 2025),
    year_split_test_year=2026,
    output_dir=PROJECT_ROOT / "data" / "calibration" / "station_stacking_v7",
)

config.resolved_optuna_storage_path()
""",
            ),
            _cell(
                "code",
                """result = run_station_year_split_experiment(config)
result.scoreboard
""",
            ),
            _cell(
                "markdown",
                """## NBM Raw High
""",
            ),
            _cell(
                "code",
                """nbm_raw_metrics = result.metrics.loc[result.metrics["method"].eq("nbm_raw")].copy()
nbm_raw_metrics
""",
            ),
            _cell(
                "markdown",
                """## Morning Trend Coverage
""",
            ),
            _cell(
                "code",
                """trend_coverage = (
    result.features[TREND_COLUMNS]
    .notna()
    .mean()
    .mul(100)
    .sort_values(ascending=False)
    .rename("coverage_pct")
    .reset_index()
    .rename(columns={"index": "feature"})
)

trend_coverage
""",
            ),
            _cell(
                "code",
                """result.feature_columns.loc[result.feature_columns["feature"].isin(TREND_COLUMNS)]
""",
            ),
            _cell(
                "markdown",
                """## Rounded Within 1F Accuracy
""",
            ),
            _cell(
                "code",
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
""",
            ),
            _cell(
                "markdown",
                """## Version Comparison
""",
            ),
            _cell(
                "code",
                """comparison_frames = []
for version, folder in [
    ("v1", PROJECT_ROOT / "data" / "calibration" / "station_stacking"),
    ("v2", PROJECT_ROOT / "data" / "calibration" / "station_stacking_v2"),
    ("v3", PROJECT_ROOT / "data" / "calibration" / "station_stacking_v3"),
    ("v4", PROJECT_ROOT / "data" / "calibration" / "station_stacking_v4"),
    ("v5", PROJECT_ROOT / "data" / "calibration" / "station_stacking_v5"),
    ("v6", PROJECT_ROOT / "data" / "calibration" / "station_stacking_v6"),
    ("v7", PROJECT_ROOT / "data" / "calibration" / "station_stacking_v7"),
]:
    path = folder / f"{STATION_ID}_year_split_scoreboard.csv"
    if path.exists():
        frame = pd.read_csv(path)
        frame["version"] = version
        comparison_frames.append(frame)

version_comparison = pd.concat(comparison_frames, ignore_index=True) if comparison_frames else pd.DataFrame()
if not version_comparison.empty:
    version_comparison = version_comparison.sort_values(["period", "mae_f", "version", "method"]).reset_index(drop=True)
version_comparison
""",
            ),
            _cell(
                "markdown",
                """## 2026 OOF Weather Brackets
""",
            ),
            _cell(
                "code",
                """result.bracket_metrics
""",
            ),
        ],
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
    for station_id in TARGET_STATIONS:
        path = out_dir / f"stacking_{station_id}_v7.ipynb"
        path.write_text(json.dumps(_notebook(station_id), indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
