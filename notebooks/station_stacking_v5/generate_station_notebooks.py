from __future__ import annotations

import json
from pathlib import Path


TARGET_STATIONS = ("KATL", "KAUS", "KORD", "KDAL", "KHOU", "KLAX", "KMIA", "KLGA", "KSEA")


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
                f"""# Station Stacking v5 - {station_id}

Wide HRRR/GFS same-day 11am notebook for `{station_id}`.

This version uses the same feature engineering as v4, but tunes Optuna trials against MAE instead of RMSE. Artifacts are written to `data/calibration/station_stacking_v5`.
""",
            ),
            _cell(
                "code",
                """from pathlib import Path
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

STATION_ID = "__STATION_ID__"
FAST_MODE = False
OPTUNA_TRIALS = 100
STACK_OPTUNA_TRIALS = 50
OPTUNA_METRIC = "mae_f"
OPTUNA_VERBOSE = True
PROJECT_ROOT
""".replace("__STATION_ID__", station_id),
            ),
            _cell(
                "code",
                """from src.calibration.station_stacking import (
    StationStackingConfig,
    V5_FEATURE_COLUMNS,
    missing_model_dependencies,
    run_station_year_split_experiment,
)
""",
            ),
            _cell(
                "markdown",
                """## V5 Feature Engineering

Source-owned `feature_version="v5"` feature set: v2/v3 temperature features plus forecast and observed precipitation signals. Forecast precipitation comes from provider-prefixed SDK summary columns such as `gfs_forecast_precip_total_mm`, with `gfs_precip_amount` as a fallback for older caches.
""",
            ),
            _cell(
                "code",
                """import numpy as np
import pandas as pd

V5_FEATURE_COLUMNS
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
    fast_mode=FAST_MODE,
    optuna_trials=OPTUNA_TRIALS,
    stack_optuna_trials=STACK_OPTUNA_TRIALS,
    optuna_metric=OPTUNA_METRIC,
    optuna_verbose=OPTUNA_VERBOSE,
    feature_version="v5",
    output_dir=PROJECT_ROOT / "data" / "calibration" / "station_stacking_v5",
)
result = run_station_year_split_experiment(config)
result.scoreboard
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
        result.test_predictions.assign(period="test_2026"),
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
                """## Precipitation Feature Coverage
""",
            ),
            _cell(
                "code",
                """precip_columns = [column for column in result.features.columns if "precip" in column.lower()]
coverage = (
    result.features[precip_columns]
    .notna()
    .mean()
    .mul(100)
    .sort_values(ascending=False)
    .rename("coverage_pct")
    .reset_index()
    .rename(columns={"index": "feature"})
)
coverage
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
                """## 2026 Weather Brackets
""",
            ),
            _cell(
                "code",
                """result.bracket_metrics
""",
            ),
            _cell(
                "code",
                """import pandas as pd


def adjacent_brackets(bracket):
    if pd.isna(bracket):
        return []
    text = str(bracket).strip()
    if not text or "-" not in text:
        return []
    try:
        lower = int(text.split("-", 1)[0])
    except ValueError:
        return []
    return [
        f"{lower - 2}-{lower - 1}",
        f"{lower}-{lower + 1}",
        f"{lower + 2}-{lower + 3}",
    ]


bracket_3way = result.bracket_predictions.copy()

valid = bracket_3way["actual_bracket"].notna() & bracket_3way["predicted_bracket"].astype(str).str.strip().ne("")
bracket_3way = bracket_3way.loc[valid].copy()
bracket_3way["picked_brackets"] = bracket_3way["predicted_bracket"].map(adjacent_brackets)
bracket_3way["three_bracket_hit"] = bracket_3way.apply(
    lambda row: row["actual_bracket"] in row["picked_brackets"],
    axis=1,
)

three_bracket_accuracy = (
    bracket_3way
    .groupby("method", as_index=False)
    .agg(
        count=("three_bracket_hit", "size"),
        exact_bracket_accuracy_pct=("bracket_hit", lambda x: x.mean() * 100),
        three_bracket_accuracy_pct=("three_bracket_hit", lambda x: x.mean() * 100),
    )
    .sort_values("three_bracket_accuracy_pct", ascending=False)
)

three_bracket_accuracy
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
    for station_id in TARGET_STATIONS:
        path = out_dir / f"stacking_{station_id}_v5.ipynb"
        path.write_text(json.dumps(_notebook(station_id), indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
