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
                f"""# Station Stacking v3 - {station_id}

Wide HRRR/GFS same-day 11am notebook for `{station_id}`.

This version keeps the v2 notebook feature engineering and adds SDK-backed 11 AM high-so-far features from `observed_high_temp_through_as_of_f`. Artifacts are written to `data/calibration/station_stacking_v3`.
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
OPTUNA_TRIALS = 10
STACK_OPTUNA_TRIALS = 10
OPTUNA_VERBOSE = True
PROJECT_ROOT
""".replace("__STATION_ID__", station_id),
            ),
            _cell(
                "code",
                """from src.calibration.station_stacking import (
    StationStackingConfig,
    missing_model_dependencies,
    run_station_year_split_experiment,
)
""",
            ),
            _cell(
                "markdown",
                """## V3 Feature Engineering

Adds v2 features plus SDK-backed 11 AM high-so-far signals. The high-so-far feature is computed from same-day station observations at or before the 11 AM local as-of time.
""",
            ),
            _cell(
                "code",
                """import numpy as np
import pandas as pd
import src.calibration.station_stacking as station_stacking_module

V3_FEATURE_COLUMNS = [
    "v2_recent_heat_anomaly_f",
    "v2_recent_heat_momentum_f",
    "v2_morning_warmup_to_consensus_f",
    "v2_consensus_minus_7d_actual_f",
    "v2_spread_per_warmup_f",
    "v2_humidity_warmup_interaction",
    "v3_high_so_far_above_current_f",
    "v3_remaining_warmup_from_high_so_far_f",
    "v3_high_so_far_minus_lag_1d_f",
    "v3_high_so_far_minus_7d_actual_f",
    "v3_remaining_warmup_per_spread_f",
    "v3_humidity_remaining_warmup_interaction",
]


def add_v3_feature_engineering(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    observed_temp = pd.to_numeric(out.get("observed_temp_at_as_of_f"), errors="coerce")
    high_so_far = pd.to_numeric(out.get("observed_high_temp_through_as_of_f"), errors="coerce")
    observed_humidity = pd.to_numeric(out.get("observed_humidity_at_as_of"), errors="coerce")
    provider_mean = pd.to_numeric(out.get("provider_mean_high_f"), errors="coerce")
    provider_spread = pd.to_numeric(out.get("provider_spread_high_f"), errors="coerce")
    lag_1d = pd.to_numeric(out.get("actual_high_lag_1d"), errors="coerce")
    roll_7d = pd.to_numeric(out.get("actual_high_roll_7d_mean"), errors="coerce")
    roll_30d = pd.to_numeric(out.get("actual_high_roll_30d_mean"), errors="coerce")

    warmup_to_consensus = provider_mean - observed_temp
    remaining_warmup = provider_mean - high_so_far
    out["v2_recent_heat_anomaly_f"] = lag_1d - roll_30d
    out["v2_recent_heat_momentum_f"] = roll_7d - roll_30d
    out["v2_morning_warmup_to_consensus_f"] = warmup_to_consensus
    out["v2_consensus_minus_7d_actual_f"] = provider_mean - roll_7d
    out["v2_spread_per_warmup_f"] = provider_spread / warmup_to_consensus.abs().clip(lower=1.0)
    out["v2_humidity_warmup_interaction"] = (observed_humidity / 100.0) * warmup_to_consensus
    out["v3_high_so_far_above_current_f"] = high_so_far - observed_temp
    out["v3_remaining_warmup_from_high_so_far_f"] = remaining_warmup
    out["v3_high_so_far_minus_lag_1d_f"] = high_so_far - lag_1d
    out["v3_high_so_far_minus_7d_actual_f"] = high_so_far - roll_7d
    out["v3_remaining_warmup_per_spread_f"] = remaining_warmup / provider_spread.abs().clip(lower=1.0)
    out["v3_humidity_remaining_warmup_interaction"] = (observed_humidity / 100.0) * remaining_warmup
    return out


if not hasattr(station_stacking_module, "_v3_original_build_station_wide_dataset"):
    station_stacking_module._v3_original_build_station_wide_dataset = station_stacking_module.build_station_wide_dataset


def build_station_wide_dataset_v3(*args, **kwargs):
    frame = station_stacking_module._v3_original_build_station_wide_dataset(*args, **kwargs)
    return add_v3_feature_engineering(frame)


station_stacking_module.build_station_wide_dataset = build_station_wide_dataset_v3
V3_FEATURE_COLUMNS
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
    optuna_verbose=OPTUNA_VERBOSE,
    output_dir=PROJECT_ROOT / "data" / "calibration" / "station_stacking_v3",
)
result = run_station_year_split_experiment(config)
result.scoreboard
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
        path = out_dir / f"stacking_{station_id}_v3.ipynb"
        path.write_text(json.dumps(_notebook(station_id), indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
