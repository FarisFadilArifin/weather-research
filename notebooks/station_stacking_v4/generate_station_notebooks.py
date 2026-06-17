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
                f"""# Station Stacking v4 - {station_id}

Wide HRRR/GFS same-day 11am notebook for `{station_id}`.

This version keeps the v3 high-so-far features and adds SDK-backed precipitation features from HRRR/GFS forecast precipitation plus 11 AM METAR precipitation codes. Artifacts are written to `data/calibration/station_stacking_v4`.
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
                """## V4 Feature Engineering

Adds v2/v3 temperature features plus forecast and observed precipitation signals. Forecast precipitation comes from provider-prefixed SDK summary columns such as `gfs_forecast_precip_total_mm`, with `gfs_precip_amount` as a fallback for older caches.
""",
            ),
            _cell(
                "code",
                """import numpy as np
import pandas as pd
import src.calibration.station_stacking as station_stacking_module

V4_PROVIDERS = ("gfs", "hrrr")
V4_FEATURE_COLUMNS = [
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
    "v4_forecast_precip_total_mean_mm",
    "v4_forecast_precip_total_max_mm",
    "v4_forecast_precip_total_spread_mm",
    "v4_forecast_precip_max_1h_mean_mm",
    "v4_forecast_precip_hours_mean",
    "v4_forecast_precip_intensity_mean",
    "v4_forecast_precip_intensity_max",
    "v4_any_forecast_precip",
    "v4_all_forecast_precip",
    "v4_observed_precip_any",
    "v4_observed_precip_recent_mm_est",
    "v4_forecast_total_minus_observed_recent_mm",
    "v4_forecast_observed_precip_match",
    "v4_forecast_wet_observed_dry",
    "v4_observed_wet_forecast_dry",
    "v4_precip_humidity_interaction",
    "v4_precip_remaining_warmup_interaction",
]


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype="float64")


def _bool_num(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(0, index=frame.index, dtype="int64")
    series = frame[column]
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(int)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).gt(0).astype(int)
    return series.astype("string").str.lower().isin({"1", "true", "yes", "y"}).astype(int)


def _provider_series(frame: pd.DataFrame, provider: str, primary: str, fallback: str | None = None) -> pd.Series:
    primary_column = f"{provider}_{primary}"
    if primary_column in frame:
        return _num(frame, primary_column)
    if fallback is not None:
        return _num(frame, f"{provider}_{fallback}")
    return pd.Series(np.nan, index=frame.index, dtype="float64")


def _provider_matrix(frame: pd.DataFrame, primary: str, fallback: str | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        {provider: _provider_series(frame, provider, primary, fallback) for provider in V4_PROVIDERS},
        index=frame.index,
    )


def add_v4_feature_engineering(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    observed_temp = _num(out, "observed_temp_at_as_of_f")
    high_so_far = _num(out, "observed_high_temp_through_as_of_f")
    observed_humidity = _num(out, "observed_humidity_at_as_of")
    provider_mean = _num(out, "provider_mean_high_f")
    provider_spread = _num(out, "provider_spread_high_f")
    lag_1d = _num(out, "actual_high_lag_1d")
    roll_7d = _num(out, "actual_high_roll_7d_mean")
    roll_30d = _num(out, "actual_high_roll_30d_mean")

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

    precip_total = _provider_matrix(out, "forecast_precip_total_mm", fallback="precip_amount")
    precip_max_1h = _provider_matrix(out, "forecast_precip_max_1h_mm")
    precip_hours = _provider_matrix(out, "forecast_precip_hours_count")
    precip_intensity = _provider_matrix(out, "forecast_precip_intensity_code")
    has_precip = _provider_matrix(out, "forecast_has_precip").fillna(0).clip(lower=0, upper=1)

    out["v4_forecast_precip_total_mean_mm"] = precip_total.mean(axis=1)
    out["v4_forecast_precip_total_max_mm"] = precip_total.max(axis=1)
    out["v4_forecast_precip_total_spread_mm"] = precip_total.max(axis=1) - precip_total.min(axis=1)
    out["v4_forecast_precip_max_1h_mean_mm"] = precip_max_1h.mean(axis=1)
    out["v4_forecast_precip_hours_mean"] = precip_hours.mean(axis=1)
    out["v4_forecast_precip_intensity_mean"] = precip_intensity.mean(axis=1)
    out["v4_forecast_precip_intensity_max"] = precip_intensity.max(axis=1)
    out["v4_any_forecast_precip"] = has_precip.max(axis=1).fillna(0).astype(int)
    out["v4_all_forecast_precip"] = has_precip.min(axis=1).fillna(0).astype(int)

    observed_any = (
        _bool_num(out, "observed_is_raining_at_as_of")
        | _bool_num(out, "observed_is_drizzle_at_as_of")
        | _bool_num(out, "observed_is_snowing_at_as_of")
    ).astype(int)
    observed_recent_mm = _num(out, "observed_precip_recent_at_as_of") * 25.4
    out["v4_observed_precip_any"] = observed_any
    out["v4_observed_precip_recent_mm_est"] = observed_recent_mm
    out["v4_forecast_total_minus_observed_recent_mm"] = out["v4_forecast_precip_total_mean_mm"] - observed_recent_mm
    out["v4_forecast_observed_precip_match"] = out["v4_any_forecast_precip"].eq(observed_any).astype(int)
    out["v4_forecast_wet_observed_dry"] = (out["v4_any_forecast_precip"].eq(1) & observed_any.eq(0)).astype(int)
    out["v4_observed_wet_forecast_dry"] = (observed_any.eq(1) & out["v4_any_forecast_precip"].eq(0)).astype(int)
    out["v4_precip_humidity_interaction"] = out["v4_forecast_precip_total_mean_mm"] * (observed_humidity / 100.0)
    out["v4_precip_remaining_warmup_interaction"] = out["v4_forecast_precip_total_mean_mm"] * remaining_warmup
    return out


if not hasattr(station_stacking_module, "_v4_original_build_station_wide_dataset"):
    station_stacking_module._v4_original_build_station_wide_dataset = station_stacking_module.build_station_wide_dataset


def build_station_wide_dataset_v4(*args, **kwargs):
    frame = station_stacking_module._v4_original_build_station_wide_dataset(*args, **kwargs)
    return add_v4_feature_engineering(frame)


station_stacking_module.build_station_wide_dataset = build_station_wide_dataset_v4
V4_FEATURE_COLUMNS
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
    output_dir=PROJECT_ROOT / "data" / "calibration" / "station_stacking_v4",
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
        path = out_dir / f"stacking_{station_id}_v4.ipynb"
        path.write_text(json.dumps(_notebook(station_id), indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
