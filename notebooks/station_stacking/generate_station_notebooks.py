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
                f"""# Station Stacking - {station_id}

Wide 3-provider same-day 11am notebook for `{station_id}`.

Set `FAST_MODE = False` after the provider pulls are complete to run the fuller walk-forward experiment.
""",
            ),
            _cell(
                "code",
                """from pathlib import Path
import sys

PROJECT_ROOT = Path.cwd().resolve()
while not (PROJECT_ROOT / "src" / "calibration" / "station_stacking.py").exists():
    if PROJECT_ROOT.parent == PROJECT_ROOT:
        raise RuntimeError("Could not find project root containing src/calibration/station_stacking.py")
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

STATION_ID = "__STATION_ID__"
FAST_MODE = True
PROJECT_ROOT
""".replace("__STATION_ID__", station_id),
            ),
            _cell(
                "code",
                """import pandas as pd

from src.calibration.station_stacking import (
    BASELINE_METHODS,
    BASE_MODEL_METHODS,
    PROVIDER_NUMERIC_COLUMNS,
    STACK_METHOD,
    StationStackingConfig,
    build_station_wide_dataset,
    provider_availability,
    run_station_stacking_experiment,
)
""",
            ),
            _cell(
                "markdown",
                """## Provider Availability

This is a quick check of which same-day 11am provider rows are already available for this station.
""",
            ),
            _cell(
                "code",
                """availability = provider_availability(PROJECT_ROOT)
station_availability = availability.loc[availability["station_id"].eq(STATION_ID)].copy()
station_availability
""",
            ),
            _cell(
                "markdown",
                """## Wide Feature Table

One row per station-date. GFS, HRRR, and NBM forecasts are side-by-side, with leak-safe lagged actual/error features.
""",
            ),
            _cell(
                "code",
                """features = build_station_wide_dataset(PROJECT_ROOT, station_id=STATION_ID)
feature_summary = pd.DataFrame(
    [
        {
            "station_id": STATION_ID,
            "rows": len(features),
            "complete_3_provider_rows": int(features["all_provider_highs_available"].sum()),
            "first_contract_date": features["contract_date"].min(),
            "last_contract_date": features["contract_date"].max(),
        }
    ]
)
display(feature_summary)
features.tail()
""",
            ),
            _cell(
                "code",
                """provider_schema = pd.DataFrame(
    [
        {
            "provider": provider,
            "feature": feature,
            "column": f"{provider}_{feature}" if feature != "raw_forecast_high_f" else f"{provider}_high_f",
            "present": (f"{provider}_{feature}" if feature != "raw_forecast_high_f" else f"{provider}_high_f") in features.columns,
            "non_null_rows": int(
                features[
                    f"{provider}_{feature}" if feature != "raw_forecast_high_f" else f"{provider}_high_f"
                ].notna().sum()
            )
            if (f"{provider}_{feature}" if feature != "raw_forecast_high_f" else f"{provider}_high_f") in features.columns
            else 0,
        }
        for provider in ("gfs", "hrrr", "nbm")
        for feature in PROVIDER_NUMERIC_COLUMNS
    ]
)
provider_schema.pivot_table(
    index="feature",
    columns="provider",
    values="non_null_rows",
    aggfunc="first",
).loc[PROVIDER_NUMERIC_COLUMNS]
""",
            ),
            _cell(
                "markdown",
                """## Baselines, Base Models, And Stack

Raw provider baselines are evaluated first. XGBoost, LightGBM, and CatBoost are shown separately, then the Ridge meta-model is evaluated on leak-safe walk-forward base predictions.
""",
            ),
            _cell(
                "code",
                """config = StationStackingConfig(
    station_id=STATION_ID,
    project_root=PROJECT_ROOT,
    fast_mode=FAST_MODE,
)
result = run_station_stacking_experiment(config)
result.metrics
""",
            ),
            _cell(
                "code",
                """separate_model_methods = [*BASE_MODEL_METHODS, STACK_METHOD]
result.metrics.loc[result.metrics["method"].isin(separate_model_methods)].sort_values(
    ["evaluation_scope", "mae_f", "method"]
)
""",
            ),
            _cell(
                "markdown",
                """## Predictions And Output Files
""",
            ),
            _cell(
                "code",
                """result.predictions.tail(30)
""",
            ),
            _cell(
                "code",
                """prediction_wide = result.predictions.pivot_table(
    index="contract_date",
    columns="method",
    values="predicted_high_f",
    aggfunc="first",
)
actual = result.predictions.groupby("contract_date")["actual_high_f"].first()
plot_frame = prediction_wide.join(actual)
if not plot_frame.empty:
    ax = plot_frame.tail(120).plot(figsize=(14, 6), title=f"{STATION_ID} predictions vs actual")
    ax.set_ylabel("Daily high (F)")
else:
    print("No predictions available yet. Re-run after all 3 provider pulls have enough overlapping dates.")
""",
            ),
            _cell(
                "code",
                """display(result.feature_columns)
result.output_paths
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
        path = out_dir / f"stacking_{station_id}.ipynb"
        path.write_text(json.dumps(_notebook(station_id), indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
