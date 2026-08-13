from __future__ import annotations

import json
from pathlib import Path


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def notebook() -> dict:
    cells = [
        md(
            """# KATL/KDAL Directional Residual Audit V1

This notebook asks whether the V20 regression error direction is predictable in stable subgroups.
It is a diagnostic, not a boundary eligibility rule: every row is retained.

Subgroups are fixed from live-safe forecast context. Signals must have adequate sample sizes, the
same direction in 2023, 2024, and 2025, a minimum per-year effect, and Benjamini-Hochberg adjusted
`q < 0.05`. The untouched 2026 rows only confirm or reject development signals.

Two questions are tested separately:

1. **Residual direction:** is the actual high above or below the point prediction?
2. **Bucket utility:** among decisive actionable cases, would the alternative bucket recover more
   losses than it damages half-up wins?
"""
        ),
        md("## Environment\n"),
        code(
            """from pathlib import Path
import sys
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

warnings.simplefilter("ignore", PerformanceWarning)
PROJECT_ROOT = Path.cwd().resolve()
while PROJECT_ROOT != PROJECT_ROOT.parent and not (PROJECT_ROOT / "pyproject.toml").is_file():
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.directional_residual_audit import (
    audit_directional_residual_result,
    run_directional_residual_audit,
)
from src.calibration.round_override_v4 import build_utility_override_frame
from src.calibration.v19_bucket import crossfit_ridge_predictions

STATIONS = {
    "KATL": {"pipeline": PROJECT_ROOT / "data/calibration/station_stacking_v20_peak_timing", "peak": True},
    "KDAL": {"pipeline": PROJECT_ROOT / "data/calibration/station_stacking_v20_kdal_no_peak", "peak": False},
}
print(PROJECT_ROOT)
"""
        ),
        md("## Build honest 2023-2026 point-prediction frames\n"),
        code(
            """frames, sources = {}, {}
for station, config in STATIONS.items():
    features = pd.read_csv(config["pipeline"] / f"{station}_features.csv", low_memory=False)
    validation = pd.read_csv(config["pipeline"] / f"{station}_year_split_validation_predictions.csv")
    test = pd.read_csv(config["pipeline"] / f"{station}_year_split_test_predictions.csv")
    development_point = crossfit_ridge_predictions(validation)
    confirmation_point = test.loc[
        test["method"].eq("ridge_stack"),
        ["contract_date", "actual_high_f", "predicted_high_f"],
    ].copy()
    confirmation_point["train_through_year"] = 2025
    confirmation_point["validation_year"] = 2026
    point = pd.concat([development_point, confirmation_point], ignore_index=True)
    base = pd.concat([validation, test], ignore_index=True)
    frame = build_utility_override_frame(
        features, point, base, include_peak_features=config["peak"]
    )
    frames[station] = frame
    sources[station] = {"features": features, "validation": validation, "test": test, "point": point}

pd.DataFrame([
    {
        "station": station,
        "rows": len(frame),
        "start": frame["contract_date"].min().date(),
        "end": frame["contract_date"].max().date(),
        **{f"rows_{year}": int(frame["year"].eq(year).sum()) for year in (2023, 2024, 2025, 2026)},
    }
    for station, frame in frames.items()
])
"""
        ),
        md("## Run fixed subgroup and multiple-testing audit\n"),
        code(
            """results = {
    station: run_directional_residual_audit(frame, station_id=station)
    for station, frame in frames.items()
}
yearly_metrics = pd.concat(
    [result["yearly_group_metrics"] for result in results.values()], ignore_index=True
)
residual_stability = pd.concat(
    [result["residual_stability"] for result in results.values()], ignore_index=True
)
utility_stability = pd.concat(
    [result["utility_stability"] for result in results.values()], ignore_index=True
)
print("Audit analysis complete")
"""
        ),
        md("## Overall regression residual direction\n"),
        code(
            """overall_rows = []
for station, result in results.items():
    frame = result["frame"]
    for year, part in frame.groupby("year"):
        decisive = part.loc[part["decisive_override"]]
        overall_rows.append({
            "station": station,
            "year": int(year),
            "rows": len(part),
            "mean_residual_f": part["residual_f"].mean(),
            "median_residual_f": part["residual_f"].median(),
            "underprediction_rate": part["underprediction_target"].mean(),
            "actionable_rows": int(part["override_actionable"].sum()),
            "decisive_rows": len(decisive),
            "alternative_recovery_share": decisive["recovery_target"].mean(),
        })
overall_metrics = pd.DataFrame(overall_rows)
overall_metrics
"""
        ),
        md("## Development-stable residual-direction signals\n"),
        code(
            """signal_columns = [
    "station", "group_name", "group_value", "development_count", "development_rate",
    "expected_direction", "minimum_year_count", "worst_year_edge", "fdr_q_value",
    "confirmation_count_2026", "confirmation_rate_2026", "confirmation_matches_2026",
    "confirmed_stable_signal",
]
stable_residual = residual_stability.loc[residual_stability["stable_development_signal"]]
stable_residual[signal_columns].sort_values(["station", "fdr_q_value"])
"""
        ),
        code(
            """confirmed_residual = stable_residual.loc[stable_residual["confirmed_stable_signal"]]
if not confirmed_residual.empty:
    labels = confirmed_residual["station"] + " | " + confirmed_residual["group_name"] + " | " + confirmed_residual["group_value"]
    plot = confirmed_residual.assign(label=labels).set_index("label")[["development_rate", "confirmation_rate_2026"]]
    ax = plot.plot(kind="barh", figsize=(10, max(4, len(plot) * 0.55)))
    ax.axvline(0.5, color="black", linewidth=1)
    ax.set_xlabel("P(actual > point prediction)")
    ax.set_title("Residual-direction signals confirmed in 2026")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.show()
"""
        ),
        md("## Development-stable alternative-bucket utility signals\n"),
        code(
            """stable_utility = utility_stability.loc[utility_stability["stable_development_signal"]]
stable_utility[signal_columns].sort_values(["station", "fdr_q_value"])
"""
        ),
        md("## What survives the untouched 2026 confirmation?\n"),
        code(
            """summary = pd.DataFrame([
    {
        "station": station,
        "development_stable_residual_signals": int(result["residual_stability"]["stable_development_signal"].sum()),
        "confirmed_residual_signals_2026": int(result["residual_stability"]["confirmed_stable_signal"].sum()),
        "development_stable_utility_signals": int(result["utility_stability"]["stable_development_signal"].sum()),
        "confirmed_utility_signals_2026": int(result["utility_stability"]["confirmed_stable_signal"].sum()),
    }
    for station, result in results.items()
])
summary
"""
        ),
        code(
            """assert int(summary["confirmed_utility_signals_2026"].sum()) == 0
print(
    "Conclusion: some continuous residual-direction patterns repeat, but no subgroup demonstrates "
    "a development-stable and 2026-confirmed advantage for changing the half-up bucket."
)
"""
        ),
        md("## Executable integrity audit\n"),
        code(
            """audit_report = pd.concat(
    [
        audit_directional_residual_result(frames[station], result).assign(station=station)
        for station, result in results.items()
    ],
    ignore_index=True,
)[["station", "audit", "passed", "detail"]]
display(audit_report)
failed = audit_report.loc[~audit_report["passed"]]
assert failed.empty, failed.to_dict(orient="records")
print(f"All {len(audit_report)} directional residual audits passed")
"""
        ),
        md("## Export audit artifacts\n"),
        code(
            """OUTPUT_ROOT = PROJECT_ROOT / "data/calibration/directional_residual_audit_v1"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
overall_metrics.to_csv(OUTPUT_ROOT / "overall_yearly_metrics.csv", index=False)
yearly_metrics.to_csv(OUTPUT_ROOT / "yearly_subgroup_metrics.csv", index=False)
residual_stability.to_csv(OUTPUT_ROOT / "residual_direction_stability.csv", index=False)
utility_stability.to_csv(OUTPUT_ROOT / "alternative_bucket_utility_stability.csv", index=False)
summary.to_csv(OUTPUT_ROOT / "signal_summary.csv", index=False)
audit_report.to_csv(OUTPUT_ROOT / "audit.csv", index=False)
print(OUTPUT_ROOT)
"""
        ),
        md(
            """## Interpretation contract

A confirmed residual-direction signal means a subgroup tends to be above or below the continuous
point prediction. It does **not** automatically justify changing the 2-degree bucket. Only the
alternative-bucket utility test answers that decision question. No utility subgroup may be used
unless it passes development stability and untouched 2026 confirmation.
"""
        ),
    ]
    for index, cell in enumerate(cells):
        cell["id"] = f"directional-audit-v1-{index:02d}"
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    output = Path(__file__).resolve().parent / "directional_residual_audit_v1.ipynb"
    output.write_text(json.dumps(notebook(), indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
