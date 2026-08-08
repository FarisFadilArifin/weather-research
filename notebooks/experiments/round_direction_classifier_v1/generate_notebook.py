from __future__ import annotations

import json
from pathlib import Path


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


def notebook() -> dict:
    cells = [
        _markdown(
            """# KATL/KDAL V20 Binary Floor-or-Ceil Classifier

This notebook implements the exact binary task: `0 = floor(regression prediction)` and
`1 = ceil(regression prediction)`. Every honest regression prediction receives one class. There are
no boundary windows, eligibility filters, abstentions, or tuned action thresholds. The class cutoff
is fixed at 0.5.

KATL uses V20 peak-timing features and KDAL uses V20 no-peak. Hyperparameters are selected on an
inner chronological window, probabilities are Platt-calibrated on a later prior window, and the
following year is evaluated without leakage.
"""
        ),
        _markdown("## Environment and source contracts\n"),
        _code(
            """from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path.cwd().resolve()
while PROJECT_ROOT != PROJECT_ROOT.parent and not (PROJECT_ROOT / "pyproject.toml").is_file():
    PROJECT_ROOT = PROJECT_ROOT.parent
if not (PROJECT_ROOT / "pyproject.toml").is_file():
    raise RuntimeError("Run inside the weather-research repository")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.round_direction import (
    audit_round_direction_system,
    build_round_direction_frame,
    continuous_round_direction_comparison,
    fit_round_direction_system,
    predict_round_direction,
    round_direction_metrics,
    serializable_round_bundle,
)
from src.calibration.v19_bucket import crossfit_ridge_predictions

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 180)

STATIONS = {
    "KATL": {
        "pipeline_dir": PROJECT_ROOT / "data/calibration/station_stacking_v20_peak_timing",
        "include_peak_features": True,
        "continuous_forward": PROJECT_ROOT / "data/calibration/station_continuous_residual_probability/KATL/KATL_forward_continuous_predictions.csv",
    },
    "KDAL": {
        "pipeline_dir": PROJECT_ROOT / "data/calibration/station_stacking_v20_kdal_no_peak",
        "include_peak_features": False,
        "continuous_forward": PROJECT_ROOT / "data/calibration/station_continuous_residual_probability/KDAL/KDAL_forward_continuous_predictions.csv",
    },
}
print(PROJECT_ROOT)
"""
        ),
        _markdown("## Build the all-row binary target\n"),
        _code(
            """frames = {}
sources = {}
summary_rows = []
for station, config in STATIONS.items():
    pipeline = config["pipeline_dir"]
    features = pd.read_csv(pipeline / f"{station}_features.csv", low_memory=False)
    validation = pd.read_csv(pipeline / f"{station}_year_split_validation_predictions.csv")
    point = crossfit_ridge_predictions(validation)
    frame = build_round_direction_frame(
        features,
        point,
        validation,
        include_peak_features=config["include_peak_features"],
    )
    frames[station] = frame
    sources[station] = {"features": features, "validation": validation, "point": point}
    summary_rows.append(
        {
            "station": station,
            "rows": len(frame),
            "start": frame["contract_date"].min().date(),
            "end": frame["contract_date"].max().date(),
            "round_up_rate": frame["round_up"].mean(),
            "floor_ceil_change_bucket_rate": frame["floor_ceil_change_bucket"].mean(),
            "default_half_up_direction_accuracy": frame["default_half_up"].eq(frame["round_up"]).mean(),
        }
    )
dataset_summary = pd.DataFrame(summary_rows)
dataset_summary
"""
        ),
        _markdown(
            """### Label definition

For a non-integer prediction `p`, class `1` is correct when the integer settlement is greater than
`p`; otherwise class `0` is correct. The selected corrected degree is always exactly `ceil(p)` or
`floor(p)`. When floor and ceil map to the same 2°F bucket, the classification still exists but
cannot change the market bucket.
"""
        ),
        _code(
            """example_columns = [
    "contract_date", "point_prediction_f", "floor_degree_f", "ceil_degree_f",
    "prediction_fraction_f", "round_up", "default_half_up", "floor_bucket_label",
    "ceil_bucket_label", "floor_ceil_change_bucket", "actual_high_f",
]
pd.concat(
    [frame.assign(station=station).head(5) for station, frame in frames.items()],
    ignore_index=True,
)[["station", *example_columns]]
"""
        ),
        _markdown("## Chronological hyperparameter selection and probability calibration\n"),
        _code(
            """results = {}
for station, config in STATIONS.items():
    print(f"Training {station}...")
    results[station] = fit_round_direction_system(
        frames[station],
        station_id=station,
        include_peak_features=config["include_peak_features"],
    )
print("Training complete")
"""
        ),
        _markdown("## Selected candidates and chronological cutoffs\n"),
        _code(
            """fold_states = pd.concat(
    [result["fold_states"].assign(station=station) for station, result in results.items()],
    ignore_index=True,
)
fold_states[[
    "station", "validation_year", "selected_family", "selected_params_json",
    "model_training_cutoff", "calibration_start", "calibration_cutoff",
]]
"""
        ),
        _markdown("## Honest 2024–2025 result\n"),
        _code(
            """forward_metrics = pd.concat(
    [result["forward_metrics"].assign(station=station) for station, result in results.items()],
    ignore_index=True,
)
metric_columns = [
    "station", "period", "count", "round_up_rate", "binary_log_loss", "brier", "ece",
    "roc_auc", "direction_accuracy", "default_half_up_direction_accuracy",
    "point_degree_exact_accuracy", "corrected_degree_exact_accuracy",
    "point_bucket_hit_rate", "corrected_bucket_hit_rate", "bucket_hit_rate_lift",
    "actionable_count", "actionable_direction_accuracy", "bucket_switch_count",
    "recovered_losses", "damaged_wins", "net_recovered_wins", "recovery_damage_ratio",
]
forward_metrics[metric_columns]
"""
        ),
        _code(
            """combined = forward_metrics.loc[forward_metrics["period"].eq("2024-2025")]
plot_data = combined.set_index("station")[["point_bucket_hit_rate", "corrected_bucket_hit_rate"]]
ax = plot_data.plot(kind="bar", figsize=(8, 4), ylim=(0, max(0.6, plot_data.to_numpy().max() + 0.05)))
ax.set_title("Half-up regression bucket vs binary floor/ceil correction")
ax.set_ylabel("Bucket hit rate")
ax.grid(axis="y", alpha=0.25)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
"""
        ),
        _markdown("## Continuous-residual round-direction baseline\n"),
        _code(
            """comparison_rows = []
for station, config in STATIONS.items():
    continuous = pd.read_csv(config["continuous_forward"])
    baseline = continuous_round_direction_comparison(
        results[station]["forward_predictions"], continuous
    ).assign(station=station, model="continuous_student_t")
    comparison_rows.append(baseline)
    selected = results[station]["forward_metrics"].loc[
        results[station]["forward_metrics"]["period"].eq("2024-2025")
    ].copy()
    selected["station"] = station
    selected["model"] = "binary_floor_ceil_classifier"
    comparison_rows.append(selected)
continuous_comparison = pd.concat(comparison_rows, ignore_index=True)
continuous_comparison[[
    "station", "model", "count", "binary_log_loss", "brier", "ece", "roc_auc",
    "direction_accuracy", "point_bucket_hit_rate", "corrected_bucket_hit_rate",
    "net_recovered_wins",
]].sort_values(["station", "binary_log_loss"])
"""
        ),
        _markdown("## Exploratory 2026 result\n"),
        _code(
            """holdout_predictions = {}
holdout_rows = []
for station, config in STATIONS.items():
    source = sources[station]
    test = pd.read_csv(config["pipeline_dir"] / f"{station}_year_split_test_predictions.csv")
    test_point = test.loc[
        test["method"].eq("ridge_stack"),
        ["contract_date", "actual_high_f", "predicted_high_f"],
    ]
    all_point = pd.concat(
        [source["point"][["contract_date", "actual_high_f", "predicted_high_f"]], test_point],
        ignore_index=True,
    )
    all_base = pd.concat([source["validation"], test], ignore_index=True)
    all_frame = build_round_direction_frame(
        source["features"], all_point, all_base,
        include_peak_features=config["include_peak_features"],
    )
    holdout = all_frame.loc[all_frame["year"].eq(2026)]
    predicted = predict_round_direction(results[station]["final_state"], holdout)
    holdout_predictions[station] = predicted
    holdout_rows.append({"station": station, **round_direction_metrics(predicted)})
holdout_metrics = pd.DataFrame(holdout_rows)
holdout_metrics[[
    "station", "count", "binary_log_loss", "ece", "direction_accuracy",
    "point_bucket_hit_rate", "corrected_bucket_hit_rate", "bucket_hit_rate_lift",
    "bucket_switch_count", "recovered_losses", "damaged_wins", "net_recovered_wins",
]]
"""
        ),
        _markdown("## Executable correctness and leakage audit\n"),
        _code(
            """audit_parts = []
for station, config in STATIONS.items():
    audit_parts.append(
        audit_round_direction_system(
            frames[station], results[station],
            include_peak_features=config["include_peak_features"],
        ).assign(station=station)
    )
    frame = frames[station]
    audit_parts.append(
        pd.DataFrame(
            [
                {
                    "station": station,
                    "audit": "unique_station_dates",
                    "passed": not frame["contract_date"].duplicated().any(),
                    "detail": f"duplicate_dates={int(frame['contract_date'].duplicated().sum())}",
                },
                {
                    "station": station,
                    "audit": "integer_settlement_labels",
                    "passed": bool(np.allclose(frame["actual_high_f"], np.round(frame["actual_high_f"]))),
                    "detail": f"rows={len(frame)}",
                },
                {
                    "station": station,
                    "audit": "no_boundary_window_or_abstention",
                    "passed": "nearest_boundary_distance_f" not in results[station]["feature_names"],
                    "detail": "every outer row must receive class 0 or 1",
                },
            ]
        )
    )
audit_report = pd.concat(audit_parts, ignore_index=True)[["station", "audit", "passed", "detail"]]
display(audit_report)
failed = audit_report.loc[~audit_report["passed"]]
assert failed.empty, failed.to_dict(orient="records")
print(f"All {len(audit_report)} audits passed")
"""
        ),
        _markdown("## Export research artifacts\n"),
        _code(
            """OUTPUT_ROOT = PROJECT_ROOT / "data/calibration/station_round_direction_classifier"
export_rows = []
for station, result in results.items():
    output = OUTPUT_ROOT / station
    weights = output / "model_weights"
    weights.mkdir(parents=True, exist_ok=True)
    result["forward_predictions"].to_csv(output / f"{station}_forward_round_predictions.csv", index=False)
    result["forward_metrics"].to_csv(output / f"{station}_forward_round_metrics.csv", index=False)
    result["fold_states"].to_csv(output / f"{station}_round_fold_states.csv", index=False)
    result["tuning"].to_csv(output / f"{station}_round_inner_tuning.csv", index=False)
    holdout_predictions[station].to_csv(output / f"{station}_2026_round_predictions.csv", index=False)
    bundle_path = weights / f"{station}_floor_ceil_classifier_v1.joblib"
    joblib.dump(serializable_round_bundle(result), bundle_path)
    export_rows.append(
        {
            "station": station,
            "bundle": str(bundle_path),
            "selected_family": result["final_state"]["selected_spec"].family,
            "class_threshold": 0.5,
        }
    )
audit_report.to_csv(OUTPUT_ROOT / "round_direction_audit.csv", index=False)
dataset_summary.to_csv(OUTPUT_ROOT / "dataset_summary.csv", index=False)
continuous_comparison.to_csv(OUTPUT_ROOT / "continuous_baseline_comparison.csv", index=False)
holdout_metrics.to_csv(OUTPUT_ROOT / "exploratory_2026_metrics.csv", index=False)
pd.DataFrame(export_rows)
"""
        ),
        _markdown(
            """## Decision rule

The model always emits one binary action: probability at least 0.5 selects `ceil`, otherwise
`floor`. Deployment requires a positive honest bucket-hit lift and net recovered wins versus the
existing half-up regression. The exploratory 2026 section cannot override the 2024–2025 result.
"""
        ),
    ]
    for index, cell in enumerate(cells):
        cell["id"] = f"round-{index:02d}"
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
    output = Path(__file__).resolve().parent / "floor_ceil_classifier_v1.ipynb"
    output.write_text(json.dumps(notebook(), indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
