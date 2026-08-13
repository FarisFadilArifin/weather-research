from __future__ import annotations

import json
from pathlib import Path


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


def notebook() -> dict:
    cells = [
        md(
            """# KATL/KDAL V20 Half-Up Override Classifier V3

V3 keeps the regression's ordinary nearest half-up rounded bucket by default. Its binary target is
`1` only when changing to the opposite floor/ceil choice produces the settled 2°F bucket while the
default bucket loses; otherwise it is `0`. Same-bucket floor/ceil rows are non-actionable class `0`.

All rows are modeled. There are no boundary-distance windows. Model hyperparameters, Platt
calibration, and override threshold are selected in separate chronological windows before the next
outer year is evaluated.
"""
        ),
        md("## Environment and contracts\n"),
        code(
            """from pathlib import Path
import sys
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path.cwd().resolve()
while PROJECT_ROOT != PROJECT_ROOT.parent and not (PROJECT_ROOT / "pyproject.toml").is_file():
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.round_override import (
    audit_round_override_system,
    build_round_override_frame,
    continuous_override_comparison,
    fit_round_override_system,
    predict_round_override,
    round_override_metrics,
    serializable_override_bundle,
)
from src.calibration.v19_bucket import crossfit_ridge_predictions

STATIONS = {
    "KATL": {
        "pipeline": PROJECT_ROOT / "data/calibration/station_stacking_v20_peak_timing",
        "peak": True,
        "continuous": PROJECT_ROOT / "data/calibration/station_continuous_residual_probability/KATL/KATL_forward_continuous_predictions.csv",
    },
    "KDAL": {
        "pipeline": PROJECT_ROOT / "data/calibration/station_stacking_v20_kdal_no_peak",
        "peak": False,
        "continuous": PROJECT_ROOT / "data/calibration/station_continuous_residual_probability/KDAL/KDAL_forward_continuous_predictions.csv",
    },
}
print(PROJECT_ROOT)
"""
        ),
        md("## Build the bucket-aware override label\n"),
        code(
            """frames, sources, summaries = {}, {}, []
for station, config in STATIONS.items():
    features = pd.read_csv(config["pipeline"] / f"{station}_features.csv", low_memory=False)
    validation = pd.read_csv(config["pipeline"] / f"{station}_year_split_validation_predictions.csv")
    point = crossfit_ridge_predictions(validation)
    frame = build_round_override_frame(
        features, point, validation, include_peak_features=config["peak"]
    )
    frames[station] = frame
    sources[station] = {"features": features, "validation": validation, "point": point}
    summaries.append({
        "station": station,
        "rows": len(frame),
        "start": frame["contract_date"].min().date(),
        "end": frame["contract_date"].max().date(),
        "actionable_rate": frame["override_actionable"].mean(),
        "override_target_rate": frame["override_target"].mean(),
        "default_bucket_hit_rate": frame["default_bucket_win"].mean(),
    })
dataset_summary = pd.DataFrame(summaries)
dataset_summary
"""
        ),
        code(
            """columns = [
    "contract_date", "point_prediction_f", "default_degree_f", "default_bucket_label",
    "alternative_degree_f", "alternative_bucket_label", "override_actionable",
    "actual_high_f", "actual_bucket_label", "default_bucket_win",
    "alternative_bucket_win", "override_target",
]
pd.concat([frame.assign(station=station).head(6) for station, frame in frames.items()])[["station", *columns]]
"""
        ),
        md("## Nested V3 training, calibration, and threshold selection\n"),
        code(
            """results = {}
for station, config in STATIONS.items():
    print(f"Training V3 {station}...")
    results[station] = fit_round_override_system(
        frames[station], station_id=station, include_peak_features=config["peak"]
    )
print("Training complete")
"""
        ),
        md("## Selected fold configurations\n"),
        code(
            """fold_states = pd.concat(
    [result["fold_states"].assign(station=station) for station, result in results.items()],
    ignore_index=True,
)
fold_states[[
    "station", "validation_year", "selected_family", "selected_model_params_json",
    "selected_actionable_weight", "override_threshold", "model_training_cutoff",
    "calibration_start", "calibration_cutoff", "policy_start", "policy_cutoff",
]]
"""
        ),
        md("## Honest 2024–2025 result\n"),
        code(
            """forward_metrics = pd.concat(
    [result["forward_metrics"].assign(station=station) for station, result in results.items()],
    ignore_index=True,
)
metric_columns = [
    "station", "period", "count", "override_target_rate", "mean_override_probability",
    "binary_log_loss", "brier", "ece", "roc_auc", "pr_auc", "actionable_count",
    "default_bucket_hit_rate", "final_bucket_hit_rate", "bucket_hit_rate_lift",
    "override_count", "override_precision", "override_recall", "recovered_losses",
    "damaged_wins", "net_recovered_wins", "recovery_damage_ratio",
]
forward_metrics[metric_columns]
"""
        ),
        code(
            """combined = forward_metrics.loc[forward_metrics["period"].eq("2024-2025")]
plot_data = combined.set_index("station")[["default_bucket_hit_rate", "final_bucket_hit_rate"]]
ax = plot_data.plot(kind="bar", figsize=(8, 4), ylim=(0, max(0.6, plot_data.to_numpy().max() + 0.05)))
ax.set_title("Half-up default vs V3 override bucket hit rate")
ax.set_ylabel("Bucket hit rate")
ax.grid(axis="y", alpha=0.25)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
"""
        ),
        md("## Continuous-residual override baseline\n"),
        code(
            """comparison_parts = []
for station, config in STATIONS.items():
    continuous = pd.read_csv(config["continuous"])
    baseline = continuous_override_comparison(
        results[station]["forward_predictions"], continuous
    ).assign(station=station, model="continuous_student_t")
    comparison_parts.append(baseline)
    model = results[station]["forward_metrics"].loc[
        results[station]["forward_metrics"]["period"].eq("2024-2025")
    ].assign(station=station, model="v3_override_classifier")
    comparison_parts.append(model)
continuous_comparison = pd.concat(comparison_parts, ignore_index=True)
continuous_comparison[[
    "station", "model", "binary_log_loss", "brier", "ece", "roc_auc", "pr_auc",
    "default_bucket_hit_rate", "final_bucket_hit_rate", "net_recovered_wins",
]].sort_values(["station", "binary_log_loss"])
"""
        ),
        md("## Exploratory 2026\n"),
        code(
            """holdout_predictions, holdout_rows = {}, []
for station, config in STATIONS.items():
    source = sources[station]
    test = pd.read_csv(config["pipeline"] / f"{station}_year_split_test_predictions.csv")
    test_point = test.loc[test["method"].eq("ridge_stack"), ["contract_date", "actual_high_f", "predicted_high_f"]]
    all_point = pd.concat([source["point"][["contract_date", "actual_high_f", "predicted_high_f"]], test_point], ignore_index=True)
    all_base = pd.concat([source["validation"], test], ignore_index=True)
    all_frame = build_round_override_frame(
        source["features"], all_point, all_base, include_peak_features=config["peak"]
    )
    holdout = all_frame.loc[all_frame["year"].eq(2026)]
    predicted = predict_round_override(results[station]["final_state"], holdout)
    holdout_predictions[station] = predicted
    holdout_rows.append({"station": station, **round_override_metrics(predicted)})
holdout_metrics = pd.DataFrame(holdout_rows)
holdout_metrics[[
    "station", "count", "binary_log_loss", "ece", "default_bucket_hit_rate",
    "final_bucket_hit_rate", "bucket_hit_rate_lift", "override_count",
    "override_precision", "recovered_losses", "damaged_wins", "net_recovered_wins",
]]
"""
        ),
        md("## Executable V3 audit\n"),
        code(
            """audit_parts = []
for station, config in STATIONS.items():
    audit_parts.append(
        audit_round_override_system(
            frames[station], results[station], include_peak_features=config["peak"]
        ).assign(station=station)
    )
    frame = frames[station]
    audit_parts.append(pd.DataFrame([
        {"station": station, "audit": "unique_station_dates", "passed": not frame["contract_date"].duplicated().any(), "detail": f"duplicates={int(frame['contract_date'].duplicated().sum())}"},
        {"station": station, "audit": "integer_settlement_labels", "passed": bool(np.allclose(frame["actual_high_f"], np.round(frame["actual_high_f"]))), "detail": f"rows={len(frame)}"},
    ]))
audit_report = pd.concat(audit_parts, ignore_index=True)[["station", "audit", "passed", "detail"]]
display(audit_report)
failed = audit_report.loc[~audit_report["passed"]]
assert failed.empty, failed.to_dict(orient="records")
print(f"All {len(audit_report)} V3 audits passed")
"""
        ),
        md("## Export V3 research artifacts\n"),
        code(
            """OUTPUT_ROOT = PROJECT_ROOT / "data/calibration/station_round_override_v3"
exports = []
for station, result in results.items():
    output, weights = OUTPUT_ROOT / station, OUTPUT_ROOT / station / "model_weights"
    weights.mkdir(parents=True, exist_ok=True)
    result["forward_predictions"].to_csv(output / f"{station}_v3_forward_predictions.csv", index=False)
    result["forward_metrics"].to_csv(output / f"{station}_v3_forward_metrics.csv", index=False)
    result["fold_states"].to_csv(output / f"{station}_v3_fold_states.csv", index=False)
    result["tuning"].to_csv(output / f"{station}_v3_inner_tuning.csv", index=False)
    result["threshold_tuning"].to_csv(output / f"{station}_v3_threshold_tuning.csv", index=False)
    holdout_predictions[station].to_csv(output / f"{station}_v3_2026_predictions.csv", index=False)
    bundle = weights / f"{station}_half_up_override_classifier_v3.joblib"
    joblib.dump(serializable_override_bundle(result), bundle)
    exports.append({"station": station, "bundle": str(bundle), "final_family": result["final_state"]["selected_spec"].family, "final_threshold": result["final_state"]["override_threshold"]})
audit_report.to_csv(OUTPUT_ROOT / "v3_audit.csv", index=False)
dataset_summary.to_csv(OUTPUT_ROOT / "dataset_summary.csv", index=False)
continuous_comparison.to_csv(OUTPUT_ROOT / "continuous_baseline_comparison.csv", index=False)
holdout_metrics.to_csv(OUTPUT_ROOT / "exploratory_2026_metrics.csv", index=False)
pd.DataFrame(exports)
"""
        ),
        md(
            """## Acceptance rule

V3 is accepted only if it produces positive net recovered wins and bucket-hit lift in each honest
outer year without materially worse probability calibration than the continuous baseline. A small
exploratory 2026 gain cannot override a negative 2024 or 2025 result.
"""
        ),
    ]
    for index, cell in enumerate(cells):
        cell["id"] = f"override-v3-{index:02d}"
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
    output = Path(__file__).resolve().parent / "half_up_override_classifier_v3.ipynb"
    output.write_text(json.dumps(notebook(), indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
