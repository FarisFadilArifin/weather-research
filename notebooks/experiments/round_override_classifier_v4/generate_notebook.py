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
            """# KATL/KDAL V20 Cost-Sensitive Half-Up Override V4

V4 keeps nearest half-up rounding as the default. It fits two calibrated binary heads on
**actionable rows only**:

- recovery: the alternative bucket wins and half-up loses;
- damage: half-up wins and the alternative bucket loses.

The decision is `p(recovery) - damage_penalty * p(damage) > minimum_margin`, with a separate
minimum recovery probability. A policy is enabled only when it has positive `recoveries -
2*damages`, is positive in at least two of three earlier chronological policy folds, and is harmful
in none. Otherwise V4 automatically makes no overrides.
"""
        ),
        md("## Environment and contracts\n"),
        code(
            """from pathlib import Path
import sys
import warnings
import joblib
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

from src.calibration.round_override_v4 import (
    audit_utility_override_system,
    build_utility_override_frame,
    fit_utility_override_system,
    predict_utility_override,
    serializable_utility_bundle,
    utility_override_metrics,
)
from src.calibration.v19_bucket import crossfit_ridge_predictions

STATIONS = {
    "KATL": {"pipeline": PROJECT_ROOT / "data/calibration/station_stacking_v20_peak_timing", "peak": True},
    "KDAL": {"pipeline": PROJECT_ROOT / "data/calibration/station_stacking_v20_kdal_no_peak", "peak": False},
}
print(PROJECT_ROOT)
"""
        ),
        md("## Build exact recovery/damage targets and prior-only continuous features\n"),
        code(
            """frames, sources, summaries = {}, {}, []
for station, config in STATIONS.items():
    features = pd.read_csv(config["pipeline"] / f"{station}_features.csv", low_memory=False)
    validation = pd.read_csv(config["pipeline"] / f"{station}_year_split_validation_predictions.csv")
    point = crossfit_ridge_predictions(validation)
    frame = build_utility_override_frame(
        features, point, validation, include_peak_features=config["peak"]
    )
    frames[station] = frame
    sources[station] = {"features": features, "validation": validation, "point": point}
    actionable = frame.loc[frame["override_actionable"].eq(1)]
    summaries.append({
        "station": station,
        "rows": len(frame),
        "start": frame["contract_date"].min().date(),
        "end": frame["contract_date"].max().date(),
        "actionable_rows": len(actionable),
        "recovery_rate_given_actionable": actionable["recovery_target"].mean(),
        "damage_rate_given_actionable": actionable["damage_target"].mean(),
        "default_bucket_hit_rate": frame["default_bucket_win"].mean(),
    })
dataset_summary = pd.DataFrame(summaries)
dataset_summary
"""
        ),
        code(
            """display_columns = [
    "contract_date", "point_prediction_f", "default_bucket_label", "alternative_bucket_label",
    "override_actionable", "actual_bucket_label", "recovery_target", "damage_target",
    "prior_residual_mean_180d_f", "prior_residual_std_180d_f",
    "continuous_default_bucket_probability_180d",
    "continuous_alternative_bucket_probability_180d",
]
pd.concat(
    [frame.assign(station=station).tail(5) for station, frame in frames.items()]
)[["station", *display_columns]]
"""
        ),
        md("## Nested chronological V4 fitting\n"),
        code(
            """results = {}
for station, config in STATIONS.items():
    print(f"Training V4 {station}...")
    results[station] = fit_utility_override_system(
        frames[station], station_id=station, include_peak_features=config["peak"]
    )
print("Training complete")
"""
        ),
        md("## Selected heads and stability-gated policies\n"),
        code(
            """fold_states = pd.concat(
    [result["fold_states"].assign(station=station) for station, result in results.items()],
    ignore_index=True,
)
fold_states[[
    "station", "validation_year", "recovery_family", "damage_family", "policy_enabled",
    "damage_penalty", "minimum_recovery_probability", "minimum_utility_margin",
    "training_actionable_rows", "training_non_actionable_rows", "model_training_cutoff",
    "calibration_start", "calibration_cutoff", "policy_start", "policy_cutoff",
]]
"""
        ),
        md("## Honest 2024-2025 forward result\n"),
        code(
            """forward_metrics = pd.concat(
    [result["forward_metrics"].assign(station=station) for station, result in results.items()],
    ignore_index=True,
)
result_columns = [
    "station", "period", "count", "actionable_count", "default_bucket_hit_rate",
    "final_bucket_hit_rate", "bucket_hit_rate_lift", "override_count", "recovered_losses",
    "damaged_wins", "net_recovered_wins", "recovery_damage_ratio",
    "recovery_log_loss", "recovery_ece", "recovery_roc_auc",
    "damage_log_loss", "damage_ece", "damage_roc_auc",
]
forward_metrics[result_columns]
"""
        ),
        code(
            """combined = forward_metrics.loc[forward_metrics["period"].eq("2024-2025")]
plot_data = combined.set_index("station")[["default_bucket_hit_rate", "final_bucket_hit_rate"]]
ax = plot_data.plot(kind="bar", figsize=(8, 4), ylim=(0, 0.6))
ax.set_title("Half-up default vs stability-gated V4")
ax.set_ylabel("Exact 2-degree bucket hit rate")
ax.grid(axis="y", alpha=0.25)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
"""
        ),
        md("## Exploratory sequential 2026 evaluation\n"),
        code(
            """holdout_predictions, holdout_rows = {}, []
for station, config in STATIONS.items():
    source = sources[station]
    test = pd.read_csv(config["pipeline"] / f"{station}_year_split_test_predictions.csv")
    test_point = test.loc[
        test["method"].eq("ridge_stack"),
        ["contract_date", "actual_high_f", "predicted_high_f"],
    ]
    all_point = pd.concat(
        [source["point"][["contract_date", "actual_high_f", "predicted_high_f"]], test_point],
        ignore_index=True,
    )
    all_base = pd.concat([source["validation"], test], ignore_index=True)
    all_frame = build_utility_override_frame(
        source["features"], all_point, all_base, include_peak_features=config["peak"]
    )
    holdout = all_frame.loc[all_frame["year"].eq(2026)]
    predicted = predict_utility_override(results[station]["final_state"], holdout)
    holdout_predictions[station] = predicted
    holdout_rows.append({"station": station, **utility_override_metrics(predicted)})
holdout_metrics = pd.DataFrame(holdout_rows)
holdout_metrics[[
    "station", "count", "default_bucket_hit_rate", "final_bucket_hit_rate",
    "bucket_hit_rate_lift", "override_count", "recovered_losses", "damaged_wins",
    "net_recovered_wins", "recovery_log_loss", "damage_log_loss",
]]
"""
        ),
        md("## Executable V4 audit\n"),
        code(
            """audit_parts = []
for station, config in STATIONS.items():
    audit_parts.append(
        audit_utility_override_system(
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
print(f"All {len(audit_report)} V4 audits passed")
"""
        ),
        md("## Export research artifacts\n"),
        code(
            """OUTPUT_ROOT = PROJECT_ROOT / "data/calibration/station_round_override_v4"
exports = []
for station, result in results.items():
    output = OUTPUT_ROOT / station
    weights = output / "model_weights"
    weights.mkdir(parents=True, exist_ok=True)
    result["forward_predictions"].to_csv(output / f"{station}_v4_forward_predictions.csv", index=False)
    result["forward_metrics"].to_csv(output / f"{station}_v4_forward_metrics.csv", index=False)
    result["fold_states"].to_csv(output / f"{station}_v4_fold_states.csv", index=False)
    result["tuning"].to_csv(output / f"{station}_v4_head_tuning.csv", index=False)
    result["policy_tuning"].to_csv(output / f"{station}_v4_policy_tuning.csv", index=False)
    result["final_tuning"].to_csv(output / f"{station}_v4_final_head_tuning.csv", index=False)
    result["final_policy_tuning"].to_csv(output / f"{station}_v4_final_policy_tuning.csv", index=False)
    holdout_predictions[station].to_csv(output / f"{station}_v4_2026_predictions.csv", index=False)
    bundle = weights / f"{station}_half_up_utility_override_v4.joblib"
    joblib.dump(serializable_utility_bundle(result), bundle)
    exports.append({
        "station": station,
        "bundle": str(bundle),
        "final_policy_enabled": result["final_state"]["policy_enabled"],
    })
audit_report.to_csv(OUTPUT_ROOT / "v4_audit.csv", index=False)
dataset_summary.to_csv(OUTPUT_ROOT / "dataset_summary.csv", index=False)
holdout_metrics.to_csv(OUTPUT_ROOT / "exploratory_2026_metrics.csv", index=False)
pd.DataFrame(exports)
"""
        ),
        md(
            """## Acceptance rule

V4 is accepted for overrides only if each honest outer year has positive net recovered wins, no
year has negative bucket-hit lift, recovery/damage exceeds 1, and the final policy passes the
three-fold prior stability gate. An abstaining policy is a valid safety result but does not prove
that the classifier adds value over half-up.
"""
        ),
    ]
    for index, cell in enumerate(cells):
        cell["id"] = f"override-v4-{index:02d}"
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    output = Path(__file__).resolve().parent / "half_up_utility_override_v4.ipynb"
    output.write_text(json.dumps(notebook(), indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
