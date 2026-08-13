from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.bucket_probability import score_probabilities
from src.calibration.kdal_ordinal_challenger import (
    _inner_split,
    apply_no_override_policy,
    build_frames,
    policy_metrics,
    serialize_prediction_columns,
    tune_no_override_policy,
)
from src.calibration.kdal_tree_probability_challenger import (
    TREE_ARMS,
    best_tree_configs,
    fit_tree_config,
    nested_tree_evaluation,
    tune_tree_arms,
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "calibration"
    / "station_training_baseline"
    / "KDAL"
    / "tree_probability_challenger_v1"
)
ORDINAL_COMPARISON_PATH = (
    PROJECT_ROOT
    / "data"
    / "calibration"
    / "station_training_baseline"
    / "KDAL"
    / "ordinal_challenger_v1"
    / "KDAL_2026_exploratory_candidate_comparison.csv"
)


def prediction_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    probabilities = np.vstack(predictions["offset_probabilities"].to_numpy())
    scores = score_probabilities(
        predictions["offset_class"].to_numpy(dtype=int),
        probabilities,
    )
    return {
        **scores,
        "count": int(len(predictions)),
        "bucket_log_loss": float(
            -np.log(
                predictions["actual_bucket_probability"].clip(lower=1e-12)
            ).mean()
        ),
        "bucket_accuracy": float(
            predictions["probability_top_bucket_hit"].mean()
        ),
        "point_bucket_accuracy": float(
            predictions["point_bucket_hit"].mean()
        ),
    }


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    development, holdout, _ = build_frames(PROJECT_ROOT)
    development = development.loc[
        development["year"].between(2023, 2025)
    ].copy()

    nested_predictions, nested_selections, nested_tuning = (
        nested_tree_evaluation(development)
    )
    if not (
        pd.to_datetime(nested_predictions["model_training_cutoff"])
        < pd.to_datetime(nested_predictions["contract_date"])
    ).all():
        raise AssertionError("nested tree predictions are not chronological")
    nested_metrics = pd.DataFrame(
        [
            {"arm": arm, **prediction_metrics(group)}
            for arm, group in nested_predictions.groupby("arm")
        ]
    ).sort_values(["bucket_log_loss", "ranked_probability_score"])

    final_inner_train, final_inner_valid = _inner_split(development)
    final_tuning = tune_tree_arms(final_inner_train, final_inner_valid)
    final_configs = best_tree_configs(final_tuning)
    holdout_rows: list[dict[str, object]] = []
    final_config_rows: list[dict[str, object]] = []
    for arm in TREE_ARMS:
        config = final_configs[arm]
        _, policy_predictions, _ = fit_tree_config(
            final_inner_train,
            final_inner_valid,
            config,
        )
        policy = tune_no_override_policy(policy_predictions)
        _, predictions, _ = fit_tree_config(
            development,
            holdout,
            config,
        )
        predictions = apply_no_override_policy(predictions, policy)
        metrics = {
            **prediction_metrics(predictions),
            **{
                f"policy_{key}": value
                for key, value in policy_metrics(predictions).items()
            },
        }
        holdout_rows.append({"arm": arm, **metrics})
        final_config_rows.append(
            {
                **config.__dict__,
                **{
                    f"threshold_{key}": value
                    for key, value in policy.items()
                    if key != "overrides_enabled"
                },
            }
        )
        serialize_prediction_columns(predictions).to_csv(
            OUTPUT_DIR / f"KDAL_{arm}_2026_exploratory_predictions.csv",
            index=False,
        )

    holdout_comparison = pd.DataFrame(holdout_rows).sort_values(
        ["bucket_log_loss", "ranked_probability_score"]
    )
    combined = holdout_comparison.assign(source="tree_challenger")
    if ORDINAL_COMPARISON_PATH.is_file():
        source = pd.read_csv(ORDINAL_COMPARISON_PATH)
        ordinal = pd.DataFrame(
            {
                "arm": source["candidate_name"],
                "log_loss": source["holdout_log_loss"],
                "brier": source["holdout_brier"],
                "ranked_probability_score": source[
                    "holdout_ranked_probability_score"
                ],
                "offset_accuracy": source["holdout_offset_accuracy"],
                "top_two_accuracy": source["holdout_top_two_accuracy"],
                "calibration_error": source["holdout_calibration_error"],
                "count": source["holdout_count"],
                "bucket_log_loss": source["holdout_bucket_log_loss"],
                "bucket_accuracy": source["holdout_bucket_accuracy"],
                "point_bucket_accuracy": source[
                    "holdout_point_bucket_accuracy"
                ],
                "source": "ordinal_challenger",
            }
        )
        columns = list(ordinal.columns)
        combined = pd.concat(
            [combined[columns], ordinal[columns]],
            ignore_index=True,
        ).sort_values(["bucket_log_loss", "ranked_probability_score"])

    serialize_prediction_columns(nested_predictions).to_csv(
        OUTPUT_DIR / "KDAL_tree_nested_predictions.csv",
        index=False,
    )
    nested_selections.to_csv(
        OUTPUT_DIR / "KDAL_tree_nested_selections.csv",
        index=False,
    )
    nested_tuning.to_csv(
        OUTPUT_DIR / "KDAL_tree_nested_tuning.csv",
        index=False,
    )
    nested_metrics.to_csv(
        OUTPUT_DIR / "KDAL_tree_nested_metrics.csv",
        index=False,
    )
    final_tuning.to_csv(
        OUTPUT_DIR / "KDAL_tree_final_inner_tuning.csv",
        index=False,
    )
    pd.DataFrame(final_config_rows).to_csv(
        OUTPUT_DIR / "KDAL_tree_frozen_configs.csv",
        index=False,
    )
    holdout_comparison.to_csv(
        OUTPUT_DIR / "KDAL_tree_2026_exploratory_comparison.csv",
        index=False,
    )
    combined.to_csv(
        OUTPUT_DIR / "KDAL_tree_and_ordinal_2026_comparison.csv",
        index=False,
    )
    summary = {
        "experiment": "kdal_tree_probability_challenger_v1",
        "selection_period": "2023-2025_chronological_only",
        "holdout_status": "2026_exploratory_previously_inspected",
        "arms": list(TREE_ARMS),
        "nested_metrics": nested_metrics.to_dict(orient="records"),
        "final_configs": final_config_rows,
        "holdout_metrics": holdout_comparison.to_dict(orient="records"),
        "promotion_approved": False,
        "promotion_blocker": "fresh_shadow_data_required",
    }
    (OUTPUT_DIR / "KDAL_tree_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(nested_metrics.to_string(index=False))
    print(holdout_comparison.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
