from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.kdal_ordinal_challenger import (
    FRESH_SHADOW_START,
    FROZEN_CANDIDATE_ROLES,
    _inner_split,
    apply_no_override_policy,
    build_frames,
    export_frozen_candidate,
    fit_and_predict,
    frozen_candidate_rows,
    nested_forward_evaluation,
    policy_metrics,
    row_to_config,
    serialize_prediction_columns,
    sha256_file,
    tune_candidates,
    tune_no_override_policy,
)
from src.calibration.bucket_probability import score_probabilities


STATION_ID = "KDAL"
POINT_MODEL_VERSION = "station_high_regressor_baseline_kdal_no_peak_stack"
OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "calibration"
    / "station_training_baseline"
    / "KDAL"
    / "ordinal_challenger_v1"
)


def _prediction_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    probabilities = np.vstack(
        predictions["offset_probabilities"].to_numpy()
    )
    scores = score_probabilities(
        predictions["offset_class"].to_numpy(dtype=int),
        probabilities,
    )
    selected = predictions.get(
        "shadow_trade",
        pd.Series(False, index=predictions.index),
    ).astype(bool)
    return {
        **scores,
        "count": int(len(predictions)),
        "bucket_log_loss": float(
            -np.log(
                predictions["actual_bucket_probability"].clip(lower=1e-12)
            ).mean()
        ),
        "probability_top_bucket_accuracy": float(
            predictions["probability_top_bucket_hit"].mean()
        ),
        "point_bucket_accuracy": float(
            predictions["point_bucket_hit"].mean()
        ),
        "shadow_coverage": float(selected.mean()),
        "shadow_point_bucket_accuracy": (
            float(predictions.loc[selected, "point_bucket_hit"].mean())
            if selected.any()
            else float("nan")
        ),
        "override_count": int(
            predictions.get(
                "overrides_point_bucket",
                pd.Series(False, index=predictions.index),
            ).sum()
        ),
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def run_challenger() -> dict[str, object]:
    """Train, evaluate, and export the three frozen KDAL ordinal arms."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    development, holdout, paths = build_frames(PROJECT_ROOT)
    development = development.loc[
        development["year"].between(2023, 2025)
    ].copy()
    if holdout.empty:
        raise SystemExit("2026 exploratory holdout is empty")

    forward, selections, outer_tuning = nested_forward_evaluation(
        development
    )
    if set(forward["outer_validation_year"].astype(int)) != {2024, 2025}:
        raise AssertionError("nested forward evaluation did not cover 2024/2025")
    if not (
        pd.to_datetime(forward["model_training_cutoff"])
        < pd.to_datetime(forward["contract_date"])
    ).all():
        raise AssertionError("nested forward chronology failed")
    if int(forward["overrides_point_bucket"].sum()) != 0:
        raise AssertionError("nested forward policy attempted a bucket override")

    forward_metrics = _prediction_metrics(forward)
    forward_by_year = pd.DataFrame(
        [
            {
                "validation_year": int(year),
                **_prediction_metrics(group),
            }
            for year, group in forward.groupby("outer_validation_year")
        ]
    )

    final_inner_train, final_inner_valid = _inner_split(development)
    final_tuning = tune_candidates(final_inner_train, final_inner_valid)
    frozen_rows = frozen_candidate_rows(final_tuning)
    if tuple(frozen_rows["candidate_role"]) != FROZEN_CANDIDATE_ROLES:
        raise AssertionError("unexpected frozen challenger role contract")
    candidate_comparison: list[dict[str, object]] = []
    frozen_manifests: list[str] = []
    frozen_bundles: list[str] = []

    for position, row in frozen_rows.iterrows():
        config = row_to_config(row)
        candidate_name = (
            f"kdal_ordinal_challenger_v1_{position + 1}_"
            f"{config.family}_{config.feature_set}"
        )
        _, policy_validation, _ = fit_and_predict(
            final_inner_train,
            final_inner_valid,
            config,
        )
        policy = tune_no_override_policy(policy_validation)
        holdout_metrics, holdout_predictions, state = fit_and_predict(
            development,
            holdout,
            config,
        )
        holdout_predictions = apply_no_override_policy(
            holdout_predictions, policy
        )
        if int(holdout_predictions["overrides_point_bucket"].sum()) != 0:
            raise AssertionError(f"{candidate_name} attempted a bucket override")
        for probability_column in (
            "offset_probabilities",
            "degree_probabilities",
            "bucket_probabilities",
        ):
            values = holdout_predictions[probability_column]
            assert values.map(
                lambda probabilities: np.isclose(
                    sum(
                        probabilities.values()
                        if isinstance(probabilities, dict)
                        else probabilities
                    ),
                    1.0,
                    atol=1e-10,
                )
            ).all()
        no_override_metrics = {
            **holdout_metrics,
            **{
                f"policy_{key}": value
                for key, value in policy_metrics(
                    holdout_predictions
                ).items()
            },
        }
        bundle_path, manifest_path = export_frozen_candidate(
            OUTPUT_DIR / "model_weights",
            station_id=STATION_ID,
            point_model_version=POINT_MODEL_VERSION,
            point_bundle_path=paths["point_bundle"],
            config=config,
            state=state,
            policy=policy,
            historical_metrics={
                "2026_exploratory": no_override_metrics,
                "selection_inner_bucket_log_loss": float(
                    row["bucket_log_loss"]
                ),
                "selection_inner_ranked_probability_score": float(
                    row["ranked_probability_score"]
                ),
            },
            candidate_name=candidate_name,
        )
        frozen_manifests.append(
            str(manifest_path.relative_to(PROJECT_ROOT))
        )
        frozen_bundles.append(
            str(bundle_path.relative_to(PROJECT_ROOT))
        )
        serialize_prediction_columns(holdout_predictions).to_csv(
            OUTPUT_DIR / f"{candidate_name}_2026_exploratory_predictions.csv",
            index=False,
        )
        candidate_comparison.append(
            {
                "candidate_name": candidate_name,
                "candidate_role": row["candidate_role"],
                **{
                    key: row[key]
                    for key in (
                        "family",
                        "feature_set",
                        "feature_count",
                        "c",
                        "temperature",
                        "model_weight",
                        "prior_strength",
                        "bucket_log_loss",
                        "ranked_probability_score",
                        "log_loss",
                    )
                },
                **{
                    f"holdout_{key}": value
                    for key, value in no_override_metrics.items()
                },
                "bundle_sha256": sha256_file(bundle_path),
                "manifest_path": str(
                    manifest_path.relative_to(PROJECT_ROOT)
                ),
            }
        )

    comparison = pd.DataFrame(candidate_comparison)
    serialize_prediction_columns(forward).to_csv(
        OUTPUT_DIR / "KDAL_nested_forward_predictions.csv",
        index=False,
    )
    selections.to_csv(
        OUTPUT_DIR / "KDAL_nested_forward_selections.csv", index=False
    )
    outer_tuning.to_csv(
        OUTPUT_DIR / "KDAL_nested_outer_tuning.csv", index=False
    )
    forward_by_year.to_csv(
        OUTPUT_DIR / "KDAL_nested_forward_metrics_by_year.csv",
        index=False,
    )
    final_tuning.to_csv(
        OUTPUT_DIR / "KDAL_final_inner_tuning.csv", index=False
    )
    frozen_rows.to_csv(
        OUTPUT_DIR / "KDAL_frozen_candidate_configs.csv", index=False
    )
    comparison.to_csv(
        OUTPUT_DIR / "KDAL_2026_exploratory_candidate_comparison.csv",
        index=False,
    )
    summary = {
        "experiment": "kdal_ordinal_challenger_v1",
        "station_id": STATION_ID,
        "point_model_version": POINT_MODEL_VERSION,
        "historical_selection_primary_metric": "bucket_log_loss",
        "historical_selection_secondary_metric": "ranked_probability_score",
        "feature_sets": ["market_core_21", "compact_27", "full_59"],
        "families": [
            "empirical",
            "shared_slope_ordinal_logistic",
            "ordinal_logistic",
        ],
        "ordinal_model_weights": [0.25, 0.5, 0.75, 1.0],
        "nested_policy_evaluation": True,
        "bucket_overrides_enabled": False,
        "nested_forward_metrics": forward_metrics,
        "nested_forward_selections": selections.to_dict(orient="records"),
        "frozen_candidate_count": int(len(frozen_rows)),
        "frozen_candidate_roles": list(FROZEN_CANDIDATE_ROLES),
        "frozen_bundles": frozen_bundles,
        "frozen_manifests": frozen_manifests,
        "holdout_status": "2026_exploratory_previously_inspected",
        "fresh_shadow_start_contract_date": FRESH_SHADOW_START,
        "promotion_approved": False,
        "promotion_blocker": "fresh_shadow_data_required",
    }
    (OUTPUT_DIR / "KDAL_summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "summary": summary,
        "comparison": comparison,
        "frozen_configs": frozen_rows,
        "bundle_paths": [
            PROJECT_ROOT / relative_path
            for relative_path in frozen_bundles
        ],
        "manifest_paths": [
            PROJECT_ROOT / relative_path
            for relative_path in frozen_manifests
        ],
        "output_dir": OUTPUT_DIR,
    }


def main() -> int:
    result = run_challenger()
    print(json.dumps(_json_safe(result["summary"]), indent=2, sort_keys=True))
    print(result["comparison"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
