from __future__ import annotations

import numpy as np
import pandas as pd

from src.calibration.bucket_probability import (
    CandidateSpec,
    MANDATORY_SOURCE_FEATURES,
    OFFSET_LABELS,
    _fit_candidate,
    _predict_candidate,
    fit_tail_policy,
    predict_probability_bundle,
)
from src.calibration.kdal_ordinal_challenger import (
    FROZEN_CANDIDATE_ROLES,
    apply_no_override_policy,
    feature_sets,
    frozen_candidate_rows,
    tune_no_override_policy,
)


def test_shared_slope_ordinal_model_returns_valid_probabilities() -> None:
    rows = 180
    frame = pd.DataFrame(
        {
            "point_prediction_f": np.linspace(80.0, 100.0, rows),
            "base_prediction_spread_f": np.tile(
                np.linspace(0.0, 4.0, 9), 20
            ),
            "offset_class": np.tile(np.arange(len(OFFSET_LABELS)), 20),
        }
    )
    features = ["point_prediction_f", "base_prediction_spread_f"]
    state = _fit_candidate(
        frame,
        features,
        CandidateSpec(
            "shared_slope_ordinal_logistic",
            {"C": 0.03, "class_weight": None},
        ),
        random_state=42,
    )
    probabilities = _predict_candidate(state, frame.iloc[:12], features)
    assert state["family"] == "shared_slope_ordinal_logistic"
    assert probabilities.shape == (12, len(OFFSET_LABELS))
    assert np.isfinite(probabilities).all()
    assert (probabilities >= 0.0).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_ordinal_lightgbm_returns_valid_ordered_probabilities() -> None:
    rows = 180
    frame = pd.DataFrame(
        {
            "point_prediction_f": np.linspace(80.0, 100.0, rows),
            "base_prediction_spread_f": np.tile(
                np.linspace(0.0, 4.0, 9), 20
            ),
            "offset_class": np.tile(np.arange(len(OFFSET_LABELS)), 20),
        }
    )
    features = ["point_prediction_f", "base_prediction_spread_f"]
    state = _fit_candidate(
        frame,
        features,
        CandidateSpec(
            "ordinal_lightgbm",
            {
                "learning_rate": 0.05,
                "n_estimators": 20,
                "num_leaves": 5,
                "min_child_samples": 10,
                "reg_lambda": 10.0,
            },
        ),
        random_state=42,
    )
    probabilities = _predict_candidate(state, frame.iloc[:12], features)
    assert state["family"] == "ordinal_lightgbm"
    assert len(state["threshold_models"]) == len(OFFSET_LABELS) - 1
    assert probabilities.shape == (12, len(OFFSET_LABELS))
    assert np.isfinite(probabilities).all()
    assert (probabilities >= 0.0).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_feature_ablations_are_strict_subsets_of_full_contract() -> None:
    sets = feature_sets()
    assert len(sets["market_core_21"]) == 21
    assert len(sets["compact_27"]) == 27
    assert len(sets["full_59"]) == 59
    assert set(sets["market_core_21"]).issubset(sets["full_59"])
    assert set(sets["compact_27"]).issubset(sets["full_59"])


def test_no_override_policy_never_changes_point_bucket() -> None:
    predictions = pd.DataFrame(
        {
            "top_bucket_probability": [0.70, 0.30, 0.55, 0.40],
            "top_two_margin": [0.30, 0.05, 0.20, 0.10],
            "point_bucket": ["90-91", "92-93", "94-95", "96-97"],
            "actual_bucket": ["90-91", "94-95", "94-95", "96-97"],
            "point_bucket_hit": [True, False, True, True],
        }
    )
    policy = tune_no_override_policy(predictions, target_coverage=0.50)
    scored = apply_no_override_policy(predictions, policy)
    assert scored["recommended_bucket"].equals(scored["point_bucket"])
    assert not scored["overrides_point_bucket"].any()
    assert scored["recommended_bucket_hit"].equals(scored["point_bucket_hit"])


def test_serving_respects_frozen_no_override_contract() -> None:
    counts = [0] * len(OFFSET_LABELS)
    counts[6] = 100
    bundle = {
        "model_version": "test-no-override",
        "mandatory_source_features": list(MANDATORY_SOURCE_FEATURES),
        "feature_names": [],
        "selected_family": "empirical",
        "empirical_state": {
            "global_counts": counts,
            "month_counts": {"7": counts},
            "alpha": 0.5,
        },
        "empirical_prior_strength": 30.0,
        "tail_policy": fit_tail_policy([-4, 0, 2, 4]),
        "decision_thresholds": {
            "minimum_top_probability": 0.20,
            "minimum_top_two_margin": 0.0,
            "target_coverage": 0.60,
            "overrides_enabled": False,
        },
        "overrides_enabled": False,
    }
    features = {
        "contract_date": "2026-07-31",
        "month": 7,
        "point_prediction_f": 92.0,
        "xgboost_predicted_high_f": 92.0,
        "lightgbm_predicted_high_f": 92.0,
        "catboost_predicted_high_f": 92.0,
        "gfs_high_f": 92.0,
        "hrrr_high_f": 92.0,
        "nbm_high_f": 92.0,
        "observed_temp_at_as_of_f": 85.0,
        "observed_high_temp_through_as_of_f": 85.0,
        "observed_as_of_age_minutes": 0.0,
    }
    result = predict_probability_bundle(bundle, features)
    assert result["probability_top_bucket_label"] == "94-95"
    assert result["point_bucket_label"] == "92-93"
    assert result["recommended_bucket_label"] == "92-93"
    assert not result["overrides_enabled"]
    assert not result["overrides_point_bucket"]


def test_frozen_candidates_have_stable_three_arm_contract() -> None:
    tuning = pd.DataFrame(
        [
            {
                "family": "empirical",
                "feature_set": "historical_only",
                "feature_count": 0,
                "c": 0.0,
                "class_weight": None,
                "temperature": 1.0,
                "model_weight": 0.0,
                "prior_strength": 30.0,
                "bucket_log_loss": 1.10,
                "ranked_probability_score": 0.12,
                "log_loss": 2.0,
            },
            {
                "family": "shared_slope_ordinal_logistic",
                "feature_set": "market_core_21",
                "feature_count": 21,
                "c": 0.03,
                "class_weight": None,
                "temperature": 1.0,
                "model_weight": 0.50,
                "prior_strength": 30.0,
                "bucket_log_loss": 1.00,
                "ranked_probability_score": 0.11,
                "log_loss": 1.9,
            },
            {
                "family": "ordinal_logistic",
                "feature_set": "compact_27",
                "feature_count": 27,
                "c": 0.03,
                "class_weight": None,
                "temperature": 1.0,
                "model_weight": 0.75,
                "prior_strength": 30.0,
                "bucket_log_loss": 1.02,
                "ranked_probability_score": 0.10,
                "log_loss": 1.8,
            },
            {
                "family": "ordinal_logistic",
                "feature_set": "full_59",
                "feature_count": 59,
                "c": 0.03,
                "class_weight": None,
                "temperature": 1.0,
                "model_weight": 1.0,
                "prior_strength": 15.0,
                "bucket_log_loss": 1.05,
                "ranked_probability_score": 0.09,
                "log_loss": 1.85,
            },
        ]
    )
    frozen = frozen_candidate_rows(tuning)
    assert tuple(frozen["candidate_role"]) == FROZEN_CANDIDATE_ROLES
    assert tuple(frozen["family"]) == (
        "ordinal_logistic",
        "shared_slope_ordinal_logistic",
        "ordinal_logistic",
    )
    assert frozen.iloc[0]["model_weight"] < 1.0
    assert frozen.iloc[2]["model_weight"] == 1.0
