from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .bucket_probability import (
    CandidateSpec,
    _fit_candidate,
    _predict_candidate,
    empirical_probabilities,
    fit_tail_policy,
    temperature_scale,
)
from .kdal_ordinal_challenger import (
    _blend,
    distribution_metrics,
    feature_sets,
)


TREE_ARMS = (
    "lightgbm_multiclass_pure",
    "ordinal_lightgbm_pure",
    "ordinal_lightgbm_blended",
)
TREE_TEMPERATURES = (0.75, 1.0, 1.25, 1.5)
TREE_FEATURE_SETS = ("market_core_21", "full_59")
TREE_PARAMETER_GRID = (
    {
        "learning_rate": 0.03,
        "n_estimators": 100,
        "num_leaves": 5,
        "min_child_samples": 60,
        "reg_lambda": 20.0,
        "colsample_bytree": 0.8,
    },
    {
        "learning_rate": 0.03,
        "n_estimators": 100,
        "num_leaves": 9,
        "min_child_samples": 30,
        "reg_lambda": 10.0,
        "colsample_bytree": 0.8,
    },
    {
        "learning_rate": 0.03,
        "n_estimators": 200,
        "num_leaves": 5,
        "min_child_samples": 30,
        "reg_lambda": 10.0,
        "colsample_bytree": 0.8,
    },
    {
        "learning_rate": 0.03,
        "n_estimators": 200,
        "num_leaves": 9,
        "min_child_samples": 60,
        "reg_lambda": 20.0,
        "colsample_bytree": 0.8,
    },
)


@dataclass(frozen=True)
class TreeConfig:
    arm: str
    family: str
    feature_set: str
    params_json: str
    temperature: float
    model_weight: float
    prior_strength: float


def _arm_candidates(family: str) -> tuple[tuple[str, tuple[float, ...], tuple[float, ...]], ...]:
    if family == "lightgbm_multiclass":
        return (("lightgbm_multiclass_pure", (1.0,), (15.0,)),)
    if family == "ordinal_lightgbm":
        return (
            ("ordinal_lightgbm_pure", (1.0,), (15.0,)),
            (
                "ordinal_lightgbm_blended",
                (0.50, 0.75),
                (15.0, 30.0, 60.0),
            ),
        )
    raise ValueError(f"unsupported tree family: {family}")


def tune_tree_arms(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    random_state: int = 42,
) -> pd.DataFrame:
    sets = feature_sets()
    tail_policy = fit_tail_policy(train["exact_offset"].astype(int))
    rows: list[dict[str, Any]] = []
    empirical_cache = {
        prior: empirical_probabilities(train, valid, prior)
        for prior in (15.0, 30.0, 60.0)
    }
    for family in ("lightgbm_multiclass", "ordinal_lightgbm"):
        for feature_set in TREE_FEATURE_SETS:
            features = sets[feature_set]
            for params in TREE_PARAMETER_GRID:
                spec = CandidateSpec(family, params)
                state = _fit_candidate(
                    train,
                    features,
                    spec,
                    random_state=random_state,
                )
                raw = _predict_candidate(state, valid, features)
                for temperature in TREE_TEMPERATURES:
                    calibrated = temperature_scale(raw, temperature)
                    for arm, model_weights, prior_strengths in _arm_candidates(
                        family
                    ):
                        for prior_strength in prior_strengths:
                            empirical = empirical_cache[prior_strength]
                            for model_weight in model_weights:
                                probabilities = _blend(
                                    empirical,
                                    calibrated,
                                    model_weight,
                                )
                                metrics, _ = distribution_metrics(
                                    valid,
                                    probabilities,
                                    tail_policy=tail_policy,
                                )
                                rows.append(
                                    {
                                        "arm": arm,
                                        "family": family,
                                        "feature_set": feature_set,
                                        "feature_count": len(features),
                                        "params_json": json.dumps(
                                            params, sort_keys=True
                                        ),
                                        "temperature": temperature,
                                        "model_weight": model_weight,
                                        "prior_strength": prior_strength,
                                        **metrics,
                                    }
                                )
    tuning = pd.DataFrame(rows)
    return tuning.sort_values(
        [
            "arm",
            "bucket_log_loss",
            "ranked_probability_score",
            "log_loss",
            "feature_count",
            "params_json",
            "temperature",
            "model_weight",
            "prior_strength",
        ]
    ).reset_index(drop=True)


def best_tree_configs(tuning: pd.DataFrame) -> dict[str, TreeConfig]:
    configs: dict[str, TreeConfig] = {}
    for arm in TREE_ARMS:
        row = tuning.loc[tuning["arm"].eq(arm)].iloc[0]
        configs[arm] = TreeConfig(
            arm=arm,
            family=str(row["family"]),
            feature_set=str(row["feature_set"]),
            params_json=str(row["params_json"]),
            temperature=float(row["temperature"]),
            model_weight=float(row["model_weight"]),
            prior_strength=float(row["prior_strength"]),
        )
    return configs


def fit_tree_config(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    config: TreeConfig,
    *,
    random_state: int = 42,
) -> tuple[dict[str, float], pd.DataFrame, Mapping[str, Any]]:
    features = feature_sets()[config.feature_set]
    spec = CandidateSpec(config.family, json.loads(config.params_json))
    state = _fit_candidate(
        train,
        features,
        spec,
        random_state=random_state,
    )
    raw = _predict_candidate(state, valid, features)
    calibrated = temperature_scale(raw, config.temperature)
    empirical = empirical_probabilities(
        train,
        valid,
        config.prior_strength,
    )
    probabilities = _blend(empirical, calibrated, config.model_weight)
    metrics, predictions = distribution_metrics(
        valid,
        probabilities,
        tail_policy=fit_tail_policy(train["exact_offset"].astype(int)),
    )
    return metrics, predictions, state


def nested_tree_evaluation(
    development: pd.DataFrame,
    *,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_frames: list[pd.DataFrame] = []
    selections: list[dict[str, Any]] = []
    tuning_frames: list[pd.DataFrame] = []
    for validation_year in (2024, 2025):
        outer_train = development.loc[
            development["year"].lt(validation_year)
        ].copy()
        outer_valid = development.loc[
            development["year"].eq(validation_year)
        ].copy()
        split_at = outer_train["contract_date"].max() - pd.Timedelta(days=89)
        inner_train = outer_train.loc[
            outer_train["contract_date"].lt(split_at)
        ].copy()
        inner_valid = outer_train.loc[
            outer_train["contract_date"].ge(split_at)
        ].copy()
        if (
            inner_train.empty
            or inner_valid.empty
            or outer_valid.empty
            or inner_train["contract_date"].max()
            >= inner_valid["contract_date"].min()
            or inner_valid["contract_date"].max()
            >= outer_valid["contract_date"].min()
        ):
            raise AssertionError("tree challenger chronology failed")
        tuning = tune_tree_arms(
            inner_train,
            inner_valid,
            random_state=random_state,
        )
        tuning["outer_validation_year"] = validation_year
        tuning_frames.append(tuning)
        for arm, config in best_tree_configs(tuning).items():
            metrics, predictions, _ = fit_tree_config(
                outer_train,
                outer_valid,
                config,
                random_state=random_state,
            )
            predictions["arm"] = arm
            predictions["outer_validation_year"] = validation_year
            predictions["model_training_cutoff"] = outer_train[
                "contract_date"
            ].max()
            prediction_frames.append(predictions)
            selections.append(
                {
                    "outer_validation_year": validation_year,
                    **config.__dict__,
                    **{f"outer_{key}": value for key, value in metrics.items()},
                }
            )
    return (
        pd.concat(prediction_frames, ignore_index=True),
        pd.DataFrame(selections),
        pd.concat(tuning_frames, ignore_index=True),
    )
