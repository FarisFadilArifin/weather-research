from __future__ import annotations

import numpy as np
import pandas as pd

from src.calibration.round_override_v4 import (
    V4_PEAK_FEATURES,
    _rolling_bucket_probability,
    tune_utility_policy,
    utility_feature_names,
)
from src.calibration.win_classifier import KATL_PEAK_FEATURES


def test_kdal_v4_inventory_excludes_every_peak_feature() -> None:
    features = set(utility_feature_names(include_peak_features=False))
    assert features.isdisjoint(KATL_PEAK_FEATURES)
    assert features.isdisjoint(V4_PEAK_FEATURES)
    assert len(features) <= 36


def test_continuous_bucket_probability_prefers_centered_bucket() -> None:
    point = pd.Series([90.2, 90.2])
    labels = pd.Series(["90-91", "92-93"])
    mean = pd.Series([0.0, 0.0])
    std = pd.Series([0.8, 0.8])
    probability = _rolling_bucket_probability(point, labels, mean, std)
    assert probability.between(0, 1).all()
    assert probability.iloc[0] > probability.iloc[1]


def _policy_frame(*, unstable: bool) -> pd.DataFrame:
    count = 90
    frame = pd.DataFrame(
        {
            "contract_date": pd.date_range("2023-10-03", periods=count),
            "override_actionable": 1,
            "recovery_probability": 0.1,
            "damage_probability": 0.4,
            "recovery_target": 0,
            "damage_target": 0,
        }
    )
    for fold_start in (0, 30, 60):
        recovery_rows = range(fold_start, fold_start + 3)
        frame.loc[recovery_rows, ["recovery_probability", "damage_probability", "recovery_target"]] = [0.9, 0.05, 1]
    if unstable:
        frame.loc[60:64, ["recovery_probability", "damage_probability", "damage_target"]] = [0.9, 0.05, 1]
        frame.loc[60:64, "recovery_target"] = 0
    return frame


def test_utility_policy_requires_multi_fold_stability() -> None:
    decision, table = tune_utility_policy(_policy_frame(unstable=False))
    assert decision["policy_enabled"]
    assert decision["policy_stable"]
    assert table["minimum_recovery_probability"].min() == 0.05
    assert table["minimum_recovery_probability"].max() == 0.95
    assert table["damage_penalty"].min() == 2.0
    assert table.loc[table["eligible"], "negative_policy_folds"].eq(0).all()
    assert table.loc[table["eligible"], "fixed_cost_utility"].gt(0).all()


def test_utility_policy_abstains_when_one_fold_is_harmful() -> None:
    decision, _ = tune_utility_policy(_policy_frame(unstable=True))
    assert not decision["policy_enabled"]
    assert not decision["policy_stable"]
    assert np.isinf(decision["minimum_utility_margin"])
