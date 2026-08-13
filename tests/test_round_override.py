from __future__ import annotations

import pandas as pd

from src.calibration.round_override import audit_round_override_system, override_feature_names
from src.calibration.win_classifier import KATL_PEAK_FEATURES


def test_kdal_override_inventory_excludes_peak_features() -> None:
    assert set(override_feature_names(include_peak_features=False)).isdisjoint(KATL_PEAK_FEATURES)


def test_override_audit_accepts_bucket_aware_binary_contract() -> None:
    source = pd.DataFrame(
        {
            "default_half_up": [0, 1],
            "floor_degree_f": [93, 92],
            "ceil_degree_f": [94, 93],
            "alternative_degree_f": [94, 92],
            "default_degree_f": [93, 93],
            "point_degree_f": [93, 93],
            "default_bucket_label": ["92-93", "92-93"],
            "alternative_bucket_label": ["94-95", "92-93"],
            "actual_bucket_label": ["94-95", "92-93"],
            "override_actionable": [1, 0],
            "override_target": [1, 0],
            "year": [2024, 2024],
            "train_through_year": [2023, 2023],
        }
    )
    forward = pd.DataFrame(
        {
            "model_training_cutoff": ["2023-08-31"] * 2,
            "calibration_start": ["2023-09-01"] * 2,
            "calibration_cutoff": ["2023-10-31"] * 2,
            "policy_start": ["2023-11-01"] * 2,
            "policy_cutoff": ["2023-12-31"] * 2,
            "contract_date": ["2024-01-01", "2024-01-02"],
            "override_probability": [0.8, 0.1],
            "override": [True, False],
            "override_actionable": [1, 0],
            "final_bucket_win": [1, 1],
            "default_bucket_win": [0, 1],
            "recovered_loss": [1, 0],
            "damaged_win": [0, 0],
        }
    )
    result = {"forward_predictions": forward, "feature_names": override_feature_names(include_peak_features=False)}
    audit = audit_round_override_system(source, result, include_peak_features=False)
    assert audit["passed"].all(), audit.to_dict(orient="records")
