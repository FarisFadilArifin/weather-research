from __future__ import annotations

import pandas as pd

from src.calibration.round_direction import audit_round_direction_system, round_feature_names
from src.calibration.win_classifier import KATL_PEAK_FEATURES


def test_kdal_round_inventory_excludes_peak_features() -> None:
    assert set(round_feature_names(include_peak_features=False)).isdisjoint(KATL_PEAK_FEATURES)


def test_round_audit_accepts_exact_binary_contract() -> None:
    source = pd.DataFrame(
        {
            "point_prediction_f": [93.4, 92.6],
            "actual_high_f": [94.0, 92.0],
            "floor_degree_f": [93, 92],
            "ceil_degree_f": [94, 93],
            "round_up": [1, 0],
            "point_degree_f": [93, 93],
            "year": [2024, 2024],
            "train_through_year": [2023, 2023],
        }
    )
    forward = pd.DataFrame(
        {
            "model_training_cutoff": ["2023-10-31"] * 2,
            "calibration_start": ["2023-11-01"] * 2,
            "calibration_cutoff": ["2023-12-31"] * 2,
            "contract_date": ["2024-01-01", "2024-01-02"],
            "round_up_probability": [0.8, 0.2],
            "predicted_round_up": [1, 0],
            "corrected_degree_f": [94, 92],
            "floor_degree_f": [93, 92],
            "ceil_degree_f": [94, 93],
            "corrected_bucket_win": [1, 1],
            "point_bucket_win": [0, 1],
            "recovered_loss": [1, 0],
            "damaged_win": [0, 0],
        }
    )
    result = {"forward_predictions": forward, "feature_names": round_feature_names(include_peak_features=False)}
    audit = audit_round_direction_system(source, result, include_peak_features=False)
    assert audit["passed"].all(), audit.to_dict(orient="records")
