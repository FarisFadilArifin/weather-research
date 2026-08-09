from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from src.calibration.station_stacking import StationStackingConfig
from src.export_station_stacking_v2_models import (
    CELSIUS_BUCKET_CONTRACT,
    HKO_BUCKET_CONTRACT,
    _base_prediction_transform,
    _bucket_probability_policy,
    _feature_pipeline_name,
    _feature_missingness_audit,
    _git_identity,
    _ridge_residual_calibrator,
    _runtime_package_versions,
    _sha256_file,
)


def test_export_identity_helpers_cover_mixed_live_contracts(tmp_path) -> None:
    bundle = tmp_path / "model.joblib"
    bundle.write_bytes(b"immutable-model")
    assert _sha256_file(bundle) == hashlib.sha256(b"immutable-model").hexdigest()
    assert _feature_pipeline_name("v20_peak_timing") == "station_stacking_v20_peak_timing"
    assert (
        _feature_pipeline_name("v11_settlement_fix_temp")
        == "station_stacking_v11_settlement_fix"
    )


def test_export_runtime_and_source_identity_are_machine_readable() -> None:
    versions = _runtime_package_versions()
    assert set(versions) == {
        "catboost",
        "joblib",
        "lightgbm",
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
    }
    identity = _git_identity(__import__("pathlib").Path(__file__).resolve().parents[1])
    assert set(identity) == {"git_commit", "git_dirty"}


def test_hko_bucket_policy_is_one_degree_celsius() -> None:
    calibrator = {"method": "ridge_stack"}
    policy = _bucket_probability_policy(calibrator, HKO_BUCKET_CONTRACT)
    assert policy["bucket_rounding"] == "floor_1c"
    assert policy["bucket_unit"] == "celsius"
    assert policy["bucket_width_c"] == 1.0
    assert policy["bucket_interval"] == "[n,n+1)"
    assert policy["continuity_correction_c"] == 0.5
    assert policy["continuity_correction_f"] == 0.9


def test_tokyo_bucket_policy_is_nearest_whole_degree_celsius() -> None:
    policy = _bucket_probability_policy(
        {"method": "ridge_stack"}, CELSIUS_BUCKET_CONTRACT
    )
    assert policy["bucket_rounding"] == "polymarket_half_up_1c"
    assert policy["bucket_unit"] == "celsius"
    assert policy["bucket_width_c"] == 1.0
    assert policy["point_bucket"] == "floor(predictedHighC+0.5)"


def test_final_refit_missingness_audit_enforces_three_percent_boundary() -> None:
    train = pd.DataFrame(
        {
            "accepted": [np.nan] * 3 + [1.0] * 97,
            "rejected": [np.nan] * 4 + [1.0] * 96,
        }
    )
    audit = _feature_missingness_audit(
        train,
        [],
        ["accepted", "rejected"],
        max_missing_fraction=0.03,
    )
    by_feature = {row["feature"]: row for row in audit}
    assert by_feature["accepted"]["selected"] is True
    assert by_feature["accepted"]["missing_fraction"] == 0.03
    assert by_feature["rejected"]["selected"] is False
    assert by_feature["rejected"]["exclusion_reason"] == "above_missingness_threshold"


def test_cross_station_remaining_delta_export_does_not_clamp_to_observation() -> None:
    config = StationStackingConfig(
        station_id="HKO",
        target_mode="remaining_warmup",
        observation_target_same_station=False,
    )

    transform = _base_prediction_transform(config)

    assert transform == "predicted_high_f=observed_high_temp_through_as_of_f+model_output"
    assert "max(" not in transform


def test_residual_calibrator_uses_only_selected_stack_features() -> None:
    class ZeroModel:
        def predict(self, values):
            return np.zeros(len(values))

    rows = []
    for contract_date, actual in [("2025-01-01", 80.0), ("2025-01-02", 82.0)]:
        for method, predicted in [("xgboost", 79.0), ("lightgbm", 80.0), ("catboost", 81.0)]:
            rows.append(
                {
                    "contract_date": contract_date,
                    "method": method,
                    "actual_high_f": actual,
                    "predicted_high_f": predicted,
                }
            )
    predictions = pd.DataFrame(rows)
    features = [
        "xgboost_predicted_high_f",
        "lightgbm_predicted_high_f",
        "catboost_predicted_high_f",
    ]

    calibrator = _ridge_residual_calibrator(
        predictions,
        ZeroModel(),
        features,
        ("xgboost", "lightgbm", "catboost"),
    )

    assert calibrator["source"] == "validation_predictions"
    assert calibrator["row_count"] == 2
