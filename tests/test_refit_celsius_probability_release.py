from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/refit-celsius-probability-release.py"
SPEC = importlib.util.spec_from_file_location("refit_celsius_probability_release", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RemainingModel:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(frame), self.value)


class StackModel:
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return frame.mean(axis=1).to_numpy()


def test_serving_point_predictions_use_exact_point_bundle_models() -> None:
    features = pd.DataFrame(
        {
            "contract_date": ["2025-12-31", "2026-01-01", "2026-01-02"],
            "actual_high_f": [50.0, 60.0, 61.0],
            "observed_high_temp_through_as_of_f": [45.0, 55.0, np.nan],
            "feature": [1.0, 2.0, 3.0],
        }
    )
    bundle = {
        "feature_names": ["feature"],
        "base_models": {
            "xgboost": RemainingModel(1.0),
            "lightgbm": RemainingModel(2.0),
            "catboost": RemainingModel(3.0),
        },
        "stack_features": [
            "xgboost_predicted_high_f",
            "lightgbm_predicted_high_f",
            "catboost_predicted_high_f",
        ],
        "stack_model": StackModel(),
    }

    point, base = MODULE.serving_point_predictions(features, bundle)

    assert point["contract_date"].dt.year.tolist() == [2026]
    assert point["predicted_high_f"].tolist() == [57.0]
    assert sorted(base["predicted_high_f"].tolist()) == [56.0, 57.0, 58.0]


def test_validate_point_artifact_requires_exact_hash() -> None:
    bundle = {"station_id": "RJTT", "model_version": "point-v1"}
    manifest = {
        "model_version": "point-v1",
        "training": {"last_contract_date": "2026-07-25"},
        "artifact_integrity": {"bundle_sha256": "a" * 64},
    }

    try:
        MODULE.validate_point_artifact(bundle, manifest, "b" * 64)
    except ValueError as error:
        assert str(error) == "point bundle hash mismatch"
    else:
        raise AssertionError("mismatched point artifact was accepted")


def test_sha256_file_is_stable_for_release_inputs(tmp_path: Path) -> None:
    artifact = tmp_path / "input.csv"
    artifact.write_bytes(b"contract_date,value\n2026-01-01,1\n")

    assert MODULE.sha256_file(artifact) == (
        "d4c975da24dd0d5b07bf96f08bc9bf479641807f052425a6e1cf24db130338d9"
    )
