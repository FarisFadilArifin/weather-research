from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.calibration.expert_ensemble import (
    EXPERT_METHODS,
    FittedExpert,
    ExpertFitAudit,
    fold_feature_contract,
    route_expert_features,
    target_values,
    export_point_bundle,
    sha256_file,
    validate_frozen_feature_contract,
)


class _ConstantPipeline:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, values) -> np.ndarray:
        return np.full(len(values), self.value, dtype=float)


def _frame() -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    return pd.DataFrame(
        {
            "contract_date": dates,
            "actual_high_f": np.linspace(70.0, 90.0, len(dates)),
            "settlement_high_f": np.linspace(70.0, 90.0, len(dates)),
            "observed_high_temp_through_as_of_f": np.linspace(65.0, 84.0, len(dates)),
            "observed_temp_at_as_of_f": np.linspace(64.0, 83.0, len(dates)),
            "observed_humidity_at_as_of": 50.0,
            "provider_mean_high_f": np.linspace(69.0, 89.0, len(dates)),
            "gfs_high_f": np.linspace(69.0, 89.0, len(dates)),
            "gfs_forecast_lead_hours": 6.0,
            "gfs_rolling_bias_7d_f": 0.2,
            "gfs_cloud_cover_mean": 25.0,
            "v11sf_forecast_temp_11am_minus_observed_f": 1.0,
            "actual_high_lag_1d": np.linspace(69.0, 89.0, len(dates)),
            "actual_high_roll_7d_mean": np.linspace(68.0, 88.0, len(dates)),
            "day_of_year": dates.dayofyear,
            "mostly_missing": [np.nan] * 4 + [1.0] * 96,
        }
    )


def test_expert_feature_routing_enforces_role_exclusions() -> None:
    frame = _frame()
    routes = {method: route_expert_features(frame, method) for method in EXPERT_METHODS}
    assert "gfs_high_f" in routes["full_xgboost"]
    assert "gfs_rolling_bias_7d_f" in routes["forecast_huber"]
    assert "observed_humidity_at_as_of" in routes["observation_catboost"]
    assert "gfs_high_f" not in routes["observation_catboost"]
    assert "actual_high_lag_1d" not in routes["observation_catboost"]
    assert set(routes["seasonal_ridge"]) == {"actual_high_lag_1d", "actual_high_roll_7d_mean", "day_of_year"}
    assert all("settlement" not in name for names in routes.values() for name in names)


def test_fold_missingness_gate_is_owned_by_exact_training_population() -> None:
    frame = _frame()
    eligible, rejected = fold_feature_contract(frame, "full_xgboost")
    assert "mostly_missing" not in eligible
    assert rejected["mostly_missing"] == pytest.approx(0.04)
    later = frame.iloc[4:].copy()
    eligible_later, rejected_later = fold_feature_contract(later, "full_xgboost")
    assert "mostly_missing" in eligible_later
    assert "mostly_missing" not in rejected_later


def test_target_transforms_and_physical_floor() -> None:
    frame = _frame().iloc[:2].copy()
    assert np.allclose(target_values(frame, "full_xgboost"), frame["actual_high_f"] - frame["observed_high_temp_through_as_of_f"])
    assert np.allclose(target_values(frame, "forecast_huber"), frame["actual_high_f"] - frame["provider_mean_high_f"])
    audit = ExpertFitAudit("full_xgboost", "2023-01-01", "2023-01-02", 2, 1, 1, 0.03, ("observed_temp_at_as_of_f",), {}, "remaining_warmup", {})
    expert = FittedExpert("full_xgboost", ("observed_temp_at_as_of_f",), _ConstantPipeline(-10.0), "remaining_warmup", audit)
    assert np.allclose(expert.predict(frame), frame["observed_high_temp_through_as_of_f"])


def test_frozen_contract_fails_closed_and_export_hash_is_verified(tmp_path) -> None:
    frame = _frame()
    audit = ExpertFitAudit("full_xgboost", "2023-01-01", "2023-04-10", 100, 1, 1, 0.03, ("observed_temp_at_as_of_f",), {}, "remaining_warmup", {})
    experts = {
        method: FittedExpert(method, ("observed_temp_at_as_of_f",), _ConstantPipeline(1.0), "remaining_warmup", audit)
        for method in EXPERT_METHODS
    }
    validate_frozen_feature_contract(frame, experts["full_xgboost"])
    broken = frame.copy()
    broken.loc[:4, "observed_temp_at_as_of_f"] = np.nan
    with pytest.raises(ValueError, match="exceeds 3%"):
        validate_frozen_feature_contract(broken, experts["full_xgboost"])
    bundle_path, manifest_path = export_point_bundle(
        tmp_path,
        station_id="KDAL",
        model_version="test",
        experts=experts,
        weights={method: 0.25 for method in EXPERT_METHODS},
        station_contract={"bucket": "2f"},
        source_identity={"source_pipeline": "test"},
        chronology={"through_year": 2025},
    )
    import json

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["artifact_integrity"]["bundle_sha256"] == sha256_file(bundle_path)
    assert manifest["research_only"] is True
    assert manifest["live_refit_enabled"] is False
