from __future__ import annotations

import json

import pytest

from src.tokyo_runtime_package import (
    EXPECTED_REPLAY,
    RUNTIME_CONTRACT_SHA256,
    validate_contract,
    validate_replay,
)


def replay_payload() -> dict:
    return {
        "method": {
            "station": "RJTT/Tokyo",
            "selector": "point_bucket_c",
            "entry_policy": "no_filter",
            "cost_cap": 0.47,
        },
        "flat_4": dict(EXPECTED_REPLAY),
    }


def test_reference_replay_is_exact(tmp_path):
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(replay_payload()), encoding="utf-8")
    assert validate_replay(path) == EXPECTED_REPLAY


def test_reference_replay_rejects_material_mismatch(tmp_path):
    payload = replay_payload()
    payload["flat_4"]["entries"] = 95
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="replay_entries_mismatch"):
        validate_replay(path)


def test_runtime_contract_hash_is_pinned():
    assert len(RUNTIME_CONTRACT_SHA256) == 64
    assert set(RUNTIME_CONTRACT_SHA256) <= set("0123456789abcdef")


def model_contract_fixture() -> tuple[dict, dict]:
    features = ["observed_high_temp_through_as_of_f", "gfs_high_f"]
    bundle = {
        "station_id": "RJTT",
        "model_version": "station_high_regressor_baseline_tokyo_no_peak_stack",
        "feature_version": "v20_asia_no_peak",
        "timing_mode": "asia_same_day_11am_live_safe",
        "target_source": "wunderground_only",
        "providers": ("gfs", "gefs", "jma_msm"),
        "feature_names": features,
        "base_models": {"xgboost": object()},
        "stack_model": object(),
    }
    manifest = {
        "station_id": "RJTT",
        "model_contract": {
            "feature_version": "v20_asia_no_peak",
            "providers": ["gfs", "gefs", "jma_msm"],
        },
        "features": {"all": list(features)},
    }
    return bundle, manifest


def test_runtime_contract_rejects_training_only_daily_high_features():
    bundle, manifest = model_contract_fixture()
    bundle["feature_names"].append("iem_daily_high_f")
    manifest["features"]["all"].append("iem_daily_high_f")
    with pytest.raises(ValueError, match="point_in_time_unsafe_features:iem_daily_high_f"):
        validate_contract(bundle, manifest)


def test_runtime_contract_requires_exact_manifest_feature_order():
    bundle, manifest = model_contract_fixture()
    manifest["features"]["all"].reverse()
    with pytest.raises(ValueError, match="bundle_manifest_feature_order_mismatch"):
        validate_contract(bundle, manifest)
