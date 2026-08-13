from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "adopt-celsius-probability-release.py"
SPEC = importlib.util.spec_from_file_location("adopt_celsius_probability_release", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def source_bundle() -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "station_celsius_market_probability_model",
        "station_id": "RJTT",
        "model_version": "source-probability",
        "point_model_version": "evaluation-point",
        "point_bundle_sha256": "a" * 64,
        "feature_profile": "asia_no_peak",
        "selection_excludes_holdout": True,
        "training_cutoff": "2025-12-31",
        "decision_thresholds": {"minimum_top_probability": 0.525},
        "model_state": {"sentinel": [1.0, 2.0, 3.0]},
    }


def test_adoption_changes_only_serving_identity_and_preserves_frozen_policy() -> None:
    source = source_bundle()
    adopted = module.adopted_bundle(
        source,
        source_bundle_hash="b" * 64,
        source_manifest_hash="c" * 64,
        target_point_model_version="evaluation-point",
        target_point_bundle_hash="a" * 64,
        adopted_model_version="live-probability",
    )

    assert adopted["model_version"] == "live-probability"
    assert adopted["point_model_version"] == "evaluation-point"
    assert adopted["point_bundle_sha256"] == "a" * 64
    assert adopted["decision_thresholds"] == source["decision_thresholds"]
    assert adopted["model_state"] == source["model_state"]
    assert adopted["serving_adoption"]["fitting_performed"] is False
    assert adopted["serving_adoption"]["threshold_selection_performed"] is False


def test_adoption_rejects_different_point_release() -> None:
    try:
        module.adopted_bundle(
            source_bundle(),
            source_bundle_hash="b" * 64,
            source_manifest_hash="c" * 64,
            target_point_model_version="different-live-point",
            target_point_bundle_hash="d" * 64,
            adopted_model_version="live-probability",
        )
    except ValueError as error:
        assert "exact source point model" in str(error)
    else:
        raise AssertionError("metadata-only point-model rebinding was allowed")


def test_source_validation_rejects_holdout_trained_probability(tmp_path: Path) -> None:
    bundle = source_bundle()
    bundle["training_cutoff"] = "2026-07-25"
    bundle_path = tmp_path / "bundle.joblib"
    bundle_path.write_bytes(b"bundle")
    manifest = {
        "artifact_type": bundle["artifact_type"],
        "artifact_integrity": {"bundle_sha256": module.sha256_file(bundle_path)},
        "point_bundle_sha256": bundle["point_bundle_sha256"],
        "decision_thresholds": bundle["decision_thresholds"],
    }

    try:
        module.validate_source_probability(bundle, manifest, bundle_path)
    except ValueError as error:
        assert str(error) == "probability_training_cutoff_not_pre_2026"
    else:
        raise AssertionError("post-2025 probability training cutoff was accepted")
