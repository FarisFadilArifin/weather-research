from __future__ import annotations

import hashlib

from src.export_station_stacking_v2_models import (
    _feature_pipeline_name,
    _git_identity,
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
