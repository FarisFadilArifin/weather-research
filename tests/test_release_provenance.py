from __future__ import annotations

from pathlib import Path

import pytest

from src.calibration import release_provenance


def _files(root: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    artifacts = {
        "dataset": root / "data" / "dataset.parquet",
        "features": root / "data" / "features.parquet",
        "notebook": root / "notebooks" / "train.ipynb",
        "export": root / "exports" / "forward.csv",
        "model": root / "weights" / "model.joblib",
        "model_manifest": root / "weights" / "model.json",
    }
    runtime = {"lock": root / "uv.lock", "project": root / "pyproject.toml"}
    for position, path in enumerate([*artifacts.values(), *runtime.values()]):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"input-{position}\n", encoding="utf-8")
    return artifacts, runtime


def _release_inputs(root: Path) -> dict:
    artifacts, runtime = _files(root)
    return {
        "project_root": root,
        "release_id": "rjtt-2026-08-08",
        "station_id": "RJTT",
        "model_version": "station_bucket_baseline_tokyo_1c_market_ordinal",
        "artifact_paths": artifacts,
        "runtime_paths": runtime,
        "provider_contract": {
            "providers": ["gfs", "gefs", "jma_msm"],
            "timezone": "Asia/Tokyo",
            "inference_cutoff": "11:00 local",
        },
        "required_history": {
            "training_years": [2024, 2025],
            "forward_validation_years": [2025],
            "holdout_year": 2026,
            "selection_excludes_holdout": True,
        },
        "missingness": {
            "max_feature_missing_fraction": 0.03,
            "observed_fraction_by_feature": {"gfs_high_f": 1.0, "gefs_high_f": 0.99},
        },
    }


def test_release_manifest_records_and_verifies_all_immutable_inputs(tmp_path, monkeypatch) -> None:
    source = {"commit": "a" * 40, "worktree_clean": True}
    monkeypatch.setattr(release_provenance, "source_revision", lambda _: source)
    inputs = _release_inputs(tmp_path)

    manifest = release_provenance.build_release_manifest(**inputs)
    output = release_provenance.write_release_manifest(
        tmp_path / "release-registry" / "rjtt-2026-08-08.json", manifest
    )

    assert manifest["source"] == source
    assert set(manifest["artifacts"]) == {
        "dataset", "features", "notebook", "export", "model", "model_manifest"
    }
    assert set(manifest["runtime"]) == {"lock", "project"}
    assert output.is_file()
    assert release_provenance.load_and_verify_release_manifest(
        tmp_path, output, expected_provider_contract=inputs["provider_contract"]
    ) == manifest


def test_release_manifest_rejects_hash_drift_and_overwrite(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        release_provenance,
        "source_revision",
        lambda _: {"commit": "b" * 40, "worktree_clean": True},
    )
    inputs = _release_inputs(tmp_path)
    manifest = release_provenance.build_release_manifest(**inputs)
    output = release_provenance.write_release_manifest(tmp_path / "release.json", manifest)
    inputs["artifact_paths"]["model"].write_text("changed\n", encoding="utf-8")

    with pytest.raises(release_provenance.ReleaseProvenanceError, match="hash mismatch"):
        release_provenance.load_and_verify_release_manifest(tmp_path, output)
    with pytest.raises(release_provenance.ReleaseProvenanceError, match="immutable"):
        release_provenance.write_release_manifest(output, manifest)


def test_release_manifest_fails_closed_for_dirty_source_and_missingness(tmp_path, monkeypatch) -> None:
    def dirty_git_output(_: Path, *args: str) -> str:
        if args[:2] == ("rev-parse", "HEAD"):
            return "c" * 40 + "\n"
        return " M src/calibration/release_provenance.py\0"

    monkeypatch.setattr(release_provenance, "_git_output", dirty_git_output)
    with pytest.raises(release_provenance.ReleaseProvenanceError, match="clean worktree"):
        release_provenance.source_revision(tmp_path)

    monkeypatch.setattr(
        release_provenance,
        "source_revision",
        lambda _: {"commit": "c" * 40, "worktree_clean": True},
    )
    inputs = _release_inputs(tmp_path)
    inputs["missingness"] = {
        "max_feature_missing_fraction": 0.03,
        "observed_fraction_by_feature": {"gfs_high_f": 0.96},
    }
    with pytest.raises(release_provenance.ReleaseProvenanceError, match="missingness exceeds"):
        release_provenance.build_release_manifest(**inputs)
