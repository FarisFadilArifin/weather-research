#!/usr/bin/env python3
"""Adopt a frozen Celsius probability estimator for an immutable point release.

This command never fits a model or changes probability thresholds. It creates a
new serving bundle whose only behavioral dependency change is the explicitly
validated point-model version/hash used at inference.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping


PROBABILITY_ARTIFACT_TYPE = "station_celsius_market_probability_model"
POINT_ARTIFACT_TYPE = "station_high_regression_model_weights"
EXPECTED_PROFILE = "asia_no_peak"
EXPECTED_STATION = "RJTT"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def clean_source_identity(repo: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise ValueError("research source tree must be clean before adoption")
    return {
        "git_commit": commit,
        "git_dirty": False,
        "identity_role": "celsius_probability_serving_adoption",
    }


def validate_source_probability(
    bundle: Mapping[str, Any], manifest: Mapping[str, Any], bundle_path: Path
) -> None:
    checks = (
        (bundle.get("schema_version") == 1, "probability_bundle_schema_mismatch"),
        (bundle.get("artifact_type") == PROBABILITY_ARTIFACT_TYPE, "probability_bundle_type_mismatch"),
        (str(bundle.get("station_id") or "").upper() == EXPECTED_STATION, "probability_station_mismatch"),
        (bundle.get("feature_profile") == EXPECTED_PROFILE, "probability_feature_profile_mismatch"),
        (bundle.get("selection_excludes_holdout") is True, "probability_holdout_selection_mismatch"),
        (str(bundle.get("training_cutoff") or "") < "2026-01-01", "probability_training_cutoff_not_pre_2026"),
        (manifest.get("artifact_type") == PROBABILITY_ARTIFACT_TYPE, "probability_manifest_type_mismatch"),
        (
            (manifest.get("artifact_integrity") or {}).get("bundle_sha256")
            == sha256_file(bundle_path),
            "probability_source_bundle_hash_mismatch",
        ),
        (manifest.get("point_bundle_sha256") == bundle.get("point_bundle_sha256"), "probability_source_point_hash_mismatch"),
        (manifest.get("decision_thresholds") == bundle.get("decision_thresholds"), "probability_threshold_mismatch"),
    )
    for passed, reason in checks:
        if not passed:
            raise ValueError(reason)


def validate_point_release(
    bundle: Mapping[str, Any], manifest: Mapping[str, Any], bundle_path: Path
) -> None:
    checks = (
        (bundle.get("schema_version") == 1, "point_bundle_schema_mismatch"),
        (str(bundle.get("station_id") or "").upper() == EXPECTED_STATION, "point_station_mismatch"),
        (manifest.get("artifact_type") == POINT_ARTIFACT_TYPE, "point_manifest_type_mismatch"),
        (manifest.get("model_version") == bundle.get("model_version"), "point_model_version_mismatch"),
        (
            (manifest.get("artifact_integrity") or {}).get("bundle_sha256")
            == sha256_file(bundle_path),
            "point_bundle_hash_mismatch",
        ),
        ((manifest.get("source_identity") or {}).get("git_dirty") is False, "point_source_not_clean"),
        (set(bundle.get("providers") or ()) == {"gfs", "gefs", "jma_msm"}, "point_provider_contract_mismatch"),
    )
    for passed, reason in checks:
        if not passed:
            raise ValueError(reason)


def adopted_bundle(
    source: Mapping[str, Any],
    *,
    source_bundle_hash: str,
    source_manifest_hash: str,
    target_point_model_version: str,
    target_point_bundle_hash: str,
    adopted_model_version: str,
) -> dict[str, Any]:
    output = copy.deepcopy(dict(source))
    original_thresholds = copy.deepcopy(source["decision_thresholds"])
    original_model_state = output["model_state"]
    output["model_version"] = adopted_model_version
    output["point_model_version"] = target_point_model_version
    output["point_bundle_sha256"] = target_point_bundle_hash
    output["serving_adoption"] = {
        "method": "metadata_only_point_release_binding",
        "fitting_performed": False,
        "threshold_selection_performed": False,
        "source_probability_bundle_sha256": source_bundle_hash,
        "source_probability_manifest_sha256": source_manifest_hash,
        "source_probability_model_version": source["model_version"],
        "source_point_model_version": source["point_model_version"],
        "source_point_bundle_sha256": source["point_bundle_sha256"],
        "target_point_model_version": target_point_model_version,
        "target_point_bundle_sha256": target_point_bundle_hash,
    }
    if output["model_state"] is not original_model_state:
        raise AssertionError("probability estimator state was unexpectedly replaced")
    if output["decision_thresholds"] != original_thresholds:
        raise AssertionError("probability thresholds changed during adoption")
    return output


def write_release(
    bundle: Mapping[str, Any], output_dir: Path, source_identity: Mapping[str, Any]
) -> tuple[Path, Path]:
    import joblib

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{bundle['station_id']}_{bundle['model_version']}"
    bundle_path = output_dir / f"{stem}.joblib"
    manifest_path = output_dir / f"{stem}.json"
    joblib.dump(dict(bundle), bundle_path)
    manifest_keys = [key for key in bundle if key != "model_state"]
    manifest = {key: copy.deepcopy(bundle[key]) for key in manifest_keys}
    manifest["source_identity"] = dict(source_identity)
    manifest["activation_status"] = "operator_requested_local_integration_not_deployed"
    manifest["artifact_integrity"] = {"bundle_sha256": sha256_file(bundle_path)}
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return bundle_path, manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-probability-bundle", type=Path, required=True)
    parser.add_argument("--source-probability-manifest", type=Path, required=True)
    parser.add_argument("--point-bundle", type=Path, required=True)
    parser.add_argument("--point-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import joblib

    source_bundle = joblib.load(args.source_probability_bundle)
    source_manifest = load_json(args.source_probability_manifest)
    point_bundle = joblib.load(args.point_bundle)
    point_manifest = load_json(args.point_manifest)
    validate_source_probability(source_bundle, source_manifest, args.source_probability_bundle)
    validate_point_release(point_bundle, point_manifest, args.point_bundle)
    source_identity = clean_source_identity(args.repo.resolve())
    bundle = adopted_bundle(
        source_bundle,
        source_bundle_hash=sha256_file(args.source_probability_bundle),
        source_manifest_hash=sha256_file(args.source_probability_manifest),
        target_point_model_version=str(point_bundle["model_version"]),
        target_point_bundle_hash=sha256_file(args.point_bundle),
        adopted_model_version=args.model_version,
    )
    bundle_path, manifest_path = write_release(bundle, args.output_dir, source_identity)
    print(json.dumps({
        "bundlePath": str(bundle_path.resolve()),
        "bundleSha256": sha256_file(bundle_path),
        "manifestPath": str(manifest_path.resolve()),
        "manifestSha256": sha256_file(manifest_path),
        "pointModelVersion": bundle["point_model_version"],
        "pointBundleSha256": bundle["point_bundle_sha256"],
        "fittingPerformed": False,
        "thresholdSelectionPerformed": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
