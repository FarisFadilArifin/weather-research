from __future__ import annotations

import argparse
import copy
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib

from src.tokyo_runtime_package import (
    FEATURE_PIPELINE,
    PREDICTION_UNIT,
    RUNTIME_CONTRACT_SHA256,
    assert_prediction_parity,
    git_identity,
    sha256_file,
    validate_contract,
    validate_replay,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote the immutable RJTT runtime model")
    parser.add_argument("--source-bundle", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--features-csv", required=True, type=Path)
    parser.add_argument("--replay-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args()


def runtime_package_versions(project_root: Path) -> dict[str, str]:
    versions: dict[str, str] = {}
    for raw_line in (project_root / "requirements-ml-runtime.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            name, version = line.split("==", 1)
            versions[name] = version
    return versions


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    identity = git_identity(root)
    if identity.git_dirty:
        raise SystemExit("refusing export: source worktree is dirty")

    source_bundle = joblib.load(args.source_bundle)
    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    validate_contract(source_bundle, source_manifest)
    replay = validate_replay(args.replay_summary)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    stem = "RJTT_station_high_regressor_baseline_tokyo_no_peak_stack"
    bundle_path = output / f"{stem}.joblib"
    manifest_path = output / f"{stem}.json"
    parity_path = output / f"{stem}.parity.json"

    joblib.dump(source_bundle, bundle_path, compress=3)
    promoted_bundle = joblib.load(bundle_path)
    parity = assert_prediction_parity(source_bundle, promoted_bundle, args.features_csv)
    parity_path.write_text(json.dumps(parity, indent=2) + "\n", encoding="utf-8")

    manifest = copy.deepcopy(source_manifest)
    manifest["created_at_utc"] = datetime.now(UTC).isoformat()
    manifest["source_pipeline"] = "scripts/promote_tokyo_runtime_model.py"
    manifest["source_artifact_dir"] = "external/rjtt-training-artifacts"
    manifest["bundle_path"] = bundle_path.name
    manifest["source_identity"] = {
        "git_commit": identity.git_commit,
        "git_dirty": False,
    }
    manifest["package_runtime_compatibility"]["runtime_contract"] = "requirements-ml-runtime.txt"
    manifest["package_runtime_compatibility"]["runtime_contract_sha256"] = (
        RUNTIME_CONTRACT_SHA256
    )
    manifest["package_runtime_compatibility"]["package_versions"] = (
        runtime_package_versions(root)
    )
    manifest["package_runtime_compatibility"]["feature_pipeline"] = FEATURE_PIPELINE
    manifest.setdefault("model_contract", {})["prediction_temperature_unit"] = PREDICTION_UNIT
    manifest["promotion"] = {
        "immutable_runtime_package": True,
        "training_source_identity": copy.deepcopy(source_manifest.get("source_identity") or {}),
        "source_bundle_sha256": sha256_file(args.source_bundle),
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "fixed_reference_rows": len(parity),
        "maximum_prediction_difference_f": max(row["absolute_difference_f"] for row in parity),
        "parity_evidence": parity_path.name,
        "reference_replay": replay,
    }
    manifest["artifact_integrity"] = {
        "bundle_sha256": sha256_file(bundle_path),
        "parity_sha256": sha256_file(parity_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "bundle": str(bundle_path),
        "manifest": str(manifest_path),
        "parity": str(parity_path),
        "bundle_sha256": sha256_file(bundle_path),
        "manifest_sha256": sha256_file(manifest_path),
        "source_git_commit": identity.git_commit,
        "source_git_dirty": False,
        "reference_replay": replay,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
