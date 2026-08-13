"""Package settlement-trained v11 bundles for the weather-bot runtime.

This is deliberately a packaging step, not training: it preserves the
settlement-first model weights and gives the runtime copies provenance and a
pinned-package compatibility contract.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import joblib


REPO_ROOT = Path(__file__).resolve().parents[1]
BOT_ROOT = Path(r"D:\dev\polymarket-weather-prediction")
MODEL_VERSION = "station_high_regressor_v11_wunderground_settlement_stack"
SOURCE_PIPELINE = "notebooks/experiments/station_stacking_v11_settlement"
FEATURE_PIPELINE = "station_stacking_v11"
TARGET_SOURCE = "settlement_first"
STATIONS = ("KATL", "KDAL", "KHOU", "KSEA")
HARD_REQUIRED_LIVE_FEATURES = {
    "gfs_high_f",
    "hrrr_high_f",
    "nbm_high_f",
    "observed_temp_at_as_of_f",
    "observed_high_temp_through_as_of_f",
    "observed_as_of_age_minutes",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=REPO_ROOT / "data/calibration/station_stacking_v11_settlement/model_weights",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=REPO_ROOT / "data/calibration/station_stacking_v11_settlement",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BOT_ROOT / "data/calibration/station_stacking_v11/model_weights",
    )
    parser.add_argument(
        "--runtime-requirements",
        type=Path,
        default=BOT_ROOT / "requirements-ml-runtime.txt",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pinned_package_versions(path: Path) -> dict[str, str]:
    versions: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        package, version = line.split("==", 1)
        versions[package.strip().lower()] = version.strip()
    if not versions:
        raise ValueError(f"no pinned package versions found in {path}")
    return versions


def prediction_period(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        dates = [row["contract_date"] for row in csv.DictReader(handle) if row.get("contract_date")]
    if not dates:
        raise ValueError(f"no contract dates in {path}")
    return {"start": min(dates), "end": max(dates)}


def validate_source(station: str, bundle: dict[str, Any], manifest: dict[str, Any]) -> None:
    contract = manifest.get("model_contract") or {}
    if manifest.get("station_id") != station or bundle.get("station_id") != station:
        raise ValueError(f"{station}: station identity mismatch")
    if manifest.get("model_version") != MODEL_VERSION or bundle.get("model_version") != MODEL_VERSION:
        raise ValueError(f"{station}: model version mismatch")
    if manifest.get("source_pipeline") != SOURCE_PIPELINE:
        raise ValueError(f"{station}: source pipeline is not settlement v11")
    if contract.get("target_source") != TARGET_SOURCE or bundle.get("target_source") != TARGET_SOURCE:
        raise ValueError(f"{station}: target source is not settlement-first")
    if contract.get("feature_version") != "v11" or bundle.get("feature_version") != "v11":
        raise ValueError(f"{station}: incompatible v11 feature version")
    features = set((manifest.get("features") or {}).get("all") or [])
    missing = HARD_REQUIRED_LIVE_FEATURES - features
    if missing:
        raise ValueError(f"{station}: missing live feature contract fields: {sorted(missing)}")


def main() -> int:
    args = parse_args()
    runtime_versions = pinned_package_versions(args.runtime_requirements)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for station in STATIONS:
        stem = f"{station}_{MODEL_VERSION}"
        source_bundle = args.source_dir / f"{stem}.joblib"
        source_manifest = args.source_dir / f"{stem}.json"
        manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
        bundle = joblib.load(source_bundle)
        if not isinstance(bundle, dict):
            raise ValueError(f"{station}: unsupported non-dictionary joblib bundle")
        validate_source(station, bundle, manifest)

        manifest["bundle_path"] = str(args.output_dir / source_bundle.name)
        manifest["validation_test_period"] = {
            "validation": prediction_period(
                args.artifact_dir / f"{station}_year_split_validation_predictions.csv"
            ),
            "test": prediction_period(args.artifact_dir / f"{station}_year_split_test_predictions.csv"),
        }
        manifest["package_runtime_compatibility"] = {
            "runtime_contract": "requirements-ml-runtime.txt",
            "package_versions": runtime_versions,
            "python": ">=3.11",
            "feature_pipeline": FEATURE_PIPELINE,
        }
        manifest["packaged_source"] = {
            "source_bundle_sha256": sha256(source_bundle),
            "source_manifest_sha256": sha256(source_manifest),
            "source_pipeline": SOURCE_PIPELINE,
            "target_source": TARGET_SOURCE,
        }

        target_bundle = args.output_dir / source_bundle.name
        target_manifest = args.output_dir / source_manifest.name
        shutil.copy2(source_bundle, target_bundle)
        target_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"packaged {station}: {target_bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
