from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from src.asia_11am import CITY_PROFILES, run_live
from src.calibration.asia_station_stacking import build_asia_live_feature_row


ARTIFACT_TYPE = "weather_bot_tokyo_feature_artifact_v1"
FEATURE_VERSION = "v20_asia_no_peak"
FEATURE_LINEAGE = "station_stacking_v20_asia_no_peak"
TIMING_MODE = "asia_same_day_11am_live_safe"
PROVIDERS = ("gfs", "gefs", "jma_msm")
TARGET_FIELDS = {
    "actual_high_f",
    "actual_high_c",
    "settlement_high_f",
    "settlement_high_c",
    "remaining_warmup_f",
    "remaining_warmup_from_observed_high_so_far_f",
}
REQUIRED_FIELDS = (
    "gfs_high_f",
    "gefs_high_f",
    "jma_msm_high_f",
    "gfs_forecast_temp_at_as_of_f",
    "gefs_forecast_temp_at_as_of_f",
    "jma_msm_forecast_temp_at_as_of_f",
    "observed_temp_at_as_of_f",
    "observed_high_temp_through_as_of_f",
    "observed_as_of_age_minutes",
    "observed_humidity_at_as_of",
    "observed_visibility_at_as_of",
    "observed_precip_recent_at_as_of",
)
REQUIRED_TEXT_FIELDS = ("observed_weather_code_at_as_of",)


def clean_value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    return value


def git_commit(project_root: Path) -> str:
    declared = os.environ.get("WEATHER_RESEARCH_SOURCE_COMMIT", "").strip()
    if declared:
        commit = declared
    elif (project_root / ".source-commit").is_file():
        commit = (project_root / ".source-commit").read_text(encoding="utf-8").strip()
    else:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    if len(commit) != 40 or any(ch not in "0123456789abcdef" for ch in commit.lower()):
        raise ValueError("invalid_source_commit")
    return commit


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_provider_contract(project_root: Path) -> dict[str, Any]:
    path = project_root / "config" / "tokyo_iem_asos_observation_contract.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    provider = "iem_asos_global_metar"
    if (
        contract.get("trainingProvider") != provider
        or contract.get("runtimeProvider") != provider
    ):
        raise ValueError("tokyo_provider_contract_not_iem_asos_global_metar")
    required = set(contract.get("requiredRuntimeFields", []))
    expected = {
        "observed_humidity_at_as_of",
        "observed_precip_recent_at_as_of",
        "observed_visibility_at_as_of",
        "observed_weather_code_at_as_of",
    }
    if not expected.issubset(required):
        raise ValueError("tokyo_provider_contract_missing_required_runtime_fields")
    weather_policy = contract.get("weatherCodePolicy", {})
    if (
        weather_policy.get("sourceField") != "IEM wxcodes"
        or weather_policy.get("clearWeatherSentinel") != "NONE"
    ):
        raise ValueError("tokyo_provider_contract_weather_code_policy_invalid")
    return contract


def source_identity(project_root: Path, *, source_commit: str) -> dict[str, str]:
    """Read the immutable worker manifest shipped with an archive release.

    The manifest is deliberately inside the archive and has a checksum in the
    payload; it names the clean commit without attempting to hash itself.
    """
    manifest_path = project_root / "WORKER-MANIFEST.json"
    if not manifest_path.is_file():
        raise ValueError("missing_worker_archive_manifest")
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("artifactType") != "weather_research_tokyo_worker_v1":
        raise ValueError("invalid_worker_archive_manifest_type")
    if manifest.get("sourceCommit") != source_commit:
        raise ValueError("worker_archive_commit_mismatch")
    files = manifest.get("files", {})
    for name in (
        "scripts/publish_tokyo_live_feature_artifact.py",
        "src/asia_11am.py",
        "config/tokyo_iem_asos_observation_contract.json",
    ):
        entry = files.get(name)
        path = project_root / name
        if not isinstance(entry, dict) or not path.is_file():
            raise ValueError(f"worker_archive_missing_runtime_file:{name}")
        if entry.get("sha256") != _sha256_file(path):
            raise ValueError(f"worker_archive_file_checksum_mismatch:{name}")
    return {
        "cleanCommit": source_commit,
        "workerArchiveManifestSha256": hashlib.sha256(raw).hexdigest(),
        "workerArchiveArtifactType": str(manifest["artifactType"]),
    }


def validate_alignment(
    alignment: dict[str, Any], contract_date: date, *, runtime_provider: str
) -> None:
    """Require the IEM-backed, point-in-time alignment schema before publish."""
    expected = {
        "alignmentStatus": "aligned",
        "stationId": "RJTT",
        "contractDate": contract_date.isoformat(),
        "timezone": "Asia/Tokyo",
        "jmaLineage": "jma_msm_previous_day1",
        "jmaAvailabilityBasis": "open_meteo_previous_day1_variable",
        "metarSource": runtime_provider,
        "timingMode": TIMING_MODE,
    }
    for field, value in expected.items():
        if alignment.get(field) != value:
            raise ValueError(f"alignment_{field}_mismatch")
    for field in (
        "featureCutoffLocal",
        "featureCutoffUtc",
        "collectionNotBeforeUtc",
        "gfsCycleUtc",
        "gefsCycleUtc",
        "metarObservedAtUtc",
    ):
        if not isinstance(alignment.get(field), str) or not alignment[field].strip():
            raise ValueError(f"alignment_{field}_missing")
    sources = alignment.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("alignment_sources_missing")
    for name in ("gfs", "gefs", "jma_msm", "metar"):
        source = sources.get(name)
        if not isinstance(source, dict):
            raise ValueError(f"alignment_{name}_source_missing")
        urls = source.get("sourceUrls")
        checksum = str(source.get("sourceChecksum") or "")
        if not isinstance(urls, list) or not urls or not all(str(url).strip() for url in urls):
            raise ValueError(f"alignment_{name}_source_url_missing")
        if len(checksum) != 64 or any(
            character not in "0123456789abcdef" for character in checksum.lower()
        ):
            raise ValueError(f"alignment_{name}_source_checksum_missing")


def build_payload(
    frame: pd.DataFrame,
    contract_date: date,
    *,
    source_commit: str,
    generated_at: datetime,
    provider_contract: dict[str, Any],
    archive_identity: dict[str, str],
) -> dict[str, Any]:
    alignment = frame.attrs.get("alignment")
    if not isinstance(alignment, dict):
        raise ValueError("alignment_status_not_aligned")
    runtime_provider = str(provider_contract["runtimeProvider"])
    validate_alignment(alignment, contract_date, runtime_provider=runtime_provider)
    selected = frame.loc[
        frame["contract_date"].astype(str).eq(contract_date.isoformat())
    ]
    if len(selected) != 1:
        raise ValueError("expected_exactly_one_tokyo_feature_row")
    row = selected.iloc[0]
    inputs = {
        str(name): clean_value(value)
        for name, value in row.items()
        if str(name) not in TARGET_FIELDS
    }
    missing = [
        name
        for name in REQUIRED_FIELDS
        if not isinstance(inputs.get(name), (int, float))
        or isinstance(inputs.get(name), bool)
        or not math.isfinite(float(inputs[name]))
    ]
    if missing:
        raise ValueError("missing_required_live_features:" + ",".join(missing))
    missing_text = [
        name
        for name in REQUIRED_TEXT_FIELDS
        if not isinstance(inputs.get(name), str) or not inputs[name].strip()
    ]
    if missing_text:
        raise ValueError("missing_required_live_features:" + ",".join(missing_text))
    if inputs.get("observed_source") != runtime_provider:
        raise ValueError("live_observation_source_contract_mismatch")
    if inputs.get("observed_data_source") != f"{runtime_provider}_live":
        raise ValueError("live_observation_data_source_contract_mismatch")
    if archive_identity.get("cleanCommit") != source_commit:
        raise ValueError("source_identity_commit_mismatch")
    observed = datetime.fromisoformat(str(inputs.get("observed_as_of_time_local") or ""))
    if observed.date() != contract_date or not (
        (observed.hour == 10 and observed.minute >= 40)
        or (observed.hour == 11 and observed.minute == 0)
    ):
        raise ValueError("observation_outside_1040_1100_local_window")
    return {
        "artifactType": ARTIFACT_TYPE,
        "stationId": "RJTT",
        "contractDate": contract_date.isoformat(),
        "featureVersion": FEATURE_VERSION,
        "featureLineage": FEATURE_LINEAGE,
        "timingMode": TIMING_MODE,
        "providers": list(PROVIDERS),
        "predictionTemperatureUnit": "celsius",
        "pointInTimeSafe": True,
        "alignmentStatus": "aligned",
        "alignment": alignment,
        "observationSource": inputs.get("observed_source"),
        "generatedAtUtc": generated_at.astimezone(UTC).isoformat(),
        "acquisitionSourceCommit": source_commit,
        "providerContract": {
            "contractId": provider_contract["contractId"],
            "trainingProvider": provider_contract["trainingProvider"],
            "runtimeProvider": provider_contract["runtimeProvider"],
            "population": provider_contract["population"],
            "requiredRuntimeFields": provider_contract["requiredRuntimeFields"],
            "weatherCodePolicy": provider_contract["weatherCodePolicy"],
        },
        "sourceIdentity": archive_identity,
        "featureInputs": inputs,
    }


def _fsync_file(path: Path) -> None:
    # Windows rejects fsync on a read-only CRT descriptor.
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    # Directory fsync is unavailable on Windows; file fsync still protects the
    # staged payload before the atomic rename there.
    try:
        descriptor = os.open(path, getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_release(release_dir: Path, filename: str, payload: dict[str, Any]) -> None:
    artifact = release_dir / filename
    sidecar = release_dir / f"{filename}.sha256"
    if not artifact.is_file() or not sidecar.is_file():
        raise ValueError("release_artifact_or_sidecar_missing")
    raw = artifact.read_bytes()
    expected = hashlib.sha256(raw).hexdigest()
    if sidecar.read_text(encoding="utf-8").split() != [expected, filename]:
        raise ValueError("release_sidecar_checksum_mismatch")
    if json.loads(raw) != payload:
        raise ValueError("release_payload_validation_failed")


def publish(payload: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    """Publish a complete release, then atomically switch only ``current``.

    Consumers resolve both public files through one symlink.  They can never
    observe a new JSON document paired with an older sidecar checksum.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    releases = output_dir / "releases"
    releases.mkdir(exist_ok=True)
    filename = f"RJTT_{payload['contractDate']}.json"
    raw = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    digest = hashlib.sha256(raw).hexdigest()
    release_name = f"RJTT_{payload['contractDate']}_{digest[:16]}"
    destination = releases / release_name
    if destination.exists():
        _validate_release(destination, filename, payload)
    else:
        with tempfile.TemporaryDirectory(dir=releases, prefix=".stage-") as temporary:
            root = Path(temporary)
            artifact = root / filename
            sidecar = root / f"{filename}.sha256"
            artifact.write_bytes(raw)
            sidecar.write_text(f"{digest}  {filename}\n", encoding="utf-8")
            _fsync_file(artifact)
            _fsync_file(sidecar)
            _validate_release(root, filename, payload)
            _fsync_directory(root)
            os.replace(root, destination)
            _fsync_directory(releases)
            _validate_release(destination, filename, payload)
    current = output_dir / "current"
    temporary_link = output_dir / f".current-{os.getpid()}-{digest[:12]}"
    try:
        os.symlink(Path("releases") / release_name, temporary_link, target_is_directory=True)
        os.replace(temporary_link, current)
    finally:
        if temporary_link.is_symlink():
            temporary_link.unlink()
    _fsync_directory(output_dir)
    return current / filename, current / f"{filename}.sha256"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish the live-safe RJTT feature artifact")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contract-date", type=date.fromisoformat)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--skip-live-pull", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    current = datetime.now(UTC)
    day = args.contract_date or current.astimezone(ZoneInfo("Asia/Tokyo")).date()
    if not args.skip_live_pull:
        result = run_live(
            args.data_root,
            [CITY_PROFILES["tokyo"]],
            contract_date=day,
            workers=max(1, args.workers),
            now=current,
        )
        if result.get("status") != "complete":
            raise SystemExit("Tokyo live acquisition is incomplete")
    frame = build_asia_live_feature_row(
        args.data_root,
        "tokyo",
        day,
        generated_at=current,
        feature_version=FEATURE_VERSION,
        providers=PROVIDERS,
    )
    payload = build_payload(
        frame,
        day,
        source_commit=git_commit(project_root),
        generated_at=current,
        provider_contract=load_provider_contract(project_root),
        archive_identity=source_identity(
            project_root, source_commit=git_commit(project_root)
        ),
    )
    artifact, sidecar = publish(payload, args.output_dir)
    print(json.dumps({"status": "ok", "artifact": str(artifact), "sidecar": str(sidecar)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
