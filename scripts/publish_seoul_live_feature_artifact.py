from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
from datetime import UTC, date, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from src.asia_11am import CITY_PROFILES, run_live
from src.calibration.asia_station_stacking import build_asia_live_feature_row


ARTIFACT_TYPE = "weather_bot_seoul_feature_artifact_v1"
FEATURE_VERSION = "v20_asia_no_peak"
FEATURE_LINEAGE = "station_stacking_v20_asia_no_peak"
TIMING_MODE = "asia_same_day_11am_live_safe"
STATION_ID = "RKSI"
CITY_ID = "seoul"
TIMEZONE = "Asia/Seoul"
PROVIDERS = ("gfs", "gefs", "jma_msm")
TARGET_FIELDS = {
    "actual_high_f",
    "actual_high_c",
    "settlement_high_f",
    "settlement_high_c",
    "remaining_warmup_f",
    "remaining_warmup_from_observed_high_so_far_f",
}
# Raw fields required by the v20 Asia no-peak feature contract. The feature
# builder derives the remaining model inputs from these provider summaries.
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
WORKER_MANIFEST_SCHEMA_VERSION = 1
EXPECTED_WORKER_RUNTIME_PAYLOADS = (
    ".source-commit",
    "config/seoul_iem_asos_observation_contract.json",
    "requirements.txt",
    "scripts/publish_seoul_live_feature_artifact.py",
    "scripts/run_asia_11am_pull.py",
    "src/__init__.py",
    "src/asia_11am.py",
    "src/calibration/__init__.py",
    "src/calibration/asia_station_stacking.py",
    "src/calibration/data_quality.py",
    "src/calibration/dataset.py",
    "src/calibration/sdk_pipeline.py",
    "src/calibration/station_stacking.py",
    "src/calibration/time_rules.py",
    "src/current_observations.py",
    "src/direct_nwp_fetch.py",
    "src/wunderground_history.py",
)


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
    marker = project_root / ".source-commit"
    if declared:
        commit = declared
    elif marker.is_file():
        commit = marker.read_text(encoding="utf-8").strip()
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


def _safe_runtime_payload_path(name: object) -> bool:
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        return False
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts and str(path) == name


def load_provider_contract(project_root: Path) -> dict[str, Any]:
    path = project_root / "config" / "seoul_iem_asos_observation_contract.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    provider = "iem_asos_global_metar"
    if contract.get("trainingProvider") != provider or contract.get("runtimeProvider") != provider:
        raise ValueError("seoul_provider_contract_not_iem_asos_global_metar")
    required = set(contract.get("requiredRuntimeFields", []))
    expected = {
        "observed_humidity_at_as_of",
        "observed_precip_recent_at_as_of",
        "observed_visibility_at_as_of",
        "observed_weather_code_at_as_of",
    }
    if not expected.issubset(required):
        raise ValueError("seoul_provider_contract_missing_required_runtime_fields")
    policy = contract.get("weatherCodePolicy", {})
    if policy.get("sourceField") != "IEM wxcodes" or policy.get("clearWeatherSentinel") != "NONE":
        raise ValueError("seoul_provider_contract_weather_code_policy_invalid")
    return contract


def source_identity(
    project_root: Path,
    *,
    source_commit: str,
    worker_archive_sha256: str | None = None,
) -> dict[str, str]:
    archive_sha256 = (
        worker_archive_sha256
        if worker_archive_sha256 is not None
        else os.environ.get("WEATHER_RESEARCH_SEOUL_WORKER_ARCHIVE_SHA256", "")
    ).strip().lower()
    if len(archive_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in archive_sha256):
        raise ValueError("worker_archive_sha256_missing_or_invalid")
    manifest_path = project_root / "WORKER-MANIFEST.json"
    if not manifest_path.is_file():
        raise ValueError("missing_worker_archive_manifest")
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if not isinstance(manifest, dict):
        raise ValueError("invalid_worker_archive_manifest")
    if manifest.get("schemaVersion") != WORKER_MANIFEST_SCHEMA_VERSION:
        raise ValueError("invalid_worker_archive_manifest_schema")
    if manifest.get("artifactType") != "weather_research_seoul_worker_v1":
        raise ValueError("invalid_worker_archive_manifest_type")
    if manifest.get("sourceCommit") != source_commit:
        raise ValueError("worker_archive_commit_mismatch")
    if manifest.get("entrypoint") != "scripts.publish_seoul_live_feature_artifact":
        raise ValueError("worker_archive_entrypoint_mismatch")
    runtime_payloads = manifest.get("runtimePayloads")
    if (
        not isinstance(runtime_payloads, list)
        or any(not _safe_runtime_payload_path(name) for name in runtime_payloads)
        or len(runtime_payloads) != len(set(runtime_payloads))
        or runtime_payloads != list(EXPECTED_WORKER_RUNTIME_PAYLOADS)
    ):
        raise ValueError("worker_archive_manifest_runtime_payloads_mismatch")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("worker_archive_manifest_files_invalid")
    if any(not _safe_runtime_payload_path(name) for name in files):
        raise ValueError("worker_archive_unsafe_runtime_path")
    if set(files) != set(EXPECTED_WORKER_RUNTIME_PAYLOADS):
        raise ValueError("worker_archive_manifest_files_mismatch")
    for name in EXPECTED_WORKER_RUNTIME_PAYLOADS:
        entry = files[name]
        path = project_root / name
        if (
            not isinstance(entry, dict)
            or set(entry) != {"sha256", "size"}
            or not isinstance(entry.get("sha256"), str)
            or len(entry["sha256"]) != 64
            or any(ch not in "0123456789abcdef" for ch in entry["sha256"].lower())
            or not isinstance(entry.get("size"), int)
            or isinstance(entry.get("size"), bool)
            or entry["size"] < 0
        ):
            raise ValueError(f"worker_archive_manifest_file_entry_invalid:{name}")
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"worker_archive_missing_runtime_file:{name}")
        if path.stat().st_size != entry["size"]:
            raise ValueError(f"worker_archive_file_size_mismatch:{name}")
        if _sha256_file(path) != entry["sha256"]:
            raise ValueError(f"worker_archive_file_checksum_mismatch:{name}")
    if (project_root / ".source-commit").read_bytes() != (source_commit + "\n").encode():
        raise ValueError("worker_archive_source_commit_payload_mismatch")
    return {
        "cleanCommit": source_commit,
        "workerArchiveSha256": archive_sha256,
        "workerArchiveManifestSha256": hashlib.sha256(raw).hexdigest(),
        "workerArchiveArtifactType": str(manifest["artifactType"]),
    }


def validate_alignment(
    alignment: dict[str, Any], contract_date: date, *, runtime_provider: str
) -> None:
    expected = {
        "alignmentStatus": "aligned",
        "stationId": STATION_ID,
        "contractDate": contract_date.isoformat(),
        "timezone": TIMEZONE,
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
        if len(checksum) != 64 or any(ch not in "0123456789abcdef" for ch in checksum.lower()):
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
    selected = frame.loc[frame["contract_date"].astype(str).eq(contract_date.isoformat())]
    if len(selected) != 1:
        raise ValueError("expected_exactly_one_seoul_feature_row")
    inputs = {
        str(name): clean_value(value)
        for name, value in selected.iloc[0].items()
        if str(name) not in TARGET_FIELDS
    }
    missing = [
        name
        for name in REQUIRED_FIELDS
        if not isinstance(inputs.get(name), (int, float))
        or isinstance(inputs.get(name), bool)
        or not math.isfinite(float(inputs[name]))
    ]
    missing += [
        name
        for name in REQUIRED_TEXT_FIELDS
        if not isinstance(inputs.get(name), str) or not inputs[name].strip()
    ]
    if missing:
        raise ValueError("missing_required_live_features:" + ",".join(missing))
    if inputs.get("station_id") != STATION_ID:
        raise ValueError("feature_station_id_mismatch")
    if inputs.get("timezone") != TIMEZONE:
        raise ValueError("feature_timezone_mismatch")
    if inputs.get("feature_version") != FEATURE_VERSION:
        raise ValueError("feature_version_mismatch")
    if inputs.get("timing_mode") != TIMING_MODE:
        raise ValueError("timing_mode_mismatch")
    if inputs.get("observed_source") != runtime_provider:
        raise ValueError("live_observation_source_contract_mismatch")
    if inputs.get("observed_data_source") != f"{runtime_provider}_live":
        raise ValueError("live_observation_data_source_contract_mismatch")
    if archive_identity.get("cleanCommit") != source_commit:
        raise ValueError("source_identity_commit_mismatch")
    observed = datetime.fromisoformat(str(inputs.get("observed_as_of_time_local") or ""))
    if observed.tzinfo is None or observed.utcoffset() != ZoneInfo(TIMEZONE).utcoffset(observed):
        raise ValueError("observation_timezone_mismatch")
    if observed.date() != contract_date or not (
        (observed.hour == 10 and observed.minute >= 40)
        or (observed.hour == 11 and observed.minute == 0)
    ):
        raise ValueError("observation_outside_1040_1100_local_window")
    return {
        "artifactType": ARTIFACT_TYPE,
        "stationId": STATION_ID,
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
            field: provider_contract[field]
            for field in (
                "contractId",
                "trainingProvider",
                "runtimeProvider",
                "population",
                "requiredRuntimeFields",
                "weatherCodePolicy",
            )
        },
        "sourceIdentity": archive_identity,
        "featureInputs": inputs,
    }


def _fsync_file(path: Path) -> None:
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
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
    digest = hashlib.sha256(raw).hexdigest()
    if sidecar.read_text(encoding="utf-8").split() != [digest, filename]:
        raise ValueError("release_sidecar_checksum_mismatch")
    if json.loads(raw) != payload:
        raise ValueError("release_payload_validation_failed")


def publish(payload: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    releases = output_dir / "releases"
    releases.mkdir(exist_ok=True)
    filename = f"{STATION_ID}_{payload['contractDate']}.json"
    raw = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    digest = hashlib.sha256(raw).hexdigest()
    release_name = f"{STATION_ID}_{payload['contractDate']}_{digest[:16]}"
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


def collection_not_due(day: date, current: datetime) -> tuple[bool, datetime]:
    profile = CITY_PROFILES[CITY_ID]
    local_now = current.astimezone(ZoneInfo(profile.timezone))
    cutoff = datetime.combine(
        day,
        datetime.min.time().replace(
            hour=profile.as_of_hour_local,
            minute=profile.live_delay_minutes,
        ),
        tzinfo=ZoneInfo(profile.timezone),
    )
    return local_now.date() == day and local_now < cutoff, cutoff


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish the live-safe RKSI feature artifact")
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
    day = args.contract_date or current.astimezone(ZoneInfo(TIMEZONE)).date()
    if not args.skip_live_pull:
        not_due, cutoff = collection_not_due(day, current)
        if not_due:
            print(
                json.dumps(
                    {
                        "status": "not_due",
                        "contractDate": day.isoformat(),
                        "collectionNotBefore": cutoff.isoformat(),
                    }
                )
            )
            return 0
        result = run_live(
            args.data_root,
            [CITY_PROFILES[CITY_ID]],
            contract_date=day,
            workers=max(1, args.workers),
            now=current,
        )
        if result.get("status") != "complete":
            raise SystemExit("Seoul live acquisition is incomplete")
    frame = build_asia_live_feature_row(
        args.data_root,
        CITY_ID,
        day,
        generated_at=current,
        feature_version=FEATURE_VERSION,
        providers=PROVIDERS,
    )
    commit = git_commit(project_root)
    payload = build_payload(
        frame,
        day,
        source_commit=commit,
        generated_at=current,
        provider_contract=load_provider_contract(project_root),
        archive_identity=source_identity(project_root, source_commit=commit),
    )
    artifact, sidecar = publish(payload, args.output_dir)
    print(json.dumps({"status": "ok", "artifact": str(artifact), "sidecar": str(sidecar)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
