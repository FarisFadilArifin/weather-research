from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "publish_tokyo_live_feature_artifact.py"
SPEC = importlib.util.spec_from_file_location("publish_tokyo_live_feature_artifact", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def feature_frame() -> pd.DataFrame:
    row = {name: 90.0 for name in MODULE.REQUIRED_FIELDS}
    row.update(
        contract_date="2026-08-06",
        observed_as_of_time_local="2026-08-06T11:00:00+09:00",
        observed_source="iem_asos_global_metar",
        observed_data_source="iem_asos_global_metar_live",
        observed_weather_code_at_as_of="-RA",
        actual_high_f=99.0,
        optional_missing=float("nan"),
    )
    frame = pd.DataFrame([row])
    source = {
        "retrievedAtUtc": "2026-08-06T02:10:00Z",
        "sourceUrls": ["https://example.test/source"],
        "sourceChecksum": "c" * 64,
    }
    frame.attrs["alignment"] = {
        "alignmentStatus": "aligned",
        "stationId": "RJTT",
        "contractDate": "2026-08-06",
        "timezone": "Asia/Tokyo",
        "featureCutoffLocal": "2026-08-06T11:00:00+09:00",
        "featureCutoffUtc": "2026-08-06T02:00:00Z",
        "collectionNotBeforeUtc": "2026-08-06T02:10:00Z",
        "gfsCycleUtc": "2026-08-05T18:00:00Z",
        "gefsCycleUtc": "2026-08-05T18:00:00Z",
        "jmaLineage": "jma_msm_previous_day1",
        "jmaAvailabilityBasis": "open_meteo_previous_day1_variable",
        "metarObservedAtUtc": "2026-08-06T02:00:00Z",
        "metarSource": "iem_asos_global_metar",
        "timingMode": "asia_same_day_11am_live_safe",
        "sources": {name: dict(source) for name in ("gfs", "gefs", "jma_msm", "metar")},
    }
    return frame


def provider_contract() -> dict[str, object]:
    return {
        "contractId": "rjtt_iem_asos_metar_training_population_v1",
        "trainingProvider": "iem_asos_global_metar",
        "runtimeProvider": "iem_asos_global_metar",
        "population": "RJTT METAR observations at or before 11:00 Asia/Tokyo",
        "requiredRuntimeFields": [
            "observed_humidity_at_as_of",
            "observed_precip_recent_at_as_of",
            "observed_visibility_at_as_of",
            "observed_weather_code_at_as_of",
        ],
        "weatherCodePolicy": {
            "sourceField": "IEM wxcodes",
            "clearWeatherSentinel": "NONE",
        },
    }


def archive_identity(commit: str = "a" * 40) -> dict[str, str]:
    return {
        "cleanCommit": commit,
        "workerArchiveManifestSha256": "b" * 64,
        "workerArchiveArtifactType": "weather_research_tokyo_worker_v1",
    }


def test_payload_is_target_free_and_publishes_verified_sidecar(tmp_path, monkeypatch):
    payload = MODULE.build_payload(
        feature_frame(),
        date(2026, 8, 6),
        source_commit="a" * 40,
        generated_at=datetime(2026, 8, 6, 2, 10, tzinfo=UTC),
        provider_contract=provider_contract(),
        archive_identity=archive_identity(),
    )
    assert "actual_high_f" not in payload["featureInputs"]
    assert payload["featureInputs"]["optional_missing"] is None
    symlink_calls: list[tuple[Path, Path]] = []

    def fake_symlink(target, link, *, target_is_directory):
        symlink_calls.append((Path(target), Path(link)))
        link.mkdir()
        for child in (tmp_path / target).iterdir():
            shutil.copy2(child, link / child.name)

    # CI Windows workers can lack the privilege to create a directory symlink.
    # The production code still calls os.symlink and deliberately has no copy
    # fallback; this fixture only models the completed current target.
    monkeypatch.setattr(MODULE.os, "symlink", fake_symlink)
    artifact, sidecar = MODULE.publish(payload, tmp_path)
    raw = artifact.read_bytes()
    assert sidecar.read_text().split()[0] == hashlib.sha256(raw).hexdigest()
    assert json.loads(raw)["providers"] == ["gfs", "gefs", "jma_msm"]
    assert json.loads(raw)["alignmentStatus"] == "aligned"
    assert artifact.parent.name == "current"
    assert symlink_calls[0][0].parts[0] == "releases"
    assert json.loads(raw)["providerContract"]["runtimeProvider"] == "iem_asos_global_metar"
    assert not list((tmp_path / "releases").glob(".stage-*"))


def test_payload_rejects_missing_provider_and_stale_observation():
    frame = feature_frame()
    frame.loc[0, "gefs_high_f"] = float("nan")
    with pytest.raises(ValueError, match="missing_required_live_features:gefs_high_f"):
        MODULE.build_payload(
            frame,
            date(2026, 8, 6),
            source_commit="a" * 40,
            generated_at=datetime.now(UTC),
            provider_contract=provider_contract(),
            archive_identity=archive_identity(),
        )
    frame = feature_frame()
    frame.loc[0, "observed_as_of_time_local"] = "2026-08-06T10:39:00+09:00"
    with pytest.raises(ValueError, match="observation_outside"):
        MODULE.build_payload(
            frame,
            date(2026, 8, 6),
            source_commit="a" * 40,
            generated_at=datetime.now(UTC),
            provider_contract=provider_contract(),
            archive_identity=archive_identity(),
        )


def test_payload_rejects_non_iem_source_and_missing_weather_code():
    frame = feature_frame()
    frame.loc[0, "observed_source"] = "aviation_weather_center_metar"
    with pytest.raises(ValueError, match="source_contract_mismatch"):
        MODULE.build_payload(
            frame,
            date(2026, 8, 6),
            source_commit="a" * 40,
            generated_at=datetime.now(UTC),
            provider_contract=provider_contract(),
            archive_identity=archive_identity(),
        )

    frame = feature_frame()
    frame.attrs["alignment"]["metarSource"] = "aviation_weather_center_metar"
    with pytest.raises(ValueError, match="alignment_metarSource_mismatch"):
        MODULE.build_payload(
            frame,
            date(2026, 8, 6),
            source_commit="a" * 40,
            generated_at=datetime.now(UTC),
            provider_contract=provider_contract(),
            archive_identity=archive_identity(),
        )


def test_checked_in_contract_declares_source_truthful_clear_weather_policy():
    contract = MODULE.load_provider_contract(Path(__file__).resolve().parents[1])
    assert contract["trainingProvider"] == contract["runtimeProvider"] == "iem_asos_global_metar"
    assert contract["weatherCodePolicy"]["clearWeatherSentinel"] == "NONE"
    assert "never supplies precipitation" in contract["weatherCodePolicy"]["measurementRule"]
    frame = feature_frame()
    frame.loc[0, "observed_weather_code_at_as_of"] = ""
    with pytest.raises(ValueError, match="observed_weather_code_at_as_of"):
        MODULE.build_payload(
            frame,
            date(2026, 8, 6),
            source_commit="a" * 40,
            generated_at=datetime.now(UTC),
            provider_contract=provider_contract(),
            archive_identity=archive_identity(),
        )


def test_source_commit_can_be_declared_for_archive_deployments(tmp_path, monkeypatch):
    monkeypatch.setenv("WEATHER_RESEARCH_SOURCE_COMMIT", "b" * 40)
    assert MODULE.git_commit(tmp_path) == "b" * 40

    monkeypatch.delenv("WEATHER_RESEARCH_SOURCE_COMMIT")
    (tmp_path / ".source-commit").write_text("c" * 40 + "\n", encoding="utf-8")
    assert MODULE.git_commit(tmp_path) == "c" * 40


def write_worker_archive_root(tmp_path: Path, commit: str = "a" * 40) -> dict[str, object]:
    files: dict[str, bytes] = {}
    for index, name in enumerate(MODULE.EXPECTED_WORKER_RUNTIME_PAYLOADS):
        raw = (commit + "\n").encode() if name == ".source-commit" else f"runtime-{index}:{name}".encode()
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        files[name] = raw
    manifest = {
        "schemaVersion": MODULE.WORKER_MANIFEST_SCHEMA_VERSION,
        "artifactType": "weather_research_tokyo_worker_v1",
        "sourceCommit": commit,
        "entrypoint": "scripts.publish_tokyo_live_feature_artifact",
        "runtimePayloads": list(MODULE.EXPECTED_WORKER_RUNTIME_PAYLOADS),
        "files": {
            name: {"sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)}
            for name, raw in files.items()
        },
    }
    (tmp_path / "WORKER-MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_source_identity_requires_complete_matching_worker_archive_manifest(tmp_path):
    write_worker_archive_root(tmp_path)
    identity = MODULE.source_identity(
        tmp_path, source_commit="a" * 40, worker_archive_sha256="b" * 64
    )
    assert identity["cleanCommit"] == "a" * 40
    assert identity["workerArchiveSha256"] == "b" * 64
    assert len(identity["workerArchiveManifestSha256"]) == 64


def test_source_identity_requires_pinned_worker_archive_hash(tmp_path, monkeypatch):
    write_worker_archive_root(tmp_path)
    monkeypatch.delenv("WEATHER_RESEARCH_WORKER_ARCHIVE_SHA256", raising=False)
    with pytest.raises(ValueError, match="worker_archive_sha256_missing_or_invalid"):
        MODULE.source_identity(tmp_path, source_commit="a" * 40)


@pytest.mark.parametrize(
    ("name", "same_size", "error"),
    (
        (
            "src/calibration/asia_station_stacking.py",
            True,
            "worker_archive_file_checksum_mismatch:src/calibration/asia_station_stacking.py",
        ),
        (
            "src/current_observations.py",
            False,
            "worker_archive_file_size_mismatch:src/current_observations.py",
        ),
    ),
)
def test_source_identity_rejects_mutated_runtime_payloads(
    tmp_path: Path, name: str, same_size: bool, error: str
) -> None:
    write_worker_archive_root(tmp_path)
    with (tmp_path / name).open("r+b" if same_size else "ab") as handle:
        if same_size:
            handle.write(b"X")
        else:
            handle.write(b"mutated")
    with pytest.raises(ValueError, match=error):
        MODULE.source_identity(
            tmp_path, source_commit="a" * 40, worker_archive_sha256="b" * 64
        )


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        ("missing", "worker_archive_manifest_files_mismatch"),
        ("extra", "worker_archive_manifest_files_mismatch"),
        ("unsafe", "worker_archive_unsafe_runtime_path"),
    ),
)
def test_source_identity_rejects_missing_extra_and_unsafe_manifest_entries(
    tmp_path: Path, mutation: str, error: str
) -> None:
    manifest = write_worker_archive_root(tmp_path)
    files = manifest["files"]
    assert isinstance(files, dict)
    if mutation == "missing":
        del files["src/calibration/asia_station_stacking.py"]
    elif mutation == "extra":
        files["unexpected.py"] = {"sha256": "a" * 64, "size": 0}
    else:
        files["../outside.py"] = {"sha256": "a" * 64, "size": 0}
    (tmp_path / "WORKER-MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        MODULE.source_identity(
            tmp_path, source_commit="a" * 40, worker_archive_sha256="b" * 64
        )
