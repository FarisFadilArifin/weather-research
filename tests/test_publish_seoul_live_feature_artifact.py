from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "publish_seoul_live_feature_artifact.py"
SPEC = importlib.util.spec_from_file_location("publish_seoul_live_feature_artifact", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def feature_frame() -> pd.DataFrame:
    row = {name: 90.0 for name in MODULE.REQUIRED_FIELDS}
    row.update(
        contract_date="2026-08-06",
        station_id="RKSI",
        timezone="Asia/Seoul",
        feature_version="v20_asia_no_peak",
        timing_mode="asia_same_day_11am_live_safe",
        observed_as_of_time_local="2026-08-06T11:00:00+09:00",
        observed_source="iem_asos_global_metar",
        observed_data_source="iem_asos_global_metar_live",
        observed_weather_code_at_as_of="NONE",
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
        "stationId": "RKSI",
        "contractDate": "2026-08-06",
        "timezone": "Asia/Seoul",
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
        "contractId": "rksi_iem_asos_metar_training_population_v1",
        "trainingProvider": "iem_asos_global_metar",
        "runtimeProvider": "iem_asos_global_metar",
        "population": "RKSI METAR observations at or before 11:00 Asia/Seoul",
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
        "workerArchiveArtifactType": "weather_research_seoul_worker_v1",
    }


def test_collection_window_uses_asia_seoul() -> None:
    not_due, cutoff = MODULE.collection_not_due(
        date(2026, 8, 10), datetime(2026, 8, 10, 1, 50, tzinfo=UTC)
    )
    assert not_due is True
    assert cutoff.isoformat() == "2026-08-10T11:10:00+09:00"
    assert MODULE.collection_not_due(
        date(2026, 8, 10), datetime(2026, 8, 10, 2, 10, tzinfo=UTC)
    )[0] is False


def test_payload_is_rksi_celsius_v20_target_free_and_atomic(tmp_path, monkeypatch) -> None:
    payload = MODULE.build_payload(
        feature_frame(),
        date(2026, 8, 6),
        source_commit="a" * 40,
        generated_at=datetime(2026, 8, 6, 2, 10, tzinfo=UTC),
        provider_contract=provider_contract(),
        archive_identity=archive_identity(),
    )
    assert payload["stationId"] == "RKSI"
    assert payload["predictionTemperatureUnit"] == "celsius"
    assert payload["featureVersion"] == "v20_asia_no_peak"
    assert "actual_high_f" not in payload["featureInputs"]
    assert payload["featureInputs"]["optional_missing"] is None

    def fake_symlink(target, link, *, target_is_directory):
        link = Path(link)
        link.mkdir()
        for child in (tmp_path / target).iterdir():
            shutil.copy2(child, link / child.name)

    monkeypatch.setattr(MODULE.os, "symlink", fake_symlink)
    artifact, sidecar = MODULE.publish(payload, tmp_path)
    raw = artifact.read_bytes()
    assert artifact.name == "RKSI_2026-08-06.json"
    assert sidecar.read_text().split()[0] == hashlib.sha256(raw).hexdigest()
    assert json.loads(raw)["alignment"]["timezone"] == "Asia/Seoul"
    assert not list((tmp_path / "releases").glob(".stage-*"))


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        ("station", "alignment_stationId_mismatch"),
        ("timezone", "alignment_timezone_mismatch"),
        ("version", "feature_version_mismatch"),
        ("provider", "live_observation_source_contract_mismatch"),
        ("missing", "missing_required_live_features:gefs_high_f"),
    ),
)
def test_payload_fails_closed_on_rksi_v20_contract_drift(mutation: str, error: str) -> None:
    frame = feature_frame()
    if mutation == "station":
        frame.attrs["alignment"]["stationId"] = "RJTT"
    elif mutation == "timezone":
        frame.attrs["alignment"]["timezone"] = "Asia/Tokyo"
    elif mutation == "version":
        frame.loc[0, "feature_version"] = "v19"
    elif mutation == "provider":
        frame.loc[0, "observed_source"] = "aviation_weather_center_metar"
    else:
        frame.loc[0, "gefs_high_f"] = float("nan")
    with pytest.raises(ValueError, match=error):
        MODULE.build_payload(
            frame,
            date(2026, 8, 6),
            source_commit="a" * 40,
            generated_at=datetime.now(UTC),
            provider_contract=provider_contract(),
            archive_identity=archive_identity(),
        )


def write_worker_archive_root(tmp_path: Path, commit: str = "a" * 40) -> None:
    files: dict[str, bytes] = {}
    for index, name in enumerate(MODULE.EXPECTED_WORKER_RUNTIME_PAYLOADS):
        raw = (commit + "\n").encode() if name == ".source-commit" else f"runtime-{index}".encode()
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        files[name] = raw
    manifest = {
        "schemaVersion": MODULE.WORKER_MANIFEST_SCHEMA_VERSION,
        "artifactType": "weather_research_seoul_worker_v1",
        "sourceCommit": commit,
        "entrypoint": "scripts.publish_seoul_live_feature_artifact",
        "runtimePayloads": list(MODULE.EXPECTED_WORKER_RUNTIME_PAYLOADS),
        "files": {
            name: {"sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)}
            for name, raw in files.items()
        },
    }
    (tmp_path / "WORKER-MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_source_identity_is_seoul_scoped_and_hashes_every_runtime_file(tmp_path) -> None:
    write_worker_archive_root(tmp_path)
    identity = MODULE.source_identity(
        tmp_path, source_commit="a" * 40, worker_archive_sha256="b" * 64
    )
    assert identity["workerArchiveArtifactType"] == "weather_research_seoul_worker_v1"
    (tmp_path / "src" / "asia_11am.py").write_bytes(b"mutated")
    with pytest.raises(ValueError, match="worker_archive_file_size_mismatch:src/asia_11am.py"):
        MODULE.source_identity(
            tmp_path, source_commit="a" * 40, worker_archive_sha256="b" * 64
        )


def test_checked_in_provider_contract_is_rksi_iem_population() -> None:
    contract = MODULE.load_provider_contract(Path(__file__).resolve().parents[1])
    assert contract["contractId"] == "rksi_iem_asos_metar_training_population_v1"
    assert contract["trainingProvider"] == contract["runtimeProvider"] == "iem_asos_global_metar"
    assert contract["weatherCodePolicy"]["clearWeatherSentinel"] == "NONE"
