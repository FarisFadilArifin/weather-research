from __future__ import annotations

import hashlib
import importlib.util
import json
import tarfile
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_seoul_worker_archive.py"
SPEC = importlib.util.spec_from_file_location("build_seoul_worker_archive", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_worker_archive_is_deterministic_relative_and_seoul_scoped(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    first = tmp_path / "seoul-worker-a.tar"
    second = tmp_path / "seoul-worker-b.tar"
    for output in (first, second):
        MODULE.build_archive(root, output, commit="a" * 40, require_clean=False)
    assert hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(
        second.read_bytes()
    ).digest()
    with tarfile.open(first) as archive:
        names = archive.getnames()
        assert all(not Path(name).is_absolute() and ".." not in Path(name).parts for name in names)
        assert "scripts/publish_seoul_live_feature_artifact.py" in names
        assert "scripts/publish_tokyo_live_feature_artifact.py" not in names
        assert "config/seoul_iem_asos_observation_contract.json" in names
        manifest = json.load(archive.extractfile("WORKER-MANIFEST.json"))
        assert manifest["artifactType"] == "weather_research_seoul_worker_v1"
        assert manifest["sourceCommit"] == "a" * 40
        assert manifest["entrypoint"] == "scripts.publish_seoul_live_feature_artifact"
        assert manifest["runtimePayloads"] == list(MODULE.RUNTIME_PAYLOADS)
        assert set(manifest["files"]) == set(MODULE.RUNTIME_PAYLOADS)
        for name in MODULE.RUNTIME_PAYLOADS:
            raw = archive.extractfile(name).read()
            assert manifest["files"][name] == {
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size": len(raw),
            }


def test_archive_rejects_backslash_runtime_paths() -> None:
    try:
        MODULE._tar_info("src\\asia_11am.py", 1)
    except ValueError as error:
        assert str(error) == "unsafe_archive_path:src\\asia_11am.py"
    else:
        raise AssertionError("backslash archive path unexpectedly accepted")
