from __future__ import annotations

import importlib.util
import hashlib
import json
import tarfile
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_tokyo_worker_archive.py"
SPEC = importlib.util.spec_from_file_location("build_tokyo_worker_archive", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_worker_archive_is_relative_manifested_and_free_of_retired_identifiers(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "tokyo-worker.tar"
    report = MODULE.build_archive(
        root,
        output,
        commit="a" * 40,
        require_clean=False,
    )
    assert report["fileCount"] == len(MODULE.RUNTIME_FILES) + 2
    with tarfile.open(output) as archive:
        names = archive.getnames()
        assert all(not Path(name).is_absolute() and ".." not in Path(name).parts for name in names)
        assert "src/polymarket_parse.py" not in names
        assert "src/settlement_actuals.py" not in names
        assert "src/wunderground_history.py" in names
        assert "scripts/run_asia_11am_pull.py" in names
        manifest = json.load(archive.extractfile("WORKER-MANIFEST.json"))
        assert manifest["schemaVersion"] == MODULE.WORKER_MANIFEST_SCHEMA_VERSION
        assert manifest["sourceCommit"] == "a" * 40
        assert manifest["runtimePayloads"] == list(MODULE.RUNTIME_PAYLOADS)
        assert set(manifest["files"]) == set(MODULE.RUNTIME_PAYLOADS)
        assert "WORKER-MANIFEST.json" not in manifest["files"]
        for name in MODULE.RUNTIME_PAYLOADS:
            raw = archive.extractfile(name).read()
            assert manifest["files"][name] == {
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size": len(raw),
            }
        assert MODULE.RETIRED_IDENTIFIER_FREE_FILES <= set(MODULE.RUNTIME_FILES)
        for name in MODULE.RETIRED_IDENTIFIER_FREE_FILES:
            raw = archive.extractfile(name).read().lower()
            assert b"hko" not in raw
            assert b"hong kong" not in raw
            assert b"hong_kong" not in raw


def test_archive_rejects_backslash_runtime_paths() -> None:
    try:
        MODULE._tar_info("src\\calibration\\asia_station_stacking.py", 1)
    except ValueError as error:
        assert str(error) == "unsafe_archive_path:src\\calibration\\asia_station_stacking.py"
    else:
        raise AssertionError("backslash archive path unexpectedly accepted")
