from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path, PurePosixPath


RUNTIME_FILES = (
    "requirements.txt",
    "scripts/publish_tokyo_live_feature_artifact.py",
    "src/__init__.py",
    "src/asia_11am.py",
    "src/current_observations.py",
    "src/direct_nwp_fetch.py",
    "src/calibration/__init__.py",
    "src/calibration/asia_station_stacking.py",
    "src/calibration/data_quality.py",
    "src/calibration/dataset.py",
    "src/calibration/sdk_pipeline.py",
    "src/calibration/station_stacking.py",
    "src/calibration/time_rules.py",
)
BANNED_IDENTIFIERS = (b"hko", b"hong kong", b"hong_kong")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def source_commit(project_root: Path) -> str:
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


def require_clean_tracked_tree(project_root: Path) -> None:
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise ValueError("tracked_worktree_is_dirty")


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe_archive_path:{name}")
    info = tarfile.TarInfo(str(path))
    info.size = size
    info.mode = 0o644
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    return info


def build_archive(
    project_root: Path,
    output: Path,
    *,
    commit: str,
    require_clean: bool = True,
) -> dict[str, object]:
    project_root = project_root.resolve()
    if require_clean:
        require_clean_tracked_tree(project_root)
    if len(commit) != 40 or any(ch not in "0123456789abcdef" for ch in commit.lower()):
        raise ValueError("invalid_source_commit")

    payloads: dict[str, bytes] = {}
    for relative in RUNTIME_FILES:
        path = project_root / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        raw = path.read_bytes()
        lowered = raw.lower()
        if any(identifier in lowered for identifier in BANNED_IDENTIFIERS):
            raise ValueError(f"retired_station_identifier:{relative}")
        payloads[relative] = raw
    payloads[".source-commit"] = (commit + "\n").encode()

    manifest = {
        "artifactType": "weather_research_tokyo_worker_v1",
        "sourceCommit": commit,
        "entrypoint": "scripts.publish_tokyo_live_feature_artifact",
        "files": {
            name: {"sha256": sha256_bytes(raw), "size": len(raw)}
            for name, raw in sorted(payloads.items())
        },
    }
    payloads["WORKER-MANIFEST.json"] = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()

    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w", format=tarfile.PAX_FORMAT) as archive:
        for name, raw in sorted(payloads.items()):
            archive.addfile(_tar_info(name, len(raw)), io.BytesIO(raw))
    return {
        "status": "ok",
        "archive": str(output),
        "archiveSha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "sourceCommit": commit,
        "fileCount": len(payloads),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the immutable Tokyo worker archive")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = build_archive(root, args.output, commit=source_commit(root))
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
