"""Immutable, fail-closed provenance records for station-training releases."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
_RELEASE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{2,127}$")
_REQUIRED_ARTIFACTS = frozenset(
    {"dataset", "features", "notebook", "export", "model", "model_manifest"}
)


class ReleaseProvenanceError(RuntimeError):
    """Raised when release evidence is incomplete, changed, or unsafe to promote."""


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(project_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ReleaseProvenanceError(
            f"could not inspect git state: {result.stderr.strip() or 'git failed'}"
        )
    return result.stdout


def source_revision(project_root: Path | str) -> dict[str, Any]:
    """Return the current source identity, rejecting an unclean worktree."""
    root = Path(project_root).resolve()
    commit = _git_output(root, "rev-parse", "HEAD").strip()
    dirty = _git_output(root, "status", "--porcelain=v1", "-z")
    if dirty:
        raise ReleaseProvenanceError(
            "release provenance requires a clean worktree; refusing to record mutable inputs"
        )
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ReleaseProvenanceError("git did not return a full source commit hash")
    return {"commit": commit, "worktree_clean": True}


def _relative_file(project_root: Path, path: Path | str, *, label: str) -> tuple[str, str]:
    candidate = Path(path).resolve()
    try:
        relative = candidate.relative_to(project_root)
    except ValueError as error:
        raise ReleaseProvenanceError(f"{label} must stay inside the project root") from error
    if not candidate.is_file():
        raise ReleaseProvenanceError(f"{label} is missing or is not a file: {relative}")
    return relative.as_posix(), sha256_file(candidate)


def _normalize_provider_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(contract, Mapping):
        raise ReleaseProvenanceError("provider_contract must be a mapping")
    providers = contract.get("providers")
    if not isinstance(providers, Sequence) or isinstance(providers, (str, bytes)):
        raise ReleaseProvenanceError("provider_contract.providers must be a non-empty list")
    normalized = [str(provider).strip() for provider in providers]
    if not normalized or any(not provider for provider in normalized):
        raise ReleaseProvenanceError("provider_contract.providers must not be empty")
    if len(set(normalized)) != len(normalized):
        raise ReleaseProvenanceError("provider_contract.providers must not contain duplicates")
    result = dict(contract)
    result["providers"] = normalized
    return result


def _normalize_history(history: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(history, Mapping):
        raise ReleaseProvenanceError("required_history must be a mapping")
    result = dict(history)
    for key in ("training_years", "forward_validation_years"):
        values = result.get(key)
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ReleaseProvenanceError(f"required_history.{key} must be a non-empty list")
        try:
            normalized = [int(value) for value in values]
        except (TypeError, ValueError) as error:
            raise ReleaseProvenanceError(
                f"required_history.{key} must contain integer years"
            ) from error
        if not normalized or normalized != sorted(set(normalized)):
            raise ReleaseProvenanceError(
                f"required_history.{key} must be sorted, unique, and non-empty"
            )
        result[key] = normalized
    try:
        holdout_year = int(result.get("holdout_year"))
    except (TypeError, ValueError) as error:
        raise ReleaseProvenanceError(
            "required_history.holdout_year must be an integer"
        ) from error
    if holdout_year <= max(result["training_years"]):
        raise ReleaseProvenanceError("required_history.holdout_year must follow training years")
    if not bool(result.get("selection_excludes_holdout")):
        raise ReleaseProvenanceError("required_history must exclude the holdout from selection")
    result["holdout_year"] = holdout_year
    result["selection_excludes_holdout"] = True
    return result


def _normalize_missingness(missingness: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(missingness, Mapping):
        raise ReleaseProvenanceError("missingness must be a mapping")
    try:
        maximum = float(missingness.get("max_feature_missing_fraction"))
    except (TypeError, ValueError) as error:
        raise ReleaseProvenanceError(
            "missingness.max_feature_missing_fraction must be a number"
        ) from error
    if not 0.0 <= maximum <= 1.0:
        raise ReleaseProvenanceError("missingness.max_feature_missing_fraction must be within [0, 1]")
    observed = missingness.get("observed_fraction_by_feature")
    if not isinstance(observed, Mapping) or not observed:
        raise ReleaseProvenanceError("missingness.observed_fraction_by_feature must be non-empty")
    try:
        normalized_observed = {
            str(name): float(value) for name, value in observed.items()
        }
    except (TypeError, ValueError) as error:
        raise ReleaseProvenanceError(
            "missingness observed fractions must be numeric"
        ) from error
    if any(not name or value < 0.0 or value > 1.0 for name, value in normalized_observed.items()):
        raise ReleaseProvenanceError("missingness observed fractions must be named values within [0, 1]")
    failing = sorted(
        name for name, value in normalized_observed.items() if 1.0 - value > maximum
    )
    if failing:
        raise ReleaseProvenanceError(
            "missingness exceeds the release gate for: " + ", ".join(failing)
        )
    return {
        "max_feature_missing_fraction": maximum,
        "observed_fraction_by_feature": normalized_observed,
    }


def build_release_manifest(
    project_root: Path | str,
    *,
    release_id: str,
    station_id: str,
    model_version: str,
    artifact_paths: Mapping[str, Path | str],
    runtime_paths: Mapping[str, Path | str],
    provider_contract: Mapping[str, Any],
    required_history: Mapping[str, Any],
    missingness: Mapping[str, Any],
) -> dict[str, Any]:
    """Create a release record after validating every promotion input.

    This function intentionally does not write model bundles or mutate any existing
    artifact.  Calling it from a dirty worktree fails before a release record can be
    created, preventing a mutable research state from being promoted by accident.
    """
    root = Path(project_root).resolve()
    if not _RELEASE_ID.fullmatch(release_id):
        raise ReleaseProvenanceError("release_id must be a stable lowercase identifier")
    if not str(station_id).strip() or not str(model_version).strip():
        raise ReleaseProvenanceError("station_id and model_version must be non-empty")
    if set(artifact_paths) != _REQUIRED_ARTIFACTS:
        missing = sorted(_REQUIRED_ARTIFACTS - set(artifact_paths))
        unexpected = sorted(set(artifact_paths) - _REQUIRED_ARTIFACTS)
        raise ReleaseProvenanceError(
            "artifact_paths must contain exactly "
            f"{sorted(_REQUIRED_ARTIFACTS)}; missing={missing}, unexpected={unexpected}"
        )
    if not runtime_paths:
        raise ReleaseProvenanceError("at least one hashed runtime input is required")

    source = source_revision(root)
    artifacts = {
        name: {"path": path, "sha256": digest}
        for name, value in sorted(artifact_paths.items())
        for path, digest in [_relative_file(root, value, label=f"artifact {name}")]
    }
    runtime = {
        name: {"path": path, "sha256": digest}
        for name, value in sorted(runtime_paths.items())
        for path, digest in [_relative_file(root, value, label=f"runtime input {name}")]
    }
    if any(not str(name).strip() for name in runtime):
        raise ReleaseProvenanceError("runtime input names must be non-empty")
    return {
        "schema_version": SCHEMA_VERSION,
        "release_id": release_id,
        "station_id": str(station_id).strip(),
        "model_version": str(model_version).strip(),
        "source": source,
        "artifacts": artifacts,
        "runtime": runtime,
        "provider_contract": _normalize_provider_contract(provider_contract),
        "required_history": _normalize_history(required_history),
        "missingness": _normalize_missingness(missingness),
    }


def write_release_manifest(path: Path | str, manifest: Mapping[str, Any]) -> Path:
    """Write a new immutable record; existing records are never overwritten."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(dict(manifest), handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as error:
        raise ReleaseProvenanceError(
            f"release manifest already exists and is immutable: {output}"
        ) from error
    return output


def _verify_hashed_paths(
    project_root: Path, entries: Mapping[str, Any], *, label: str
) -> None:
    if not isinstance(entries, Mapping) or not entries:
        raise ReleaseProvenanceError(f"{label} must be a non-empty mapping")
    for name, entry in entries.items():
        if not isinstance(entry, Mapping):
            raise ReleaseProvenanceError(f"{label}.{name} must be a mapping")
        relative = entry.get("path")
        expected_hash = entry.get("sha256")
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise ReleaseProvenanceError(f"{label}.{name}.path must be a relative path")
        if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            raise ReleaseProvenanceError(f"{label}.{name}.sha256 must be a SHA-256 digest")
        candidate = (project_root / relative).resolve()
        try:
            candidate.relative_to(project_root)
        except ValueError as error:
            raise ReleaseProvenanceError(f"{label}.{name}.path escapes the project root") from error
        if not candidate.is_file():
            raise ReleaseProvenanceError(f"{label}.{name} is missing: {relative}")
        if sha256_file(candidate) != expected_hash:
            raise ReleaseProvenanceError(f"{label}.{name} hash mismatch: {relative}")


def verify_release_manifest(
    project_root: Path | str,
    manifest: Mapping[str, Any],
    *,
    expected_provider_contract: Mapping[str, Any] | None = None,
) -> None:
    """Verify a release record against the current clean checkout and local files."""
    root = Path(project_root).resolve()
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ReleaseProvenanceError("unsupported release provenance schema")
    if not _RELEASE_ID.fullmatch(str(manifest.get("release_id", ""))):
        raise ReleaseProvenanceError("release manifest has an invalid release_id")
    current_source = source_revision(root)
    if manifest.get("source") != current_source:
        raise ReleaseProvenanceError("source commit or clean-worktree assertion does not match")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != _REQUIRED_ARTIFACTS:
        raise ReleaseProvenanceError("release manifest has an incomplete artifact hash set")
    _verify_hashed_paths(root, artifacts, label="artifacts")
    _verify_hashed_paths(root, manifest.get("runtime"), label="runtime")
    provider_contract = _normalize_provider_contract(manifest.get("provider_contract", {}))
    if expected_provider_contract is not None and provider_contract != _normalize_provider_contract(expected_provider_contract):
        raise ReleaseProvenanceError("provider contract does not match the expected release contract")
    _normalize_history(manifest.get("required_history", {}))
    _normalize_missingness(manifest.get("missingness", {}))


def load_and_verify_release_manifest(
    project_root: Path | str,
    path: Path | str,
    *,
    expected_provider_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest_path = Path(path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReleaseProvenanceError(f"could not load release manifest: {manifest_path}") from error
    if not isinstance(manifest, dict):
        raise ReleaseProvenanceError("release manifest must contain a JSON object")
    verify_release_manifest(
        project_root, manifest, expected_provider_contract=expected_provider_contract
    )
    return manifest
