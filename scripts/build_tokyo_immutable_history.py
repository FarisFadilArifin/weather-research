from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


STATION_ID = "RJTT"
OBSERVATION_PROVIDER = "iem_asos_global_metar"
FORECAST_PROVIDERS = ("gfs", "gefs", "jma_msm")
REQUIRED_OBSERVATION_FIELDS = (
    "observed_humidity_at_as_of",
    "observed_precip_recent_at_as_of",
    "observed_visibility_at_as_of",
    "observed_weather_code_at_as_of",
)
REQUIRED_FEATURE_MAX_MISSINGNESS = 0.0
OPTIONAL_FEATURE_MAX_MISSINGNESS_PER_CALENDAR_MONTH = 0.5
TARGET_COLUMNS = frozenset(
    {
        "actual_high_f",
        "actual_high_c",
        "settlement_high_f",
        "settlement_high_c",
        "iem_daily_high_f",
        "iem_daily_high_c",
        "remaining_warmup_f",
        "remaining_warmup_from_observed_high_so_far_f",
    }
)
METADATA_COLUMNS = frozenset(
    {
        "station_id",
        "stationId",
        "contract_date",
        "contractDate",
        "observed_source",
        "observed_data_source",
        "truth_source",
        "settlement_source",
        "target_source",
        "truth_finalized",
    }
)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def _require_commit(commit: str) -> str:
    value = str(commit).strip().lower()
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("invalid_clean_source_commit")
    return value


def _column(frame: pd.DataFrame, *names: str) -> str:
    for name in names:
        if name in frame.columns:
            return name
    raise ValueError("missing_required_column:" + names[0])


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _is_finalized_truth(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"true", "finalized"}


def _expected_dates(start_date: date, end_date: date) -> list[str]:
    values: list[str] = []
    current = start_date
    while current <= end_date:
        values.append(current.isoformat())
        current += timedelta(days=1)
    return values


def _calendar_month_readiness(
    working: pd.DataFrame,
    *,
    date_column: str,
    feature_columns: list[str],
    required_feature_columns: list[str],
    providers: tuple[str, ...],
    truth_source_column: str,
) -> list[dict[str, Any]]:
    """Validate rolling readiness independently for every calendar month."""
    readiness: list[dict[str, Any]] = []
    optional_columns = sorted(set(feature_columns) - set(required_feature_columns))
    month_keys = pd.to_datetime(working[date_column]).dt.strftime("%Y-%m")
    for month, group in working.groupby(month_keys, sort=True):
        provider_missingness = {
            provider: float(group[f"{provider}_high_f"].map(_finite).eq(False).mean())
            for provider in providers
        }
        required_missingness = {
            field: float(group[field].map(_finite).eq(False).mean())
            for field in required_feature_columns
            if field != "observed_weather_code_at_as_of"
        }
        required_missingness["observed_weather_code_at_as_of"] = float(
            group["observed_weather_code_at_as_of"].map(_nonempty_text).eq(False).mean()
        )
        optional_missingness = {
            field: float(group[field].map(lambda value: _clean(value) is None).mean())
            for field in optional_columns
        }
        incomplete_providers = [
            provider
            for provider, fraction in provider_missingness.items()
            if fraction > REQUIRED_FEATURE_MAX_MISSINGNESS
        ]
        incomplete_required = [
            field
            for field, fraction in required_missingness.items()
            if fraction > REQUIRED_FEATURE_MAX_MISSINGNESS
        ]
        excessive_optional = [
            field
            for field, fraction in optional_missingness.items()
            if fraction > OPTIONAL_FEATURE_MAX_MISSINGNESS_PER_CALENDAR_MONTH
        ]
        if incomplete_providers:
            raise ValueError(
                f"history_calendar_month_provider_coverage_missing:{month}:"
                + ",".join(incomplete_providers)
            )
        if incomplete_required:
            raise ValueError(
                f"history_calendar_month_required_missingness_exceeded:{month}:"
                + ",".join(sorted(incomplete_required))
            )
        if excessive_optional:
            raise ValueError(
                f"history_calendar_month_optional_missingness_exceeded:{month}:"
                + ",".join(excessive_optional)
            )
        finalized = group["truth_finalized"].map(_is_finalized_truth)
        if not finalized.all():
            raise ValueError(f"history_calendar_month_truth_not_finalized:{month}")
        readiness.append(
            {
                "calendarMonth": str(month),
                "rowCount": int(len(group)),
                "providerMissingness": provider_missingness,
                "requiredFeatureMissingness": required_missingness,
                "optionalFeatureMissingness": optional_missingness,
                "finalizedTruthRows": int(finalized.sum()),
                "truthLineageColumn": truth_source_column,
                "ready": True,
            }
        )
    return readiness


def validate_history(
    frame: pd.DataFrame,
    *,
    start_date: date,
    end_date: date,
    truth_column: str,
    providers: Iterable[str] = FORECAST_PROVIDERS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return deterministic, target-free rows only after the seed gate passes."""
    if end_date < start_date:
        raise ValueError("history_end_before_start")
    station_column = _column(frame, "station_id", "stationId")
    date_column = _column(frame, "contract_date", "contractDate")
    if truth_column not in frame:
        raise ValueError("missing_truth_column:" + truth_column)
    for field in (
        "observed_source",
        "observed_data_source",
        "truth_finalized",
        *REQUIRED_OBSERVATION_FIELDS,
    ):
        _column(frame, field)
    for provider in providers:
        _column(frame, f"{provider}_high_f")
    forbidden = sorted((TARGET_COLUMNS - {truth_column}) & set(frame.columns))
    if forbidden:
        raise ValueError("forbidden_leakage_column:" + ",".join(forbidden))

    working = frame.copy()
    working[date_column] = pd.to_datetime(working[date_column], errors="coerce").dt.date
    if working[date_column].isna().any():
        raise ValueError("invalid_contract_date")
    if not working[station_column].astype(str).eq(STATION_ID).all():
        raise ValueError("history_station_must_be_RJTT")
    if not working["observed_source"].astype(str).eq(OBSERVATION_PROVIDER).all():
        raise ValueError("history_observation_provider_mismatch")
    if not working["observed_data_source"].astype(str).str.startswith(OBSERVATION_PROVIDER).all():
        raise ValueError("history_observation_lineage_mismatch")
    if working[date_column].duplicated().any():
        raise ValueError("duplicate_contract_date")

    expected_dates = _expected_dates(start_date, end_date)
    actual_dates = sorted(day.isoformat() for day in working[date_column])
    if actual_dates != expected_dates:
        raise ValueError("history_date_range_or_coverage_mismatch")

    truth_source_column = next(
        (name for name in ("truth_source", "settlement_source", "target_source") if name in working),
        None,
    )
    if truth_source_column is None or not working[truth_source_column].map(_nonempty_text).all():
        raise ValueError("missing_truth_lineage")
    if not working["truth_finalized"].map(_is_finalized_truth).all():
        raise ValueError("history_truth_not_finalized")

    required_numeric = [truth_column, *(f"{provider}_high_f" for provider in providers)]
    required_numeric.extend(
        field for field in REQUIRED_OBSERVATION_FIELDS if field != "observed_weather_code_at_as_of"
    )
    missing = [
        field
        for field in required_numeric
        if not working[field].map(_finite).all()
    ]
    if missing:
        raise ValueError("history_required_values_missing:" + ",".join(sorted(missing)))
    if not working["observed_weather_code_at_as_of"].map(_nonempty_text).all():
        raise ValueError("history_required_values_missing:observed_weather_code_at_as_of")

    feature_columns = sorted(
        name
        for name in working.columns
        if name not in TARGET_COLUMNS and name not in METADATA_COLUMNS
    )
    required_feature_columns = sorted(
        set(required_numeric + ["observed_weather_code_at_as_of"])
    )
    calendar_month_readiness = _calendar_month_readiness(
        working,
        date_column=date_column,
        feature_columns=feature_columns,
        required_feature_columns=required_feature_columns,
        providers=tuple(providers),
        truth_source_column=truth_source_column,
    )
    records: list[dict[str, Any]] = []
    missingness: dict[str, float] = {}
    for name in feature_columns:
        missingness[name] = float(working[name].map(lambda value: _clean(value) is None).mean())
    for _, row in working.sort_values(date_column).iterrows():
        records.append(
            {
                "stationId": STATION_ID,
                "contractDate": row[date_column].isoformat(),
                "providers": list(providers),
                "observationLineage": {
                    "provider": OBSERVATION_PROVIDER,
                    "dataSource": str(row["observed_data_source"]),
                },
                "truth": {
                    "column": truth_column,
                    "value": float(row[truth_column]),
                    "source": str(row[truth_source_column]),
                    "finalized": True,
                },
                "featureInputs": {name: _clean(row[name]) for name in feature_columns},
            }
        )
    return records, {
        "requiredNonNullFields": required_feature_columns,
        "optionalFeatureMissingness": {name: missingness[name] for name in sorted(missingness)},
        "truthColumn": truth_column,
        "truthLineageColumn": truth_source_column,
        "rollingCalendarMonthReadiness": {
            "providerCoverageMaximumMissingFraction": REQUIRED_FEATURE_MAX_MISSINGNESS,
            "requiredFeatureMaximumMissingFraction": REQUIRED_FEATURE_MAX_MISSINGNESS,
            "optionalFeatureMaximumMissingFraction": (
                OPTIONAL_FEATURE_MAX_MISSINGNESS_PER_CALENDAR_MONTH
            ),
            "finalizedTruthRequired": True,
            "calendarMonths": calendar_month_readiness,
        },
    }


def _atomic_write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def build_history_seed(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    start_date: date,
    end_date: date,
    source_commit: str,
    input_sha256: str,
    truth_column: str = "actual_high_c",
    providers: Iterable[str] = FORECAST_PROVIDERS,
    archive_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    commit = _require_commit(source_commit)
    input_sha256 = str(input_sha256).strip().lower()
    if len(input_sha256) != 64 or any(character not in "0123456789abcdef" for character in input_sha256):
        raise ValueError("invalid_input_sha256")
    if archive_manifest_sha256 is not None:
        archive_manifest_sha256 = str(archive_manifest_sha256).strip().lower()
        if len(archive_manifest_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in archive_manifest_sha256
        ):
            raise ValueError("invalid_worker_archive_manifest_sha256")
    providers = tuple(providers)
    if providers != FORECAST_PROVIDERS:
        raise ValueError("tokyo_history_provider_contract_mismatch")
    records, requirements = validate_history(
        frame,
        start_date=start_date,
        end_date=end_date,
        truth_column=truth_column,
        providers=providers,
    )
    history_raw = b"".join(
        (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for record in records
    )
    history_sha = sha256_bytes(history_raw)
    manifest = {
        "artifactType": "weather_research_tokyo_immutable_history_v1",
        "stationId": STATION_ID,
        "sourceIdentity": {
            "cleanCommit": commit,
            "workerArchiveManifestSha256": archive_manifest_sha256,
        },
        "observationProviderContract": {
            "trainingProvider": OBSERVATION_PROVIDER,
            "runtimeProvider": OBSERVATION_PROVIDER,
            "population": "RJTT METAR observations at or before 11:00 Asia/Tokyo",
        },
        "forecastProviders": list(providers),
        "history": {
            "startDate": start_date.isoformat(),
            "endDate": end_date.isoformat(),
            "rowCount": len(records),
            "inputSha256": input_sha256,
            "jsonlSha256": history_sha,
            "deterministicOrder": "contractDate_ascending",
            "duplicatesAllowed": False,
            "leakagePolicy": "target_columns_are_truth_only_and_never_feature_inputs",
        },
        "missingnessRequirements": requirements,
    }
    manifest_raw = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    checksums = {
        "tokyo_history.jsonl": history_sha,
        "tokyo_history.manifest.json": sha256_bytes(manifest_raw),
    }
    checksums_raw = (json.dumps(checksums, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    _atomic_write(output_dir / "tokyo_history.jsonl", history_raw)
    _atomic_write(output_dir / "tokyo_history.manifest.json", manifest_raw)
    _atomic_write(output_dir / "tokyo_history.checksums.json", checksums_raw)
    return {
        "status": "ok",
        "history": str(output_dir / "tokyo_history.jsonl"),
        "manifest": str(output_dir / "tokyo_history.manifest.json"),
        "checksums": str(output_dir / "tokyo_history.checksums.json"),
        "rowCount": len(records),
        "historySha256": history_sha,
    }


def _read_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    raise ValueError("unsupported_history_input_format")


def _clean_git_commit(project_root: Path) -> str:
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise ValueError("tracked_worktree_is_dirty")
    return _require_commit(
        subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=project_root, check=True, capture_output=True, text=True
        ).stdout
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a deterministic immutable RJTT history seed")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-date", type=date.fromisoformat, required=True)
    parser.add_argument("--end-date", type=date.fromisoformat, required=True)
    parser.add_argument("--truth-column", default="actual_high_c")
    parser.add_argument("--archive-manifest", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    archive_sha = None
    if args.archive_manifest:
        manifest = json.loads(args.archive_manifest.read_text(encoding="utf-8"))
        if manifest.get("sourceCommit") != _clean_git_commit(project_root):
            raise ValueError("worker_archive_commit_mismatch")
        archive_sha = sha256_file(args.archive_manifest)
    report = build_history_seed(
        _read_frame(args.input),
        args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        source_commit=_clean_git_commit(project_root),
        input_sha256=sha256_file(args.input),
        truth_column=args.truth_column,
        providers=FORECAST_PROVIDERS,
        archive_manifest_sha256=archive_sha,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
