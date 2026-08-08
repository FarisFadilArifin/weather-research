from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_tokyo_immutable_history.py"
SPEC = importlib.util.spec_from_file_location("build_tokyo_immutable_history", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def history_frame() -> pd.DataFrame:
    rows = []
    for day, high in (("2026-08-07", 31.5), ("2026-08-06", 30.0)):
        rows.append(
            {
                "station_id": "RJTT",
                "contract_date": day,
                "actual_high_c": high,
                "truth_source": "wunderground_station_history",
                "truth_finalized": True,
                "observed_source": "iem_asos_global_metar",
                "observed_data_source": "iem_asos_global_metar_raw",
                "observed_humidity_at_as_of": 74.0,
                "observed_precip_recent_at_as_of": 0.03,
                "observed_visibility_at_as_of": 6.0,
                "observed_weather_code_at_as_of": "-RA",
                "gfs_high_f": 85.0,
                "gefs_high_f": 84.0,
                "jma_msm_high_f": 83.0,
                "safe_optional_feature": 1.0,
            }
        )
    return pd.DataFrame(rows)


def build(frame: pd.DataFrame, output: Path) -> dict[str, object]:
    return MODULE.build_history_seed(
        frame,
        output,
        start_date=date(2026, 8, 6),
        end_date=date(2026, 8, 7),
        source_commit="a" * 40,
        input_sha256="b" * 64,
    )


def test_history_seed_is_deterministic_target_free_and_manifested(tmp_path: Path) -> None:
    first = build(history_frame(), tmp_path / "first")
    second = build(history_frame().iloc[::-1], tmp_path / "second")
    first_history = Path(first["history"]).read_bytes()
    assert first_history == Path(second["history"]).read_bytes()
    records = [json.loads(line) for line in first_history.splitlines()]
    assert [record["contractDate"] for record in records] == ["2026-08-06", "2026-08-07"]
    assert all("actual_high_c" not in record["featureInputs"] for record in records)
    assert all("truth_finalized" not in record["featureInputs"] for record in records)
    assert records[0]["truth"]["source"] == "wunderground_station_history"
    assert records[0]["truth"]["finalized"] is True

    manifest = json.loads(Path(first["manifest"]).read_text(encoding="utf-8"))
    checksums = json.loads(Path(first["checksums"]).read_text(encoding="utf-8"))
    assert manifest["sourceIdentity"]["cleanCommit"] == "a" * 40
    assert manifest["history"]["jsonlSha256"] == hashlib.sha256(first_history).hexdigest()
    assert checksums["tokyo_history.manifest.json"] == hashlib.sha256(
        Path(first["manifest"]).read_bytes()
    ).hexdigest()
    assert "manifestSha256" not in manifest
    readiness = manifest["missingnessRequirements"]["rollingCalendarMonthReadiness"]
    assert readiness["providerCoverageMaximumMissingFraction"] == 0.0
    assert readiness["finalizedTruthRequired"] is True
    assert [item["calendarMonth"] for item in readiness["calendarMonths"]] == ["2026-08"]
    assert readiness["calendarMonths"][0]["ready"] is True


def test_history_seed_rejects_duplicate_dates_and_target_leakage(tmp_path: Path) -> None:
    duplicated = pd.concat([history_frame(), history_frame().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate_contract_date"):
        build(duplicated, tmp_path / "duplicate")
    leaked = history_frame()
    leaked["actual_high_f"] = 88.0
    with pytest.raises(ValueError, match="forbidden_leakage_column:actual_high_f"):
        build(leaked, tmp_path / "leaked")


def test_history_seed_rejects_provider_contract_drift(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="provider_contract_mismatch"):
        MODULE.build_history_seed(
            history_frame(),
            tmp_path,
            start_date=date(2026, 8, 6),
            end_date=date(2026, 8, 7),
            source_commit="a" * 40,
            input_sha256="b" * 64,
            providers=("gfs",),
        )


def test_history_seed_requires_finalized_truth_and_calendar_month_missingness_limits(
    tmp_path: Path,
) -> None:
    unfinalized = history_frame()
    unfinalized.loc[0, "truth_finalized"] = False
    with pytest.raises(ValueError, match="history_truth_not_finalized"):
        build(unfinalized, tmp_path / "unfinalized")

    sparse_optional = history_frame()
    sparse_optional["safe_optional_feature"] = None
    with pytest.raises(ValueError, match="optional_missingness_exceeded:2026-08"):
        build(sparse_optional, tmp_path / "sparse")

    missing_provider = history_frame()
    missing_provider.loc[0, "gfs_high_f"] = None
    with pytest.raises(ValueError, match="history_required_values_missing:gfs_high_f"):
        build(missing_provider, tmp_path / "provider")
