from __future__ import annotations

import hashlib
import importlib.util
import json
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
        observed_source="rjtt_metar",
        actual_high_f=99.0,
        optional_missing=float("nan"),
    )
    return pd.DataFrame([row])


def test_payload_is_target_free_and_publishes_verified_sidecar(tmp_path):
    payload = MODULE.build_payload(
        feature_frame(),
        date(2026, 8, 6),
        source_commit="a" * 40,
        generated_at=datetime(2026, 8, 6, 2, 10, tzinfo=UTC),
    )
    assert "actual_high_f" not in payload["featureInputs"]
    assert payload["featureInputs"]["optional_missing"] is None
    artifact, sidecar = MODULE.publish(payload, tmp_path)
    raw = artifact.read_bytes()
    assert sidecar.read_text().split()[0] == hashlib.sha256(raw).hexdigest()
    assert json.loads(raw)["providers"] == ["gfs", "gefs", "jma_msm"]


def test_payload_rejects_missing_provider_and_stale_observation():
    frame = feature_frame()
    frame.loc[0, "gefs_high_f"] = float("nan")
    with pytest.raises(ValueError, match="missing_required_live_features:gefs_high_f"):
        MODULE.build_payload(
            frame,
            date(2026, 8, 6),
            source_commit="a" * 40,
            generated_at=datetime.now(UTC),
        )
    frame = feature_frame()
    frame.loc[0, "observed_as_of_time_local"] = "2026-08-06T10:39:00+09:00"
    with pytest.raises(ValueError, match="observation_outside"):
        MODULE.build_payload(
            frame,
            date(2026, 8, 6),
            source_commit="a" * 40,
            generated_at=datetime.now(UTC),
        )
