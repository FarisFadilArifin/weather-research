from __future__ import annotations

import json

import pytest

from src.tokyo_runtime_package import (
    EXPECTED_REPLAY,
    RUNTIME_CONTRACT_SHA256,
    validate_replay,
)


def replay_payload() -> dict:
    return {
        "method": {
            "station": "RJTT/Tokyo",
            "selector": "point_bucket_c",
            "entry_policy": "no_filter",
            "cost_cap": 0.47,
        },
        "flat_4": dict(EXPECTED_REPLAY),
    }


def test_reference_replay_is_exact(tmp_path):
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(replay_payload()), encoding="utf-8")
    assert validate_replay(path) == EXPECTED_REPLAY


def test_reference_replay_rejects_material_mismatch(tmp_path):
    payload = replay_payload()
    payload["flat_4"]["entries"] = 95
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="replay_entries_mismatch"):
        validate_replay(path)


def test_runtime_contract_hash_is_pinned():
    assert len(RUNTIME_CONTRACT_SHA256) == 64
    assert set(RUNTIME_CONTRACT_SHA256) <= set("0123456789abcdef")
