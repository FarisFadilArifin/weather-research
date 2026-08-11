from __future__ import annotations

import importlib.util
from pathlib import Path

import joblib
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_kdal_ordinal_challenger_v1.py"
SPEC = importlib.util.spec_from_file_location("kdal_ordinal_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_runner_fails_before_training_when_point_version_does_not_match(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "point.joblib"
    joblib.dump({"model_version": "different"}, bundle)
    MODULE.build_frames = lambda _: ({}, {}, {"point_bundle": bundle})

    with pytest.raises(ValueError, match="does not match"):
        MODULE.run_challenger(
            point_model_version="expected",
            point_bundle_path=bundle,
            output_dir=tmp_path / "output",
        )
