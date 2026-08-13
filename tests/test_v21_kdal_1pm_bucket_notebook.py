from __future__ import annotations

import json
from pathlib import Path


def test_v21_kdal_1pm_bucket_notebook_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    folder = root / "notebooks/experiments/station_stacking_v21_kdal_1pm_bucket"
    notebook = json.loads((folder / "v21_kdal_1pm_bucket.ipynb").read_text(encoding="utf-8"))
    source = "".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert notebook["nbformat"] == 4
    assert "same_day_1pm_live_safe" in source
    assert "v20_kdal_1pm_no_peak" in source
    assert "station_bucket_v21_kdal_1pm" in source
    assert "--feature-profile', 'kdal_1pm" in source
    assert "audit_v21_kdal_1pm_bucket.py" in source
    assert "RESEARCH-ONLY" in source
