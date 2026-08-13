from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = (
    ROOT
    / "notebooks"
    / "experiments"
    / "station_stacking_v24_kdal_no_peak_diverse_ensemble"
)
GENERATOR = NOTEBOOK_DIR / "generate_station_notebook.py"
NOTEBOOK = NOTEBOOK_DIR / "stacking_KDAL_v24_diverse_ensemble.ipynb"


def test_v24_generator_writes_diverse_ensemble_notebook() -> None:
    subprocess.run([sys.executable, str(GENERATOR)], cwd=ROOT, check=True)
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert 'base_model_methods=("xgboost", "extra_trees", "ridge")' in source
    assert "station_high_regressor_v24_kdal_no_peak_diverse_stack" in source
    assert "station_stacking_v24_kdal_no_peak_diverse_ensemble" in source
    assert 'feature_version="v11_settlement_fix_temp"' in source
    assert 'training_profile="v20_aligned"' in source
    assert 'target_source="wunderground_only"' in source
    assert "EXPORT_MODEL_WEIGHTS = True" in source
