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
    / "station_stacking_v25_kdal_xgb_extra_blend"
)
GENERATOR = NOTEBOOK_DIR / "generate_station_notebook.py"
NOTEBOOK = NOTEBOOK_DIR / "stacking_KDAL_v25_xgb_extra_blend.ipynb"


def test_v25_generator_writes_no_retuning_constrained_blend_notebook() -> None:
    subprocess.run([sys.executable, str(GENERATOR)], cwd=ROOT, check=True)
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "station_stacking_v20_kdal_no_peak" in source
    assert "station_stacking_v24_kdal_no_peak_diverse_ensemble" in source
    assert "scan_two_model_weights" in source
    assert "GRID_STEP = 0.001" in source
    assert "xgb_extra_constrained_blend" in source
    assert "tune_year_split_base_models" not in source
    assert "optuna" not in source.lower()
