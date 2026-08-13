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
    / "station_stacking_v26_kdal_xgb_extra_svr_blend"
)
GENERATOR = NOTEBOOK_DIR / "generate_station_notebook.py"
NOTEBOOK = NOTEBOOK_DIR / "stacking_KDAL_v26_xgb_extra_svr_blend.ipynb"


def test_v26_generator_writes_svr_only_tuning_and_simplex_blend() -> None:
    subprocess.run([sys.executable, str(GENERATOR)], cwd=ROOT, check=True)
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert 'base_model_methods=("svr",)' in source
    assert "SVR_TRIALS = 30" in source
    assert "SIMPLEX_STEP = 0.005" in source
    assert "scan_three_model_simplex_weights" in source
    assert "xgb_extra_svr_simplex_blend" in source
    assert '"xgboost": False' in source
    assert '"extra_trees": False' in source
    assert '"svr": True' in source
