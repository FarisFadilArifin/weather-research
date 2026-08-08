from __future__ import annotations

import json
from pathlib import Path


def test_round_direction_notebook_is_executed_and_audited() -> None:
    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / "notebooks"
        / "experiments"
        / "round_direction_classifier_v1"
        / "floor_ceil_classifier_v1.ipynb"
    )
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert all("id" in cell for cell in notebook["cells"])
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    assert all(cell["execution_count"] is not None for cell in code_cells)
    outputs = [output for cell in code_cells for output in cell.get("outputs", [])]
    assert not [output for output in outputs if output.get("output_type") == "error"]
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    saved_outputs = json.dumps(outputs, sort_keys=True)
    assert "0 = floor(regression prediction)" in source
    assert "1 = ceil(regression prediction)" in source
    assert "no_boundary_window_or_abstention" in source
    assert "classification_threshold" not in source or "0.5" in source
    assert "all_outer_rows_scored" in saved_outputs
    assert "binary_target_formula" in saved_outputs
    assert "point_predictions_are_forward" in saved_outputs
