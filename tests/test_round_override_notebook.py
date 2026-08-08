from __future__ import annotations

import json
import csv
from pathlib import Path


def test_round_override_v3_notebook_is_executed_and_audited() -> None:
    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / "notebooks"
        / "experiments"
        / "round_override_classifier_v3"
        / "half_up_override_classifier_v3.ipynb"
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
    assert "half-up" in source.lower()
    assert "override_target" in source
    assert "no boundary-distance windows" in source.lower()
    assert "default_is_half_up" in saved_outputs
    assert "override_target_formula" in saved_outputs
    assert "chronological_fit_calibration_policy_validation" in saved_outputs
    assert "no_boundary_distance_window" in saved_outputs

    audit_path = root / "data" / "calibration" / "station_round_override_v3" / "v3_audit.csv"
    with audit_path.open(encoding="utf-8", newline="") as handle:
        audit_rows = list(csv.DictReader(handle))
    assert len(audit_rows) == 28
    assert all(row["passed"] == "True" for row in audit_rows)
