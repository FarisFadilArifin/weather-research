from __future__ import annotations

import csv
import json
from pathlib import Path


def test_directional_residual_audit_notebook_is_executed() -> None:
    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / "notebooks"
        / "experiments"
        / "directional_residual_audit_v1"
        / "directional_residual_audit_v1.ipynb"
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
    assert "diagnostic, not a boundary eligibility rule" in source.lower()
    assert "benjamini-hochberg" in source.lower()
    assert "2026 rows only confirm" in source.lower()
    assert "zero subgroup" in saved_outputs.lower() or "no subgroup" in saved_outputs.lower()
    assert "point_predictions_are_forward" in saved_outputs

    audit_path = root / "data" / "calibration" / "directional_residual_audit_v1" / "audit.csv"
    with audit_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 26
    assert all(row["passed"] == "True" for row in rows)


def test_no_utility_signal_survives_2026_confirmation() -> None:
    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / "data"
        / "calibration"
        / "directional_residual_audit_v1"
        / "signal_summary.csv"
    )
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["station"] for row in rows} == {"KATL", "KDAL"}
    assert all(int(row["confirmed_utility_signals_2026"]) == 0 for row in rows)
