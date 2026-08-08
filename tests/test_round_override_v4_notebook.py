from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib


def test_round_override_v4_notebook_is_executed_and_audited() -> None:
    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / "notebooks"
        / "experiments"
        / "round_override_classifier_v4"
        / "half_up_utility_override_v4.ipynb"
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
    assert "p(recovery) - damage_penalty * p(damage)" in source
    assert "actionable rows only" in source.lower()
    assert "three earlier chronological policy folds" in source.lower()
    assert "recovery_target_formula" in saved_outputs
    assert "damage_target_formula" in saved_outputs
    assert "continuous_features_are_strictly_prior" in saved_outputs
    assert "policy_selection_matches_eligibility" in saved_outputs
    assert "final_policy_matches_eligibility" in saved_outputs

    output_root = root / "data" / "calibration" / "station_round_override_v4"
    with (output_root / "v4_audit.csv").open(encoding="utf-8", newline="") as handle:
        audit_rows = list(csv.DictReader(handle))
    assert len(audit_rows) == 50
    assert all(row["passed"] == "True" for row in audit_rows)


def test_round_override_v4_exported_bundles_are_loadable_and_safe() -> None:
    root = Path(__file__).resolve().parents[1]
    output_root = root / "data" / "calibration" / "station_round_override_v4"
    for station in ("KATL", "KDAL"):
        path = (
            output_root
            / station
            / "model_weights"
            / f"{station}_half_up_utility_override_v4.joblib"
        )
        bundle = joblib.load(path)
        assert bundle["artifact_type"] == "station_half_up_utility_override_v4_research"
        assert bundle["schema_version"] == 4
        assert bundle["station_id"] == station
        assert set(bundle["state"]) >= {"recovery", "damage", "policy_enabled"}
        assert not bundle["state"]["policy_enabled"]
