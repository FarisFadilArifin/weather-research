from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_v27_runner_freezes_ordinal_contract() -> None:
    source = (
        PROJECT_ROOT / "scripts" / "run_v27_kdal_no_peak_ordinal.py"
    ).read_text(encoding="utf-8")
    assert '"ordinal_logistic"' in source
    assert '"1.0"' in source
    assert '"0.25,0.5,0.75,1.0"' in source
    assert '"exploratory"' in source
    assert "station_high_regressor_v20_kdal_no_peak_stack" in source


def test_v27_audit_guards_point_identity_and_fresh_shadow_requirement() -> None:
    source = (
        PROJECT_ROOT / "scripts" / "audit_v27_kdal_no_peak_ordinal.py"
    ).read_text(encoding="utf-8")
    assert "point_bundle_sha256" in source
    assert "forced_ordinal" in source
    assert "probability_simplex" in source
    assert "chronology" in source
    assert "holdout_prediction_export" in source
    assert "fresh_shadow_data_required" in source


def test_v27_notebook_is_generated_from_source() -> None:
    directory = (
        PROJECT_ROOT
        / "notebooks"
        / "experiments"
        / "station_stacking_v27_kdal_no_peak_ordinal"
    )
    generator = (directory / "generate_notebook.py").read_text(encoding="utf-8")
    readme = (directory / "README.md").read_text(encoding="utf-8")
    assert "run_v27_kdal_no_peak_ordinal.py" in generator
    assert "2026 period is exploratory" in generator
    assert "promotion: prohibited" in readme
