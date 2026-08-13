from __future__ import annotations

import json
from pathlib import Path


def test_v20_kdal_no_peak_notebook_matches_v20_contract_and_exports() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook_root = (
        root
        / "notebooks"
        / "experiments"
        / "station_stacking_v20_kdal_no_peak"
    )
    generator = (notebook_root / "generate_station_notebook.py").read_text(encoding="utf-8")

    assert 'STATION_ID = "KDAL"' in generator
    assert 'training_profile="v20_aligned"' in generator
    assert 'feature_version="v11_settlement_fix_temp"' in generator
    assert 'target_source="wunderground_only"' in generator
    assert "max_feature_missing_fraction=0.03" in generator
    assert "year_split_folds=V20_EXPANDING_FOLDS" in generator
    assert "year_split_validation_weights={2022: 1.0, 2023: 1.0, 2024: 1.0, 2025: 1.0}" in generator
    assert "EXPORT_MODEL_WEIGHTS = True" in generator

    path = notebook_root / "stacking_KDAL_v20_no_peak.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert notebook["nbformat"] == 4
    assert source.count("config = StationStackingConfig(") == 1
    assert 'training_profile="v20_aligned"' in source
    assert 'feature_version="v11_settlement_fix_temp"' in source
    assert 'target_source="wunderground_only"' in source
    assert "max_feature_missing_fraction=0.03" in source
    assert "year_split_folds=V20_EXPANDING_FOLDS" in source
    assert "year_split_validation_weights={2022: 1.0, 2023: 1.0, 2024: 1.0, 2025: 1.0}" in source
    assert "EXPORT_MODEL_WEIGHTS = True" in source
    assert "EXPORT_MODEL_WEIGHTS = False" not in source
    assert "station_high_regressor_v20_kdal_no_peak_stack" in source
    assert 'source_pipeline="notebooks/experiments/station_stacking_v20_kdal_no_peak"' in source
    assert '"station_stacking_v20_kdal_no_peak"' in source
    assert "export_station_model_weights(" in source
    assert "training_profile=config.effective_training_profile" in source
    assert 'feature_version="v20_peak_timing"' not in source
    assert "V20_PEAK_TIMING_RAW_FEATURE_COLUMNS" not in source
    assert "V20_ENGINEERED_FEATURE_COLUMNS" not in source
    assert "v20_peak_timing_readiness" not in source

    from src.calibration.station_stacking import (
        StationStackingConfig,
        V20_EXPANDING_FOLDS,
        _uses_expanding_stack_validation,
    )

    config = StationStackingConfig(
        station_id="KDAL",
        feature_version="v11_settlement_fix_temp",
        training_profile="v20_aligned",
    )
    assert config.effective_year_split_folds == V20_EXPANDING_FOLDS
    assert _uses_expanding_stack_validation(config)
