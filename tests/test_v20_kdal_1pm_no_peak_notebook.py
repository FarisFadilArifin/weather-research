from __future__ import annotations

import json
from pathlib import Path


def test_v20_kdal_1pm_no_peak_notebook_has_consistent_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook_root = (
        root
        / "notebooks"
        / "experiments"
        / "station_stacking_v20_kdal_1pm_no_peak"
    )
    generator = (notebook_root / "generate_station_notebook.py").read_text(encoding="utf-8")
    notebook = json.loads((notebook_root / "stacking_KDAL_v20_1pm_no_peak.ipynb").read_text(encoding="utf-8"))
    source = "".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert 'TIMING_MODE = "same_day_1pm_live_safe"' in generator
    assert 'FEATURE_VERSION = "v20_kdal_1pm_no_peak"' in generator
    assert 'TARGET_MODE = "remaining_warmup"' in generator
    assert source.count("config = StationStackingConfig(") == 1
    assert 'TIMING_MODE = "same_day_1pm_live_safe"' in source
    assert 'feature_version="v20_kdal_1pm_no_peak"' in source
    assert 'target_mode="remaining_warmup"' in source
    assert 'target_source="wunderground_only"' in source
    assert 'training_profile="v20_aligned"' in source
    assert "V20_KDAL_1PM_TEMP_FEATURE_COLUMNS" in source
    assert "v13sf_forecast_temp_1pm_minus_observed_f" in source
    assert "station_high_regressor_v20_kdal_1pm_no_peak_stack" in source
    assert 'source_pipeline="notebooks/experiments/station_stacking_v20_kdal_1pm_no_peak"' in source
    assert '"station_stacking_v20_kdal_1pm_no_peak"' in source
    assert "blocking_issue_count" in source
    assert 'feature_version="v11_settlement_fix_temp"' not in source
    assert 'TIMING_MODE = "same_day_11am_live_safe"' not in source
    assert "v11sf_forecast_temp_11am_minus_observed_f" not in source
    assert "V11_SETTLEMENT_FIX_TEMP_FEATURE_COLUMNS" not in source
    assert "V11_DROPPED_FEATURE_COLUMNS" not in source
    assert "V11_FEATURE_COLUMNS" not in source
    assert "V20_PEAK_TIMING_RAW_FEATURE_COLUMNS" not in source
    assert "V20_ENGINEERED_FEATURE_COLUMNS" not in source
    assert 'f"{STATION_ID}_1pm_feature_coverage.csv"' in source
    assert 'f"{STATION_ID}_1pm_vs_11am_common_date_comparison.csv"' in source
    assert 'f"{STATION_ID}_11am_feature_coverage.csv"' not in source
    assert 'f"{STATION_ID}_v11_common_date_comparison.csv"' not in source
