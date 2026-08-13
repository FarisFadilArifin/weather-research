from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "notebooks" / "experiments" / "kdal_hrrr_9am_v1"
GENERATOR = EXPERIMENT / "generate_notebook.py"
CONFIG = EXPERIMENT / "config.json"
NOTEBOOK = EXPERIMENT / "train_KDAL.ipynb"


def _module():
    spec = importlib.util.spec_from_file_location("kdal_hrrr_9am_v1_generator", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source(notebook: dict) -> str:
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_config_freezes_provider_specific_timing_and_research_paths():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["hrrr_timing_mode"] == "same_day_11am_hrrr_9am_cycle_v1"
    assert config["hrrr_expected_rows"] == 2046
    assert config["hrrr_date_range"] == ["2021-01-01", "2026-08-08"]
    assert config["observation_cutoff_local"] == "11:00"
    assert config["prediction_decision_time_local"] == "11:15"
    assert config["point_evaluation_train_years"] == [2021, 2025]
    assert config["point_max_feature_missing_fraction"] == 0.03
    assert config["ordinal_challenger_enabled"] is False
    assert config["data_project_root"] == "D:/dev/weather-research"


def test_generated_notebook_is_isolated_and_preserves_kdal_lineage():
    notebook = _module().build_notebook()
    source = _source(notebook)
    metadata = notebook["metadata"]["station_training_baseline"]

    assert "TIMING_MODE = \"same_day_11am_live_safe\"" in source
    assert "DATA_PROJECT_ROOT = Path('D:/dev/weather-research').resolve()" in source
    assert "project_root=DATA_PROJECT_ROOT" in source
    assert "same_day_11am_hrrr_9am_cycle_v1" in source
    assert "set(issue_hours_utc) == {14, 15}" in source
    assert "forecast_hour_min" in source and "forecast_hour_max" in source
    assert "baseline[\"provider\"]" in source
    assert '.ne("hrrr")' in source
    assert 'set(hrrr_rows["timing_mode"].astype(str)) == {HRRR_9AM_TIMING_MODE}' in source
    assert 'set(station_availability["provider"].astype(str)) == set(PROVIDERS)' in source
    assert "load_current_observation_features" in source
    assert "training_profile=\"v20_aligned\"" in source
    assert 'feature_version="v11_settlement_fix_temp"' in source
    assert 'target_source="wunderground_only"' in source
    assert "V20_EXPANDING_FOLDS" in source
    assert "POINT_MAX_FEATURE_MISSING_FRACTION = 0.03" in source
    assert "PROBABILITY_FEATURE_COUNT = 59" in source
    assert "run_challenger" not in source
    assert 'PROJECT_ROOT / "data" / "calibration" / "experiments" / "kdal_hrrr_9am_v1"' in source
    assert 'PROJECT_ROOT / "data" / "calibration" / "station_training_baseline" / "KDAL"' not in source
    assert "notebooks/experiments/kdal_hrrr_9am_v1" in source
    assert "notebooks/station_training_baseline/experiments/kdal_hrrr_9am_v1" not in source
    assert metadata["status"] == "research_only"
    assert metadata["production_export"] is False
    assert metadata["deployed"] is False


def test_checked_in_notebook_matches_generator():
    generated = _module().build_notebook()
    checked_in = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert checked_in == generated
