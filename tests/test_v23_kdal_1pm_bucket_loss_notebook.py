from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "notebooks/experiments/station_stacking_v23_kdal_1pm_bucket_loss/generate_notebook.py"


def _module():
    spec = importlib.util.spec_from_file_location("v23_generator", GENERATOR)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v23_preserves_data_contract_and_changes_selection_metric() -> None:
    notebook = _module().notebook()
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert 'TIMING_MODE = "same_day_1pm_live_safe"' in source
    assert 'OPTUNA_METRIC = "bucket_log_loss"' in source
    assert 'feature_version="v20_kdal_1pm_no_peak"' in source
    assert 'target_mode="remaining_warmup"' in source
    assert 'target_source="wunderground_only"' in source
    assert 'output_dir=PROJECT_ROOT / "data" / "calibration" / "station_stacking_v23_kdal_1pm_bucket_loss"' in source
    assert "station_high_regressor_v23_kdal_1pm_bucket_loss_stack" in source
    assert "V11_FEATURE_COLUMNS" not in source
    assert "WU-IEM" not in source
    assert '"station_stacking_v20_kdal_1pm_no_peak" / "audit" / "audit_result.json"' in source
    assert "catboost_max_iterations=1500" in source
    assert "catboost_max_depth=8" in source
    assert "catboost_max_border_count=128" in source
    assert "--feature-profile', 'kdal_1pm'" in source
    assert "audit_v23_kdal_1pm_bucket_loss.py" in source
    assert "holdout_rows_used_for_selection" not in source or "selector_audit" in source
    assert "point_bundle_sha256') == current_point_sha" in source
