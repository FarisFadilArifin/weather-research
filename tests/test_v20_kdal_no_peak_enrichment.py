from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from src.calibration import station_stacking


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_audit_module():
    path = REPO_ROOT / "scripts" / "audit_v20_kdal_no_peak_enrichment.py"
    spec = importlib.util.spec_from_file_location("audit_v20_kdal_no_peak_enrichment", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v20_kdal_enrichment_contract_is_11am_no_peak() -> None:
    module = _load_audit_module()
    assert module.STATION == "KDAL"
    assert module.TIMING_MODE == "same_day_11am_live_safe"
    assert module.FEATURE_VERSION == "v11_settlement_fix_temp"
    assert "v11sf_forecast_temp_11am_mean_f" in module.REQUIRED_TAIL_COLUMNS
    assert all("peak" not in column for column in module.REQUIRED_TAIL_COLUMNS)


def test_v20_kdal_enrichment_runner_uses_parallel_isolated_shards() -> None:
    source = (
        REPO_ROOT / "scripts" / "run_v20_kdal_no_peak_enrichment.ps1"
    ).read_text(encoding="utf-8")
    assert '[int]$ShardDays = 1' in source
    assert '[int]$MaxParallel = 12' in source
    assert '[int]$ForecastFxxWorkers = 3' in source
    assert "sdk_11am_live_safe_v20_kdal_enrich_" in source
    assert "direct_nbm_v20_kdal_enrich_" in source
    assert "sdk_current_obs_v20_kdal_enrich_" in source
    assert "--include-weather-features" in source
    assert "audit_v20_kdal_no_peak_enrichment.py" in source


def test_wunderground_only_adds_dates_beyond_iem_actual_spine(tmp_path: Path) -> None:
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    pd.DataFrame(
        {
            "station_id": ["KDAL"],
            "contract_date": ["2026-07-29"],
            "settlement_high_f": [99.0],
            "settlement_source": ["wunderground_station_history"],
            "quality_flag": ["ok"],
        }
    ).to_csv(processed / "settlement_actual_highs.csv", index=False)
    actuals = pd.DataFrame(
        {
            "contract_date": ["2026-07-28"],
            "actual_high_f": [98.0],
            "iem_actual_high_f": [98.0],
            "settlement_high_f": [pd.NA],
            "settlement_source": [pd.NA],
            "settlement_quality_flag": [pd.NA],
            "target_source": ["iem_hourly"],
            "actual_source": ["iem"],
            "actual_data_quality_flag": ["ok"],
            "actual_raw_observation_count": [24],
        }
    )

    result = station_stacking._apply_settlement_first_actuals(
        tmp_path,
        "KDAL",
        actuals,
        target_source="wunderground_only",
    )

    assert result["contract_date"].tolist() == ["2026-07-28", "2026-07-29"]
    tail = result.loc[result["contract_date"].eq("2026-07-29")].iloc[0]
    assert tail["actual_high_f"] == 99.0
    assert pd.isna(tail["iem_actual_high_f"])
    assert tail["actual_source"] == "wunderground_station_history"
