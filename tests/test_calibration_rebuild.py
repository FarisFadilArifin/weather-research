from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from src.calibration.dataset import build_calibration_samples
from src.calibration.modeling import WalkForwardConfig, walk_forward_baseline_predictions
from src.calibration.time_rules import forecast_as_of_utc


def _write_minimal_project(tmp_path, days: int = 4) -> None:
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    dates = pd.date_range("2026-01-01", periods=days, freq="D").date
    pd.DataFrame(
        {
            "station_code": ["KATL"],
            "station_name": ["Atlanta/Hartsfield-Jackson Intl"],
            "airport_name": ["Atlanta/Hartsfield-Jackson Intl"],
            "lat": [33.63],
            "lon": [-84.44],
            "timezone": ["America/New_York"],
            "country": ["US"],
        }
    ).to_csv(processed / "station_registry.csv", index=False)
    pd.DataFrame(
        {
            "station_code": ["KATL"] * days,
            "date_local": [d.isoformat() for d in dates],
            "actual_high_f": [71 + i for i in range(days)],
        }
    ).to_csv(processed / "actual_highs.csv", index=False)
    pd.DataFrame(
        {
            "station_code": ["KATL"] * days,
            "station_name": ["Atlanta/Hartsfield-Jackson Intl"] * days,
            "airport_name": ["Atlanta/Hartsfield-Jackson Intl"] * days,
            "provider": ["nws"] * days,
            "model": ["nbm"] * days,
            "issue_time_utc": [f"{d.isoformat()}T06:00:00+00:00" for d in dates],
            "target_date_local": [d.isoformat() for d in dates],
            "forecast_horizon_hours": [0] * days,
            "forecast_high_f": [70.0] * days,
            "source_file_or_url": ["nbm tmax"] * days,
        }
    ).to_csv(processed / "nws_forecast_snapshots.csv", index=False)


def test_zero_hour_is_six_am_local_time_with_dst() -> None:
    assert forecast_as_of_utc("2026-01-15", "America/New_York").isoformat() == "2026-01-15T11:00:00+00:00"
    assert forecast_as_of_utc("2026-07-15", "America/New_York").isoformat() == "2026-07-15T10:00:00+00:00"


def test_legacy_nws_nbm_rows_are_relabelled_to_nbm(tmp_path) -> None:
    _write_minimal_project(tmp_path)
    samples = build_calibration_samples(tmp_path)
    assert set(samples["provider"]) == {"nbm"}
    assert set(samples["model"]) == {"nbm_tmax"}
    assert "nws" not in set(samples["provider"])


def test_rolling_bias_features_use_only_past_rows(tmp_path) -> None:
    _write_minimal_project(tmp_path, days=4)
    samples = build_calibration_samples(tmp_path).sort_values("contract_date")
    assert math.isnan(samples.iloc[0]["rolling_provider_bias_7d"])
    assert samples.iloc[3]["rolling_provider_bias_7d"] == 2.0
    assert samples.iloc[3]["past_provider_sample_count"] == 3


def test_station_month_climatology_uses_prior_dates_not_prior_same_day_rows(tmp_path) -> None:
    processed = tmp_path / "data" / "processed"
    calibration = tmp_path / "data" / "calibration"
    processed.mkdir(parents=True)
    calibration.mkdir(parents=True)
    dates = pd.date_range("2026-01-01", periods=6, freq="D").date
    pd.DataFrame(
        {
            "station_code": ["KATL"],
            "station_name": ["Atlanta/Hartsfield-Jackson Intl"],
            "airport_name": ["Atlanta/Hartsfield-Jackson Intl"],
            "lat": [33.63],
            "lon": [-84.44],
            "timezone": ["America/New_York"],
            "country": ["US"],
        }
    ).to_csv(processed / "station_registry.csv", index=False)
    pd.DataFrame(
        {
            "station_code": ["KATL"] * len(dates),
            "date_local": [d.isoformat() for d in dates],
            "actual_high_f": [60 + i for i in range(len(dates))],
        }
    ).to_csv(processed / "actual_highs.csv", index=False)
    forecast_rows = []
    for d in dates:
        for provider in ["gfs", "hrrr"]:
            forecast_rows.append(
                {
                    "station_id": "KATL",
                    "provider": provider,
                    "model": provider,
                    "contract_date": d.isoformat(),
                    "issued_at": f"{d.isoformat()}T06:00:00+00:00",
                    "raw_forecast_high_f": 60.0,
                    "fetch_status": "ok",
                }
            )
    pd.DataFrame(forecast_rows).to_csv(calibration / "mostlyright_nwp_0h_cache.csv", index=False)
    samples = build_calibration_samples(tmp_path)
    day_six = samples.loc[samples["contract_date"] == "2026-01-06"].sort_values("provider")
    assert list(day_six["station_month_past_actual_mean_f"]) == [62.0, 62.0]


def test_hierarchical_shrinkage_walk_forward_is_finite(tmp_path) -> None:
    _write_minimal_project(tmp_path, days=6)
    samples = build_calibration_samples(tmp_path)
    predictions = walk_forward_baseline_predictions(samples, WalkForwardConfig(min_train_days=3, shrinkage_k=10.0))
    shrinkage = predictions.loc[predictions["method"] == "hierarchical_shrinkage"]
    assert not shrinkage.empty
    assert shrinkage["predicted_calibration_bias_f"].notna().all()


def test_ml_workflow_is_exposed_as_notebook() -> None:
    path = Path(__file__).resolve().parents[1] / "notebooks" / "calibration_ml_walkforward.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )
    assert "walk_forward_ml_predictions" in source
    assert "ml_results.csv" in source
    assert "recommended_calibration_rules.csv" in source
