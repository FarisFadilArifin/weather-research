from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from src.calibration.asia_station_stacking import (
    ASIA_PROVIDERS,
    asia_expanding_folds,
    build_asia_station_wide_dataset,
)
from src.calibration.station_stacking import (
    POINT_IN_TIME_UNSAFE_FEATURE_COLUMNS,
    StationStackingConfig,
    V20_ASIA_NO_PEAK_FEATURE_VERSION,
    feature_columns,
    run_station_year_split_experiment,
)


def _write_fixture(root: Path, city: str, station: str) -> None:
    base = root / "normalized"
    settlement_dir = base / "settlements" / city
    observation_dir = base / "observations" / city
    gfs_dir = base / "forecasts" / "gfs" / city
    gefs_dir = base / "forecasts" / "gefs_ensemble_daily" / city
    jma_dir = base / "forecasts" / "jma_msm_previous_day1" / city
    for directory in (settlement_dir, observation_dir, gfs_dir, gefs_dir, jma_dir):
        directory.mkdir(parents=True, exist_ok=True)

    dates = ["2022-07-03", "2022-07-04"]
    pd.DataFrame(
        [
            {"station_id": station, "contract_date": dates[0], "settlement_high_c": 30.0, "settlement_high_f": 86.0, "settlement_source": "wunderground_station_history", "quality_flag": "ok"},
            {"station_id": station, "contract_date": dates[1], "settlement_high_c": 31.0, "settlement_high_f": 87.8, "settlement_source": "wunderground_station_history", "quality_flag": "ok"},
            {"station_id": station, "contract_date": dates[1], "settlement_high_c": 32.0, "settlement_high_f": 89.6, "settlement_source": "wunderground_station_history", "quality_flag": "ok"},
        ]
    ).to_parquet(settlement_dir / "2022-07.parquet", index=False)
    pd.DataFrame(
        [
            {
                "station_id": station,
                "contract_date": day,
                "observed_temp_at_as_of_f": 77.0,
                "observed_high_temp_through_as_of_f": 78.0,
                "observed_dewpoint_at_as_of_f": 66.0,
                "observed_humidity_at_as_of": 60.0,
                "observed_wind_speed_at_as_of": 5.0,
                "observed_wind_direction_at_as_of": 180.0,
                "observed_as_of_age_minutes": 0.0,
                "observed_fetch_status": "ok",
                "observed_as_of_time_local": f"{day}T11:00:00+09:00",
            }
            for day in dates
        ]
    ).to_parquet(observation_dir / "2022-07.parquet", index=False)
    hourly = []
    for day in dates:
        for hour, temp in ((8, 25.0), (20, 28.0)):
            hourly.append(
                {
                    "station_id": station,
                    "contract_date": day,
                    "forecast_hour": hour,
                    "temp_c_2m": temp,
                    "dewpoint_c_2m": 18.0,
                    "relative_humidity_pct_2m": 60.0,
                    "wind_speed_ms_10m": 2.0,
                    "wind_direction_deg_10m": 180.0,
                    "wind_gust_ms": 3.0,
                    "precip_mm_1h": 1.0 if hour == 8 else 2.0,
                    "cloud_cover_pct": 20.0,
                    "forecast_as_of_utc": f"{day}T02:00:00Z",
                    "issued_at_utc": f"{day}T00:00:00Z",
                    "lineage": "gfs_fixture",
                    "source_url": "fixture",
                    "fetch_status": "ok",
                }
            )
    pd.DataFrame(hourly).to_parquet(gfs_dir / "2022-07.parquet", index=False)
    pd.DataFrame(
        [
            {
                "station_id": station,
                "contract_date": day,
                "gefs_mean_high_c": 27.0,
                "gefs_std_high_c": 1.0,
                "gefs_p50_c": 27.2,
                "gefs_spread_c": 3.0,
                "fetch_status": "ok",
            }
            for day in dates
        ]
    ).to_parquet(gefs_dir / "2022-07.parquet", index=False)
    jma = []
    for day in dates:
        for hour, temp in ((11, 24.0), (23, 28.0)):
            jma.append(
                {
                    "station_id": station,
                    "contract_date": day,
                    "forecast_hour_local": hour,
                    "temp_2m_c": temp,
                    "dewpoint_2m_c": 18.0,
                    "relative_humidity_2m_pct": 60.0,
                    "wind_speed_10m_kmh": 8.0,
                    "wind_direction_10m_deg": 180.0,
                    "wind_gusts_10m_kmh": 12.0,
                    "precipitation_mm": 0.5 if hour == 11 else 1.5,
                    "cloud_cover_pct": 20.0,
                    "forecast_as_of_utc": f"{day}T02:00:00Z",
                    "issued_at_utc": f"{day}T00:00:00Z",
                    "lineage": "jma_fixture",
                    "source_url": "fixture",
                    "fetch_status": "ok",
                }
            )
    pd.DataFrame(jma).to_parquet(jma_dir / "2022-07.parquet", index=False)


def test_asia_builder_converts_and_deduplicates_fixture(tmp_path: Path) -> None:
    _write_fixture(tmp_path, "tokyo", "RJTT")
    frame = build_asia_station_wide_dataset(tmp_path, "tokyo")

    assert frame["contract_date"].tolist() == ["2022-07-03", "2022-07-04"]
    assert frame.loc[frame["contract_date"].eq("2022-07-04"), "actual_high_f"].iloc[0] == 89.6
    assert frame.loc[frame["contract_date"].eq("2022-07-04"), "actual_high_c"].iloc[0] == 32.0
    assert frame["actual_high_c_source"].eq("settlement_high_c").all()
    assert frame.loc[frame["contract_date"].eq("2022-07-03"), "gfs_high_f"].iloc[0] == 82.4
    assert frame.loc[frame["contract_date"].eq("2022-07-03"), "jma_msm_high_f"].iloc[0] == 82.4
    row = frame.loc[frame["contract_date"].eq("2022-07-03")].iloc[0]
    expected_optional = {
        "gfs_dewpoint_mean_f": 64.4,
        "gfs_humidity_mean": 60.0,
        "gfs_wind_speed_mean": 2.0,
        "gfs_wind_speed_max": 2.0,
        "gfs_wind_direction_mean": 180.0,
        "gfs_wind_gust_max": 3.0,
        "gfs_forecast_precip_total_mm": 3.0,
        "gfs_forecast_precip_max_1h_mm": 2.0,
        "gfs_forecast_precip_hours_count": 2.0,
        "gfs_cloud_cover_mean": 20.0,
        "gfs_cloud_cover_max": 20.0,
        "jma_msm_dewpoint_mean_f": 64.4,
        "jma_msm_humidity_mean": 60.0,
        "jma_msm_wind_speed_mean": 8.0,
        "jma_msm_wind_speed_max": 8.0,
        "jma_msm_wind_direction_mean": 180.0,
        "jma_msm_wind_gust_max": 12.0,
        "jma_msm_forecast_precip_total_mm": 2.0,
        "jma_msm_forecast_precip_max_1h_mm": 1.5,
        "jma_msm_forecast_precip_hours_count": 2.0,
        "jma_msm_cloud_cover_mean": 20.0,
        "jma_msm_cloud_cover_max": 20.0,
    }
    for column, expected in expected_optional.items():
        assert row[column] == pytest.approx(expected), column
    assert set(frame["station_id"]) == {"RJTT"}
    assert frame["strict_quality_ok"].all()


def test_asia_builder_rejects_physically_invalid_optional_provider_fields(
    tmp_path: Path,
) -> None:
    _write_fixture(tmp_path, "tokyo", "RJTT")
    gfs_path = (
        tmp_path
        / "normalized"
        / "forecasts"
        / "gfs"
        / "tokyo"
        / "2022-07.parquet"
    )
    gfs = pd.read_parquet(gfs_path)
    gfs.loc[0, "relative_humidity_pct_2m"] = 150.0
    gfs.to_parquet(gfs_path, index=False)

    with pytest.raises(
        ValueError,
        match="provider_optional_field_out_of_bounds:gfs:relative_humidity_pct_2m:percentage",
    ):
        build_asia_station_wide_dataset(tmp_path, "tokyo")


def test_asia_folds_start_with_available_history() -> None:
    folds = asia_expanding_folds()
    assert [fold.validation_year for fold in folds] == [2023, 2024, 2025]
    assert all(fold.train_start_year == 2022 for fold in folds)
    assert all(fold.validation_year != 2022 for fold in folds)


def test_asia_point_model_excludes_final_same_day_iem_highs() -> None:
    frame = pd.DataFrame(
        {
            "actual_high_f": [86.0],
            "observed_high_temp_through_as_of_f": [80.0],
            "iem_daily_high_f": [86.0],
            "iem_daily_high_c": [30.0],
        }
    )
    config = StationStackingConfig(
        station_id="RJTT",
        providers=ASIA_PROVIDERS,
        feature_version=V20_ASIA_NO_PEAK_FEATURE_VERSION,
        training_profile="v20_aligned",
        target_mode="remaining_warmup",
        target_source="wunderground_only",
    )

    categorical, numeric = feature_columns(frame, config)
    selected = set(categorical) | set(numeric)

    assert "observed_high_temp_through_as_of_f" in selected
    assert "actual_high_c" not in selected
    assert "settlement_high_c" not in selected
    assert selected.isdisjoint(POINT_IN_TIME_UNSAFE_FEATURE_COLUMNS)


def test_asia_reduced_year_split_smoke_runs_without_network(tmp_path: Path) -> None:
    dates = pd.date_range("2022-01-01", "2026-01-10", freq="14D")
    rows = []
    for index, stamp in enumerate(dates):
        high = 80.0 + (index % 5)
        row = {
            "contract_date": stamp.strftime("%Y-%m-%d"),
            "station_id": "RJTT",
            "actual_high_f": high,
            "observed_high_temp_through_as_of_f": high - 2.0,
            "observed_temp_at_as_of_f": high - 3.0,
            "observed_fetch_status": "ok",
            "observed_as_of_age_minutes": 0.0,
            "all_provider_highs_available": True,
            "strict_quality_ok": True,
            "year": stamp.year,
            "month": stamp.month,
        }
        for offset, provider in enumerate(ASIA_PROVIDERS):
            row[f"{provider}_high_f"] = high - 1.0 + offset * 0.2
            row[f"{provider}_forecast_temp_at_as_of_f"] = high - 3.0 + offset * 0.1
        rows.append(row)
    features = pd.DataFrame(rows)
    config = StationStackingConfig(
        station_id="RJTT",
        providers=ASIA_PROVIDERS,
        prebuilt_features=features,
        feature_version=V20_ASIA_NO_PEAK_FEATURE_VERSION,
        training_profile="v20_aligned",
        target_mode="remaining_warmup",
        target_source="wunderground_only",
        base_model_methods=("xgboost",),
        stack_enabled=False,
        fast_mode=True,
        min_train_rows=1,
        optuna_trials=1,
        year_split_folds=asia_expanding_folds(),
        year_split_validation_weights={2023: 1.0, 2024: 1.0, 2025: 1.0},
        year_split_test_train_years=(2022, 2025),
        year_split_test_year=2026,
        output_dir=tmp_path / "artifacts",
    )
    result = run_station_year_split_experiment(config)
    assert not result.features.empty
    assert not result.test_predictions.empty
    assert set(result.test_predictions["contract_date"].astype(str).str[:4]) == {"2026"}


def test_asia_notebook_generator_matches_committed_outputs() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook_root = (
        root
        / "notebooks"
        / "experiments"
        / "station_stacking_v20_asia_no_peak"
    )
    generator_path = notebook_root / "generate_city_notebooks.py"
    spec = importlib.util.spec_from_file_location("asia_notebook_generator", generator_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def source_contract(notebook: dict) -> dict:
        return {
            "nbformat": notebook["nbformat"],
            "nbformat_minor": notebook["nbformat_minor"],
            "cells": [
                {
                    "cell_type": cell["cell_type"],
                    "source": cell.get("source", []),
                }
                for cell in notebook["cells"]
            ],
        }

    for city in ("tokyo", "seoul"):
        generated = module._notebook(city)
        stored = json.loads((notebook_root / f"stacking_{city.title()}_v20_no_peak.ipynb").read_text(encoding="utf-8"))
        assert source_contract(generated) == source_contract(stored)
        source = "".join("".join(cell.get("source", [])) for cell in stored["cells"])
        assert f'CITY_ID = "{city}"' in source
        assert "v20_asia_no_peak" in source
        assert 'PROVIDERS = ("gfs", "gefs", "jma_msm")' in source
        assert "V20_PEAK_TIMING" not in source
        assert "hrrr" not in source.lower()
        assert "nbm" not in source.lower()
        assert all(
            cell.get("execution_count") is None
            for cell in generated["cells"]
            if cell["cell_type"] == "code"
        )
        assert all(
            not cell.get("outputs")
            for cell in generated["cells"]
            if cell["cell_type"] == "code"
        )
