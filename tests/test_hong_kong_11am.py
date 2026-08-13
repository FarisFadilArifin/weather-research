from __future__ import annotations

import json
import io
import zipfile
from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import src.hong_kong_11am as hong_kong
from src.calibration.station_stacking import (
    OBSERVED_HIGH_SO_FAR_COLUMN,
    TARGET_SOURCE_HKO_DAILY_MAX,
    V20_EXPANDING_FOLDS,
    V20_HKO_GFS_NO_PEAK_FEATURE_VERSION,
    _modeling_frame,
    _optuna_study_name,
    _prediction_output_to_high,
    feature_columns,
)
from src.direct_nwp_fetch import _point_value_bilinear
from src.hong_kong_11am import (
    END_DATE,
    FORECAST_HOURS,
    GFS_ALLOWED_GAP_END_DATE,
    GFS_USABLE_START_DATE,
    MODEL_PROVIDERS,
    OBSERVATION_SOURCE_CONTRACT,
    RESTRICTED_PROVIDERS,
    START_DATE,
    _validate_imported_provider_frame,
    add_celsius_prediction_columns,
    apply_celsius_bucket_metrics_to_summary,
    build_rolling_climatology,
    cleanup_gfs_month_raw,
    celsius_bucket_metrics,
    fahrenheit_to_celsius,
    floor_celsius_bucket,
    forecast_timing,
    hong_kong_stacking_config,
    normalize_iem_metar_csv,
    normalize_hko_open_data_archives,
    parse_hko_daily_extract,
    parse_hko_daily_summary,
    provider_modeling_coverage,
    summarize_gfs_values,
    target_dates,
    write_quote_packets,
)


def test_target_date_contract_has_2027_rows() -> None:
    dates = target_dates()
    assert len(dates) == 2027
    assert dates[0] == START_DATE
    assert dates[-1] == END_DATE


def test_modeling_profile_is_gfs_only_but_restricted_imports_remain_available() -> None:
    assert MODEL_PROVIDERS == ("gfs",)
    assert RESTRICTED_PROVIDERS == ("ifs", "icon")
    assert GFS_ALLOWED_GAP_END_DATE == date(2021, 3, 23)
    assert GFS_USABLE_START_DATE == date(2021, 3, 24)


def test_gfs_coverage_allows_only_the_known_early_gap(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    usable_dates = [day.isoformat() for day in target_dates(GFS_USABLE_START_DATE, END_DATE)]
    frame = pd.DataFrame({"contract_date": usable_dates})
    monkeypatch.setattr(hong_kong, "_load_forecast_provider", lambda _root, _provider: frame.copy())

    coverage = provider_modeling_coverage(tmp_path, "gfs")

    assert coverage["modeling_ready"] is True
    assert coverage["allowed_early_gap_rows"] == 82
    assert coverage["ok_rows"] == len(usable_dates)
    assert coverage["missing_usable_dates"] == []


def test_gfs_coverage_rejects_a_gap_after_usable_start(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    usable_dates = [day.isoformat() for day in target_dates(GFS_USABLE_START_DATE, END_DATE)]
    missing_date = usable_dates[len(usable_dates) // 2]
    frame = pd.DataFrame({"contract_date": [value for value in usable_dates if value != missing_date]})
    monkeypatch.setattr(hong_kong, "_load_forecast_provider", lambda _root, _provider: frame.copy())

    coverage = provider_modeling_coverage(tmp_path, "gfs")

    assert coverage["modeling_ready"] is False
    assert coverage["missing_usable_dates"] == [missing_date]


def test_gfs_coverage_accepts_recovered_rows_inside_the_allowed_early_period(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    usable_dates = [day.isoformat() for day in target_dates(GFS_USABLE_START_DATE, END_DATE)]
    frame = pd.DataFrame({"contract_date": [START_DATE.isoformat(), *usable_dates]})
    monkeypatch.setattr(hong_kong, "_load_forecast_provider", lambda _root, _provider: frame.copy())

    coverage = provider_modeling_coverage(tmp_path, "gfs")

    assert coverage["modeling_ready"] is True
    assert coverage["unexpected_dates"] == []


def test_hko_config_uses_v20_folds_target_and_three_percent_gate(tmp_path) -> None:
    config = hong_kong_stacking_config(tmp_path, tmp_path / "data")
    assert config.providers == ("gfs",)
    assert config.effective_feature_version == V20_HKO_GFS_NO_PEAK_FEATURE_VERSION
    assert config.effective_target_source == TARGET_SOURCE_HKO_DAILY_MAX
    assert config.effective_year_split_folds == V20_EXPANDING_FOLDS
    assert config.effective_year_split_validation_weights == {
        2022: 1.0,
        2023: 1.0,
        2024: 1.0,
        2025: 1.0,
    }
    assert config.effective_max_feature_missing_fraction == pytest.approx(0.03)
    assert config.observation_target_same_station is True
    assert config.observation_source == OBSERVATION_SOURCE_CONTRACT
    assert _optuna_study_name(config, stage="base", method="xgboost").endswith(
        "_obs_hko_open_data_archive_1min"
    )


def test_hko_same_station_observation_filters_impossible_target_and_clamps_prediction(tmp_path) -> None:
    config = hong_kong_stacking_config(tmp_path, tmp_path / "data")
    frame = pd.DataFrame(
        {
            "contract_date": ["2026-07-20"],
            "year": [2026],
            "actual_high_f": [80.0],
            OBSERVED_HIGH_SO_FAR_COLUMN: [83.0],
            "observed_temp_at_as_of_f": [82.0],
            "observed_fetch_status": ["ok"],
            "observed_as_of_age_minutes": [0.0],
            "actual_data_quality_flag": ["ok"],
            "gfs_high_f": [84.0],
            "gfs_forecast_temp_at_as_of_f": [82.5],
            "all_provider_highs_available": [True],
        }
    )

    modeling_frame, _, _ = _modeling_frame(frame, config)

    assert modeling_frame.empty
    predicted = _prediction_output_to_high(np.array([-3.0]), frame, config)
    assert predicted.tolist() == pytest.approx([83.0])


def _snapshot_zip(filename: str, content: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(filename, content)
    return buffer.getvalue()


def test_hko_daily_archives_merge_into_month_parser_input() -> None:
    first = _snapshot_zip(
        "archive/20260701-1100-latest_1min_temperature.csv",
        "Date time,Automatic Weather Station,Air Temperature(degree Celsius)\n"
        "202607011100,HK Observatory,31.0\n",
    )
    second = _snapshot_zip(
        "archive/20260702-1100-latest_1min_temperature.csv",
        "Date time,Automatic Weather Station,Air Temperature(degree Celsius)\n"
        "202607021100,HK Observatory,32.0\n",
    )

    assert hong_kong._historical_archive_snapshot_dates(first) == {date(2026, 7, 1)}
    merged = hong_kong._merge_historical_archive_zips([first, second])
    parsed = hong_kong.parse_hko_historical_archive_zip(merged)

    assert parsed["observed_at_local"].dt.date.tolist() == [
        date(2026, 7, 1),
        date(2026, 7, 2),
    ]


def test_hko_open_data_archive_uses_latest_safe_headquarters_snapshots() -> None:
    temperature = _snapshot_zip(
        "temperature.csv",
        """Date time,Automatic Weather Station,Air Temperature(degree Celsius)
202607200900,HK Observatory,29.0
202607201000,HK Observatory,30.0
202607201100,HK Observatory,31.0
202607201110,HK Observatory,32.0
""",
    )
    maxmin = _snapshot_zip(
        "maxmin.csv",
        """Date time,Automatic Weather Station,MaximumAir Temperature Since Midnight(degree Celsius),Minimum Air Temperature Since Midnight(degree Celsius)
202607200900,HK Observatory,29.2,25.0
202607201100,HK Observatory,31.2,25.0
202607201110,HK Observatory,32.1,25.0
""",
    )
    humidity = _snapshot_zip(
        "humidity.csv",
        """Date time,Automatic Weather Station,Relative Humidity(percent)
202607201100,HK Observatory,70
202607201110,HK Observatory,68
""",
    )

    frame = normalize_hko_open_data_archives(
        temperature,
        maxmin,
        humidity,
        [date(2026, 7, 20)],
    )

    row = frame.iloc[0]
    assert row["station_id"] == "HKO"
    assert row["observed_data_source"] == OBSERVATION_SOURCE_CONTRACT
    assert row["observed_temp_at_as_of_f"] == pytest.approx(87.8)
    assert row["observed_high_temp_through_as_of_f"] == pytest.approx(88.16)
    assert row["observed_humidity_at_as_of"] == pytest.approx(70.0)
    assert row["observed_high_so_far_change_since_9am_f"] == pytest.approx(3.6)
    assert row["observed_as_of_age_minutes"] == pytest.approx(0.0)
    assert row["observed_fetch_status"] == "ok"


def test_hko_feature_contract_prunes_single_provider_degenerates(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "contract_date": ["2026-07-20"],
            "actual_high_f": [86.0],
            "actual_high_c": [30.0],
            "gfs_high_f": [86.5],
            "provider_mean_high_f": [86.5],
            "provider_spread_high_f": [0.0],
            "provider_std_high_f": [np.nan],
            "gfs_rank_high": [1.0],
            "v11sf_forecast_temp_11am_mean_f": [84.0],
            "v11sf_forecast_temp_11am_median_f": [84.0],
            "v11sf_forecast_temp_11am_minus_observed_f": [-0.5],
            "v11sf_forecast_temp_11am_spread_f": [0.0],
            "v11sf_forecast_temp_11am_provider_count": [1],
        }
    )
    config = hong_kong_stacking_config(tmp_path, tmp_path / "data")

    _, numeric = feature_columns(frame, config)

    assert "gfs_high_f" in numeric
    assert "v11sf_forecast_temp_11am_mean_f" in numeric
    assert "v11sf_forecast_temp_11am_minus_observed_f" in numeric
    assert "actual_high_c" not in numeric
    assert "provider_mean_high_f" not in numeric
    assert "provider_spread_high_f" not in numeric
    assert "provider_std_high_f" not in numeric
    assert "gfs_rank_high" not in numeric
    assert "v11sf_forecast_temp_11am_median_f" not in numeric
    assert "v11sf_forecast_temp_11am_spread_f" not in numeric
    assert "v11sf_forecast_temp_11am_provider_count" not in numeric


def test_celsius_reporting_uses_floor_buckets_and_lower_two_bucket_pair() -> None:
    assert fahrenheit_to_celsius(86.0) == pytest.approx(30.0)
    assert floor_celsius_bucket(31.2) == 31
    assert floor_celsius_bucket(32.6) == 32
    assert floor_celsius_bucket(-1.5) == -2
    predictions = pd.DataFrame(
        {
            "method": ["gfs_raw", "ridge_stack"],
            "actual_high_f": [86.0, 86.9],
            "predicted_high_f": [86.9, 86.0],
            "error_f": [0.9, -0.9],
            "absolute_error_f": [0.9, 0.9],
        }
    )

    dual = add_celsius_prediction_columns(predictions)

    assert dual["actual_high_c"].tolist() == pytest.approx([30.0, 30.5])
    assert dual["predicted_high_c"].tolist() == pytest.approx([30.5, 30.0])
    assert dual["actual_bucket_c"].tolist() == [30, 30]
    assert dual["predicted_bucket_c"].tolist() == [30, 30]
    assert dual["two_bucket_lower_c"].tolist() == [29, 29]
    assert dual["two_bucket_upper_c"].tolist() == [30, 30]
    assert dual["two_bucket_hit_c"].tolist() == [True, True]

    bucket_metrics = celsius_bucket_metrics(predictions)
    assert bucket_metrics["bucket_unit"].eq("celsius").all()
    assert bucket_metrics["bucket_width_c"].eq(1.0).all()
    assert bucket_metrics["rounding_rule"].eq("floor_integer_celsius").all()
    assert bucket_metrics["two_bucket_accuracy"].eq(1.0).all()

    inherited_summary = pd.DataFrame(
        {
            "evaluation_scope": ["year_split_test", "year_split_test"],
            "method": ["gfs_raw", "ridge_stack"],
            "mae_f": [0.9, 0.9],
            "bucket_log_loss": [1.2, 1.3],
            "bucket_accuracy_pct": [50.0, 50.0],
        }
    )
    summary = apply_celsius_bucket_metrics_to_summary(inherited_summary, predictions.assign(evaluation_scope="year_split_test"))
    assert "bucket_log_loss" not in summary
    assert "bucket_accuracy_pct" not in summary
    assert summary["bucket_width_c"].eq(1.0).all()
    assert "exact_bucket_accuracy_pct" in summary


def test_forecast_timing_is_previous_18z_and_f009_to_f021() -> None:
    timing = forecast_timing("2026-07-20")
    assert timing["issue_utc"] == datetime(2026, 7, 19, 18, tzinfo=UTC)
    assert timing["as_of_utc"] == datetime(2026, 7, 20, 3, tzinfo=UTC)
    assert timing["window_end_utc"] == datetime(2026, 7, 20, 16, tzinfo=UTC)
    assert timing["forecast_hours"] == tuple(range(9, 22))


def test_parse_hko_daily_extract_uses_absolute_daily_max_column() -> None:
    payload = {
        "stn": {
            "data": [
                {
                    "month": 1,
                    "dayData": [
                        ["01", "1025.5", "15.0", "11.8", "8.6"],
                        ["02", "1022.9", "17.8", "14.0", "10.4"],
                        ["Mean/Total", "1024.2", "16.4", "12.9", "9.5"],
                    ],
                }
            ]
        }
    }
    frame = parse_hko_daily_extract(payload, 2021)
    assert frame["contract_date"].tolist() == ["2021-01-01", "2021-01-02"]
    assert frame["actual_high_c"].tolist() == [15.0, 17.8]
    assert frame.loc[0, "actual_high_f"] == pytest.approx(59.0)


def test_hko_daily_summary_parser_uses_hko_max_and_validates_date() -> None:
    payload = {
        "DYN_DAT_MINDS_RYES": {
            "ReportTimeInfoDate": {"Val_Eng": "20260720"},
            "HKOReadingsMaxTemp": {"Val_Eng": "29.7"},
        }
    }
    row = parse_hko_daily_summary(payload, date(2026, 7, 20))
    assert row["actual_high_c"] == pytest.approx(29.7)
    assert row["actual_high_f"] == pytest.approx(85.46)
    assert row["actual_source"].endswith("provisional")
    with pytest.raises(ValueError, match="date mismatch"):
        parse_hko_daily_summary(payload, date(2026, 7, 19))


def test_iem_metar_keeps_rain_code_but_never_uses_non_us_rain_amount() -> None:
    csv = """station,valid,tmpf,dwpf,drct,sknt,p01i,alti,mslp,vsby,gust,skyc1,skyc2,skyc3,skyc4,skyl1,skyl2,skyl3,skyl4,wxcodes,peak_wind_gust,peak_wind_drct,peak_wind_time,metar
VHHH,2021-07-20 09:00,82.4,77.0,180,8,0.00,29.70,null,6.21,null,FEW,SCT,null,null,1000,2500,null,null,null,null,null,null,VHHH 200100Z 18008KT 9999 FEW010 SCT025 28/25 Q1006
VHHH,2021-07-20 10:00,84.2,77.0,190,9,0.00,29.68,null,4.00,15,FEW,BKN,null,null,800,1800,null,null,-SHRA,null,null,null,VHHH 200200Z 19009G15KT 6000 -SHRA FEW008 BKN018 29/25 Q1005
VHHH,2021-07-20 11:00,86.0,78.8,200,10,0.00,29.65,null,3.00,18,FEW,BKN,null,null,800,1500,null,null,SHRA,null,null,null,VHHH 200300Z 20010G18KT 4800 SHRA FEW008 BKN015 30/26 Q1004
"""
    frame = normalize_iem_metar_csv(csv.encode(), [date(2021, 7, 20)])
    row = frame.iloc[0]
    assert row["observed_fetch_status"] == "ok"
    assert row["observed_temp_at_as_of_f"] == pytest.approx(86.0)
    assert row["observed_high_temp_through_as_of_f"] == pytest.approx(86.0)
    assert row["observed_weather_code_at_as_of"] == "SHRA"
    assert pd.isna(row["observed_precip_recent_at_as_of"])
    assert not bool(row["observed_precip_amount_available"])
    assert row["observed_temp_change_last_1h_f"] == pytest.approx(1.8)


def test_iem_metar_rejects_observation_older_than_60_minutes() -> None:
    csv = """station,valid,tmpf,dwpf,drct,sknt,p01i,alti,mslp,vsby,gust,skyc1,skyc2,skyc3,skyc4,skyl1,skyl2,skyl3,skyl4,wxcodes,peak_wind_gust,peak_wind_drct,peak_wind_time,metar
VHHH,2021-07-20 09:30,82.4,77.0,180,8,0.00,29.70,null,6.21,null,FEW,null,null,null,1000,null,null,null,null,null,null,null,VHHH 200130Z 18008KT 9999 FEW010 28/25 Q1006
"""
    frame = normalize_iem_metar_csv(csv.encode(), [date(2021, 7, 20)])
    assert frame.loc[0, "observed_fetch_status"] == "unavailable"
    assert "60 minutes" in frame.loc[0, "observed_unavailable_reason"]


def test_gfs_summary_uses_hko_high_and_hko_11am_temperature() -> None:
    hko = {}
    for hour in FORECAST_HOURS:
        hko[hour] = {
            "temp_k_2m": 300.0 + (hour - 9) * 0.1,
            "dewpoint_k_2m": 294.0,
            "relative_humidity_pct_2m": 75.0,
            "wind_speed_ms_10m": 4.0,
            "wind_direction_deg_10m": 180.0,
            "wind_gust_ms": 6.0,
            "precip_mm_1h": 0.5,
            "cloud_cover_pct": 60.0,
        }
    row = summarize_gfs_values(date(2026, 7, 20), {"HKO": hko})
    assert row["forecast_temp_at_as_of_f"] == pytest.approx((300.0 - 273.15) * 9 / 5 + 32)
    assert row["raw_forecast_high_f"] == pytest.approx((301.2 - 273.15) * 9 / 5 + 32)
    assert row["forecast_precip_total_mm"] == pytest.approx(6.5)
    assert row["forecast_hour_count_returned"] == 13
    assert row["fetch_status"] == "ok"


def test_gfs_month_cleanup_deletes_only_issues_owned_by_that_target_month(tmp_path) -> None:
    raw = tmp_path / "raw/nwp_subsets/gfs"
    raw.mkdir(parents=True)
    owned = [
        raw / "gfs_temp_k_2m_2020123118_f009.grib2",
        raw / ".gfs_cloud_cover_pct_2021013018_f021.grib2.123.tmp",
    ]
    preserved = [
        raw / "gfs_temp_k_2m_2021013118_f009.grib2",
        raw / "unrelated.bin",
    ]
    for path in [*owned, *preserved]:
        path.write_bytes(b"raw")

    result = cleanup_gfs_month_raw(tmp_path, "2021-01")

    assert result["status"] == "complete"
    assert result["deleted_files"] == 2
    assert result["deleted_bytes"] == 6
    assert all(not path.exists() for path in owned)
    assert all(path.exists() for path in preserved)


def test_bilinear_interpolation_uses_four_regular_grid_cells() -> None:
    dataset = xr.Dataset(
        {"t2m": (("latitude", "longitude"), np.array([[0.0, 2.0], [2.0, 4.0]]))},
        coords={"latitude": [1.0, 0.0], "longitude": [0.0, 1.0]},
    )
    assert _point_value_bilinear(dataset, "t2m", 0.5, 0.5) == pytest.approx(2.0)


def test_restricted_import_rejects_wrong_cycle() -> None:
    frame = pd.DataFrame(
        [
            {
                "contract_date": "2026-07-20",
                "provider": "ifs",
                "issued_at": "2026-07-20T00:00:00Z",
                "forecast_as_of": "2026-07-20T03:00:00Z",
                "forecast_hour_min": 9,
                "forecast_hour_max": 21,
                "raw_forecast_high_f": 90.0,
                "forecast_temp_at_as_of_f": 84.0,
                "fetch_status": "ok",
            }
        ]
    )
    with pytest.raises(ValueError, match="timing mismatch"):
        _validate_imported_provider_frame(frame, "ifs")


def test_rolling_climatology_never_uses_target_or_future_year() -> None:
    labels = pd.DataFrame(
        {
            "contract_date": [f"{year}-07-20" for year in range(2011, 2022)],
            "actual_high_f": [float(year) for year in range(2011, 2022)],
        }
    )
    normals = build_rolling_climatology(labels)
    row = normals.loc[normals["target_year"].eq(2021)].iloc[0]
    assert row["climatology_high_10y_count"] == 10
    assert row["climatology_high_10y_f"] == pytest.approx(np.mean(range(2011, 2021)))
    assert row["climatology_source_end_year"] == 2020


def test_quote_stage_records_zero_spend_authority(tmp_path) -> None:
    paths = write_quote_packets(tmp_path)
    assert all(path.exists() for path in paths.values())
    access = json.loads((tmp_path / "access/provider_access.json").read_text(encoding="utf-8"))
    assert access["spend_authorized"] is False
    assert access["usage_class"] == "personal_research"
