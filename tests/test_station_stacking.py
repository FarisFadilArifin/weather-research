from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import export_station_stacking_v2_models
from src.calibration import station_stacking
from src.calibration.dataset import CALIBRATION_COLUMNS, WEATHER_NUMERIC_COLUMNS
from src.calibration.modeling import _feature_columns
from src.calibration.station_stacking import (
    OBSERVED_CATEGORICAL_FEATURES,
    OBSERVED_NUMERIC_COLUMNS,
    PROVIDER_NUMERIC_COLUMNS,
    StationStackingConfig,
    build_station_wide_dataset,
    feature_columns,
    load_current_observation_features,
    load_same_day_provider_forecasts,
    missing_expected_model_methods,
    provider_availability,
    raw_baseline_predictions,
    tune_year_split_base_models,
    year_split_feature_importance,
    year_split_test_predictions,
)


FORECAST_AT_AS_OF_COLUMNS = {
    "forecast_temp_at_as_of_f",
    "dewpoint_at_as_of_f",
    "humidity_at_as_of",
    "wind_speed_at_as_of",
    "wind_direction_at_as_of",
}

EXPERIMENTAL_FORECAST_COLUMNS = {
    "cloud_cover_mean",
    "cloud_cover_max",
    "pressure_mslp_mean",
    "pressure_surface_mean",
    "visibility_mean",
    "ceiling_min",
}


def _write_station_stacking_fixture(root: Path, days: int = 8) -> None:
    processed = root / "data" / "processed"
    calibration = root / "data" / "calibration"
    processed.mkdir(parents=True)
    calibration.mkdir(parents=True)
    dates = pd.date_range("2026-01-01", periods=days, freq="D").date
    pd.DataFrame(
        {
            "station_code": ["KATL"],
            "station_name": ["Atlanta/Hartsfield-Jackson Intl"],
            "airport_name": ["Atlanta/Hartsfield-Jackson Intl"],
            "city_label": ["Atlanta"],
            "lat": [33.62972],
            "lon": [-84.44223],
            "timezone": ["America/New_York"],
            "country": ["US"],
        }
    ).to_csv(processed / "station_registry.csv", index=False)
    pd.DataFrame(
        {
            "station_code": ["KATL"] * days,
            "date_local": [day.isoformat() for day in dates],
            "actual_high_f": [70 + i for i in range(days)],
        }
    ).to_csv(processed / "actual_highs.csv", index=False)

    offsets = {"gfs": -1.0, "hrrr": 0.0, "nbm": 1.0}
    for provider, offset in offsets.items():
        cache_dir = calibration / f"sdk_11am_{provider}_fixture"
        cache_file = "sdk_nwp_0h_cache.csv"
        if provider == "nbm":
            cache_dir = calibration / "direct_nbm_fixture"
            cache_file = "direct_nbm_0h_cache.csv"
        cache_dir.mkdir()
        rows = []
        for i, day in enumerate(dates):
            rows.append(
                {
                    "station_id": "KATL",
                    "station_name": "Atlanta/Hartsfield-Jackson Intl",
                    "airport_name": "Atlanta/Hartsfield-Jackson Intl",
                    "provider": provider,
                    "model": provider,
                    "source_label": f"fixture_{provider}",
                    "timing_mode": "same_day_11am",
                    "cycle_selection_policy": "fixture",
                    "contract_date": day.isoformat(),
                    "forecast_as_of": f"{day.isoformat()}T16:00:00+00:00",
                    "issued_at": f"{day.isoformat()}T12:00:00+00:00",
                    "forecast_window_start": f"{day.isoformat()}T16:00:00+00:00",
                    "forecast_window_end": f"{day.isoformat()}T23:00:00+00:00",
                    "horizon_hours": 0,
                    "raw_forecast_high_f": 70 + i + offset,
                    "forecast_hour_min": 4,
                    "forecast_hour_max": 16,
                    "grid_dist_km_mean": 10,
                    "cloud_cover_mean": 40 + i,
                    "cloud_cover_max": 60 + i,
                    "precip_amount": 0.01 * i,
                    "wind_speed_mean": 5 + i,
                    "wind_speed_max": 8 + i,
                    "wind_speed_at_as_of": 6 + i,
                    "wind_direction_mean": 180,
                    "wind_direction_at_as_of": 190,
                    "wind_gust_max": 12 + i,
                    "dewpoint_mean_f": 50,
                    "dewpoint_at_as_of_f": 49,
                    "humidity_mean": 55,
                    "humidity_at_as_of": 56,
                    "data_source": "fixture",
                    "source_file_or_url": "fixture",
                    "fetch_status": "ok",
                    "unavailable_reason": "",
                }
            )
        pd.DataFrame(rows).to_csv(cache_dir / cache_file, index=False)

    obs_dir = calibration / "sdk_current_obs_fixture"
    obs_dir.mkdir()
    obs_rows = []
    for i, day in enumerate(dates):
        obs_rows.append(
            {
                "station_id": "KATL",
                "station_name": "Atlanta/Hartsfield-Jackson Intl",
                "airport_name": "Atlanta/Hartsfield-Jackson Intl",
                "contract_date": day.isoformat(),
                "timing_mode": "same_day_11am",
                "observed_temp_at_as_of_f": 70 + i,
                "observed_dewpoint_at_as_of_f": 60 + i,
                "observed_humidity_at_as_of": 55,
                "observed_wind_speed_at_as_of": 8,
                "observed_wind_direction_at_as_of": 180,
                "observed_wind_gust_at_as_of": 12,
                "observed_peak_wind_gust_at_as_of": 15,
                "observed_peak_wind_direction_at_as_of": 190,
                "observed_peak_wind_time_utc": f"{day.isoformat()}T15:30:00Z",
                "observed_pressure_at_as_of": 1012,
                "observed_pressure_source": "sea_level_pressure_mb",
                "observed_altimeter_inhg_at_as_of": 29.9,
                "observed_sea_level_pressure_mb_at_as_of": 1012,
                "observed_visibility_at_as_of": 2 if i == 0 else 10,
                "observed_ceiling_at_as_of": 1500,
                "observed_cloud_cover_at_as_of": 75,
                "observed_weather_code_at_as_of": "-RA BR" if i == 0 else "",
                "observed_precip_recent_at_as_of": 0.02 if i == 0 else 0,
                "observed_snow_depth_at_as_of": "",
                "observed_as_of_time_local": f"{day.isoformat()}T10:55:00-05:00",
                "observed_as_of_time_utc": f"{day.isoformat()}T15:55:00Z",
                "observed_as_of_age_minutes": 5,
                "observed_source": "fixture",
                "observed_observation_type": "METAR",
                "observed_qc_field": "",
                "observed_raw_metar": "fixture metar",
                "observed_data_source": "fixture",
                "observed_fetch_status": "ok",
                "observed_unavailable_reason": "",
            }
        )
    pd.DataFrame(obs_rows).to_csv(obs_dir / "sdk_current_observations_11am.csv", index=False)


def test_station_wide_features_are_provider_wide_and_lag_safe(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("WEATHER_RESEARCH_INCLUDE_DIRECT_NBM", "1")
    _write_station_stacking_fixture(tmp_path)

    frame = build_station_wide_dataset(tmp_path, station_id="KATL", providers=("gfs", "hrrr", "nbm"))

    assert frame["all_provider_highs_available"].all()
    assert {"gfs_high_f", "hrrr_high_f", "nbm_high_f"}.issubset(frame.columns)
    for provider in ("gfs", "hrrr", "nbm"):
        expected = {
            f"{provider}_{column}"
            for column in PROVIDER_NUMERIC_COLUMNS
            if column != "raw_forecast_high_f"
        }
        assert expected.issubset(frame.columns)
        excluded = {f"{provider}_{column}" for column in FORECAST_AT_AS_OF_COLUMNS}
        assert excluded.isdisjoint(frame.columns)
    assert "gfs_error_f" not in frame.columns
    assert frame.loc[1, "actual_high_lag_1d"] == 70
    assert frame.loc[1, "gfs_error_lag_1d_f"] == 1
    assert frame.loc[1, "provider_mean_high_f"] == 71
    assert frame.loc[1, "gfs_minus_actual_high_lag_1d_f"] == 0
    assert frame.loc[0, "gfs_prior_month_bias_f"] != frame.loc[0, "gfs_prior_month_bias_f"]
    assert frame.loc[1, "gfs_prior_month_bias_f"] != frame.loc[1, "gfs_prior_month_bias_f"]
    assert frame.loc[2, "gfs_prior_month_bias_f"] == 1
    assert frame.loc[2, "gfs_high_plus_prior_month_bias_f"] == frame.loc[2, "actual_high_f"]
    assert frame.loc[0, "gfs_hrrr_high_f_diff_f"] == -1
    assert frame.loc[0, "gfs_hrrr_high_f_abs_diff_f"] == 1
    assert frame.loc[0, "gfs_high_minus_observed_temp_f"] == -1
    assert frame.loc[0, "hrrr_high_minus_observed_temp_f"] == 0


def test_station_wide_features_include_current_observations(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("WEATHER_RESEARCH_INCLUDE_DIRECT_NBM", "1")
    _write_station_stacking_fixture(tmp_path)

    observations = load_current_observation_features(tmp_path, station_id="KATL")
    frame = build_station_wide_dataset(tmp_path, station_id="KATL")
    config = StationStackingConfig(station_id="KATL", project_root=tmp_path)
    categorical, numeric = feature_columns(frame, config)

    assert len(observations) == 8
    assert set(OBSERVED_NUMERIC_COLUMNS).issubset(frame.columns)
    assert frame.loc[0, "observed_dewpoint_depression_f"] == 10
    assert round(frame.loc[0, "observed_wind_dir_sin"], 6) == 0
    assert round(frame.loc[0, "observed_wind_dir_cos"], 6) == -1
    assert frame.loc[0, "observed_is_raining_at_as_of"]
    assert frame.loc[0, "observed_is_fog_or_mist_at_as_of"]
    assert frame.loc[1, "observed_temp_minus_actual_high_lag_1d_f"] == 1
    assert frame.loc[1, "observed_temp_minus_actual_high_roll_7d_mean_f"] == 1
    assert set(OBSERVED_CATEGORICAL_FEATURES).issubset(categorical)
    assert "observed_temp_at_as_of_f" in numeric
    assert "observed_dewpoint_depression_f" in numeric
    assert "observed_raw_metar" not in numeric


def test_station_stacking_loads_direct_nbm_cache(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("WEATHER_RESEARCH_INCLUDE_DIRECT_NBM", "1")
    _write_station_stacking_fixture(tmp_path)

    forecasts = load_same_day_provider_forecasts(tmp_path, providers=("gfs", "hrrr", "nbm"))

    nbm = forecasts.loc[forecasts["provider"].eq("nbm")]
    assert not nbm.empty
    assert set(nbm["source_cache_dir"]) == {"direct_nbm_fixture"}


def test_provider_availability_and_raw_baselines(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("WEATHER_RESEARCH_INCLUDE_DIRECT_NBM", "1")
    _write_station_stacking_fixture(tmp_path)

    availability = provider_availability(tmp_path, providers=("gfs", "hrrr", "nbm"))
    assert set(availability["provider"]) == {"gfs", "hrrr", "nbm"}
    assert set(availability["row_count"]) == {8}

    frame = build_station_wide_dataset(tmp_path, station_id="KATL", providers=("gfs", "hrrr", "nbm"))
    config = StationStackingConfig(
        station_id="KATL",
        project_root=tmp_path,
        providers=("gfs", "hrrr", "nbm"),
        min_train_rows=2,
        refit_days=1,
    )
    predictions = raw_baseline_predictions(frame, config)

    assert {"gfs_raw", "hrrr_raw", "nbm_raw", "provider_mean", "provider_median", "best_raw_provider"}.issubset(
        set(predictions["method"])
    )
    assert predictions.loc[predictions["method"].eq("best_raw_provider"), "contract_date"].min() == "2026-01-03"


def test_missing_expected_model_methods_flags_baseline_only_metrics() -> None:
    metrics = pd.DataFrame(
        {
            "method": ["gfs_raw", "hrrr_raw", "provider_mean", "provider_median"],
            "mae_f": [1.0, 1.1, 0.9, 0.9],
        }
    )

    missing = missing_expected_model_methods(metrics)

    assert missing == ["xgboost", "lightgbm", "catboost", "ridge_stack"]


def test_selected_hyperparameters_uses_rmse_not_mae() -> None:
    tuning = pd.DataFrame(
        [
            {"method": "xgboost", "param_key": "trial_low_mae", "status": "ok", "mae_f": 1.0, "rmse_f": 10.0},
            {"method": "xgboost", "param_key": "trial_low_mae", "status": "ok", "mae_f": 1.2, "rmse_f": 9.0},
            {"method": "xgboost", "param_key": "trial_low_rmse", "status": "ok", "mae_f": 4.0, "rmse_f": 2.0},
            {"method": "xgboost", "param_key": "trial_low_rmse", "status": "ok", "mae_f": 4.2, "rmse_f": 2.5},
        ]
    )

    selected = station_stacking._selected_hyperparameters(tuning)

    assert selected.iloc[0]["param_key"] == "trial_low_rmse"
    assert selected.iloc[0]["mean_validation_rmse_f"] == 2.25


def test_year_split_tuning_stack_scoreboard_and_brackets(monkeypatch) -> None:
    class FakeTrial:
        def __init__(self, number: int):
            self.number = number

        def suggest_float(self, name, low, high, log=False):
            return 1.0

        def suggest_categorical(self, name, choices):
            if "models_plus_raw" in choices:
                return "models_plus_raw"
            if True in choices:
                return True
            return choices[0]

    class FakeStudy:
        def optimize(self, objective, n_trials, show_progress_bar=False, catch=()):
            for number in range(n_trials):
                objective(FakeTrial(number))

    class MeanEstimator:
        def fit(self, x, y):
            self.mean_ = float(pd.Series(y).mean())
            return self

        def predict(self, x):
            return [self.mean_] * len(x)

    def fake_fit_predict_base_model(**kwargs):
        train = kwargs["train"]
        valid = kwargs["valid"]
        predicted = [float(pd.Series(train["actual_high_f"]).mean())] * len(valid)
        metadata = {
            "numeric_features": ",".join(kwargs["numeric"]),
            "categorical_features": ",".join(kwargs["categorical"]),
            "best_iteration": 1,
        }
        return predicted, metadata

    monkeypatch.setattr(station_stacking, "_create_optuna_study", lambda *args, **kwargs: FakeStudy())
    monkeypatch.setattr(station_stacking, "_create_stack_optuna_study", lambda *args, **kwargs: FakeStudy())
    monkeypatch.setattr(station_stacking, "_suggest_hyperparameters", lambda *args, **kwargs: {})
    monkeypatch.setattr(station_stacking, "_fit_predict_base_model", fake_fit_predict_base_model)
    rows = []
    for year in range(2021, 2027):
        for month in range(1, 4):
            actual = 70 + (year - 2021) + month
            rows.append(
                {
                    "contract_date": f"{year}-{month:02d}-01",
                    "actual_high_f": actual,
                    "year": year,
                    "month": month,
                    "day_of_week": "Monday",
                    "gfs_high_f": actual - 1,
                    "hrrr_high_f": actual + 1,
                    "all_provider_highs_available": True,
                    "provider_mean_high_f": actual,
                    "provider_median_high_f": actual,
                    "observed_temp_at_as_of_f": actual - 5,
                }
            )
    frame = pd.DataFrame(rows)
    config = StationStackingConfig(station_id="KATL", providers=("gfs", "hrrr"), fast_mode=True, min_meta_train_rows=2)
    categorical, numeric = ["day_of_week"], ["gfs_high_f", "hrrr_high_f", "observed_temp_at_as_of_f"]

    tuning, validation_predictions, selected = tune_year_split_base_models(frame, config, categorical, numeric)
    baseline_validation = station_stacking.year_split_baseline_predictions(frame, config)
    validation_predictions = pd.concat([baseline_validation, validation_predictions], ignore_index=True)
    test_predictions = year_split_test_predictions(frame, config, categorical, numeric, selected)
    stack_predictions = station_stacking.year_split_stack_test_predictions(validation_predictions, test_predictions, config)
    test_predictions = pd.concat([test_predictions, stack_predictions], ignore_index=True)
    scoreboard = station_stacking.year_split_scoreboard(validation_predictions, test_predictions)
    bracket_predictions = station_stacking.year_split_bracket_predictions(test_predictions)
    bracket_metrics = station_stacking.year_split_bracket_metrics(bracket_predictions)

    assert set(tuning["fold"].dropna()) == {"fold_2021_2023_to_2024", "fold_2022_2024_to_2025"}
    assert set(validation_predictions["contract_date"].str[:4]) == {"2024", "2025"}
    assert set(test_predictions["contract_date"].str[:4]) == {"2026"}
    assert {"xgboost", "lightgbm", "catboost"}.issubset(set(selected["method"]))
    assert set(stack_predictions["method"]) == {"ridge_stack"}
    assert set(scoreboard["method"]) == {"xgboost", "lightgbm", "catboost", "ridge_stack", "hrrr_raw", "gfs_raw"}
    assert set(bracket_metrics["method"]) == {"xgboost", "lightgbm", "catboost", "ridge_stack", "hrrr_raw", "gfs_raw"}


def test_polymarket_temperature_bracket_rounds_half_up_to_two_degree_bins() -> None:
    assert station_stacking.polymarket_temperature_bracket(80) == "80-81"
    assert station_stacking.polymarket_temperature_bracket(81) == "80-81"
    assert station_stacking.polymarket_temperature_bracket(82) == "82-83"
    assert station_stacking.polymarket_temperature_bracket(84.5) == "84-85"
    assert station_stacking.round_temperature_half_up(84.5) == 85


def test_year_split_feature_importance_uses_2026_test(monkeypatch) -> None:
    from sklearn.base import BaseEstimator

    class FirstFeatureEstimator(BaseEstimator):
        def fit(self, x, y):
            return self

        def predict(self, x):
            return pd.to_numeric(x.iloc[:, 0], errors="coerce").fillna(0)

    monkeypatch.setattr(station_stacking, "_build_base_model_pipeline", lambda *args, **kwargs: FirstFeatureEstimator())
    rows = []
    for year in range(2021, 2027):
        for month in range(1, 11):
            driver = float(month * 10)
            rows.append(
                {
                    "contract_date": f"{year}-{month:02d}-01",
                    "actual_high_f": driver,
                    "year": year,
                    "driver_feature": driver,
                    "noise_feature": 1.0,
                }
            )
    frame = pd.DataFrame(rows)
    config = StationStackingConfig(
        station_id="KATL",
        providers=("gfs", "hrrr"),
        fast_mode=True,
        feature_importance_repeats=5,
    )
    selected = pd.DataFrame({"method": ["xgboost"], "param_key": ["trial_0"], "mean_validation_rmse_f": [0.0]})

    importance = year_split_feature_importance(frame, config, [], ["driver_feature", "noise_feature"], selected)

    assert not importance.empty
    assert set(importance["test_year"]) == {2026}
    assert set(importance["train_start_year"]) == {2021}
    assert set(importance["train_end_year"]) == {2025}
    assert importance.iloc[0]["feature"] == "driver_feature"
    assert importance.iloc[0]["importance_mean_mae_f"] > 0


def test_export_station_stacking_v2_model_weights(tmp_path, monkeypatch) -> None:
    from sklearn.dummy import DummyRegressor
    import json
    import joblib

    artifact_dir = tmp_path / "data" / "calibration" / "station_stacking_v2"
    artifact_dir.mkdir(parents=True)
    rows = []
    for year in range(2021, 2027):
        for month in range(1, 3):
            actual = 70 + (year - 2021) + month
            rows.append(
                {
                    "station_id": "KATL",
                    "contract_date": f"{year}-{month:02d}-01",
                    "actual_high_f": actual,
                    "year": year,
                    "month": month,
                    "day_of_week": "Monday",
                    "gfs_high_f": actual - 1,
                    "hrrr_high_f": actual + 1,
                    "provider_mean_high_f": actual,
                    "all_provider_highs_available": True,
                    "observed_temp_at_as_of_f": actual - 4,
                    "v2_recent_heat_anomaly_f": 0.5,
                }
            )
    pd.DataFrame(rows).to_csv(artifact_dir / "KATL_features.csv", index=False)
    pd.DataFrame(
        {
            "method": ["xgboost", "lightgbm", "catboost"],
            "param_key": ["trial_0", "trial_0", "trial_0"],
            "mean_validation_rmse_f": [1.0, 1.1, 1.2],
            "mean_validation_mae_f": [0.8, 0.9, 1.0],
        }
    ).to_csv(artifact_dir / "KATL_year_split_selected_hyperparameters.csv", index=False)

    validation_rows = []
    for date, actual in [("2024-01-01", 74.0), ("2025-01-01", 75.0)]:
        predictions = {
            "xgboost": actual,
            "lightgbm": actual + 0.1,
            "catboost": actual - 0.1,
            "hrrr_raw": actual + 1.0,
            "gfs_raw": actual - 1.0,
        }
        for method, predicted in predictions.items():
            validation_rows.append(
                {
                    "contract_date": date,
                    "method": method,
                    "actual_high_f": actual,
                    "predicted_high_f": predicted,
                }
            )
    pd.DataFrame(validation_rows).to_csv(artifact_dir / "KATL_year_split_validation_predictions.csv", index=False)
    pd.DataFrame(
        {
            "method": ["ridge_stack"],
            "trial_number": [0],
            "param_key": ["stack_trial_0"],
            "feature_set": ["models_plus_raw"],
            "alpha": [1.0],
            "fit_intercept": [True],
            "mae_f": [0.5],
            "rmse_f": [0.6],
            "count": [2],
            "status": ["ok"],
            "error": [""],
        }
    ).to_csv(artifact_dir / "KATL_year_split_stack_tuning.csv", index=False)

    monkeypatch.setattr(
        export_station_stacking_v2_models,
        "_build_base_model_pipeline",
        lambda *args, **kwargs: DummyRegressor(strategy="mean"),
    )

    exported = export_station_stacking_v2_models.export_station_model_weights(
        project_root=tmp_path,
        station_id="KATL",
        artifact_dir=artifact_dir,
    )
    bundle = joblib.load(exported.bundle_path)
    manifest = json.loads(exported.manifest_path.read_text(encoding="utf-8"))

    assert exported.bundle_path.exists()
    assert exported.manifest_path.exists()
    assert set(bundle["base_models"]) == {"xgboost", "lightgbm", "catboost"}
    assert bundle["stack_features"] == [
        "xgboost_predicted_high_f",
        "lightgbm_predicted_high_f",
        "catboost_predicted_high_f",
        "hrrr_raw_predicted_high_f",
        "gfs_raw_predicted_high_f",
    ]
    assert manifest["station_id"] == "KATL"
    assert manifest["training"]["mode"] == "production_refit_all_available_actuals"


def test_forecast_at_as_of_columns_are_not_calibration_features() -> None:
    frame = pd.DataFrame(
        {
            "station_id": ["KATL"],
            "provider": ["gfs"],
            "model": ["gfs"],
            "timing_mode": ["same_day_11am"],
            "month": [1],
            "day_of_week": [1],
            "rain_regime": ["dry"],
            "cloud_regime": ["clear"],
            "raw_forecast_high_f": [72.0],
            **{column: [1.0] for column in FORECAST_AT_AS_OF_COLUMNS},
        }
    )

    _, numeric = _feature_columns(frame)

    assert FORECAST_AT_AS_OF_COLUMNS.isdisjoint(CALIBRATION_COLUMNS)
    assert FORECAST_AT_AS_OF_COLUMNS.isdisjoint(WEATHER_NUMERIC_COLUMNS)
    assert FORECAST_AT_AS_OF_COLUMNS.isdisjoint(PROVIDER_NUMERIC_COLUMNS)
    assert FORECAST_AT_AS_OF_COLUMNS.isdisjoint(numeric)
    assert EXPERIMENTAL_FORECAST_COLUMNS.isdisjoint(CALIBRATION_COLUMNS)
    assert EXPERIMENTAL_FORECAST_COLUMNS.isdisjoint(WEATHER_NUMERIC_COLUMNS)
    assert EXPERIMENTAL_FORECAST_COLUMNS.isdisjoint(PROVIDER_NUMERIC_COLUMNS)
    assert EXPERIMENTAL_FORECAST_COLUMNS.isdisjoint(numeric)
