from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.calibration.station_stacking import (
    OBSERVED_HIGH_SO_FAR_COLUMN,
    StationStackingConfig,
    _add_ensemble_features,
    _add_forecast_history_delta_features,
    _add_forecast_shape_features,
    _add_lagged_provider_error_features,
    _add_observation_forecast_delta_features,
    _add_prior_month_provider_error_features,
    _add_provider_availability_features,
    _add_provider_cross_model_features,
    _build_base_model_pipeline,
    _fit_feature_columns,
    _model_target_values,
    _modeling_frame,
    _params_from_selected_row,
    _round_half_up_series,
    _temperature_bracket_from_rounded,
    add_versioned_feature_engineering,
    build_station_wide_dataset,
)
from src.export_station_stacking_v2_models import _fit_stack_model


START_DATE = "2026-07-03"
END_DATE = "2026-07-14"
PROVIDERS = ("gfs", "hrrr", "nbm")


def _load_bundle(path: str) -> dict:
    return joblib.load(ROOT / path)


def _fit_kdal_v20_bundle() -> dict:
    artifact_dir = ROOT / "data/calibration/station_stacking_v20_peak_timing"
    features = pd.read_csv(artifact_dir / "KDAL_features.csv", low_memory=False)
    selected = pd.read_csv(artifact_dir / "KDAL_year_split_selected_hyperparameters.csv")
    validation = pd.read_csv(artifact_dir / "KDAL_year_split_validation_predictions.csv")
    stack_tuning = pd.read_csv(artifact_dir / "KDAL_year_split_stack_tuning.csv")
    config = StationStackingConfig(
        station_id="KDAL",
        project_root=ROOT,
        timing_mode="same_day_11am_live_safe",
        providers=PROVIDERS,
        feature_version="v20_peak_timing",
        optuna_metric="mae_f",
        target_mode="remaining_warmup",
        target_source="wunderground_only",
        base_model_methods=("xgboost", "lightgbm", "catboost"),
        stack_enabled=True,
    )
    modeling, categorical, numeric = _modeling_frame(features, config)
    train = modeling.copy()
    fit_categorical, fit_numeric = _fit_feature_columns(train, categorical, numeric)
    feature_names = [*fit_categorical, *fit_numeric]
    selected_by_method = {str(row.method): row for _, row in selected.iterrows()}
    base_models = {}
    for method in config.effective_base_model_methods:
        params = _params_from_selected_row(selected_by_method[method])
        estimator = _build_base_model_pipeline(config, fit_categorical, fit_numeric, method, params)
        estimator.fit(train[feature_names], _model_target_values(train, config))
        base_models[method] = estimator
    stack_model, stack_manifest = _fit_stack_model(
        validation,
        stack_tuning,
        metric_col="mae_f",
        base_model_methods=config.effective_base_model_methods,
        providers=PROVIDERS,
    )
    return {
        "station_id": "KDAL",
        "feature_version": "v20_peak_timing",
        "target_mode": "remaining_warmup",
        "base_model_methods": config.effective_base_model_methods,
        "base_models": base_models,
        "stack_model": stack_model,
        "stack_features": stack_manifest["features"],
        "feature_names": feature_names,
    }


def _enriched_station_frame(station: str) -> pd.DataFrame:
    os.environ["WEATHER_RESEARCH_INCLUDE_DIRECT_NBM"] = "1"
    frame = build_station_wide_dataset(
        ROOT,
        station_id=station,
        timing_mode="same_day_11am_live_safe",
        providers=PROVIDERS,
        feature_version="v20_peak_timing",
        target_source="wunderground_only",
    )
    in_window = frame["contract_date"].astype(str).between(START_DATE, END_DATE)
    # Peak timing uses the same remaining-day curves and point-in-time cutoff as
    # the standard provider-high fields. Fill only archive rows not recovered by
    # the direct provider pull.
    frame.loc[in_window, "hrrr_high_f"] = frame.loc[in_window, "hrrr_high_f"].fillna(
        frame.loc[in_window, "hrrr_max_post11_f"]
    )
    frame.loc[in_window, "hrrr_forecast_temp_at_as_of_f"] = frame.loc[
        in_window, "hrrr_forecast_temp_at_as_of_f"
    ].fillna(frame.loc[in_window, "hrrr_t11l_f"])
    frame.loc[in_window, "nbm_high_f"] = frame.loc[in_window, "nbm_high_f"].fillna(
        frame.loc[in_window, "nbm_max_post11_f"]
    )
    frame.loc[in_window, "nbm_forecast_temp_at_as_of_f"] = frame.loc[
        in_window, "nbm_forecast_temp_at_as_of_f"
    ].fillna(frame.loc[in_window, "nbm_t11l_f"])

    # Recompute all provider-derived features affected by the filled high/11 AM
    # values. Historical features are recomputed on the full date spine.
    frame = _add_provider_availability_features(frame, PROVIDERS)
    frame = _add_ensemble_features(frame, PROVIDERS)
    frame = _add_forecast_shape_features(frame, PROVIDERS)
    frame = _add_provider_cross_model_features(frame, PROVIDERS)
    frame = _add_lagged_provider_error_features(frame, PROVIDERS)
    frame = _add_prior_month_provider_error_features(frame, PROVIDERS)
    frame = _add_forecast_history_delta_features(frame, PROVIDERS)
    frame = _add_observation_forecast_delta_features(frame, PROVIDERS)
    frame = add_versioned_feature_engineering(frame, feature_version="v20_peak_timing", providers=PROVIDERS)
    return frame.loc[in_window].copy()


def _predict(bundle: dict, frame: pd.DataFrame) -> pd.Series:
    required = list(bundle["feature_names"])
    model_frame = frame.copy()
    for column in required:
        if column not in model_frame:
            model_frame[column] = np.nan
    observed_high = pd.to_numeric(model_frame[OBSERVED_HIGH_SO_FAR_COLUMN], errors="coerce")
    base_highs: dict[str, np.ndarray] = {}
    for method in bundle["base_model_methods"]:
        raw = np.asarray(bundle["base_models"][method].predict(model_frame[required]), dtype=float)
        if bundle.get("target_mode") == "remaining_warmup":
            base_highs[method] = np.maximum(observed_high.to_numpy(), observed_high.to_numpy() + raw)
        else:
            base_highs[method] = raw
    stack_source = pd.DataFrame(
        {f"{method}_predicted_high_f": values for method, values in base_highs.items()},
        index=model_frame.index,
    )
    for provider in PROVIDERS:
        stack_source[f"{provider}_raw_predicted_high_f"] = pd.to_numeric(
            model_frame.get(f"{provider}_high_f"), errors="coerce"
        )
    return pd.Series(
        bundle["stack_model"].predict(stack_source[bundle["stack_features"]]),
        index=model_frame.index,
    )


def main() -> None:
    bundles = {
        ("KATL", "v11_settlement"): _load_bundle(
            "data/calibration/station_stacking_v11_settlement/model_weights/"
            "KATL_station_high_regressor_v11_wunderground_settlement_stack.joblib"
        ),
        ("KATL", "v20"): _load_bundle(
            "data/calibration/station_stacking_v20_peak_timing/model_weights/"
            "KATL_station_high_regressor_v20_peak_timing_stack.joblib"
        ),
        ("KDAL", "v11_settlement"): _load_bundle(
            "data/calibration/station_stacking_v11_settlement/model_weights/"
            "KDAL_station_high_regressor_v11_wunderground_settlement_stack.joblib"
        ),
        ("KDAL", "v11_settlement_fix"): _load_bundle(
            "data/calibration/station_stacking_v11_settlement_fix/model_weights/"
            "KDAL_station_high_regressor_v11_settlement_fix_temp_stack.joblib"
        ),
        ("KDAL", "v20"): _fit_kdal_v20_bundle(),
    }

    rows = []
    for station in ("KATL", "KDAL"):
        frame = _enriched_station_frame(station)
        for (bundle_station, model), bundle in bundles.items():
            if bundle_station != station:
                continue
            predicted = _predict(bundle, frame)
            result = pd.DataFrame(
                {
                    "station": station,
                    "contract_date": frame["contract_date"].astype(str).str[:10],
                    "model": model,
                    "actual_high_f": pd.to_numeric(frame["actual_high_f"], errors="coerce"),
                    "predicted_high_f": predicted,
                    "observed_high_11am_f": pd.to_numeric(
                        frame[OBSERVED_HIGH_SO_FAR_COLUMN], errors="coerce"
                    ),
                    "gfs_high_f": pd.to_numeric(frame["gfs_high_f"], errors="coerce"),
                    "hrrr_high_f": pd.to_numeric(frame["hrrr_high_f"], errors="coerce"),
                    "nbm_high_f": pd.to_numeric(frame["nbm_high_f"], errors="coerce"),
                }
            )
            result["actual_rounded_f"] = _round_half_up_series(result["actual_high_f"])
            result["predicted_rounded_f"] = _round_half_up_series(result["predicted_high_f"])
            result["actual_bucket"] = result["actual_rounded_f"].map(_temperature_bracket_from_rounded)
            result["predicted_bucket"] = result["predicted_rounded_f"].map(_temperature_bracket_from_rounded)
            result["bucket_hit"] = result["actual_bucket"].eq(result["predicted_bucket"])
            result["error_f"] = result["actual_high_f"] - result["predicted_high_f"]
            result["absolute_error_f"] = result["error_f"].abs()
            rows.append(result)

    detail = pd.concat(rows, ignore_index=True).sort_values(["station", "model", "contract_date"])
    summary = (
        detail.groupby(["station", "model"], as_index=False)
        .agg(
            days=("bucket_hit", "size"),
            bucket_hits=("bucket_hit", "sum"),
            bucket_hitrate=("bucket_hit", "mean"),
            mae_f=("absolute_error_f", "mean"),
            bias_f=("error_f", "mean"),
        )
    )
    summary["bucket_hitrate_pct"] = summary.pop("bucket_hitrate") * 100.0
    out_dir = ROOT / "reports/july_3_14_bucket_hitrate"
    out_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out_dir / "detail.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
