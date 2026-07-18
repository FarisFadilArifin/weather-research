from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.calibration.station_stacking import (  # noqa: E402
    TARGET,
    TARGET_SOURCE_SETTLEMENT_FIRST,
    StationStackingConfig,
    _build_base_model_estimator,
    _build_preprocessor,
    _fit_base_estimator,
    _fit_feature_columns,
    _modeling_frame,
    _model_target_values,
    _params_from_selected_row,
    _prediction_output_to_high,
    _round_half_up_series,
    _stack_features_for_set,
    _temperature_bracket_from_rounded,
    _year_split_stack_source_frame,
)


DEFAULT_STATIONS = ("KATL", "KAUS", "KDAL", "KHOU", "KLAX", "KLGA", "KMIA", "KORD", "KSEA")
PROVIDERS = ("gfs", "hrrr", "nbm")
BASE_METHODS = ("xgboost", "lightgbm", "catboost")
TERMS = ("precip", "wind", "dewpoint", "humidity", "ceiling", "weather_code")
SPECIAL_DERIVED_COLUMNS = {
    "v4_precip_humidity_interaction",
    "v4_precip_remaining_warmup_interaction",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablate v11 settlement forecast-weather fields in the 2026 test set.")
    parser.add_argument("--stations", default=",".join(DEFAULT_STATIONS))
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=REPO_ROOT / "data/calibration/station_stacking_v11_settlement",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "reports/v11_settlement_nan_forecast_fields",
    )
    return parser.parse_args()


def forecast_weather_columns(feature_columns: list[str]) -> list[str]:
    selected: list[str] = []
    for column in feature_columns:
        name = column.lower()
        relevant = any(term in name for term in TERMS)
        forecast_derived = (
            name.startswith(("gfs_", "hrrr_", "nbm_"))
            or name.startswith("v4_forecast_")
            or name.startswith("v8_")
            or ("forecast" in name and not name.startswith("observed_"))
            or name in SPECIAL_DERIVED_COLUMNS
        )
        if relevant and forecast_derived:
            selected.append(column)
    return selected


def base_predictions(
    frame: pd.DataFrame,
    masked_columns: list[str],
    config: StationStackingConfig,
    categorical: list[str],
    numeric: list[str],
    selected: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], pd.DataFrame]:
    years = pd.to_numeric(frame["year"], errors="coerce")
    train = frame.loc[years.between(2021, 2025)].copy()
    test = frame.loc[years.eq(2026)].copy()
    masked_test = test.copy()
    for column in masked_columns:
        masked_test[column] = np.nan
    original: dict[str, np.ndarray] = {}
    masked: dict[str, np.ndarray] = {}

    for _, selected_row in selected.iterrows():
        method = str(selected_row["method"])
        params = _params_from_selected_row(selected_row)
        fit_categorical, fit_numeric = _fit_feature_columns(train, categorical, numeric)
        feature_names = [*fit_categorical, *fit_numeric]
        preprocessor = _build_preprocessor(fit_categorical, fit_numeric)
        x_train = preprocessor.fit_transform(train[feature_names])
        x_original = preprocessor.transform(test[feature_names])
        x_masked = preprocessor.transform(masked_test[feature_names])
        y_train = _model_target_values(train, config)
        y_test = _model_target_values(test, config)
        estimator = _build_base_model_estimator(config, method, params, early_stopping_rounds=None)
        _fit_base_estimator(estimator, method, x_train, y_train, x_original, y_test, early_stopping_rounds=None)
        original[method] = _prediction_output_to_high(estimator.predict(x_original), test, config)
        masked[method] = _prediction_output_to_high(estimator.predict(x_masked), masked_test, config)
        print(f"  fitted {method}", flush=True)
    return original, masked, test


def replace_base_predictions(
    stored_test: pd.DataFrame,
    test: pd.DataFrame,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    result = stored_test.loc[~stored_test["method"].eq("ridge_stack")].copy()
    lookup_dates = pd.Series(test["contract_date"].astype(str).to_numpy())
    for method, values in predictions.items():
        replacement = pd.DataFrame({"contract_date": lookup_dates, "replacement": values})
        method_mask = result["method"].eq(method)
        method_rows = result.loc[method_mask].drop(columns=["predicted_high_f"]).merge(
            replacement, on="contract_date", how="left"
        )
        method_rows = method_rows.rename(columns={"replacement": "predicted_high_f"})
        result = pd.concat([result.loc[~method_mask], method_rows], ignore_index=True)
    return result


def stack_predictions(
    validation: pd.DataFrame,
    test_predictions: pd.DataFrame,
    tuning: pd.DataFrame,
) -> pd.DataFrame:
    best = tuning.loc[tuning["status"].eq("ok")].sort_values(["mae_f", "param_key"]).iloc[0]
    methods = [*BASE_METHODS, *(f"{provider}_raw" for provider in PROVIDERS)]
    train_source = _year_split_stack_source_frame(validation, methods)
    test_source = _year_split_stack_source_frame(test_predictions, methods)
    features = _stack_features_for_set(str(best["feature_set"]), BASE_METHODS, PROVIDERS)
    train = train_source.dropna(subset=[*features, TARGET])
    test = test_source.dropna(subset=[*features, TARGET])
    model = Ridge(alpha=float(best["alpha"]), fit_intercept=bool(best["fit_intercept"]))
    model.fit(train[features], train[TARGET])
    result = test[["contract_date", TARGET]].copy()
    result["predicted_high_f"] = model.predict(test[features])
    return result


def comparison_frame(
    station: str,
    original: pd.DataFrame,
    masked: pd.DataFrame,
    published: pd.DataFrame,
) -> pd.DataFrame:
    comparison = original.rename(columns={"predicted_high_f": "refit_prediction_f"}).merge(
        masked[["contract_date", "predicted_high_f"]].rename(columns={"predicted_high_f": "masked_prediction_f"}),
        on="contract_date",
    )
    comparison = comparison.merge(
        published[["contract_date", "predicted_high_f"]].rename(
            columns={"predicted_high_f": "published_prediction_f"}
        ),
        on="contract_date",
    )
    comparison["station_id"] = station
    comparison["prediction_delta_f"] = comparison["masked_prediction_f"] - comparison["refit_prediction_f"]
    comparison["refit_error_f"] = comparison[TARGET] - comparison["refit_prediction_f"]
    comparison["masked_error_f"] = comparison[TARGET] - comparison["masked_prediction_f"]
    comparison["refit_minus_published_f"] = comparison["refit_prediction_f"] - comparison["published_prediction_f"]
    for prefix, column in (
        ("actual", TARGET),
        ("refit", "refit_prediction_f"),
        ("masked", "masked_prediction_f"),
    ):
        comparison[f"{prefix}_rounded_f"] = _round_half_up_series(comparison[column])
        comparison[f"{prefix}_bucket"] = comparison[f"{prefix}_rounded_f"].map(_temperature_bracket_from_rounded)
    return comparison


def station_summary(station: str, comparison: pd.DataFrame, masked_feature_count: int) -> dict[str, object]:
    return {
        "station_id": station,
        "count": len(comparison),
        "masked_feature_count": masked_feature_count,
        "baseline_mae_f": comparison["refit_error_f"].abs().mean(),
        "masked_mae_f": comparison["masked_error_f"].abs().mean(),
        "baseline_rmse_f": np.sqrt(np.mean(comparison["refit_error_f"] ** 2)),
        "masked_rmse_f": np.sqrt(np.mean(comparison["masked_error_f"] ** 2)),
        "baseline_bucket_hit_pct": 100 * comparison["actual_bucket"].eq(comparison["refit_bucket"]).mean(),
        "masked_bucket_hit_pct": 100 * comparison["actual_bucket"].eq(comparison["masked_bucket"]).mean(),
        "mean_prediction_delta_f": comparison["prediction_delta_f"].mean(),
        "mean_abs_prediction_delta_f": comparison["prediction_delta_f"].abs().mean(),
        "p95_abs_prediction_delta_f": comparison["prediction_delta_f"].abs().quantile(0.95),
        "max_abs_prediction_delta_f": comparison["prediction_delta_f"].abs().max(),
        "rounded_prediction_changed_pct": 100
        * comparison["refit_rounded_f"].ne(comparison["masked_rounded_f"]).mean(),
        "bucket_changed_pct": 100 * comparison["refit_bucket"].ne(comparison["masked_bucket"]).mean(),
        "max_abs_refit_minus_published_f": comparison["refit_minus_published_f"].abs().max(),
    }


def anchored_summaries(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    anchored = detail.copy()
    anchored["anchored_masked_prediction_f"] = (
        anchored["published_prediction_f"] + anchored["prediction_delta_f"]
    )
    anchored["published_error_f"] = anchored[TARGET] - anchored["published_prediction_f"]
    anchored["anchored_masked_error_f"] = anchored[TARGET] - anchored["anchored_masked_prediction_f"]
    anchored["published_rounded_f"] = _round_half_up_series(anchored["published_prediction_f"])
    anchored["anchored_masked_rounded_f"] = _round_half_up_series(anchored["anchored_masked_prediction_f"])
    anchored["published_bucket"] = anchored["published_rounded_f"].map(_temperature_bracket_from_rounded)
    anchored["anchored_masked_bucket"] = anchored["anchored_masked_rounded_f"].map(
        _temperature_bracket_from_rounded
    )

    def summarize(group: pd.DataFrame, station_id: str) -> dict[str, object]:
        return {
            "station_id": station_id,
            "count": len(group),
            "baseline_mae_f": group["published_error_f"].abs().mean(),
            "masked_mae_f": group["anchored_masked_error_f"].abs().mean(),
            "mae_delta_f": group["anchored_masked_error_f"].abs().mean()
            - group["published_error_f"].abs().mean(),
            "baseline_rmse_f": np.sqrt(np.mean(group["published_error_f"] ** 2)),
            "masked_rmse_f": np.sqrt(np.mean(group["anchored_masked_error_f"] ** 2)),
            "baseline_bucket_hit_pct": 100
            * group["actual_bucket"].eq(group["published_bucket"]).mean(),
            "masked_bucket_hit_pct": 100
            * group["actual_bucket"].eq(group["anchored_masked_bucket"]).mean(),
            "bucket_hit_delta_pp": 100
            * (
                group["actual_bucket"].eq(group["anchored_masked_bucket"]).mean()
                - group["actual_bucket"].eq(group["published_bucket"]).mean()
            ),
            "mean_prediction_delta_f": group["prediction_delta_f"].mean(),
            "mean_abs_prediction_delta_f": group["prediction_delta_f"].abs().mean(),
            "p95_abs_prediction_delta_f": group["prediction_delta_f"].abs().quantile(0.95),
            "max_abs_prediction_delta_f": group["prediction_delta_f"].abs().max(),
            "rounded_prediction_changed_pct": 100
            * group["published_rounded_f"].ne(group["anchored_masked_rounded_f"]).mean(),
            "bucket_changed_pct": 100
            * group["published_bucket"].ne(group["anchored_masked_bucket"]).mean(),
        }

    station_rows = [summarize(group, str(station)) for station, group in anchored.groupby("station_id")]
    station_frame = pd.DataFrame(station_rows).sort_values("station_id")
    pooled_frame = pd.DataFrame([summarize(anchored, "POOLED")])
    return anchored, station_frame, pooled_frame


def main() -> None:
    warnings.filterwarnings("ignore")
    args = parse_args()
    stations = tuple(value.strip().upper() for value in args.stations.split(",") if value.strip())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "station_summary.csv"
    detail_path = args.output_dir / "prediction_comparison.csv"
    existing_summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    existing_detail = pd.read_csv(detail_path) if detail_path.exists() else pd.DataFrame()
    completed = set(existing_summary.get("station_id", pd.Series(dtype="string")).astype(str))

    for station in stations:
        if station in completed:
            print(f"SKIP {station}: checkpoint exists", flush=True)
            continue
        print(f"START {station}", flush=True)
        features = pd.read_csv(args.artifact_dir / f"{station}_features.csv", low_memory=False)
        feature_inventory = pd.read_csv(args.artifact_dir / f"{station}_feature_columns.csv")
        categorical = feature_inventory.loc[feature_inventory["kind"].eq("categorical"), "feature"].tolist()
        numeric = feature_inventory.loc[feature_inventory["kind"].eq("numeric"), "feature"].tolist()
        masked_columns = forecast_weather_columns([*categorical, *numeric])
        config = StationStackingConfig(
            station_id=station,
            project_root=REPO_ROOT,
            timing_mode="same_day_11am_live_safe",
            providers=PROVIDERS,
            feature_version="v11",
            target_mode="remaining_warmup",
            target_source=TARGET_SOURCE_SETTLEMENT_FIRST,
            hyperparameter_space="wide",
            base_model_methods=BASE_METHODS,
            stack_enabled=True,
            year_split_test_train_years=(2021, 2025),
            year_split_test_year=2026,
            output_dir=args.artifact_dir,
            optuna_verbose=False,
        )
        frame, model_categorical, model_numeric = _modeling_frame(features, config)
        if categorical != model_categorical or numeric != model_numeric:
            raise ValueError(f"{station}: saved feature inventory does not match reconstructed modeling columns")
        selected = pd.read_csv(args.artifact_dir / f"{station}_year_split_selected_hyperparameters.csv")
        original_base, masked_base, test = base_predictions(
            frame, masked_columns, config, categorical, numeric, selected
        )
        stored_test = pd.read_csv(args.artifact_dir / f"{station}_year_split_test_predictions.csv")
        validation = pd.read_csv(args.artifact_dir / f"{station}_year_split_validation_predictions.csv")
        tuning = pd.read_csv(args.artifact_dir / f"{station}_year_split_stack_tuning.csv")
        original_stack = stack_predictions(
            validation, replace_base_predictions(stored_test, test, original_base), tuning
        )
        masked_stack = stack_predictions(
            validation, replace_base_predictions(stored_test, test, masked_base), tuning
        )
        published = stored_test.loc[stored_test["method"].eq("ridge_stack")].copy()
        detail = comparison_frame(station, original_stack, masked_stack, published)
        summary = pd.DataFrame([station_summary(station, detail, len(masked_columns))])
        existing_summary = pd.concat([existing_summary, summary], ignore_index=True)
        existing_detail = pd.concat([existing_detail, detail], ignore_index=True)
        existing_summary.to_csv(summary_path, index=False)
        existing_detail.to_csv(detail_path, index=False)
        row = summary.iloc[0]
        print(
            f"DONE {station}: MAE {row['baseline_mae_f']:.4f}->{row['masked_mae_f']:.4f}; "
            f"bucket {row['baseline_bucket_hit_pct']:.2f}%->{row['masked_bucket_hit_pct']:.2f}%",
            flush=True,
        )

    anchored, anchored_station, anchored_pooled = anchored_summaries(existing_detail)
    anchored.to_csv(args.output_dir / "anchored_prediction_comparison.csv", index=False)
    anchored_station.to_csv(args.output_dir / "anchored_station_summary.csv", index=False)
    anchored_pooled.to_csv(args.output_dir / "anchored_pooled_summary.csv", index=False)
    print(existing_summary.sort_values("station_id").to_string(index=False), flush=True)
    print("\nAnchored station summary", flush=True)
    print(anchored_station.to_string(index=False), flush=True)
    print("\nAnchored pooled summary", flush=True)
    print(anchored_pooled.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
