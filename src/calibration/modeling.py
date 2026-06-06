from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BASELINE_METHODS = [
    "global_provider_mean_bias",
    "station_provider_mean_bias",
    "station_provider_month_mean_bias",
    "station_provider_horizon_month_mean_bias",
    "hierarchical_shrinkage",
]

TARGET = "calibration_bias_f"


@dataclass(frozen=True)
class WalkForwardConfig:
    min_train_days: int = 90
    shrinkage_k: float = 30.0
    random_state: int = 42
    ml_refit_days: int = 30


def train_and_evaluate(
    calibration_dir: str | Path,
    min_train_days: int = 90,
    shrinkage_k: float = 30.0,
    ml_refit_days: int = 30,
    run_ml: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out_dir = Path(calibration_dir)
    samples_path = out_dir / "calibration_samples.csv"
    if not samples_path.exists():
        raise FileNotFoundError(f"Missing calibration samples: {samples_path}")
    samples = pd.read_csv(samples_path)
    config = WalkForwardConfig(
        min_train_days=min_train_days,
        shrinkage_k=shrinkage_k,
        ml_refit_days=ml_refit_days,
    )
    baseline_predictions = walk_forward_baseline_predictions(samples, config)
    baseline_results = summarize_prediction_results(baseline_predictions, model_family="baseline")
    baseline_results.to_csv(out_dir / "baseline_results.csv", index=False)

    if run_ml:
        ml_predictions = walk_forward_ml_predictions(samples, config)
        ml_results = summarize_prediction_results(ml_predictions, model_family="ml")
    else:
        ml_results = pd.DataFrame(columns=baseline_results.columns)
    ml_results.to_csv(out_dir / "ml_results.csv", index=False)

    recommended = recommended_rules(samples, baseline_predictions, baseline_results, ml_results)
    recommended.to_csv(out_dir / "recommended_calibration_rules.csv", index=False)
    return baseline_results, ml_results, recommended


def walk_forward_baseline_predictions(samples: pd.DataFrame, config: WalkForwardConfig) -> pd.DataFrame:
    clean = _clean_samples(samples)
    if clean.empty:
        return _empty_predictions()
    rows: list[pd.DataFrame] = []
    dates = sorted(clean["contract_date"].unique())
    for date_value in dates:
        train = clean.loc[clean["contract_date"] < date_value]
        if train["contract_date"].nunique() < config.min_train_days:
            continue
        valid = clean.loc[clean["contract_date"] == date_value].copy()
        if valid.empty:
            continue
        for method in BASELINE_METHODS:
            pred = valid.copy()
            pred["method"] = method
            pred["predicted_calibration_bias_f"] = _predict_baseline(method, train, valid, config.shrinkage_k)
            rows.append(_prediction_columns(pred))
    return pd.concat(rows, ignore_index=True) if rows else _empty_predictions()


def walk_forward_ml_predictions(samples: pd.DataFrame, config: WalkForwardConfig) -> pd.DataFrame:
    clean = _clean_samples(samples)
    if clean.empty:
        return _empty_predictions()
    categorical_cols, numeric_cols = _feature_columns(clean)
    models = _build_ml_models(config.random_state, categorical_cols, numeric_cols)
    if not models:
        return _empty_predictions()
    rows: list[pd.DataFrame] = []
    dates = sorted(clean["contract_date"].unique())
    feature_cols = categorical_cols + numeric_cols
    step = max(1, int(config.ml_refit_days))
    for start_idx in range(0, len(dates), step):
        date_value = dates[start_idx]
        block_dates = dates[start_idx : start_idx + step]
        train = clean.loc[clean["contract_date"] < date_value]
        if train["contract_date"].nunique() < config.min_train_days:
            continue
        valid = clean.loc[clean["contract_date"].isin(block_dates)].copy()
        if valid.empty:
            continue
        for name, estimator in models.items():
            try:
                estimator.fit(train[feature_cols], train[TARGET])
                predicted = estimator.predict(valid[feature_cols])
            except Exception:
                continue
            pred = valid.copy()
            pred["method"] = name
            pred["predicted_calibration_bias_f"] = predicted
            rows.append(_prediction_columns(pred))
    return pd.concat(rows, ignore_index=True) if rows else _empty_predictions()


def summarize_prediction_results(predictions: pd.DataFrame, model_family: str) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(
            columns=[
                "model_family",
                "method",
                "evaluation_scope",
                "station_id",
                "provider",
                "model",
                "timing_mode",
                "month",
                "count",
                "mae_before_f",
                "mae_after_f",
                "mae_improvement_f",
                "rmse_before_f",
                "rmse_after_f",
                "bias_before_f",
                "bias_after_f",
                "within_1f_before_pct",
                "within_1f_after_pct",
                "within_2f_before_pct",
                "within_2f_after_pct",
                "within_3f_before_pct",
                "within_3f_after_pct",
            ]
        )
    frames = [
        _summarize_scope(predictions, ["method"], "overall"),
        _summarize_scope(predictions, ["method", "station_id"], "station"),
        _summarize_scope(predictions, ["method", "provider", "model", "timing_mode"], "provider_model"),
        _summarize_scope(predictions, ["method", "month"], "month"),
        _summarize_scope(
            predictions,
            ["method", "station_id", "provider", "model", "timing_mode", "month"],
            "station_provider_month",
        ),
    ]
    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "model_family", model_family)
    for column in ["station_id", "provider", "model", "timing_mode", "month"]:
        if column not in out:
            out[column] = pd.NA
    return out


def recommended_rules(
    samples: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    baseline_results: pd.DataFrame,
    ml_results: pd.DataFrame,
) -> pd.DataFrame:
    clean = _clean_samples(samples)
    if clean.empty:
        return pd.DataFrame()
    shrinkage = baseline_predictions.loc[baseline_predictions["method"] == "hierarchical_shrinkage"].copy()
    if shrinkage.empty:
        return pd.DataFrame()
    latest_date = clean["contract_date"].max()
    rows: list[dict[str, Any]] = []
    for keys, group in clean.groupby(["station_id", "provider", "model", "timing_mode", "month"], dropna=False):
        station_id, provider, model, timing_mode, month = keys
        train = clean.loc[
            (clean["contract_date"] <= latest_date)
            & (clean["station_id"] == station_id)
            & (clean["provider"] == provider)
            & (clean["model"] == model)
            & (clean["timing_mode"] == timing_mode)
        ]
        representative = pd.DataFrame(
            [
                {
                    "station_id": station_id,
                    "provider": provider,
                    "model": model,
                    "timing_mode": timing_mode,
                    "month": month,
                    "horizon_hours": 0,
                }
            ]
        )
        rule_bias = float(_predict_hierarchical_shrinkage(train, representative, k=30.0).iloc[0])
        sample_count = int(group[TARGET].notna().sum())
        rows.append(
            {
                "station_id": station_id,
                "provider": provider,
                "model": model,
                "timing_mode": timing_mode,
                "month": int(month) if pd.notna(month) else pd.NA,
                "horizon_hours": 0,
                "recommended_method": _choose_recommendation(baseline_results, ml_results),
                "calibration_add_f": rule_bias,
                "sample_count": sample_count,
                "source": "walk_forward_hierarchical_shrinkage",
                "valid_after_contract_date": latest_date,
            }
        )
    return pd.DataFrame(rows).sort_values(["provider", "model", "timing_mode", "station_id", "month"]).reset_index(drop=True)


def _predict_baseline(method: str, train: pd.DataFrame, valid: pd.DataFrame, shrinkage_k: float) -> pd.Series:
    if method == "global_provider_mean_bias":
        return _group_mean_predict(train, valid, ["provider", "model", "timing_mode"])
    if method == "station_provider_mean_bias":
        return _group_mean_predict(train, valid, ["station_id", "provider", "model", "timing_mode"])
    if method == "station_provider_month_mean_bias":
        return _group_mean_predict(train, valid, ["station_id", "provider", "model", "timing_mode", "month"])
    if method == "station_provider_horizon_month_mean_bias":
        return _group_mean_predict(train, valid, ["station_id", "provider", "model", "timing_mode", "horizon_hours", "month"])
    if method == "hierarchical_shrinkage":
        return _predict_hierarchical_shrinkage(train, valid, shrinkage_k)
    raise ValueError(f"Unknown baseline method: {method}")


def _group_mean_predict(train: pd.DataFrame, valid: pd.DataFrame, group_cols: list[str]) -> pd.Series:
    global_mean = float(train[TARGET].mean()) if not train.empty else 0.0
    provider_mean = train.groupby(["provider", "model", "timing_mode"], dropna=False)[TARGET].mean()
    means = train.groupby(group_cols, dropna=False)[TARGET].mean()
    values: list[float] = []
    for _, row in valid.iterrows():
        key = tuple(row[col] for col in group_cols)
        if key in means.index:
            values.append(float(means.loc[key]))
            continue
        provider_key = (row["provider"], row["model"], row["timing_mode"])
        values.append(float(provider_mean.loc[provider_key]) if provider_key in provider_mean.index else global_mean)
    return pd.Series(values, index=valid.index, dtype=float)


def _predict_hierarchical_shrinkage(train: pd.DataFrame, valid: pd.DataFrame, k: float) -> pd.Series:
    global_mean = float(train[TARGET].mean()) if not train.empty else 0.0
    provider_stats = _stats(train, ["provider", "model", "timing_mode"])
    station_stats = _stats(train, ["station_id", "provider", "model", "timing_mode"])
    month_stats = _stats(train, ["station_id", "provider", "model", "timing_mode", "month"])
    horizon_stats = _stats(train, ["station_id", "provider", "model", "timing_mode", "horizon_hours", "month"])
    values: list[float] = []
    for _, row in valid.iterrows():
        provider_key = (row["provider"], row["model"], row["timing_mode"])
        station_key = (row["station_id"], row["provider"], row["model"], row["timing_mode"])
        month_key = (row["station_id"], row["provider"], row["model"], row["timing_mode"], row["month"])
        horizon_key = (
            row["station_id"],
            row["provider"],
            row["model"],
            row["timing_mode"],
            row["horizon_hours"],
            row["month"],
        )
        prior = _blend(global_mean, provider_stats.get(provider_key), k)
        prior = _blend(prior, station_stats.get(station_key), k)
        prior = _blend(prior, month_stats.get(month_key), k)
        prior = _blend(prior, horizon_stats.get(horizon_key), k)
        values.append(float(prior))
    return pd.Series(values, index=valid.index, dtype=float)


def _stats(train: pd.DataFrame, group_cols: list[str]) -> dict[tuple[Any, ...], tuple[float, int]]:
    if train.empty:
        return {}
    grouped = train.groupby(group_cols, dropna=False)[TARGET].agg(["mean", "count"])
    return {key if isinstance(key, tuple) else (key,): (float(row["mean"]), int(row["count"])) for key, row in grouped.iterrows()}


def _blend(prior: float, stat: tuple[float, int] | None, k: float) -> float:
    if stat is None:
        return prior
    mean, count = stat
    weight = count / (count + k)
    return weight * mean + (1 - weight) * prior


def _summarize_scope(predictions: pd.DataFrame, group_cols: list[str], scope: str) -> pd.DataFrame:
    grouped = predictions.groupby(group_cols, dropna=False).apply(_metric_row, include_groups=False).reset_index()
    grouped["evaluation_scope"] = scope
    return grouped


def _metric_row(group: pd.DataFrame) -> pd.Series:
    before = pd.to_numeric(group["calibration_bias_f"], errors="coerce")
    after = pd.to_numeric(group["residual_after_calibration_f"], errors="coerce")
    before_abs = before.abs()
    after_abs = after.abs()
    return pd.Series(
        {
            "count": int(after.notna().sum()),
            "mae_before_f": float(before_abs.mean()),
            "mae_after_f": float(after_abs.mean()),
            "mae_improvement_f": float(before_abs.mean() - after_abs.mean()),
            "rmse_before_f": float(np.sqrt((before**2).mean())),
            "rmse_after_f": float(np.sqrt((after**2).mean())),
            "bias_before_f": float(before.mean()),
            "bias_after_f": float(after.mean()),
            "within_1f_before_pct": float((before_abs <= 1).mean() * 100),
            "within_1f_after_pct": float((after_abs <= 1).mean() * 100),
            "within_2f_before_pct": float((before_abs <= 2).mean() * 100),
            "within_2f_after_pct": float((after_abs <= 2).mean() * 100),
            "within_3f_before_pct": float((before_abs <= 3).mean() * 100),
            "within_3f_after_pct": float((after_abs <= 3).mean() * 100),
        }
    )


def _build_ml_models(random_state: int, categorical: list[str], numeric: list[str]) -> dict[str, Any]:
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import ElasticNet, Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except Exception:
        return {}

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numeric,
            ),
        ],
        remainder="drop",
    )
    tree_preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
            ("num", SimpleImputer(strategy="median"), numeric),
        ],
        remainder="drop",
    )
    return {
        "ridge": Pipeline([("prep", preprocessor), ("model", Ridge(alpha=2.0))]),
        "elasticnet": Pipeline([("prep", preprocessor), ("model", ElasticNet(alpha=0.03, l1_ratio=0.2, max_iter=5000))]),
        "random_forest": Pipeline(
            [
                ("prep", tree_preprocessor),
                ("model", RandomForestRegressor(n_estimators=80, min_samples_leaf=10, random_state=random_state, n_jobs=-1)),
            ]
        ),
        "extra_trees": Pipeline(
            [
                ("prep", tree_preprocessor),
                ("model", ExtraTreesRegressor(n_estimators=80, min_samples_leaf=10, random_state=random_state, n_jobs=-1)),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("prep", tree_preprocessor),
                ("model", HistGradientBoostingRegressor(max_iter=100, learning_rate=0.05, l2_regularization=0.2, random_state=random_state)),
            ]
        ),
    }


def _feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical = ["station_id", "provider", "model", "timing_mode", "month", "day_of_week", "rain_regime", "cloud_regime"]
    numeric = [
        "horizon_hours",
        "raw_forecast_high_f",
        "day_of_year_sin",
        "day_of_year_cos",
        "forecast_issue_hour_utc",
        "forecast_lead_hours",
        "dewpoint_mean_f",
        "humidity_mean",
        "wind_speed_mean",
        "wind_speed_max",
        "observed_temp_at_as_of_f",
        "observed_dewpoint_at_as_of_f",
        "observed_humidity_at_as_of",
        "observed_wind_speed_at_as_of",
        "observed_pressure_at_as_of",
        "observed_visibility_at_as_of",
        "observed_as_of_age_minutes",
        "provider_disagreement_f",
        "rolling_provider_bias_7d",
        "rolling_provider_bias_30d",
        "rolling_provider_bias_90d",
        "expanding_provider_bias",
        "past_provider_sample_count",
        "raw_forecast_station_month_z",
    ]
    cols = categorical + numeric
    for col in cols:
        if col not in frame:
            frame[col] = pd.NA
    usable_numeric = [col for col in numeric if pd.to_numeric(frame[col], errors="coerce").notna().any()]
    return categorical, usable_numeric


def _clean_samples(samples: pd.DataFrame) -> pd.DataFrame:
    if samples.empty:
        return pd.DataFrame()
    clean = samples.dropna(subset=["contract_date", TARGET, "raw_forecast_high_f"]).copy()
    clean["contract_date"] = pd.to_datetime(clean["contract_date"], errors="coerce").dt.date.astype(str)
    clean = clean.dropna(subset=["contract_date"])
    clean["month"] = pd.to_numeric(clean["month"], errors="coerce").astype("Int64")
    clean["horizon_hours"] = pd.to_numeric(clean["horizon_hours"], errors="coerce").fillna(0).astype(int)
    for col in ["station_id", "provider", "model", "timing_mode", "day_of_week", "rain_regime", "cloud_regime"]:
        if col not in clean:
            clean[col] = "strict_6am" if col == "timing_mode" else "unknown"
        clean[col] = clean[col].astype("string").fillna("unknown")
    clean["timing_mode"] = clean["timing_mode"].replace("unknown", "strict_6am")
    return clean.sort_values(["contract_date", "station_id", "provider", "model"]).reset_index(drop=True)


def _prediction_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["residual_after_calibration_f"] = out["calibration_bias_f"] - out["predicted_calibration_bias_f"]
    out["absolute_error_after"] = out["residual_after_calibration_f"].abs()
    columns = [
        "station_id",
        "provider",
        "model",
        "timing_mode",
        "contract_date",
        "month",
        "method",
        "raw_forecast_high_f",
        "actual_high_f",
        "calibration_bias_f",
        "predicted_calibration_bias_f",
        "residual_after_calibration_f",
        "absolute_error_before",
        "absolute_error_after",
    ]
    return out[columns]


def _empty_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "station_id",
            "provider",
            "model",
            "timing_mode",
            "contract_date",
            "month",
            "method",
            "raw_forecast_high_f",
            "actual_high_f",
            "calibration_bias_f",
            "predicted_calibration_bias_f",
            "residual_after_calibration_f",
            "absolute_error_before",
            "absolute_error_after",
        ]
    )


def _choose_recommendation(baseline_results: pd.DataFrame, ml_results: pd.DataFrame) -> str:
    base = _overall_method_row(baseline_results, "hierarchical_shrinkage")
    if base is None or ml_results.empty:
        return "hierarchical_shrinkage"
    best_ml = ml_results.loc[ml_results["evaluation_scope"] == "overall"].sort_values("mae_after_f").head(1)
    if best_ml.empty:
        return "hierarchical_shrinkage"
    ml_row = best_ml.iloc[0]
    improvement = float(base["mae_after_f"]) - float(ml_row["mae_after_f"])
    return str(ml_row["method"]) if improvement >= 0.10 and _ml_stability_ok(ml_results, str(ml_row["method"])) else "hierarchical_shrinkage"


def _overall_method_row(results: pd.DataFrame, method: str) -> pd.Series | None:
    if results.empty:
        return None
    rows = results.loc[(results["evaluation_scope"] == "overall") & (results["method"] == method)]
    return None if rows.empty else rows.iloc[0]


def _ml_stability_ok(ml_results: pd.DataFrame, method: str) -> bool:
    split = ml_results.loc[
        (ml_results["method"] == method)
        & (ml_results["evaluation_scope"] == "station_provider_month")
        & (ml_results["count"] >= 10)
    ]
    if split.empty:
        return False
    return float((split["mae_improvement_f"] > 0).mean()) >= 0.70
