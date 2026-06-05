from __future__ import annotations

import numpy as np
import pandas as pd

from .features import add_weather_regime_buckets


METRIC_COLUMNS = [
    "count",
    "mean_error_f",
    "median_error_f",
    "mae_f",
    "rmse_f",
    "error_std_f",
    "error_p10_f",
    "error_p25_f",
    "error_p75_f",
    "error_p90_f",
    "within_1f_pct",
    "within_2f_pct",
    "within_3f_pct",
    "within_5f_pct",
    "warm_bias_frequency",
    "cool_bias_frequency",
]


def add_error_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["error_f"] = out["actual_high_f"] - out["forecast_high_f"]
    out["abs_error_f"] = out["error_f"].abs()
    out["squared_error_f"] = out["error_f"] ** 2
    return out


def summarize_errors(df: pd.DataFrame) -> pd.Series:
    errors = pd.to_numeric(df["error_f"], errors="coerce").dropna()
    if errors.empty:
        return pd.Series({column: np.nan for column in METRIC_COLUMNS})
    abs_errors = errors.abs()
    return pd.Series(
        {
            "count": int(errors.count()),
            "mean_error_f": float(errors.mean()),
            "median_error_f": float(errors.median()),
            "mae_f": float(abs_errors.mean()),
            "rmse_f": float(np.sqrt((errors**2).mean())),
            "error_std_f": float(errors.std(ddof=1)) if errors.count() > 1 else 0.0,
            "error_p10_f": float(errors.quantile(0.10)),
            "error_p25_f": float(errors.quantile(0.25)),
            "error_p75_f": float(errors.quantile(0.75)),
            "error_p90_f": float(errors.quantile(0.90)),
            "within_1f_pct": float((abs_errors <= 1).mean() * 100),
            "within_2f_pct": float((abs_errors <= 2).mean() * 100),
            "within_3f_pct": float((abs_errors <= 3).mean() * 100),
            "within_5f_pct": float((abs_errors <= 5).mean() * 100),
            "warm_bias_frequency": float((errors > 0).mean() * 100),
            "cool_bias_frequency": float((errors < 0).mean() * 100),
        }
    )


def metrics_by(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=group_cols + METRIC_COLUMNS)
    clean = df.dropna(subset=["error_f"])
    if clean.empty:
        return pd.DataFrame(columns=group_cols + METRIC_COLUMNS)
    return clean.groupby(group_cols, dropna=False).apply(summarize_errors, include_groups=False).reset_index()


def write_metric_outputs(model_errors: pd.DataFrame, output_dir: str) -> dict[str, pd.DataFrame]:
    from pathlib import Path

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    regime_errors = add_weather_regime_buckets(model_errors)
    outputs = {
        "metrics_by_station_provider_horizon.csv": metrics_by(
            model_errors,
            ["station_code", "station_name", "airport_name", "provider", "model", "forecast_horizon_hours"],
        ),
        "mae_by_provider_station_horizon_month.csv": metrics_by(
            model_errors,
            [
                "provider",
                "model",
                "station_code",
                "station_name",
                "airport_name",
                "forecast_horizon_hours",
                "month",
            ],
        ),
        "metrics_by_month.csv": metrics_by(model_errors, ["month", "station_code", "provider", "model"]),
        "metrics_by_season.csv": metrics_by(model_errors, ["season", "station_code", "provider", "model"]),
        "metrics_by_weather_regime.csv": metrics_by(
            regime_errors,
            ["cloud_cover_bucket", "precipitation_bucket", "wind_speed_bucket", "provider_disagreement_bucket", "provider", "model"],
        ),
    }
    for filename, frame in outputs.items():
        frame.to_csv(out / filename, index=False)
    return outputs
