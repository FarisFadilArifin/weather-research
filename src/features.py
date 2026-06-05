from __future__ import annotations

import numpy as np
import pandas as pd


def month_from_date(date_series: pd.Series) -> pd.Series:
    return pd.to_datetime(date_series, errors="coerce").dt.month


def season_from_month(month: int | float | None) -> str | None:
    if pd.isna(month):
        return None
    month = int(month)
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "fall"


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dates = pd.to_datetime(out["target_date_local"], errors="coerce")
    out["month"] = dates.dt.month
    out["season"] = out["month"].map(season_from_month)
    out["day_of_week"] = dates.dt.day_name()
    return out


def add_weather_regime_buckets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["cloud_cover_bucket"] = pd.cut(
        pd.to_numeric(out.get("cloud_cover_mean"), errors="coerce"),
        bins=[-0.1, 25, 50, 75, 100],
        labels=["clear", "partly_cloudy", "mostly_cloudy", "overcast"],
    )
    out["precipitation_bucket"] = pd.cut(
        pd.to_numeric(out.get("precip_amount"), errors="coerce").fillna(0),
        bins=[-0.001, 0.01, 0.1, 0.5, np.inf],
        labels=["dry", "light", "moderate", "heavy"],
    )
    out["wind_speed_bucket"] = pd.cut(
        pd.to_numeric(out.get("wind_speed_mean"), errors="coerce"),
        bins=[-0.1, 5, 12, 20, np.inf],
        labels=["calm", "breezy", "windy", "very_windy"],
    )
    out["provider_disagreement_bucket"] = pd.cut(
        pd.to_numeric(out.get("provider_disagreement_f"), errors="coerce"),
        bins=[-0.1, 1, 3, 5, np.inf],
        labels=["low", "medium", "high", "very_high"],
    )
    return out


def compute_provider_disagreement(forecasts: pd.DataFrame) -> pd.DataFrame:
    if forecasts.empty:
        forecasts["provider_disagreement_f"] = pd.NA
        return forecasts
    out = forecasts.copy()
    grouped = out.groupby(["station_code", "target_date_local", "forecast_horizon_hours"], dropna=False)[
        "forecast_high_f"
    ]
    spread = grouped.transform(lambda s: s.max(skipna=True) - s.min(skipna=True))
    out["provider_disagreement_f"] = spread
    return out
