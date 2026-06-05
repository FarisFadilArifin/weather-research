from __future__ import annotations

from pathlib import Path

import pandas as pd

from .features import add_time_features, add_weather_regime_buckets, compute_provider_disagreement
from .metrics import add_error_columns


MODEL_ERROR_COLUMNS = [
    "market_slug",
    "event_id",
    "market_id",
    "polymarket_city",
    "station_code",
    "station_name",
    "airport_name",
    "provider",
    "model",
    "issue_time_utc",
    "issue_time_local",
    "target_date_local",
    "forecast_horizon_hours",
    "forecast_high_f",
    "actual_high_f",
    "error_f",
    "abs_error_f",
    "squared_error_f",
    "month",
    "season",
    "day_of_week",
    "cloud_cover_mean",
    "cloud_cover_max",
    "precip_amount",
    "wind_speed_mean",
    "dewpoint_mean_f",
    "humidity_mean",
    "provider_disagreement_f",
    "outcome_buckets",
    "resolved_polymarket_bucket",
]


def combine_forecasts(frames: list[pd.DataFrame]) -> pd.DataFrame:
    nonempty = [frame for frame in frames if frame is not None and not frame.empty]
    if not nonempty:
        return pd.DataFrame()
    return pd.concat(nonempty, ignore_index=True)


def join_forecasts_actuals_markets(
    markets: pd.DataFrame,
    station_map: pd.DataFrame,
    actuals: pd.DataFrame,
    forecasts: pd.DataFrame,
    include_active_training: bool = False,
) -> pd.DataFrame:
    if forecasts.empty or actuals.empty or station_map.empty:
        return pd.DataFrame(columns=MODEL_ERROR_COLUMNS)
    forecasts = compute_provider_disagreement(forecasts)
    forecasts = forecasts.dropna(subset=["forecast_high_f"])
    actuals = actuals.dropna(subset=["actual_high_f"])

    joined = forecasts.merge(
        actuals[["station_code", "date_local", "actual_high_f"]],
        left_on=["station_code", "target_date_local"],
        right_on=["station_code", "date_local"],
        how="inner",
    )
    joined = joined.merge(
        station_map[["market_slug", "target_date_local", "station_code", "polymarket_city"]],
        on=["target_date_local", "station_code"],
        how="left",
    )
    market_cols = [
        "slug",
        "event_id",
        "market_id",
        "outcome_buckets",
        "final_outcome",
        "is_active",
        "is_resolved",
    ]
    event_market_rows = markets[market_cols].sort_values("market_id").drop_duplicates(subset=["slug"])
    joined = joined.merge(event_market_rows, left_on="market_slug", right_on="slug", how="left")
    if not include_active_training:
        joined = joined.loc[joined["is_resolved"].fillna(True).astype(bool)]

    joined["resolved_polymarket_bucket"] = joined["final_outcome"]
    joined = add_error_columns(joined)
    joined = add_time_features(joined)
    joined = add_weather_regime_buckets(joined)
    joined = joined.rename(columns={"market_slug": "market_slug"})
    for column in MODEL_ERROR_COLUMNS:
        if column not in joined:
            joined[column] = pd.NA
    return joined[MODEL_ERROR_COLUMNS]


def write_model_errors(
    markets: pd.DataFrame,
    station_map: pd.DataFrame,
    actuals: pd.DataFrame,
    forecasts: pd.DataFrame,
    processed_dir: str | Path,
    include_active_training: bool = False,
) -> pd.DataFrame:
    processed = Path(processed_dir)
    processed.mkdir(parents=True, exist_ok=True)
    frame = join_forecasts_actuals_markets(
        markets,
        station_map,
        actuals,
        forecasts,
        include_active_training=include_active_training,
    )
    frame.to_csv(processed / "model_errors.csv", index=False)
    return frame
