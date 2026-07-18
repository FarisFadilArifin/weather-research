from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .station_stacking import (
    StationStackingConfig,
    _modeling_frame,
    _validation_predictions_for_selected_params,
    summarize_year_split_predictions,
    tune_year_split_base_models,
    tune_year_split_stack_model,
    year_split_baseline_predictions,
    year_split_bracket_metrics,
    year_split_bracket_predictions,
    year_split_feature_importance,
    year_split_scoreboard,
    year_split_test_predictions,
)
from .v11_settlement_enrichment import (
    FORBIDDEN_EXACT,
    FORBIDDEN_PATTERNS,
    PROVIDERS,
    add_cross_provider_features,
    add_forecast_observation_deltas,
    extended_prediction_metrics,
)


VARIANTS = ("cleaned_v11", "observation_only", "forecast_only", "combined")


def load_enriched_feature_frames(
    *,
    baseline_dir: str | Path,
    cache_root: str | Path,
    stations: Iterable[str],
) -> dict[str, pd.DataFrame]:
    baseline_root = Path(baseline_dir)
    cache = Path(cache_root)
    forecast = _wide_forecast(pd.read_csv(cache / "forecast_daily_enriched.csv"))
    observed = pd.read_csv(cache / "observation_daily_enriched.csv")
    frames: dict[str, pd.DataFrame] = {}
    for station in stations:
        station_id = station.upper()
        base = pd.read_csv(baseline_root / f"{station_id}_features.csv")
        base["station_id"] = station_id
        merged = base.merge(
            forecast.loc[forecast["station_id"].eq(station_id)],
            on=["station_id", "contract_date"],
            how="left",
            suffixes=("", "_enriched_forecast"),
        )
        observed_station = observed.loc[observed["station_id"].eq(station_id)].drop(columns=["observed_source"], errors="ignore")
        new_observed = [column for column in observed_station if column.startswith("observed_") and column not in base]
        merged = merged.merge(
            observed_station[["station_id", "contract_date", *new_observed]],
            on=["station_id", "contract_date"],
            how="left",
        )
        frames[station_id] = merged
    return frames


def make_variant(
    frame: pd.DataFrame,
    variant: str,
    *,
    admitted_enriched_features: Iterable[str],
) -> pd.DataFrame:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}")
    admitted = set(admitted_enriched_features)
    base_columns = [
        column
        for column in frame
        if not column.endswith("_enriched_forecast")
        and not _is_enriched_column(column)
        and not _legacy_optional_forecast_feature(column)
    ]
    base_columns = [column for column in base_columns if not _forbidden(column)]
    observation = {column for column in admitted if column.startswith("observed_")}
    forecast = admitted - observation
    selected = set(base_columns)
    if variant in {"observation_only", "combined"}:
        selected |= observation
    if variant in {"forecast_only", "combined"}:
        selected |= forecast
    out = frame[[column for column in frame if column in selected]].copy()
    if variant in {"forecast_only", "combined"}:
        out = add_cross_provider_features(out, admitted=selected)
    if variant == "combined":
        out = add_forecast_observation_deltas(out, admitted=selected)
    return out


def frozen_validation_predictions(
    features: pd.DataFrame,
    config: StationStackingConfig,
    selected_hyperparameters: pd.DataFrame,
) -> pd.DataFrame:
    modeling, categorical, numeric = _enriched_modeling_frame(features, config)
    baseline = year_split_baseline_predictions(modeling, config, config.effective_year_split_folds)
    modeled = _validation_predictions_for_selected_params(
        modeling,
        config,
        categorical,
        numeric,
        config.effective_year_split_folds,
        selected_hyperparameters,
    )
    predictions = pd.concat([baseline, modeled], ignore_index=True)
    predictions["station_id"] = config.station_id.upper()
    return predictions


def full_tuned_experiment_from_features(
    features: pd.DataFrame,
    config: StationStackingConfig,
    *,
    artifact_dir: str | Path,
) -> dict[str, pd.DataFrame]:
    """Run the existing V11 architecture on a prebuilt, gated feature frame."""
    folds = config.effective_year_split_folds
    modeling, categorical, numeric = _enriched_modeling_frame(features, config)
    baseline_validation = year_split_baseline_predictions(modeling, config, folds)
    tuning, validation, selected = tune_year_split_base_models(modeling, config, categorical, numeric, folds)
    test = year_split_test_predictions(
        modeling,
        config,
        categorical,
        numeric,
        selected,
        train_years=config.effective_year_split_test_train_years,
        test_year=config.effective_year_split_test_year,
    )
    all_validation = pd.concat([baseline_validation, validation], ignore_index=True)
    stack_test, stack_tuning = tune_year_split_stack_model(
        all_validation,
        test,
        config,
        test_year=config.effective_year_split_test_year,
    )
    if not stack_test.empty:
        test = pd.concat([test, stack_test], ignore_index=True)
    importance = year_split_feature_importance(
        modeling,
        config,
        categorical,
        numeric,
        selected,
        train_years=config.effective_year_split_test_train_years,
        test_year=config.effective_year_split_test_year,
    )
    metrics = summarize_year_split_predictions(all_validation, test)
    scoreboard = year_split_scoreboard(all_validation, test)
    brackets = year_split_bracket_predictions(test, test_year=config.effective_year_split_test_year)
    bracket_metrics = year_split_bracket_metrics(brackets)
    all_validation["station_id"] = config.station_id.upper()
    test["station_id"] = config.station_id.upper()
    extended = extended_prediction_metrics(pd.concat([all_validation, test], ignore_index=True))
    artifacts = {
        "features": features,
        "validation_predictions": all_validation,
        "test_predictions": test,
        "tuning": tuning,
        "selected_hyperparameters": selected,
        "stack_tuning": stack_tuning,
        "feature_importance": importance,
        "metrics": metrics,
        "extended_metrics": extended,
        "scoreboard": scoreboard,
        "bracket_predictions": brackets,
        "bracket_metrics": bracket_metrics,
        "feature_columns": pd.DataFrame(
            [{"feature": value, "kind": "categorical"} for value in categorical]
            + [{"feature": value, "kind": "numeric"} for value in numeric]
        ),
    }
    output = Path(artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    station = config.station_id.upper()
    for name, artifact in artifacts.items():
        artifact.to_csv(output / f"{station}_{name}.csv", index=False)
    return artifacts


def _wide_forecast(frame: pd.DataFrame) -> pd.DataFrame:
    identity = ["station_id", "contract_date"]
    value_columns = [
        column
        for column in frame.columns
        if column not in {*identity, "provider", "issued_at", "forecast_as_of", "forecast_window_start", "forecast_window_end", "fetch_status"}
    ]
    pieces: list[pd.DataFrame] = []
    for provider in PROVIDERS:
        part = frame.loc[frame["provider"].eq(provider), [*identity, *value_columns]].copy()
        part = part.rename(columns={column: f"{provider}_{column}" for column in value_columns})
        pieces.append(part)
    output = pieces[0]
    for part in pieces[1:]:
        output = output.merge(part, on=identity, how="outer")
    return output


def _is_enriched_column(column: str) -> bool:
    forecast_markers = (
        "dewpoint_at_11am_f",
        "dewpoint_remaining_mean_f",
        "humidity_at_11am_pct",
        "humidity_remaining_mean_pct",
        "precip_total_mm",
        "precip_max_hourly_mm",
        "precip_wet_hour_count",
        "precip_any",
        "cloud_at_11am_pct",
        "cloud_remaining_mean_pct",
        "cloud_remaining_max_pct",
        "wind_u_at_11am_mph",
        "wind_v_at_11am_mph",
        "wind_speed_at_11am_mph",
        "wind_speed_remaining_mean_mph",
        "wind_speed_remaining_max_mph",
        "wind_vector_mean_direction_sin",
        "wind_vector_mean_direction_cos",
    )
    observed_markers = (
        "_change_1h",
        "_change_3h",
        "temperature_acceleration",
        "morning_temperature_range",
        "minutes_since_high",
        "calm_wind",
        "variable_wind",
        "cloud_category",
        "ceiling_present",
        "wind_u_at_as_of",
        "wind_v_at_as_of",
    )
    return any(marker in column for marker in (*forecast_markers, *observed_markers))


def _forbidden(column: str) -> bool:
    return column in FORBIDDEN_EXACT or any(pattern in column for pattern in FORBIDDEN_PATTERNS)


def _legacy_optional_forecast_feature(column: str) -> bool:
    provider_related = any(column.startswith(f"{provider}_") for provider in PROVIDERS) or column.startswith("provider_")
    pair_related = any(column.startswith(f"{left}_{right}_") for left in PROVIDERS for right in PROVIDERS if left != right)
    weather_related = any(
        marker in column
        for marker in ("dewpoint", "humidity", "precip", "cloud", "wind", "visibility", "ceiling", "pressure", "radiation")
    )
    return (provider_related or pair_related) and weather_related


def _enriched_modeling_frame(
    features: pd.DataFrame,
    config: StationStackingConfig,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    modeling, categorical, numeric = _modeling_frame(features, config)
    for column in ("observed_cloud_category",):
        if column in modeling:
            modeling[column] = modeling[column].astype("string").fillna("missing")
            if column not in categorical:
                categorical.append(column)
            numeric = [value for value in numeric if value != column]
    return modeling, categorical, numeric
