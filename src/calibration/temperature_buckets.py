from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


POLYMARKET_CELSIUS_1C = "polymarket_half_up_1c"
POLYMARKET_FAHRENHEIT_2F = "polymarket_half_up_2f"
FLOOR_CELSIUS_1C = "floor_1c"
SUPPORTED_BUCKET_CONTRACTS = frozenset(
    {
        POLYMARKET_CELSIUS_1C,
        POLYMARKET_FAHRENHEIT_2F,
        FLOOR_CELSIUS_1C,
    }
)


def validate_bucket_contract(value: str) -> str:
    contract = str(value).strip().lower()
    if contract not in SUPPORTED_BUCKET_CONTRACTS:
        supported = ", ".join(sorted(SUPPORTED_BUCKET_CONTRACTS))
        raise ValueError(f"unsupported bucket contract {value!r}; expected one of: {supported}")
    return contract


def fahrenheit_to_celsius(values: Any) -> Any:
    return (values - 32.0) * 5.0 / 9.0


def _round_half_up(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return np.floor(numeric + 0.5).astype("Int64")


def _actual_celsius(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    actual = pd.Series(np.nan, index=frame.index, dtype=float)
    source = pd.Series("actual_high_f_converted_to_c", index=frame.index, dtype="string")
    for column in ("actual_high_c", "settlement_high_c"):
        if column not in frame:
            continue
        candidate = pd.to_numeric(frame[column], errors="coerce")
        use = actual.isna() & candidate.notna()
        actual.loc[use] = candidate.loc[use]
        source.loc[use] = column
    fallback = fahrenheit_to_celsius(
        pd.to_numeric(frame["actual_high_f"], errors="coerce")
    )
    return actual.fillna(fallback), source


def point_bucket_predictions(
    predictions: pd.DataFrame,
    bucket_contract: str,
) -> pd.DataFrame:
    """Map continuous point predictions to the market's exact settlement buckets."""
    contract = validate_bucket_contract(bucket_contract)
    required = {"actual_high_f", "predicted_high_f"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError("point predictions are missing columns: " + ", ".join(missing))

    frame = predictions.copy()
    actual_f = pd.to_numeric(frame["actual_high_f"], errors="coerce")
    predicted_f = pd.to_numeric(frame["predicted_high_f"], errors="coerce")
    frame["actual_high_f"] = actual_f
    frame["predicted_high_f"] = predicted_f

    if contract == POLYMARKET_FAHRENHEIT_2F:
        actual_degree = _round_half_up(actual_f)
        predicted_degree = _round_half_up(predicted_f)
        frame["actual_market_bucket"] = actual_degree.map(
            lambda value: pd.NA if pd.isna(value) else int(value) - int(value) % 2
        ).astype("Int64")
        frame["predicted_market_bucket"] = predicted_degree.map(
            lambda value: pd.NA if pd.isna(value) else int(value) - int(value) % 2
        ).astype("Int64")
        bucket_width = 2.0
        frame["actual_bucket_source"] = "actual_high_f"
    else:
        actual_c, actual_source = _actual_celsius(frame)
        predicted_c = fahrenheit_to_celsius(predicted_f)
        frame["actual_high_c"] = actual_c
        frame["predicted_high_c"] = predicted_c
        frame["actual_bucket_source"] = actual_source
        if contract == POLYMARKET_CELSIUS_1C:
            frame["actual_market_bucket"] = _round_half_up(actual_c)
            frame["predicted_market_bucket"] = _round_half_up(predicted_c)
        else:
            frame["actual_market_bucket"] = np.floor(actual_c).astype("Int64")
            frame["predicted_market_bucket"] = np.floor(predicted_c).astype("Int64")
        bucket_width = 1.0

    valid = frame["actual_market_bucket"].notna() & frame["predicted_market_bucket"].notna()
    frame["bucket_error"] = (
        frame["predicted_market_bucket"] - frame["actual_market_bucket"]
    ).astype("Float64") / bucket_width
    frame["bucket_hit"] = (
        frame["predicted_market_bucket"].eq(frame["actual_market_bucket"])
        .where(valid)
        .astype("boolean")
    )
    frame["within_one_bucket"] = (
        frame["bucket_error"].abs().le(1.0).where(valid).astype("boolean")
    )
    frame["bucket_contract"] = contract
    return frame


def point_bucket_metrics(
    predictions: pd.DataFrame,
    bucket_contract: str,
) -> pd.DataFrame:
    """Return an exact point-bucket scoreboard, independent of probability models."""
    frame = point_bucket_predictions(predictions, bucket_contract)
    valid = frame.loc[frame["bucket_hit"].notna()].copy()
    if valid.empty:
        return pd.DataFrame()
    hits = int(valid["bucket_hit"].sum())
    count = int(len(valid))
    within_one = float(valid["within_one_bucket"].mean())
    errors = pd.to_numeric(valid["bucket_error"], errors="coerce")
    return pd.DataFrame(
        [
            {
                "bucket_contract": validate_bucket_contract(bucket_contract),
                "count": count,
                "bucket_hits": hits,
                "exact_bucket_hit_rate": hits / count,
                "exact_bucket_hit_pct": hits / count * 100.0,
                "within_one_bucket_rate": within_one,
                "within_one_bucket_pct": within_one * 100.0,
                "mean_bucket_error": float(errors.mean()),
                "mean_absolute_bucket_error": float(errors.abs().mean()),
                "first_contract_date": (
                    str(pd.to_datetime(valid["contract_date"]).min().date())
                    if "contract_date" in valid
                    else None
                ),
                "last_contract_date": (
                    str(pd.to_datetime(valid["contract_date"]).max().date())
                    if "contract_date" in valid
                    else None
                ),
            }
        ]
    )
