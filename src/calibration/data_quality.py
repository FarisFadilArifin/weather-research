from __future__ import annotations

from typing import Iterable

import pandas as pd


STRICT_QUALITY_OK_COLUMN = "strict_quality_ok"
STRICT_QUALITY_ISSUES_COLUMN = "strict_quality_issues"

MIN_ACTUAL_OBSERVATION_COUNT = 18
MIN_PLAUSIBLE_TEMP_F = -80.0
MAX_PLAUSIBLE_TEMP_F = 140.0
MAX_CURRENT_OBSERVATION_AGE_MINUTES = 20.0


def add_strict_quality_flags(
    frame: pd.DataFrame,
    providers: Iterable[str] | None = None,
    *,
    compare_observation_to_actual: bool = True,
) -> pd.DataFrame:
    """Annotate rows that should be excluded from strict training/evaluation."""
    out = frame.copy()
    if out.empty:
        out[STRICT_QUALITY_OK_COLUMN] = pd.Series(dtype=bool)
        out[STRICT_QUALITY_ISSUES_COLUMN] = pd.Series(dtype="string")
        return out

    issues: dict[str, pd.Series] = {}
    actual_high = _numeric(out, "actual_high_f")
    if "actual_high_f" in out:
        issues["missing_actual_high"] = actual_high.isna()
        issues["actual_high_out_of_range"] = actual_high.notna() & ~_plausible_temperature(actual_high)

    quality = _first_text(out, ["actual_data_quality_flag", "data_quality_flag"])
    if quality is not None:
        normalized = quality.astype("string").str.strip().str.lower()
        present = normalized.notna() & normalized.ne("") & normalized.ne("<na>") & normalized.ne("nan")
        issues["actual_quality_not_ok"] = present & normalized.ne("ok")

    actual_count = _first_numeric(out, ["actual_raw_observation_count", "raw_observation_count", "obs_count"])
    if actual_count is not None:
        issues["actual_observation_count_low"] = actual_count.notna() & actual_count.lt(MIN_ACTUAL_OBSERVATION_COUNT)

    observed_status = _first_text(out, ["observed_fetch_status"])
    if observed_status is not None:
        normalized_status = observed_status.astype("string").str.strip().str.lower()
        status_present = normalized_status.notna() & normalized_status.ne("") & normalized_status.ne("<na>") & normalized_status.ne("nan")
        status_ok = status_present & normalized_status.eq("ok")
        issues["observed_missing_fetch_status"] = ~status_present
        issues["observed_fetch_not_ok"] = status_present & ~normalized_status.eq("ok")
    else:
        status_ok = pd.Series(False, index=out.index)

    observed_temp = _numeric(out, "observed_temp_at_as_of_f")
    if "observed_temp_at_as_of_f" in out:
        issues["observed_ok_missing_temp"] = status_ok & observed_temp.isna()
        issues["observed_temp_out_of_range"] = observed_temp.notna() & ~_plausible_temperature(observed_temp)
        if compare_observation_to_actual and "actual_high_f" in out:
            issues["observed_temp_above_actual_high"] = observed_temp.notna() & actual_high.notna() & observed_temp.gt(actual_high)

    observed_high = _numeric(out, "observed_high_temp_through_as_of_f")
    if "observed_high_temp_through_as_of_f" in out:
        issues["observed_ok_missing_high_so_far"] = status_ok & observed_high.isna()
        issues["observed_high_out_of_range"] = observed_high.notna() & ~_plausible_temperature(observed_high)
        if compare_observation_to_actual and "actual_high_f" in out:
            issues["actual_below_observed_high_so_far"] = (
                observed_high.notna() & actual_high.notna() & actual_high.lt(observed_high)
            )

    observed_age = _numeric(out, "observed_as_of_age_minutes")
    if "observed_as_of_age_minutes" in out:
        issues["observed_stale_age_gt_20_minutes"] = observed_age.notna() & observed_age.gt(MAX_CURRENT_OBSERVATION_AGE_MINUTES)

    raw_forecast_high = _numeric(out, "raw_forecast_high_f")
    if "raw_forecast_high_f" in out:
        issues["forecast_high_out_of_range"] = raw_forecast_high.notna() & ~_plausible_temperature(raw_forecast_high)

    for provider in providers or ():
        high_col = f"{str(provider).lower()}_high_f"
        if high_col not in out:
            continue
        high = pd.to_numeric(out[high_col], errors="coerce")
        issues[f"{high_col}_out_of_range"] = high.notna() & ~_plausible_temperature(high)

    if issues:
        issue_frame = pd.DataFrame(issues, index=out.index).fillna(False).astype(bool)
        out[STRICT_QUALITY_OK_COLUMN] = ~issue_frame.any(axis=1)
        out[STRICT_QUALITY_ISSUES_COLUMN] = _join_issue_names(issue_frame)
    else:
        out[STRICT_QUALITY_OK_COLUMN] = True
        out[STRICT_QUALITY_ISSUES_COLUMN] = ""
    return out


def filter_strict_training_rows(
    frame: pd.DataFrame,
    providers: Iterable[str] | None = None,
    require_provider_highs: bool = False,
) -> pd.DataFrame:
    out = add_strict_quality_flags(frame, providers=providers)
    mask = out[STRICT_QUALITY_OK_COLUMN].fillna(False)
    if require_provider_highs:
        high_cols = [f"{str(provider).lower()}_high_f" for provider in providers or ()]
        high_cols = [column for column in high_cols if column in out]
        if high_cols:
            mask &= out[high_cols].notna().all(axis=1)
    return out.loc[mask].copy()


def plausible_temperature_mask(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.notna() & _plausible_temperature(values)


def _plausible_temperature(values: pd.Series) -> pd.Series:
    return values.between(MIN_PLAUSIBLE_TEMP_F, MAX_PLAUSIBLE_TEMP_F, inclusive="both")


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(pd.NA, index=frame.index, dtype="Float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _first_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.Series | None:
    for column in columns:
        if column in frame:
            return pd.to_numeric(frame[column], errors="coerce")
    return None


def _first_text(frame: pd.DataFrame, columns: list[str]) -> pd.Series | None:
    for column in columns:
        if column in frame:
            return frame[column]
    return None


def _join_issue_names(issue_frame: pd.DataFrame) -> pd.Series:
    names = issue_frame.columns.to_list()
    values: list[str] = []
    for row in issue_frame.itertuples(index=False, name=None):
        values.append(";".join(name for name, flagged in zip(names, row, strict=False) if flagged))
    return pd.Series(values, index=issue_frame.index, dtype="string")
