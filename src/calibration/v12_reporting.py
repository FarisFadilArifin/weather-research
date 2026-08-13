from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .station_stacking import (
    GUARDED_BLEND_CAPS_F,
    TARGET,
    TARGET_STATIONS,
    round_temperature_half_up,
    select_guarded_blend_cap,
)


ACTIVE_V12_GATE_STATIONS = ("KATL", "KDAL", "KMIA", "KORD", "KSEA")
V12_REQUIRED_LABEL_START = "2026-05-19"
V12_REQUIRED_LABEL_END = "2026-06-21"
V12_MODEL_VERSION = "station_high_regressor_v12_guarded_blend"
V12_HARD_LIVE_FEATURES = (
    "gfs_high_f",
    "hrrr_high_f",
    "nbm_high_f",
    "observed_temp_at_as_of_f",
    "observed_high_temp_through_as_of_f",
    "observed_as_of_age_minutes",
)


@dataclass(frozen=True)
class V12ReportSummary:
    selected_cap_f: float | None
    selected_method: str | None
    acceptance_passed: bool
    provider_mean_mae_f: float | None
    selected_mae_f: float | None
    missing_required_settlement_labels: int


def write_v12_research_artifacts(
    artifact_dir: str | Path,
    *,
    stations: Iterable[str] = TARGET_STATIONS,
    required_label_stations: Iterable[str] = ACTIVE_V12_GATE_STATIONS,
    required_label_start: str = V12_REQUIRED_LABEL_START,
    required_label_end: str = V12_REQUIRED_LABEL_END,
    model_version: str = V12_MODEL_VERSION,
) -> V12ReportSummary:
    artifacts = Path(artifact_dir)
    station_list = tuple(str(station).upper() for station in stations)
    features = _read_station_frames(artifacts, station_list, "features")
    test_predictions = _read_station_frames(artifacts, station_list, "year_split_test_predictions")
    bracket_predictions = _read_station_frames(artifacts, station_list, "year_split_bracket_predictions")

    target_comparison = build_target_source_comparison(features, required_label_stations, required_label_start, required_label_end)
    live_audit = build_live_equivalent_feature_audit(features)
    baseline_comparison = build_provider_baseline_comparison(test_predictions)
    monthly_metrics = build_monthly_station_metrics(test_predictions)
    harmful = build_harmful_corrections(test_predictions)
    rounded = build_rounded_bucket_backtest(bracket_predictions if not bracket_predictions.empty else test_predictions)
    cap_selection = select_guarded_blend_cap(test_predictions, caps_f=GUARDED_BLEND_CAPS_F)

    target_comparison.to_csv(artifacts / "v12_target_source_comparison.csv", index=False)
    live_audit.to_csv(artifacts / "v12_live_equivalent_feature_audit.csv", index=False)
    baseline_comparison.to_csv(artifacts / "v12_provider_baseline_comparison.csv", index=False)
    monthly_metrics.to_csv(artifacts / "v12_monthly_station_metrics.csv", index=False)
    harmful.to_csv(artifacts / "v12_harmful_corrections.csv", index=False)
    rounded.to_csv(artifacts / "v12_rounded_bucket_backtest.csv", index=False)
    cap_selection.to_csv(artifacts / "v12_guarded_cap_selection.csv", index=False)

    summary = summarize_v12_acceptance(target_comparison, cap_selection)
    _write_handoff(
        artifacts / "v12_candidate_model_handoff.md",
        summary=summary,
        model_version=model_version,
        required_label_start=required_label_start,
        required_label_end=required_label_end,
    )
    return summary


def build_target_source_comparison(
    features: pd.DataFrame,
    required_label_stations: Iterable[str] = ACTIVE_V12_GATE_STATIONS,
    required_label_start: str = V12_REQUIRED_LABEL_START,
    required_label_end: str = V12_REQUIRED_LABEL_END,
) -> pd.DataFrame:
    columns = [
        "station_id",
        "contract_date",
        "actual_high_f",
        "iem_actual_high_f",
        "settlement_high_f",
        "target_source",
        "target_source_diff_f",
        "settlement_source",
        "settlement_quality_flag",
        "is_required_label_window",
        "missing_required_settlement_label",
    ]
    if features.empty:
        return pd.DataFrame(columns=columns)
    out = features.copy()
    for column in columns:
        if column not in out:
            out[column] = pd.NA
    dates = pd.to_datetime(out["contract_date"], errors="coerce")
    required_stations = {str(station).upper() for station in required_label_stations}
    required = (
        out["station_id"].astype(str).str.upper().isin(required_stations)
        & dates.between(pd.Timestamp(required_label_start), pd.Timestamp(required_label_end))
    )
    out["is_required_label_window"] = required
    out["missing_required_settlement_label"] = required & pd.to_numeric(out["settlement_high_f"], errors="coerce").isna()
    return out[columns].sort_values(["station_id", "contract_date"]).reset_index(drop=True)


def build_live_equivalent_feature_audit(features: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "station_id",
        "row_count",
        "first_contract_date",
        "last_contract_date",
        "hard_feature",
        "non_null_rows",
        "missing_rows",
        "missing_pct",
        "observations_in_1040_1100_local_rows",
    ]
    if features.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for station, group in features.groupby("station_id", dropna=False):
        obs_window_rows = _observation_window_ok_count(group)
        for feature in V12_HARD_LIVE_FEATURES:
            if feature in group:
                non_null = int(pd.to_numeric(group[feature], errors="coerce").notna().sum())
            else:
                non_null = 0
            rows.append(
                {
                    "station_id": station,
                    "row_count": int(len(group)),
                    "first_contract_date": str(group["contract_date"].min()),
                    "last_contract_date": str(group["contract_date"].max()),
                    "hard_feature": feature,
                    "non_null_rows": non_null,
                    "missing_rows": int(len(group) - non_null),
                    "missing_pct": float((len(group) - non_null) / len(group) * 100.0) if len(group) else 100.0,
                    "observations_in_1040_1100_local_rows": obs_window_rows,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def build_provider_baseline_comparison(test_predictions: pd.DataFrame) -> pd.DataFrame:
    columns = ["scope", "method", "count", "mae_f", "rmse_f", "bias_f", "bucket_hit_pct"]
    if test_predictions.empty:
        return pd.DataFrame(columns=columns)
    frame = test_predictions.loc[test_predictions["evaluation_scope"].eq("year_split_test")].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    return _metrics_by_scope(frame, ["method"], scope="all_2026")[columns]


def build_monthly_station_metrics(test_predictions: pd.DataFrame) -> pd.DataFrame:
    columns = ["scope", "station_id", "month", "method", "count", "mae_f", "rmse_f", "bias_f", "bucket_hit_pct"]
    if test_predictions.empty:
        return pd.DataFrame(columns=columns)
    frame = test_predictions.loc[test_predictions["evaluation_scope"].eq("year_split_test")].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame["month"] = pd.to_datetime(frame["contract_date"], errors="coerce").dt.month.astype("Int64")
    metrics = _metrics_by_scope(frame, ["station_id", "month", "method"], scope="station_month_2026")
    return metrics[columns].sort_values(["station_id", "month", "method"]).reset_index(drop=True)


def build_harmful_corrections(test_predictions: pd.DataFrame, ml_method: str = "ridge_stack") -> pd.DataFrame:
    columns = [
        "station_id",
        "contract_date",
        "actual_high_f",
        "provider_mean_predicted_high_f",
        "ml_predicted_high_f",
        "provider_mean_absolute_error_f",
        "ml_absolute_error_f",
        "ml_minus_provider_mean_f",
    ]
    if test_predictions.empty:
        return pd.DataFrame(columns=columns)
    frame = test_predictions.loc[test_predictions["method"].isin(["provider_mean", ml_method])].copy()
    key_columns = ["station_id", "contract_date"] if "station_id" in frame else ["contract_date"]
    pivot = frame.pivot_table(index=key_columns, columns="method", values="predicted_high_f", aggfunc="first")
    if "provider_mean" not in pivot or ml_method not in pivot:
        return pd.DataFrame(columns=columns)
    actuals = frame.groupby(key_columns, dropna=False)[TARGET].first()
    out = pivot.join(actuals).reset_index()
    out["provider_mean_absolute_error_f"] = (out[TARGET] - out["provider_mean"]).abs()
    out["ml_absolute_error_f"] = (out[TARGET] - out[ml_method]).abs()
    out = out.loc[out["ml_absolute_error_f"].gt(out["provider_mean_absolute_error_f"])].copy()
    out["ml_minus_provider_mean_f"] = out[ml_method] - out["provider_mean"]
    out = out.rename(
        columns={
            TARGET: "actual_high_f",
            "provider_mean": "provider_mean_predicted_high_f",
            ml_method: "ml_predicted_high_f",
        }
    )
    for column in columns:
        if column not in out:
            out[column] = pd.NA
    return out[columns].sort_values(["station_id", "contract_date"] if "station_id" in out else ["contract_date"])


def build_rounded_bucket_backtest(predictions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scope",
        "method",
        "count",
        "mae_f",
        "rmse_f",
        "bias_f",
        "bucket_hit_pct",
        "two_bucket_package_hit_pct",
    ]
    if predictions.empty:
        return pd.DataFrame(columns=columns)
    frame = predictions.copy()
    if "bracket_hit" in frame:
        frame["bucket_hit"] = frame["bracket_hit"]
    else:
        frame["actual_rounded_high_f"] = frame[TARGET].map(round_temperature_half_up)
        frame["predicted_rounded_high_f"] = frame["predicted_high_f"].map(round_temperature_half_up)
        frame["bucket_hit"] = frame["actual_rounded_high_f"].eq(frame["predicted_rounded_high_f"])
    frame["two_bucket_package_hit"] = _two_bucket_package_hit(frame)
    metrics = _metrics_by_scope(frame, ["method"], scope="all_2026")
    two_bucket = (
        frame.groupby("method", dropna=False)["two_bucket_package_hit"]
        .mean()
        .mul(100.0)
        .rename("two_bucket_package_hit_pct")
        .reset_index()
    )
    return metrics.merge(two_bucket, on="method", how="left")[columns]


def summarize_v12_acceptance(target_comparison: pd.DataFrame, cap_selection: pd.DataFrame) -> V12ReportSummary:
    missing_required = (
        int(target_comparison["missing_required_settlement_label"].fillna(False).sum())
        if "missing_required_settlement_label" in target_comparison
        else 0
    )
    if cap_selection.empty:
        return V12ReportSummary(None, None, False, None, None, missing_required)
    selected = cap_selection.iloc[0]
    selected_mae = float(selected["mae_f"])
    provider_mae = float(selected["provider_mean_mae_f"])
    passed = bool(selected_mae < provider_mae and missing_required == 0)
    return V12ReportSummary(
        selected_cap_f=float(selected["cap_f"]),
        selected_method=str(selected["method"]),
        acceptance_passed=passed,
        provider_mean_mae_f=provider_mae,
        selected_mae_f=selected_mae,
        missing_required_settlement_labels=missing_required,
    )


def _read_station_frames(artifact_dir: Path, stations: Iterable[str], suffix: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for station in stations:
        path = artifact_dir / f"{station}_{suffix}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, low_memory=False)
        if frame.empty:
            continue
        frame["station_id"] = station
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _metrics_by_scope(frame: pd.DataFrame, group_columns: list[str], *, scope: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, group in frame.groupby(group_columns, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        error = pd.to_numeric(group["error_f"], errors="coerce")
        abs_error = error.abs()
        bucket_hit = _bucket_hit(group)
        row = {
            "scope": scope,
            "count": int(error.notna().sum()),
            "mae_f": float(abs_error.mean()),
            "rmse_f": float(np.sqrt((error**2).mean())),
            "bias_f": float(error.mean()),
            "bucket_hit_pct": float(bucket_hit.mean() * 100.0),
        }
        row.update(dict(zip(group_columns, key_values, strict=False)))
        rows.append(row)
    return pd.DataFrame(rows)


def _bucket_hit(frame: pd.DataFrame) -> pd.Series:
    if "bracket_hit" in frame:
        return frame["bracket_hit"].astype("boolean")
    actual = frame[TARGET].map(round_temperature_half_up)
    predicted = frame["predicted_high_f"].map(round_temperature_half_up)
    return actual.eq(predicted).astype("boolean")


def _two_bucket_package_hit(frame: pd.DataFrame) -> pd.Series:
    actual = frame[TARGET].map(round_temperature_half_up)
    predicted = frame["predicted_high_f"].map(round_temperature_half_up)
    values = []
    for actual_value, predicted_value in zip(actual, predicted, strict=False):
        if pd.isna(actual_value) or pd.isna(predicted_value):
            values.append(pd.NA)
            continue
        values.append(abs(int(actual_value) - int(predicted_value)) <= 1)
    return pd.Series(values, index=frame.index, dtype="boolean")


def _observation_window_ok_count(frame: pd.DataFrame) -> int:
    if "observed_as_of_time_local" not in frame:
        return 0
    times = pd.to_datetime(frame["observed_as_of_time_local"], errors="coerce")
    minutes = times.dt.hour.mul(60).add(times.dt.minute)
    return int(minutes.between(10 * 60 + 40, 11 * 60).sum())


def _write_handoff(
    path: Path,
    *,
    summary: V12ReportSummary,
    model_version: str,
    required_label_start: str,
    required_label_end: str,
) -> None:
    status = "PASS" if summary.acceptance_passed else "BLOCKED"
    lines = [
        "# V12 Candidate Model Handoff",
        "",
        f"Model version: `{model_version}`",
        "",
        f"Acceptance status: **{status}**",
        "",
        f"Selected guarded method: `{summary.selected_method or 'none'}`",
        f"Selected cap F: `{summary.selected_cap_f if summary.selected_cap_f is not None else 'none'}`",
        f"Selected MAE F: `{summary.selected_mae_f if summary.selected_mae_f is not None else 'n/a'}`",
        f"Provider mean MAE F: `{summary.provider_mean_mae_f if summary.provider_mean_mae_f is not None else 'n/a'}`",
        f"Missing required settlement labels: `{summary.missing_required_settlement_labels}`",
        "",
        "Primary gate: selected v12 guarded blend must beat provider mean on all 9 supported stations across 2026 common-date rows.",
        f"Required settlement-label window: `{required_label_start}` through `{required_label_end}` for active stations.",
        "",
        "Generated artifacts:",
        "",
        "- `v12_target_source_comparison.csv`",
        "- `v12_live_equivalent_feature_audit.csv`",
        "- `v12_provider_baseline_comparison.csv`",
        "- `v12_monthly_station_metrics.csv`",
        "- `v12_harmful_corrections.csv`",
        "- `v12_rounded_bucket_backtest.csv`",
        "- `v12_guarded_cap_selection.csv`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
