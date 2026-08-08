from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import binomtest


DEVELOPMENT_YEARS = (2023, 2024, 2025)
CONFIRMATION_YEAR = 2026
MIN_RESIDUAL_YEAR_COUNT = 20
MIN_UTILITY_YEAR_DECISIONS = 8
MIN_CONFIRMATION_RESIDUAL_COUNT = 15
MIN_CONFIRMATION_UTILITY_DECISIONS = 5
GROUP_SOURCE_COLUMNS = {
    "month",
    "prediction_fraction_f",
    "default_half_up",
    "base_prediction_spread_f",
    "provider_spread_high_f",
    "base_mean_minus_point_f",
    "provider_mean_minus_point_f",
    "prior_residual_bias_30d_f",
    "point_minus_observed_high_f",
    "observed_cloud_cover_at_as_of",
    "observed_precip_recent_at_as_of",
    "point_prediction_f",
    "models_supporting_alternative_bucket",
    "models_supporting_default_bucket",
    "continuous_alternative_probability_advantage_180d",
}
TARGET_COLUMNS = {
    "actual_high_f",
    "actual_bucket_label",
    "continuous_residual_f",
    "residual_f",
    "underprediction_target",
    "recovery_target",
    "damage_target",
    "realized_override_utility",
}


def add_directional_audit_groups(frame: pd.DataFrame) -> pd.DataFrame:
    """Add fixed, target-independent diagnostic groups; no row is filtered."""
    out = frame.copy()
    dates = pd.to_datetime(out["contract_date"])
    out["audit_month"] = dates.dt.month.map(lambda value: f"{int(value):02d}")
    out["audit_season"] = dates.dt.month.map(_season)
    out["audit_fraction_quarter"] = pd.cut(
        out["prediction_fraction_f"],
        [-np.inf, 0.25, 0.50, 0.75, np.inf],
        labels=("Q1_[0,.25]", "Q2_(.25,.50]", "Q3_(.50,.75]", "Q4_(.75,1)"),
    ).astype(str)
    out["audit_half_up_direction"] = np.where(out["default_half_up"].eq(1), "up", "down")
    out["audit_base_spread"] = _fixed_cut(
        out["base_prediction_spread_f"], [-np.inf, 0.5, 1.0, np.inf],
        ("low_<=0.5", "medium_(0.5,1]", "high_>1"),
    )
    out["audit_provider_spread"] = _fixed_cut(
        out["provider_spread_high_f"], [-np.inf, 3.0, 6.0, np.inf],
        ("low_<=3", "medium_(3,6]", "high_>6"),
    )
    out["audit_base_direction"] = _fixed_cut(
        out["base_mean_minus_point_f"], [-np.inf, -0.25, 0.25, np.inf],
        ("lower_<-0.25", "neutral_[-0.25,0.25]", "higher_>0.25"),
    )
    out["audit_provider_direction"] = _fixed_cut(
        out["provider_mean_minus_point_f"], [-np.inf, -1.0, 0.0, 1.0, np.inf],
        ("strong_lower_<-1", "lower_[-1,0]", "higher_(0,1]", "strong_higher_>1"),
    )
    out["audit_recent_bias"] = _fixed_cut(
        out["prior_residual_bias_30d_f"], [-np.inf, -0.5, 0.0, 0.5, np.inf],
        ("negative_<-0.5", "slight_negative_[-0.5,0]", "slight_positive_(0,0.5]", "positive_>0.5"),
    )
    out["audit_observation_gap"] = _fixed_cut(
        out["point_minus_observed_high_f"], [-np.inf, 5.0, 8.0, 11.0, np.inf],
        ("<=5F", "(5,8]F", "(8,11]F", ">11F"),
    )
    cloud = pd.to_numeric(out["observed_cloud_cover_at_as_of"], errors="coerce")
    out["audit_cloud_regime"] = np.select(
        [cloud.le(25), cloud.ge(75)], ["clear_<=25", "cloudy_>=75"], default="mixed_(25,75)"
    )
    precipitation = pd.to_numeric(out["observed_precip_recent_at_as_of"], errors="coerce")
    out["audit_precip_regime"] = np.where(precipitation.gt(0), "wet", "dry")
    out["audit_temperature_band"] = _fixed_cut(
        out["point_prediction_f"], [-np.inf, 75.0, 85.0, 95.0, np.inf],
        ("<75F", "[75,85)F", "[85,95)F", ">=95F"),
    )
    support_balance = (
        pd.to_numeric(out["models_supporting_alternative_bucket"], errors="coerce")
        - pd.to_numeric(out["models_supporting_default_bucket"], errors="coerce")
    )
    out["audit_bucket_support_balance"] = _fixed_cut(
        support_balance, [-np.inf, -2.0, 1.0, np.inf],
        ("default_leads_by_2+", "close", "alternative_leads_by_2+"),
    )
    out["audit_continuous_advantage"] = _fixed_cut(
        out["continuous_alternative_probability_advantage_180d"],
        [-np.inf, -0.10, 0.0, 0.10, np.inf],
        ("default_>0.10", "default_(0,0.10]", "alternative_(0,0.10]", "alternative_>0.10"),
    )
    out["audit_season_x_base_direction"] = (
        out["audit_season"] + " | " + out["audit_base_direction"]
    )
    out["audit_fraction_x_base_direction"] = (
        out["audit_fraction_quarter"] + " | " + out["audit_base_direction"]
    )
    out["audit_bias_x_base_direction"] = (
        out["audit_recent_bias"] + " | " + out["audit_base_direction"]
    )
    return out


def audit_group_names() -> list[str]:
    return [
        "audit_month",
        "audit_season",
        "audit_fraction_quarter",
        "audit_half_up_direction",
        "audit_base_spread",
        "audit_provider_spread",
        "audit_base_direction",
        "audit_provider_direction",
        "audit_recent_bias",
        "audit_observation_gap",
        "audit_cloud_regime",
        "audit_precip_regime",
        "audit_temperature_band",
        "audit_bucket_support_balance",
        "audit_continuous_advantage",
        "audit_season_x_base_direction",
        "audit_fraction_x_base_direction",
        "audit_bias_x_base_direction",
    ]


def run_directional_residual_audit(
    frame: pd.DataFrame,
    *,
    station_id: str,
) -> dict[str, pd.DataFrame]:
    grouped = add_directional_audit_groups(frame)
    grouped["residual_f"] = grouped["actual_high_f"] - grouped["point_prediction_f"]
    grouped["underprediction_target"] = grouped["residual_f"].gt(0).astype(int)
    grouped["decisive_override"] = (
        grouped["recovery_target"] + grouped["damage_target"]
    ).eq(1)
    yearly = summarize_directional_groups(grouped, group_names=audit_group_names())
    residual_stability = _stability_table(
        yearly,
        metric="residual",
        minimum_year_count=MIN_RESIDUAL_YEAR_COUNT,
        minimum_confirmation_count=MIN_CONFIRMATION_RESIDUAL_COUNT,
    )
    utility_stability = _stability_table(
        yearly,
        metric="utility",
        minimum_year_count=MIN_UTILITY_YEAR_DECISIONS,
        minimum_confirmation_count=MIN_CONFIRMATION_UTILITY_DECISIONS,
    )
    for table in (yearly, residual_stability, utility_stability):
        table.insert(0, "station", station_id.upper())
    return {
        "frame": grouped,
        "yearly_group_metrics": yearly,
        "residual_stability": residual_stability,
        "utility_stability": utility_stability,
    }


def summarize_directional_groups(
    frame: pd.DataFrame,
    *,
    group_names: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_name in group_names:
        for (year, group_value), part in frame.groupby(["year", group_name], dropna=False):
            decisive = part.loc[part["decisive_override"]]
            recovery = int(decisive["recovery_target"].sum())
            damage = int(decisive["damage_target"].sum())
            rows.append(
                {
                    "year": int(year),
                    "group_name": group_name.removeprefix("audit_"),
                    "group_value": str(group_value),
                    "row_count": int(len(part)),
                    "underprediction_count": int(part["underprediction_target"].sum()),
                    "underprediction_rate": float(part["underprediction_target"].mean()),
                    "mean_residual_f": float(part["residual_f"].mean()),
                    "median_residual_f": float(part["residual_f"].median()),
                    "actionable_count": int(part["override_actionable"].sum()),
                    "decisive_override_count": int(len(decisive)),
                    "recovery_count": recovery,
                    "damage_count": damage,
                    "alternative_recovery_share": float(recovery / len(decisive))
                    if len(decisive)
                    else math.nan,
                    "mean_override_utility": float(
                        part.loc[part["override_actionable"].eq(1), "realized_override_utility"].mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def _stability_table(
    yearly: pd.DataFrame,
    *,
    metric: str,
    minimum_year_count: int,
    minimum_confirmation_count: int,
) -> pd.DataFrame:
    rows = []
    for (group_name, group_value), part in yearly.groupby(["group_name", "group_value"]):
        development = part.loc[part["year"].isin(DEVELOPMENT_YEARS)].set_index("year")
        if set(development.index) != set(DEVELOPMENT_YEARS):
            continue
        if metric == "residual":
            counts = development["row_count"]
            successes = development["underprediction_count"]
            rates = development["underprediction_rate"]
        else:
            counts = development["decisive_override_count"]
            successes = development["recovery_count"]
            rates = development["alternative_recovery_share"]
        total_count = int(counts.sum())
        total_success = int(successes.sum())
        combined_rate = total_success / total_count if total_count else math.nan
        directions = np.sign(rates.to_numpy(dtype=float) - 0.5)
        consistent = bool(np.all(directions == directions[0]) and directions[0] != 0)
        minimum_count_ok = bool(counts.ge(minimum_year_count).all())
        worst_year_edge = float(np.nanmin(np.abs(rates.to_numpy(dtype=float) - 0.5)))
        p_value = (
            float(binomtest(total_success, total_count, p=0.5).pvalue)
            if total_count
            else math.nan
        )
        confirmation = part.loc[part["year"].eq(CONFIRMATION_YEAR)]
        if confirmation.empty:
            confirmation_count = 0
            confirmation_rate = math.nan
        elif metric == "residual":
            confirmation_count = int(confirmation.iloc[0]["row_count"])
            confirmation_rate = float(confirmation.iloc[0]["underprediction_rate"])
        else:
            confirmation_count = int(confirmation.iloc[0]["decisive_override_count"])
            confirmation_rate = float(confirmation.iloc[0]["alternative_recovery_share"])
        expected_direction = "up" if combined_rate > 0.5 else "down"
        confirmation_direction = (
            "up" if confirmation_rate > 0.5 else "down" if confirmation_rate < 0.5 else "tie"
        ) if np.isfinite(confirmation_rate) else "missing"
        rows.append(
            {
                "group_name": group_name,
                "group_value": group_value,
                "development_count": total_count,
                "development_successes": total_success,
                "development_rate": combined_rate,
                "expected_direction": expected_direction,
                "minimum_year_count": int(counts.min()),
                "minimum_year_count_pass": minimum_count_ok,
                "consistent_direction_2023_2025": consistent,
                "worst_year_edge": worst_year_edge,
                "binomial_p_value": p_value,
                "confirmation_count_2026": confirmation_count,
                "confirmation_rate_2026": confirmation_rate,
                "confirmation_direction_2026": confirmation_direction,
                "confirmation_sample_pass": confirmation_count >= minimum_confirmation_count,
                "confirmation_matches_2026": (
                    confirmation_count >= minimum_confirmation_count
                    and confirmation_direction == expected_direction
                ),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["fdr_q_value"] = _benjamini_hochberg(table["binomial_p_value"].to_numpy(float))
    minimum_edge = 0.03 if metric == "residual" else 0.05
    table["stable_development_signal"] = (
        table["minimum_year_count_pass"]
        & table["consistent_direction_2023_2025"]
        & table["worst_year_edge"].ge(minimum_edge)
        & table["fdr_q_value"].lt(0.05)
    )
    table["confirmed_stable_signal"] = (
        table["stable_development_signal"] & table["confirmation_matches_2026"]
    )
    return table.sort_values(
        ["stable_development_signal", "fdr_q_value", "development_count"],
        ascending=[False, True, False],
        ignore_index=True,
    )


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    output = np.full(len(values), np.nan)
    finite_positions = np.flatnonzero(np.isfinite(values))
    if not len(finite_positions):
        return output
    finite = values[finite_positions]
    order = np.argsort(finite)
    ranked = finite[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0.0, 1.0)
    output[finite_positions] = restored
    return output


def audit_directional_residual_result(
    source_frame: pd.DataFrame,
    result: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    frame = result["frame"]
    residual = frame["actual_high_f"] - frame["point_prediction_f"]
    yearly = result["yearly_group_metrics"]
    residual_stability = result["residual_stability"]
    utility_stability = result["utility_stability"]
    rows = [
        ("residual_formula", np.allclose(frame["residual_f"], residual), "actual minus honest point prediction"),
        ("direction_target_formula", frame["underprediction_target"].eq(residual.gt(0).astype(int)).all(), "1 means actual exceeds prediction"),
        ("override_utility_formula", frame["realized_override_utility"].eq(frame["recovery_target"] - frame["damage_target"]).all(), "recovery minus damage"),
        ("group_sources_are_target_independent", GROUP_SOURCE_COLUMNS.isdisjoint(TARGET_COLUMNS), f"sources={len(GROUP_SOURCE_COLUMNS)}"),
        ("diagnostic_groups_do_not_filter_rows", len(frame) == len(source_frame), f"rows={len(frame)}"),
        ("development_selection_excludes_2026", set(DEVELOPMENT_YEARS).isdisjoint({CONFIRMATION_YEAR}), "2026 only confirms selected signals"),
        ("yearly_metrics_cover_development", set(DEVELOPMENT_YEARS).issubset(set(yearly["year"])), f"years={sorted(yearly['year'].unique())}"),
        ("fdr_values_are_valid", pd.concat([residual_stability["fdr_q_value"], utility_stability["fdr_q_value"]]).dropna().between(0, 1).all(), "Benjamini-Hochberg q-values"),
        ("fdr_not_below_raw_p", (residual_stability["fdr_q_value"] + 1e-15 >= residual_stability["binomial_p_value"]).all() and (utility_stability["fdr_q_value"] + 1e-15 >= utility_stability["binomial_p_value"]).all(), "multiple-testing correction is conservative"),
        ("confirmation_requires_minimum_sample", (~residual_stability["confirmed_stable_signal"] | residual_stability["confirmation_sample_pass"]).all() and (~utility_stability["confirmed_stable_signal"] | utility_stability["confirmation_sample_pass"]).all(), "confirmed rows pass 2026 sample floor"),
        ("unique_station_dates", not frame["contract_date"].duplicated().any(), f"duplicates={int(frame['contract_date'].duplicated().sum())}"),
        ("integer_settlement_labels", bool(np.allclose(frame["actual_high_f"], np.round(frame["actual_high_f"]))), f"rows={len(frame)}"),
    ]
    if "train_through_year" in frame:
        valid = frame.dropna(subset=["train_through_year", "year"])
        rows.append(("point_predictions_are_forward", (valid["train_through_year"] < valid["year"]).all(), f"checked_rows={len(valid)}"))
    return pd.DataFrame(
        [{"audit": name, "passed": bool(passed), "detail": detail} for name, passed, detail in rows]
    )


def _fixed_cut(values: pd.Series, bins: Sequence[float], labels: Sequence[str]) -> pd.Series:
    return pd.cut(values, bins=bins, labels=labels, include_lowest=True).astype(str)


def _season(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"
