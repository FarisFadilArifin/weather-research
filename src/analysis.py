from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .metrics import metrics_by


TRADING_COLUMNS = [
    "polymarket_city",
    "station_code",
    "station_name",
    "airport_name",
    "provider",
    "model",
    "forecast_horizon_hours",
    "sample_size",
    "bias_f",
    "mae_f",
    "rmse_f",
    "error_std_f",
    "within_1f_pct",
    "within_2f_pct",
    "within_3f_pct",
    "edge_reliability_score",
    "recommended_use",
    "notes",
]


def build_trading_research_table(model_errors: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    if model_errors.empty:
        return pd.DataFrame(columns=TRADING_COLUMNS)
    grouped = metrics_by(
        model_errors,
        [
            "polymarket_city",
            "station_code",
            "station_name",
            "airport_name",
            "provider",
            "model",
            "forecast_horizon_hours",
        ],
    )
    if grouped.empty:
        return pd.DataFrame(columns=TRADING_COLUMNS)
    out = grouped.rename(columns={"count": "sample_size", "mean_error_f": "bias_f"}).copy()
    out["edge_reliability_score"] = (out["bias_f"].abs() / out["error_std_f"].replace(0, pd.NA)).fillna(0)
    thresholds = settings.get("trading", {})
    min_sample = int(thresholds.get("min_sample_size", 20))
    high_std = float(thresholds.get("high_error_std_f", 5.0))
    strong = float(thresholds.get("strong_score_threshold", 0.5))
    possible = float(thresholds.get("possible_score_threshold", 0.25))
    out["recommended_use"] = out.apply(
        lambda row: _recommend(row, min_sample=min_sample, high_std=high_std, strong=strong, possible=possible),
        axis=1,
    )
    out["notes"] = out.apply(lambda row: _notes(row, min_sample=min_sample, high_std=high_std), axis=1)
    for column in TRADING_COLUMNS:
        if column not in out:
            out[column] = pd.NA
    return out[TRADING_COLUMNS]


def _recommend(row: pd.Series, min_sample: int, high_std: float, strong: float, possible: float) -> str:
    if row["sample_size"] < min_sample or row["error_std_f"] >= high_std:
        return "avoid"
    if row["edge_reliability_score"] >= strong:
        return "strong_research_candidate"
    if row["edge_reliability_score"] >= possible:
        return "possible_candidate"
    return "avoid"


def _notes(row: pd.Series, min_sample: int, high_std: float) -> str:
    notes: list[str] = []
    if row["sample_size"] < min_sample:
        notes.append("sample too small")
    if row["error_std_f"] >= high_std:
        notes.append("high error volatility")
    if row["bias_f"] > 0:
        notes.append("actuals tend warmer than forecast")
    elif row["bias_f"] < 0:
        notes.append("actuals tend cooler than forecast")
    return "; ".join(notes)


def write_trading_table(model_errors: pd.DataFrame, settings: dict[str, Any], output_dir: str | Path) -> pd.DataFrame:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame = build_trading_research_table(model_errors, settings)
    frame.to_csv(out / "trading_research_table.csv", index=False)
    return frame


def write_research_summary(
    station_map: pd.DataFrame,
    model_errors: pd.DataFrame,
    trading_table: pd.DataFrame,
    output_dir: str | Path,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "research_summary.md"
    mapped = station_map.loc[~station_map["needs_manual_review"].fillna(True).astype(bool)] if not station_map.empty else station_map
    review = station_map.loc[station_map["needs_manual_review"].fillna(True).astype(bool)] if not station_map.empty else station_map
    best = _best_worst(model_errors, best=True)
    worst = _best_worst(model_errors, best=False)
    best_horizon = _best_horizon_by_station(model_errors, best=True)
    worst_horizon = _best_horizon_by_station(model_errors, best=False)
    hrrr_vs_open = _provider_comparison(model_errors, "hrrr", "openmeteo")
    nws_vs_all = _nws_wins(model_errors)
    volatility = trading_table.loc[trading_table["recommended_use"] == "avoid"] if not trading_table.empty else trading_table

    content = f"""# Polymarket Weather Research Summary

## Discovered Stations
{_bullet_list(mapped, ["station_code", "station_name", "airport_name", "polymarket_city"])}

## Needs Manual Review
{_bullet_list(review, ["market_slug", "polymarket_city", "resolution_source_text"])}

## Best And Worst Accuracy
- Best station overall by MAE: {best}
- Worst station overall by MAE: {worst}

## Horizon Findings
Best forecast horizon for each station:
{best_horizon}

Worst forecast horizon for each station:
{worst_horizon}

## Bias Findings
Warm-bias combinations:
{_bias_list(model_errors, warm=True)}

Cool-bias combinations:
{_bias_list(model_errors, warm=False)}

## Provider Comparisons
- HRRR vs Open-Meteo: {hrrr_vs_open}
- NWS/NBM-style wins: {nws_vs_all}

## Trading Window Notes
{_trading_notes(trading_table)}

## Warning List
{_bullet_list(volatility.head(20), ["station_code", "provider", "model", "forecast_horizon_hours", "error_std_f", "notes"])}

## Data Limitations And Leakage Risks
- Station mappings come only from Polymarket rule/source text or manual overrides.
- Historical NWS API coverage is limited; archived NBM is needed for rigorous historical NWS-style comparisons.
- HRRR availability is domain- and forecast-hour-limited; 72h snapshots are unavailable unless the archive contains the required forecast hours.
- Wunderground displayed settlement values can differ from raw official observations because of rounding, finalization timing, or source presentation.
- Active markets should not be used for training until actuals are finalized.
"""
    path.write_text(content, encoding="utf-8")
    return path


def _best_worst(model_errors: pd.DataFrame, best: bool) -> str:
    if model_errors.empty:
        return "not enough data"
    grouped = metrics_by(model_errors, ["station_code"])
    if grouped.empty:
        return "not enough data"
    row = grouped.sort_values("mae_f", ascending=best).iloc[0]
    return f"{row['station_code']} (MAE {row['mae_f']:.2f}F, n={int(row['count'])})"


def _best_horizon_by_station(model_errors: pd.DataFrame, best: bool) -> str:
    if model_errors.empty:
        return "- not enough data"
    grouped = metrics_by(model_errors, ["station_code", "provider", "model", "forecast_horizon_hours"])
    if grouped.empty:
        return "- not enough data"
    rows = []
    for station, group in grouped.groupby("station_code"):
        row = group.sort_values("mae_f", ascending=best).iloc[0]
        rows.append(f"- {station}: {row['provider']} {int(row['forecast_horizon_hours'])}h (MAE {row['mae_f']:.2f}F)")
    return "\n".join(rows)


def _bias_list(model_errors: pd.DataFrame, warm: bool) -> str:
    if model_errors.empty:
        return "- not enough data"
    grouped = metrics_by(model_errors, ["station_code", "provider", "model", "forecast_horizon_hours"])
    if grouped.empty:
        return "- not enough data"
    selected = grouped.loc[grouped["mean_error_f"] > 0] if warm else grouped.loc[grouped["mean_error_f"] < 0]
    selected = selected.reindex(selected["mean_error_f"].abs().sort_values(ascending=False).index).head(10)
    if selected.empty:
        return "- none identified"
    return "\n".join(
        f"- {r.station_code} {r.provider} {int(r.forecast_horizon_hours)}h: bias {r.mean_error_f:.2f}F, n={int(r['count'])}"
        for _, r in selected.iterrows()
    )


def _provider_comparison(model_errors: pd.DataFrame, left: str, right: str) -> str:
    if model_errors.empty:
        return "not enough data"
    grouped = metrics_by(model_errors, ["station_code", "forecast_horizon_hours", "provider"])
    pivot = grouped.pivot_table(index=["station_code", "forecast_horizon_hours"], columns="provider", values="mae_f")
    if left not in pivot or right not in pivot:
        return "not enough overlapping data"
    wins = (pivot[left] < pivot[right]).sum()
    losses = (pivot[left] > pivot[right]).sum()
    return f"{left} lower MAE in {wins} station/horizon groups; {right} lower MAE in {losses}"


def _nws_wins(model_errors: pd.DataFrame) -> str:
    if model_errors.empty or "nws" not in set(model_errors.get("provider", [])):
        return "not enough NWS/NBM-style data"
    grouped = metrics_by(model_errors, ["station_code", "forecast_horizon_hours", "provider"])
    winners = grouped.sort_values("mae_f").groupby(["station_code", "forecast_horizon_hours"]).first().reset_index()
    count = int((winners["provider"] == "nws").sum())
    return f"NWS/NBM-style provider wins {count} station/horizon groups"


def _trading_notes(trading_table: pd.DataFrame) -> str:
    if trading_table.empty:
        return "- not enough data"
    strong = trading_table.loc[trading_table["recommended_use"] == "strong_research_candidate"]
    possible = trading_table.loc[trading_table["recommended_use"] == "possible_candidate"]
    return (
        f"- Strong candidates: {len(strong)} station/provider/horizon groups\n"
        f"- Possible candidates: {len(possible)} station/provider/horizon groups\n"
        "- Use table-level recommendations as research filters only, not trading instructions."
    )


def _bullet_list(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame is None or frame.empty:
        return "- none"
    lines = []
    for _, row in frame.drop_duplicates(subset=[c for c in columns if c in frame]).head(30).iterrows():
        values = [str(row.get(column, "")).strip() for column in columns if pd.notna(row.get(column, pd.NA))]
        text = " | ".join(v[:160] for v in values if v)
        if text:
            lines.append(f"- {text}")
    return "\n".join(lines) if lines else "- none"
