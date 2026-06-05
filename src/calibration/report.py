from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_report(calibration_dir: str | Path) -> Path:
    out_dir = Path(calibration_dir)
    samples = _read(out_dir / "calibration_samples.csv")
    baselines = _read(out_dir / "baseline_results.csv")
    ml = _read(out_dir / "ml_results.csv")
    rules = _read(out_dir / "recommended_calibration_rules.csv")
    path = out_dir / "calibration_report.md"
    path.write_text(_render_report(samples, baselines, ml, rules), encoding="utf-8")
    return path


def _render_report(samples: pd.DataFrame, baselines: pd.DataFrame, ml: pd.DataFrame, rules: pd.DataFrame) -> str:
    providers = ", ".join(sorted(samples["provider"].dropna().astype(str).unique())) if not samples.empty else "none"
    date_range = "none"
    if not samples.empty:
        dates = pd.to_datetime(samples["contract_date"], errors="coerce")
        date_range = f"{dates.min().date()} to {dates.max().date()}"
    best_baseline = _best_overall(baselines)
    shrinkage = _overall_method(baselines, "hierarchical_shrinkage")
    best_ml = _best_overall(ml)
    reference_baseline = shrinkage if shrinkage is not None else best_baseline
    recommendation = _recommendation(reference_baseline, best_ml, ml)
    return f"""# Weather Forecast Calibration Report

## Data
- Samples: {len(samples)}
- Stations: {_count_unique(samples, "station_id")}
- Providers: {providers}
- Date range: {date_range}
- Horizon: 0h, defined as forecast available at 06:00 local station time

## Best Walk-Forward Results
{_format_result("Best baseline", best_baseline)}
{_format_result("Best ML", best_ml)}

## Recommendation
{recommendation}

## Provider Lineage
- `nbm` rows are NOAA NBM archive/TMAX rows, not exact weather.gov/NWS text forecasts.
- `nws` rows are only allowed for captured weather.gov forecasts. No such rows are emitted unless local capture files exist.
- `gfs` and `hrrr` rows are expected to come from Mostly Right `forecast_nwp` cache rows.

## Rule Output
- Recommended calibration rules written: {len(rules)}
- Rules file: `recommended_calibration_rules.csv`
"""


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _count_unique(frame: pd.DataFrame, column: str) -> int:
    return 0 if frame.empty or column not in frame else int(frame[column].nunique(dropna=True))


def _best_overall(results: pd.DataFrame) -> pd.Series | None:
    if results.empty:
        return None
    overall = results.loc[results["evaluation_scope"] == "overall"].copy()
    if overall.empty:
        return None
    return overall.sort_values("mae_after_f").iloc[0]


def _overall_method(results: pd.DataFrame, method: str) -> pd.Series | None:
    if results.empty:
        return None
    rows = results.loc[(results["evaluation_scope"] == "overall") & (results["method"] == method)]
    return None if rows.empty else rows.iloc[0]


def _format_result(label: str, row: pd.Series | None) -> str:
    if row is None:
        return f"- {label}: not enough data"
    return (
        f"- {label}: `{row['method']}` MAE {row['mae_after_f']:.3f}F "
        f"vs raw {row['mae_before_f']:.3f}F, bias after {row['bias_after_f']:.3f}F, n={int(row['count'])}"
    )


def _recommendation(best_baseline: pd.Series | None, best_ml: pd.Series | None, ml_results: pd.DataFrame) -> str:
    if best_baseline is None:
        return "Not enough walk-forward data to recommend a calibration model."
    if best_ml is None:
        return (
            f"Use `{best_baseline['method']}`. ML was unavailable or not run, and the baseline is leakage-safe "
            "under the current walk-forward evaluation."
        )
    improvement = float(best_baseline["mae_after_f"]) - float(best_ml["mae_after_f"])
    stable = _ml_stability_ok(ml_results, str(best_ml["method"]))
    if improvement >= 0.10 and stable:
        return (
            f"Use ML model `{best_ml['method']}` for research deployment candidates. It beats the best baseline by "
            f"{improvement:.3f}F MAE and passes the station/month stability screen."
        )
    return (
        f"Use `{best_baseline['method']}`. The best ML improvement is {improvement:.3f}F MAE and does not clear "
        "the robustness threshold for live trading calibration."
    )


def _ml_stability_ok(results: pd.DataFrame, method: str) -> bool:
    split = results.loc[
        (results["method"] == method)
        & (results["evaluation_scope"] == "station_provider_month")
        & (results["count"] >= 10)
    ]
    if split.empty:
        return False
    return float((split["mae_improvement_f"] > 0).mean()) >= 0.70
