"""Generate the complete through-V20 feature EDA notebook."""

from __future__ import annotations

import json
from pathlib import Path


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


cells = [
    markdown(
        """# Complete Feature EDA Through V20

This notebook explores the complete 113-feature input set used by the V20 peak-timing model (`V20_FEATURE_COLUMNS`). It covers features inherited from V2-V11, the V11 settlement-alignment additions, and the V20 raw and engineered additions. It provides:

- dataset shape, date range, station coverage, and duplicate checks;
- missing-value and descriptive-statistics tables;
- one distribution histogram and station-level boxplot for each feature;
- IQR-based outlier counts (diagnostic only; no values are removed).

Run from any working directory inside this repository. Change `STATIONS`, `START_DATE`, or `END_DATE` in the configuration cell to focus the analysis.
"""
    ),
    code(
        """from pathlib import Path
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", 100)
pd.set_option("display.float_format", lambda value: f"{value:,.3f}")


def find_project_root(start: Path | None = None) -> Path:
    candidates = [Path(start or Path.cwd()).resolve(), Path.cwd().resolve()]
    for candidate in candidates:
        for path in (candidate, *candidate.parents):
            if (path / "pyproject.toml").exists() and (path / "src").exists():
                return path
    raise FileNotFoundError("Could not find the weather-research project root.")


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.station_stacking import (
    V11_FEATURE_COLUMNS,
    V11_SETTLEMENT_FIX_TEMP_FEATURE_COLUMNS,
    V20_ENGINEERED_FEATURE_COLUMNS,
    V20_FEATURE_COLUMNS,
    V20_PEAK_TIMING_RAW_FEATURE_COLUMNS,
)

plt.style.use("seaborn-v0_8-whitegrid")
print(f"Project root: {PROJECT_ROOT}")
"""
    ),
    markdown("## Configuration\n"),
    code(
        """# Use None to keep every available station/date.
STATIONS = None          # Example: ["KATL", "KDAL"]
START_DATE = None        # Example: "2021-01-01"
END_DATE = None          # Example: "2026-07-14"

HISTOGRAM_BINS = 30
FIGURE_SIZE = (13, 4.5)
STATION_COLORS = {"KATL": "#2563eb", "KDAL": "#ea580c"}
"""
    ),
    markdown("## Load the complete V20 modeling features\n"),
    code(
        """feature_dir = PROJECT_ROOT / "data" / "calibration" / "station_stacking_v20_peak_timing"
available_files = sorted(feature_dir.glob("*_features.csv"))
if not available_files:
    raise RuntimeError(f"No exported V20 feature files found under {feature_dir}")

selected_stations = {station.upper() for station in STATIONS} if STATIONS else None
frames = []
for path in available_files:
    station = path.name.removesuffix("_features.csv").upper()
    if selected_stations and station not in selected_stations:
        continue
    frame = pd.read_csv(path, low_memory=False)
    if "station_id" not in frame:
        frame["station_id"] = station
    frames.append(frame)

if not frames:
    raise RuntimeError("No V20 feature files matched the STATIONS filter.")

v20 = pd.concat(frames, ignore_index=True, sort=False)
v20["contract_date"] = pd.to_datetime(v20["contract_date"], errors="coerce")
if START_DATE:
    v20 = v20[v20["contract_date"] >= pd.Timestamp(START_DATE)].copy()
if END_DATE:
    v20 = v20[v20["contract_date"] <= pd.Timestamp(END_DATE)].copy()

feature_columns = list(V20_FEATURE_COLUMNS)
missing_columns = [column for column in feature_columns if column not in v20.columns]
if missing_columns:
    raise KeyError(f"Exported V20 data is missing model features: {missing_columns}")

features = v20[["station_id", "contract_date", *feature_columns]].copy()
for column in feature_columns:
    features[column] = pd.to_numeric(features[column], errors="coerce")

feature_groups = {
    **{column: "Inherited through V11" for column in V11_FEATURE_COLUMNS},
    **{column: "V11 settlement alignment" for column in V11_SETTLEMENT_FIX_TEMP_FEATURE_COLUMNS},
    **{column: "V20 raw peak timing" for column in V20_PEAK_TIMING_RAW_FEATURE_COLUMNS},
    **{column: "V20 engineered" for column in V20_ENGINEERED_FEATURE_COLUMNS},
}

print(f"Rows: {len(features):,}")
print(f"Complete features through V20: {len(feature_columns)}")
print(f"Stations: {', '.join(sorted(features['station_id'].dropna().unique()))}")
print(f"Date range: {features['contract_date'].min().date()} to {features['contract_date'].max().date()}")
print(f"Duplicate station/date keys: {features.duplicated(['station_id', 'contract_date']).sum():,}")
display(pd.Series(feature_groups).value_counts().rename_axis("feature_group").to_frame("features"))
display(features.head())
"""
    ),
    markdown("## Coverage and descriptive statistics\n"),
    code(
        """coverage = pd.DataFrame({
    "group": pd.Series(feature_groups),
    "dtype": features[feature_columns].dtypes.astype(str),
    "non_null": features[feature_columns].notna().sum(),
    "missing": features[feature_columns].isna().sum(),
    "missing_pct": features[feature_columns].isna().mean().mul(100),
    "unique": features[feature_columns].nunique(dropna=True),
}).sort_values(["missing_pct", "unique"], ascending=[False, True])
display(coverage)

summary = features[feature_columns].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
summary.insert(0, "group", pd.Series(feature_groups))
summary["skew"] = features[feature_columns].skew(numeric_only=True)
display(summary)
"""
    ),
    markdown("## Coverage by station\n"),
    code(
        """station_coverage = (
    features.groupby("station_id")[feature_columns]
    .agg(lambda series: series.notna().mean() * 100)
    .T
    .round(2)
)
station_coverage.index.name = "feature"
display(station_coverage)
"""
    ),
    markdown(
        """## Distribution and boxplot for all 113 features through V20

The histogram overlays stations and includes an overall median marker. The boxplot is split by station, making station shifts and extreme values easy to spot. Missing values are excluded independently for each feature.
"""
    ),
    code(
        """def sensible_bins(values: pd.Series, default_bins: int = 30):
    clean = values.dropna()
    unique = np.sort(clean.unique())
    if len(unique) <= 20 and len(unique) > 0 and np.allclose(unique, np.round(unique)):
        return np.arange(unique.min() - 0.5, unique.max() + 1.5, 1)
    return default_bins


def plot_feature_distribution(frame: pd.DataFrame, feature: str) -> None:
    stations = sorted(frame["station_id"].dropna().unique())
    clean_all = frame[feature].dropna()
    fig, (ax_hist, ax_box) = plt.subplots(
        1, 2, figsize=FIGURE_SIZE, gridspec_kw={"width_ratios": [2.1, 1]}
    )

    if clean_all.empty:
        for axis in (ax_hist, ax_box):
            axis.text(0.5, 0.5, "No non-null values", ha="center", va="center")
            axis.set_axis_off()
        fig.suptitle(feature, fontsize=14, fontweight="bold")
        plt.show()
        plt.close(fig)
        return

    bins = sensible_bins(clean_all, HISTOGRAM_BINS)
    for index, station in enumerate(stations):
        station_values = frame.loc[frame["station_id"].eq(station), feature].dropna()
        if station_values.empty:
            continue
        color = STATION_COLORS.get(station, plt.cm.tab10(index % 10))
        ax_hist.hist(
            station_values,
            bins=bins,
            alpha=0.42,
            density=True,
            label=f"{station} (n={len(station_values):,})",
            color=color,
            edgecolor="white",
            linewidth=0.4,
        )

    median = clean_all.median()
    ax_hist.axvline(median, color="#111827", linestyle="--", linewidth=1.5, label=f"Median = {median:,.2f}")
    ax_hist.set_title("Distribution")
    ax_hist.set_xlabel(feature)
    ax_hist.set_ylabel("Density")
    ax_hist.legend(fontsize=8)

    box_values = []
    box_labels = []
    box_colors = []
    for index, station in enumerate(stations):
        values = frame.loc[frame["station_id"].eq(station), feature].dropna().to_numpy()
        if values.size:
            box_values.append(values)
            box_labels.append(station)
            box_colors.append(STATION_COLORS.get(station, plt.cm.tab10(index % 10)))
    box = ax_box.boxplot(box_values, tick_labels=box_labels, patch_artist=True, showfliers=True)
    for patch, color in zip(box["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    ax_box.set_title("Boxplot by station")
    ax_box.set_ylabel(feature)

    missing = frame[feature].isna().sum()
    fig.suptitle(
        f"{feature}  |  {feature_groups.get(feature, 'Other')}  |  non-null={len(clean_all):,}, missing={missing:,} ({missing / len(frame):.1%})",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    plt.show()
    plt.close(fig)


for feature in feature_columns:
    plot_feature_distribution(features, feature)
"""
    ),
    markdown("## IQR outlier diagnostic\n"),
    code(
        """outlier_rows = []
for feature in feature_columns:
    values = features[feature].dropna()
    if values.empty:
        outlier_rows.append({
            "feature": feature, "group": feature_groups.get(feature),
            "q1": np.nan, "q3": np.nan, "iqr": np.nan,
            "lower_fence": np.nan, "upper_fence": np.nan, "outliers": 0,
            "outlier_pct": np.nan,
        })
        continue
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = ((values < lower) | (values > upper)).sum()
    outlier_rows.append({
        "feature": feature,
        "group": feature_groups.get(feature),
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_fence": lower,
        "upper_fence": upper,
        "outliers": int(outliers),
        "outlier_pct": outliers / len(values) * 100,
    })

outlier_summary = pd.DataFrame(outlier_rows).set_index("feature").sort_values("outlier_pct", ascending=False)
display(outlier_summary)
"""
    ),
    markdown(
        """## Notes for interpretation

- The IQR rule flags unusual values; it does **not** prove a data-quality error.
- Several weather variables are naturally zero-inflated or bounded, so skewed distributions can be expected.
- Compare station-level boxes before applying global clipping or transformations because KATL and KDAL may have different climatologies.
- Review missingness alongside distributions: a clean-looking histogram can still represent only a subset of dates.
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

output_path = Path(__file__).with_name("v20_raw_feature_eda.ipynb")
output_path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"Wrote {output_path}")
