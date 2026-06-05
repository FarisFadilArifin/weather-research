from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .features import add_weather_regime_buckets
from .metrics import metrics_by


def save_required_plots(model_errors: pd.DataFrame, trading_table: pd.DataFrame, plots_dir: str | Path) -> list[Path]:
    out = Path(plots_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if model_errors.empty:
        return paths
    model_errors = add_weather_regime_buckets(model_errors)

    grouped = metrics_by(model_errors, ["station_code", "provider", "forecast_horizon_hours"])
    paths.append(_bar_metric(grouped, "mae_f", "MAE by Station and Horizon", out / "mae_by_station_horizon.png"))
    paths.append(_bar_metric(grouped, "rmse_f", "RMSE by Station and Horizon", out / "rmse_by_station_horizon.png"))
    paths.append(_bar_metric(grouped, "mean_error_f", "Bias by Station and Horizon", out / "bias_by_station_horizon.png"))
    paths.append(_hist(model_errors, "error_f", "Error Distribution by Station", out / "error_distribution_by_station.png", group="station_code"))
    paths.append(_hist(model_errors, "error_f", "Error Distribution by Horizon", out / "error_distribution_by_horizon.png", group="forecast_horizon_hours"))
    paths.append(_provider_compare(grouped, out / "provider_comparison.png"))
    paths.append(_scatter(model_errors, out / "actual_vs_forecast_scatter.png"))
    paths.append(_heatmap(grouped, "mae_f", out / "heatmap_station_horizon_mae.png"))
    paths.append(_heatmap(grouped, "mean_error_f", out / "heatmap_station_horizon_bias.png"))
    paths.append(_rolling_error(model_errors, out / "rolling_7day_error_by_station_provider.png"))
    paths.append(_box(model_errors, "cloud_cover_bucket", out / "error_by_cloud_cover_bucket.png"))
    paths.append(_box(model_errors, "month", out / "error_by_month.png"))
    paths.append(_sample_size(model_errors, out / "station_sample_size.png"))
    paths.append(_best_worst(trading_table, out / "best_worst_station_horizon_combinations.png"))
    return [path for path in paths if path.exists()]


def _bar_metric(frame: pd.DataFrame, metric: str, title: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 6))
    if not frame.empty:
        data = frame.sort_values(metric).tail(30)
        labels = data["station_code"].astype(str) + " " + data["forecast_horizon_hours"].astype(str) + "h"
        ax.bar(labels, data[metric])
        ax.tick_params(axis="x", rotation=75)
    ax.set_title(title)
    ax.set_ylabel(metric)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _hist(frame: pd.DataFrame, metric: str, title: str, path: Path, group: str) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, sub in frame.groupby(group):
        ax.hist(pd.to_numeric(sub[metric], errors="coerce").dropna(), bins=20, alpha=0.35, label=str(label))
    ax.set_title(title)
    ax.set_xlabel(metric)
    if len(frame[group].dropna().unique()) <= 10:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _provider_compare(frame: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    if not frame.empty:
        pivot = frame.pivot_table(index="forecast_horizon_hours", columns="provider", values="mae_f", aggfunc="mean")
        pivot.plot(ax=ax)
    ax.set_title("HRRR vs Open-Meteo vs NWS")
    ax.set_ylabel("MAE F")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _scatter(frame: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(frame["forecast_high_f"], frame["actual_high_f"], alpha=0.5)
    vals = pd.concat([frame["forecast_high_f"], frame["actual_high_f"]]).dropna()
    if not vals.empty:
        ax.plot([vals.min(), vals.max()], [vals.min(), vals.max()], color="black", linewidth=1)
    ax.set_xlabel("Forecast high F")
    ax.set_ylabel("Actual high F")
    ax.set_title("Actual High vs Forecast High")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _heatmap(frame: pd.DataFrame, metric: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 8))
    if not frame.empty:
        pivot = frame.pivot_table(index="station_code", columns="forecast_horizon_hours", values=metric, aggfunc="mean")
        im = ax.imshow(pivot.fillna(0), aspect="auto")
        ax.set_xticks(range(len(pivot.columns)), labels=pivot.columns)
        ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
        fig.colorbar(im, ax=ax, label=metric)
    ax.set_title(path.stem.replace("_", " ").title())
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _rolling_error(frame: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 6))
    data = frame.copy()
    data["date"] = pd.to_datetime(data["target_date_local"], errors="coerce")
    for (station, provider), sub in data.sort_values("date").groupby(["station_code", "provider"]):
        sub = sub.set_index("date")
        sub["error_f"].rolling("7D").mean().plot(ax=ax, label=f"{station}-{provider}")
    ax.set_title("Rolling 7-Day Forecast Error")
    if len(data.groupby(["station_code", "provider"])) <= 10:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _box(frame: pd.DataFrame, column: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    data = [sub["error_f"].dropna() for _, sub in frame.dropna(subset=[column]).groupby(column)]
    labels = [str(label) for label, _ in frame.dropna(subset=[column]).groupby(column)]
    if data:
        ax.boxplot(data, labels=labels)
    ax.set_title(path.stem.replace("_", " ").title())
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _sample_size(frame: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    frame.groupby("station_code")["error_f"].count().sort_values().plot(kind="bar", ax=ax)
    ax.set_title("Polymarket Station Sample Size")
    ax.set_ylabel("sample size")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _best_worst(trading_table: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 6))
    if not trading_table.empty:
        data = pd.concat([trading_table.nsmallest(10, "mae_f"), trading_table.nlargest(10, "mae_f")])
        labels = data["station_code"].astype(str) + " " + data["provider"].astype(str) + " " + data["forecast_horizon_hours"].astype(str) + "h"
        ax.bar(labels, data["mae_f"])
        ax.tick_params(axis="x", rotation=75)
    ax.set_title("Best/Worst Station-Horizon Combinations")
    ax.set_ylabel("MAE F")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
