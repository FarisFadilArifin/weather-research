from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_v11_settlement_enriched_experiment import (  # noqa: E402
    BASELINE_DIR,
    FOLDS,
    OUTPUT_DIR,
)
from src.calibration.v11_settlement_enriched_experiment import (  # noqa: E402
    VARIANTS,
    load_enriched_feature_frames,
    make_variant,
)
from src.calibration.v11_settlement_enrichment import (  # noqa: E402
    OBSERVED_BASE_FIELDS,
    MISSING_INDICATOR_THRESHOLD,
    STATIONS,
    expanding_fold_coverage_inventory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate and materialize V11 enriched features before model training")
    parser.add_argument("--stations", default=",".join(STATIONS))
    parser.add_argument("--baseline-dir", type=Path, default=BASELINE_DIR)
    parser.add_argument("--cache-root", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--prepared-dir", type=Path, default=OUTPUT_DIR / "prepared")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stations = tuple(value.strip().upper() for value in args.stations.split(",") if value.strip())
    forecast_path = args.cache_root / "forecast_daily_enriched.csv"
    observation_path = args.cache_root / "observation_daily_enriched.csv"
    parity_path = args.cache_root / "parity" / "iem_awc_parity_report.csv"
    missing = [path for path in (forecast_path, observation_path, parity_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Enrichment caches are incomplete: {[str(path) for path in missing]}")

    frames = load_enriched_feature_frames(
        baseline_dir=args.baseline_dir,
        cache_root=args.cache_root,
        stations=stations,
    )
    original_frames = {
        station: pd.read_csv(args.baseline_dir / f"{station}_features.csv")
        for station in stations
    }
    combined = pd.concat(frames.values(), ignore_index=True)
    candidates = sorted(column for column in combined if _candidate_feature(column))
    reproducible = set(candidates)
    parity = pd.read_csv(parity_path)
    failed_observed = set(parity.loc[~parity["parity_pass"].fillna(False), "feature"].astype(str))
    reproducible -= failed_observed
    inventory = expanding_fold_coverage_inventory(
        combined,
        candidates,
        folds=FOLDS,
        stations=stations,
        reproducible_features=reproducible,
        parent_map=_parent_map(candidates),
    )
    admitted = set(inventory.loc[inventory["admitted_all_folds"].fillna(False), "feature"].astype(str))
    optional_observed = set(OBSERVED_BASE_FIELDS) - {
        "observed_temp_at_as_of_f",
        "observed_high_temp_through_as_of_f",
    }
    blocked_observed = optional_observed - admitted
    frames = {
        station: frame.drop(columns=sorted(blocked_observed), errors="ignore")
        for station, frame in frames.items()
    }
    variants = {
        variant: {
            station: _add_training_only_missing_indicators(
                make_variant(frame, variant, admitted_enriched_features=admitted)
            )
            for station, frame in frames.items()
        }
        for variant in VARIANTS
    }

    args.prepared_dir.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(args.prepared_dir / "gated_feature_inventory_by_fold.csv", index=False)
    pd.DataFrame({"feature": sorted(admitted)}).to_csv(
        args.prepared_dir / "gated_feature_inventory.csv", index=False
    )
    _missingness_report(combined, candidates).to_csv(
        args.prepared_dir / "feature_missingness_by_station_year.csv", index=False
    )
    _write_live_handoff(args.prepared_dir / "production_live_field_handoff.md", admitted)
    manifest_rows: list[dict[str, object]] = []
    for station, frame in original_frames.items():
        target = args.prepared_dir / "original_v11" / f"{station}_features.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(target, index=False)
        manifest_rows.append(_manifest_row("original_v11", station, frame, target))
    for variant, station_frames in variants.items():
        for station, frame in station_frames.items():
            target = args.prepared_dir / variant / f"{station}_features.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(target, index=False)
            manifest_rows.append(_manifest_row(variant, station, frame, target))
    pd.DataFrame(manifest_rows).to_csv(args.prepared_dir / "prepared_manifest.csv", index=False)
    print(f"Prepared {len(manifest_rows)} station/variant feature files in {args.prepared_dir}")


def _manifest_row(variant: str, station: str, frame: pd.DataFrame, path: Path) -> dict[str, object]:
    return {
        "feature_version": "v11_settlement_enriched_v1",
        "variant": variant,
        "station_id": station,
        "row_count": int(len(frame)),
        "feature_count": int(len(frame.columns)),
        "date_min": str(frame["contract_date"].min()),
        "date_max": str(frame["contract_date"].max()),
        "path": str(path.resolve()),
    }


def _candidate_feature(column: str) -> bool:
    forecast = any(column.startswith(f"{provider}_") for provider in ("gfs", "hrrr", "nbm")) and any(
        marker in column
        for marker in (
            "dewpoint_at_11am", "remaining_mean", "precip_", "cloud_at_11am", "cloud_remaining",
            "wind_u_at_11am", "wind_v_at_11am", "wind_speed_at_11am", "wind_speed_remaining",
            "wind_vector_mean",
        )
    )
    observed = column in set(OBSERVED_BASE_FIELDS) - {
        "observed_temp_at_as_of_f", "observed_high_temp_through_as_of_f"
    } or column.startswith("observed_") and any(
        marker in column
        for marker in (
            "_change_1h", "_change_3h", "temperature_acceleration", "morning_temperature_range",
            "minutes_since_high", "calm_wind", "variable_wind", "cloud_category", "ceiling_present",
            "wind_u_at_as_of", "wind_v_at_as_of",
        )
    )
    return forecast or observed


def _parent_map(candidates: list[str]) -> dict[str, tuple[str, ...]]:
    parents: dict[str, tuple[str, ...]] = {}
    roots = {
        "dewpoint_f": "observed_dewpoint_at_as_of_f",
        "humidity_pct": "observed_humidity_at_as_of",
        "pressure_hpa": "observed_pressure_at_as_of",
        "visibility_miles": "observed_visibility_at_as_of",
        "cloud_pct": "observed_cloud_cover_at_as_of",
        "wind_speed_mph": "observed_wind_speed_at_as_of",
        "wind_u_mph": "observed_wind_u_at_as_of_mph",
        "wind_v_mph": "observed_wind_v_at_as_of_mph",
    }
    for feature in candidates:
        for root, parent in roots.items():
            if feature.startswith(f"observed_{root}_change_"):
                parents[feature] = (parent,)
        if "calm_wind" in feature or "variable_wind" in feature:
            parents[feature] = ("observed_wind_speed_at_as_of",)
        elif "cloud_category" in feature or "ceiling_present" in feature:
            parents[feature] = ("observed_cloud_cover_at_as_of",)
    return parents


def _add_training_only_missing_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    year = pd.to_numeric(out.get("year", pd.to_datetime(out["contract_date"]).dt.year), errors="coerce")
    for column in list(out.columns):
        if not pd.api.types.is_numeric_dtype(out[column]):
            continue
        if any(
            len(train := out.loc[year.between(fold.train_start_year, fold.train_end_year), column])
            and train.isna().mean() > MISSING_INDICATOR_THRESHOLD
            for fold in FOLDS
        ):
            out[f"{column}__missing"] = out[column].isna().astype("int8")
    return out


def _missingness_report(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    data = frame.copy()
    data["year"] = pd.to_numeric(data.get("year", pd.to_datetime(data["contract_date"]).dt.year), errors="coerce")
    rows = []
    for (station, year), group in data.groupby(["station_id", "year"]):
        for feature in features:
            rows.append({
                "station_id": station,
                "year": int(year),
                "feature": feature,
                "row_count": int(len(group)),
                "missing_count": int(group[feature].isna().sum()),
                "missingness": float(group[feature].isna().mean()),
            })
    return pd.DataFrame(rows)


def _write_live_handoff(path: Path, admitted: set[str]) -> None:
    lines = [
        "# Future live-fetcher handoff (not implemented in production)", "",
        "- Select forecast cycles available at or before 11:00 local.",
        "- Decode hourly GFS, HRRR, and NBM through local midnight.",
        "- Use the last parity-compatible AWC report in 10:40–11:00 local.",
        "- Never derive gust, observed precipitation, raw weather code, heat index, wind chill, or raw ceiling.",
        "- Use training-bundle medians/categories; never fill meteorological unknowns with zero.", "",
        "## Gated live fields", "", *[f"- `{feature}`" for feature in sorted(admitted)], "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
