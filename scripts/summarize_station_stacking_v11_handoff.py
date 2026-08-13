from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_VERSION = "station_high_regressor_v11_huber_ridge_stack"
PROVIDERS = ("gfs", "hrrr", "nbm")
HARD_REQUIRED_LIVE_FEATURES = {
    "gfs_high_f",
    "hrrr_high_f",
    "nbm_high_f",
    "observed_temp_at_as_of_f",
    "observed_high_temp_through_as_of_f",
    "observed_as_of_age_minutes",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize v11 station-stacking artifacts for Polymarket handoff.")
    parser.add_argument("--artifact-dir", type=Path, default=Path("data/calibration/station_stacking_v11"))
    parser.add_argument("--model-version", default=MODEL_VERSION)
    parser.add_argument("--feature-inventory-out", type=Path, default=Path("POLYMARKET_11AM_ML_V11_FEATURE_INVENTORY.csv"))
    parser.add_argument("--artifact-summary-out", type=Path, default=Path("POLYMARKET_11AM_ML_V11_ARTIFACT_SUMMARY.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_dir = args.artifact_dir
    inventory = build_feature_inventory(artifact_dir)
    summary = build_artifact_summary(artifact_dir, args.model_version)

    inventory.to_csv(args.feature_inventory_out, index=False)
    summary.to_csv(args.artifact_summary_out, index=False)

    print(f"Wrote {args.feature_inventory_out} ({len(inventory)} features)")
    print(f"Wrote {args.artifact_summary_out} ({len(summary)} stations)")


def build_feature_inventory(artifact_dir: Path) -> pd.DataFrame:
    features_by_station: dict[str, set[str]] = {}
    kind_by_feature: dict[str, set[str]] = {}
    missing_rows: list[dict[str, object]] = []

    for feature_path in sorted(artifact_dir.glob("*_feature_columns.csv")):
        station = feature_path.name.split("_", 1)[0]
        feature_columns = pd.read_csv(feature_path)
        features_by_station[station] = set(feature_columns["feature"].astype(str))
        for row in feature_columns.itertuples(index=False):
            kind_by_feature.setdefault(str(row.feature), set()).add(str(row.kind))

        features = pd.read_csv(artifact_dir / f"{station}_features.csv", low_memory=False)
        modeling_rows = _strict_modeling_rows(features)
        for row in feature_columns.itertuples(index=False):
            feature = str(row.feature)
            kind = str(row.kind)
            all_missing_pct, all_non_null_rows = _missingness(features, feature, kind)
            modeling_missing_pct, modeling_non_null_rows = _missingness(modeling_rows, feature, kind)
            missing_rows.append(
                {
                    "station": station,
                    "feature": feature,
                    "kind": kind,
                    "feature_rows": len(features),
                    "all_non_null_rows": all_non_null_rows,
                    "all_rows_missing_pct": all_missing_pct,
                    "modeling_rows": len(modeling_rows),
                    "modeling_non_null_rows": modeling_non_null_rows,
                    "modeling_rows_missing_pct": modeling_missing_pct,
                }
            )

    missing = pd.DataFrame(missing_rows)
    stations = sorted(features_by_station)
    union_features = sorted(set().union(*features_by_station.values()))
    rows: list[dict[str, object]] = []
    for feature in union_features:
        feature_missing = missing.loc[missing["feature"].eq(feature)]
        station_present = sorted(feature_missing["station"].unique())
        kinds = sorted(kind_by_feature.get(feature, []))
        all_weights = pd.to_numeric(feature_missing["feature_rows"], errors="coerce").fillna(0)
        modeling_weights = pd.to_numeric(feature_missing["modeling_rows"], errors="coerce").fillna(0)
        if feature_missing.empty:
            all_min_missing = all_max_missing = all_mean_missing = 100.0
            modeling_min_missing = modeling_max_missing = modeling_mean_missing = 100.0
        else:
            all_min_missing = float(feature_missing["all_rows_missing_pct"].min())
            all_max_missing = float(feature_missing["all_rows_missing_pct"].max())
            if float(all_weights.sum()) > 0:
                all_mean_missing = float(np.average(feature_missing["all_rows_missing_pct"], weights=all_weights))
            else:
                all_mean_missing = float(feature_missing["all_rows_missing_pct"].mean())
            modeling_min_missing = float(feature_missing["modeling_rows_missing_pct"].min())
            modeling_max_missing = float(feature_missing["modeling_rows_missing_pct"].max())
            if float(modeling_weights.sum()) > 0:
                modeling_mean_missing = float(
                    np.average(feature_missing["modeling_rows_missing_pct"], weights=modeling_weights)
                )
            else:
                modeling_mean_missing = float(feature_missing["modeling_rows_missing_pct"].mean())
        rows.append(
            {
                "feature": feature,
                "kind": "/".join(kinds),
                "family": feature_family(feature),
                "used_station_count": len(station_present),
                "used_stations": " ".join(station_present),
                "missing_pct_min_all_rows": round(all_min_missing, 3),
                "missing_pct_max_all_rows": round(all_max_missing, 3),
                "missing_pct_weighted_mean_all_rows": round(all_mean_missing, 3),
                "missing_pct_min_modeling_rows": round(modeling_min_missing, 3),
                "missing_pct_max_modeling_rows": round(modeling_max_missing, 3),
                "missing_pct_weighted_mean_modeling_rows": round(modeling_mean_missing, 3),
                "expected_nan_policy": expected_nan_policy(feature, kinds[0] if kinds else "", modeling_max_missing),
                "schema_note": "station_specific_extra" if len(station_present) < len(stations) else "",
            }
        )
    return pd.DataFrame(rows)


def build_artifact_summary(artifact_dir: Path, model_version: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature_path in sorted(artifact_dir.glob("*_feature_columns.csv")):
        station = feature_path.name.split("_", 1)[0]
        feature_columns = pd.read_csv(feature_path)
        features = pd.read_csv(artifact_dir / f"{station}_features.csv", low_memory=False)
        manifest_path = artifact_dir / "model_weights" / f"{station}_{model_version}.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        stack = manifest["stack_model"]
        scoreboard = pd.read_csv(artifact_dir / f"{station}_year_split_scoreboard.csv")
        ridge = scoreboard.loc[
            scoreboard["period"].eq("test_2026") & scoreboard["method"].eq("ridge_stack")
        ].iloc[0]
        rows.append(
            {
                "station": station,
                "feature_rows": len(features),
                "feature_count": len(feature_columns),
                "categorical_count": int(feature_columns["kind"].eq("categorical").sum()),
                "numeric_count": int(feature_columns["kind"].eq("numeric").sum()),
                "modeling_rows": len(_strict_modeling_rows(features)),
                "export_train_rows": manifest["training"]["train_rows"],
                "stack_feature_set": stack.get("feature_set"),
                "stack_features": " ".join(stack.get("features", [])),
                "test_2026_ridge_mae_f": round(float(ridge["mae_f"]), 4),
                "test_2026_ridge_rmse_f": round(float(ridge["rmse_f"]), 4),
                "test_2026_count": int(ridge["count"]),
            }
        )
    return pd.DataFrame(rows).sort_values("station").reset_index(drop=True)


def _strict_modeling_rows(features: pd.DataFrame) -> pd.DataFrame:
    mask = pd.Series(True, index=features.index)
    for column in ["actual_high_f", "observed_high_temp_through_as_of_f"]:
        if column in features:
            mask &= pd.to_numeric(features[column], errors="coerce").notna()
    for provider in PROVIDERS:
        column = f"{provider}_high_f"
        if column in features:
            mask &= pd.to_numeric(features[column], errors="coerce").notna()
    if "strict_quality_ok" in features:
        mask &= features["strict_quality_ok"].fillna(False).astype(bool)
    if "all_provider_highs_available" in features:
        mask &= features["all_provider_highs_available"].fillna(False).astype(bool)
    return features.loc[mask].copy()


def _missingness(frame: pd.DataFrame, feature: str, kind: str) -> tuple[float, int]:
    if frame.empty or feature not in frame:
        return 100.0, 0
    series = frame[feature]
    if kind == "numeric":
        missing = pd.to_numeric(series, errors="coerce").isna()
    else:
        text = series.astype("string")
        missing = text.isna() | text.str.strip().isin(["", "<NA>", "nan", "NaN"])
    return float(missing.mean() * 100.0), int((~missing).sum())


def feature_family(feature: str) -> str:
    if feature == "day_of_week":
        return "calendar/categorical"
    if feature in {"month", "day_of_year", "day_of_year_sin", "day_of_year_cos", "year"}:
        return "calendar"
    if feature.startswith("climatology_") or "climatology" in feature:
        return "v9/v11 climatology"
    if feature.startswith("v8_"):
        return "v8 remaining-warmup engineered"
    if feature.startswith("v4_"):
        return "v4 precipitation engineered"
    if feature.startswith("v3_"):
        return "v3 high-so-far engineered"
    if feature.startswith("v2_"):
        return "v2 heat/warmup engineered"
    if feature.startswith("observed_"):
        if any(token in feature for token in ["change_last", "warmup_rate", "since_9am"]):
            return "current observation trend"
        if any(token in feature for token in ["history", "lag", "roll"]):
            return "observation history"
        return "current observation"
    if feature.startswith("provider_"):
        return "provider ensemble/climatology delta"
    if any(feature.startswith(f"{provider}_") for provider in PROVIDERS):
        if any(token in feature for token in ["lag", "roll", "bias", "mae", "error"]):
            return "provider historical error/bias"
        if "minus_observed" in feature or "minus_obs" in feature:
            return "provider-observation delta"
        if "minus_" in feature or "_diff_" in feature:
            return "provider cross-model delta"
        return "provider forecast/raw"
    if feature.startswith("actual_high_lag") or feature.startswith("actual_high_roll"):
        return "prior actual history"
    if feature in {"lat", "lon", "elevation_ft"}:
        return "station metadata"
    return "other numeric"


def expected_nan_policy(feature: str, kind: str, max_missing: float) -> str:
    if feature in HARD_REQUIRED_LIVE_FEATURES:
        return "required_live_non_null; fail closed if missing"
    if kind == "categorical":
        return 'may_be_null; categorical imputer fills "missing"'
    family = feature_family(feature)
    if family in {"prior actual history", "provider historical error/bias", "observation history"}:
        return "expected_nan_early_history_or_missing_prior; numeric median imputer handles"
    if family == "v9/v11 climatology":
        return "should_be_present_after_normals_join; investigate if high missingness"
    if max_missing == 0:
        return "observed_non_null_in_v11_artifacts; still build column defensively"
    return "may_be_null; numeric median imputer handles"


if __name__ == "__main__":
    main()
