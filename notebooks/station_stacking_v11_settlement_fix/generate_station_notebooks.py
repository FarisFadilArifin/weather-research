from __future__ import annotations

import importlib.util
import json
from pathlib import Path


TARGET_STATIONS = ("KATL", "KDAL")

# Source-owned experiment markers used by regression tests:
# feature_version="v11_settlement_fix_temp"
# target_source="settlement_first"
# max_feature_missing_fraction=0.03
# EXPORT_MODEL_WEIGHTS = False
# no V15 arms: this notebook runs one configuration only.


def _load_v11_settlement_generator():
    source = Path(__file__).resolve().parents[1] / "station_stacking_v11_settlement" / "generate_station_notebooks.py"
    spec = importlib.util.spec_from_file_location("station_stacking_v11_settlement_generator", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load v11 settlement notebook generator from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def _code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def _notebook(station_id: str) -> dict:
    v11_settlement = _load_v11_settlement_generator()
    notebook = v11_settlement._notebook(station_id)
    replacements = [
        ("Station Stacking v11 Settlement", "Station Stacking v11 Settlement Fix"),
        (
            "station_high_regressor_v11_wunderground_settlement_stack",
            "station_high_regressor_v11_settlement_fix_temp_stack",
        ),
        ('feature_version="v11"', 'feature_version="v11_settlement_fix_temp"'),
        (
            '    target_source="settlement_first",\n',
            '    target_source="settlement_first",\n'
            '    max_feature_missing_fraction=0.03,\n',
        ),
        (
            'output_dir=PROJECT_ROOT / "data" / "calibration" / "station_stacking_v11_settlement"',
            'output_dir=PROJECT_ROOT / "data" / "calibration" / "station_stacking_v11_settlement_fix"',
        ),
        (
            'source_pipeline="notebooks/station_stacking_v11_settlement"',
            'source_pipeline="notebooks/station_stacking_v11_settlement_fix"',
        ),
        (
            "This controlled rerun keeps the exact v11 feature/model contract, replaces daily-high labels with settlement-first Wunderground station history when available, and writes separate artifacts to `data/calibration/station_stacking_v11_settlement`.",
            "This experiment keeps the settlement-first v11 remaining-warmup contract, removes features above 3% missingness within each training fold, adds the expanded live-safe 11 AM forecast-temperature feature family, and writes artifacts to `data/calibration/station_stacking_v11_settlement_fix`.",
        ),
    ]
    for cell in notebook["cells"]:
        cell["source"] = [_replace_all(line, replacements) for line in cell.get("source", [])]

    setup = "".join(notebook["cells"][1]["source"])
    setup = setup.replace("PROJECT_ROOT\n", "EXPORT_MODEL_WEIGHTS = False\nPROJECT_ROOT\n")
    notebook["cells"][1]["source"] = setup.splitlines(keepends=True)

    imports = "".join(notebook["cells"][2]["source"])
    imports = imports.replace(
        "    StationStackingConfig,\n",
        "    StationStackingConfig,\n"
        "    V11_SETTLEMENT_FIX_TEMP_FEATURE_COLUMNS,\n"
        "    _fit_feature_columns,\n"
        "    _modeling_frame,\n",
    )
    notebook["cells"][2]["source"] = imports.splitlines(keepends=True)

    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if "exported_weights = export_station_model_weights(" in source:
            indented = "\n".join(f"    {line}" if line else "" for line in source.splitlines())
            source = (
                "if EXPORT_MODEL_WEIGHTS:\n"
                f"{indented}\n"
                "else:\n"
                "    print(\"Model export disabled for this experimental notebook.\")\n"
            )
            cell["source"] = source.splitlines(keepends=True)

    notebook["cells"].extend(_reporting_cells())
    return notebook


def _reporting_cells() -> list[dict]:
    return [
        _markdown("## Train-Fold 3% Missingness Audit\n"),
        _code(
            """modeling_frame, candidate_categorical, candidate_numeric = _modeling_frame(result.features, config)
candidate_features = [*candidate_categorical, *candidate_numeric]
audit_specs = [
    (fold.name, fold.train_start_year, fold.train_end_year)
    for fold in YEAR_SPLIT_EXPANDING_FOLDS
] + [("test_refit_2021_2025", 2021, 2025)]

missingness_rows = []
years = pd.to_numeric(modeling_frame["year"], errors="coerce")
for fold_name, train_start, train_end in audit_specs:
    train = modeling_frame.loc[years.between(train_start, train_end)].copy()
    retained_categorical, retained_numeric = _fit_feature_columns(
        train,
        candidate_categorical,
        candidate_numeric,
        max_missing_fraction=config.effective_max_feature_missing_fraction,
    )
    retained = set(retained_categorical) | set(retained_numeric)
    for feature in candidate_features:
        numeric_feature = feature in candidate_numeric
        values = pd.to_numeric(train[feature], errors="coerce") if numeric_feature else train[feature]
        missingness_rows.append(
            {
                "fold": fold_name,
                "train_start_year": train_start,
                "train_end_year": train_end,
                "feature": feature,
                "kind": "numeric" if numeric_feature else "categorical",
                "missing_fraction": float(values.isna().mean()),
                "retained": feature in retained,
            }
        )

fold_feature_missingness = pd.DataFrame(missingness_rows)
retained_dropped_summary = (
    fold_feature_missingness.groupby(["fold", "retained"], as_index=False)
    .agg(feature_count=("feature", "nunique"), maximum_missing_fraction=("missing_fraction", "max"))
)
fold_feature_missingness.to_csv(config.resolved_output_dir() / f"{STATION_ID}_fold_feature_missingness.csv", index=False)
retained_dropped_summary, fold_feature_missingness.loc[~fold_feature_missingness["retained"]].sort_values(
    ["fold", "missing_fraction"], ascending=[True, False]
)
"""
        ),
        _markdown("## Expanded 11 AM Feature Coverage and Provider Count\n"),
        _code(
            """new_feature_coverage = (
    result.features[V11_SETTLEMENT_FIX_TEMP_FEATURE_COLUMNS]
    .notna()
    .mean()
    .mul(100)
    .rename("coverage_pct")
    .reset_index()
    .rename(columns={"index": "feature"})
)
provider_count_coverage = (
    result.features["v11sf_forecast_temp_11am_provider_count"]
    .value_counts(dropna=False)
    .sort_index()
    .rename_axis("available_provider_count")
    .reset_index(name="row_count")
)
provider_count_coverage["row_pct"] = provider_count_coverage["row_count"] / len(result.features) * 100
new_feature_coverage.to_csv(config.resolved_output_dir() / f"{STATION_ID}_11am_feature_coverage.csv", index=False)
new_feature_coverage, provider_count_coverage
"""
        ),
        _markdown("## New-Feature Importance\n"),
        _code(
            """new_feature_importance = result.feature_importance.loc[
    result.feature_importance["feature"].isin(V11_SETTLEMENT_FIX_TEMP_FEATURE_COLUMNS)
].sort_values(["method", "importance_mean_mae_f"], ascending=[True, False])
new_feature_importance
"""
        ),
        _markdown("## 2026 Monthly Metrics\n"),
        _code(
            """monthly_predictions = result.test_predictions.copy()
monthly_predictions["month"] = pd.to_datetime(monthly_predictions["contract_date"], errors="coerce").dt.month
monthly_metrics = (
    monthly_predictions.dropna(subset=["month", "error_f"])
    .groupby(["method", "month"], as_index=False)
    .agg(
        count=("error_f", "size"),
        mae_f=("absolute_error_f", "mean"),
        rmse_f=("error_f", lambda values: float(np.sqrt(np.mean(np.square(values))))),
        bias_f=("error_f", "mean"),
    )
)
monthly_metrics.to_csv(config.resolved_output_dir() / f"{STATION_ID}_2026_monthly_metrics.csv", index=False)
monthly_metrics
"""
        ),
        _markdown("## Performance by Warm/Cool 11 AM Forecast Delta\n"),
        _code(
            """delta_by_date = result.features[[
    "contract_date",
    "v11sf_forecast_temp_11am_minus_observed_f",
]].copy()
delta_predictions = result.test_predictions.merge(delta_by_date, on="contract_date", how="left")
delta_predictions["forecast_temp_delta_bucket"] = pd.cut(
    delta_predictions["v11sf_forecast_temp_11am_minus_observed_f"],
    bins=[-np.inf, -2.0, -0.5, 0.5, 2.0, np.inf],
    labels=["cool_gt_2f", "cool_0.5_to_2f", "near_match", "warm_0.5_to_2f", "warm_gt_2f"],
)
warm_cool_metrics = (
    delta_predictions.dropna(subset=["forecast_temp_delta_bucket", "error_f"])
    .groupby(["method", "forecast_temp_delta_bucket"], observed=True, as_index=False)
    .agg(count=("error_f", "size"), mae_f=("absolute_error_f", "mean"), bias_f=("error_f", "mean"))
)
warm_cool_metrics.to_csv(config.resolved_output_dir() / f"{STATION_ID}_warm_cool_delta_metrics.csv", index=False)
warm_cool_metrics
"""
        ),
        _markdown("## Common-Date Comparison with Existing V11 Settlement\n"),
        _code(
            """baseline_path = (
    PROJECT_ROOT
    / "data"
    / "calibration"
    / "station_stacking_v11_settlement"
    / f"{STATION_ID}_year_split_test_predictions.csv"
)
baseline_predictions = pd.read_csv(baseline_path)
baseline_predictions["contract_date"] = baseline_predictions["contract_date"].astype(str).str[:10]
fix_predictions = result.test_predictions.copy()
fix_predictions["contract_date"] = fix_predictions["contract_date"].astype(str).str[:10]
comparison = baseline_predictions.merge(
    fix_predictions,
    on=["contract_date", "method"],
    suffixes=("_baseline", "_fix"),
)
comparison["baseline_abs_error_f"] = (
    pd.to_numeric(comparison["actual_high_f_baseline"], errors="coerce")
    - pd.to_numeric(comparison["predicted_high_f_baseline"], errors="coerce")
).abs()
comparison["fix_abs_error_f"] = (
    pd.to_numeric(comparison["actual_high_f_fix"], errors="coerce")
    - pd.to_numeric(comparison["predicted_high_f_fix"], errors="coerce")
).abs()
common_date_comparison = (
    comparison.groupby("method", as_index=False)
    .agg(
        common_date_count=("contract_date", "size"),
        baseline_mae_f=("baseline_abs_error_f", "mean"),
        fix_mae_f=("fix_abs_error_f", "mean"),
        fix_better_days=("fix_abs_error_f", lambda values: int((values < comparison.loc[values.index, "baseline_abs_error_f"]).sum())),
        baseline_better_days=("fix_abs_error_f", lambda values: int((values > comparison.loc[values.index, "baseline_abs_error_f"]).sum())),
    )
)
common_date_comparison["delta_mae_f"] = common_date_comparison["fix_mae_f"] - common_date_comparison["baseline_mae_f"]
common_date_comparison.to_csv(config.resolved_output_dir() / f"{STATION_ID}_v11_common_date_comparison.csv", index=False)
common_date_comparison.sort_values("delta_mae_f")
"""
        ),
    ]


def _replace_all(text: str, replacements: list[tuple[str, str]]) -> str:
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    for station in TARGET_STATIONS:
        notebook = _notebook(station)
        path = out_dir / f"stacking_{station}_v11_settlement_fix.ipynb"
        path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
