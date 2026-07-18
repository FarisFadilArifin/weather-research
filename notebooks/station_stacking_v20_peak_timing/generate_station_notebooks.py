from __future__ import annotations

import importlib.util
import json
from pathlib import Path


TARGET_STATIONS = ("KATL", "KDAL")

# Regression-test markers for the single V20 arm:
# training_profile="v20_aligned"
# feature_version="v20_peak_timing"
# target_source="wunderground_only"
# max_feature_missing_fraction=0.03
# year_split_folds=V20_EXPANDING_FOLDS
# EXPORT_MODEL_WEIGHTS = False
# EVALUATION_END_DATE = "2026-07-01"  # provisional until July 2-14 labels arrive


def _load_v11_fix_generator():
    source = Path(__file__).resolve().parents[1] / "station_stacking_v11_settlement_fix" / "generate_station_notebooks.py"
    spec = importlib.util.spec_from_file_location("station_stacking_v11_fix_generator", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load V11 Settlement Fix notebook generator from {source}")
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
    v11_fix = _load_v11_fix_generator()
    notebook = v11_fix._notebook(station_id)
    replacements = [
        ("Station Stacking v11 Settlement Fix", "Station Stacking V20 Peak Timing"),
        ("station_high_regressor_v11_settlement_fix_temp_stack", "station_high_regressor_v20_peak_timing_stack"),
        ('feature_version="v11_settlement_fix_temp"', 'feature_version="v20_peak_timing"'),
        ('target_source="settlement_first"', 'target_source="wunderground_only"'),
        ("YEAR_SPLIT_EXPANDING_FOLDS", "V20_EXPANDING_FOLDS"),
        (
            'year_split_folds=V20_EXPANDING_FOLDS,\n',
            'year_split_folds=V20_EXPANDING_FOLDS,\n'
            '    year_split_validation_weights={2022: 1.0, 2023: 1.0, 2024: 1.0, 2025: 1.0},\n',
        ),
        ('output_dir=PROJECT_ROOT / "data" / "calibration" / "station_stacking_v11_settlement_fix"',
         'output_dir=PROJECT_ROOT / "data" / "calibration" / "station_stacking_v20_peak_timing"'),
        ('source_pipeline="notebooks/station_stacking_v11_settlement_fix"',
         'source_pipeline="notebooks/station_stacking_v20_peak_timing"'),
        ('/ "station_stacking_v11_settlement"', '/ "station_stacking_v11_settlement_fix"'),
        ("fix_predictions", "v20_predictions"),
        ("fix_abs_error_f", "v20_abs_error_f"),
        ("fix_mae_f", "v20_mae_f"),
        ("fix_better_days", "v20_better_days"),
        ("delta_mae_f", "v20_delta_mae_f"),
        ("_fix", "_v20"),
        ("station_stacking_v11_settlement_v20", "station_stacking_v11_settlement_fix"),
        ("Common-Date Comparison with Existing V11 Settlement", "Common-Date Comparison with Existing V11 Settlement Fix"),
    ]
    for cell in notebook["cells"]:
        cell["source"] = [_replace_all(line, replacements) for line in cell.get("source", [])]

    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if "config = StationStackingConfig(" in source:
            source = source.replace(
                '    feature_version="v20_peak_timing",\n',
                '    feature_version="v20_peak_timing",\n'
                '    training_profile="v20_aligned",\n',
            )
        if "export_station_model_weights(" in source:
            source = source.replace(
                "        feature_version=config.effective_feature_version,\n",
                "        feature_version=config.effective_feature_version,\n"
                "        training_profile=config.effective_training_profile,\n",
            )
        cell["source"] = source.splitlines(keepends=True)

    notebook["cells"][0]["source"] = [
        f"# {station_id} Station Stacking V20 Peak Timing\n",
        "\n",
        "Single-arm Wunderground-only experiment using V11 Settlement Fix temperature alignment, curated live-safe HRRR/NBM peak-timing features, a 3% train-fold missingness gate, and expanding 2021–2025 validation folds. The readiness section can run while shards are still being pulled and blocks model tuning until coverage is sufficient.\n",
    ]

    imports = "".join(notebook["cells"][2]["source"])
    imports = imports.replace(
        "    StationStackingConfig,\n",
        "    StationStackingConfig,\n"
        "    V20_ENGINEERED_FEATURE_COLUMNS,\n"
        "    V20_PEAK_TIMING_RAW_FEATURE_COLUMNS,\n"
        "    build_station_wide_dataset,\n"
        "    v20_peak_timing_readiness,\n",
    )
    notebook["cells"][2]["source"] = imports.splitlines(keepends=True)

    readiness_cells = [
        _markdown("## Peak-Timing and Wunderground Readiness Gate\n"),
        _code(
            """readiness_features = build_station_wide_dataset(
    PROJECT_ROOT,
    station_id=STATION_ID,
    timing_mode=TIMING_MODE,
    providers=PROVIDERS,
    feature_version="v20_peak_timing",
    target_source="wunderground_only",
)
EVALUATION_END_DATE = "2026-07-01"  # provisional; official V20 cutoff remains 2026-07-14
v20_ready, readiness_summary, readiness_missing_dates, readiness_fold_missingness = v20_peak_timing_readiness(
    readiness_features,
    station_id=STATION_ID,
    folds=V20_EXPANDING_FOLDS,
    max_missing_fraction=0.03,
    end_date=EVALUATION_END_DATE,
)
readiness_dir = PROJECT_ROOT / "data" / "calibration" / "station_stacking_v20_peak_timing"
readiness_dir.mkdir(parents=True, exist_ok=True)
readiness_summary.to_csv(readiness_dir / f"{STATION_ID}_readiness_summary.csv", index=False)
readiness_missing_dates.to_csv(readiness_dir / f"{STATION_ID}_readiness_missing_dates.csv", index=False)
readiness_fold_missingness.to_csv(readiness_dir / f"{STATION_ID}_readiness_fold_feature_missingness.csv", index=False)
display(readiness_summary)
display(readiness_fold_missingness.groupby(["fold", "retained"], as_index=False).agg(feature_count=("feature", "nunique")))
if not v20_ready:
    raise RuntimeError(
        "V20 readiness failed. Audit artifacts were written; rerun this notebook after peak-timing and "
        "Wunderground pulls reduce each station-year missing fraction to 3% or less."
    )
"""
        ),
    ]
    config_index = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if "config = StationStackingConfig(" in "".join(cell.get("source", []))
    )
    notebook["cells"][config_index:config_index] = readiness_cells
    notebook["cells"].extend(_v20_reporting_cells())
    return notebook


def _v20_reporting_cells() -> list[dict]:
    return [
        _markdown("## V20 Peak-Timing Feature Coverage\n"),
        _code(
            """v20_feature_columns = [*V20_PEAK_TIMING_RAW_FEATURE_COLUMNS, *V20_ENGINEERED_FEATURE_COLUMNS]
v20_feature_coverage = (
    result.features[v20_feature_columns]
    .notna().mean().mul(100)
    .rename("coverage_pct").reset_index().rename(columns={"index": "feature"})
)
v20_feature_coverage.to_csv(
    config.resolved_output_dir() / f"{STATION_ID}_v20_peak_feature_coverage.csv", index=False
)
v20_feature_coverage.sort_values("coverage_pct")
"""
        ),
        _markdown("## Fold Metrics and Peak-Feature Importance\n"),
        _code(
            """fold_metrics = (
    result.validation_predictions.groupby(["fold", "method"], as_index=False)
    .agg(count=("absolute_error_f", "size"), mae_f=("absolute_error_f", "mean"), bias_f=("error_f", "mean"))
)
peak_feature_importance = result.feature_importance.loc[
    result.feature_importance["feature"].isin(v20_feature_columns)
].sort_values(["method", "importance_mean_mae_f"], ascending=[True, False])
fold_metrics.to_csv(config.resolved_output_dir() / f"{STATION_ID}_fold_metrics.csv", index=False)
fold_metrics, peak_feature_importance
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
        path = out_dir / f"stacking_{station}_v20_peak_timing.ipynb"
        path.write_text(json.dumps(_notebook(station), indent=2), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
