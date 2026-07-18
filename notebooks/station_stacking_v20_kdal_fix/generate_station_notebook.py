from __future__ import annotations

import importlib.util
import json
from pathlib import Path


STATION_ID = "KDAL"

# Single KDAL patch markers:
# feature_version="v20_kdal_nbm_physics"
# target_source="wunderground_only"
# max_feature_missing_fraction=0.03
# EVALUATION_END_DATE = "2026-07-01"
# EXPORT_MODEL_WEIGHTS = False


def _load_v20_generator():
    source = Path(__file__).resolve().parents[1] / "station_stacking_v20_peak_timing" / "generate_station_notebooks.py"
    spec = importlib.util.spec_from_file_location("station_stacking_v20_generator", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load V20 notebook generator from {source}")
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


def _notebook() -> dict:
    v20 = _load_v20_generator()
    notebook = v20._notebook(STATION_ID)
    replacements = [
        ("Station Stacking V20 Peak Timing", "Station Stacking V20 KDAL NBM-Physics Fix"),
        ('feature_version="v20_peak_timing"', 'feature_version="v20_kdal_nbm_physics"'),
        ("station_high_regressor_v20_peak_timing_stack", "station_high_regressor_v20_kdal_nbm_physics_stack"),
        ("station_stacking_v20_peak_timing", "station_stacking_v20_kdal_fix"),
    ]
    for cell in notebook["cells"]:
        cell["source"] = [_replace_all(line, replacements) for line in cell.get("source", [])]
    notebook["cells"][0]["source"] = [
        "# KDAL Station Stacking V20 NBM-Physics Fix\n",
        "\n",
        "KDAL-only patch retaining NBM temperature timing and HRRR solar/cloud/precipitation physics while excluding HRRR temperature-curve features. The final ridge prediction receives a capped monthly correction learned only from forward OOF stack residuals. This provisional run evaluates through July 1, 2026.\n",
    ]

    imports = "".join(notebook["cells"][2]["source"])
    imports = imports.replace(
        "    StationStackingConfig,\n",
        "    StationStackingConfig,\n"
        "    V20_KDAL_NBM_PHYSICS_ENGINEERED_FEATURE_COLUMNS,\n"
        "    V20_KDAL_NBM_PHYSICS_RAW_FEATURE_COLUMNS,\n"
        "    kdal_oof_residual_calibrated_stack_predictions,\n",
    )
    notebook["cells"][2]["source"] = imports.splitlines(keepends=True)

    result_index = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if "result = run_station_year_split_experiment(config)" in "".join(cell.get("source", []))
    )
    notebook["cells"][result_index + 1 : result_index + 1] = _calibration_cells()
    notebook["cells"].extend(_comparison_cells())
    return notebook


def _calibration_cells() -> list[dict]:
    return [
        _markdown("## KDAL Forward-OOF Residual Calibration\n"),
        _code(
            """calibrated_predictions, residual_calibration, calibration_oof_predictions = (
    kdal_oof_residual_calibrated_stack_predictions(
        result.validation_predictions,
        result.test_predictions,
        result.stack_tuning_results,
        config,
        min_month_rows=60,
        shrinkage_rows=60,
        correction_cap_f=0.75,
    )
)
if calibrated_predictions.empty:
    raise RuntimeError("KDAL OOF residual calibration produced no test predictions.")

calibrated_predictions.to_csv(
    config.resolved_output_dir() / f"{STATION_ID}_oof_calibrated_test_predictions.csv", index=False
)
residual_calibration.to_csv(
    config.resolved_output_dir() / f"{STATION_ID}_oof_residual_calibration.csv", index=False
)
calibration_oof_predictions.to_csv(
    config.resolved_output_dir() / f"{STATION_ID}_stack_calibration_oof_predictions.csv", index=False
)
residual_calibration
"""
        ),
        _code(
            """calibrated_error = pd.to_numeric(calibrated_predictions["error_f"], errors="coerce")
calibrated_absolute_error = pd.to_numeric(calibrated_predictions["absolute_error_f"], errors="coerce")
calibrated_metrics = pd.DataFrame([
    {
        "method": "ridge_stack_oof_calibrated",
        "count": int(calibrated_absolute_error.notna().sum()),
        "mae_f": float(calibrated_absolute_error.mean()),
        "rmse_f": float(np.sqrt(np.mean(np.square(calibrated_error.dropna())))),
        "bias_f": float(calibrated_error.mean()),
        "p95_absolute_error_f": float(calibrated_absolute_error.quantile(0.95)),
        "large_error_5f_pct": float(calibrated_absolute_error.ge(5.0).mean() * 100.0),
        "within_1f_pct": float(calibrated_absolute_error.le(1.0).mean() * 100.0),
        "within_2f_pct": float(calibrated_absolute_error.le(2.0).mean() * 100.0),
        "within_3f_pct": float(calibrated_absolute_error.le(3.0).mean() * 100.0),
    }
])
calibrated_metrics.to_csv(
    config.resolved_output_dir() / f"{STATION_ID}_oof_calibrated_metrics.csv", index=False
)
calibrated_metrics
"""
        ),
    ]


def _comparison_cells() -> list[dict]:
    return [
        _markdown("## Calibrated Common-Date Comparison with KDAL V11 Settlement Fix\n"),
        _code(
            """v11_fix_path = (
    PROJECT_ROOT / "data" / "calibration" / "station_stacking_v11_settlement_fix"
    / f"{STATION_ID}_year_split_test_predictions.csv"
)
v11_fix = pd.read_csv(v11_fix_path)
v11_fix = v11_fix.loc[v11_fix["method"].eq("ridge_stack")].copy()
v11_fix["contract_date"] = v11_fix["contract_date"].astype(str).str[:10]
candidate = calibrated_predictions.copy()
candidate["contract_date"] = candidate["contract_date"].astype(str).str[:10]
common = candidate.merge(v11_fix, on="contract_date", suffixes=("_candidate", "_v11_fix"))
common["candidate_absolute_error_f"] = (
    pd.to_numeric(common["actual_high_f_candidate"], errors="coerce")
    - pd.to_numeric(common["predicted_high_f_candidate"], errors="coerce")
).abs()
common["v11_fix_absolute_error_f"] = (
    pd.to_numeric(common["actual_high_f_v11_fix"], errors="coerce")
    - pd.to_numeric(common["predicted_high_f_v11_fix"], errors="coerce")
).abs()
common_date_comparison = pd.DataFrame([
    {
        "common_date_count": len(common),
        "candidate_mae_f": common["candidate_absolute_error_f"].mean(),
        "v11_fix_mae_f": common["v11_fix_absolute_error_f"].mean(),
        "delta_mae_f": common["candidate_absolute_error_f"].mean() - common["v11_fix_absolute_error_f"].mean(),
        "candidate_better_days": int((common["candidate_absolute_error_f"] < common["v11_fix_absolute_error_f"]).sum()),
        "v11_fix_better_days": int((common["candidate_absolute_error_f"] > common["v11_fix_absolute_error_f"]).sum()),
    }
])
common_date_comparison.to_csv(
    config.resolved_output_dir() / f"{STATION_ID}_calibrated_v11_fix_common_date_comparison.csv", index=False
)
common_date_comparison
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
    path = out_dir / "stacking_KDAL_v20_kdal_fix.ipynb"
    path.write_text(json.dumps(_notebook(), indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
