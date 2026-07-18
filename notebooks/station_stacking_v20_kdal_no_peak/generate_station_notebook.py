from __future__ import annotations

import importlib.util
import json
from pathlib import Path


STATION_ID = "KDAL"

# Regression-test markers for the KDAL V20-aligned no-peak arm:
# feature_version="v11_settlement_fix_temp"
# target_source="wunderground_only"
# max_feature_missing_fraction=0.03
# year_split_folds=V20_EXPANDING_FOLDS
# year_split_validation_weights={2022: 1.0, 2023: 1.0, 2024: 1.0, 2025: 1.0}
# EXPORT_MODEL_WEIGHTS = True
# No V20 peak-timing feature family is loaded by this experiment.


def _load_v11_fix_generator():
    source = Path(__file__).resolve().parents[1] / "station_stacking_v11_settlement_fix" / "generate_station_notebooks.py"
    spec = importlib.util.spec_from_file_location("station_stacking_v11_fix_generator", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load V11 Settlement Fix notebook generator from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _notebook() -> dict:
    v11_fix = _load_v11_fix_generator()
    notebook = v11_fix._notebook(STATION_ID)
    replacements = [
        ("Station Stacking v11 Settlement Fix", "Station Stacking V20 KDAL No Peak"),
        (
            "station_high_regressor_v11_settlement_fix_temp_stack",
            "station_high_regressor_v20_kdal_no_peak_stack",
        ),
        ('target_source="settlement_first"', 'target_source="wunderground_only"'),
        ("YEAR_SPLIT_EXPANDING_FOLDS", "V20_EXPANDING_FOLDS"),
        (
            "year_split_folds=V20_EXPANDING_FOLDS,\n",
            "year_split_folds=V20_EXPANDING_FOLDS,\n"
            "    year_split_validation_weights={2022: 1.0, 2023: 1.0, 2024: 1.0, 2025: 1.0},\n",
        ),
        (
            'output_dir=PROJECT_ROOT / "data" / "calibration" / "station_stacking_v11_settlement_fix"',
            'output_dir=PROJECT_ROOT / "data" / "calibration" / "station_stacking_v20_kdal_no_peak"',
        ),
        (
            'source_pipeline="notebooks/station_stacking_v11_settlement_fix"',
            'source_pipeline="notebooks/station_stacking_v20_kdal_no_peak"',
        ),
        ("EXPORT_MODEL_WEIGHTS = False", "EXPORT_MODEL_WEIGHTS = True"),
    ]
    for cell in notebook["cells"]:
        cell["source"] = [_replace_all(line, replacements) for line in cell.get("source", [])]

    notebook["cells"][0]["source"] = [
        "# KDAL Station Stacking V20 No Peak\n",
        "\n",
        "KDAL-only V20-aligned experiment using the V11 Settlement Fix live-safe 11 AM temperature-alignment "
        "features with no HRRR/NBM peak-timing feature family. It uses Wunderground-only targets, a 3% "
        "train-fold missingness gate, four equal-weight expanding validation folds covering 2022-2025, and "
        "exports the fitted model bundle after a successful full run.\n",
    ]
    return notebook


def _replace_all(text: str, replacements: list[tuple[str, str]]) -> str:
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "stacking_KDAL_v20_no_peak.ipynb"
    path.write_text(json.dumps(_notebook(), indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
