from __future__ import annotations

import importlib.util
import json
from pathlib import Path


STATION_ID = "KDAL"


def _load_v20_generator():
    source = (
        Path(__file__).resolve().parents[1]
        / "station_stacking_v20_kdal_no_peak"
        / "generate_station_notebook.py"
    )
    spec = importlib.util.spec_from_file_location("station_stacking_v20_kdal_no_peak_generator", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load KDAL V20 no-peak notebook generator from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _notebook() -> dict:
    v20 = _load_v20_generator()
    notebook = v20._notebook()
    replacements = [
        ("Station Stacking V20 KDAL No Peak", "Station Stacking V24 KDAL Diverse Ensemble"),
        ("KDAL Station Stacking V20 No Peak", "KDAL Station Stacking V24 Diverse Ensemble"),
        (
            "station_high_regressor_v20_kdal_no_peak_stack",
            "station_high_regressor_v24_kdal_no_peak_diverse_stack",
        ),
        (
            'base_model_methods=("xgboost", "lightgbm", "catboost")',
            'base_model_methods=("xgboost", "extra_trees", "ridge")',
        ),
        (
            'output_dir=PROJECT_ROOT / "data" / "calibration" / "station_stacking_v20_kdal_no_peak"',
            'output_dir=PROJECT_ROOT / "data" / "calibration" / '
            '"station_stacking_v24_kdal_no_peak_diverse_ensemble"',
        ),
        (
            'source_pipeline="notebooks/experiments/station_stacking_v20_kdal_no_peak"',
            'source_pipeline="notebooks/experiments/station_stacking_v24_kdal_no_peak_diverse_ensemble"',
        ),
    ]
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        for old, new in replacements:
            source = source.replace(old, new)
        cell["source"] = source.splitlines(keepends=True)

    notebook["cells"][0]["source"] = [
        "# KDAL Station Stacking V24 Diverse Ensemble\n",
        "\n",
        "KDAL-only ablation based on V20 no-peak. It preserves the live-safe 11 AM feature, target, "
        "missingness, and expanding-year validation contracts while replacing the three boosted-tree "
        "base learners with XGBoost, Extra Trees, and scaled Ridge. The Ridge meta-model and raw-provider "
        "candidate features remain enabled, and the fitted bundle is exported after a successful run.\n",
    ]
    return notebook


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "stacking_KDAL_v24_diverse_ensemble.ipynb"
    path.write_text(json.dumps(_notebook(), indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
