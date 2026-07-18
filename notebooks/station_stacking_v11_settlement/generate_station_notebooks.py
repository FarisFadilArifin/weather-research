import importlib.util
import json
from pathlib import Path


TARGET_STATIONS = ("KATL", "KAUS", "KORD", "KDAL", "KHOU", "KLAX", "KMIA", "KLGA", "KSEA")


def _load_v11_generator():
    source = Path(__file__).resolve().parents[1] / "station_stacking_v11" / "generate_station_notebooks.py"
    spec = importlib.util.spec_from_file_location("station_stacking_v11_generator", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load v11 notebook generator from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _notebook(station_id: str) -> dict:
    v11 = _load_v11_generator()
    notebook = v11._notebook(station_id)
    replacements = [
        ("Station Stacking v11", "Station Stacking v11 Settlement"),
        (
            'MODEL_VERSION = "station_high_regressor_v11_huber_ridge_stack"',
            'MODEL_VERSION = "station_high_regressor_v11_wunderground_settlement_stack"',
        ),
        (
            'output_dir=PROJECT_ROOT / "data" / "calibration" / "station_stacking_v11"',
            'output_dir=PROJECT_ROOT / "data" / "calibration" / "station_stacking_v11_settlement"',
        ),
        (
            "This version keeps the v9 remaining-warmup feature contract, trains XGBoost/LightGBM/CatBoost with Huber-style objectives, keeps ridge stacking enabled, and writes artifacts to `data/calibration/station_stacking_v11`.",
            "This controlled rerun keeps the exact v11 feature/model contract, replaces daily-high labels with settlement-first Wunderground station history when available, and writes separate artifacts to `data/calibration/station_stacking_v11_settlement`.",
        ),
        (
            '    target_mode="remaining_warmup",\n',
            '    target_mode="remaining_warmup",\n'
            '    target_source="settlement_first",\n',
        ),
        (
            '    target_mode=config.effective_target_mode,\n',
            '    target_mode=config.effective_target_mode,\n'
            '    target_source=config.effective_target_source,\n',
        ),
        ('source_pipeline="notebooks/station_stacking_v11"', 'source_pipeline="notebooks/station_stacking_v11_settlement"'),
    ]
    for cell in notebook["cells"]:
        cell["source"] = [_replace_all(line, replacements) for line in cell.get("source", [])]
    return notebook


def _replace_all(text: str, replacements: list[tuple[str, str]]) -> str:
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    for station in TARGET_STATIONS:
        notebook = _notebook(station)
        path = out_dir / f"stacking_{station}_v11_settlement.ipynb"
        path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
