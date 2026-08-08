import importlib.util
import json
from pathlib import Path


# The historical 9 AM forecast and observation pulls currently cover these stations.
TARGET_STATIONS = ("KATL", "KDAL", "KMIA", "KSEA")


def _load_v11_settlement_generator():
    source = Path(__file__).resolve().parents[1] / "station_stacking_v11_settlement" / "generate_station_notebooks.py"
    spec = importlib.util.spec_from_file_location("station_stacking_v11_settlement_generator", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load v11 settlement notebook generator from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _notebook(station_id: str) -> dict:
    v11_settlement = _load_v11_settlement_generator()
    notebook = v11_settlement._notebook(station_id)
    replacements = [
        ("Station Stacking v11 Settlement", "Station Stacking v11 9AM Settlement"),
        (
            "This controlled rerun keeps the exact v11 feature/model contract, replaces daily-high labels with settlement-first Wunderground station history when available, and writes separate artifacts to `data/calibration/station_stacking_v11_settlement`.",
            "This controlled rerun keeps the exact v11 feature/model contract, uses only point-in-time-safe 9 AM forecast and observation data, requires Wunderground station-history daily-high labels with no fallback, and writes separate artifacts to `data/calibration/station_stacking_v11_9AM_settlement`.",
        ),
        ('TIMING_MODE = "same_day_11am_live_safe"', 'TIMING_MODE = "same_day_9am_live_safe"'),
        (
            'MODEL_VERSION = "station_high_regressor_v11_wunderground_settlement_stack"',
            'MODEL_VERSION = "station_high_regressor_v11_9AM_wunderground_settlement_stack"',
        ),
        (
            'output_dir=PROJECT_ROOT / "data" / "calibration" / "station_stacking_v11_settlement"',
            'output_dir=PROJECT_ROOT / "data" / "calibration" / "station_stacking_v11_9AM_settlement"',
        ),
        ('    target_source="settlement_first",\n', '    target_source="wunderground_only",\n'),
        (
            'source_pipeline="notebooks/experiments/station_stacking_v11_settlement"',
            'source_pipeline="notebooks/experiments/station_stacking_v11_9AM_settlement"',
        ),
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
    out_dir.mkdir(parents=True, exist_ok=True)
    for station in TARGET_STATIONS:
        notebook = _notebook(station)
        path = out_dir / f"stacking_{station}_v11_9AM_settlement.ipynb"
        path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
