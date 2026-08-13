import importlib.util
import json
from pathlib import Path


TARGET_STATIONS = ("KATL", "KAUS", "KORD", "KDAL", "KHOU", "KLAX", "KMIA", "KLGA", "KSEA")


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
        ("Station Stacking v11 Settlement", "Station Stacking v18 Wunderground Physics"),
        (
            "This controlled rerun keeps the exact v11 feature/model contract, replaces daily-high labels with settlement-first Wunderground station history when available, and writes separate artifacts to `data/calibration/station_stacking_v11_settlement`.",
            "This v18 run keeps the v11 remaining-warmup ridge-stack backbone, requires Wunderground station-history labels, adds NBM hourly curve and HRRR/RAP physics shard features, and writes isolated artifacts to `data/calibration/station_stacking_v18`.",
        ),
        (
            '`feature_version="v11"` keeps the v9 feature contract and remaining-warmup target, but trains base learners with Huber-style objectives while retaining the ridge stack selected by validation MAE.',
            '`feature_version="v18"` adds coverage-gated NBM hourly curve and HRRR/RAP physics features while selecting by validation MAE.',
        ),
        (
            'MODEL_VERSION = "station_high_regressor_v11_wunderground_settlement_stack"',
            'MODEL_VERSION = "station_high_regressor_v18_nbm_hrrr_physics_settlement_stack"',
        ),
        ("OPTUNA_TRIALS = 30", "OPTUNA_TRIALS = 100"),
        ("STACK_OPTUNA_TRIALS = 30", "STACK_OPTUNA_TRIALS = 100"),
        ("OPTUNA_STARTUP_TRIALS = 15", "OPTUNA_STARTUP_TRIALS = 40"),
        ("STACK_OPTUNA_STARTUP_TRIALS = 15", "STACK_OPTUNA_STARTUP_TRIALS = 40"),
        ('hyperparameter_space="wide"', 'hyperparameter_space="wide_plus"'),
        ('feature_version="v11"', 'feature_version="v18"'),
        ('target_source="settlement_first"', 'target_source="wunderground_only"'),
        (
            'output_dir=PROJECT_ROOT / "data" / "calibration" / "station_stacking_v11_settlement"',
            'output_dir=PROJECT_ROOT / "data" / "calibration" / "station_stacking_v18"',
        ),
        ('source_pipeline="notebooks/experiments/station_stacking_v11_settlement"', 'source_pipeline="notebooks/experiments/station_stacking_v18"'),
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
        path = out_dir / f"stacking_{station}_v18.ipynb"
        path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
