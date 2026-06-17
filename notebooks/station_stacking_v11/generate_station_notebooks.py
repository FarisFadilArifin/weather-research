import importlib.util
import json
from pathlib import Path


TARGET_STATIONS = ("KATL", "KAUS", "KORD", "KDAL", "KHOU", "KLAX", "KMIA", "KLGA", "KSEA")

# Source-owned v11 contract markers:
# feature_version="v11"
# timing_mode=TIMING_MODE
# providers=PROVIDERS
# target_mode="remaining_warmup"
# base_model_methods=("xgboost", "lightgbm", "catboost")
# stack_enabled=True
# export_station_model_weights
# year_split_folds=YEAR_SPLIT_EXPANDING_FOLDS
# hyperparameter_space="wide"
# MODEL_VERSION = "station_high_regressor_v11_huber_ridge_stack"
# OPTUNA_TRIALS = 30
# OPTUNA_STARTUP_TRIALS = 15
# STACK_OPTUNA_TRIALS = 30
# STACK_OPTUNA_STARTUP_TRIALS = 15


def _load_v9_generator():
    source = Path(__file__).resolve().parents[1] / "station_stacking_v9" / "generate_station_notebooks.py"
    spec = importlib.util.spec_from_file_location("station_stacking_v9_generator", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load v9 notebook generator from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _notebook(station_id: str) -> dict:
    v9 = _load_v9_generator()
    notebook = v9._notebook(station_id)
    replacements = [
        ("Station Stacking v9", "Station Stacking v11"),
        ('feature_version="v9"', 'feature_version="v11"'),
        (
            'MODEL_VERSION = "station_high_regressor_v9_remaining_warmup"',
            'MODEL_VERSION = "station_high_regressor_v11_huber_ridge_stack"',
        ),
        (
            'output_dir=PROJECT_ROOT / "data" / "calibration" / "station_stacking_v9"',
            'output_dir=PROJECT_ROOT / "data" / "calibration" / "station_stacking_v11"',
        ),
        (
            '    ("v9", PROJECT_ROOT / "data" / "calibration" / "station_stacking_v9"),',
            '    ("v9", PROJECT_ROOT / "data" / "calibration" / "station_stacking_v9"),\n'
            '    ("v10", PROJECT_ROOT / "data" / "calibration" / "station_stacking_v10"),\n'
            '    ("v11", PROJECT_ROOT / "data" / "calibration" / "station_stacking_v11"),',
        ),
        ("V9_DROPPED_FEATURE_COLUMNS", "V11_DROPPED_FEATURE_COLUMNS"),
        ("V9_FEATURE_COLUMNS", "V11_FEATURE_COLUMNS"),
        ("V9 Contract", "V11 Contract"),
        ("V9 Feature Coverage", "V11 Feature Coverage"),
        ("v9_feature_coverage", "v11_feature_coverage"),
        ("## V9", "## V11"),
        ("`feature_version=\\\"v9\\\"`", "`feature_version=\\\"v11\\\"`"),
        (
            "This version keeps the v8 live-safe GFS/HRRR/NBM contract, adds source-owned 10-year calendar-day max-temperature climatology features, and trains base learners on remaining warmup from the observed high-so-far. Artifacts are written to `data/calibration/station_stacking_v9`.",
            "This version keeps the v9 remaining-warmup feature contract, trains XGBoost/LightGBM/CatBoost with Huber-style objectives, keeps ridge stacking enabled, and writes artifacts to `data/calibration/station_stacking_v11`.",
        ),
        (
            '`feature_version="v11"` keeps the v8 remaining-warmup feature set, adds leakage-safe 10-year calendar-day climatology features, and uses `target_mode="remaining_warmup"` so base learners fit `actual_high_f - observed_high_temp_through_as_of_f` before converting back to `predicted_high_f`.',
            '`feature_version="v11"` keeps the v9 feature contract and remaining-warmup target, but trains base learners with Huber-style objectives while retaining the ridge stack selected by validation MAE.',
        ),
        (
            '    target_mode="remaining_warmup",\n',
            '    target_mode="remaining_warmup",\n'
            '    base_model_methods=("xgboost", "lightgbm", "catboost"),\n'
            '    stack_enabled=True,\n',
        ),
        (
            '    target_mode=config.effective_target_mode,\n',
            '    target_mode=config.effective_target_mode,\n'
            '    base_model_methods=tuple(config.effective_base_model_methods),\n'
            '    stack_enabled=config.stack_enabled,\n',
        ),
        ('source_pipeline="notebooks/station_stacking_v9"', 'source_pipeline="notebooks/station_stacking_v11"'),
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
        path = out_dir / f"stacking_{station}_v11.ipynb"
        path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
