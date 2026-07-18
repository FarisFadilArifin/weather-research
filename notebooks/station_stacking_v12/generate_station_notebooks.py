import importlib.util
import json
from pathlib import Path


TARGET_STATIONS = ("KATL", "KAUS", "KORD", "KDAL", "KHOU", "KLAX", "KMIA", "KLGA", "KSEA")

# Source-owned v12 contract markers:
# feature_version="v12"
# timing_mode=TIMING_MODE
# providers=PROVIDERS
# target_mode="remaining_warmup"
# target_source="settlement_first"
# base_model_methods=("xgboost", "lightgbm", "catboost")
# stack_enabled=True
# guarded_blend_cap_candidates=(1.0, 2.0, 3.0)
# export_station_model_weights
# year_split_folds=YEAR_SPLIT_EXPANDING_FOLDS
# hyperparameter_space="wide"
# MODEL_VERSION = "station_high_regressor_v12_guarded_blend"
# OPTUNA_TRIALS = 30
# OPTUNA_STARTUP_TRIALS = 15
# STACK_OPTUNA_TRIALS = 30
# STACK_OPTUNA_STARTUP_TRIALS = 15


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
        ("Station Stacking v11", "Station Stacking v12"),
        ('feature_version="v11"', 'feature_version="v12"'),
        (
            'MODEL_VERSION = "station_high_regressor_v11_huber_ridge_stack"',
            'MODEL_VERSION = "station_high_regressor_v12_guarded_blend"',
        ),
        (
            'output_dir=PROJECT_ROOT / "data" / "calibration" / "station_stacking_v11"',
            'output_dir=PROJECT_ROOT / "data" / "calibration" / "station_stacking_v12"',
        ),
        (
            '    ("v11", PROJECT_ROOT / "data" / "calibration" / "station_stacking_v11"),',
            '    ("v11", PROJECT_ROOT / "data" / "calibration" / "station_stacking_v11"),\n'
            '    ("v12", PROJECT_ROOT / "data" / "calibration" / "station_stacking_v12"),',
        ),
        ("V11_DROPPED_FEATURE_COLUMNS", "V12_DROPPED_FEATURE_COLUMNS"),
        ("V11_FEATURE_COLUMNS", "V12_FEATURE_COLUMNS"),
        ("V11 Contract", "V12 Contract"),
        ("V11 Feature Coverage", "V12 Feature Coverage"),
        ("v11_feature_coverage", "v12_feature_coverage"),
        ("## V11", "## V12"),
        ("`feature_version=\\\"v11\\\"`", "`feature_version=\\\"v12\\\"`"),
        (
            "This version keeps the v9 remaining-warmup feature contract, trains XGBoost/LightGBM/CatBoost with Huber-style objectives, keeps ridge stacking enabled, and writes artifacts to `data/calibration/station_stacking_v11`.",
            "This version keeps the v11 feature and Huber objective lineage, uses settlement-first labels, evaluates guarded provider-mean blend caps, and writes artifacts to `data/calibration/station_stacking_v12`.",
        ),
        (
            '`feature_version="v11"` keeps the v9 feature contract and remaining-warmup target, but trains base learners with Huber-style objectives while retaining the ridge stack selected by validation MAE.',
            '`feature_version="v12"` keeps the v11 feature contract and remaining-warmup target, adds settlement-first target-source reporting, and evaluates 1F/2F/3F provider-mean capped stack predictions.',
        ),
        (
            '`feature_version="v12"` keeps the v9 feature contract and remaining-warmup target, but trains base learners with Huber-style objectives while retaining the ridge stack selected by validation MAE.',
            '`feature_version="v12"` keeps the v11 feature contract and remaining-warmup target, adds settlement-first target-source reporting, and evaluates 1F/2F/3F provider-mean capped stack predictions.',
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
        ('source_pipeline="notebooks/station_stacking_v11"', 'source_pipeline="notebooks/station_stacking_v12"'),
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
        path = out_dir / f"stacking_{station}_v12.ipynb"
        path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
