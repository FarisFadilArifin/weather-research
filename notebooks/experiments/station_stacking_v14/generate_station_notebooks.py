import importlib.util
import json
from pathlib import Path


TARGET_STATIONS = ("KATL", "KAUS", "KORD", "KDAL", "KHOU", "KLAX", "KMIA", "KLGA", "KSEA")

# Source-owned v14 contract markers:
# feature_version="v14"
# timing_mode=TIMING_MODE
# providers=PROVIDERS
# target_mode="remaining_warmup"
# base_model_methods=("xgboost", "lightgbm", "catboost")
# stack_enabled=True
# export_station_model_weights
# year_split_folds=YEAR_SPLIT_EXPANDING_FOLDS
# hyperparameter_space="wide"
# MODEL_VERSION = "station_high_regressor_v14_curated_weather_stack"
# OPTUNA_TRIALS = 30
# OPTUNA_STARTUP_TRIALS = 15
# STACK_OPTUNA_TRIALS = 30
# STACK_OPTUNA_STARTUP_TRIALS = 15


def _load_v13_generator():
    source = Path(__file__).resolve().parents[1] / "station_stacking_v13" / "generate_station_notebooks.py"
    spec = importlib.util.spec_from_file_location("station_stacking_v13_generator", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load v13 notebook generator from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _notebook(station_id: str) -> dict:
    v13 = _load_v13_generator()
    notebook = v13._notebook(station_id)
    replacements = [
        ("Station Stacking v13", "Station Stacking v14"),
        ('feature_version="v13"', 'feature_version="v14"'),
        (
            'MODEL_VERSION = "station_high_regressor_v13_weather_warmup_stack"',
            'MODEL_VERSION = "station_high_regressor_v14_curated_weather_stack"',
        ),
        (
            'output_dir=PROJECT_ROOT / "data" / "calibration" / "station_stacking_v13"',
            'output_dir=PROJECT_ROOT / "data" / "calibration" / "station_stacking_v14"',
        ),
        (
            '    ("v13", PROJECT_ROOT / "data" / "calibration" / "station_stacking_v13"),',
            '    ("v13", PROJECT_ROOT / "data" / "calibration" / "station_stacking_v13"),\n'
            '    ("v14", PROJECT_ROOT / "data" / "calibration" / "station_stacking_v14"),',
        ),
        ("V13_DROPPED_FEATURE_COLUMNS", "V14_DROPPED_FEATURE_COLUMNS"),
        ("V13_FEATURE_COLUMNS", "V14_FEATURE_COLUMNS"),
        ("V13 Contract", "V14 Contract"),
        ("V13 Feature Coverage", "V14 Feature Coverage"),
        ("v13_feature_coverage", "v14_feature_coverage"),
        ("## V13", "## V14"),
        ("`feature_version=\\\"v13\\\"`", "`feature_version=\\\"v14\\\"`"),
        (
            "This version keeps the v11 remaining-warmup target and Huber/ridge stack, adds direct GFS/HRRR weather fields for rain/cloud/dewpoint/humidity, and writes artifacts to `data/calibration/station_stacking_v13`.",
            "This version keeps the v11 remaining-warmup target and Huber/ridge stack, computes v13 weather aggregates, trains only on a curated v11-plus-weather allowlist, and writes artifacts to `data/calibration/station_stacking_v14`.",
        ),
        (
            '`feature_version="v13"` keeps the v11 remaining-warmup target and stack semantics, then adds direct forecast-side weather features so rainy-day behavior can be evaluated against v11.',
            '`feature_version="v14"` keeps the v11 feature base, adds only curated aggregate weather features that pass coverage, and avoids raw provider weather feature sprawl.',
        ),
        (
            '`feature_version="v14"` keeps the v9 feature contract and remaining-warmup target, but trains base learners with Huber-style objectives while retaining the ridge stack selected by validation MAE.',
            '`feature_version="v14"` keeps the v11 remaining-warmup, Huber, and ridge-stack lineage, then adds only curated aggregate weather features that pass coverage.',
        ),
        (
            'source_pipeline="notebooks/experiments/station_stacking_v13"',
            'source_pipeline="notebooks/experiments/station_stacking_v14"',
        ),
        ("Rain-Day V11 vs V13 MAE", "Rain-Day V11 vs V14 MAE"),
        ('_prediction_mae_by_method(result.test_predictions, "v13", rainy_dates),', '_prediction_mae_by_method(result.test_predictions, "v14", rainy_dates),'),
        ('{"v11", "v13"}.issubset(rain_mae_wide.columns)', '{"v11", "v14"}.issubset(rain_mae_wide.columns)'),
        ('rain_mae_wide["v13_minus_v11_mae_f"] = rain_mae_wide["v13"] - rain_mae_wide["v11"]', 'rain_mae_wide["v14_minus_v11_mae_f"] = rain_mae_wide["v14"] - rain_mae_wide["v11"]'),
        ('"v13_minus_v11_mae_f"', '"v14_minus_v11_mae_f"'),
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
        path = out_dir / f"stacking_{station}_v14.ipynb"
        path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
