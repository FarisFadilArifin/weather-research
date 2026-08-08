import importlib.util
import json
from pathlib import Path


TARGET_STATIONS = ("KATL", "KAUS", "KORD", "KDAL", "KHOU", "KLAX", "KMIA", "KLGA", "KSEA")

# Source-owned v9 contract markers:
# feature_version="v9"
# timing_mode=TIMING_MODE
# providers=PROVIDERS
# target_mode="remaining_warmup"
# year_split_folds=YEAR_SPLIT_EXPANDING_FOLDS
# hyperparameter_space="wide"
# OPTUNA_TRIALS = 30
# OPTUNA_STARTUP_TRIALS = 15


def _load_v8_generator():
    source = Path(__file__).resolve().parents[1] / "station_stacking_v8" / "generate_station_notebooks.py"
    spec = importlib.util.spec_from_file_location("station_stacking_v8_generator", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load v8 notebook generator from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _notebook(station_id: str) -> dict:
    v8 = _load_v8_generator()
    notebook = v8._notebook(station_id)
    replacements = [
        ("Station Stacking v8", "Station Stacking v9"),
        ("feature_version=\"v8\"", "feature_version=\"v9\""),
        (
            'output_dir=PROJECT_ROOT / "data" / "calibration" / "station_stacking_v8"',
            'output_dir=PROJECT_ROOT / "data" / "calibration" / "station_stacking_v9"',
        ),
        (
            '    ("v8", PROJECT_ROOT / "data" / "calibration" / "station_stacking_v8"),',
            '    ("v8", PROJECT_ROOT / "data" / "calibration" / "station_stacking_v8"),\n'
            '    ("v9", PROJECT_ROOT / "data" / "calibration" / "station_stacking_v9"),',
        ),
        ("V8_DROPPED_FEATURE_COLUMNS", "V9_DROPPED_FEATURE_COLUMNS"),
        ("V8_FEATURE_COLUMNS", "V9_FEATURE_COLUMNS"),
        ("V8 Contract", "V9 Contract"),
        ("V8 Feature Coverage", "V9 Feature Coverage"),
        ("v8_feature_coverage", "v9_feature_coverage"),
        ("## V8", "## V9"),
        ("`feature_version=\\\"v8\\\"`", "`feature_version=\\\"v9\\\"`"),
        ("OPTUNA_TRIALS = 50", "OPTUNA_TRIALS = 30"),
        ("STACK_OPTUNA_TRIALS = 50", "STACK_OPTUNA_TRIALS = 30"),
        ("OPTUNA_STARTUP_TRIALS = 20", "OPTUNA_STARTUP_TRIALS = 15"),
        ("STACK_OPTUNA_STARTUP_TRIALS = 20", "STACK_OPTUNA_STARTUP_TRIALS = 15"),
        (
            "OPTUNA_VERBOSE = True",
            'OPTUNA_VERBOSE = True\nMODEL_VERSION = "station_high_regressor_v9_remaining_warmup"',
        ),
        (
            "from src.calibration.station_stacking import (",
            "from src.export_station_stacking_v2_models import export_station_model_weights\nfrom src.calibration.station_stacking import (",
        ),
        (
            "This version keeps the v7 live-safe GFS/HRRR/NBM contract, adds source-owned remaining-warmup feature engineering, and drops only conservative zero-importance input fields. Artifacts are written to `data/calibration/station_stacking_v8`.",
            "This version keeps the v8 live-safe GFS/HRRR/NBM contract, adds source-owned 10-year calendar-day max-temperature climatology features, and trains base learners on remaining warmup from the observed high-so-far. Artifacts are written to `data/calibration/station_stacking_v9`.",
        ),
        (
            '`feature_version="v9"` keeps the v7 live-safe NBM setup and direct `actual_high_f` target. V8 adds remaining-warmup features and removes only conservative zero-importance model inputs from the feature matrix.',
            '`feature_version="v9"` keeps the v8 remaining-warmup feature set, adds leakage-safe 10-year calendar-day climatology features, and uses `target_mode="remaining_warmup"` so base learners fit `actual_high_f - observed_high_temp_through_as_of_f` before converting back to `predicted_high_f`.',
        ),
        (
            '    feature_version="v9",\n',
            '    feature_version="v9",\n    target_mode="remaining_warmup",\n',
        ),
    ]
    for cell in notebook["cells"]:
        cell["source"] = [_replace_all(line, replacements) for line in cell.get("source", [])]
    _insert_export_cell(notebook, v8)
    return notebook


def _insert_export_cell(notebook: dict, v8) -> None:
    export_cell = v8._cell(
        "code",
        """exported_weights = export_station_model_weights(
    project_root=PROJECT_ROOT,
    station_id=STATION_ID,
    artifact_dir=config.resolved_output_dir(),
    model_version=MODEL_VERSION,
    timing_mode=config.timing_mode,
    providers=tuple(config.providers),
    feature_version=config.effective_feature_version,
    optuna_metric=config.effective_optuna_metric,
    target_mode=config.effective_target_mode,
    source_pipeline="notebooks/experiments/station_stacking_v9",
)

exported_weights.bundle_path, exported_weights.manifest_path
""",
    )
    for index, cell in enumerate(notebook["cells"]):
        source = "".join(cell.get("source", []))
        if "result = run_station_year_split_experiment(config)" in source:
            notebook["cells"].insert(index + 1, export_cell)
            return
    notebook["cells"].append(export_cell)


def _replace_all(text: str, replacements: list[tuple[str, str]]) -> str:
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    for station in TARGET_STATIONS:
        notebook = _notebook(station)
        path = out_dir / f"stacking_{station}_v9.ipynb"
        path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
