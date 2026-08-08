from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_ROOT = Path(__file__).resolve().parent


def _markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def _code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def _notebook(city_id: str) -> dict:
    city = city_id.lower()
    label = city.title()
    station = {"tokyo": "RJTT", "seoul": "RKSI"}[city]
    timezone = {"tokyo": "Asia/Tokyo", "seoul": "Asia/Seoul"}[city]
    cells = [
        _markdown(
            f"# {label} Station Stacking V20 No-Peak\n\n"
            f"{label} (`{station}`) adaptation of the KDAL V20-aligned station-stacking workflow. "
            "The notebook uses the existing Asia 11 AM parquet contract, Fahrenheit-native modeling, "
            "Wunderground-only settlement highs, and an expanding 2022–2025 validation design.\n"
        ),
        _code(
            "from pathlib import Path\n"
            "import sys\n"
            "import warnings\n"
            "\n"
            "warnings.filterwarnings(\"ignore\", message=\"IProgress not found.*\")\n"
            "warnings.filterwarnings(\"ignore\", message=\"Skipping features without any observed values.*\")\n"
            "\n"
            "PROJECT_ROOT = Path.cwd().resolve()\n"
            "while not (PROJECT_ROOT / \"src\" / \"calibration\" / \"asia_station_stacking.py\").exists():\n"
            "    if PROJECT_ROOT.parent == PROJECT_ROOT:\n"
            "        raise RuntimeError(\"Could not find project root containing src/calibration/asia_station_stacking.py\")\n"
            "    PROJECT_ROOT = PROJECT_ROOT.parent\n"
            "if str(PROJECT_ROOT) not in sys.path:\n"
            "    sys.path.insert(0, str(PROJECT_ROOT))\n"
            "\n"
            f"CITY_ID = \"{city}\"\n"
            f"CITY_LABEL = \"{label}\"\n"
            f"STATION_ID = \"{station}\"\n"
            f"TIMEZONE = \"{timezone}\"\n"
            "DATA_ROOT = PROJECT_ROOT / \"data\" / \"calibration\" / \"asia_11am\"\n"
            "OUTPUT_DIR = DATA_ROOT / \"models\" / f\"v20_{CITY_ID}_no_peak\"\n"
            "PROVIDERS = (\"gfs\", \"gefs\", \"jma_msm\")\n"
            "TIMING_MODE = \"asia_same_day_11am_live_safe\"\n"
            "FEATURE_VERSION = \"v20_asia_no_peak\"\n"
            "TRAINING_PROFILE = \"v20_aligned\"\n"
            "TARGET_SOURCE = \"wunderground_only\"\n"
            "TARGET_MODE = \"remaining_warmup\"\n"
            "OPTUNA_TRIALS = 30\n"
            "STACK_OPTUNA_TRIALS = 30\n"
            "OPTUNA_STARTUP_TRIALS = 15\n"
            "STACK_OPTUNA_STARTUP_TRIALS = 15\n"
            "FAST_MODE = False\n"
            "EXPORT_MODEL_WEIGHTS = True\n"
            "MODEL_VERSION = f\"station_high_regressor_v20_{CITY_ID}_no_peak_stack\"\n"
            "PROJECT_ROOT\n"
        ),
        _code(
            "import numpy as np\n"
            "import pandas as pd\n"
            "\n"
            "from src.calibration.asia_station_stacking import (\n"
            "    ASIA_PROVIDERS,\n"
            "    ASIA_TEST_YEAR,\n"
            "    ASIA_TIMING_MODE,\n"
            "    asia_expanding_folds,\n"
            "    build_asia_station_wide_dataset,\n"
            "    provider_readiness,\n"
            ")\n"
            "from src.calibration.station_stacking import (\n"
            "    StationStackingConfig,\n"
            "    V20_ASIA_NO_PEAK_FEATURE_VERSION,\n"
            "    missing_model_dependencies,\n"
            "    run_station_year_split_experiment,\n"
            ")\n"
            "from src.export_station_stacking_v2_models import export_station_model_weights\n"
        ),
        _markdown(
            "## City contract\n\n"
            "- Existing Asia parquet data rooted at `data/calibration/asia_11am`\n"
            "- Local 11 AM live-safe observation cutoff\n"
            "- GFS, GEFS, and JMA MSM forecast inputs\n"
            "- Wunderground-only daily settlement high target\n"
            "- Fahrenheit-native model values with Celsius reporting\n"
        ),
        _code(
            "fold_spec = pd.DataFrame(\n"
            "    [\n"
            "        {\n"
            "            \"fold\": fold.name,\n"
            "            \"train_start_year\": fold.train_start_year,\n"
            "            \"train_end_year\": fold.train_end_year,\n"
            "            \"validation_year\": fold.validation_year,\n"
            "            \"validation_weight\": 1.0,\n"
            "        }\n"
            "        for fold in asia_expanding_folds()\n"
            "    ]\n"
            ")\n"
            "fold_spec\n"
        ),
        _markdown("## Provider readiness\n"),
        _code(
            "readiness = provider_readiness(DATA_ROOT, CITY_ID, providers=PROVIDERS)\n"
            "readiness\n"
        ),
        _markdown("## Build the live-safe modeling frame\n"),
        _code(
            "features = build_asia_station_wide_dataset(\n"
            "    DATA_ROOT,\n"
            "    CITY_ID,\n"
            "    feature_version=FEATURE_VERSION,\n"
            "    providers=PROVIDERS,\n"
            ")\n"
            "features[[\n"
            "    \"contract_date\",\n"
            "    \"actual_high_f\",\n"
            "    \"observed_high_temp_through_as_of_f\",\n"
            "    \"gfs_high_f\",\n"
            "    \"gefs_high_f\",\n"
            "    \"jma_msm_high_f\",\n"
            "    \"strict_quality_ok\",\n"
            "]].head()\n"
        ),
        _code(
            "feature_coverage = (\n"
            "    features[[\"actual_high_f\", \"observed_high_temp_through_as_of_f\", *[f\"{p}_high_f\" for p in PROVIDERS]]]\n"
            "    .notna()\n"
            "    .mean()\n"
            "    .rename(\"non_null_fraction\")\n"
            "    .to_frame()\n"
            ")\n"
            "feature_coverage\n"
        ),
        _markdown("## Train and score\n"),
        _code(
            "missing_packages = missing_model_dependencies((\"xgboost\", \"lightgbm\", \"catboost\", \"optuna\"))\n"
            "if missing_packages:\n"
            "    raise ImportError(\n"
            "        \"Missing station-stacking ML packages: \"\n"
            "        + \", \".join(missing_packages)\n"
            "        + \". Install them with: python -m pip install -r requirements.txt\"\n"
            "    )\n"
            "\n"
            "config = StationStackingConfig(\n"
            "    station_id=STATION_ID,\n"
            "    project_root=PROJECT_ROOT,\n"
            "    timing_mode=TIMING_MODE,\n"
            "    providers=PROVIDERS,\n"
            "    fast_mode=FAST_MODE,\n"
            "    optuna_trials=OPTUNA_TRIALS,\n"
            "    stack_optuna_trials=STACK_OPTUNA_TRIALS,\n"
            "    optuna_startup_trials=OPTUNA_STARTUP_TRIALS,\n"
            "    stack_optuna_startup_trials=STACK_OPTUNA_STARTUP_TRIALS,\n"
            "    optuna_metric=\"mae_f\",\n"
            "    feature_version=FEATURE_VERSION,\n"
            "    training_profile=TRAINING_PROFILE,\n"
            "    target_mode=TARGET_MODE,\n"
            "    target_source=TARGET_SOURCE,\n"
            "    max_feature_missing_fraction=0.03,\n"
            "    base_model_methods=(\"xgboost\", \"lightgbm\", \"catboost\"),\n"
            "    stack_enabled=True,\n"
            "    hyperparameter_space=\"wide\",\n"
            "    year_split_folds=asia_expanding_folds(),\n"
            "    year_split_validation_weights={2023: 1.0, 2024: 1.0, 2025: 1.0},\n"
            "    year_split_test_train_years=(2022, 2025),\n"
            "    year_split_test_year=ASIA_TEST_YEAR,\n"
            "    output_dir=OUTPUT_DIR,\n"
            "    prebuilt_features=features,\n"
            ")\n"
            "config.resolved_optuna_storage_path()\n"
        ),
        _code(
            "result = run_station_year_split_experiment(config)\n"
            "result.scoreboard\n"
        ),
        _markdown("## Celsius reporting and export\n"),
        _code(
            "celsius_predictions = result.test_predictions.copy()\n"
            "for column in (\"actual_high_f\", \"predicted_high_f\", \"error_f\"):\n"
            "    if column in celsius_predictions:\n"
            "        celsius_predictions[column.replace(\"_f\", \"_c\")] = pd.to_numeric(celsius_predictions[column], errors=\"coerce\") * 5.0 / 9.0\n"
            "celsius_predictions.head()\n"
        ),
        _code(
            "if EXPORT_MODEL_WEIGHTS:\n"
            "    exported_weights = export_station_model_weights(\n"
            "        project_root=PROJECT_ROOT,\n"
            "        station_id=STATION_ID,\n"
            "        city_id=CITY_ID,\n"
            "        artifact_dir=config.resolved_output_dir(),\n"
            "        model_version=MODEL_VERSION,\n"
            "        timing_mode=config.timing_mode,\n"
            "        providers=tuple(config.providers),\n"
            "        feature_version=config.effective_feature_version,\n"
            "        training_profile=config.effective_training_profile,\n"
            "        optuna_metric=config.effective_optuna_metric,\n"
            "        target_mode=config.effective_target_mode,\n"
            "        target_source=config.effective_target_source,\n"
            "        base_model_methods=tuple(config.effective_base_model_methods),\n"
            "        stack_enabled=config.stack_enabled,\n"
            "        source_pipeline=f\"notebooks/experiments/station_stacking_v20_asia_no_peak/{CITY_ID}\",\n"
            "    )\n"
            "    exported_weights.bundle_path, exported_weights.manifest_path\n"
            "else:\n"
            "    print(\"Model export disabled for this notebook.\")\n"
        ),
        _code(
            "result.output_paths\n"
        ),
    ]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def main() -> None:
    NOTEBOOK_ROOT.mkdir(parents=True, exist_ok=True)
    for city in ("tokyo", "seoul"):
        path = NOTEBOOK_ROOT / f"stacking_{city.title()}_v20_no_peak.ipynb"
        path.write_text(json.dumps(_notebook(city), indent=2) + "\n", encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
