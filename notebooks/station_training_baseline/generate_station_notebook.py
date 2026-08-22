from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BASELINE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BASELINE_ROOT.parents[1]
STATION_IDS = ("KDAL", "RJTT", "RKSI", "RKPK")


def _markdown(source: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def _code(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def _load_config(path: str | Path) -> dict[str, Any]:
    from src.calibration.station_baseline import load_station_config

    return load_station_config(path)


def build_notebook(config: dict[str, Any]) -> dict[str, Any]:
    station = config["station_id"]
    config_relative = f"notebooks/station_training_baseline/configs/{station}.json"
    cells = [
        _markdown(
            f"# Station Training Baseline — {station}: {config['station_name']}\n\n"
            "This is the canonical single-model station workflow. It trains one XGBoost point "
            "regressor, a conditional Gaussian residual probability baseline, and four ordinal "
            "research candidates. Blended, shared-slope, and pure ordinal form the canonical "
            "two-of-three ensemble; the native ordinal reference is non-voting. Validation is "
            "chronological, and production artifacts are separately labeled candidates.\n"
        ),
        _code(
            "from pathlib import Path\n"
            "import sys\n\n"
            "PROJECT_ROOT = Path.cwd().resolve()\n"
            "while PROJECT_ROOT != PROJECT_ROOT.parent and not (PROJECT_ROOT / 'pyproject.toml').is_file():\n"
            "    PROJECT_ROOT = PROJECT_ROOT.parent\n"
            "if not (PROJECT_ROOT / 'pyproject.toml').is_file():\n"
            "    raise FileNotFoundError('weather-research project root not found')\n"
            "if str(PROJECT_ROOT) not in sys.path:\n"
            "    sys.path.insert(0, str(PROJECT_ROOT))\n\n"
            "from src.calibration.station_baseline import (\n"
            "    ARCHITECTURE_VERSION,\n"
            "    load_station_config,\n"
            "    run_station_baseline,\n"
            ")\n"
        ),
        _markdown("## Station and chronology contract\n"),
        _code(
            f"CONFIG_PATH = PROJECT_ROOT / {config_relative!r}\n"
            "station_config = load_station_config(CONFIG_PATH)\n"
            "{\n"
            "    'architecture': ARCHITECTURE_VERSION,\n"
            "    'station_id': station_config['station_id'],\n"
            "    'timezone': station_config['timezone'],\n"
            "    'providers': station_config['providers'],\n"
            "    'point_model': 'xgboost',\n"
            "    'optuna_trials': station_config['optuna_trials'],\n"
            "    'optuna_startup_trials': station_config['optuna_startup_trials'],\n"
            "    'probability_benchmark': 'conditional_gaussian_residual',\n"
            "    'ordinal_candidates': [\n"
            "        'native_ordinal_reference',\n"
            "        'blended_ordinal',\n"
            "        'shared_slope_ordinal',\n"
            "        'pure_ordinal',\n"
            "    ],\n"
            "    'ordinal_ensemble': {\n"
            "        'voting_members': ['blended_ordinal', 'shared_slope_ordinal', 'pure_ordinal'],\n"
            "        'required_votes': 2,\n"
            "        'aggregation': 'median_selected_bucket',\n"
            "    },\n"
            "}\n"
        ),
        _markdown(
            "## Train, validate, compare, and export\n\n"
            "This call writes frozen validation artifacts, forward/holdout comparison reports, and "
            "separate live-production candidate artifacts. Persistent Optuna storage resumes until "
            "the configured total of 100 XGBoost trials is complete.\n"
        ),
        _code(
            "run = run_station_baseline(\n"
            "    CONFIG_PATH,\n"
            "    project_root=PROJECT_ROOT,\n"
            "    export_production=True,  # exports an unapproved production candidate\n"
            ")\n"
            "run.point_scoreboard\n"
        ),
        _markdown("## Gaussian, ordinal candidates, and ensemble comparison\n"),
        _code("run.probability_comparison\n"),
        _markdown("## Exported validation and production-candidate artifacts\n"),
        _code("{name: str(path) for name, path in run.artifact_paths.items()}\n"),
        _markdown("## Reports\n"),
        _code("{name: str(path) for name, path in run.report_paths.items()}\n"),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            "station_training_baseline": {
                "architecture_version": "station_training_baseline_xgboost_probability_v2",
                "station_id": station,
                "station_name": config["station_name"],
                "point_model": "xgboost",
                "ensemble_enabled": True,
                "ordinal_required_votes": 2,
                "ordinal_aggregation": "median_selected_bucket",
                "probability_models": [
                    "conditional_gaussian_residual",
                    "native_ordinal_reference",
                    "blended_ordinal",
                    "shared_slope_ordinal",
                    "pure_ordinal",
                    "ordinal_ensemble_median",
                ],
                "optuna_trials": int(config["optuna_trials"]),
                "optuna_startup_trials": int(config["optuna_startup_trials"]),
                "production_export": "candidate_only",
                "production_export_default": True,
                "point_oos_lineage": "nested_chronological_outer_fold",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def generate(config_path: str | Path) -> Path:
    config = _load_config(config_path)
    output = BASELINE_ROOT / config["notebook_path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_notebook(config), indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate station-code baseline notebooks")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if args.all == (args.config is not None):
        parser.error("provide exactly one of --config or --all")
    paths = (
        [BASELINE_ROOT / "configs" / f"{station}.json" for station in STATION_IDS]
        if args.all
        else [args.config]
    )
    for path in paths:
        print(generate(path))


if __name__ == "__main__":
    main()
