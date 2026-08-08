from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def test_v20_hko_no_peak_notebook_matches_gfs_only_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook_root = (
        root / "notebooks" / "experiments" / "station_stacking_v20_hko_no_peak"
    )
    generator_path = notebook_root / "generate_station_notebook.py"
    generator = generator_path.read_text(encoding="utf-8")
    notebook = json.loads((notebook_root / "stacking_HKO_v20_no_peak.ipynb").read_text(encoding="utf-8"))
    source = "".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] == 5
    outputs = [
        output
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        for output in cell.get("outputs", [])
    ]
    assert not [
        output for output in outputs if output.get("output_type") == "error"
    ]
    assert 'STATION_ID = "HKO"' in source
    assert 'PROVIDERS = ("gfs",)' in source
    assert 'FEATURE_VERSION = "v20_hko_gfs_no_peak"' in source
    assert 'TARGET_SOURCE = "hko_daily_max"' in source
    assert "V20_EXPANDING_FOLDS" in source
    assert "config.observation_target_same_station is True" in source
    assert "config.observation_source == OBSERVATION_SOURCE_CONTRACT" in source
    assert "HKO Headquarters 1-minute temperature" in source
    assert "DATA.GOV.HK archive" in source
    assert "VHHH observations" not in source
    assert "max_feature_missing_fraction=0.03" in source
    assert "OPTUNA_TRIALS = 30" in source
    assert "EXPORT_MODEL_WEIGHTS = True" in source
    assert "run_hong_kong_year_split_experiment(" in source
    assert "export_station_model_weights(" in source
    assert "add_celsius_prediction_columns" in source
    assert 'bucket_contract="floor_1c"' in source
    assert "observation_target_same_station=True" in source
    assert "observation_source=OBSERVATION_SOURCE_CONTRACT" in source
    assert '"bucket_width_c"' in source
    assert "floor_integer_celsius" in source
    assert "Existing Fahrenheit Bracket Diagnostics" not in source
    assert "polymarket_half_up_2f" not in source
    assert "GFS Raw Uplift" in source
    assert "Warm/Cool 11 AM Forecast Delta" in source
    assert "V20_PEAK_TIMING_RAW_FEATURE_COLUMNS" not in source
    assert "V20_ENGINEERED_FEATURE_COLUMNS" not in source
    assert "v20_peak_timing" not in source
    assert 'source_pipeline="notebooks/experiments/station_stacking_v20_hko_no_peak"' in source
    assert "do not execute" not in generator.lower()


def test_v20_hko_no_peak_generator_matches_committed_notebook() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook_root = (
        root / "notebooks" / "experiments" / "station_stacking_v20_hko_no_peak"
    )
    generator_path = notebook_root / "generate_station_notebook.py"
    spec = importlib.util.spec_from_file_location("hko_notebook_generator", generator_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    generated = module._notebook()
    stored = json.loads((notebook_root / "stacking_HKO_v20_no_peak.ipynb").read_text(encoding="utf-8"))

    def source_contract(notebook: dict) -> dict:
        return {
            "nbformat": notebook["nbformat"],
            "nbformat_minor": notebook["nbformat_minor"],
            "cells": [
                {
                    "cell_type": cell["cell_type"],
                    "source": cell.get("source", []),
                }
                for cell in notebook["cells"]
            ],
        }

    assert source_contract(generated) == source_contract(stored)
    assert all(
        cell.get("execution_count") is None
        for cell in generated["cells"]
        if cell["cell_type"] == "code"
    )
    assert all(
        not cell.get("outputs")
        for cell in generated["cells"]
        if cell["cell_type"] == "code"
    )
