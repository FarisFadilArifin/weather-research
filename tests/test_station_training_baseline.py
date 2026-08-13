from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = PROJECT_ROOT / "notebooks" / "station_training_baseline"


def _generator_module():
    source = BASELINE_ROOT / "generate_station_notebook.py"
    spec = importlib.util.spec_from_file_location(
        "station_training_baseline_generator",
        source,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _notebook_source(station_key: str = "KDAL") -> tuple[dict, str]:
    notebook = json.loads(
        (
            BASELINE_ROOT
            / "stations"
            / station_key
            / f"train_{station_key}.ipynb"
        ).read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    return notebook, source


def _source_contract(notebook: dict) -> dict:
    return {
        "nbformat": notebook["nbformat"],
        "nbformat_minor": notebook["nbformat_minor"],
        "station_training_baseline": notebook["metadata"][
            "station_training_baseline"
        ],
        "cells": [
            {
                "cell_type": cell["cell_type"],
                "source": cell.get("source", []),
            }
            for cell in notebook["cells"]
        ],
    }


def test_generated_notebooks_match_configured_sources() -> None:
    generator = _generator_module()
    for station_key in ("KDAL", "Seoul", "Tokyo"):
        config = generator._load_config(
            BASELINE_ROOT / "configs" / f"{station_key}.json"
        )
        generated = generator.build_notebook(config)
        checked_in, _ = _notebook_source(station_key)
        assert _source_contract(generated) == _source_contract(checked_in)


def test_kdal_baseline_is_one_self_contained_station_workflow() -> None:
    _, source = _notebook_source()
    assert "Station Training Baseline — KDAL: Dallas Love Field" in source
    assert 'STATION_ID = "KDAL"' in source
    assert 'training_profile="v20_aligned"' in source
    assert 'target_source="wunderground_only"' in source
    assert 'base_model_methods=("xgboost", "lightgbm", "catboost")' in source
    assert "run_station_year_split_experiment(config)" in source
    assert "POINT_EVALUATION_TRAIN_YEARS = (2021, 2025)" in source
    assert 'POINT_BUCKET_CONTRACT = "polymarket_half_up_2f"' in source
    assert "POINT_MAX_FEATURE_MISSING_FRACTION = 0.03" in source
    assert "Point-model market-bucket hit rate" in source
    assert "point_bucket_metrics(" in source
    assert '"evaluation_status"] = "honest_forward"' in source
    assert '"evaluation_status"] = "exploratory_holdout"' in source
    assert "train_years=POINT_EVALUATION_TRAIN_YEARS" in source
    assert (
        "max_feature_missing_fraction=config.effective_max_feature_missing_fraction"
        in source
    )
    assert 'os.environ.get("STATION_TRAINING_EXPORT_LIVE_MODEL_WEIGHTS", "0") == "1"' in source
    assert "LIVE_POINT_MODEL_VERSION != MODEL_VERSION" in source
    assert "frozen_feature_names=frozen_point_feature_names" in source
    assert '"selection_mode"] == "frozen_evaluation_contract"' in source
    assert 'live_point_manifest["features"]["all"] == list(frozen_point_feature_names)' in source
    assert 'evaluation_point_manifest["training"]["train_end_year"]' in source
    assert 'evaluation_point_manifest["model_contract"]["max_feature_missing_fraction"]' in source
    assert "Ordinal Probabilities Model 2" in source
    assert 'forced_family="ordinal_logistic"' in source
    assert "blend_weights=(1.0,)" in source
    assert "evaluate_probability_holdout(" in source
    assert "export_probability_bundle(" in source
    assert "Required three-arm ordinal challenger export" in source
    assert "run_challenger()" in source
    assert "FROZEN_CANDIDATE_ROLES" in source
    assert 'len(challenger_run["bundle_paths"]) == 3' in source
    assert 'len(challenger_run["manifest_paths"]) == 3' in source
    assert source.index("run_station_year_split_experiment(config)") < source.index(
        "fit_probability_system("
    )
    assert source.index("fit_probability_system(") < source.index(
        "run_challenger()"
    )
    assert "Version Comparison" not in source
    assert "Common-Date Comparison with Existing V11 Settlement" not in source
    assert "regime_gated" not in source
    assert "ElasticNet" not in source


def test_kdal_baseline_freezes_probability_chronology_and_metadata() -> None:
    notebook, source = _notebook_source()
    metadata = notebook["metadata"]["station_training_baseline"]
    assert "development years: `[2023, 2024, 2025]`" in source
    assert "2026 is an exploratory holdout" in source
    assert (
        'set(ordinal_forward_predictions["validation_year"]) == set('
        in source
    )
    assert 'period="holdout_2026"' in source
    assert metadata["station_id"] == "KDAL"
    assert metadata["point_evaluation_train_years"] == [2021, 2025]
    assert metadata["point_bucket_contract"] == "polymarket_half_up_2f"
    assert metadata["point_max_feature_missing_fraction"] == 0.03
    assert (
        metadata["point_live_model_version"]
        == "station_high_regressor_live_kdal_no_peak_stack_2026"
    )
    assert metadata["point_live_export_default"] is False
    assert metadata["probability_model_label"] == "Ordinal Probabilities Model 2"
    assert metadata["probability_family"] == "ordinal_logistic"
    assert metadata["probability_blend_weight"] == 1.0
    assert metadata["probability_feature_profile"] == "common_no_peak"
    assert metadata["probability_feature_count"] == 59
    assert metadata["probability_providers"] == ["gfs", "hrrr", "nbm"]
    assert metadata["probability_development_years"] == [2023, 2024, 2025]
    assert metadata["probability_forward_validation_years"] == [2024, 2025]
    assert metadata["probability_holdout_year"] == 2026
    assert metadata["probability_holdout_status"] == "exploratory"
    assert metadata["ordinal_challenger_enabled"] is True
    assert metadata["ordinal_challenger_version"] == "kdal_ordinal_challenger_v1"
    assert metadata["ordinal_challenger_roles"] == [
        "blended_ordinal",
        "shared_slope_ordinal",
        "pure_ordinal",
    ]
    assert metadata["ordinal_challenger_exports_model_weights"] is True


def test_kdal_baseline_enforces_probability_integrity_before_export() -> None:
    _, source = _notebook_source()
    assert "assert not point_forward_predictions.empty" in source
    assert 'point_forward_predictions["train_through_year"]' in source
    assert "assert not ordinal_holdout_predictions.empty" in source
    assert "assert not ordinal_holdout_metrics.empty" in source
    assert '"offset_probabilities",' in source
    assert '"degree_probabilities",' in source
    assert '"bucket_probabilities",' in source
    assert 'ordinal_manifest["point_bundle_sha256"]' in source
    assert 'ordinal_manifest["artifact_integrity"]["bundle_sha256"]' in source
    assert 'challenger_manifest["point_bundle_sha256"]' in source
    assert 'challenger_manifest["artifact_integrity"]["bundle_sha256"]' in source
    assert "assert challenger_bundle_path.is_file()" in source


def test_seoul_and_tokyo_follow_station_baseline_contract() -> None:
    expected = {
        "Seoul": {
            "station_id": "RKSI",
            "city_id": "seoul",
            "timezone": "Asia/Seoul",
            "model_version": "station_high_regressor_baseline_seoul_no_peak_stack",
            "evaluation_years": [2022, 2025],
            "live_model_version": "station_high_regressor_live_seoul_no_peak_stack_2026",
            "optuna_storage_name": "RKSI_optuna.sqlite3",
        },
        "Tokyo": {
            "station_id": "RJTT",
            "city_id": "tokyo",
            "timezone": "Asia/Tokyo",
            "model_version": "station_high_regressor_baseline_tokyo_no_peak_stack",
            "evaluation_years": [2022, 2025],
            "live_model_version": "station_high_regressor_live_tokyo_no_peak_stack_2026",
            "optuna_storage_name": "RJTT_optuna_no_fullday_high.sqlite3",
        },
    }
    for station_key, contract in expected.items():
        notebook, source = _notebook_source(station_key)
        metadata = notebook["metadata"]["station_training_baseline"]
        assert f'CITY_ID = "{contract["city_id"]}"' in source
        assert f'STATION_ID = "{contract["station_id"]}"' in source
        assert f'TIMEZONE = "{contract["timezone"]}"' in source
        assert (
            'PROVIDERS = ("gfs", "gefs", "jma_msm")'
            in source
        )
        assert "asia_same_day_11am_live_safe" in source
        assert 'PROBABILITY_FEATURE_PROFILE = "asia_no_peak"' in source
        assert (
            "PROBABILITY_DEVELOPMENT_YEARS = (2024, 2025)"
            in source
        )
        assert "PROBABILITY_FORWARD_VALIDATION_YEARS = (2025,)" in source
        assert "run_station_year_split_experiment(config)" in source
        assert "optuna_verbose=True" in source
        assert (
            f'optuna_storage_path=OUTPUT_DIR / "{contract["optuna_storage_name"]}"'
            in source
        )
        if station_key in {"Seoul", "Tokyo"}:
            assert "fit_celsius_probability_system(" in source
            assert 'PROBABILITY_TARGET = "celsius_market_1c"' in source
            assert 'PROBABILITY_OUTPUT_SUBDIR = "celsius_market_probability"' in source
            assert "offset_c = actual_bucket_c - point_bucket_c" in source
            assert "2026 remains exploratory" in source
            assert "ordinal_probability" not in source
            assert "export_celsius_probability_bundle(" in source
        else:
            assert "fit_probability_system(" in source
            assert "export_probability_bundle(" in source
        assert "export_station_model_weights(" in source
        assert "POINT_EVALUATION_TRAIN_YEARS = (2022, 2025)" in source
        assert 'POINT_BUCKET_CONTRACT = "polymarket_half_up_1c"' in source
        assert "POINT_MAX_FEATURE_MISSING_FRACTION = 0.03" in source
        assert "Point-model market-bucket hit rate" in source
        assert "point_forward_bucket_metrics" in source
        assert "point_holdout_bucket_metrics" in source
        assert "train_years=POINT_EVALUATION_TRAIN_YEARS" in source
        assert (
            "max_feature_missing_fraction=config.effective_max_feature_missing_fraction"
            in source
        )
        assert 'os.environ.get("STATION_TRAINING_EXPORT_LIVE_MODEL_WEIGHTS", "0") == "1"' in source
        assert f'LIVE_POINT_MODEL_VERSION = "{contract["live_model_version"]}"' in source
        assert "train_years=None" in source
        assert "frozen_feature_names=frozen_point_feature_names" in source
        assert '"selection_mode"] == "frozen_evaluation_contract"' in source
        assert 'evaluation_point_manifest["training"]["train_end_year"]' in source
        assert 'evaluation_point_manifest["model_contract"]["max_feature_missing_fraction"]' in source
        assert "run_challenger()" not in source
        assert "hrrr" not in source.lower()
        assert "nbm" not in source.lower()
        probability_fit_call = (
            "fit_celsius_probability_system("
            if station_key in {"Seoul", "Tokyo"}
            else "fit_probability_system("
        )
        assert (
            source.index("run_station_year_split_experiment(config)")
            < source.index(probability_fit_call)
        )
        assert metadata["station_id"] == contract["station_id"]
        assert metadata["point_model_version"] == contract["model_version"]
        assert metadata["point_evaluation_train_years"] == contract["evaluation_years"]
        assert metadata["point_bucket_contract"] == "polymarket_half_up_1c"
        assert metadata["point_max_feature_missing_fraction"] == 0.03
        assert metadata["point_live_model_version"] == contract["live_model_version"]
        assert metadata["point_live_export_default"] is False
        assert metadata["probability_feature_profile"] == "asia_no_peak"
        assert metadata["probability_feature_count"] == 59
        assert metadata["probability_providers"] == [
            "gfs",
            "gefs",
            "jma_msm",
        ]
        assert metadata["probability_development_years"] == [2024, 2025]
        assert metadata["probability_forward_validation_years"] == [2025]
        assert metadata["probability_holdout_year"] == 2026
        assert metadata["probability_target"] == "celsius_market_1c"
        assert metadata["probability_output_subdir"] == "celsius_market_probability"
        assert metadata["ordinal_challenger_enabled"] is False
        assert metadata["ordinal_challenger_roles"] == []
        assert metadata["ordinal_challenger_exports_model_weights"] is False


def test_seoul_and_tokyo_celsius_reporting_convert_absolute_temperatures_only() -> None:
    for station_key in ("Seoul", "Tokyo"):
        _, source = _notebook_source(station_key)
        reporting_start = source.index("## Celsius reporting and export")
        reporting_end = source.index("if EXPORT_MODEL_WEIGHTS:", reporting_start)
        reporting = source[reporting_start:reporting_end]
        assert 'for column in ("actual_high_f", "predicted_high_f"):' in reporting
        assert (
            'pd.to_numeric(celsius_predictions[column], errors="coerce") - 32.0'
            in reporting
        )
        assert 'celsius_predictions["error_c"]' in reporting
        assert (
            'pd.to_numeric(celsius_predictions["error_f"], errors="coerce") * 5.0 / 9.0'
            in reporting
        )


def test_regeneration_preserves_existing_notebook_outputs_and_metadata() -> None:
    generator = _generator_module()
    generated = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": ["same\n"]},
            {
                "cell_type": "code",
                "metadata": {},
                "source": ["same_code()\n"],
                "execution_count": None,
                "outputs": [],
            },
        ],
        "metadata": {"station_training_baseline": {"station_id": "RJTT"}},
    }
    existing = {
        "cells": [
            {"cell_type": "markdown", "metadata": {"tag": "note"}, "source": ["same\n"]},
            {
                "cell_type": "code",
                "metadata": {"tag": "run"},
                "source": ["same_code()\n"],
                "execution_count": 4,
                "outputs": [{"output_type": "stream", "name": "stdout", "text": "saved\n"}],
            },
        ],
        "metadata": {"language_info": {"name": "python"}, "station_training_baseline": {"old": True}},
    }
    merged = generator._preserve_existing_notebook_state(generated, existing)
    assert merged["cells"][0]["source"] == ["same\n"]
    assert merged["cells"][0]["metadata"] == {"tag": "note"}
    assert merged["cells"][1]["outputs"] == existing["cells"][1]["outputs"]
    assert merged["cells"][1]["execution_count"] == 4
    assert merged["metadata"]["language_info"] == {"name": "python"}
    assert merged["metadata"]["station_training_baseline"] == {"station_id": "RJTT"}


def test_regeneration_allows_new_cells_without_reusing_stale_state() -> None:
    generator = _generator_module()
    generated = {
        "cells": [
            {
                "cell_type": "code",
                "metadata": {},
                "source": ["same_code()\n"],
                "execution_count": None,
                "outputs": [],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": ["new_code()\n"],
                "execution_count": None,
                "outputs": [],
            },
        ],
        "metadata": {"station_training_baseline": {"station_id": "KDAL"}},
    }
    existing = {
        "cells": [
            {
                "cell_type": "code",
                "metadata": {"tag": "run"},
                "source": ["same_code()\n"],
                "execution_count": 7,
                "outputs": [
                    {"output_type": "stream", "name": "stdout", "text": "saved\n"}
                ],
            }
        ],
        "metadata": {"station_training_baseline": {"old": True}},
    }

    merged = generator._preserve_existing_notebook_state(generated, existing)

    assert merged["cells"][0]["execution_count"] == 7
    assert merged["cells"][0]["metadata"] == {"tag": "run"}
    assert merged["cells"][1]["execution_count"] is None
    assert merged["cells"][1]["outputs"] == []
