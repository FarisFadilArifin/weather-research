from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.calibration.station_baseline as station_baseline
import src.calibration.station_probability_models as probability_models
from src.calibration.station_baseline import (
    ARCHITECTURE_VERSION,
    DIRECT_NBM_ENV,
    build_station_features,
    load_station_config,
    point_training_config,
)
from src.calibration.station_probability_models import (
    MODEL_FEATURES,
    ORDINAL_COMPACT_FEATURES,
    ORDINAL_MEMBER_ROLES,
    build_probability_frame,
    export_ordinal_ensemble_manifest,
    fit_gaussian,
    fit_ordinal,
    fit_shared_slope_ordinal,
    ordinal_ensemble_predictions,
    probability_metrics,
    probability_predictions,
    round_half_up,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = PROJECT_ROOT / "notebooks" / "station_training_baseline"
STATIONS = ("KDAL", "RJTT", "RKSI", "RKPK")


def _generator_module():
    source = BASELINE_ROOT / "generate_station_notebook.py"
    spec = importlib.util.spec_from_file_location("station_training_baseline_generator", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_contract(notebook: dict) -> dict:
    return {
        "metadata": notebook["metadata"]["station_training_baseline"],
        "cells": [
            {"cell_type": cell["cell_type"], "source": cell.get("source", [])}
            for cell in notebook["cells"]
        ],
    }


def test_station_code_configs_define_the_replacement_architecture() -> None:
    expected = {
        "KDAL": ("F", 2, "us_station"),
        "RJTT": ("C", 1, "asia_11am"),
        "RKSI": ("C", 1, "asia_11am"),
        "RKPK": ("C", 1, "asia_11am"),
    }
    for station, (unit, width, builder) in expected.items():
        config = load_station_config(BASELINE_ROOT / "configs" / f"{station}.json")
        assert config["notebook_path"] == f"stations/{station}/train_{station}.ipynb"
        assert config["feature_builder"] == builder
        assert config["probability_unit"] == unit
        assert config["market_bucket_width"] == width
        assert config["optuna_trials"] == 100
        assert config["optuna_startup_trials"] == 40
        assert "xgboost" in config["point_evaluation_model_version"]
        assert "gaussian_residual" in config["gaussian_evaluation_model_version"]
        assert set(config["ordinal_candidate_model_versions"]) == {
            "native_ordinal_reference",
            *ORDINAL_MEMBER_ROLES,
        }
        for role, versions in config["ordinal_candidate_model_versions"].items():
            assert set(versions) == {"evaluation", "production"}
            assert role in versions["evaluation"]
            assert role in versions["production"]
    assert not (BASELINE_ROOT / "configs" / "Tokyo.json").exists()
    assert not (BASELINE_ROOT / "configs" / "Seoul.json").exists()


def test_generated_notebooks_match_the_generator() -> None:
    generator = _generator_module()
    for station in STATIONS:
        config = load_station_config(BASELINE_ROOT / "configs" / f"{station}.json")
        generated = generator.build_notebook(config)
        checked_in = json.loads(
            (BASELINE_ROOT / "stations" / station / f"train_{station}.ipynb").read_text(encoding="utf-8")
        )
        assert _source_contract(generated) == _source_contract(checked_in)


def test_notebooks_use_one_xgboost_and_four_ordinal_candidates() -> None:
    for station in STATIONS:
        notebook = json.loads(
            (BASELINE_ROOT / "stations" / station / f"train_{station}.ipynb").read_text(encoding="utf-8")
        )
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        metadata = notebook["metadata"]["station_training_baseline"]
        assert metadata["architecture_version"] == ARCHITECTURE_VERSION
        assert "run_station_baseline(" in source
        assert "conditional_gaussian_residual" in source
        assert "blended_ordinal" in source
        assert "shared_slope_ordinal" in source
        assert "pure_ordinal" in source
        assert metadata["point_model"] == "xgboost"
        assert metadata["ensemble_enabled"] is True
        assert metadata["ordinal_required_votes"] == 2
        assert metadata["ordinal_aggregation"] == "median_selected_bucket"
        assert set(metadata["probability_models"]) == {
            "conditional_gaussian_residual",
            "native_ordinal_reference",
            "blended_ordinal",
            "shared_slope_ordinal",
            "pure_ordinal",
            "ordinal_ensemble_median",
        }
        assert metadata["optuna_trials"] == 100
        assert metadata["optuna_startup_trials"] == 40
        assert "lightgbm" not in source.lower()
        assert "catboost" not in source.lower()
        assert "ridge stack" not in source.lower()


def test_point_config_is_single_xgboost_with_tpe_startup_contract() -> None:
    config = load_station_config(BASELINE_ROOT / "configs" / "RJTT.json")
    features = pd.DataFrame({"contract_date": ["2025-01-01"], "actual_high_f": [50.0]})
    point = point_training_config(config, PROJECT_ROOT, features)
    assert point.effective_base_model_methods == ("xgboost",)
    assert point.stack_enabled is False
    assert point.effective_optuna_trials == 100
    assert point.effective_optuna_startup_trials == 40


def test_us_baseline_enables_direct_nbm_only_while_building(monkeypatch) -> None:
    observed_env: list[str | None] = []

    def fake_build(*args, **kwargs):
        observed_env.append(station_baseline.os.environ.get(DIRECT_NBM_ENV))
        return pd.DataFrame(
            {
                "gfs_high_f": [70.0],
                "hrrr_high_f": [69.0],
                "nbm_high_f": [71.0],
            }
        )

    monkeypatch.delenv(DIRECT_NBM_ENV, raising=False)
    monkeypatch.setattr(station_baseline, "build_station_wide_dataset", fake_build)
    frame = build_station_features(
        {
            "station_id": "KDAL",
            "feature_builder": "us_station",
            "providers": ["gfs", "hrrr", "nbm"],
            "timing_mode": "same_day_11am_live_safe",
            "feature_version": "v11_settlement_fix_temp",
            "target_source": "wunderground_only",
        },
        PROJECT_ROOT,
    )

    assert len(frame) == 1
    assert observed_env == ["1"]
    assert DIRECT_NBM_ENV not in station_baseline.os.environ


def _synthetic_frames(rows: int = 260) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(7)
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    point = 70.0 + 12.0 * np.sin(np.arange(rows) * 2.0 * np.pi / 365.25)
    actual = point + rng.normal(0.15, 1.6, rows)
    predictions = pd.DataFrame(
        {"contract_date": dates, "actual_high_f": actual, "predicted_high_f": point}
    )
    features = pd.DataFrame(
        {
            "contract_date": dates,
            "actual_high_f": actual,
            "gfs_high_f": point + rng.normal(0.0, 1.0, rows),
            "hrrr_high_f": point + rng.normal(0.0, 1.0, rows),
            "nbm_high_f": point + rng.normal(0.0, 1.0, rows),
            "observed_temp_at_as_of_f": point - 3.0,
            "observed_high_temp_through_as_of_f": point - 1.0,
            "observed_as_of_age_minutes": 5.0,
            "day_of_year_sin": np.sin(np.arange(rows) * 2.0 * np.pi / 365.25),
            "day_of_year_cos": np.cos(np.arange(rows) * 2.0 * np.pi / 365.25),
        }
    )
    return features, predictions


def test_gaussian_and_ordinal_models_produce_normalized_market_probabilities() -> None:
    features, predictions = _synthetic_frames()
    frame = build_probability_frame(
        features,
        predictions,
        providers=("gfs", "hrrr", "nbm"),
        unit="F",
        bucket_width=2,
    )
    feature_names = list(frame.attrs["feature_names"])
    assert set(MODEL_FEATURES).issubset(feature_names)
    gaussian = fit_gaussian(frame, feature_names, alpha=10.0, scale_multiplier=1.0)
    ordinal = fit_ordinal(
        frame,
        feature_names,
        tail=4,
        c=0.3,
        class_weight=None,
        temperature=1.0,
    )
    combined = []
    for family, state in (("gaussian", gaussian), ("ordinal", ordinal)):
        output = probability_predictions(
            frame.tail(20),
            family=family,
            state=state,
            unit="F",
            bucket_width=2,
            period="test",
        )
        assert len(output) == 20
        for raw in output["market_bucket_probabilities"]:
            assert np.isclose(sum(json.loads(raw).values()), 1.0, atol=1e-10)
        combined.append(output)
    metrics = probability_metrics(pd.concat(combined, ignore_index=True))
    assert metrics["count"] == 40
    assert np.isfinite(metrics["market_log_loss"])
    assert np.isfinite(metrics["market_brier"])


def test_three_member_ordinal_ensemble_produces_votes_and_normalized_probabilities() -> None:
    features, predictions = _synthetic_frames()
    frame = build_probability_frame(
        features,
        predictions,
        providers=("gfs", "hrrr", "nbm"),
        unit="F",
        bucket_width=2,
    )
    full_features = list(frame.attrs["feature_names"])
    compact_features = list(ORDINAL_COMPACT_FEATURES)
    pure = fit_ordinal(
        frame,
        full_features,
        tail=4,
        c=0.3,
        class_weight=None,
        temperature=1.0,
    )
    pure["decision_thresholds"] = {
        "minimum_top_probability": 0.0,
            "minimum_top_two_margin": -1.0,
    }
    blended = {
        **pure,
        "candidate_role": "blended_ordinal",
        "family": "blended_cumulative_ordinal_logistic",
        "blend_weight": 1.0,
    }
    shared = fit_shared_slope_ordinal(
        frame,
        compact_features,
        tail=4,
        c=0.3,
        class_weight=None,
        temperature=1.0,
    )
    shared["decision_thresholds"] = {
        "minimum_top_probability": 0.0,
            "minimum_top_two_margin": -1.0,
    }
    members, ensemble = ordinal_ensemble_predictions(
        frame.tail(20),
        {
            "blended_ordinal": blended,
            "shared_slope_ordinal": shared,
            "pure_ordinal": pure,
        },
        unit="F",
        bucket_width=2,
        period="test",
    )
    assert set(members["family"]) == set(ORDINAL_MEMBER_ROLES)
    assert ensemble["ordinal_approved"].all()
    assert ensemble["ordinal_votes"].eq(3).all()
    for raw in ensemble["market_bucket_probabilities"]:
        assert np.isclose(sum(json.loads(raw).values()), 1.0, atol=1e-10)


def test_four_candidate_fit_keeps_native_reference_outside_voting_members(monkeypatch) -> None:
    monkeypatch.setattr(probability_models, "ORDINAL_C_GRID", (0.3,))
    monkeypatch.setattr(probability_models, "ORDINAL_CLASS_WEIGHTS", (None,))
    monkeypatch.setattr(probability_models, "ORDINAL_TEMPERATURE_GRID", (1.0,))
    monkeypatch.setattr(probability_models, "ORDINAL_BLEND_WEIGHT_GRID", (0.75,))
    monkeypatch.setattr(probability_models, "ORDINAL_PRIOR_STRENGTH_GRID", (30.0,))
    features, predictions = _synthetic_frames()
    frame = build_probability_frame(
        features,
        predictions,
        providers=("gfs", "hrrr", "nbm"),
        unit="F",
        bucket_width=2,
    )
    states, tuning = probability_models.fit_ordinal_candidates(
        frame,
        tail=4,
        unit="F",
        bucket_width=2,
    )
    assert set(states) == {"native_ordinal_reference", *ORDINAL_MEMBER_ROLES}
    assert states["native_ordinal_reference"]["candidate_role"] == "native_ordinal_reference"
    assert set(ORDINAL_MEMBER_ROLES).isdisjoint({"native_ordinal_reference"})
    assert set(tuning["candidate_role"].dropna()) == {
        "native_ordinal_reference",
        *ORDINAL_MEMBER_ROLES,
    }


def test_ensemble_manifest_marks_only_three_candidates_as_voters(tmp_path: Path) -> None:
    features, predictions = _synthetic_frames(80)
    frame = build_probability_frame(
        features,
        predictions,
        providers=("gfs", "hrrr", "nbm"),
        unit="F",
        bucket_width=2,
    )
    full_features = list(frame.attrs["feature_names"])
    native_features = [
        "point_prediction_native",
        "point_rounding_remainder_native",
        "point_distance_to_round_boundary_native",
    ]
    pure = fit_ordinal(
        frame, full_features, tail=1, c=0.3, class_weight=None, temperature=1.0
    )
    native = fit_ordinal(
        frame, native_features, tail=1, c=1.0, class_weight=None, temperature=1.0
    )
    native.update(
        {
            "family": "native_cumulative_ordinal_logistic",
            "candidate_role": "native_ordinal_reference",
            "feature_profile": "native_minimal_reference",
            "reference_contract": "single_xgboost_native_unit_ordinal_reference_v2",
        }
    )
    blended = {
        **pure,
        "family": "blended_cumulative_ordinal_logistic",
        "candidate_role": "blended_ordinal",
        "empirical_state": probability_models._fit_empirical_state(frame, 1),
        "empirical_prior_strength": 30.0,
        "blend_weight": 0.75,
    }
    shared = fit_shared_slope_ordinal(
        frame,
        list(ORDINAL_COMPACT_FEATURES),
        tail=1,
        c=0.3,
        class_weight=None,
        temperature=1.0,
    )
    states = {
        "native_ordinal_reference": native,
        "blended_ordinal": blended,
        "shared_slope_ordinal": shared,
        "pure_ordinal": pure,
    }
    for role, state in states.items():
        probabilities = probability_models.predict_ordinal(state, frame.tail(4))
        assert probabilities.shape == (4, 3), role
        assert np.isfinite(probabilities).all(), role
        assert np.allclose(probabilities.sum(axis=1), 1.0), role
        rows = probability_predictions(
            frame.tail(4),
            family=role,
            state=state,
            unit="F",
            bucket_width=2,
            period="test",
        )
        assert len(rows) == 4
        assert rows["market_bucket_probabilities"].notna().all()

    artifacts = {
        role: probability_models.export_probability_artifact(
            state,
            tmp_path / role,
            artifact_type=probability_models.ORDINAL_ARTIFACT_TYPE,
            station_id="KDAL",
            model_version=f"test_{role}",
            point_model_version="test_xgboost",
            point_bundle_sha256="a" * 64,
            unit="F",
            bucket_width=2,
            training_frame=frame,
            validation_metrics=[],
            source_identity={"git_dirty": True},
            release_role="frozen_evaluation",
        )
        for role, state in states.items()
    }
    output = export_ordinal_ensemble_manifest(
        tmp_path / "ordinal_ensemble" / "evaluation_manifest.json",
        station_id="KDAL",
        point_model_version="test_xgboost",
        point_bundle_sha256="a" * 64,
        unit="F",
        bucket_width=2,
        member_artifacts=artifacts,
        source_identity={"git_dirty": True},
        release_role="frozen_evaluation",
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["required_votes"] == 2
    assert payload["require_all_models"] is True
    assert payload["voting_roles"] == list(ORDINAL_MEMBER_ROLES)
    assert payload["reference_required_for_voting"] is False
    assert payload["members"]["native_ordinal_reference"]["voting_member"] is False
    assert all(payload["members"][role]["voting_member"] for role in ORDINAL_MEMBER_ROLES)


def test_native_celsius_probability_target_prefers_settlement_celsius() -> None:
    features, predictions = _synthetic_frames(80)
    features["actual_high_c"] = (features["actual_high_f"] - 32.0) * 5.0 / 9.0
    features["actual_source"] = "synthetic_settlement"
    features["actual_high_c_source"] = "settlement_high_c"
    features["actual_high_c_settlement_source"] = "synthetic settlement"
    frame = build_probability_frame(
        features,
        predictions,
        providers=("gfs", "hrrr", "nbm"),
        unit="C",
        bucket_width=1,
    )
    assert frame["actual_market_bucket"].str.endswith("C").all()
    assert np.allclose(frame["actual_high_native"], features["actual_high_c"])


def test_native_celsius_target_requires_matching_source_identity() -> None:
    features, predictions = _synthetic_frames(80)
    features["actual_high_c"] = (features["actual_high_f"] - 32.0) * 5.0 / 9.0
    features["actual_source"] = "settlement_a"
    features["actual_high_c_source"] = "settlement_high_c"
    features["actual_high_c_settlement_source"] = "settlement_b"
    with pytest.raises(ValueError, match="provenance differs"):
        build_probability_frame(features, predictions, providers=("gfs", "hrrr", "nbm"), unit="C", bucket_width=1)
    features["actual_high_c_settlement_source"] = pd.NA
    with pytest.raises(ValueError, match="provenance is missing"):
        build_probability_frame(features, predictions, providers=("gfs", "hrrr", "nbm"), unit="C", bucket_width=1)


def test_probability_rounding_and_join_audit_fail_closed() -> None:
    assert [round_half_up(value) for value in (-2.5, -1.5, -0.5, 0.5, 1.5)] == [-2, -1, 0, 1, 2]
    features, predictions = _synthetic_frames(4)
    bad = predictions.copy()
    bad.loc[0, "contract_date"] = pd.Timestamp("2030-01-01")
    with pytest.raises(ValueError, match="no matching feature row"):
        build_probability_frame(features, bad, providers=("gfs", "hrrr", "nbm"), unit="F", bucket_width=2)
    frame = build_probability_frame(features, predictions, providers=("gfs", "hrrr", "nbm"), unit="F", bucket_width=2)
    assert frame.attrs["row_completeness"]["unmatched_point_rows"] == 0
    assert frame.attrs["feature_missingness_before_imputation"]
