from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from src.calibration.bucket_probability import probability_feature_names
from src.calibration.celsius_market_probability import (
    CelsiusCandidate,
    OFFSET_LABELS_C,
    build_celsius_probability_frame,
    export_celsius_probability_bundle,
    fahrenheit_to_celsius,
    fit_celsius_probability_system,
    offset_class_index_c,
    offset_to_market_bucket_probabilities_c,
    predict_celsius_probability_bundle,
    round_half_up,
    sha256_file,
)


def test_round_half_up_market_boundaries() -> None:
    assert round_half_up(12.49) == 12
    assert round_half_up(12.50) == 13
    assert round_half_up(12.51) == 13
    assert round_half_up(-1.50) == -1


def test_fahrenheit_to_celsius_conversion() -> None:
    assert fahrenheit_to_celsius(32.0) == 0.0
    assert fahrenheit_to_celsius(68.0) == 20.0
    assert math.isclose(fahrenheit_to_celsius(91.4), 33.0)


def test_exact_celsius_bucket_and_offset_construction() -> None:
    dates = pd.to_datetime(["2025-01-01"])
    features = pd.DataFrame(
        {
            "contract_date": dates,
            "gfs_high_f": [70.0],
            "gefs_high_f": [69.0],
            "jma_msm_high_f": [68.0],
            "observed_temp_at_as_of_f": [64.0],
            "observed_high_temp_through_as_of_f": [65.0],
            "observed_as_of_age_minutes": [0.0],
        }
    )
    point = pd.DataFrame(
        {
            "contract_date": dates,
            "actual_high_f": [70.7],  # 21.5 C -> 22 C with half-up
            "predicted_high_f": [68.9],  # 20.5 C -> 21 C with half-up
        }
    )
    base = pd.DataFrame(
        [
            {"contract_date": dates[0], "method": method, "predicted_high_f": value}
            for method, value in zip(
                ("xgboost", "lightgbm", "catboost"), (68.0, 69.0, 70.0), strict=True
            )
        ]
    )
    frame = build_celsius_probability_frame(
        features,
        point,
        base,
        include_peak_features=False,
        feature_profile="asia_no_peak",
    )
    row = frame.iloc[0]
    assert row["point_bucket_c"] == 21
    assert row["actual_bucket_c"] == 22
    assert row["offset_c"] == 1
    assert row["offset_class_c"] == offset_class_index_c(1)
    assert row["actual_high_c_source"] == "actual_high_f_converted_to_c"


def test_offset_mapping_is_exact_complete_and_recommends_maximum() -> None:
    offset = dict(zip(OFFSET_LABELS_C, (0.05, 0.10, 0.15, 0.20, 0.30, 0.10, 0.10), strict=True))
    tail = {
        "low_exact_offset_weights": {"-4": 0.25, "-3": 0.75},
        "high_exact_offset_weights": {"3": 1.0},
    }
    market = offset_to_market_bucket_probabilities_c(20, offset, tail)
    assert np.isclose(sum(market.values()), 1.0)
    assert np.isclose(market[16], 0.0125)
    assert np.isclose(market[17], 0.0375)
    assert np.isclose(market[21], 0.30)
    assert max(market, key=market.get) == 21


def _synthetic_training_frame() -> pd.DataFrame:
    dates = list(pd.date_range("2024-01-01", periods=220, freq="D"))
    dates += list(pd.date_range("2025-01-01", periods=50, freq="D"))
    dates += list(pd.date_range("2026-01-01", periods=10, freq="D"))
    frame = pd.DataFrame({"contract_date": dates})
    frame["year"] = frame["contract_date"].dt.year
    frame["month"] = frame["contract_date"].dt.month
    frame["point_prediction_f"] = 68.0 + np.sin(np.arange(len(frame)) / 10.0)
    frame["point_prediction_c"] = frame["point_prediction_f"].map(fahrenheit_to_celsius)
    frame["point_bucket_c"] = frame["point_prediction_c"].map(round_half_up)
    frame["offset_c"] = (np.arange(len(frame)) % 7) - 3
    frame["offset_class_c"] = frame["offset_c"].map(offset_class_index_c)
    frame["actual_bucket_c"] = frame["point_bucket_c"] + frame["offset_c"]
    frame["actual_high_c"] = frame["actual_bucket_c"].astype(float)
    frame["actual_high_f"] = frame["actual_high_c"] * 9.0 / 5.0 + 32.0
    frame["actual_high_c_source"] = "actual_high_f_converted_to_c"
    frame["observed_high_temp_through_as_of_f"] = 65.0
    for position, name in enumerate(
        probability_feature_names(include_peak_features=False, feature_profile="asia_no_peak")
    ):
        if name not in frame:
            frame[name] = np.sin(np.arange(len(frame)) / (position + 2.0))
    return frame


def test_chronology_holdout_exclusion_and_manifest_integrity(tmp_path) -> None:
    frame = _synthetic_training_frame()
    point_bundle = tmp_path / "point.joblib"
    point_bundle.write_bytes(b"fresh point bundle")
    bundle, forward, _ = fit_celsius_probability_system(
        frame,
        station_id="RJTT",
        point_model_version="point-v1",
        point_bundle_sha256=sha256_file(point_bundle),
        feature_profile="asia_no_peak",
        model_version="station_bucket_baseline_tokyo_1c_market_ordinal",
        development_years=(2024, 2025),
        forward_validation_years=(2025,),
        calibration_days=30,
        min_train_rows=100,
        candidates=(CelsiusCandidate(0.1, None),),
        temperature_grid=(1.0,),
    )
    assert bundle["training_cutoff"] < "2026-01-01"
    assert bundle["development_years"] == [2024, 2025]
    assert pd.to_datetime(forward["model_training_cutoff"]).lt(
        pd.to_datetime(forward["contract_date"])
    ).all()
    assert pd.to_datetime(forward["calibration_validation_cutoff"]).lt(
        pd.to_datetime(forward["contract_date"])
    ).all()
    assert forward["market_bucket_probabilities_c"].map(
        lambda probabilities: np.isclose(sum(probabilities.values()), 1.0)
    ).all()
    assert forward.apply(
        lambda row: row["recommended_bucket_c"]
        == int(max(row["market_bucket_probabilities_c"], key=row["market_bucket_probabilities_c"].get)),
        axis=1,
    ).all()

    artifact = tmp_path / "forward.csv"
    forward.drop(columns=["celsius_offset_probabilities", "market_bucket_probabilities_c"]).to_csv(
        artifact, index=False
    )
    bundle_path, manifest_path = export_celsius_probability_bundle(
        bundle,
        tmp_path / "weights",
        source_identity={"notebook": "train_Tokyo.ipynb"},
        artifact_paths=[artifact],
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["point_bundle_sha256"] == sha256_file(point_bundle)
    assert manifest["artifact_integrity"]["bundle_sha256"] == sha256_file(bundle_path)
    assert manifest["artifact_integrity"]["artifact_sha256"][artifact.name] == sha256_file(artifact)


def test_confidence_decision_uses_celsius_market_probabilities() -> None:
    feature_names = probability_feature_names(
        include_peak_features=False, feature_profile="asia_no_peak"
    )
    values = {name: 1.0 for name in feature_names if not name.endswith("__missing")}
    values.update(
        {
            "contract_date": "2026-01-01",
            "point_prediction_f": 68.0,
            "xgboost_predicted_high_f": 68.0,
            "lightgbm_predicted_high_f": 68.0,
            "catboost_predicted_high_f": 68.0,
            "gfs_high_f": 68.0,
            "gefs_high_f": 68.0,
            "jma_msm_high_f": 68.0,
            "observed_temp_at_as_of_f": 65.0,
            "observed_high_temp_through_as_of_f": 65.0,
            "observed_as_of_age_minutes": 0.0,
        }
    )
    bundle = {
        "model_version": "test",
        "feature_profile": "asia_no_peak",
        "feature_names": feature_names,
        "mandatory_source_features": [
            "gfs_high_f",
            "gefs_high_f",
            "jma_msm_high_f",
            "observed_temp_at_as_of_f",
            "observed_high_temp_through_as_of_f",
            "observed_as_of_age_minutes",
        ],
        "model_state": {"threshold_models": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]},
        "temperature": 1.0,
        "tail_policy": {
            "low_exact_offset_weights": {"-3": 1.0},
            "high_exact_offset_weights": {"3": 1.0},
        },
        "decision_thresholds": {
            "minimum_top_probability": 0.35,
            "minimum_top_two_margin": 0.20,
            "minimum_switch_advantage": 0.20,
            "tail_ambiguity_rule_enabled": True,
        },
    }
    result = predict_celsius_probability_bundle(bundle, values)
    assert result["status"] == "ok"
    assert result["recommended_bucket_c"] == 23
    assert result["market_top_probability_c"] == 0.4
    assert result["market_probability_decision"] == "shadow_trade"
    assert result["market_probability_decision_reason"] == "confidence_passed"
