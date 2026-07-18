from __future__ import annotations

import json

import pandas as pd
import pytest

from src.calibration.station_stacking import _fit_feature_columns
from src.calibration.v19_bucket import (
    CumulativeOrdinalClassifier,
    _offset_probabilities_to_buckets,
    bucket_decision_metrics,
    crossfit_ridge_predictions,
    empirical_modal_bucket_decisions,
    feature_missingness_audit,
    paired_bootstrap_bucket_gain,
    ordinal_blend_bucket_decisions,
    temperature_bucket_label,
)


def test_fit_feature_columns_applies_three_percent_training_gate() -> None:
    frame = pd.DataFrame(
        {
            "dense": list(range(97)) + [None, None, None],
            "too_sparse": list(range(96)) + [None, None, None, None],
            "category": ["ok"] * 97 + [None, None, None],
        }
    )

    categorical, numeric = _fit_feature_columns(
        frame,
        ["category"],
        ["dense", "too_sparse"],
        max_missing_fraction=0.03,
    )

    assert categorical == ["category"]
    assert numeric == ["dense"]


def test_feature_missingness_audit_uses_training_years_only() -> None:
    frame = pd.DataFrame(
        {
            "year": [2025, 2025, 2026],
            "feature": [1.0, 2.0, None],
        }
    )

    audit = feature_missingness_audit(frame, [], ["feature"], train_years=(2021, 2025))

    assert audit.iloc[0]["missing_pct"] == pytest.approx(0.0)
    assert bool(audit.iloc[0]["keep_v19"])


@pytest.mark.parametrize(
    ("value", "expected"),
    [(90.49, "90-91"), (91.5, "92-93"), (89.5, "90-91"), (-1.5, "-2--1")],
)
def test_temperature_bucket_label_uses_half_up_rounding(value: float, expected: str) -> None:
    assert temperature_bucket_label(value) == expected


def test_empirical_modal_bucket_can_differ_from_rounded_point() -> None:
    residuals = pd.DataFrame(
        {
            "contract_date": pd.date_range("2023-07-01", periods=10),
            "residual_f": [1.2] * 8 + [-0.1] * 2,
        }
    )
    test = pd.DataFrame(
        {
            "contract_date": ["2026-07-14"],
            "method": ["ridge_stack"],
            "actual_high_f": [92.4],
            "predicted_high_f": [91.2],
        }
    )

    decisions = empirical_modal_bucket_decisions(test, residuals, monthly_shrinkage=0.0)

    row = decisions.iloc[0]
    assert row["point_bucket"] == "90-91"
    assert row["modal_bucket"] == "92-93"
    assert bool(row["modal_bucket_hit"])
    probabilities = json.loads(row["bucket_probabilities_json"])
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert probabilities["92-93"] == pytest.approx(0.8)


def test_crossfit_ridge_uses_only_earlier_validation_years() -> None:
    rows = []
    methods = ("xgboost", "lightgbm", "catboost", "gfs_raw", "hrrr_raw", "nbm_raw")
    for year in (2022, 2023):
        for day in range(1, 4):
            actual = 70.0 + day + (year - 2022)
            for offset, method in enumerate(methods):
                rows.append(
                    {
                        "contract_date": f"{year}-01-{day:02d}",
                        "method": method,
                        "actual_high_f": actual,
                        "predicted_high_f": actual + (offset - 2) * 0.1,
                    }
                )
    tuning = pd.DataFrame(
        {
            "status": ["ok"],
            "mae_f": [0.2],
            "param_key": ["stack_trial_0"],
            "feature_set": ["models_only"],
            "alpha": [1.0],
            "fit_intercept": [True],
        }
    )

    predictions = crossfit_ridge_predictions(pd.DataFrame(rows), tuning, min_train_rows=3)

    assert predictions["validation_year"].unique().tolist() == [2023]
    assert predictions["train_through_year"].unique().tolist() == [2022]
    assert len(predictions) == 3


def test_nested_ridge_predictions_are_invariant_to_future_year_changes() -> None:
    methods = ("xgboost", "lightgbm", "catboost", "gfs_raw", "hrrr_raw", "nbm_raw")
    rows = []
    for year in (2022, 2023, 2024, 2025):
        for day in range(1, 6):
            actual = 70.0 + day
            for method_index, method in enumerate(methods):
                rows.append(
                    {
                        "contract_date": f"{year}-01-{day:02d}",
                        "method": method,
                        "actual_high_f": actual,
                        "predicted_high_f": actual + 0.1 * (method_index - 2),
                    }
                )
    original = pd.DataFrame(rows)
    changed = original.copy()
    changed.loc[changed["contract_date"].str.startswith("2025-"), "predicted_high_f"] += 50.0

    before = crossfit_ridge_predictions(original, min_train_rows=5)
    after = crossfit_ridge_predictions(changed, min_train_rows=5)
    before = before.loc[before["validation_year"].le(2024)].reset_index(drop=True)
    after = after.loc[after["validation_year"].le(2024)].reset_index(drop=True)

    pd.testing.assert_frame_equal(before, after)


def test_cumulative_ordinal_classifier_returns_ordered_class_distribution() -> None:
    x = pd.DataFrame({"signal": [-2, -1, 0, 1, 2] * 20})
    y = pd.Series([-2, -1, 0, 1, 2] * 20)
    model = CumulativeOrdinalClassifier(random_state=1).fit(x, y)

    probabilities = model.predict_proba(pd.DataFrame({"signal": [-2, 0, 2]}))

    assert model.classes_.tolist() == [-2, -1, 0, 1, 2]
    assert probabilities.sum(axis=1).tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert probabilities[0].argmax() < probabilities[2].argmax()


def test_censored_tail_probability_uses_empirical_tail_shape() -> None:
    probabilities = _offset_probabilities_to_buckets(
        90.2,
        classes=pd.Series([-2, -1, 0, 1, 2]).to_numpy(),
        probabilities=pd.Series([0.2, 0.1, 0.4, 0.1, 0.2]).to_numpy(),
        empirical_reference={"84-85": 0.1, "86-87": 0.1, "88-89": 0.1, "90-91": 0.4, "92-93": 0.1, "94-95": 0.1, "96-97": 0.1},
    )

    assert probabilities["84-85"] == pytest.approx(0.1)
    assert probabilities["86-87"] == pytest.approx(0.1)
    assert probabilities["94-95"] == pytest.approx(0.1)
    assert probabilities["96-97"] == pytest.approx(0.1)
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_metrics_and_bootstrap_report_paired_gain() -> None:
    decisions = pd.DataFrame(
        {
            "point_bucket_hit": [True, False, False, True],
            "modal_bucket_hit": [True, True, False, True],
            "bucket_log_loss": [0.2, 0.3, 1.0, 0.1],
        }
    )

    metrics = bucket_decision_metrics(decisions)
    gain = paired_bootstrap_bucket_gain(decisions, repetitions=200, random_state=1)

    assert metrics.set_index("decision").loc["rounded_point", "bucket_accuracy_pct"] == pytest.approx(50.0)
    assert metrics.set_index("decision").loc["empirical_modal", "bucket_accuracy_pct"] == pytest.approx(75.0)
    assert gain["gain_pp"] == pytest.approx(25.0)


def test_ordinal_blend_selects_weight_without_using_test_labels_for_tuning() -> None:
    methods = ("xgboost", "lightgbm", "catboost", "gfs_raw", "hrrr_raw", "nbm_raw")
    validation_rows = []
    residual_rows = []
    for year in (2023, 2024, 2025):
        for day in range(1, 31):
            point = 80.2 + (day % 5)
            offset = (-1, 0, 1)[day % 3]
            actual = point + 2.0 * offset
            date = f"{year}-07-{day:02d}"
            residual_rows.append(
                {
                    "contract_date": date,
                    "actual_high_f": actual,
                    "predicted_high_f": point,
                    "residual_f": actual - point,
                    "validation_year": year,
                }
            )
            for method_index, method in enumerate(methods):
                validation_rows.append(
                    {
                        "contract_date": date,
                        "method": method,
                        "actual_high_f": actual,
                        "predicted_high_f": point + 0.05 * (method_index - 2),
                    }
                )
    test_rows = []
    for day in range(1, 11):
        point = 82.2 + (day % 4)
        actual = point + 2.0 * ((-1, 0, 1)[day % 3])
        date = f"2026-07-{day:02d}"
        for method_index, method in enumerate((*methods, "ridge_stack")):
            predicted = point if method == "ridge_stack" else point + 0.05 * (method_index - 2)
            test_rows.append(
                {
                    "contract_date": date,
                    "method": method,
                    "actual_high_f": actual,
                    "predicted_high_f": predicted,
                }
            )

    decisions, tuning, metadata = ordinal_blend_bucket_decisions(
        pd.DataFrame(validation_rows),
        pd.DataFrame(test_rows),
        pd.DataFrame(residual_rows),
        monthly_shrinkage=10.0,
        min_train_rows=20,
    )

    assert len(decisions) == 10
    assert set(tuning["ordinal_weight"]) == {0.0, 0.25, 0.5, 0.75, 1.0}
    assert metadata["classifier_oof_rows"] == 60
    assert metadata["selected_ordinal_weight"] in set(tuning["ordinal_weight"])
    assert sum(json.loads(decisions.iloc[0]["bucket_probabilities_json"]).values()) == pytest.approx(1.0)
