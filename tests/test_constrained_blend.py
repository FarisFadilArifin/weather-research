from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.calibration.constrained_blend import (
    FixedWeightBlendRegressor,
    FixedWeightSimplexBlendRegressor,
    blend_predictions,
    blend_simplex_predictions,
    merge_multiple_prediction_sources,
    merge_prediction_sources,
    scan_three_model_simplex_weights,
    scan_simplex_weights,
    select_simplex_weights,
    scan_two_model_weights,
    select_three_model_simplex_weights,
    select_two_model_weight,
    selected_fold_metrics,
)


def _predictions(method: str, values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "contract_date": ["2022-01-01", "2022-01-02", "2023-01-01", "2023-01-02"],
            "actual_high_f": [10.0, 20.0, 30.0, 40.0],
            "method": method,
            "predicted_high_f": values,
            "evaluation_scope": "year_split_validation",
            "fold": ["fold_2022", "fold_2022", "fold_2023", "fold_2023"],
        }
    )


def test_constrained_blend_selects_fold_balanced_weight() -> None:
    xgboost = _predictions("xgboost", [10.0, 20.0, 32.0, 42.0])
    extra_trees = _predictions("extra_trees", [8.0, 18.0, 30.0, 40.0])
    merged = merge_prediction_sources(xgboost, extra_trees)

    scan = scan_two_model_weights(merged, grid_step=0.5)
    selected = select_two_model_weight(scan, grid_step=0.5)
    blended = blend_predictions(merged, selected)
    folds = selected_fold_metrics(blended, selected)

    assert selected.primary_weight == pytest.approx(0.5)
    assert selected.secondary_weight == pytest.approx(0.5)
    assert set(folds["fold"]) == {"fold_2022", "fold_2023"}
    assert blended["method"].eq("xgb_extra_constrained_blend").all()


def test_fixed_weight_blend_regressor_predicts_dataframe_and_array() -> None:
    model = FixedWeightBlendRegressor(
        ("xgboost_predicted_high_f", "extra_trees_predicted_high_f"),
        (0.6, 0.4),
    )
    frame = pd.DataFrame(
        {
            "xgboost_predicted_high_f": [10.0, 20.0],
            "extra_trees_predicted_high_f": [15.0, 25.0],
        }
    )

    expected = np.array([12.0, 22.0])
    assert np.allclose(model.predict(frame), expected)
    assert np.allclose(model.predict(frame.to_numpy()), expected)


def test_fixed_weight_blend_regressor_rejects_invalid_weights() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        FixedWeightBlendRegressor(("x", "y"), (0.8, 0.8))


def test_three_model_simplex_scan_can_assign_zero_weight() -> None:
    xgboost = _predictions("xgboost", [10.0, 20.0, 32.0, 42.0])
    extra_trees = _predictions("extra_trees", [8.0, 18.0, 30.0, 40.0])
    svr = _predictions("svr", [20.0, 30.0, 40.0, 50.0])
    merged = merge_multiple_prediction_sources(
        {"xgboost": xgboost, "extra_trees": extra_trees, "svr": svr}
    )

    scan = scan_three_model_simplex_weights(
        merged,
        methods=("xgboost", "extra_trees", "svr"),
        grid_step=0.5,
    )
    selected = select_three_model_simplex_weights(
        scan,
        methods=("xgboost", "extra_trees", "svr"),
    )
    weights = tuple(float(selected[f"{method}_weight"]) for method in ("xgboost", "extra_trees", "svr"))
    blended = blend_simplex_predictions(
        merged,
        methods=("xgboost", "extra_trees", "svr"),
        weights=weights,
        method="xgb_extra_svr_simplex_blend",
    )
    model = FixedWeightSimplexBlendRegressor(
        (
            "xgboost_predicted_high_f",
            "extra_trees_predicted_high_f",
            "svr_predicted_high_f",
        ),
        weights,
    )

    assert weights == pytest.approx((0.5, 0.5, 0.0))
    assert np.allclose(model.predict(merged.loc[:, model.feature_names]), blended["predicted_high_f"])


def test_generic_four_model_simplex_enumerates_zero_weights() -> None:
    methods = ("a", "b", "c", "d")
    merged = merge_multiple_prediction_sources(
        {method: _predictions(method, [10.0 + index, 20.0 + index, 30.0 + index, 40.0 + index]) for index, method in enumerate(methods)}
    )
    scan = scan_simplex_weights(merged, methods=methods, grid_step=0.5)
    assert len(scan) == 10
    assert np.allclose(scan[[f"{method}_weight" for method in methods]].sum(axis=1), 1.0)
    assert (scan[[f"{method}_weight" for method in methods]] == 0.0).any().any()


def test_generic_simplex_shortlists_mean_then_prefers_worst_fold_stability() -> None:
    methods = ("a", "b", "c", "d")
    scan = pd.DataFrame(
        [
            {"a_weight": 1.0, "b_weight": 0.0, "c_weight": 0.0, "d_weight": 0.0, "mean_fold_mae_f": 1.000, "worst_fold_mae_f": 1.30, "row_mae_f": 1.0},
            {"a_weight": 0.25, "b_weight": 0.25, "c_weight": 0.25, "d_weight": 0.25, "mean_fold_mae_f": 1.009, "worst_fold_mae_f": 1.10, "row_mae_f": 1.0},
            {"a_weight": 0.0, "b_weight": 1.0, "c_weight": 0.0, "d_weight": 0.0, "mean_fold_mae_f": 1.011, "worst_fold_mae_f": 1.00, "row_mae_f": 1.0},
        ]
    )
    selected = select_simplex_weights(scan, methods=methods, mean_mae_tolerance_f=0.01)
    assert tuple(float(selected[f"{method}_weight"]) for method in methods) == pytest.approx((0.25, 0.25, 0.25, 0.25))
