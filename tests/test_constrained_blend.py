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
