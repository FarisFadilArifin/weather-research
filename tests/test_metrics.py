import pandas as pd

from src.metrics import add_error_columns, metrics_by, summarize_errors


def test_error_definition_positive_means_actual_warmer():
    df = pd.DataFrame({"actual_high_f": [92], "forecast_high_f": [90]})
    out = add_error_columns(df)
    assert out.loc[0, "error_f"] == 2
    assert out.loc[0, "abs_error_f"] == 2
    assert out.loc[0, "squared_error_f"] == 4


def test_summarize_errors_core_metrics():
    df = pd.DataFrame({"error_f": [-2, 0, 2]})
    summary = summarize_errors(df)
    assert summary["count"] == 3
    assert summary["mean_error_f"] == 0
    assert round(summary["mae_f"], 3) == 1.333
    assert round(summary["rmse_f"], 3) == 1.633
    assert summary["within_2f_pct"] == 100
    assert round(summary["warm_bias_frequency"], 3) == 33.333
    assert round(summary["cool_bias_frequency"], 3) == 33.333


def test_metrics_by_grouping():
    df = pd.DataFrame(
        {
            "station_code": ["KDAL", "KDAL", "KDFW"],
            "provider": ["hrrr", "hrrr", "openmeteo"],
            "error_f": [1, 3, -2],
        }
    )
    grouped = metrics_by(df, ["station_code", "provider"])
    assert set(grouped["station_code"]) == {"KDAL", "KDFW"}
    kdal = grouped.loc[grouped["station_code"] == "KDAL"].iloc[0]
    assert kdal["count"] == 2
    assert kdal["mae_f"] == 2
