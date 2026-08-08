from __future__ import annotations

import json
import math
import sys
import time
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path("D:/dev/weather-research-clean-export-d4a6a8b")
if not SOURCE_ROOT.exists():
    raise FileNotFoundError(f"Missing original v20 source export: {SOURCE_ROOT}")
sys.path.insert(0, str(SOURCE_ROOT))

from src.calibration.station_stacking import (  # noqa: E402
    OBSERVED_HIGH_SO_FAR_COLUMN,
    StationStackingConfig,
    _build_base_model_estimator,
    _build_preprocessor,
    _fit_feature_columns,
    _model_target_values,
    _modeling_frame,
    _params_from_selected_row,
    _round_half_up_series,
    _temperature_bracket_from_rounded,
)


ORIGINAL_ARTIFACT_DIR = (
    SOURCE_ROOT / "data/calibration/station_stacking_v20_kdal_no_peak"
)
CURRENT_ARTIFACT_DIR = (
    REPO_ROOT / "data/calibration/station_stacking_v20_kdal_no_peak"
)
BUNDLE_PATH = (
    ORIGINAL_ARTIFACT_DIR
    / "model_weights/KDAL_station_high_regressor_v20_kdal_no_peak_stack.joblib"
)
OUTPUT_DIR = REPO_ROOT / "reports/v20_kdal_refit_170_plus"
METHODS = ("xgboost", "lightgbm", "catboost")
CUTOFF = pd.Timestamp("2026-06-21")


def config() -> StationStackingConfig:
    return StationStackingConfig(
        station_id="KDAL",
        project_root=SOURCE_ROOT,
        timing_mode="same_day_11am_live_safe",
        providers=("gfs", "hrrr", "nbm"),
        feature_version="v11_settlement_fix_temp",
        training_profile="v20_aligned",
        optuna_metric="mae_f",
        target_mode="remaining_warmup",
        target_source="wunderground_only",
        base_model_methods=METHODS,
        stack_enabled=True,
        max_feature_missing_fraction=0.03,
        output_dir=ORIGINAL_ARTIFACT_DIR,
    )


def predict_with_models(
    frame: pd.DataFrame,
    *,
    feature_names: list[str],
    preprocessor: object | None,
    models: dict[str, object],
    stack_model: object,
    stack_features: list[str],
) -> tuple[float, dict[str, float]]:
    x = (
        preprocessor.transform(frame[feature_names])
        if preprocessor is not None
        else frame[feature_names]
    )
    observed = pd.to_numeric(
        frame[OBSERVED_HIGH_SO_FAR_COLUMN], errors="coerce"
    ).to_numpy(dtype=float)
    base: dict[str, float] = {}
    for method in METHODS:
        raw = np.asarray(models[method].predict(x), dtype=float)
        base[method] = float(np.maximum(observed, observed + raw)[0])
    stack_source = pd.DataFrame(
        {
            f"{method}_predicted_high_f": [base[method]]
            for method in METHODS
        }
    )
    for provider in ("gfs", "hrrr", "nbm"):
        stack_source[f"{provider}_raw_predicted_high_f"] = [
            float(frame[f"{provider}_high_f"].iloc[0])
        ]
    point = float(stack_model.predict(stack_source[stack_features])[0])
    return point, base


def fit_models(
    train: pd.DataFrame,
    categorical: list[str],
    numeric: list[str],
    params: dict[str, dict[str, object]],
) -> tuple[list[str], object, dict[str, object]]:
    cfg = config()
    fit_categorical, fit_numeric = _fit_feature_columns(
        train,
        categorical,
        numeric,
        max_missing_fraction=cfg.effective_max_feature_missing_fraction,
    )
    feature_names = [*fit_categorical, *fit_numeric]
    preprocessor = _build_preprocessor(fit_categorical, fit_numeric)
    x = preprocessor.fit_transform(train[feature_names])
    y = _model_target_values(train, cfg)
    models = {}
    for method in METHODS:
        estimator = _build_base_model_estimator(cfg, method, params[method])
        estimator.fit(x, y)
        models[method] = estimator
    return feature_names, preprocessor, models


def exact_mcnemar_pvalue(fixed_only: int, refit_only: int) -> float:
    n = fixed_only + refit_only
    if n == 0:
        return 1.0
    tail = sum(
        math.comb(n, k) * 0.5**n
        for k in range(min(fixed_only, refit_only) + 1)
    )
    return float(min(1.0, 2.0 * tail))


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = config()
    original = pd.read_csv(
        ORIGINAL_ARTIFACT_DIR / "KDAL_features.csv", low_memory=False
    )
    current = pd.read_csv(
        CURRENT_ARTIFACT_DIR / "KDAL_features.csv", low_memory=False
    )
    original["contract_date"] = pd.to_datetime(original["contract_date"])
    current["contract_date"] = pd.to_datetime(current["contract_date"])
    features = pd.concat(
        [
            original.loc[original["contract_date"].le(CUTOFF)],
            current.loc[current["contract_date"].gt(CUTOFF)],
        ],
        ignore_index=True,
        sort=False,
    )
    modeling, categorical, numeric = _modeling_frame(features, cfg)
    base_train = modeling.loc[modeling["contract_date"].le(CUTOFF)].copy()
    tail = modeling.loc[modeling["contract_date"].gt(CUTOFF)].copy()
    tail = tail.sort_values("contract_date").reset_index(drop=True)
    base_2026_days = int(
        pd.to_datetime(base_train["contract_date"]).dt.year.eq(2026).sum()
    )
    if base_2026_days != 170:
        raise ValueError(f"Expected 170 base 2026 days, found {base_2026_days}")
    if tail.empty or tail["contract_date"].max() != pd.Timestamp("2026-07-29"):
        raise ValueError("Expected an eligible tail ending 2026-07-29")

    selected = pd.read_csv(
        ORIGINAL_ARTIFACT_DIR / "KDAL_year_split_selected_hyperparameters.csv"
    )
    params = {
        method: _params_from_selected_row(
            selected.loc[selected["method"].eq(method)].iloc[0]
        )
        for method in METHODS
    }
    bundle = joblib.load(BUNDLE_PATH)
    stack_model = bundle["stack_model"]
    stack_features = list(bundle["stack_features"])

    # Rebuild model 170 once as a runtime/source-code reproduction check.
    print(
        f"Rebuilding model 170 on {len(base_train)} total rows...",
        flush=True,
    )
    rebuilt_features, rebuilt_prep, rebuilt_models = fit_models(
        base_train, categorical, numeric, params
    )

    rows: list[dict[str, object]] = []
    for index, test in tail.iterrows():
        test_frame = test.to_frame().T
        train = modeling.loc[
            modeling["contract_date"].lt(test["contract_date"])
        ].copy()
        training_2026_days = int(
            pd.to_datetime(train["contract_date"]).dt.year.eq(2026).sum()
        )
        started = time.perf_counter()

        fixed_prediction, fixed_base = predict_with_models(
            test_frame,
            feature_names=list(bundle["feature_names"]),
            preprocessor=None,
            models=bundle["base_models"],
            stack_model=stack_model,
            stack_features=stack_features,
        )
        rebuilt_170_prediction, _ = predict_with_models(
            test_frame,
            feature_names=rebuilt_features,
            preprocessor=rebuilt_prep,
            models=rebuilt_models,
            stack_model=stack_model,
            stack_features=stack_features,
        )
        if training_2026_days == 170:
            refit_prediction = fixed_prediction
            refit_base = fixed_base
        else:
            feature_names, preprocessor, models = fit_models(
                train, categorical, numeric, params
            )
            refit_prediction, refit_base = predict_with_models(
                test_frame,
                feature_names=feature_names,
                preprocessor=preprocessor,
                models=models,
                stack_model=stack_model,
                stack_features=stack_features,
            )
        row = {
            "contract_date": test["contract_date"].strftime("%Y-%m-%d"),
            "training_2026_days": training_2026_days,
            "train_rows": len(train),
            "train_last_contract_date": train["contract_date"].max().strftime(
                "%Y-%m-%d"
            ),
            "actual_high_f": float(test["actual_high_f"]),
            "fixed_170_predicted_high_f": fixed_prediction,
            "daily_refit_predicted_high_f": refit_prediction,
            "rebuilt_170_predicted_high_f": rebuilt_170_prediction,
            **{
                f"fixed_170_{method}_predicted_high_f": value
                for method, value in fixed_base.items()
            },
            **{
                f"daily_refit_{method}_predicted_high_f": value
                for method, value in refit_base.items()
            },
            "fit_seconds": time.perf_counter() - started,
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUTPUT_DIR / "checkpoint.csv", index=False)
        print(
            f"[{index + 1}/{len(tail)}] model={training_2026_days}, "
            f"date={row['contract_date']}, fixed={fixed_prediction:.3f}, "
            f"refit={refit_prediction:.3f}, seconds={row['fit_seconds']:.1f}",
            flush=True,
        )

    detail = pd.DataFrame(rows)
    detail["actual_rounded_f"] = _round_half_up_series(detail["actual_high_f"])
    detail["actual_bucket"] = detail["actual_rounded_f"].map(
        _temperature_bracket_from_rounded
    )
    for prefix in ("fixed_170", "daily_refit"):
        detail[f"{prefix}_rounded_f"] = _round_half_up_series(
            detail[f"{prefix}_predicted_high_f"]
        )
        detail[f"{prefix}_bucket"] = detail[f"{prefix}_rounded_f"].map(
            _temperature_bracket_from_rounded
        )
        detail[f"{prefix}_bucket_hit"] = detail[f"{prefix}_bucket"].eq(
            detail["actual_bucket"]
        )
        detail[f"{prefix}_error_f"] = (
            detail["actual_high_f"] - detail[f"{prefix}_predicted_high_f"]
        )
        detail[f"{prefix}_absolute_error_f"] = detail[
            f"{prefix}_error_f"
        ].abs()
    detail["refit_hit_gain"] = (
        detail["daily_refit_bucket_hit"].astype(int)
        - detail["fixed_170_bucket_hit"].astype(int)
    )
    detail["refit_minus_fixed_prediction_f"] = (
        detail["daily_refit_predicted_high_f"]
        - detail["fixed_170_predicted_high_f"]
    )
    detail["rebuilt_170_minus_artifact_f"] = (
        detail["rebuilt_170_predicted_high_f"]
        - detail["fixed_170_predicted_high_f"]
    )
    detail.to_csv(OUTPUT_DIR / "detail.csv", index=False)

    summary_rows = []
    for prefix in ("fixed_170", "daily_refit"):
        summary_rows.append(
            {
                "method": prefix,
                "days": len(detail),
                "bucket_hits": int(detail[f"{prefix}_bucket_hit"].sum()),
                "bucket_hit_rate_pct": float(
                    detail[f"{prefix}_bucket_hit"].mean() * 100
                ),
                "mae_f": float(detail[f"{prefix}_absolute_error_f"].mean()),
                "rmse_f": float(
                    np.sqrt(np.mean(np.square(detail[f"{prefix}_error_f"])))
                ),
                "bias_actual_minus_prediction_f": float(
                    detail[f"{prefix}_error_f"].mean()
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)

    fixed_only = int(
        (
            detail["fixed_170_bucket_hit"]
            & ~detail["daily_refit_bucket_hit"]
        ).sum()
    )
    refit_only = int(
        (
            ~detail["fixed_170_bucket_hit"]
            & detail["daily_refit_bucket_hit"]
        ).sum()
    )
    both = int(
        (
            detail["fixed_170_bucket_hit"]
            & detail["daily_refit_bucket_hit"]
        ).sum()
    )
    gain = detail["refit_hit_gain"].to_numpy(dtype=float)
    rng = np.random.default_rng(20260730)
    bootstrap = rng.choice(
        gain, size=(100_000, len(gain)), replace=True
    ).mean(axis=1)
    result = {
        "experiment": (
            "Frozen live model 170 versus export-style daily refits 170, "
            "171, 172, ... using only prior eligible enriched days"
        ),
        "base_cutoff": CUTOFF.strftime("%Y-%m-%d"),
        "base_2026_training_days": base_2026_days,
        "tail_last_date": tail["contract_date"].max().strftime("%Y-%m-%d"),
        "eligible_tail_days": len(tail),
        "training_day_versions_tested": detail["training_2026_days"].tolist(),
        "summary": summary.to_dict(orient="records"),
        "paired": {
            "both_hit": both,
            "fixed_only_hit": fixed_only,
            "refit_only_hit": refit_only,
            "neither_hit": len(detail) - both - fixed_only - refit_only,
            "net_hits": refit_only - fixed_only,
            "hit_rate_gain_pp": float(gain.mean() * 100),
            "bootstrap_95ci_low_pp": float(np.quantile(bootstrap, 0.025) * 100),
            "bootstrap_95ci_high_pp": float(np.quantile(bootstrap, 0.975) * 100),
            "mcnemar_exact_two_sided_pvalue": exact_mcnemar_pvalue(
                fixed_only, refit_only
            ),
        },
        "reproduction_check": {
            "mean_absolute_rebuilt_170_minus_artifact_f": float(
                detail["rebuilt_170_minus_artifact_f"].abs().mean()
            ),
            "max_absolute_rebuilt_170_minus_artifact_f": float(
                detail["rebuilt_170_minus_artifact_f"].abs().max()
            ),
        },
        "coverage_note": (
            "Only rows passing the original v20 production modeling contract "
            "are eligible. The enriched file ends July 29, but missing provider "
            "inputs create calendar-date gaps in the 18-row forward tail."
        ),
        "runtime_versions": {
            package: version(package)
            for package in (
                "numpy",
                "pandas",
                "scikit-learn",
                "xgboost",
                "lightgbm",
                "catboost",
                "joblib",
            )
        },
    }
    (OUTPUT_DIR / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "checkpoint.csv").unlink(missing_ok=True)
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
