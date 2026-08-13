from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from .constrained_blend import (
    blend_simplex_predictions,
    merge_multiple_prediction_sources,
    scan_simplex_weights,
    select_simplex_weights,
)


EXPERT_METHODS = (
    "full_xgboost",
    "forecast_huber",
    "observation_catboost",
    "seasonal_ridge",
)
TARGET = "actual_high_f"
OBSERVED_FLOOR = "observed_high_temp_through_as_of_f"
MISSINGNESS_LIMIT = 0.03
SIMPLEX_GRID_STEP = 0.025

_NON_FEATURE_EXACT = {
    TARGET,
    "actual_high_c",
    "settlement_high_f",
    "settlement_high_c",
    "contract_date",
    "year",
    "strict_quality_ok",
}
_NON_FEATURE_TOKENS = (
    "diagnostic_only",
    "settlement",
    "iem_actual",
    "iem_daily_high",
    "target_source_diff",
    "target_source",
    "settlement_source",
    "quality_flag",
    "actual_source",
    "source_uri",
    "source_checksum",
    "raw_metar",
    "unavailable_reason",
)
_FORECAST_TOKENS = (
    "provider_",
    "_high_f",
    "forecast_temp_at_as_of",
    "horizon",
    "lead_hours",
    "rolling_bias",
    "rolling_mae",
    "prior_month_bias",
    "prior_month_mae",
    "error_lag",
    "v11sf_",
    "minus_observed",
)
_OBSERVATION_TOKENS = (
    "observed_",
    "v11sf_forecast_temp_11am",
    "cloud",
    "precip",
    "dewpoint",
    "humidity",
    "wind",
    "solar",
    "shortwave",
    "day_of_year",
    "month",
    "is_weekend",
)
_OBSERVATION_EXCLUSIONS = (
    "actual_high_",
    "error_lag",
    "rolling_bias",
    "rolling_mae",
    "prior_month_",
    "provider_mean_high",
    "provider_median_high",
    "provider_min_high",
    "provider_max_high",
    "observation_adjusted_provider_high",
    "high_plus_",
    "provider_",
)


@dataclass(frozen=True)
class ExpertFitAudit:
    method: str
    training_start: str
    training_cutoff: str
    training_rows: int
    feature_count_before_gate: int
    feature_count_after_gate: int
    missingness_limit: float
    eligible_features: tuple[str, ...]
    rejected_missingness: Mapping[str, float]
    target_transform: str
    selected_params: Mapping[str, Any]


@dataclass
class FittedExpert:
    method: str
    feature_names: tuple[str, ...]
    pipeline: Any
    target_transform: str
    audit: ExpertFitAudit

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [name for name in self.feature_names if name not in frame]
        if missing:
            raise ValueError(f"{self.method} inference is missing frozen features: {missing}")
        matrix = frame.loc[:, self.feature_names].apply(pd.to_numeric, errors="coerce")
        transformed = np.asarray(self.pipeline.predict(matrix), dtype=float)
        if self.target_transform == "remaining_warmup":
            transformed = transformed + pd.to_numeric(frame[OBSERVED_FLOOR], errors="coerce").to_numpy()
        elif self.target_transform == "provider_residual":
            transformed = transformed + pd.to_numeric(frame["provider_mean_high_f"], errors="coerce").to_numpy()
        floor = pd.to_numeric(frame[OBSERVED_FLOOR], errors="coerce").to_numpy(dtype=float)
        return np.maximum(transformed, floor)


def route_expert_features(frame: pd.DataFrame, method: str) -> list[str]:
    """Return ordered numeric candidates for one expert before fold missingness gating."""
    if method not in EXPERT_METHODS:
        raise ValueError(f"unknown expert method: {method}")
    numeric = [
        name
        for name in frame.columns
        if name not in _NON_FEATURE_EXACT
        and not any(token in name.lower() for token in _NON_FEATURE_TOKENS)
        and pd.to_numeric(frame[name], errors="coerce").notna().any()
    ]
    if method == "full_xgboost":
        return numeric
    if method == "forecast_huber":
        return [name for name in numeric if any(token in name.lower() for token in _FORECAST_TOKENS)]
    if method == "observation_catboost":
        return [
            name
            for name in numeric
            if any(token in name.lower() for token in _OBSERVATION_TOKENS)
            and not any(token in name.lower() for token in _OBSERVATION_EXCLUSIONS)
            and not (name.endswith("_high_f") and not name.startswith("observed_"))
        ]
    return [
        name
        for name in numeric
        if name == "day_of_year"
        or name.startswith("actual_high_lag_")
        or name.startswith("actual_high_roll_")
        or name.startswith("actual_high_trend_")
    ]


def fold_feature_contract(
    train: pd.DataFrame,
    method: str,
    *,
    missingness_limit: float = MISSINGNESS_LIMIT,
) -> tuple[tuple[str, ...], dict[str, float]]:
    if not 0.0 <= missingness_limit < 1.0:
        raise ValueError("missingness_limit must be in [0, 1)")
    candidates = route_expert_features(train, method)
    missingness = train.loc[:, candidates].apply(pd.to_numeric, errors="coerce").isna().mean()
    eligible = tuple(name for name in candidates if float(missingness[name]) <= missingness_limit)
    rejected = {
        name: float(missingness[name])
        for name in candidates
        if float(missingness[name]) > missingness_limit
    }
    if not eligible:
        raise ValueError(f"{method} has no fold-eligible features")
    return eligible, rejected


def target_values(frame: pd.DataFrame, method: str) -> np.ndarray:
    actual = pd.to_numeric(frame[TARGET], errors="coerce").to_numpy(dtype=float)
    if method in ("full_xgboost", "observation_catboost"):
        anchor = pd.to_numeric(frame[OBSERVED_FLOOR], errors="coerce").to_numpy(dtype=float)
        return actual - anchor
    if method == "forecast_huber":
        anchor = pd.to_numeric(frame["provider_mean_high_f"], errors="coerce").to_numpy(dtype=float)
        return actual - anchor
    if method == "seasonal_ridge":
        return actual
    raise ValueError(f"unknown expert method: {method}")


def _target_transform(method: str) -> str:
    return {
        "full_xgboost": "remaining_warmup",
        "observation_catboost": "remaining_warmup",
        "forecast_huber": "provider_residual",
        "seasonal_ridge": "direct_final_high",
    }[method]


def _chronological_inner_split(train: pd.DataFrame, calibration_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = train.sort_values("contract_date").copy()
    cutoff = pd.to_datetime(ordered["contract_date"]).max() - pd.Timedelta(days=calibration_days - 1)
    inner_train = ordered.loc[pd.to_datetime(ordered["contract_date"]).lt(cutoff)].copy()
    inner_valid = ordered.loc[pd.to_datetime(ordered["contract_date"]).ge(cutoff)].copy()
    if inner_train.empty or inner_valid.empty:
        split = max(1, int(len(ordered) * 0.8))
        inner_train, inner_valid = ordered.iloc[:split].copy(), ordered.iloc[split:].copy()
    if inner_valid.empty or pd.to_datetime(inner_train["contract_date"]).max() >= pd.to_datetime(inner_valid["contract_date"]).min():
        raise ValueError("insufficient strictly chronological tuning history")
    return inner_train, inner_valid


def _linear_pipeline(method: str, features: Sequence[str], params: Mapping[str, Any]) -> Pipeline:
    if method == "forecast_huber":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("spline", SplineTransformer(n_knots=5, degree=3, include_bias=False)),
                ("scale", StandardScaler()),
                ("model", HuberRegressor(epsilon=float(params["epsilon"]), alpha=float(params["alpha"]), max_iter=1000)),
            ]
        )
    day = ["day_of_year"] if "day_of_year" in features else []
    history = [name for name in features if name != "day_of_year"]
    transformers: list[tuple[str, Any, list[str]]] = []
    if day:
        transformers.append(
            ("season", Pipeline([("imputer", SimpleImputer(strategy="median")), ("spline", SplineTransformer(n_knots=12, degree=3, extrapolation="periodic", include_bias=False))]), day)
        )
    if history:
        transformers.append(("history", Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())]), history))
    return Pipeline([("features", ColumnTransformer(transformers)), ("model", Ridge(alpha=float(params["alpha"])))])


def _booster_pipeline(method: str, params: Mapping[str, Any], random_state: int) -> Pipeline:
    if method == "full_xgboost":
        from xgboost import XGBRegressor

        estimator = XGBRegressor(
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=random_state,
            tree_method="hist",
            **params,
        )
    else:
        from catboost import CatBoostRegressor

        estimator = CatBoostRegressor(
            loss_function="MAE",
            verbose=False,
            random_seed=random_state,
            thread_count=-1,
            allow_writing_files=False,
            **params,
        )
    return Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("model", estimator)])


def _tune_params(
    method: str,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: Sequence[str],
    *,
    optuna_trials: int,
    startup_trials: int,
    random_state: int,
) -> dict[str, Any]:
    x_train = train.loc[:, features].apply(pd.to_numeric, errors="coerce")
    x_valid = valid.loc[:, features].apply(pd.to_numeric, errors="coerce")
    y_train, y_valid = target_values(train, method), target_values(valid, method)
    if method == "forecast_huber":
        candidates = (
            {"epsilon": epsilon, "alpha": alpha}
            for epsilon in (1.1, 1.35, 1.5, 1.75, 2.0)
            for alpha in (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
        )
        return min(candidates, key=lambda params: (mean_absolute_error(y_valid, _linear_pipeline(method, features, params).fit(x_train, y_train).predict(x_valid)), params["epsilon"], params["alpha"]))
    if method == "seasonal_ridge":
        candidates = ({"alpha": alpha} for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0))
        return min(candidates, key=lambda params: (mean_absolute_error(y_valid, _linear_pipeline(method, features, params).fit(x_train, y_train).predict(x_valid)), params["alpha"]))

    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: Any) -> float:
        if method == "full_xgboost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 150, 700, step=50),
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "subsample": trial.suggest_float("subsample", 0.65, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 5.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
            }
        else:
            params = {
                "iterations": trial.suggest_int("iterations", 200, 800, step=50),
                "depth": trial.suggest_int("depth", 3, 9),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
                "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
            }
        fitted = _booster_pipeline(method, params, random_state).fit(x_train, y_train)
        return float(mean_absolute_error(y_valid, fitted.predict(x_valid)))

    sampler = optuna.samplers.TPESampler(seed=random_state, n_startup_trials=startup_trials)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=optuna_trials, show_progress_bar=False)
    return dict(study.best_params)


def fit_expert(
    train: pd.DataFrame,
    method: str,
    *,
    optuna_trials: int = 30,
    startup_trials: int = 15,
    calibration_days: int = 90,
    random_state: int = 42,
) -> FittedExpert:
    required_targets = [TARGET, OBSERVED_FLOOR]
    if method == "forecast_huber":
        required_targets.append("provider_mean_high_f")
    ordered = train.sort_values("contract_date").dropna(subset=required_targets).copy()
    features, rejected = fold_feature_contract(ordered, method)
    inner_train, inner_valid = _chronological_inner_split(ordered, calibration_days)
    params = _tune_params(method, inner_train, inner_valid, features, optuna_trials=optuna_trials, startup_trials=startup_trials, random_state=random_state)
    pipeline = (
        _linear_pipeline(method, features, params)
        if method in ("forecast_huber", "seasonal_ridge")
        else _booster_pipeline(method, params, random_state)
    )
    x = ordered.loc[:, features].apply(pd.to_numeric, errors="coerce")
    pipeline.fit(x, target_values(ordered, method))
    candidates = route_expert_features(ordered, method)
    audit = ExpertFitAudit(
        method=method,
        training_start=pd.to_datetime(ordered["contract_date"]).min().date().isoformat(),
        training_cutoff=pd.to_datetime(ordered["contract_date"]).max().date().isoformat(),
        training_rows=len(ordered),
        feature_count_before_gate=len(candidates),
        feature_count_after_gate=len(features),
        missingness_limit=MISSINGNESS_LIMIT,
        eligible_features=features,
        rejected_missingness=rejected,
        target_transform=_target_transform(method),
        selected_params=params,
    )
    return FittedExpert(method, features, pipeline, _target_transform(method), audit)


def crossfit_experts(
    frame: pd.DataFrame,
    evaluation_years: Sequence[int],
    **fit_kwargs: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = frame.copy()
    data["contract_date"] = pd.to_datetime(data["contract_date"], errors="coerce")
    data["year"] = data["contract_date"].dt.year
    predictions: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for year in evaluation_years:
        train = data.loc[data["year"].lt(int(year))].copy()
        valid = data.loc[data["year"].eq(int(year))].copy()
        if train.empty or valid.empty:
            continue
        if train["contract_date"].max() >= valid["contract_date"].min():
            raise AssertionError("expert fold chronology is invalid")
        for method in EXPERT_METHODS:
            fitted = fit_expert(train, method, **fit_kwargs)
            part = valid.loc[:, ["contract_date", TARGET]].copy()
            part["method"] = method
            part["predicted_high_f"] = fitted.predict(valid)
            part["fold"] = f"year_{year}"
            part["validation_year"] = int(year)
            part["model_training_cutoff"] = train["contract_date"].max()
            predictions.append(part)
            audits.append({"validation_year": int(year), **asdict(fitted.audit)})
    if not predictions:
        raise ValueError("no expert folds were produced")
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(audits)


def forward_simplex_predictions(
    expert_oof: pd.DataFrame,
    *,
    methods: tuple[str, ...] = EXPERT_METHODS,
    grid_step: float = SIMPLEX_GRID_STEP,
    tolerance_f: float = 0.01,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = merge_multiple_prediction_sources({method: expert_oof for method in methods})
    merged["validation_year"] = pd.to_datetime(merged["contract_date"]).dt.year
    output: list[pd.DataFrame] = []
    selections: list[dict[str, Any]] = []
    for year in sorted(merged["validation_year"].unique()):
        history = merged.loc[merged["validation_year"].lt(year)].copy()
        valid = merged.loc[merged["validation_year"].eq(year)].copy()
        if history.empty:
            continue
        scan = scan_simplex_weights(history, methods=methods, grid_step=grid_step)
        selected = select_simplex_weights(scan, methods=methods, mean_mae_tolerance_f=tolerance_f)
        weights = tuple(float(selected[f"{method}_weight"]) for method in methods)
        part = blend_simplex_predictions(valid, methods=methods, weights=weights, method="four_expert_simplex_blend")
        part["weight_training_through_year"] = int(year) - 1
        output.append(part)
        selections.append({"validation_year": int(year), **selected.to_dict()})
    if not output:
        raise ValueError("at least two expert OOF years are required for forward blend weights")
    return pd.concat(output, ignore_index=True), pd.DataFrame(selections)


def select_frozen_weights(
    expert_oof: pd.DataFrame,
    *,
    methods: tuple[str, ...] = EXPERT_METHODS,
    grid_step: float = SIMPLEX_GRID_STEP,
) -> pd.Series:
    merged = merge_multiple_prediction_sources({method: expert_oof for method in methods})
    scan = scan_simplex_weights(merged, methods=methods, grid_step=grid_step)
    return select_simplex_weights(scan, methods=methods, mean_mae_tolerance_f=0.01)


def fit_final_experts(frame: pd.DataFrame, *, through_year: int = 2025, **fit_kwargs: Any) -> dict[str, FittedExpert]:
    dates = pd.to_datetime(frame["contract_date"], errors="coerce")
    training = frame.loc[dates.dt.year.le(through_year)].copy()
    return {method: fit_expert(training, method, **fit_kwargs) for method in EXPERT_METHODS}


def validate_frozen_feature_contract(frame: pd.DataFrame, expert: FittedExpert) -> None:
    missing = [name for name in expert.feature_names if name not in frame]
    if missing:
        raise ValueError(f"frozen feature contract is missing columns: {missing}")
    rates = frame.loc[:, expert.feature_names].apply(pd.to_numeric, errors="coerce").isna().mean()
    failed = {name: float(rate) for name, rate in rates.items() if float(rate) > MISSINGNESS_LIMIT}
    if failed:
        raise ValueError(f"frozen feature contract exceeds 3% missingness: {failed}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_point_bundle(
    output_dir: Path,
    *,
    station_id: str,
    model_version: str,
    experts: Mapping[str, FittedExpert],
    weights: Mapping[str, float],
    station_contract: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    chronology: Mapping[str, Any],
) -> tuple[Path, Path]:
    import joblib

    if tuple(experts) != EXPERT_METHODS:
        raise ValueError("point bundle experts must follow the four-expert contract order")
    weight_vector = np.asarray([weights[method] for method in EXPERT_METHODS], dtype=float)
    if (weight_vector < 0).any() or not np.isclose(weight_vector.sum(), 1.0):
        raise ValueError("simplex weights must be non-negative and sum to one")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    bundle = {
        "schema_version": 1,
        "artifact_type": "station_expert_ensemble_point_model",
        "station_id": station_id,
        "model_version": model_version,
        "research_only": True,
        "promotion_approved": False,
        "live_refit_enabled": False,
        "expert_methods": list(EXPERT_METHODS),
        "experts": dict(experts),
        "expert_feature_contracts": {method: list(experts[method].feature_names) for method in EXPERT_METHODS},
        "expert_missingness_audits": {method: asdict(experts[method].audit) for method in EXPERT_METHODS},
        "target_transforms": {method: experts[method].target_transform for method in EXPERT_METHODS},
        "simplex_weights": {method: float(weights[method]) for method in EXPERT_METHODS},
        "station_contract": dict(station_contract),
        "source_identity": dict(source_identity),
        "chronology": dict(chronology),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    bundle_path = output / f"{station_id}_{model_version}.joblib"
    manifest_path = output / f"{station_id}_{model_version}.json"
    joblib.dump(bundle, bundle_path)
    manifest = {key: value for key, value in bundle.items() if key != "experts"}
    manifest["artifact_integrity"] = {"bundle_sha256": sha256_file(bundle_path)}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    if sha256_file(bundle_path) != manifest["artifact_integrity"]["bundle_sha256"]:
        raise AssertionError("point bundle SHA-256 verification failed")
    return bundle_path, manifest_path
