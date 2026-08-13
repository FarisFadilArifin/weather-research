from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .round_override import build_round_override_frame
from .win_classifier import KATL_PEAK_FEATURES, expected_calibration_error


ROLLING_WINDOWS = (180, 365)
POLICY_FOLDS = 3
V4_COMMON_FEATURES = (
    "prediction_fraction_f",
    "floor_degree_is_odd",
    "alternative_round_direction",
    "models_supporting_default_bucket",
    "models_supporting_alternative_bucket",
    "base_mean_supports_alternative_f",
    "provider_mean_supports_alternative_f",
    "recent_bias_supports_alternative_7d_f",
    "recent_bias_supports_alternative_30d_f",
    "base_prediction_spread_f",
    "provider_spread_high_f",
    "point_minus_observed_temp_f",
    "point_minus_observed_high_f",
    "observed_temp_change_last_1h_f",
    "observed_temp_change_last_3h_f",
    "observed_morning_warmup_rate_f_per_hour",
    "day_of_year_sin",
    "day_of_year_cos",
    "prior_residual_mean_180d_f",
    "prior_residual_std_180d_f",
    "continuous_default_bucket_probability_180d",
    "continuous_alternative_bucket_probability_180d",
    "continuous_alternative_probability_advantage_180d",
    "prior_residual_mean_365d_f",
    "prior_residual_std_365d_f",
    "continuous_default_bucket_probability_365d",
    "continuous_alternative_bucket_probability_365d",
    "continuous_alternative_probability_advantage_365d",
)
V4_PEAK_FEATURES = (
    "v4_adjusted_mean_supports_alternative_f",
    "v4_hrrr_adjusted_supports_alternative_f",
    "v4_nbm_adjusted_supports_alternative_f",
    "v20_adjusted_high_spread_f",
    "v20_peak_hour_difference",
    "v20_solar_energy_11_14_wh_m2",
    "v20_solar_energy_15_18_wh_m2",
    "v20_rain_present_11_18",
)


@dataclass(frozen=True)
class HeadSpec:
    family: str
    model_params: Mapping[str, Any]

    @property
    def key(self) -> str:
        params = json.dumps(dict(self.model_params), sort_keys=True, separators=(",", ":"))
        return f"{self.family}|{params}"


def default_head_specs() -> list[HeadSpec]:
    return [
        HeadSpec("logistic", {"C": 0.03}),
        HeadSpec("logistic", {"C": 0.1}),
        HeadSpec("logistic", {"C": 0.3}),
        HeadSpec(
            "catboost",
            {
                "max_depth": 2,
                "max_iter": 150,
                "learning_rate": 0.03,
                "l2_regularization": 15.0,
            },
        ),
    ]


def utility_feature_names(*, include_peak_features: bool) -> list[str]:
    return [*V4_COMMON_FEATURES, *(V4_PEAK_FEATURES if include_peak_features else ())]


def build_utility_override_frame(
    feature_frame: pd.DataFrame,
    point_predictions: pd.DataFrame,
    base_validation_predictions: pd.DataFrame,
    *,
    include_peak_features: bool,
) -> pd.DataFrame:
    """Build V4 labels and strictly prior residual-distribution features."""
    frame = build_round_override_frame(
        feature_frame,
        point_predictions,
        base_validation_predictions,
        include_peak_features=include_peak_features,
    ).sort_values("contract_date", ignore_index=True)
    frame["recovery_target"] = frame["override_target"].astype(int)
    frame["damage_target"] = (
        frame["override_actionable"].eq(1)
        & frame["default_bucket_win"].eq(1)
        & frame["alternative_bucket_win"].eq(0)
    ).astype(int)
    frame["realized_override_utility"] = (
        frame["recovery_target"] - frame["damage_target"]
    ).astype(int)

    residual = (
        pd.to_numeric(frame["actual_high_f"], errors="coerce")
        - pd.to_numeric(frame["point_prediction_f"], errors="coerce")
    )
    prior_residual = residual.shift(1)
    for window in ROLLING_WINDOWS:
        rolling = prior_residual.rolling(window, min_periods=30)
        mean = rolling.mean()
        std = rolling.std(ddof=0).clip(lower=0.20)
        frame[f"prior_residual_mean_{window}d_f"] = mean
        frame[f"prior_residual_std_{window}d_f"] = std
        default_probability = _rolling_bucket_probability(
            frame["point_prediction_f"], frame["default_bucket_label"], mean, std
        )
        alternative_probability = _rolling_bucket_probability(
            frame["point_prediction_f"], frame["alternative_bucket_label"], mean, std
        )
        frame[f"continuous_default_bucket_probability_{window}d"] = default_probability
        frame[f"continuous_alternative_bucket_probability_{window}d"] = alternative_probability
        frame[f"continuous_alternative_probability_advantage_{window}d"] = (
            alternative_probability - default_probability
        )

    if include_peak_features:
        direction = pd.to_numeric(frame["alternative_round_direction"], errors="coerce")
        point = pd.to_numeric(frame["point_prediction_f"], errors="coerce")
        for source, target in (
            ("v20_adjusted_high_mean_f", "v4_adjusted_mean_supports_alternative_f"),
            ("v20_hrrr_observation_adjusted_high_f", "v4_hrrr_adjusted_supports_alternative_f"),
            ("v20_nbm_observation_adjusted_high_f", "v4_nbm_adjusted_supports_alternative_f"),
        ):
            frame[target] = direction * (pd.to_numeric(frame.get(source), errors="coerce") - point)

    for name in utility_feature_names(include_peak_features=include_peak_features):
        if name not in frame:
            frame[name] = np.nan
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    return frame


def _rolling_bucket_probability(
    point: pd.Series,
    labels: pd.Series,
    residual_mean: pd.Series,
    residual_std: pd.Series,
) -> pd.Series:
    from scipy.stats import t

    lows = pd.to_numeric(labels.astype(str).str.split("-").str[0], errors="coerce")
    lower_residual = lows - 0.5 - pd.to_numeric(point, errors="coerce")
    upper_residual = lows + 1.5 - pd.to_numeric(point, errors="coerce")
    # Preserve the rolling residual standard deviation under a Student-t(df=5) model.
    scale = pd.to_numeric(residual_std, errors="coerce") * math.sqrt(3.0 / 5.0)
    lower_z = (lower_residual - residual_mean) / scale
    upper_z = (upper_residual - residual_mean) / scale
    probability = t.cdf(upper_z, df=5) - t.cdf(lower_z, df=5)
    return pd.Series(np.clip(probability, 0.0, 1.0), index=point.index, dtype=float)


def fit_utility_override_system(
    frame: pd.DataFrame,
    *,
    station_id: str,
    include_peak_features: bool,
    candidate_specs: Sequence[HeadSpec] | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    station = station_id.strip().upper()
    if station == "KDAL" and include_peak_features:
        raise ValueError("KDAL V20 no-peak V4 model cannot include peak features")
    specs = list(candidate_specs or default_head_specs())
    feature_names = utility_feature_names(include_peak_features=include_peak_features)
    development = frame.loc[frame["year"].between(2023, 2025)].copy()
    prediction_parts: list[pd.DataFrame] = []
    tuning_parts: list[pd.DataFrame] = []
    policy_parts: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []

    for year in (2024, 2025):
        history = development.loc[development["year"].lt(year)]
        validation = development.loc[development["year"].eq(year)]
        state, tuning, policy = _fit_from_history(
            history,
            feature_names=feature_names,
            candidate_specs=specs,
            random_state=random_state,
        )
        predicted = predict_utility_override(state, validation)
        predicted["validation_year"] = year
        for key in (
            "model_training_cutoff",
            "calibration_start",
            "calibration_cutoff",
            "policy_start",
            "policy_cutoff",
        ):
            predicted[key] = state[key]
        prediction_parts.append(predicted)
        tuning["outer_validation_year"] = year
        policy["outer_validation_year"] = year
        tuning_parts.append(tuning)
        policy_parts.append(policy)
        fold_rows.append(_state_summary(state, validation_year=year))

    forward = pd.concat(prediction_parts, ignore_index=True).sort_values(
        "contract_date", ignore_index=True
    )
    metric_rows = [
        {
            "period": str(year),
            **utility_override_metrics(forward.loc[forward["validation_year"].eq(year)]),
        }
        for year in (2024, 2025)
    ]
    metric_rows.append({"period": "2024-2025", **utility_override_metrics(forward)})

    final_state, final_tuning, final_policy = _fit_from_history(
        development,
        feature_names=feature_names,
        candidate_specs=specs,
        random_state=random_state,
    )
    return {
        "station_id": station,
        "feature_profile": "compact_peak_augmented" if include_peak_features else "compact_no_peak",
        "feature_names": feature_names,
        "forward_predictions": forward,
        "forward_metrics": pd.DataFrame(metric_rows),
        "fold_states": pd.DataFrame(fold_rows),
        "tuning": pd.concat(tuning_parts, ignore_index=True),
        "policy_tuning": pd.concat(policy_parts, ignore_index=True),
        "final_state": final_state,
        "final_tuning": final_tuning,
        "final_policy_tuning": final_policy,
    }


def _fit_from_history(
    history: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    candidate_specs: Sequence[HeadSpec],
    random_state: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    core, inner, calibration, policy = _ordered_windows(history)
    states: dict[str, Any] = {}
    tuning_parts = []
    for head, target in (("recovery", "recovery_target"), ("damage", "damage_target")):
        selected, tuning = _select_head(
            core,
            inner,
            feature_names=feature_names,
            target=target,
            candidate_specs=candidate_specs,
            random_state=random_state,
        )
        refit = pd.concat([core, inner], ignore_index=True)
        model = _fit_head_model(
            refit.loc[refit["override_actionable"].eq(1)],
            feature_names,
            target,
            selected,
            random_state=random_state,
        )
        calibration_actionable = calibration.loc[calibration["override_actionable"].eq(1)]
        raw = model.predict_proba(calibration_actionable[list(feature_names)])[:, 1]
        calibrator = _fit_head_calibrator(raw, calibration_actionable[target].to_numpy(dtype=int))
        states[head] = {"model": model, "calibrator": calibrator, "selected_spec": selected}
        tuning["head"] = head
        tuning_parts.append(tuning)

    policy_probability = _predict_head_probabilities(states, policy, feature_names)
    policy_source = policy.copy()
    policy_source["recovery_probability"] = policy_probability["recovery"]
    policy_source["damage_probability"] = policy_probability["damage"]
    decision, policy_table = tune_utility_policy(policy_source)
    return {
        "feature_names": list(feature_names),
        "recovery": states["recovery"],
        "damage": states["damage"],
        **decision,
        "model_training_cutoff": pd.concat([core, inner])["contract_date"].max(),
        "calibration_start": calibration["contract_date"].min(),
        "calibration_cutoff": calibration["contract_date"].max(),
        "policy_start": policy["contract_date"].min(),
        "policy_cutoff": policy["contract_date"].max(),
        "training_actionable_rows": int(pd.concat([core, inner])["override_actionable"].sum()),
        "training_non_actionable_rows": 0,
    }, pd.concat(tuning_parts, ignore_index=True), policy_table


def _select_head(
    core: pd.DataFrame,
    inner: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    target: str,
    candidate_specs: Sequence[HeadSpec],
    random_state: int,
) -> tuple[HeadSpec, pd.DataFrame]:
    train = core.loc[core["override_actionable"].eq(1)]
    valid = inner.loc[inner["override_actionable"].eq(1)]
    rows = []
    scored = []
    for spec in candidate_specs:
        model = _fit_head_model(train, feature_names, target, spec, random_state=random_state)
        probability = model.predict_proba(valid[list(feature_names)])[:, 1]
        score = _binary_score(valid[target].to_numpy(dtype=int), probability)
        rows.append(
            {
                "candidate_key": spec.key,
                "family": spec.family,
                "model_params_json": json.dumps(dict(spec.model_params), sort_keys=True),
                "train_actionable_rows": len(train),
                "inner_actionable_rows": len(valid),
                **score,
            }
        )
        scored.append((spec, score))
    selected = sorted(
        scored,
        key=lambda item: (
            item[1]["log_loss"],
            item[1]["brier"],
            0 if item[0].family == "logistic" else 1,
        ),
    )[0][0]
    return selected, pd.DataFrame(rows)


def _fit_head_model(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    target: str,
    spec: HeadSpec,
    *,
    random_state: int,
) -> Any:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if frame["override_actionable"].ne(1).any():
        raise AssertionError("V4 heads may only fit actionable rows")
    if frame[target].nunique() < 2:
        raise ValueError(f"{target} needs both classes in the fit window")
    if spec.family == "logistic":
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=float(spec.model_params["C"]),
                        max_iter=2000,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    elif spec.family == "catboost":
        from catboost import CatBoostClassifier

        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                (
                    "classifier",
                    CatBoostClassifier(
                        depth=int(spec.model_params["max_depth"]),
                        iterations=int(spec.model_params["max_iter"]),
                        learning_rate=float(spec.model_params["learning_rate"]),
                        l2_leaf_reg=float(spec.model_params["l2_regularization"]),
                        loss_function="Logloss",
                        random_state=random_state,
                        verbose=False,
                        allow_writing_files=False,
                        thread_count=1,
                    ),
                ),
            ]
        )
    else:
        raise ValueError(f"unsupported V4 family: {spec.family}")
    model.fit(frame[list(feature_names)], frame[target].astype(int))
    return model


def _fit_head_calibrator(raw: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression

    y = np.asarray(actual, dtype=int)
    if len(np.unique(y)) < 2:
        return {"constant": float(y.mean())}
    clipped = np.clip(np.asarray(raw, dtype=float), 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1.0, max_iter=1000).fit(logits, y)
    return {"model": model}


def _apply_head_calibrator(state: Mapping[str, Any], raw: np.ndarray) -> np.ndarray:
    if "constant" in state:
        return np.full(len(raw), float(state["constant"]), dtype=float)
    clipped = np.clip(np.asarray(raw, dtype=float), 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    return np.clip(state["model"].predict_proba(logits)[:, 1], 1e-6, 1 - 1e-6)


def _predict_head_probabilities(
    states: Mapping[str, Any], frame: pd.DataFrame, feature_names: Sequence[str]
) -> dict[str, np.ndarray]:
    actionable = frame["override_actionable"].eq(1).to_numpy()
    output = {}
    for head in ("recovery", "damage"):
        raw = states[head]["model"].predict_proba(frame[list(feature_names)])[:, 1]
        probability = _apply_head_calibrator(states[head]["calibrator"], raw)
        output[head] = np.where(actionable, probability, 0.0)
    return output


def tune_utility_policy(policy: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    """Tune only on prior policy data and require stability across three time folds."""
    ordered = policy.sort_values("contract_date").copy()
    ordered["policy_fold"] = np.floor(
        np.arange(len(ordered)) * POLICY_FOLDS / max(len(ordered), 1)
    ).astype(int).clip(0, POLICY_FOLDS - 1)
    rows = []
    minimum_probabilities = np.round(np.arange(0.05, 0.951, 0.025), 3)
    margins = np.round(np.arange(0.0, 0.301, 0.05), 3)
    for damage_penalty in (2.0, 3.0, 4.0):
        score = ordered["recovery_probability"] - damage_penalty * ordered["damage_probability"]
        for minimum_probability in minimum_probabilities:
            for minimum_margin in margins:
                override = (
                    ordered["override_actionable"].eq(1)
                    & ordered["recovery_probability"].ge(minimum_probability)
                    & score.gt(minimum_margin)
                )
                recovered = int((override & ordered["recovery_target"].eq(1)).sum())
                damaged = int((override & ordered["damage_target"].eq(1)).sum())
                fold_nets = []
                for fold in range(POLICY_FOLDS):
                    mask = ordered["policy_fold"].eq(fold)
                    fold_nets.append(
                        int((override & mask & ordered["recovery_target"].eq(1)).sum())
                        - int((override & mask & ordered["damage_target"].eq(1)).sum())
                    )
                positive_folds = sum(value > 0 for value in fold_nets)
                negative_folds = sum(value < 0 for value in fold_nets)
                net = recovered - damaged
                fixed_cost_utility = recovered - 2 * damaged
                eligible = (
                    fixed_cost_utility > 0
                    and positive_folds >= 2
                    and negative_folds == 0
                    and int(override.sum()) >= 3
                )
                rows.append(
                    {
                        "damage_penalty": damage_penalty,
                        "minimum_recovery_probability": float(minimum_probability),
                        "minimum_utility_margin": float(minimum_margin),
                        "override_count": int(override.sum()),
                        "recovered_losses": recovered,
                        "damaged_wins": damaged,
                        "net_recovered_wins": net,
                        "fixed_cost_utility": fixed_cost_utility,
                        "positive_policy_folds": positive_folds,
                        "negative_policy_folds": negative_folds,
                        "policy_fold_nets_json": json.dumps(fold_nets),
                        "eligible": eligible,
                    }
                )
    table = pd.DataFrame(rows)
    eligible = table.loc[table["eligible"]].sort_values(
        [
            "fixed_cost_utility",
            "net_recovered_wins",
            "damaged_wins",
            "override_count",
            "minimum_recovery_probability",
            "minimum_utility_margin",
            "damage_penalty",
        ],
        ascending=[False, False, True, True, False, False, False],
    )
    if eligible.empty:
        return {
            "policy_enabled": False,
            "damage_penalty": 4.0,
            "minimum_recovery_probability": 1.1,
            "minimum_utility_margin": math.inf,
            "policy_stable": False,
        }, table
    selected = eligible.iloc[0]
    return {
        "policy_enabled": True,
        "damage_penalty": float(selected["damage_penalty"]),
        "minimum_recovery_probability": float(selected["minimum_recovery_probability"]),
        "minimum_utility_margin": float(selected["minimum_utility_margin"]),
        "policy_stable": True,
    }, table


def predict_utility_override(state: Mapping[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    probability = _predict_head_probabilities(state, frame, state["feature_names"])
    columns = [
        "contract_date",
        "actual_high_f",
        "actual_bucket_label",
        "point_prediction_f",
        "default_degree_f",
        "default_bucket_label",
        "alternative_degree_f",
        "alternative_bucket_label",
        "override_actionable",
        "default_bucket_win",
        "alternative_bucket_win",
        "recovery_target",
        "damage_target",
        "realized_override_utility",
    ]
    out = frame[columns].copy()
    out["recovery_probability"] = probability["recovery"]
    out["damage_probability"] = probability["damage"]
    out["expected_utility"] = (
        out["recovery_probability"] - float(state["damage_penalty"]) * out["damage_probability"]
    )
    out["policy_enabled"] = bool(state["policy_enabled"])
    out["damage_penalty"] = float(state["damage_penalty"])
    out["minimum_recovery_probability"] = float(state["minimum_recovery_probability"])
    out["minimum_utility_margin"] = float(state["minimum_utility_margin"])
    out["override"] = (
        bool(state["policy_enabled"])
        & out["override_actionable"].eq(1)
        & out["recovery_probability"].ge(float(state["minimum_recovery_probability"]))
        & out["expected_utility"].gt(float(state["minimum_utility_margin"]))
    )
    out["final_degree_f"] = np.where(
        out["override"], out["alternative_degree_f"], out["default_degree_f"]
    ).astype(int)
    out["final_bucket_label"] = np.where(
        out["override"], out["alternative_bucket_label"], out["default_bucket_label"]
    )
    out["final_bucket_win"] = out["final_bucket_label"].eq(out["actual_bucket_label"]).astype(int)
    out["recovered_loss"] = (out["override"] & out["recovery_target"].eq(1)).astype(int)
    out["damaged_win"] = (out["override"] & out["damage_target"].eq(1)).astype(int)
    return out


def utility_override_metrics(predictions: pd.DataFrame) -> dict[str, Any]:
    actionable = predictions.loc[predictions["override_actionable"].eq(1)]
    output: dict[str, Any] = {
        "count": int(len(predictions)),
        "actionable_count": int(len(actionable)),
        "actionable_rate": float(len(actionable) / len(predictions)),
        "default_bucket_hit_rate": float(predictions["default_bucket_win"].mean()),
        "final_bucket_hit_rate": float(predictions["final_bucket_win"].mean()),
        "bucket_hit_rate_lift": float(
            predictions["final_bucket_win"].mean() - predictions["default_bucket_win"].mean()
        ),
        "override_count": int(predictions["override"].sum()),
        "override_rate": float(predictions["override"].mean()),
        "recovered_losses": int(predictions["recovered_loss"].sum()),
        "damaged_wins": int(predictions["damaged_win"].sum()),
    }
    output["net_recovered_wins"] = output["recovered_losses"] - output["damaged_wins"]
    output["recovery_damage_ratio"] = (
        float(output["recovered_losses"] / output["damaged_wins"])
        if output["damaged_wins"]
        else (math.inf if output["recovered_losses"] else math.nan)
    )
    for head, target in (("recovery", "recovery_target"), ("damage", "damage_target")):
        output.update(
            {
                f"{head}_{key}": value
                for key, value in _probability_metrics(
                    actionable[target].to_numpy(dtype=int),
                    actionable[f"{head}_probability"].to_numpy(dtype=float),
                ).items()
            }
        )
    return output


def _probability_metrics(actual: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = np.asarray(actual, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-12, 1 - 1e-12)
    return {
        "target_rate": float(y.mean()),
        "mean_probability": float(p.mean()),
        "log_loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
        "brier": float(np.mean((p - y) ** 2)),
        "ece": expected_calibration_error(y, p),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else math.nan,
        "pr_auc": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else math.nan,
    }


def _binary_score(actual: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    y = np.asarray(actual, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-12, 1 - 1e-12)
    return {
        "count": int(len(y)),
        "positive_rate": float(y.mean()),
        "log_loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
        "brier": float(np.mean((p - y) ** 2)),
    }


def _ordered_windows(
    history: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = history.sort_values("contract_date")
    end = ordered["contract_date"].max()
    policy_start = end - pd.Timedelta(days=89)
    calibration_start = policy_start - pd.Timedelta(days=90)
    inner_start = calibration_start - pd.Timedelta(days=60)
    core = ordered.loc[ordered["contract_date"].lt(inner_start)]
    inner = ordered.loc[
        ordered["contract_date"].ge(inner_start)
        & ordered["contract_date"].lt(calibration_start)
    ]
    calibration = ordered.loc[
        ordered["contract_date"].ge(calibration_start)
        & ordered["contract_date"].lt(policy_start)
    ]
    policy = ordered.loc[ordered["contract_date"].ge(policy_start)]
    if len(core) < 120 or min(len(inner), len(calibration), len(policy)) < 50:
        raise ValueError("insufficient history for V4 chronological windows")
    for name, part in (("core", core), ("inner", inner), ("calibration", calibration), ("policy", policy)):
        if int(part["override_actionable"].sum()) < 20:
            raise ValueError(f"insufficient actionable rows in V4 {name} window")
    return core, inner, calibration, policy


def _state_summary(state: Mapping[str, Any], *, validation_year: int) -> dict[str, Any]:
    return {
        "validation_year": validation_year,
        "recovery_family": state["recovery"]["selected_spec"].family,
        "recovery_model_params_json": json.dumps(
            dict(state["recovery"]["selected_spec"].model_params), sort_keys=True
        ),
        "damage_family": state["damage"]["selected_spec"].family,
        "damage_model_params_json": json.dumps(
            dict(state["damage"]["selected_spec"].model_params), sort_keys=True
        ),
        "policy_enabled": state["policy_enabled"],
        "policy_stable": state["policy_stable"],
        "damage_penalty": state["damage_penalty"],
        "minimum_recovery_probability": state["minimum_recovery_probability"],
        "minimum_utility_margin": state["minimum_utility_margin"],
        "training_actionable_rows": state["training_actionable_rows"],
        "training_non_actionable_rows": state["training_non_actionable_rows"],
        "model_training_cutoff": state["model_training_cutoff"],
        "calibration_start": state["calibration_start"],
        "calibration_cutoff": state["calibration_cutoff"],
        "policy_start": state["policy_start"],
        "policy_cutoff": state["policy_cutoff"],
    }


def audit_utility_override_system(
    source_frame: pd.DataFrame,
    result: Mapping[str, Any],
    *,
    include_peak_features: bool,
) -> pd.DataFrame:
    forward = result["forward_predictions"]
    expected_recovery = (
        source_frame["override_actionable"].eq(1)
        & source_frame["alternative_bucket_win"].eq(1)
        & source_frame["default_bucket_win"].eq(0)
    ).astype(int)
    expected_damage = (
        source_frame["override_actionable"].eq(1)
        & source_frame["default_bucket_win"].eq(1)
        & source_frame["alternative_bucket_win"].eq(0)
    ).astype(int)
    chronology = (
        pd.to_datetime(forward["model_training_cutoff"])
        < pd.to_datetime(forward["calibration_start"])
    ).all() and (
        pd.to_datetime(forward["calibration_cutoff"])
        < pd.to_datetime(forward["policy_start"])
    ).all() and (
        pd.to_datetime(forward["policy_cutoff"])
        < pd.to_datetime(forward["contract_date"])
    ).all()
    accounting = (
        int(forward["final_bucket_win"].sum()) - int(forward["default_bucket_win"].sum())
        == int(forward["recovered_loss"].sum()) - int(forward["damaged_win"].sum())
    )
    probability_columns = [
        f"continuous_{bucket}_bucket_probability_{window}d"
        for window in ROLLING_WINDOWS
        for bucket in ("default", "alternative")
    ]
    probability_values = source_frame[probability_columns].stack().dropna()
    rolling_correct = True
    residual = source_frame["actual_high_f"] - source_frame["point_prediction_f"]
    for window in ROLLING_WINDOWS:
        shifted = residual.shift(1).rolling(window, min_periods=30)
        rolling_correct &= np.allclose(
            source_frame[f"prior_residual_mean_{window}d_f"],
            shifted.mean(),
            equal_nan=True,
        )
        rolling_correct &= np.allclose(
            source_frame[f"prior_residual_std_{window}d_f"],
            shifted.std(ddof=0).clip(lower=0.20),
            equal_nan=True,
        )
    fold_states = result["fold_states"]
    policy_grid = result["policy_tuning"]
    policy_selection_consistent = True
    for _, state in fold_states.iterrows():
        year_grid = policy_grid.loc[
            policy_grid["outer_validation_year"].eq(state["validation_year"])
        ]
        if bool(state["policy_enabled"]):
            chosen = year_grid.loc[
                np.isclose(year_grid["damage_penalty"], state["damage_penalty"])
                & np.isclose(
                    year_grid["minimum_recovery_probability"],
                    state["minimum_recovery_probability"],
                )
                & np.isclose(
                    year_grid["minimum_utility_margin"],
                    state["minimum_utility_margin"],
                )
            ]
            policy_selection_consistent &= len(chosen) == 1 and bool(chosen.iloc[0]["eligible"])
        else:
            policy_selection_consistent &= not year_grid["eligible"].any()
    final_state = result["final_state"]
    final_grid = result["final_policy_tuning"]
    if bool(final_state["policy_enabled"]):
        final_chosen = final_grid.loc[
            np.isclose(final_grid["damage_penalty"], final_state["damage_penalty"])
            & np.isclose(
                final_grid["minimum_recovery_probability"],
                final_state["minimum_recovery_probability"],
            )
            & np.isclose(
                final_grid["minimum_utility_margin"],
                final_state["minimum_utility_margin"],
            )
        ]
        final_policy_consistent = len(final_chosen) == 1 and bool(final_chosen.iloc[0]["eligible"])
    else:
        final_policy_consistent = not final_grid["eligible"].any()
    window_lengths_valid = (
        (
            pd.to_datetime(forward["calibration_cutoff"])
            - pd.to_datetime(forward["calibration_start"])
        ).dt.days.ge(89).all()
        and (
            pd.to_datetime(forward["policy_cutoff"])
            - pd.to_datetime(forward["policy_start"])
        ).dt.days.ge(89).all()
    )
    probability_advantage_correct = all(
        np.allclose(
            source_frame[f"continuous_alternative_probability_advantage_{window}d"],
            source_frame[f"continuous_alternative_bucket_probability_{window}d"]
            - source_frame[f"continuous_default_bucket_probability_{window}d"],
            equal_nan=True,
        )
        for window in ROLLING_WINDOWS
    )
    rows = [
        ("recovery_target_formula", expected_recovery.eq(source_frame["recovery_target"]).all(), "alternative wins and half-up loses"),
        ("damage_target_formula", expected_damage.eq(source_frame["damage_target"]).all(), "half-up wins and alternative loses"),
        ("targets_are_mutually_exclusive", ~(source_frame["recovery_target"].eq(1) & source_frame["damage_target"].eq(1)).any(), "a row cannot recover and damage simultaneously"),
        ("non_actionable_targets_are_zero", source_frame.loc[source_frame["override_actionable"].eq(0), ["recovery_target", "damage_target"]].eq(0).all(axis=None), "non-actionable rows cannot create utility"),
        ("training_uses_actionable_rows_only", fold_states["training_non_actionable_rows"].eq(0).all() and final_state["training_non_actionable_rows"] == 0, f"outer_folds={len(fold_states)} plus final"),
        ("continuous_features_are_strictly_prior", rolling_correct, "rolling residuals use shift(1)"),
        ("continuous_probabilities_are_valid", probability_values.between(0, 1).all(), f"checked={len(probability_values)}"),
        ("continuous_probability_advantage_formula", probability_advantage_correct, "alternative probability minus default probability"),
        ("compact_feature_inventory", len(result["feature_names"]) <= 36, f"features={len(result['feature_names'])}"),
        ("all_outer_rows_scored", len(forward) == int(source_frame["year"].isin([2024, 2025]).sum()) and np.isfinite(forward[["recovery_probability", "damage_probability"]]).all(axis=None), f"outer_rows={len(forward)}"),
        ("utility_formula", np.allclose(forward["expected_utility"], forward["recovery_probability"] - forward["damage_penalty"] * forward["damage_probability"]), "p_recovery - penalty*p_damage"),
        ("override_only_when_actionable", (~forward["override"] | forward["override_actionable"].eq(1)).all(), "no meaningless override"),
        ("disabled_policy_abstains", (forward["policy_enabled"] | ~forward["override"]).all(), "disabled folds make zero overrides"),
        ("chronological_fit_calibration_policy_validation", chronology, "fit < 90d calibration < 90d policy < outer validation"),
        ("long_calibration_and_policy_windows", window_lengths_valid, "calibration and policy windows are each 90 days"),
        ("full_probability_threshold_grid", policy_grid["minimum_recovery_probability"].min() <= 0.05 and policy_grid["minimum_recovery_probability"].max() >= 0.95, "grid covers 0.05 through 0.95"),
        ("three_policy_stability_folds", policy_grid["policy_fold_nets_json"].map(lambda value: len(json.loads(value)) == POLICY_FOLDS).all(), f"folds={POLICY_FOLDS}"),
        ("selected_policy_is_stable_or_disabled", (fold_states["policy_enabled"].eq(fold_states["policy_stable"])).all(), "unstable policy defaults to half-up"),
        ("policy_selection_matches_eligibility", policy_selection_consistent, "enabled selection is eligible; disabled means no eligible candidate"),
        ("final_policy_matches_eligibility", final_policy_consistent, "exported policy independently matches final eligibility grid"),
        ("win_accounting_identity", accounting, "final-default equals recovered-damaged"),
        ("kdal_no_peak_contract", include_peak_features or set(result["feature_names"]).isdisjoint(set(KATL_PEAK_FEATURES) | set(V4_PEAK_FEATURES)), f"peak_features={include_peak_features}"),
    ]
    if "train_through_year" in source_frame:
        valid = source_frame.dropna(subset=["train_through_year", "year"])
        rows.append(("point_predictions_are_forward", (valid["train_through_year"] < valid["year"]).all(), f"checked_rows={len(valid)}"))
    return pd.DataFrame(
        [{"audit": name, "passed": bool(passed), "detail": detail} for name, passed, detail in rows]
    )


def serializable_utility_bundle(result: Mapping[str, Any]) -> dict[str, Any]:
    state = dict(result["final_state"])
    for head in ("recovery", "damage"):
        state[head] = dict(state[head])
        state[head]["selected_spec"] = asdict(state[head]["selected_spec"])
    return {
        "artifact_type": "station_half_up_utility_override_v4_research",
        "schema_version": 4,
        "station_id": result["station_id"],
        "feature_profile": result["feature_profile"],
        "feature_names": result["feature_names"],
        "decision_semantics": "override when calibrated recovery benefit exceeds penalized damage risk",
        "state": state,
        "forward_metrics": result["forward_metrics"].to_dict(orient="records"),
    }
