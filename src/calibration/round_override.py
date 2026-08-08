from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .bucket_probability import BASE_METHODS, canonical_two_degree_bucket, round_half_up
from .round_direction import build_round_direction_frame, round_feature_names
from .win_classifier import KATL_PEAK_FEATURES, expected_calibration_error


OVERRIDE_FEATURES = (
    "alternative_round_direction",
    "fraction_distance_to_half_f",
    "models_supporting_default_bucket",
    "models_supporting_alternative_bucket",
    "base_mean_supports_alternative_f",
    "provider_mean_supports_alternative_f",
    "recent_bias_supports_alternative_7d_f",
    "recent_bias_supports_alternative_30d_f",
    "prior_override_rate_30d",
    "prior_override_rate_90d",
)


@dataclass(frozen=True)
class OverrideSpec:
    family: str
    model_params: Mapping[str, Any]
    actionable_weight: float

    @property
    def key(self) -> str:
        params = json.dumps(dict(self.model_params), sort_keys=True, separators=(",", ":"))
        return f"{self.family}|{params}|weight={self.actionable_weight:g}"


def default_override_specs() -> list[OverrideSpec]:
    specs = []
    for weight in (1.0, 2.0, 4.0):
        for c in (0.03, 0.1, 1.0):
            specs.append(OverrideSpec("logistic", {"C": c}, weight))
        specs.extend(
            [
                OverrideSpec(
                    "catboost",
                    {
                        "max_depth": 2,
                        "max_iter": 150,
                        "learning_rate": 0.03,
                        "l2_regularization": 10.0,
                    },
                    weight,
                ),
                OverrideSpec(
                    "catboost",
                    {
                        "max_depth": 3,
                        "max_iter": 200,
                        "learning_rate": 0.02,
                        "l2_regularization": 20.0,
                    },
                    weight,
                ),
            ]
        )
    return specs


def override_feature_names(*, include_peak_features: bool) -> list[str]:
    return list(
        dict.fromkeys(
            [
                *round_feature_names(include_peak_features=include_peak_features),
                *OVERRIDE_FEATURES,
            ]
        )
    )


def build_round_override_frame(
    feature_frame: pd.DataFrame,
    point_predictions: pd.DataFrame,
    base_validation_predictions: pd.DataFrame,
    *,
    include_peak_features: bool,
) -> pd.DataFrame:
    frame = build_round_direction_frame(
        feature_frame,
        point_predictions,
        base_validation_predictions,
        include_peak_features=include_peak_features,
    )
    default_up = frame["default_half_up"].astype(int)
    alternative_degree = np.where(
        default_up.eq(1), frame["floor_degree_f"], frame["ceil_degree_f"]
    ).astype(int)
    frame["default_degree_f"] = frame["point_degree_f"].astype(int)
    frame["default_bucket_label"] = frame["point_bucket_label"]
    frame["alternative_degree_f"] = alternative_degree
    frame["alternative_bucket_label"] = pd.Series(
        alternative_degree, index=frame.index
    ).map(canonical_two_degree_bucket)
    frame["alternative_round_direction"] = np.where(default_up.eq(1), -1.0, 1.0)
    frame["override_actionable"] = frame["alternative_bucket_label"].ne(
        frame["default_bucket_label"]
    ).astype(int)
    frame["default_bucket_win"] = frame["default_bucket_label"].eq(
        frame["actual_bucket_label"]
    ).astype(int)
    frame["alternative_bucket_win"] = frame["alternative_bucket_label"].eq(
        frame["actual_bucket_label"]
    ).astype(int)
    frame["override_target"] = (
        frame["override_actionable"].eq(1)
        & frame["alternative_bucket_win"].eq(1)
        & frame["default_bucket_win"].eq(0)
    ).astype(int)
    frame["fraction_distance_to_half_f"] = (
        frame["prediction_fraction_f"] - 0.5
    ).abs()

    support_columns = [
        *(f"{name}_predicted_high_f" for name in BASE_METHODS),
        "gfs_high_f",
        "hrrr_high_f",
        "nbm_high_f",
    ]
    default_support = np.zeros(len(frame), dtype=float)
    alternative_support = np.zeros(len(frame), dtype=float)
    point = pd.to_numeric(frame["point_prediction_f"], errors="coerce")
    direction = pd.to_numeric(frame["alternative_round_direction"], errors="coerce")
    for column in support_columns:
        values = pd.to_numeric(frame.get(column), errors="coerce")
        degrees = values.map(
            lambda value: round_half_up(float(value)) if pd.notna(value) else np.nan
        )
        labels = degrees.map(
            lambda value: canonical_two_degree_bucket(int(value))
            if pd.notna(value)
            else None
        )
        default_support += labels.eq(frame["default_bucket_label"]).to_numpy(dtype=float)
        alternative_support += labels.eq(frame["alternative_bucket_label"]).to_numpy(dtype=float)
    frame["models_supporting_default_bucket"] = default_support
    frame["models_supporting_alternative_bucket"] = alternative_support
    frame["base_mean_supports_alternative_f"] = direction * (
        pd.to_numeric(frame["base_prediction_mean_f"], errors="coerce") - point
    )
    frame["provider_mean_supports_alternative_f"] = direction * (
        pd.to_numeric(frame["provider_mean_high_f"], errors="coerce") - point
    )
    frame["recent_bias_supports_alternative_7d_f"] = direction * pd.to_numeric(
        frame["prior_residual_bias_7d_f"], errors="coerce"
    )
    frame["recent_bias_supports_alternative_30d_f"] = direction * pd.to_numeric(
        frame["prior_residual_bias_30d_f"], errors="coerce"
    )
    ordered = frame.sort_values("contract_date")
    for window in (30, 90):
        frame.loc[ordered.index, f"prior_override_rate_{window}d"] = (
            ordered["override_target"]
            .shift(1)
            .rolling(window, min_periods=max(5, window // 5))
            .mean()
            .to_numpy()
        )
    for name in override_feature_names(include_peak_features=include_peak_features):
        if name not in frame:
            frame[name] = np.nan
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    return frame.sort_values("contract_date").reset_index(drop=True)


def fit_round_override_system(
    frame: pd.DataFrame,
    *,
    station_id: str,
    include_peak_features: bool,
    candidate_specs: Sequence[OverrideSpec] | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    station = station_id.strip().upper()
    if station == "KDAL" and include_peak_features:
        raise ValueError("KDAL V20 no-peak override model cannot include peak features")
    specs = list(candidate_specs or default_override_specs())
    feature_names = override_feature_names(include_peak_features=include_peak_features)
    development = frame.loc[frame["year"].between(2023, 2025)].copy()
    parts = []
    tuning_parts = []
    cutoff_parts = []
    fold_states = []
    for year in (2024, 2025):
        history = development.loc[development["year"].lt(year)]
        validation = development.loc[development["year"].eq(year)]
        state, tuning, thresholds = _fit_from_history(
            history,
            feature_names=feature_names,
            candidate_specs=specs,
            random_state=random_state,
        )
        predicted = predict_round_override(state, validation)
        predicted["validation_year"] = year
        for key in (
            "model_training_cutoff",
            "calibration_start",
            "calibration_cutoff",
            "policy_start",
            "policy_cutoff",
        ):
            predicted[key] = state[key]
        parts.append(predicted)
        tuning["outer_validation_year"] = year
        thresholds["outer_validation_year"] = year
        tuning_parts.append(tuning)
        cutoff_parts.append(thresholds)
        fold_states.append(
            {
                "validation_year": year,
                "selected_family": state["selected_spec"].family,
                "selected_model_params_json": json.dumps(
                    dict(state["selected_spec"].model_params), sort_keys=True
                ),
                "selected_actionable_weight": state["selected_spec"].actionable_weight,
                "override_threshold": state["override_threshold"],
                "model_training_cutoff": state["model_training_cutoff"],
                "calibration_start": state["calibration_start"],
                "calibration_cutoff": state["calibration_cutoff"],
                "policy_start": state["policy_start"],
                "policy_cutoff": state["policy_cutoff"],
            }
        )
    forward = pd.concat(parts, ignore_index=True).sort_values(
        "contract_date", ignore_index=True
    )
    metrics = []
    for year in (2024, 2025):
        metrics.append(
            {
                "period": str(year),
                **round_override_metrics(forward.loc[forward["validation_year"].eq(year)]),
            }
        )
    metrics.append({"period": "2024-2025", **round_override_metrics(forward)})
    final_state, final_tuning, final_thresholds = _fit_from_history(
        development,
        feature_names=feature_names,
        candidate_specs=specs,
        random_state=random_state,
    )
    return {
        "station_id": station,
        "feature_profile": "peak_augmented" if include_peak_features else "common_no_peak",
        "feature_names": feature_names,
        "forward_predictions": forward,
        "forward_metrics": pd.DataFrame(metrics),
        "fold_states": pd.DataFrame(fold_states),
        "tuning": pd.concat(tuning_parts, ignore_index=True),
        "threshold_tuning": pd.concat(cutoff_parts, ignore_index=True),
        "final_state": final_state,
        "final_tuning": final_tuning,
        "final_threshold_tuning": final_thresholds,
    }


def predict_round_override(state: Mapping[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    raw = state["model"].predict_proba(frame[state["feature_names"]])[:, 1]
    probability = _apply_calibrator(state["calibrator"], raw)
    out = frame[
        [
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
            "override_target",
        ]
    ].copy()
    out["override_probability"] = probability
    out["override_threshold"] = float(state["override_threshold"])
    out["override"] = (
        out["override_actionable"].eq(1)
        & out["override_probability"].ge(float(state["override_threshold"]))
    )
    out["final_degree_f"] = np.where(
        out["override"], out["alternative_degree_f"], out["default_degree_f"]
    ).astype(int)
    out["final_bucket_label"] = np.where(
        out["override"], out["alternative_bucket_label"], out["default_bucket_label"]
    )
    out["final_bucket_win"] = out["final_bucket_label"].eq(
        out["actual_bucket_label"]
    ).astype(int)
    out["recovered_loss"] = (
        out["override"] & out["alternative_bucket_win"].eq(1)
    ).astype(int)
    out["damaged_win"] = (
        out["override"] & out["default_bucket_win"].eq(1)
    ).astype(int)
    return out


def round_override_metrics(predictions: pd.DataFrame) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = predictions["override_target"].to_numpy(dtype=int)
    p = np.clip(predictions["override_probability"].to_numpy(dtype=float), 1e-12, 1 - 1e-12)
    overrides = predictions["override"].astype(bool)
    recovered = int(predictions["recovered_loss"].sum())
    damaged = int(predictions["damaged_win"].sum())
    return {
        "count": int(len(predictions)),
        "override_target_rate": float(y.mean()),
        "mean_override_probability": float(p.mean()),
        "binary_log_loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
        "brier": float(np.mean((p - y) ** 2)),
        "ece": expected_calibration_error(y, p),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else math.nan,
        "pr_auc": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else math.nan,
        "actionable_count": int(predictions["override_actionable"].sum()),
        "actionable_rate": float(predictions["override_actionable"].mean()),
        "default_bucket_hit_rate": float(predictions["default_bucket_win"].mean()),
        "final_bucket_hit_rate": float(predictions["final_bucket_win"].mean()),
        "bucket_hit_rate_lift": float(
            predictions["final_bucket_win"].mean()
            - predictions["default_bucket_win"].mean()
        ),
        "override_count": int(overrides.sum()),
        "override_rate": float(overrides.mean()),
        "override_precision": float(
            predictions.loc[overrides, "override_target"].mean()
        )
        if overrides.any()
        else math.nan,
        "override_recall": float(
            predictions.loc[predictions["override_target"].eq(1), "override"].mean()
        )
        if y.sum()
        else math.nan,
        "recovered_losses": recovered,
        "damaged_wins": damaged,
        "net_recovered_wins": recovered - damaged,
        "recovery_damage_ratio": float(recovered / damaged)
        if damaged
        else (math.inf if recovered else math.nan),
    }


def continuous_override_comparison(
    predictions: pd.DataFrame,
    continuous_predictions: pd.DataFrame,
    *,
    selected_family: str = "student_t",
) -> pd.DataFrame:
    continuous = continuous_predictions.copy()
    if "model_family" in continuous:
        continuous = continuous.loc[continuous["model_family"].eq(selected_family)]
    continuous["contract_date"] = pd.to_datetime(continuous["contract_date"])
    continuous = continuous.drop_duplicates("contract_date")
    joined = predictions.merge(
        continuous[["contract_date", "bucket_probabilities_json"]],
        on="contract_date",
        how="inner",
        validate="one_to_one",
    )
    alternative_probability = []
    default_probability = []
    for _, row in joined.iterrows():
        probabilities = json.loads(row["bucket_probabilities_json"])
        alternative_probability.append(
            float(probabilities.get(row["alternative_bucket_label"], 0.0))
            if int(row["override_actionable"]) == 1
            else 0.0
        )
        default_probability.append(
            float(probabilities.get(row["default_bucket_label"], 0.0))
        )
    joined["override_probability"] = alternative_probability
    joined["override_threshold"] = 0.0
    joined["override"] = (
        joined["override_actionable"].eq(1)
        & pd.Series(alternative_probability, index=joined.index).gt(
            pd.Series(default_probability, index=joined.index)
        )
    )
    joined["final_bucket_label"] = np.where(
        joined["override"], joined["alternative_bucket_label"], joined["default_bucket_label"]
    )
    joined["final_bucket_win"] = joined["final_bucket_label"].eq(
        joined["actual_bucket_label"]
    ).astype(int)
    joined["recovered_loss"] = (
        joined["override"] & joined["alternative_bucket_win"].eq(1)
    ).astype(int)
    joined["damaged_win"] = (
        joined["override"] & joined["default_bucket_win"].eq(1)
    ).astype(int)
    return pd.DataFrame([round_override_metrics(joined)])


def audit_round_override_system(
    source_frame: pd.DataFrame,
    result: Mapping[str, Any],
    *,
    include_peak_features: bool,
) -> pd.DataFrame:
    forward = result["forward_predictions"]
    expected_alternative = np.where(
        source_frame["default_half_up"].eq(1),
        source_frame["floor_degree_f"],
        source_frame["ceil_degree_f"],
    ).astype(int)
    expected_actionable = source_frame["alternative_bucket_label"].ne(
        source_frame["default_bucket_label"]
    ).astype(int)
    expected_target = (
        expected_actionable.eq(1)
        & source_frame["alternative_bucket_label"].eq(source_frame["actual_bucket_label"])
        & source_frame["default_bucket_label"].ne(source_frame["actual_bucket_label"])
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
        int(forward["final_bucket_win"].sum())
        - int(forward["default_bucket_win"].sum())
        == int(forward["recovered_loss"].sum()) - int(forward["damaged_win"].sum())
    )
    rows = [
        ("default_is_half_up", np.array_equal(source_frame["default_degree_f"], source_frame["point_degree_f"]), "default degree equals honest point half-up degree"),
        ("alternative_is_opposite_floor_ceil", np.array_equal(expected_alternative, source_frame["alternative_degree_f"]), "opposite of default floor/ceil choice"),
        ("actionable_formula", expected_actionable.eq(source_frame["override_actionable"]).all(), "alternative and default buckets differ"),
        ("override_target_formula", expected_target.eq(source_frame["override_target"]).all(), "alternative wins and default loses"),
        ("non_actionable_targets_are_zero", source_frame.loc[source_frame["override_actionable"].eq(0), "override_target"].eq(0).all(), "same-bucket floor/ceil rows cannot request override"),
        ("all_outer_rows_scored", len(forward) == int(source_frame["year"].isin([2024, 2025]).sum()) and np.isfinite(forward["override_probability"]).all(), f"outer_rows={len(forward)}"),
        ("override_only_when_actionable", (~forward["override"] | forward["override_actionable"].eq(1)).all(), "no meaningless same-bucket overrides"),
        ("chronological_fit_calibration_policy_validation", chronology, "fit < calibration < policy < outer validation"),
        ("win_accounting_identity", accounting, "final-default equals recovered-damaged"),
        ("kdal_no_peak_contract", include_peak_features or set(result["feature_names"]).isdisjoint(KATL_PEAK_FEATURES), f"peak_overlap={len(set(result['feature_names']) & set(KATL_PEAK_FEATURES))}"),
        ("no_boundary_distance_window", "nearest_boundary_distance_f" not in result["feature_names"], "all rows are modeled; no eligibility window"),
    ]
    if "train_through_year" in source_frame:
        valid = source_frame.dropna(subset=["train_through_year", "year"])
        rows.append(("point_predictions_are_forward", (valid["train_through_year"] < valid["year"]).all(), f"checked_rows={len(valid)}"))
    return pd.DataFrame(
        [{"audit": name, "passed": bool(passed), "detail": detail} for name, passed, detail in rows]
    )


def serializable_override_bundle(result: Mapping[str, Any]) -> dict[str, Any]:
    state = dict(result["final_state"])
    state["selected_spec"] = asdict(state["selected_spec"])
    return {
        "artifact_type": "station_half_up_override_classifier_v3_research",
        "schema_version": 3,
        "station_id": result["station_id"],
        "feature_profile": result["feature_profile"],
        "feature_names": result["feature_names"],
        "class_semantics": {"0": "keep_half_up", "1": "override_to_opposite_floor_ceil"},
        "state": state,
        "forward_metrics": result["forward_metrics"].to_dict(orient="records"),
    }


def _fit_from_history(
    history: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    candidate_specs: Sequence[OverrideSpec],
    random_state: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    core, inner, calibration, policy = _ordered_windows(history)
    rows = []
    candidates = []
    for spec in candidate_specs:
        model = _fit_model(core, feature_names, spec, random_state=random_state)
        p = model.predict_proba(inner[feature_names])[:, 1]
        score = _binary_score(inner["override_target"].to_numpy(dtype=int), p)
        rows.append(
            {
                "candidate_key": spec.key,
                "family": spec.family,
                "model_params_json": json.dumps(dict(spec.model_params), sort_keys=True),
                "actionable_weight": spec.actionable_weight,
                **score,
            }
        )
        candidates.append((spec, score))
    selected = sorted(
        candidates,
        key=lambda item: (
            item[1]["log_loss"],
            item[1]["brier"],
            0 if item[0].family == "logistic" else 1,
            item[0].actionable_weight,
        ),
    )[0][0]
    refit = pd.concat([core, inner], ignore_index=True)
    model = _fit_model(refit, feature_names, selected, random_state=random_state)
    calibrator = _fit_calibrator(
        model.predict_proba(calibration[feature_names])[:, 1],
        calibration["override_target"].to_numpy(dtype=int),
    )
    base_state = {
        "feature_names": list(feature_names),
        "selected_spec": selected,
        "model": model,
        "calibrator": calibrator,
        "override_threshold": 1.1,
        "model_training_cutoff": refit["contract_date"].max(),
        "calibration_start": calibration["contract_date"].min(),
        "calibration_cutoff": calibration["contract_date"].max(),
        "policy_start": policy["contract_date"].min(),
        "policy_cutoff": policy["contract_date"].max(),
    }
    policy_probability = _apply_calibrator(
        calibrator, model.predict_proba(policy[feature_names])[:, 1]
    )
    policy_source = policy.copy()
    policy_source["override_probability"] = policy_probability
    threshold, threshold_table = _tune_threshold(policy_source)
    base_state["override_threshold"] = threshold
    return base_state, pd.DataFrame(rows), threshold_table


def _fit_model(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    spec: OverrideSpec,
    *,
    random_state: int,
) -> Any:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if spec.family == "logistic":
        estimator = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
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

        estimator = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
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
        raise ValueError(f"unknown override family: {spec.family}")
    weights = np.where(
        frame["override_actionable"].eq(1), spec.actionable_weight, 1.0
    )
    estimator.fit(
        frame[list(feature_names)],
        frame["override_target"].astype(int),
        classifier__sample_weight=weights,
    )
    return estimator


def _fit_calibrator(raw: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression

    clipped = np.clip(np.asarray(raw, dtype=float), 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1.0, max_iter=1000)
    model.fit(logits, actual)
    return {"model": model}


def _apply_calibrator(state: Mapping[str, Any], raw: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(raw, dtype=float), 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    return np.clip(state["model"].predict_proba(logits)[:, 1], 1e-6, 1 - 1e-6)


def _tune_threshold(policy: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in [*np.arange(0.05, 0.501, 0.025), 1.1]:
        override = (
            policy["override_actionable"].eq(1)
            & policy["override_probability"].ge(threshold)
        )
        recovered = int((override & policy["alternative_bucket_win"].eq(1)).sum())
        damaged = int((override & policy["default_bucket_win"].eq(1)).sum())
        rows.append(
            {
                "threshold": float(threshold),
                "override_count": int(override.sum()),
                "override_precision": float(policy.loc[override, "override_target"].mean())
                if override.any()
                else math.nan,
                "recovered_losses": recovered,
                "damaged_wins": damaged,
                "net_recovered_wins": recovered - damaged,
            }
        )
    table = pd.DataFrame(rows)
    selected = table.assign(
        precision_rank=table["override_precision"].fillna(-1.0)
    ).sort_values(
        ["net_recovered_wins", "precision_rank", "override_count", "threshold"],
        ascending=[False, False, True, False],
    ).iloc[0]
    return float(selected["threshold"]), table


def _ordered_windows(history: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = history.sort_values("contract_date")
    end = ordered["contract_date"].max()
    policy_start = end - pd.Timedelta(days=59)
    calibration_start = policy_start - pd.Timedelta(days=60)
    inner_start = calibration_start - pd.Timedelta(days=60)
    core = ordered.loc[ordered["contract_date"].lt(inner_start)]
    inner = ordered.loc[ordered["contract_date"].between(inner_start, calibration_start, inclusive="left")]
    inner = inner.loc[inner["contract_date"].lt(calibration_start)]
    calibration = ordered.loc[
        ordered["contract_date"].ge(calibration_start)
        & ordered["contract_date"].lt(policy_start)
    ]
    policy = ordered.loc[ordered["contract_date"].ge(policy_start)]
    if len(core) < 120 or min(len(inner), len(calibration), len(policy)) < 30:
        raise ValueError("insufficient history for V3 chronological windows")
    return core, inner, calibration, policy


def _binary_score(actual: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    y = np.asarray(actual, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-12, 1 - 1e-12)
    return {
        "count": len(y),
        "positive_rate": float(y.mean()),
        "log_loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
        "brier": float(np.mean((p - y) ** 2)),
    }
