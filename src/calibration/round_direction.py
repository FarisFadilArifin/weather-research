from __future__ import annotations

from dataclasses import asdict
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .bucket_probability import BASE_METHODS, canonical_two_degree_bucket, round_half_up
from .win_classifier import (
    CandidateSpec,
    KATL_PEAK_FEATURES,
    _apply_calibrator,
    _fit_calibrator,
    _fit_candidate,
    _predict_raw,
    build_win_frame,
    expected_calibration_error,
    win_feature_names,
)


ROUND_DIRECTION_FEATURES = (
    "prediction_fraction_f",
    "distance_to_floor_f",
    "distance_to_ceil_f",
    "floor_degree_is_odd",
    "floor_ceil_change_bucket",
    "base_models_supporting_up",
    "providers_supporting_up",
    "base_mean_minus_point_f",
    "provider_mean_minus_point_f",
    "prior_round_up_rate_30d",
    "prior_round_up_rate_90d",
)


def default_round_candidate_specs() -> list[CandidateSpec]:
    return [
        CandidateSpec("logistic", {"C": 0.03}, "platt"),
        CandidateSpec("logistic", {"C": 0.1}, "platt"),
        CandidateSpec("logistic", {"C": 1.0}, "platt"),
        CandidateSpec(
            "catboost",
            {
                "max_depth": 2,
                "max_iter": 150,
                "learning_rate": 0.03,
                "l2_regularization": 10.0,
            },
            "platt",
        ),
        CandidateSpec(
            "catboost",
            {
                "max_depth": 3,
                "max_iter": 200,
                "learning_rate": 0.02,
                "l2_regularization": 20.0,
            },
            "platt",
        ),
    ]


def round_feature_names(*, include_peak_features: bool) -> list[str]:
    return list(
        dict.fromkeys(
            [
                *win_feature_names(include_peak_features=include_peak_features),
                *ROUND_DIRECTION_FEATURES,
            ]
        )
    )


def build_round_direction_frame(
    feature_frame: pd.DataFrame,
    point_predictions: pd.DataFrame,
    base_validation_predictions: pd.DataFrame,
    *,
    include_peak_features: bool,
) -> pd.DataFrame:
    """Build the all-row binary floor/ceil target from honest point predictions."""
    frame = build_win_frame(
        feature_frame,
        point_predictions,
        base_validation_predictions,
        include_peak_features=include_peak_features,
    )
    point = pd.to_numeric(frame["point_prediction_f"], errors="coerce")
    actual = pd.to_numeric(frame["actual_high_f"], errors="coerce")
    floor_degree = np.floor(point).astype(int)
    ceil_degree = np.ceil(point).astype(int)
    fraction = point - floor_degree
    frame["floor_degree_f"] = floor_degree
    frame["ceil_degree_f"] = ceil_degree
    frame["prediction_fraction_f"] = fraction
    frame["distance_to_floor_f"] = fraction
    frame["distance_to_ceil_f"] = np.where(fraction.eq(0), 0.0, 1.0 - fraction)
    frame["floor_degree_is_odd"] = (floor_degree % 2 != 0).astype(float)
    frame["floor_bucket_label"] = pd.Series(floor_degree, index=frame.index).map(
        canonical_two_degree_bucket
    )
    frame["ceil_bucket_label"] = pd.Series(ceil_degree, index=frame.index).map(
        canonical_two_degree_bucket
    )
    frame["floor_ceil_change_bucket"] = frame["floor_bucket_label"].ne(
        frame["ceil_bucket_label"]
    ).astype(float)
    # Settlement labels are integer-valued. For non-integer predictions, actual > point is
    # exactly the decision that ceil is the correct direction; equality deterministically floors.
    frame["round_up"] = actual.gt(point).astype(int)
    frame["default_half_up"] = fraction.ge(0.5).astype(int)

    base_columns = [f"{name}_predicted_high_f" for name in BASE_METHODS]
    base = frame.reindex(columns=base_columns).apply(pd.to_numeric, errors="coerce")
    providers = frame.reindex(columns=["gfs_high_f", "hrrr_high_f", "nbm_high_f"]).apply(
        pd.to_numeric, errors="coerce"
    )
    frame["base_models_supporting_up"] = base.gt(point, axis=0).sum(axis=1).astype(float)
    frame["providers_supporting_up"] = providers.gt(point, axis=0).sum(axis=1).astype(float)
    frame["base_mean_minus_point_f"] = pd.to_numeric(
        frame["base_prediction_mean_f"], errors="coerce"
    ) - point
    frame["provider_mean_minus_point_f"] = pd.to_numeric(
        frame["provider_mean_high_f"], errors="coerce"
    ) - point
    ordered = frame.sort_values("contract_date")
    for window in (30, 90):
        frame.loc[ordered.index, f"prior_round_up_rate_{window}d"] = (
            ordered["round_up"]
            .shift(1)
            .rolling(window, min_periods=max(5, window // 5))
            .mean()
            .to_numpy()
        )
    for name in round_feature_names(include_peak_features=include_peak_features):
        if name not in frame:
            frame[name] = np.nan
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    return frame.sort_values("contract_date").reset_index(drop=True)


def fit_round_direction_system(
    frame: pd.DataFrame,
    *,
    station_id: str,
    include_peak_features: bool,
    candidate_specs: Sequence[CandidateSpec] | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    station = station_id.strip().upper()
    if station == "KDAL" and include_peak_features:
        raise ValueError("KDAL v20 no-peak round-direction model cannot include peak features")
    feature_names = round_feature_names(include_peak_features=include_peak_features)
    specs = list(candidate_specs or default_round_candidate_specs())
    development = frame.loc[frame["year"].between(2023, 2025)].copy()
    prediction_parts: list[pd.DataFrame] = []
    tuning_parts: list[pd.DataFrame] = []
    fold_states: list[dict[str, Any]] = []
    for validation_year in (2024, 2025):
        history = development.loc[development["year"].lt(validation_year)].copy()
        validation = development.loc[development["year"].eq(validation_year)].copy()
        if validation.empty:
            continue
        state, tuning = _fit_from_history(
            history,
            feature_names=feature_names,
            candidate_specs=specs,
            random_state=random_state,
        )
        predicted = predict_round_direction(state, validation)
        predicted["validation_year"] = validation_year
        predicted["model_training_cutoff"] = state["model_training_cutoff"]
        predicted["calibration_start"] = state["calibration_start"]
        predicted["calibration_cutoff"] = state["calibration_cutoff"]
        prediction_parts.append(predicted)
        tuning["outer_validation_year"] = validation_year
        tuning_parts.append(tuning)
        fold_states.append(
            {
                "validation_year": validation_year,
                "selected_family": state["selected_spec"].family,
                "selected_params_json": json.dumps(
                    dict(state["selected_spec"].params), sort_keys=True
                ),
                "model_training_cutoff": state["model_training_cutoff"],
                "calibration_start": state["calibration_start"],
                "calibration_cutoff": state["calibration_cutoff"],
            }
        )
    if not prediction_parts:
        raise ValueError("no chronological round-direction folds were produced")
    forward = pd.concat(prediction_parts, ignore_index=True).sort_values(
        "contract_date", ignore_index=True
    )
    metric_rows = []
    for year in (2024, 2025):
        subset = forward.loc[forward["validation_year"].eq(year)]
        metric_rows.append({"period": str(year), **round_direction_metrics(subset)})
    metric_rows.append({"period": "2024-2025", **round_direction_metrics(forward)})
    final_state, final_tuning = _fit_from_history(
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
        "forward_metrics": pd.DataFrame(metric_rows),
        "fold_states": pd.DataFrame(fold_states),
        "tuning": pd.concat(tuning_parts, ignore_index=True),
        "final_state": final_state,
        "final_tuning": final_tuning,
    }


def predict_round_direction(state: Mapping[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    raw = _predict_raw(state["model"], frame, state["feature_names"])
    probability = _apply_calibrator(state["calibrator"], raw)
    out = frame[
        [
            "contract_date",
            "actual_high_f",
            "actual_degree_f",
            "actual_bucket_label",
            "point_prediction_f",
            "point_degree_f",
            "point_bucket_label",
            "floor_degree_f",
            "ceil_degree_f",
            "floor_bucket_label",
            "ceil_bucket_label",
            "floor_ceil_change_bucket",
            "round_up",
            "default_half_up",
        ]
    ].copy()
    out["round_up_probability"] = probability
    out["predicted_round_up"] = out["round_up_probability"].ge(0.5).astype(int)
    out["corrected_degree_f"] = np.where(
        out["predicted_round_up"].eq(1), out["ceil_degree_f"], out["floor_degree_f"]
    ).astype(int)
    out["corrected_bucket_label"] = out["corrected_degree_f"].map(
        canonical_two_degree_bucket
    )
    out["point_bucket_win"] = out["point_bucket_label"].eq(
        out["actual_bucket_label"]
    ).astype(int)
    out["corrected_bucket_win"] = out["corrected_bucket_label"].eq(
        out["actual_bucket_label"]
    ).astype(int)
    out["bucket_switch"] = out["corrected_bucket_label"].ne(
        out["point_bucket_label"]
    )
    out["recovered_loss"] = (
        out["bucket_switch"] & out["corrected_bucket_win"].eq(1)
    ).astype(int)
    out["damaged_win"] = (
        out["bucket_switch"] & out["point_bucket_win"].eq(1)
    ).astype(int)
    return out


def round_direction_metrics(predictions: pd.DataFrame) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = predictions["round_up"].to_numpy(dtype=int)
    p = np.clip(predictions["round_up_probability"].to_numpy(dtype=float), 1e-12, 1 - 1e-12)
    predicted = predictions["predicted_round_up"].to_numpy(dtype=int)
    actionable = predictions["floor_ceil_change_bucket"].eq(1)
    switches = predictions["bucket_switch"].astype(bool)
    recovered = int(predictions["recovered_loss"].sum())
    damaged = int(predictions["damaged_win"].sum())
    return {
        "count": int(len(predictions)),
        "round_up_rate": float(y.mean()),
        "mean_round_up_probability": float(p.mean()),
        "binary_log_loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
        "brier": float(np.mean((p - y) ** 2)),
        "ece": expected_calibration_error(y, p),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else math.nan,
        "pr_auc": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else math.nan,
        "direction_accuracy": float(np.mean(predicted == y)),
        "default_half_up_direction_accuracy": float(
            np.mean(predictions["default_half_up"].to_numpy(dtype=int) == y)
        ),
        "point_degree_exact_accuracy": float(
            predictions["point_degree_f"].eq(predictions["actual_degree_f"]).mean()
        ),
        "corrected_degree_exact_accuracy": float(
            predictions["corrected_degree_f"].eq(predictions["actual_degree_f"]).mean()
        ),
        "point_bucket_hit_rate": float(predictions["point_bucket_win"].mean()),
        "corrected_bucket_hit_rate": float(predictions["corrected_bucket_win"].mean()),
        "bucket_hit_rate_lift": float(
            predictions["corrected_bucket_win"].mean()
            - predictions["point_bucket_win"].mean()
        ),
        "actionable_count": int(actionable.sum()),
        "actionable_rate": float(actionable.mean()),
        "actionable_direction_accuracy": float(
            predictions.loc[actionable, "predicted_round_up"].eq(
                predictions.loc[actionable, "round_up"]
            ).mean()
        ),
        "actionable_point_bucket_hit_rate": float(
            predictions.loc[actionable, "point_bucket_win"].mean()
        ),
        "actionable_corrected_bucket_hit_rate": float(
            predictions.loc[actionable, "corrected_bucket_win"].mean()
        ),
        "bucket_switch_count": int(switches.sum()),
        "recovered_losses": recovered,
        "damaged_wins": damaged,
        "net_recovered_wins": recovered - damaged,
        "recovery_damage_ratio": float(recovered / damaged)
        if damaged
        else (math.inf if recovered else math.nan),
    }


def continuous_round_direction_comparison(
    predictions: pd.DataFrame,
    continuous_predictions: pd.DataFrame,
    *,
    selected_family: str = "student_t",
) -> pd.DataFrame:
    continuous = continuous_predictions.copy()
    if "model_family" in continuous:
        continuous = continuous.loc[continuous["model_family"].eq(selected_family)]
    if "availability_status" in continuous:
        continuous = continuous.loc[continuous["availability_status"].eq("available")]
    continuous["contract_date"] = pd.to_datetime(
        continuous["contract_date"], errors="coerce"
    )
    continuous = continuous.drop_duplicates("contract_date")
    joined = predictions.merge(
        continuous[["contract_date", "degree_probabilities_json"]],
        on="contract_date",
        how="inner",
        validate="one_to_one",
    )
    probabilities = []
    for _, row in joined.iterrows():
        masses = json.loads(row["degree_probabilities_json"])
        point = float(row["point_prediction_f"])
        probabilities.append(
            sum(float(mass) for degree, mass in masses.items() if float(degree) > point)
        )
    joined["round_up_probability"] = probabilities
    joined["predicted_round_up"] = joined["round_up_probability"].ge(0.5).astype(int)
    joined["corrected_degree_f"] = np.where(
        joined["predicted_round_up"].eq(1), joined["ceil_degree_f"], joined["floor_degree_f"]
    ).astype(int)
    joined["corrected_bucket_label"] = joined["corrected_degree_f"].map(
        canonical_two_degree_bucket
    )
    joined["point_bucket_win"] = joined["point_bucket_label"].eq(
        joined["actual_bucket_label"]
    ).astype(int)
    joined["corrected_bucket_win"] = joined["corrected_bucket_label"].eq(
        joined["actual_bucket_label"]
    ).astype(int)
    joined["bucket_switch"] = joined["corrected_bucket_label"].ne(
        joined["point_bucket_label"]
    )
    joined["recovered_loss"] = (
        joined["bucket_switch"] & joined["corrected_bucket_win"].eq(1)
    ).astype(int)
    joined["damaged_win"] = (
        joined["bucket_switch"] & joined["point_bucket_win"].eq(1)
    ).astype(int)
    return pd.DataFrame([round_direction_metrics(joined)])


def audit_round_direction_system(
    source_frame: pd.DataFrame,
    result: Mapping[str, Any],
    *,
    include_peak_features: bool,
) -> pd.DataFrame:
    forward = result["forward_predictions"]
    point = pd.to_numeric(source_frame["point_prediction_f"], errors="coerce")
    actual = pd.to_numeric(source_frame["actual_high_f"], errors="coerce")
    expected_floor = np.floor(point).astype(int)
    expected_ceil = np.ceil(point).astype(int)
    expected_target = actual.gt(point).astype(int)
    expected_half_up = point.map(round_half_up)
    chronology = (
        pd.to_datetime(forward["model_training_cutoff"])
        < pd.to_datetime(forward["calibration_start"])
    ).all() and (
        pd.to_datetime(forward["calibration_cutoff"])
        < pd.to_datetime(forward["contract_date"])
    ).all()
    accounting = (
        int(forward["corrected_bucket_win"].sum())
        - int(forward["point_bucket_win"].sum())
        == int(forward["recovered_loss"].sum()) - int(forward["damaged_win"].sum())
    )
    rows = [
        {
            "audit": "floor_formula",
            "passed": bool(np.array_equal(expected_floor, source_frame["floor_degree_f"])),
            "detail": "floor(point_prediction_f)",
        },
        {
            "audit": "ceil_formula",
            "passed": bool(np.array_equal(expected_ceil, source_frame["ceil_degree_f"])),
            "detail": "ceil(point_prediction_f)",
        },
        {
            "audit": "binary_target_formula",
            "passed": bool(expected_target.eq(source_frame["round_up"]).all()),
            "detail": "round_up = actual_high_f > point_prediction_f",
        },
        {
            "audit": "point_rounding_is_half_up",
            "passed": bool(np.array_equal(expected_half_up, source_frame["point_degree_f"])),
            "detail": "independent half-up recomputation",
        },
        {
            "audit": "all_outer_rows_scored",
            "passed": bool(
                len(forward) == int(source_frame["year"].isin([2024, 2025]).sum())
                and np.isfinite(forward["round_up_probability"]).all()
            ),
            "detail": f"outer_rows={len(forward)}",
        },
        {
            "audit": "fixed_binary_threshold",
            "passed": bool(
                forward["predicted_round_up"].eq(
                    forward["round_up_probability"].ge(0.5).astype(int)
                ).all()
            ),
            "detail": "1 when probability >= 0.5, otherwise 0",
        },
        {
            "audit": "corrected_degree_is_floor_or_ceil",
            "passed": bool(
                (
                    forward["corrected_degree_f"].eq(forward["floor_degree_f"])
                    | forward["corrected_degree_f"].eq(forward["ceil_degree_f"])
                ).all()
            ),
            "detail": "no abstention or third action",
        },
        {
            "audit": "chronological_fit_calibration_validation",
            "passed": bool(chronology),
            "detail": "fit < calibration < following outer year",
        },
        {
            "audit": "policy_win_accounting_identity",
            "passed": bool(accounting),
            "detail": "corrected minus point wins equals recovered minus damaged",
        },
        {
            "audit": "station_peak_feature_contract",
            "passed": bool(
                include_peak_features
                or set(result["feature_names"]).isdisjoint(KATL_PEAK_FEATURES)
            ),
            "detail": f"peak_feature_overlap={len(set(result['feature_names']) & set(KATL_PEAK_FEATURES))}",
        },
    ]
    if "train_through_year" in source_frame:
        valid = source_frame.dropna(subset=["train_through_year", "year"])
        rows.append(
            {
                "audit": "point_predictions_are_forward",
                "passed": bool((valid["train_through_year"] < valid["year"]).all()),
                "detail": f"checked_rows={len(valid)}",
            }
        )
    return pd.DataFrame(rows)


def serializable_round_bundle(result: Mapping[str, Any]) -> dict[str, Any]:
    state = dict(result["final_state"])
    state["selected_spec"] = asdict(state["selected_spec"])
    return {
        "artifact_type": "station_floor_ceil_round_direction_classifier_research",
        "schema_version": 1,
        "station_id": result["station_id"],
        "feature_profile": result["feature_profile"],
        "feature_names": result["feature_names"],
        "class_semantics": {"0": "floor", "1": "ceil"},
        "classification_threshold": 0.5,
        "state": state,
        "forward_metrics": result["forward_metrics"].to_dict(orient="records"),
    }


def _fit_from_history(
    history: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    candidate_specs: Sequence[CandidateSpec],
    random_state: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    core, inner, calibration = _ordered_windows(history)
    core_model = core.assign(bucket_win=core["round_up"])
    inner_model = inner.assign(bucket_win=inner["round_up"])
    candidates = []
    rows = []
    for spec in candidate_specs:
        model = _fit_candidate(
            core_model, feature_names, spec, random_state=random_state
        )
        probability = _predict_raw(model, inner_model, feature_names)
        score = _binary_score(inner["round_up"].to_numpy(dtype=int), probability)
        rows.append(
            {
                "candidate_key": spec.key,
                "family": spec.family,
                "params_json": json.dumps(dict(spec.params), sort_keys=True),
                "inner_start": inner["contract_date"].min(),
                "inner_cutoff": inner["contract_date"].max(),
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
        ),
    )[0][0]
    refit = pd.concat([core, inner], ignore_index=True)
    refit_model = refit.assign(bucket_win=refit["round_up"])
    model = _fit_candidate(
        refit_model, feature_names, selected, random_state=random_state
    )
    calibrator = _fit_calibrator(
        "platt",
        _predict_raw(model, calibration, feature_names),
        calibration["round_up"].to_numpy(dtype=int),
    )
    return {
        "feature_names": list(feature_names),
        "selected_spec": selected,
        "model": model,
        "calibrator": calibrator,
        "classification_threshold": 0.5,
        "model_training_cutoff": refit["contract_date"].max(),
        "calibration_start": calibration["contract_date"].min(),
        "calibration_cutoff": calibration["contract_date"].max(),
    }, pd.DataFrame(rows)


def _ordered_windows(
    history: pd.DataFrame, *, inner_days: int = 60, calibration_days: int = 60
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = history.sort_values("contract_date")
    end = ordered["contract_date"].max()
    calibration_start = end - pd.Timedelta(days=calibration_days - 1)
    inner_start = calibration_start - pd.Timedelta(days=inner_days)
    core = ordered.loc[ordered["contract_date"].lt(inner_start)]
    inner = ordered.loc[
        ordered["contract_date"].ge(inner_start)
        & ordered["contract_date"].lt(calibration_start)
    ]
    calibration = ordered.loc[ordered["contract_date"].ge(calibration_start)]
    if len(core) < 180 or min(len(inner), len(calibration)) < 30:
        raise ValueError("insufficient chronological history for round-direction model")
    if not (
        core["contract_date"].max() < inner["contract_date"].min()
        <= inner["contract_date"].max() < calibration["contract_date"].min()
    ):
        raise AssertionError("round-direction history windows overlap")
    return core, inner, calibration


def _binary_score(actual: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    y = np.asarray(actual, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-12, 1 - 1e-12)
    return {
        "count": int(len(y)),
        "positive_rate": float(y.mean()),
        "log_loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
        "brier": float(np.mean((p - y) ** 2)),
    }
