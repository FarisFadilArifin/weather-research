from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .bucket_probability import (
    BASE_METHODS,
    MANDATORY_SOURCE_FEATURES,
    MISSING_INDICATOR_SUFFIX,
    probability_feature_names,
    probability_package_versions,
    round_half_up,
    temperature_scale,
    _resolve_feature_profile,
)


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "station_bucket_correction_model"
RELATION_LABELS = ("lower", "same", "upper")
EFFECTIVE_TIE_TOLERANCE = 1e-3


@dataclass(frozen=True)
class LogisticSpec:
    c: float
    class_weight: str | None


def default_logistic_specs() -> list[LogisticSpec]:
    return [
        LogisticSpec(c=c, class_weight=class_weight)
        for c in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
        for class_weight in (None, "balanced")
    ]


def add_bucket_correction_targets(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    raw_point_bucket = (
        pd.to_numeric(out["point_degree_f"], errors="raise").astype(int) // 2
    )
    if "observed_high_temp_through_as_of_f" in out:
        observed_degree = pd.to_numeric(
            out["observed_high_temp_through_as_of_f"], errors="coerce"
        ).map(lambda value: round_half_up(value) if pd.notna(value) else np.nan)
        observed_bucket = observed_degree // 2
        out["point_bucket_index"] = pd.concat(
            [raw_point_bucket, observed_bucket], axis=1
        ).max(axis=1).astype(int)
    else:
        out["point_bucket_index"] = raw_point_bucket
    out["actual_bucket_index"] = (
        pd.to_numeric(out["actual_degree_f"], errors="raise").astype(int) // 2
    )
    out["bucket_delta"] = out["actual_bucket_index"] - out["point_bucket_index"]
    out["point_bucket_wrong"] = out["bucket_delta"].ne(0).astype(int)
    out["bucket_relation_class"] = np.select(
        [out["bucket_delta"].lt(0), out["bucket_delta"].gt(0)],
        [0, 2],
        default=1,
    ).astype(int)
    return out


def fit_bucket_correction_system(
    frame: pd.DataFrame,
    *,
    station_id: str,
    point_model_version: str,
    point_bundle_sha256: str,
    include_peak_features: bool,
    feature_profile: str | None = None,
    model_version: str | None = None,
    specs: Sequence[LogisticSpec] | None = None,
    calibration_days: int = 90,
    min_train_rows: int = 180,
    random_state: int = 42,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    station_id = station_id.strip().upper()
    resolved_profile = _resolve_feature_profile(
        include_peak_features=include_peak_features, feature_profile=feature_profile
    )
    feature_names = probability_feature_names(
        include_peak_features=include_peak_features, feature_profile=resolved_profile
    )
    development = add_bucket_correction_targets(
        frame.loc[frame["year"].between(2023, 2025)].copy()
    )
    if len(development) < min_train_rows:
        raise ValueError("insufficient 2023-2025 bucket-correction rows")
    candidates = list(specs or default_logistic_specs())
    oof_parts: list[pd.DataFrame] = []
    tuning_rows: list[dict[str, Any]] = []

    for validation_year in (2024, 2025):
        outer_train = development.loc[development["year"].lt(validation_year)].copy()
        outer_valid = development.loc[development["year"].eq(validation_year)].copy()
        if len(outer_train) < min_train_rows or outer_valid.empty:
            continue
        inner_train, inner_valid = _calibration_split(
            outer_train, calibration_days=calibration_days
        )
        _assert_forward_history(inner_train, inner_valid, outer_train, outer_valid)

        risk_choice = _select_logistic_candidate(
            inner_train,
            inner_valid,
            feature_names,
            target="point_bucket_wrong",
            classes=(0, 1),
            specs=candidates,
            random_state=random_state,
        )
        relation_choice = _select_logistic_candidate(
            inner_train,
            inner_valid,
            feature_names,
            target="bucket_relation_class",
            classes=(0, 1, 2),
            specs=candidates,
            random_state=random_state,
        )
        risk_model = _fit_logistic(
            outer_train,
            feature_names,
            "point_bucket_wrong",
            risk_choice["spec"],
            random_state=random_state,
        )
        relation_model = _fit_logistic(
            outer_train,
            feature_names,
            "bucket_relation_class",
            relation_choice["spec"],
            random_state=random_state,
        )
        risk_probabilities = temperature_scale(
            _predict_aligned(risk_model, outer_valid, feature_names, (0, 1)),
            float(risk_choice["temperature"]),
        )
        relation_probabilities = temperature_scale(
            _predict_aligned(
                relation_model, outer_valid, feature_names, (0, 1, 2)
            ),
            float(relation_choice["temperature"]),
        )
        rows = _prediction_rows(
            outer_valid,
            risk_probabilities[:, 1],
            relation_probabilities,
            validation_year,
        )
        rows["model_training_cutoff"] = outer_train["contract_date"].max()
        rows["calibration_training_cutoff"] = inner_train["contract_date"].max()
        rows["calibration_validation_start"] = inner_valid["contract_date"].min()
        rows["calibration_validation_cutoff"] = inner_valid["contract_date"].max()
        oof_parts.append(rows)
        for stage, choice in (("risk", risk_choice), ("relation", relation_choice)):
            tuning_rows.append(
                {
                    "validation_year": validation_year,
                    "stage": stage,
                    "C": choice["spec"].c,
                    "class_weight": choice["spec"].class_weight,
                    "temperature": choice["temperature"],
                    "log_loss": choice["log_loss"],
                    "brier": choice["brier"],
                }
            )

    if not oof_parts:
        raise ValueError("no forward bucket-correction folds were produced")
    oof = pd.concat(oof_parts, ignore_index=True)
    thresholds, policy_rows = tune_override_policy(oof)

    final_inner_train, final_inner_valid = _calibration_split(
        development, calibration_days=calibration_days
    )
    final_risk_choice = _select_logistic_candidate(
        final_inner_train,
        final_inner_valid,
        feature_names,
        target="point_bucket_wrong",
        classes=(0, 1),
        specs=candidates,
        random_state=random_state,
    )
    final_relation_choice = _select_logistic_candidate(
        final_inner_train,
        final_inner_valid,
        feature_names,
        target="bucket_relation_class",
        classes=(0, 1, 2),
        specs=candidates,
        random_state=random_state,
    )
    risk_model = _fit_logistic(
        development,
        feature_names,
        "point_bucket_wrong",
        final_risk_choice["spec"],
        random_state=random_state,
    )
    relation_model = _fit_logistic(
        development,
        feature_names,
        "bucket_relation_class",
        final_relation_choice["spec"],
        random_state=random_state,
    )
    selected_policy = policy_rows.loc[policy_rows["selected"]].iloc[0].to_dict()
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "station_id": station_id,
        "model_version": model_version or f"{station_id.lower()}_bucket_correction_v1",
        "point_model_version": point_model_version,
        "point_bundle_sha256": point_bundle_sha256.lower(),
        "feature_profile": resolved_profile,
        "feature_names": feature_names,
        "mandatory_source_features": list(MANDATORY_SOURCE_FEATURES),
        "relation_labels": list(RELATION_LABELS),
        "risk_model": risk_model,
        "risk_spec": _spec_dict(final_risk_choice["spec"]),
        "risk_temperature": float(final_risk_choice["temperature"]),
        "relation_model": relation_model,
        "relation_spec": _spec_dict(final_relation_choice["spec"]),
        "relation_temperature": float(final_relation_choice["temperature"]),
        "decision_thresholds": thresholds,
        "forward_metrics": correction_metrics(oof, thresholds),
        "forward_policy_metrics": selected_policy,
        "training_start": development["contract_date"].min().date().isoformat(),
        "training_cutoff": development["contract_date"].max().date().isoformat(),
        "training_rows": int(len(development)),
        "package_versions": probability_package_versions(),
    }
    tuning = pd.concat(
        [
            pd.DataFrame(tuning_rows),
            policy_rows.assign(validation_year="policy", stage="policy"),
        ],
        ignore_index=True,
        sort=False,
    )
    return bundle, oof, tuning


def predict_bucket_correction(
    bundle: Mapping[str, Any], feature_values: Mapping[str, Any]
) -> dict[str, Any]:
    missing = [
        name
        for name in bundle["mandatory_source_features"]
        if _finite_number(feature_values.get(name)) is None
    ]
    for name in (
        "point_prediction_f",
        *(f"{method}_predicted_high_f" for method in BASE_METHODS),
    ):
        if _finite_number(feature_values.get(name)) is None:
            missing.append(name)
    if missing:
        return {
            "status": "unavailable",
            "reason": "missing_required_features:"
            + ",".join(sorted(set(missing))),
        }

    frame = pd.DataFrame([dict(feature_values)])
    frame = _ensure_feature_contract(frame, bundle["feature_names"])
    risk = temperature_scale(
        _predict_aligned(
            bundle["risk_model"], frame, bundle["feature_names"], (0, 1)
        ),
        float(bundle["risk_temperature"]),
    )[0]
    relation = temperature_scale(
        _predict_aligned(
            bundle["relation_model"],
            frame,
            bundle["feature_names"],
            (0, 1, 2),
        ),
        float(bundle["relation_temperature"]),
    )[0]
    point_degree = round_half_up(float(feature_values["point_prediction_f"]))
    observed_bucket_index = (
        round_half_up(float(feature_values["observed_high_temp_through_as_of_f"])) // 2
    )
    point_bucket_index = max(point_degree // 2, observed_bucket_index)
    decision = apply_override_policy(
        float(risk[1]),
        relation,
        bundle["decision_thresholds"],
        point_bucket_index=point_bucket_index,
        observed_high_f=float(feature_values["observed_high_temp_through_as_of_f"]),
    )
    direction = int(decision["direction"])
    recommended_index = point_bucket_index + direction
    return {
        "status": "ok",
        "model_version": bundle["model_version"],
        "rounded_point_high_f": point_degree,
        "point_bucket_label": _bucket_label(point_bucket_index),
        "recommended_bucket_label": _bucket_label(recommended_index),
        "risk_probability": float(risk[1]),
        "relation_probabilities": {
            label: float(value)
            for label, value in zip(RELATION_LABELS, relation, strict=True)
        },
        "overrides_point_bucket": bool(direction),
        "override_direction": direction,
        "decision": "shadow_override" if direction else "no_override",
        "decision_reason": decision["reason"],
    }


def tune_override_policy(
    predictions: pd.DataFrame,
    *,
    minimum_total_switches: int = 10,
    minimum_switches_per_year: int = 3,
) -> tuple[dict[str, Any], pd.DataFrame]:
    records: list[dict[str, Any]] = []
    relation = np.vstack(predictions["relation_probabilities"].to_numpy())
    risk = predictions["risk_probability"].to_numpy(dtype=float)
    bucket_delta = predictions["bucket_delta"].to_numpy(dtype=int)
    years = predictions["validation_year"].to_numpy(dtype=int)
    point_hit = bucket_delta == 0
    point_bucket_index = predictions["point_bucket_index"].to_numpy(dtype=int)
    observed_high = pd.to_numeric(
        predictions["observed_high_temp_through_as_of_f"], errors="coerce"
    ).to_numpy(dtype=float)

    for minimum_risk in np.arange(0.35, 0.801, 0.05):
        for minimum_direction in np.arange(0.35, 0.751, 0.05):
            for minimum_direction_margin in np.arange(0.0, 0.301, 0.05):
                for minimum_advantage in np.arange(0.0, 0.301, 0.05):
                    lower = relation[:, 0]
                    same = relation[:, 1]
                    upper = relation[:, 2]
                    direction = np.where(upper > lower, 1, -1)
                    direction_probability = np.maximum(lower, upper)
                    direction_margin = np.abs(upper - lower)
                    switch = (
                        (risk >= minimum_risk)
                        & (direction_probability >= minimum_direction)
                        & (direction_margin >= minimum_direction_margin)
                        & ((direction_probability - same) >= minimum_advantage)
                    )
                    minimum_bucket_index = np.floor(
                        np.floor(observed_high + 0.5) / 2.0
                    ).astype(int)
                    switch &= (point_bucket_index + direction) >= minimum_bucket_index
                    switch_count = int(switch.sum())
                    if switch_count == 0:
                        continue
                    corrected_hit = np.where(switch, bucket_delta == direction, point_hit)
                    switch_corrected = bucket_delta[switch] == direction[switch]
                    switch_point = point_hit[switch]
                    year_switch_counts: dict[str, int] = {}
                    year_lifts: dict[str, float] = {}
                    stable_years = True
                    for year in sorted(set(years)):
                        selected = switch & (years == year)
                        count = int(selected.sum())
                        year_switch_counts[str(year)] = count
                        if count:
                            lift = float(
                                (bucket_delta[selected] == direction[selected]).mean()
                                - point_hit[selected].mean()
                            )
                        else:
                            lift = float("nan")
                        year_lifts[str(year)] = lift
                        if count < minimum_switches_per_year or not math.isfinite(lift) or lift <= 0:
                            stable_years = False
                    switch_lift = float(switch_corrected.mean() - switch_point.mean())
                    corrected_accuracy = float(corrected_hit.mean())
                    point_accuracy = float(point_hit.mean())
                    stable = bool(
                        switch_count >= minimum_total_switches
                        and stable_years
                        and switch_lift > 0
                        and corrected_accuracy >= point_accuracy
                    )
                    records.append(
                        {
                            "minimum_risk_probability": float(minimum_risk),
                            "minimum_direction_probability": float(minimum_direction),
                            "minimum_direction_margin": float(minimum_direction_margin),
                            "minimum_advantage_over_same": float(minimum_advantage),
                            "switch_count": switch_count,
                            "switch_coverage": float(switch.mean()),
                            "switch_accuracy": float(switch_corrected.mean()),
                            "point_accuracy_on_switches": float(switch_point.mean()),
                            "switch_lift": switch_lift,
                            "corrected_bucket_accuracy": corrected_accuracy,
                            "point_bucket_accuracy": point_accuracy,
                            "year_switch_counts_json": json.dumps(year_switch_counts, sort_keys=True),
                            "year_switch_lifts_json": json.dumps(year_lifts, sort_keys=True),
                            "stable_forward_evidence": stable,
                        }
                    )
    if not records:
        records.append(
            {
                "minimum_risk_probability": 1.1,
                "minimum_direction_probability": 1.1,
                "minimum_direction_margin": 1.1,
                "minimum_advantage_over_same": 1.1,
                "switch_count": 0,
                "switch_coverage": 0.0,
                "switch_accuracy": float("nan"),
                "point_accuracy_on_switches": float("nan"),
                "switch_lift": float("nan"),
                "corrected_bucket_accuracy": float(point_hit.mean()),
                "point_bucket_accuracy": float(point_hit.mean()),
                "year_switch_counts_json": json.dumps(
                    {str(year): 0 for year in sorted(set(years))}, sort_keys=True
                ),
                "year_switch_lifts_json": json.dumps(
                    {str(year): float("nan") for year in sorted(set(years))},
                    sort_keys=True,
                ),
                "stable_forward_evidence": False,
            }
        )
    policies = pd.DataFrame(records)
    stable = policies.loc[policies["stable_forward_evidence"]].copy()
    if stable.empty:
        eligible = policies.loc[
            policies["switch_count"].ge(minimum_total_switches)
        ].copy()
        if eligible.empty:
            eligible = policies.copy()
        selected_index = eligible.sort_values(
            [
                "corrected_bucket_accuracy",
                "switch_lift",
                "switch_accuracy",
                "switch_count",
            ],
            ascending=[False, False, False, False],
        ).index[0]
    else:
        selected_index = stable.sort_values(
            [
                "corrected_bucket_accuracy",
                "switch_lift",
                "switch_accuracy",
                "switch_count",
            ],
            ascending=[False, False, False, False],
        ).index[0]
    policies["selected"] = policies.index == selected_index
    selected = policies.loc[selected_index]
    thresholds = {
        "minimum_risk_probability": float(selected["minimum_risk_probability"]),
        "minimum_direction_probability": float(
            selected["minimum_direction_probability"]
        ),
        "minimum_direction_margin": float(selected["minimum_direction_margin"]),
        "minimum_advantage_over_same": float(
            selected["minimum_advantage_over_same"]
        ),
        "minimum_total_switches": minimum_total_switches,
        "minimum_switches_per_year": minimum_switches_per_year,
        "stable_forward_evidence": bool(selected["stable_forward_evidence"]),
    }
    return thresholds, policies


def apply_override_policy(
    risk_probability: float,
    relation_probabilities: Sequence[float],
    thresholds: Mapping[str, Any],
    *,
    point_bucket_index: int | None = None,
    observed_high_f: float | None = None,
) -> dict[str, Any]:
    lower, same, upper = (float(value) for value in relation_probabilities)
    direction = 1 if upper > lower else -1
    direction_probability = max(lower, upper)
    if risk_probability < float(thresholds["minimum_risk_probability"]):
        return {"direction": 0, "reason": "risk_below_threshold"}
    if direction_probability < float(
        thresholds["minimum_direction_probability"]
    ):
        return {"direction": 0, "reason": "direction_probability_below_threshold"}
    if abs(upper - lower) < float(thresholds["minimum_direction_margin"]):
        return {"direction": 0, "reason": "direction_margin_below_threshold"}
    if direction_probability - same < float(
        thresholds["minimum_advantage_over_same"]
    ):
        return {"direction": 0, "reason": "advantage_over_same_below_threshold"}
    if point_bucket_index is not None and observed_high_f is not None:
        observed_bucket_index = round_half_up(observed_high_f) // 2
        if point_bucket_index + direction < observed_bucket_index:
            return {"direction": 0, "reason": "observed_high_physical_floor"}
    return {"direction": direction, "reason": "stable_override_thresholds_passed"}


def correction_metrics(
    predictions: pd.DataFrame, thresholds: Mapping[str, Any]
) -> dict[str, float]:
    relation = np.vstack(predictions["relation_probabilities"].to_numpy())
    risk = predictions["risk_probability"].to_numpy(dtype=float)
    actual_wrong = predictions["point_bucket_wrong"].to_numpy(dtype=int)
    actual_relation = predictions["bucket_relation_class"].to_numpy(dtype=int)
    bucket_delta = predictions["bucket_delta"].to_numpy(dtype=int)
    directions = np.asarray(
        [
            apply_override_policy(
                risk_value,
                relation_value,
                thresholds,
                point_bucket_index=int(point_bucket),
                observed_high_f=float(observed_high),
            )["direction"]
            for risk_value, relation_value, point_bucket, observed_high in zip(
                risk,
                relation,
                predictions["point_bucket_index"],
                predictions["observed_high_temp_through_as_of_f"],
                strict=True,
            )
        ],
        dtype=int,
    )
    switches = directions != 0
    point_hit = bucket_delta == 0
    corrected_hit = np.where(switches, bucket_delta == directions, point_hit)
    binary_probabilities = np.column_stack([1.0 - risk, risk])
    risk_brier = np.square(risk - actual_wrong).mean()
    relation_one_hot = np.eye(3)[actual_relation]
    relation_brier = np.square(relation - relation_one_hot).sum(axis=1).mean()
    result = {
        "count": float(len(predictions)),
        "risk_log_loss": float(
            -np.log(binary_probabilities[np.arange(len(actual_wrong)), actual_wrong].clip(1e-12)).mean()
        ),
        "risk_brier": float(risk_brier),
        "relation_log_loss": float(
            -np.log(relation[np.arange(len(actual_relation)), actual_relation].clip(1e-12)).mean()
        ),
        "relation_brier": float(relation_brier),
        "relation_accuracy": float((relation.argmax(axis=1) == actual_relation).mean()),
        "point_bucket_accuracy": float(point_hit.mean()),
        "corrected_bucket_accuracy": float(corrected_hit.mean()),
        "switch_count": float(switches.sum()),
        "switch_coverage": float(switches.mean()),
    }
    if switches.any():
        result["switch_accuracy"] = float(
            (bucket_delta[switches] == directions[switches]).mean()
        )
        result["point_accuracy_on_switches"] = float(point_hit[switches].mean())
    else:
        result["switch_accuracy"] = float("nan")
        result["point_accuracy_on_switches"] = float("nan")
    return result


def export_bucket_correction_bundle(
    bundle: Mapping[str, Any],
    output_dir: Path,
    *,
    source_identity: Mapping[str, Any],
) -> tuple[Path, Path]:
    import joblib

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{bundle['station_id']}_{bundle['model_version']}"
    bundle_path = output_dir / f"{stem}.joblib"
    manifest_path = output_dir / f"{stem}.json"
    joblib.dump(dict(bundle), bundle_path)
    bundle_hash = _sha256_file(bundle_path)
    manifest = {
        key: value
        for key, value in bundle.items()
        if key not in {"risk_model", "relation_model"}
    }
    manifest["source_identity"] = dict(source_identity)
    manifest["artifact_integrity"] = {"bundle_sha256": bundle_hash}
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    return bundle_path, manifest_path


def _calibration_split(
    frame: pd.DataFrame, *, calibration_days: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_at = frame["contract_date"].max() - pd.Timedelta(
        days=calibration_days - 1
    )
    train = frame.loc[frame["contract_date"].lt(split_at)].copy()
    valid = frame.loc[frame["contract_date"].ge(split_at)].copy()
    if len(train) < 60 or valid.empty:
        raise ValueError("insufficient bucket-correction calibration history")
    return train, valid


def _assert_forward_history(
    inner_train: pd.DataFrame,
    inner_valid: pd.DataFrame,
    outer_train: pd.DataFrame,
    outer_valid: pd.DataFrame,
) -> None:
    validation_start = outer_valid["contract_date"].min()
    if (
        outer_train["contract_date"].max() >= validation_start
        or inner_train["contract_date"].max() >= inner_valid["contract_date"].min()
        or inner_valid["contract_date"].max() >= validation_start
    ):
        raise AssertionError("bucket-correction history is not strictly forward")


def _select_logistic_candidate(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    target: str,
    classes: Sequence[int],
    specs: Sequence[LogisticSpec],
    random_state: int,
) -> dict[str, Any]:
    actual = valid[target].to_numpy(dtype=int)
    rows: list[dict[str, Any]] = []
    for spec in specs:
        model = _fit_logistic(
            train, feature_names, target, spec, random_state=random_state
        )
        raw = _predict_aligned(model, valid, feature_names, classes)
        for temperature in np.linspace(0.5, 3.0, 11):
            probabilities = temperature_scale(raw, float(temperature))
            one_hot = np.eye(len(classes))[actual]
            rows.append(
                {
                    "spec": spec,
                    "temperature": float(temperature),
                    "log_loss": float(
                        -np.log(
                            probabilities[np.arange(len(actual)), actual].clip(
                                1e-12
                            )
                        ).mean()
                    ),
                    "brier": float(
                        np.square(probabilities - one_hot).sum(axis=1).mean()
                    ),
                }
            )
    best_log_loss = min(row["log_loss"] for row in rows)
    near_log_loss = [
        row
        for row in rows
        if row["log_loss"] <= best_log_loss + EFFECTIVE_TIE_TOLERANCE
    ]
    best_brier = min(row["brier"] for row in near_log_loss)
    effective_ties = [
        row
        for row in near_log_loss
        if row["brier"] <= best_brier + EFFECTIVE_TIE_TOLERANCE
    ]
    return min(
        effective_ties,
        key=lambda row: (
            1 if row["spec"].class_weight == "balanced" else 0,
            row["spec"].c,
        ),
    )


def _fit_logistic(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    target: str,
    spec: LogisticSpec,
    *,
    random_state: int,
) -> Any:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    keep_empty_features=True,
                ),
            ),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=spec.c,
                    class_weight=spec.class_weight,
                    solver="lbfgs",
                    max_iter=2_000,
                    random_state=random_state,
                ),
            ),
        ]
    )
    pipeline.fit(frame.reindex(columns=feature_names), frame[target].astype(int))
    return pipeline


def _predict_aligned(
    model: Any,
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    classes: Sequence[int],
) -> np.ndarray:
    raw = np.asarray(
        model.predict_proba(frame.reindex(columns=feature_names)), dtype=float
    )
    fitted_classes = [int(value) for value in model.named_steps["classifier"].classes_]
    aligned = np.full((len(frame), len(classes)), 1e-12, dtype=float)
    for source_index, fitted_class in enumerate(fitted_classes):
        aligned[:, list(classes).index(fitted_class)] = raw[:, source_index]
    return aligned / aligned.sum(axis=1, keepdims=True)


def _prediction_rows(
    frame: pd.DataFrame,
    risk_probability: np.ndarray,
    relation_probabilities: np.ndarray,
    validation_year: int,
) -> pd.DataFrame:
    rows = frame[
        [
            "contract_date",
            "actual_high_f",
            "actual_degree_f",
            "point_prediction_f",
            "point_degree_f",
            "point_bucket_index",
            "observed_high_temp_through_as_of_f",
            "actual_bucket_index",
            "bucket_delta",
            "point_bucket_wrong",
            "bucket_relation_class",
        ]
    ].copy()
    rows["validation_year"] = validation_year
    rows["risk_probability"] = risk_probability
    rows["relation_probabilities"] = [
        values.tolist() for values in relation_probabilities
    ]
    return rows


def _ensure_feature_contract(
    frame: pd.DataFrame, feature_names: Sequence[str]
) -> pd.DataFrame:
    from .bucket_probability import add_probability_features

    out = add_probability_features(frame)
    for name in feature_names:
        if name.endswith(MISSING_INDICATOR_SUFFIX):
            source_name = name[: -len(MISSING_INDICATOR_SUFFIX)]
            source = (
                out[source_name]
                if source_name in out
                else pd.Series([np.nan] * len(out), index=out.index)
            )
            out[name] = source.map(
                lambda value: 1.0 if _finite_number(value) is None else 0.0
            )
        if name not in out:
            out[name] = np.nan
    return out


def _bucket_label(bucket_index: int) -> str:
    low = 2 * int(bucket_index)
    return f"{low}-{low + 1}"


def _spec_dict(spec: LogisticSpec) -> dict[str, Any]:
    return {"C": spec.c, "class_weight": spec.class_weight}


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
