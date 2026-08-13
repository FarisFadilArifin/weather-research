from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
CONFIG_DIR = HERE / "configs"


def _markdown(source: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip() + "\n"}


def _code(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip() + "\n",
    }


def _probability_cell(config: dict[str, Any]) -> str:
    if config["probability_target"] == "fahrenheit_2f":
        return """
from src.calibration.bucket_probability import (
    build_probability_frame, evaluate_probability_holdout,
    export_probability_bundle, fit_probability_system,
)

probability_frame = build_probability_frame(
    feature_frame, forward_blend, expert_oof,
    include_peak_features=False, feature_profile=PROBABILITY_PROFILE,
)
probability_bundle, probability_forward, probability_tuning = fit_probability_system(
    probability_frame,
    station_id=STATION_ID,
    point_model_version=POINT_MODEL_VERSION,
    point_bundle_sha256=sha256_file(point_bundle_path),
    include_peak_features=False,
    feature_profile=PROBABILITY_PROFILE,
    model_version="station_expert_ensemble_v1_ordinal_shadow",
    development_years=PROBABILITY_DEVELOPMENT_YEARS,
    forward_validation_years=PROBABILITY_FORWARD_YEARS,
)
probability_bundle["holdout_status"] = "exploratory_shadow_only"
probability_bundle["promotion_approved"] = False
probability_bundle_path, probability_manifest_path = export_probability_bundle(
    probability_bundle, OUTPUT_DIR / "probability",
    source_identity=SOURCE_IDENTITY,
)
probability_holdout, probability_holdout_metrics = evaluate_probability_holdout(
    feature_frame, point_holdout, expert_holdout, probability_bundle, holdout_year=2026,
)
probability_holdout_frame = build_probability_frame(
    feature_frame, point_holdout, expert_holdout,
    include_peak_features=False, feature_profile=PROBABILITY_PROFILE,
)
probability_holdout_frame = probability_holdout_frame.loc[probability_holdout_frame["year"].eq(2026)].copy()
probability_forward.to_csv(OUTPUT_DIR / "probability_forward_predictions.csv", index=False)
probability_tuning.to_csv(OUTPUT_DIR / "probability_tuning.csv", index=False)
probability_holdout.to_csv(OUTPUT_DIR / "probability_holdout_2026.csv", index=False)
probability_holdout_metrics
"""
    return """
from src.calibration.celsius_market_probability import (
    build_celsius_probability_frame, evaluate_celsius_probability_holdout,
    export_celsius_probability_bundle, fit_celsius_probability_system,
)

probability_frame = build_celsius_probability_frame(
    feature_frame, forward_blend, expert_oof,
    include_peak_features=False, feature_profile=PROBABILITY_PROFILE,
)
probability_bundle, probability_forward, probability_tuning = fit_celsius_probability_system(
    probability_frame,
    station_id=STATION_ID,
    point_model_version=POINT_MODEL_VERSION,
    point_bundle_sha256=sha256_file(point_bundle_path),
    feature_profile=PROBABILITY_PROFILE,
    model_version="station_expert_ensemble_v1_celsius_ordinal_shadow",
    development_years=PROBABILITY_DEVELOPMENT_YEARS,
    forward_validation_years=PROBABILITY_FORWARD_YEARS,
)
probability_bundle["promotion_approved"] = False
probability_bundle_path, probability_manifest_path = export_celsius_probability_bundle(
    probability_bundle, OUTPUT_DIR / "celsius_market_probability",
    source_identity=SOURCE_IDENTITY,
    artifact_paths=(point_bundle_path, point_manifest_path),
)
probability_holdout, probability_holdout_metrics, probability_holdout_calibration = evaluate_celsius_probability_holdout(
    feature_frame, point_holdout, expert_holdout, probability_bundle, holdout_year=2026,
)
probability_forward.to_csv(OUTPUT_DIR / "probability_forward_predictions.csv", index=False)
probability_tuning.to_csv(OUTPUT_DIR / "probability_tuning.csv", index=False)
probability_holdout.to_csv(OUTPUT_DIR / "probability_holdout_2026.csv", index=False)
probability_holdout_calibration.to_csv(OUTPUT_DIR / "probability_holdout_2026_calibration.csv", index=False)
probability_holdout_metrics
"""


def _challenger_cell(config: dict[str, Any]) -> str:
    if not config["kdal_challenger"]:
        return "print('KDAL challenger is not part of this station contract.')"
    return """
from src.calibration.kdal_ordinal_challenger import (
    _inner_split, apply_no_override_policy, export_frozen_candidate,
    feature_sets, fit_and_predict, frozen_candidate_rows,
    nested_forward_evaluation, row_to_config, tune_candidates,
    tune_no_override_policy,
)

challenger_sets = feature_sets(PROBABILITY_PROFILE)
assert tuple(challenger_sets) == ("market_core_21", "compact_29", "full_61")
challenger_forward, challenger_selections, challenger_nested_tuning = nested_forward_evaluation(
    probability_frame, feature_profile=PROBABILITY_PROFILE,
)
challenger_train, challenger_valid = _inner_split(probability_frame)
challenger_tuning = tune_candidates(
    challenger_train, challenger_valid, feature_profile=PROBABILITY_PROFILE,
)
frozen_rows = frozen_candidate_rows(challenger_tuning)
challenger_export_rows = []
for _, frozen_row in frozen_rows.iterrows():
    role = str(frozen_row["candidate_role"])
    challenger_config = row_to_config(frozen_row)
    _, inner_predictions, _ = fit_and_predict(
        challenger_train, challenger_valid, challenger_config,
        feature_profile=PROBABILITY_PROFILE,
    )
    policy = tune_no_override_policy(inner_predictions)
    holdout_metrics, holdout_predictions, challenger_state = fit_and_predict(
        probability_frame, probability_holdout_frame, challenger_config,
        feature_profile=PROBABILITY_PROFILE,
    )
    holdout_predictions = apply_no_override_policy(holdout_predictions, policy)
    assert not holdout_predictions["overrides_point_bucket"].any()
    challenger_output = OUTPUT_DIR / "kdal_ordinal_challenger"
    challenger_output.mkdir(parents=True, exist_ok=True)
    holdout_predictions.to_csv(challenger_output / f"{role}_holdout_2026.csv", index=False)
    bundle_path, manifest_path = export_frozen_candidate(
        OUTPUT_DIR / "kdal_ordinal_challenger",
        station_id=STATION_ID,
        point_model_version=POINT_MODEL_VERSION,
        point_bundle_path=point_bundle_path,
        config=challenger_config,
        state=challenger_state,
        policy=policy,
        historical_metrics={"nested_pre_2026": challenger_selections.to_dict(orient="records"), "exploratory_2026": holdout_metrics},
        candidate_name=f"station_expert_ensemble_v1_{role}",
        feature_profile=PROBABILITY_PROFILE,
    )
    challenger_export_rows.append({"role": role, "bundle": bundle_path, "manifest": manifest_path})
challenger_forward.to_csv(OUTPUT_DIR / "kdal_challenger_forward.csv", index=False)
challenger_nested_tuning.to_csv(OUTPUT_DIR / "kdal_challenger_nested_tuning.csv", index=False)
pd.DataFrame(challenger_export_rows)
"""


def build_notebook(config: dict[str, Any]) -> dict[str, Any]:
    title = f"{config['city']} / {config['station_id']} — Four-Expert Station Ensemble V1"
    settings = f"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = next(path for path in (Path.cwd(), *Path.cwd().parents) if (path / "pyproject.toml").is_file())
STATION_ID = {config['station_id']!r}
CITY = {config['city']!r}
PROVIDERS = {tuple(config['providers'])!r}
EVALUATION_YEARS = {tuple(config['evaluation_years'])!r}
PROBABILITY_DEVELOPMENT_YEARS = {tuple(config['probability_development_years'])!r}
PROBABILITY_FORWARD_YEARS = {tuple(config['probability_forward_years'])!r}
PROBABILITY_PROFILE = {config['probability_profile']!r}
BUCKET_CONTRACT = {config['bucket_contract']!r}
OPTUNA_TRIALS = 30
OPTUNA_STARTUP_TRIALS = 15
HOLDOUT_YEAR = 2026
ENABLE_LIVE_REFIT = False
POINT_MODEL_VERSION = "station_expert_ensemble_v1_point_shadow"
OUTPUT_DIR = PROJECT_ROOT / "data" / "calibration" / "station_expert_ensemble_v1" / {config['artifact_subdir']!r}
FEATURES_PATH = PROJECT_ROOT / "data" / "calibration" / "station_training_baseline" / {config['feature_artifact_subdir']!r} / {config['feature_filename']!r}
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
assert not ENABLE_LIVE_REFIT
"""
    cells = [
        _markdown(f"# {title}\n\nResearch/shadow-only. This notebook does not promote or modify the active station baseline."),
        _markdown("""## Contract and chronology

The market bucket contract is explicit below. Every expert recomputes its 3% feature-missingness gate inside each outer fit and again on the exact pre-2026 final-refit population. All tuning, preprocessing, feature selection, blend weights, calibration, and policy thresholds are strictly forward. The 2026 holdout is exploratory and cannot tune any component. Native Celsius settlement is retained for Seoul and Tokyo. Optional live refits are disabled and must fail closed against frozen feature contracts."""),
        _code(settings),
        _markdown("## 1. Source feature construction and identity"),
        _code("""
if not FEATURES_PATH.is_file():
    raise FileNotFoundError(f"Generate the canonical baseline live-safe feature snapshot first: {FEATURES_PATH}")
feature_frame = pd.read_csv(FEATURES_PATH)
feature_frame["contract_date"] = pd.to_datetime(feature_frame["contract_date"], errors="coerce")
feature_frame = feature_frame.sort_values("contract_date").drop_duplicates("contract_date").reset_index(drop=True)
assert feature_frame["contract_date"].notna().all()
assert {"actual_high_f", "observed_high_temp_through_as_of_f", "provider_mean_high_f"}.issubset(feature_frame)
assert all(f"{provider}_high_f" in feature_frame for provider in PROVIDERS)
SOURCE_IDENTITY = {
    "source_pipeline": "notebooks/experiments/station_expert_ensemble_v1",
    "feature_snapshot": FEATURES_PATH.relative_to(PROJECT_ROOT).as_posix(),
    "station_id": STATION_ID,
    "providers": list(PROVIDERS),
}
feature_frame[["contract_date", "actual_high_f", "observed_high_temp_through_as_of_f"]].describe(include="all")
"""),
        _markdown("## 2. Quality, leakage, target, and fold contracts"),
        _code("""
from src.calibration.expert_ensemble import EXPERT_METHODS, route_expert_features, target_values

feature_routes = {method: route_expert_features(feature_frame, method) for method in EXPERT_METHODS}
assert not any("DIAGNOSTIC_ONLY".lower() in name.lower() for names in feature_routes.values() for name in names)
assert not any(name in {"actual_high_f", "settlement_high_f", "settlement_high_c"} for names in feature_routes.values() for name in names)
assert not any(name.endswith("_high_f") and not name.startswith("observed_") for name in feature_routes["observation_catboost"])
pd.DataFrame({"method": method, "possible_features": len(names)} for method, names in feature_routes.items())
"""),
        _markdown("## 3. Strictly forward expert training"),
        _code("""
from src.calibration.expert_ensemble import crossfit_experts

expert_oof, expert_missingness_audits = crossfit_experts(
    feature_frame,
    EVALUATION_YEARS,
    optuna_trials=OPTUNA_TRIALS,
    startup_trials=OPTUNA_STARTUP_TRIALS,
)
assert (pd.to_datetime(expert_oof["model_training_cutoff"]) < pd.to_datetime(expert_oof["contract_date"])).all()
assert (expert_oof["predicted_high_f"] >= expert_oof.merge(feature_frame[["contract_date", "observed_high_temp_through_as_of_f"]], on="contract_date")["observed_high_temp_through_as_of_f"]).all()
expert_oof.to_csv(OUTPUT_DIR / "expert_oof_predictions.csv", index=False)
expert_missingness_audits.to_json(OUTPUT_DIR / "expert_missingness_audits.json", orient="records", indent=2)
expert_missingness_audits[["validation_year", "method", "feature_count_before_gate", "feature_count_after_gate"]]
"""),
        _markdown("## 4. Four-way simplex blend with forward-only weights"),
        _code("""
from src.calibration.expert_ensemble import forward_simplex_predictions, select_frozen_weights

forward_blend, forward_weight_selections = forward_simplex_predictions(expert_oof)
frozen_weight_row = select_frozen_weights(expert_oof)
frozen_weights = {method: float(frozen_weight_row[f"{method}_weight"]) for method in EXPERT_METHODS}
assert all(weight >= 0 for weight in frozen_weights.values())
assert np.isclose(sum(frozen_weights.values()), 1.0)
forward_blend.to_csv(OUTPUT_DIR / "forward_blend_predictions.csv", index=False)
forward_weight_selections.to_csv(OUTPUT_DIR / "forward_simplex_weights.csv", index=False)
frozen_weights
"""),
        _markdown("## 5. Common-date comparison and paired uncertainty"),
        _code("""
def _half_up(values):
    return np.floor(pd.to_numeric(values, errors="coerce") + 0.5).astype("Int64")

def _bucket_indices(frame):
    predicted_f = pd.to_numeric(frame["predicted_high_f"], errors="coerce")
    actual_f = pd.to_numeric(frame["actual_high_f"], errors="coerce")
    if BUCKET_CONTRACT.endswith("2f"):
        return _half_up(predicted_f).floordiv(2), _half_up(actual_f).floordiv(2)
    predicted_c = (predicted_f - 32.0) * 5.0 / 9.0
    native_source = feature_frame.set_index("contract_date").get("actual_high_c", pd.Series(dtype=float))
    native_c = pd.to_datetime(frame["contract_date"]).map(native_source)
    actual_c = pd.to_numeric(native_c, errors="coerce").fillna((actual_f - 32.0) * 5.0 / 9.0)
    return _half_up(predicted_c), _half_up(actual_c)

def metric_row(method, frame):
    error = frame["predicted_high_f"] - frame["actual_high_f"]
    predicted_bucket, actual_bucket = _bucket_indices(frame)
    bucket_distance = (predicted_bucket - actual_bucket).abs()
    return {"method": method, "count": len(frame), "mae_f": error.abs().mean(), "rmse_f": np.sqrt(np.square(error).mean()), "bias_f": error.mean(), "exact_bucket_hit_rate": bucket_distance.eq(0).mean(), "within_one_bucket_rate": bucket_distance.le(1).mean()}

comparison_parts = [expert_oof, forward_blend]
wide = expert_oof.pivot(index="contract_date", columns="method", values="predicted_high_f")
equal_mean = wide.mean(axis=1).rename("predicted_high_f").reset_index().merge(feature_frame[["contract_date", "actual_high_f"]], on="contract_date")
equal_mean["method"] = "equal_expert_mean"
comparison_parts.append(equal_mean)
baseline_path = FEATURES_PATH.parent / f"{STATION_ID}_year_split_validation_predictions.csv"
if baseline_path.is_file():
    baseline = pd.read_csv(baseline_path)
    baseline["contract_date"] = pd.to_datetime(baseline["contract_date"])
    baseline = baseline.loc[baseline["method"].eq("ridge_stack")]
    comparison_parts.append(baseline)
comparison_predictions = pd.concat(comparison_parts, ignore_index=True, sort=False)
common_dates = set.intersection(*(set(pd.to_datetime(part["contract_date"])) for part in comparison_parts))
common = comparison_predictions.loc[pd.to_datetime(comparison_predictions["contract_date"]).isin(common_dates)].copy()
comparison_metrics = pd.DataFrame(metric_row(method, part) for method, part in common.groupby("method"))
monthly_stability = pd.DataFrame(
    metric_row(method, month_part) | {"month": int(month)}
    for (method, month), month_part in common.assign(month=pd.to_datetime(common["contract_date"]).dt.month).groupby(["method", "month"])
)
error_wide = common.assign(error_f=common["predicted_high_f"] - common["actual_high_f"]).pivot(index="contract_date", columns="method", values="error_f")
error_correlation = error_wide.corr()
blend_errors = forward_blend.set_index("contract_date")["predicted_high_f"] - forward_blend.set_index("contract_date")["actual_high_f"]
rng = np.random.default_rng(42)
paired_uncertainty = []
for method, part in expert_oof.groupby("method"):
    competitor = part.set_index("contract_date")["predicted_high_f"] - part.set_index("contract_date")["actual_high_f"]
    paired = pd.concat([blend_errors.abs().rename("blend"), competitor.abs().rename("competitor")], axis=1).dropna()
    samples = [float((sample["blend"] - sample["competitor"]).mean()) for sample in (paired.iloc[rng.integers(0, len(paired), len(paired))] for _ in range(2000))]
    paired_uncertainty.append({"competitor": method, "mae_delta_f": float((paired["blend"] - paired["competitor"]).mean()), "ci_low_f": np.quantile(samples, .025), "ci_high_f": np.quantile(samples, .975)})
comparison_metrics.to_csv(OUTPUT_DIR / "common_date_comparison_metrics.csv", index=False)
monthly_stability.to_csv(OUTPUT_DIR / "monthly_stability.csv", index=False)
error_correlation.to_csv(OUTPUT_DIR / "error_correlation.csv")
pd.DataFrame(paired_uncertainty).to_csv(OUTPUT_DIR / "paired_uncertainty.csv", index=False)
comparison_metrics
"""),
        _markdown("## 6. Frozen pre-2026 evaluation refit and exploratory 2026 holdout"),
        _code("""
from src.calibration.expert_ensemble import fit_final_experts
from src.calibration.constrained_blend import merge_multiple_prediction_sources, blend_simplex_predictions

final_experts = fit_final_experts(
    feature_frame, through_year=2025,
    optuna_trials=OPTUNA_TRIALS, startup_trials=OPTUNA_STARTUP_TRIALS,
)
holdout_frame = feature_frame.loc[feature_frame["contract_date"].dt.year.eq(HOLDOUT_YEAR)].copy()
expert_holdout_parts = []
for method, expert in final_experts.items():
    part = holdout_frame[["contract_date", "actual_high_f"]].copy()
    part["method"] = method
    part["predicted_high_f"] = expert.predict(holdout_frame)
    part["evaluation_scope"] = "exploratory_2026"
    part["fold"] = "holdout_2026"
    expert_holdout_parts.append(part)
expert_holdout = pd.concat(expert_holdout_parts, ignore_index=True)
holdout_wide = merge_multiple_prediction_sources({method: expert_holdout for method in EXPERT_METHODS})
point_holdout = blend_simplex_predictions(
    holdout_wide, methods=EXPERT_METHODS,
    weights=tuple(frozen_weights[method] for method in EXPERT_METHODS),
    method="four_expert_simplex_blend",
)
expert_holdout.to_csv(OUTPUT_DIR / "expert_holdout_2026.csv", index=False)
point_holdout.to_csv(OUTPUT_DIR / "point_holdout_2026.csv", index=False)
pd.DataFrame(metric_row(method, part) for method, part in pd.concat([expert_holdout, point_holdout]).groupby("method"))
"""),
        _markdown("## 7. Research-only point bundle export"),
        _code("""
from src.calibration.expert_ensemble import export_point_bundle, sha256_file

STATION_CONTRACT = {
    "providers": list(PROVIDERS),
    "market_bucket_contract": BUCKET_CONTRACT,
    "native_settlement_units_preserved": BUCKET_CONTRACT.endswith("1c"),
    "missingness_gate": 0.03,
}
CHRONOLOGY = {
    "evaluation_years": list(EVALUATION_YEARS),
    "frozen_training_through_year": 2025,
    "holdout_year": 2026,
    "holdout_status": "exploratory_only",
    "selection_excludes_holdout": True,
}
point_bundle_path, point_manifest_path = export_point_bundle(
    OUTPUT_DIR / "model_weights",
    station_id=STATION_ID,
    model_version=POINT_MODEL_VERSION,
    experts=final_experts,
    weights=frozen_weights,
    station_contract=STATION_CONTRACT,
    source_identity=SOURCE_IDENTITY,
    chronology=CHRONOLOGY,
)
point_manifest = json.loads(point_manifest_path.read_text(encoding="utf-8"))
assert point_manifest["artifact_integrity"]["bundle_sha256"] == sha256_file(point_bundle_path)
point_manifest_path
"""),
        _markdown("## 8. Linked ordinal probability model (shadow only)"),
        _code(_probability_cell(config)),
        _markdown("## 9. KDAL three-arm ordinal challenger"),
        _code(_challenger_cell(config)),
        _markdown("## 10. Completion contract"),
        _code("""
assert not ENABLE_LIVE_REFIT
assert point_bundle_path.is_file() and point_manifest_path.is_file()
assert probability_bundle_path.is_file() and probability_manifest_path.is_file()
print({"station": STATION_ID, "status": "research_complete", "promotion_approved": False, "holdout": "exploratory_2026"})
"""),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            "station_expert_ensemble": {
                "station_id": config["station_id"],
                "providers": config["providers"],
                "source_pipeline": "notebooks/experiments/station_expert_ensemble_v1",
                "research_only": True,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def generate(config_name: str) -> Path:
    config_path = CONFIG_DIR / f"{config_name}.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    destination = HERE / config["notebook"]
    destination.write_text(json.dumps(build_notebook(config), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stations", nargs="*", default=["KDAL", "Seoul", "Tokyo"])
    args = parser.parse_args()
    for station in args.stations:
        print(generate(station))


if __name__ == "__main__":
    main()
