from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

import pandas as pd

from src.calibration.asia_station_stacking import build_asia_station_wide_dataset
from src.calibration.station_probability_models import (
    GAUSSIAN_ARTIFACT_TYPE,
    ORDINAL_MEMBER_ROLES,
    ORDINAL_ARTIFACT_TYPE,
    build_probability_frame,
    export_ordinal_ensemble_manifest,
    export_probability_artifact,
    fit_production_probability_models,
    ordinal_ensemble_predictions,
    probability_metrics,
    probability_predictions,
    run_probability_walk_forward,
    sha256_file,
)
from src.calibration.station_stacking import (
    StationStackingConfig,
    YearSplitFold,
    build_station_wide_dataset,
    run_station_year_split_experiment,
)
from src.export_station_stacking_v2_models import export_station_model_weights


ARCHITECTURE_VERSION = "station_training_baseline_xgboost_probability_v2"
POINT_METHOD = "xgboost"
DIRECT_NBM_ENV = "WEATHER_RESEARCH_INCLUDE_DIRECT_NBM"


@dataclass(frozen=True)
class StationBaselineRun:
    station_id: str
    config: dict[str, Any]
    output_dir: Path
    point_scoreboard: pd.DataFrame
    probability_comparison: pd.DataFrame
    artifact_paths: dict[str, Path]
    report_paths: dict[str, Path]


def load_station_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    config = json.loads(source.read_text(encoding="utf-8"))
    required = {
        "station_id",
        "station_name",
        "notebook_path",
        "feature_builder",
        "providers",
        "feature_version",
        "timing_mode",
        "target_mode",
        "target_source",
        "timezone",
        "probability_unit",
        "market_bucket_width",
        "ordinal_tail_offset",
        "year_split_folds",
        "point_evaluation_train_years",
        "point_test_year",
        "probability_development_years",
        "probability_validation_years",
        "point_evaluation_model_version",
        "point_production_model_version",
        "gaussian_evaluation_model_version",
        "gaussian_production_model_version",
        "ordinal_candidate_model_versions",
        "optuna_trials",
        "optuna_startup_trials",
        "point_nested_chronological_oos",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError("station baseline config missing: " + ",".join(missing))
    station = str(config["station_id"]).strip().upper()
    if station not in {"KDAL", "RJTT", "RKSI", "RKPK"}:
        raise ValueError(f"unsupported baseline station: {station}")
    config["station_id"] = station
    if int(config["optuna_trials"]) != 100 or int(config["optuna_startup_trials"]) != 40:
        raise ValueError("baseline Optuna contract requires 100 trials and 40 startup trials")
    if config["point_nested_chronological_oos"] is not True:
        raise ValueError("station baseline requires nested chronological point OOS lineage")
    if str(config["notebook_path"]) != f"stations/{station}/train_{station}.ipynb":
        raise ValueError("notebook path must use the station-code naming contract")
    versions = config["ordinal_candidate_model_versions"]
    expected_roles = {"native_ordinal_reference", *ORDINAL_MEMBER_ROLES}
    if set(versions) != expected_roles:
        raise ValueError(
            "ordinal candidate model versions must define exactly: "
            + ",".join(sorted(expected_roles))
        )
    for role, role_versions in versions.items():
        if set(role_versions) != {"evaluation", "production"}:
            raise ValueError(f"ordinal candidate {role} must define evaluation and production versions")
    folds = config["year_split_folds"]
    if not isinstance(folds, list) or len(folds) < 2:
        raise ValueError("year_split_folds must contain at least two chronological folds")
    last_validation = -float("inf")
    for fold in folds:
        if not isinstance(fold, Mapping) or set(fold) != {"name", "train_start_year", "train_end_year", "validation_year"}:
            raise ValueError("each year_split_fold must contain exactly name/train_start_year/train_end_year/validation_year")
        start, end, validation = (int(fold[key]) for key in ("train_start_year", "train_end_year", "validation_year"))
        if start > end or end >= validation or validation <= last_validation:
            raise ValueError("year_split_folds must be strictly chronological with training before validation")
        last_validation = validation
    if int(config["point_test_year"]) <= last_validation:
        raise ValueError("point_test_year must be after every validation fold")
    return config


def build_station_features(config: Mapping[str, Any], project_root: str | Path) -> pd.DataFrame:
    root = Path(project_root).resolve()
    providers = tuple(str(value) for value in config["providers"])
    if config["feature_builder"] == "asia_11am":
        frame = build_asia_station_wide_dataset(
            root / "data" / "calibration" / "asia_11am",
            str(config["city_id"]),
            feature_version=str(config["feature_version"]),
            providers=providers,
        )
    elif config["feature_builder"] == "us_station":
        previous_direct_nbm = os.environ.get(DIRECT_NBM_ENV)
        if "nbm" in providers:
            os.environ[DIRECT_NBM_ENV] = "1"
        try:
            frame = build_station_wide_dataset(
                root,
                station_id=str(config["station_id"]),
                timing_mode=str(config["timing_mode"]),
                providers=providers,
                feature_version=str(config["feature_version"]),
                target_source=str(config["target_source"]),
            )
        finally:
            if previous_direct_nbm is None:
                os.environ.pop(DIRECT_NBM_ENV, None)
            else:
                os.environ[DIRECT_NBM_ENV] = previous_direct_nbm
    else:
        raise ValueError(f"unknown feature builder: {config['feature_builder']}")
    if frame.empty:
        raise ValueError(f"no features available for {config['station_id']}")
    missing_providers = [
        provider
        for provider in providers
        if f"{provider}_high_f" not in frame
        or not pd.to_numeric(frame[f"{provider}_high_f"], errors="coerce").notna().any()
    ]
    if missing_providers:
        raise ValueError(
            f"missing provider features for {config['station_id']}: {','.join(missing_providers)}"
        )
    return frame


def point_training_config(
    config: Mapping[str, Any],
    project_root: str | Path,
    features: pd.DataFrame,
) -> StationStackingConfig:
    output = _output_dir(config, project_root)
    folds = tuple(
        YearSplitFold(
            str(item["name"]),
            int(item["train_start_year"]),
            int(item["train_end_year"]),
            int(item["validation_year"]),
        )
        for item in config["year_split_folds"]
    )
    evaluation_years = tuple(int(value) for value in config["point_evaluation_train_years"])
    return StationStackingConfig(
        station_id=str(config["station_id"]),
        project_root=Path(project_root).resolve(),
        timing_mode=str(config["timing_mode"]),
        providers=tuple(str(value) for value in config["providers"]),
        optuna_trials=int(config["optuna_trials"]),
        optuna_startup_trials=int(config["optuna_startup_trials"]),
        optuna_metric="mae_f",
        optuna_verbose=bool(config.get("optuna_verbose", False)),
        optuna_storage_path=output / f"{config['station_id']}_xgboost_optuna.sqlite3",
        feature_version=str(config["feature_version"]),
        training_profile="v20_aligned",
        target_mode=str(config["target_mode"]),
        target_source=str(config["target_source"]),
        max_feature_missing_fraction=float(config.get("max_feature_missing_fraction", 0.03)),
        base_model_methods=(POINT_METHOD,),
        stack_enabled=False,
        hyperparameter_space=str(config.get("hyperparameter_space", "wide")),
        year_split_folds=folds,
        year_split_validation_weights={fold.validation_year: 1.0 for fold in folds},
        year_split_test_train_years=evaluation_years,
        year_split_test_year=int(config["point_test_year"]),
        feature_importance_repeats=1,
        output_dir=output,
        observation_target_same_station=True,
        observation_source=str(config.get("observation_source", "default")),
        prebuilt_features=features,
        nested_chronological_oos=bool(config["point_nested_chronological_oos"]),
    )


def run_station_baseline(
    config_path: str | Path,
    *,
    project_root: str | Path = ".",
    export_production: bool = True,
) -> StationBaselineRun:
    root = Path(project_root).resolve()
    config = load_station_config(config_path)
    output = _output_dir(config, root)
    output.mkdir(parents=True, exist_ok=True)
    reports = output / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    features = build_station_features(config, root)
    point_config = point_training_config(config, root, features)
    point_result = run_station_year_split_experiment(point_config)
    validation_point = _point_predictions(point_result.validation_predictions)
    test_point = _point_predictions(point_result.test_predictions)
    if validation_point.empty or test_point.empty:
        raise ValueError(f"missing XGBoost validation/test predictions for {config['station_id']}")

    evaluation_export = _export_point_model(
        config,
        root,
        point_config,
        model_version=str(config["point_evaluation_model_version"]),
        train_years=tuple(int(value) for value in config["point_evaluation_train_years"]),
        model_dir=output / "model_weights" / "evaluation",
    )
    evaluation_manifest = json.loads(evaluation_export.manifest_path.read_text(encoding="utf-8"))
    frozen_features = list(evaluation_manifest["features"]["all"])

    production_export = None
    if export_production:
        production_export = _export_point_model(
            config,
            root,
            point_config,
            model_version=str(config["point_production_model_version"]),
            train_years=None,
            model_dir=output / "model_weights" / "production",
            frozen_feature_names=frozen_features,
        )

    unit = str(config["probability_unit"]).upper()
    bucket_width = int(config["market_bucket_width"])
    providers = tuple(str(value) for value in config["providers"])
    validation_frame = build_probability_frame(
        features,
        validation_point,
        providers=providers,
        unit=unit,
        bucket_width=bucket_width,
    )
    probability_run = run_probability_walk_forward(
        validation_frame,
        station_id=str(config["station_id"]),
        development_years=config["probability_development_years"],
        validation_years=config["probability_validation_years"],
        tail=int(config["ordinal_tail_offset"]),
        unit=unit,
        bucket_width=bucket_width,
    )
    holdout_frame = build_probability_frame(
        features,
        test_point,
        providers=providers,
        unit=unit,
        bucket_width=bucket_width,
    )
    gaussian_holdout = probability_predictions(
        holdout_frame,
        family="gaussian",
        state=probability_run.gaussian_state,
        unit=unit,
        bucket_width=bucket_width,
        period=f"holdout_{config['point_test_year']}",
    )
    native_ordinal_holdout = probability_predictions(
        holdout_frame,
        family="native_ordinal_reference",
        state=probability_run.ordinal_states["native_ordinal_reference"],
        unit=unit,
        bucket_width=bucket_width,
        period=f"holdout_{config['point_test_year']}",
    )
    ordinal_member_holdout, ordinal_ensemble_holdout = ordinal_ensemble_predictions(
        holdout_frame,
        probability_run.ordinal_states,
        unit=unit,
        bucket_width=bucket_width,
        period=f"holdout_{config['point_test_year']}",
    )
    all_predictions = pd.concat(
        [
            probability_run.forward_predictions,
            gaussian_holdout,
            native_ordinal_holdout,
            ordinal_member_holdout,
            ordinal_ensemble_holdout,
        ],
        ignore_index=True,
    )
    comparison = _comparison_metrics(str(config["station_id"]), all_predictions)
    tuning = probability_run.tuning.copy()

    development = validation_frame.loc[
        validation_frame["year"].isin([int(value) for value in config["probability_development_years"]])
    ].copy()
    source_identity = _source_identity(root, config)
    evaluation_metrics = _records(comparison.loc[comparison["period"].str.startswith("forward_")])
    artifact_paths: dict[str, Path] = {
        "point_evaluation_bundle": evaluation_export.bundle_path,
        "point_evaluation_manifest": evaluation_export.manifest_path,
    }
    gaussian_evaluation = export_probability_artifact(
        probability_run.gaussian_state,
        output / "probability" / "gaussian" / "evaluation",
        artifact_type=GAUSSIAN_ARTIFACT_TYPE,
        station_id=str(config["station_id"]),
        model_version=str(config["gaussian_evaluation_model_version"]),
        point_model_version=str(config["point_evaluation_model_version"]),
        point_bundle_sha256=sha256_file(evaluation_export.bundle_path),
        unit=unit,
        bucket_width=bucket_width,
        training_frame=development,
        validation_metrics=evaluation_metrics,
        source_identity=source_identity,
        release_role="frozen_evaluation",
    )
    ordinal_evaluation_artifacts: dict[str, tuple[Path, Path]] = {}
    for role, state in probability_run.ordinal_states.items():
        ordinal_evaluation_artifacts[role] = export_probability_artifact(
            state,
            output / "ordinal_candidates" / "evaluation" / role,
            artifact_type=ORDINAL_ARTIFACT_TYPE,
            station_id=str(config["station_id"]),
            model_version=str(config["ordinal_candidate_model_versions"][role]["evaluation"]),
            point_model_version=str(config["point_evaluation_model_version"]),
            point_bundle_sha256=sha256_file(evaluation_export.bundle_path),
            unit=unit,
            bucket_width=bucket_width,
            training_frame=development,
            validation_metrics=[
                value
                for value in evaluation_metrics
                if value.get("family") in {role, "ordinal_ensemble_median"}
            ],
            source_identity=source_identity,
            release_role="frozen_evaluation",
        )
    evaluation_ensemble_manifest = export_ordinal_ensemble_manifest(
        output / "ordinal_ensemble" / "evaluation_manifest.json",
        station_id=str(config["station_id"]),
        point_model_version=str(config["point_evaluation_model_version"]),
        point_bundle_sha256=sha256_file(evaluation_export.bundle_path),
        unit=unit,
        bucket_width=bucket_width,
        member_artifacts=ordinal_evaluation_artifacts,
        source_identity=source_identity,
        release_role="frozen_evaluation",
    )
    artifact_paths.update(
        {
            "gaussian_evaluation_bundle": gaussian_evaluation[0],
            "gaussian_evaluation_manifest": gaussian_evaluation[1],
            "ordinal_ensemble_evaluation_manifest": evaluation_ensemble_manifest,
        }
    )
    for role, (bundle_path, manifest_path) in ordinal_evaluation_artifacts.items():
        artifact_paths[f"{role}_evaluation_bundle"] = bundle_path
        artifact_paths[f"{role}_evaluation_manifest"] = manifest_path

    production_tuning = pd.DataFrame()
    if production_export is not None:
        artifact_paths.update(
            {
                "point_production_bundle": production_export.bundle_path,
                "point_production_manifest": production_export.manifest_path,
            }
        )
        production_points = pd.concat([validation_point, test_point], ignore_index=True).drop_duplicates(
            "contract_date", keep="last"
        )
        production_frame = build_probability_frame(
            features,
            production_points,
            providers=providers,
            unit=unit,
            bucket_width=bucket_width,
        )
        gaussian_production_state, ordinal_production_states, production_tuning = (
            fit_production_probability_models(
                production_frame,
                tail=int(config["ordinal_tail_offset"]),
                unit=unit,
                bucket_width=bucket_width,
            )
        )
        gaussian_production = export_probability_artifact(
            gaussian_production_state,
            output / "probability" / "gaussian" / "production",
            artifact_type=GAUSSIAN_ARTIFACT_TYPE,
            station_id=str(config["station_id"]),
            model_version=str(config["gaussian_production_model_version"]),
            point_model_version=str(config["point_production_model_version"]),
            point_bundle_sha256=sha256_file(production_export.bundle_path),
            unit=unit,
            bucket_width=bucket_width,
            training_frame=production_frame,
            validation_metrics=[],
            external_evaluation_evidence=evaluation_metrics,
            source_identity=source_identity,
            release_role="live_production_candidate",
        )
        ordinal_production_artifacts: dict[str, tuple[Path, Path]] = {}
        for role, state in ordinal_production_states.items():
            ordinal_production_artifacts[role] = export_probability_artifact(
                state,
                output / "ordinal_candidates" / "production" / role,
                artifact_type=ORDINAL_ARTIFACT_TYPE,
                station_id=str(config["station_id"]),
                model_version=str(config["ordinal_candidate_model_versions"][role]["production"]),
                point_model_version=str(config["point_production_model_version"]),
                point_bundle_sha256=sha256_file(production_export.bundle_path),
                unit=unit,
                bucket_width=bucket_width,
                training_frame=production_frame,
                validation_metrics=[],
                external_evaluation_evidence=[
                    value
                    for value in evaluation_metrics
                    if value.get("family") in {role, "ordinal_ensemble_median"}
                ],
                source_identity=source_identity,
                release_role="live_production_candidate",
            )
        production_ensemble_manifest = export_ordinal_ensemble_manifest(
            output / "ordinal_ensemble" / "production_manifest.json",
            station_id=str(config["station_id"]),
            point_model_version=str(config["point_production_model_version"]),
            point_bundle_sha256=sha256_file(production_export.bundle_path),
            unit=unit,
            bucket_width=bucket_width,
            member_artifacts=ordinal_production_artifacts,
            source_identity=source_identity,
            release_role="live_production_candidate",
        )
        artifact_paths.update(
            {
                "gaussian_production_bundle": gaussian_production[0],
                "gaussian_production_manifest": gaussian_production[1],
                "ordinal_ensemble_production_manifest": production_ensemble_manifest,
            }
        )
        for role, (bundle_path, manifest_path) in ordinal_production_artifacts.items():
            artifact_paths[f"{role}_production_bundle"] = bundle_path
            artifact_paths[f"{role}_production_manifest"] = manifest_path

    prediction_path = reports / f"{config['station_id']}_probability_predictions.csv"
    comparison_path = reports / f"{config['station_id']}_probability_comparison.csv"
    monthly_path = reports / f"{config['station_id']}_monthly_probability_metrics.csv"
    agreement_path = reports / f"{config['station_id']}_ordinal_member_agreement.csv"
    gate_path = reports / f"{config['station_id']}_ordinal_gate_metrics.csv"
    trading_input_path = reports / f"{config['station_id']}_trading_backtest_input.csv"
    tuning_path = reports / f"{config['station_id']}_probability_tuning.csv"
    summary_path = reports / f"{config['station_id']}_baseline_summary.json"
    candidate_comparison_path = (
        output / "ordinal_candidates" / "evaluation" / "candidate_comparison.json"
    )
    all_predictions.to_csv(prediction_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    monthly_metrics = _monthly_probability_metrics(str(config["station_id"]), all_predictions)
    member_agreement = _ordinal_member_agreement(all_predictions)
    gate_metrics = _ordinal_gate_metrics(str(config["station_id"]), all_predictions)
    monthly_metrics.to_csv(monthly_path, index=False)
    member_agreement.to_csv(agreement_path, index=False)
    gate_metrics.to_csv(gate_path, index=False)
    all_predictions.reindex(
        columns=[
            "contract_date",
            "period",
            "family",
            "actual_market_bucket",
            "point_market_bucket",
            "point_market_probability",
            "top_market_bucket",
            "top_market_probability",
            "top_two_margin",
            "ordinal_votes",
            "ordinal_approved",
            "market_bucket_probabilities",
        ]
    ).to_csv(trading_input_path, index=False)
    pd.concat([tuning, production_tuning], ignore_index=True).to_csv(tuning_path, index=False)
    candidate_comparison_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_comparison_path.write_text(
        json.dumps(
            {
                "architecture_version": ARCHITECTURE_VERSION,
                "station_id": config["station_id"],
                "reference_role": "native_ordinal_reference",
                "voting_roles": list(ORDINAL_MEMBER_ROLES),
                "canonical_ensemble": "ordinal_ensemble_median",
                "comparison": _records(
                    comparison.loc[
                        comparison["family"].isin(
                            [
                                "native_ordinal_reference",
                                *ORDINAL_MEMBER_ROLES,
                                "ordinal_ensemble_median",
                            ]
                        )
                    ]
                ),
                "agreement": _records(member_agreement),
                "gate": _records(gate_metrics),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    artifact_paths["ordinal_candidate_comparison"] = candidate_comparison_path
    summary = {
        "architecture_version": ARCHITECTURE_VERSION,
        "station_id": config["station_id"],
        "point_model": POINT_METHOD,
        "optuna": {
            "trials": int(config["optuna_trials"]),
            "startup_trials": int(config["optuna_startup_trials"]),
            "sampler": "TPESampler",
        },
        "probability_models": {
            "benchmark": "conditional_gaussian_residual",
            "ordinal_candidates": [
                "native_ordinal_reference",
                *ORDINAL_MEMBER_ROLES,
            ],
            "canonical_ensemble": "ordinal_ensemble_median",
            "voting_policy": "two_of_three",
        },
        "comparison": _records(comparison),
        "artifacts": {key: str(path) for key, path in artifact_paths.items()},
        "source_identity": source_identity,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return StationBaselineRun(
        station_id=str(config["station_id"]),
        config=config,
        output_dir=output,
        point_scoreboard=point_result.scoreboard,
        probability_comparison=comparison,
        artifact_paths=artifact_paths,
        report_paths={
            "predictions": prediction_path,
            "comparison": comparison_path,
            "monthly_probability_metrics": monthly_path,
            "ordinal_member_agreement": agreement_path,
            "ordinal_gate_metrics": gate_path,
            "trading_backtest_input": trading_input_path,
            "ordinal_candidate_comparison": candidate_comparison_path,
            "tuning": tuning_path,
            "summary": summary_path,
        },
    )


def _point_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame.loc[frame["method"].eq(POINT_METHOD)].copy()
    if selected.empty:
        return selected
    selected["contract_date"] = pd.to_datetime(selected["contract_date"], errors="coerce")
    selected = selected.dropna(subset=["contract_date", "actual_high_f", "predicted_high_f"])
    selected["validation_year"] = selected["contract_date"].dt.year
    return selected.sort_values("contract_date").drop_duplicates("contract_date", keep="last")


def _export_point_model(
    config: Mapping[str, Any],
    root: Path,
    point_config: StationStackingConfig,
    *,
    model_version: str,
    train_years: tuple[int, int] | None,
    model_dir: Path,
    frozen_feature_names: list[str] | None = None,
):
    return export_station_model_weights(
        project_root=root,
        station_id=str(config["station_id"]),
        artifact_dir=point_config.resolved_output_dir(),
        model_dir=model_dir,
        train_years=train_years,
        model_version=model_version,
        timing_mode=point_config.timing_mode,
        providers=tuple(point_config.providers),
        feature_version=point_config.effective_feature_version,
        training_profile=point_config.effective_training_profile,
        optuna_metric=point_config.effective_optuna_metric,
        target_mode=point_config.effective_target_mode,
        target_source=point_config.effective_target_source,
        base_model_methods=(POINT_METHOD,),
        stack_enabled=False,
        source_pipeline=f"notebooks/station_training_baseline/{config['notebook_path']}",
        feature_pipeline=str(config["feature_version"]),
        max_feature_missing_fraction=point_config.effective_max_feature_missing_fraction,
        bucket_contract=str(config["point_bucket_contract"]),
        observation_target_same_station=True,
        observation_source=str(config.get("observation_source", "default")),
        city_id=config.get("city_id"),
        frozen_feature_names=frozen_feature_names,
        feature_contract_source={
            "architecture_version": ARCHITECTURE_VERSION,
            "station_config": f"notebooks/station_training_baseline/configs/{config['station_id']}.json",
        },
        release_role=("frozen_evaluation" if train_years is not None else "live_production_candidate"),
        approval_status=("frozen_evaluation_artifact" if train_years is not None else "unapproved_production_candidate"),
        year_split_folds=point_config.effective_year_split_folds,
        year_split_validation_weights=point_config.effective_year_split_validation_weights,
        year_split_test_train_years=point_config.effective_year_split_test_train_years,
        year_split_test_year=point_config.effective_year_split_test_year,
    )


def _comparison_metrics(station_id: str, predictions: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"station_id": station_id, "period": period, "family": family, **probability_metrics(group)}
            for (period, family), group in predictions.groupby(["period", "family"], sort=True)
        ]
    )


def _monthly_probability_metrics(station_id: str, predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame["month"] = pd.to_datetime(frame["contract_date"], errors="coerce").dt.to_period("M").astype(str)
    return pd.DataFrame(
        [
            {
                "station_id": station_id,
                "month": month,
                "family": family,
                **probability_metrics(group),
            }
            for (month, family), group in frame.groupby(["month", "family"], sort=True)
        ]
    )


def _ordinal_member_agreement(predictions: pd.DataFrame) -> pd.DataFrame:
    roles = ["native_ordinal_reference", *ORDINAL_MEMBER_ROLES]
    selected = predictions.loc[predictions["family"].isin(roles)].copy()
    rows: list[dict[str, Any]] = []
    for period, period_frame in selected.groupby("period", sort=True):
        by_role = {
            role: group.set_index("contract_date")
            for role, group in period_frame.groupby("family", sort=True)
        }
        for left_index, left_role in enumerate(roles):
            for right_role in roles[left_index + 1 :]:
                if left_role not in by_role or right_role not in by_role:
                    continue
                merged = by_role[left_role][
                    ["top_market_bucket", "point_market_probability"]
                ].join(
                    by_role[right_role][["top_market_bucket", "point_market_probability"]],
                    how="inner",
                    lsuffix="_left",
                    rsuffix="_right",
                )
                if merged.empty:
                    continue
                rows.append(
                    {
                        "period": period,
                        "left_role": left_role,
                        "right_role": right_role,
                        "count": int(len(merged)),
                        "top_bucket_agreement": float(
                            merged["top_market_bucket_left"].eq(
                                merged["top_market_bucket_right"]
                            ).mean()
                        ),
                        "point_probability_correlation": float(
                            merged["point_market_probability_left"].corr(
                                merged["point_market_probability_right"]
                            )
                        ),
                        "point_probability_mean_absolute_difference": float(
                            (
                                merged["point_market_probability_left"]
                                - merged["point_market_probability_right"]
                            ).abs().mean()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _ordinal_gate_metrics(station_id: str, predictions: pd.DataFrame) -> pd.DataFrame:
    selected = predictions.loc[predictions["family"].eq("ordinal_ensemble_median")].copy()
    rows: list[dict[str, Any]] = []
    for period, group in selected.groupby("period", sort=True):
        approved = group["ordinal_approved"].fillna(False).astype(bool)
        point_hit = group["point_market_bucket"].eq(group["actual_market_bucket"])
        rows.append(
            {
                "station_id": station_id,
                "period": period,
                "count": int(len(group)),
                "approved_count": int(approved.sum()),
                "approval_coverage": float(approved.mean()),
                "mean_votes": float(group["ordinal_votes"].mean()),
                "point_bucket_accuracy": float(point_hit.mean()),
                "approved_point_bucket_accuracy": (
                    float(point_hit.loc[approved].mean()) if approved.any() else None
                ),
                "ensemble_top_bucket_accuracy": float(group["top_market_hit"].mean()),
                "approved_ensemble_top_bucket_accuracy": (
                    float(group.loc[approved, "top_market_hit"].mean())
                    if approved.any()
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


def _source_identity(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
    )
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "architecture_version": ARCHITECTURE_VERSION,
        "notebook": f"notebooks/station_training_baseline/{config['notebook_path']}",
        "config": f"notebooks/station_training_baseline/configs/{config['station_id']}.json",
    }


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def _output_dir(config: Mapping[str, Any], project_root: str | Path) -> Path:
    return (
        Path(project_root).resolve()
        / "data"
        / "calibration"
        / "station_training_baseline"
        / str(config["station_id"])
    )
