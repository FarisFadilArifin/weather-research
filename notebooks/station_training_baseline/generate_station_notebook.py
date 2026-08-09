from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


BASELINE_ROOT = Path(__file__).resolve().parent
NOTEBOOKS_ROOT = BASELINE_ROOT.parent


def _markdown(source: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def _code(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "station_id",
        "station_name",
        "base_generator",
        "base_builder",
        "notebook_path",
        "artifact_subdir",
        "point_workflow_label",
        "point_source_model_version",
        "point_source_output_token",
        "point_source_pipeline_token",
        "point_model_version",
        "point_evaluation_train_years",
        "point_bucket_contract",
        "point_max_feature_missing_fraction",
        "point_live_model_version",
        "probability_model_label",
        "probability_model_version",
        "probability_feature_profile",
        "probability_feature_count",
        "probability_providers",
        "probability_development_years",
        "probability_forward_validation_years",
        "probability_holdout_year",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError("station config is missing: " + ", ".join(missing))
    config["station_id"] = str(config["station_id"]).strip().upper()
    for setting in (
        "probability_providers",
        "probability_development_years",
        "probability_forward_validation_years",
    ):
        if not config[setting]:
            raise ValueError(f"{setting} must not be empty")
    evaluation_years = tuple(
        int(year) for year in config["point_evaluation_train_years"]
    )
    if len(evaluation_years) != 2 or evaluation_years[0] > evaluation_years[1]:
        raise ValueError(
            "point_evaluation_train_years must be [START_YEAR, END_YEAR]"
        )
    if evaluation_years[1] != int(config["probability_holdout_year"]) - 1:
        raise ValueError(
            "point evaluation training must end in the year before the holdout"
        )
    config["point_evaluation_train_years"] = list(evaluation_years)
    bucket_contract = str(config["point_bucket_contract"]).strip().lower()
    if bucket_contract not in {
        "polymarket_half_up_1c",
        "polymarket_half_up_2f",
        "floor_1c",
    }:
        raise ValueError("point_bucket_contract is unsupported")
    config["point_bucket_contract"] = bucket_contract
    missingness_threshold = float(config["point_max_feature_missing_fraction"])
    if not 0.0 <= missingness_threshold <= 0.03:
        raise ValueError(
            "point_max_feature_missing_fraction must be between 0 and 0.03"
        )
    config["point_max_feature_missing_fraction"] = missingness_threshold
    live_version = str(config["point_live_model_version"]).strip()
    if not live_version or live_version == str(config["point_model_version"]).strip():
        raise ValueError("point_live_model_version must be non-empty and distinct")
    config["point_live_model_version"] = live_version
    return config


def _load_base_notebook(config: dict[str, Any]) -> dict[str, Any]:
    source = (NOTEBOOKS_ROOT / config["base_generator"]).resolve()
    spec = importlib.util.spec_from_file_location(
        f"station_training_source_{config['station_id'].lower()}",
        source,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load station notebook generator from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    builder = getattr(module, config["base_builder"])
    argument = config.get("base_builder_argument")
    notebook = builder() if argument is None else builder(argument)
    if not isinstance(notebook, dict) or "cells" not in notebook:
        raise TypeError(f"{source} did not return a notebook dictionary")
    return notebook


def _replace_in_cells(notebook: dict[str, Any], old: str, new: str) -> int:
    replacements = 0
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        replacements += source.count(old)
        cell["source"] = source.replace(old, new).splitlines(keepends=True)
    return replacements


def _replace_required(
    notebook: dict[str, Any],
    old: str,
    new: str,
    *,
    setting: str,
) -> None:
    if _replace_in_cells(notebook, old, new) == 0:
        raise ValueError(
            f"base notebook does not contain configured {setting}: {old!r}"
        )


def _remove_sections(notebook: dict[str, Any], headings: set[str]) -> None:
    """Remove historical comparisons that are not dependencies of a fresh run."""
    retained: list[dict[str, Any]] = []
    skip_next_code = False
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", [])).strip()
        if cell.get("cell_type") == "markdown" and source in headings:
            skip_next_code = True
            continue
        if skip_next_code and cell.get("cell_type") == "code":
            skip_next_code = False
            continue
        skip_next_code = False
        retained.append(cell)
    notebook["cells"] = retained


def _configure_point_workflow(
    notebook: dict[str, Any], config: dict[str, Any]
) -> None:
    station_id = config["station_id"]
    artifact_expression = (
        'PROJECT_ROOT / "data" / "calibration" / '
        f'"station_training_baseline" / "{config["artifact_subdir"]}"'
    )
    _replace_required(
        notebook,
        config["point_source_model_version"],
        config["point_model_version"],
        setting="point_source_model_version",
    )
    _replace_required(
        notebook,
        config["point_source_output_token"],
        artifact_expression,
        setting="point_source_output_token",
    )
    _replace_required(
        notebook,
        config["point_source_pipeline_token"],
        "source_pipeline="
        f'"notebooks/station_training_baseline/'
        f'{Path(config["notebook_path"]).parent.as_posix()}"',
        setting="point_source_pipeline_token",
    )
    if "optuna_verbose" in config or "optuna_storage_name" in config:
        optuna_settings = "    optuna_trials=OPTUNA_TRIALS,\n"
        if "optuna_verbose" in config:
            optuna_settings += (
                f"    optuna_verbose={bool(config['optuna_verbose'])!r},\n"
            )
        if "optuna_storage_name" in config:
            storage_name = str(config["optuna_storage_name"]).strip()
            if not storage_name or Path(storage_name).name != storage_name:
                raise ValueError("optuna_storage_name must be a file name")
            optuna_settings += (
                f'    optuna_storage_path=OUTPUT_DIR / "{storage_name}",\n'
            )
        _replace_required(
            notebook,
            "    optuna_trials=OPTUNA_TRIALS,\n",
            optuna_settings,
            setting="optuna_settings",
        )
    _replace_required(
        notebook,
        "    max_feature_missing_fraction=0.03,\n",
        "    max_feature_missing_fraction=POINT_MAX_FEATURE_MISSING_FRACTION,\n",
        setting="point_training_feature_missingness_gate",
    )
    export_cells = [
        cell
        for cell in notebook["cells"]
        if "export_station_model_weights(" in "".join(cell.get("source", []))
    ]
    if len(export_cells) != 1:
        raise ValueError(
            "base notebook must contain exactly one point-model export cell"
        )
    export_source = "".join(export_cells[0]["source"])
    if "max_feature_missing_fraction=" not in export_source:
        _replace_required(
            notebook,
            "        stack_enabled=config.stack_enabled,\n",
            "        stack_enabled=config.stack_enabled,\n"
            "        max_feature_missing_fraction="
            "config.effective_max_feature_missing_fraction,\n",
            setting="point_export_feature_missingness_gate",
        )
    _replace_required(
        notebook,
        "        source_pipeline=",
        "        bucket_contract=POINT_BUCKET_CONTRACT,\n"
        "        source_pipeline=",
        setting="point_export_bucket_contract",
    )
    _replace_required(
        notebook,
        "        model_version=MODEL_VERSION,\n",
        "        model_version=MODEL_VERSION,\n"
        "        train_years=POINT_EVALUATION_TRAIN_YEARS,\n",
        setting="point_evaluation_train_years",
    )
    export_source = "".join(export_cells[0]["source"]).rstrip()
    export_verification = """

if EXPORT_MODEL_WEIGHTS:
    import json as _point_export_json

    evaluation_point_manifest = _point_export_json.loads(
        exported_weights.manifest_path.read_text(encoding="utf-8")
    )
    assert evaluation_point_manifest["model_version"] == MODEL_VERSION
    assert evaluation_point_manifest["training"]["train_start_year"] == POINT_EVALUATION_TRAIN_YEARS[0]
    assert evaluation_point_manifest["training"]["train_end_year"] == POINT_EVALUATION_TRAIN_YEARS[1]
    assert evaluation_point_manifest["model_contract"]["max_feature_missing_fraction"] == POINT_MAX_FEATURE_MISSING_FRACTION
    assert evaluation_point_manifest["model_contract"]["bucket_contract"] == POINT_BUCKET_CONTRACT
    assert all(
        row["missing_fraction"] <= POINT_MAX_FEATURE_MISSING_FRACTION
        for row in evaluation_point_manifest["features"]["missingness_audit"]
        if row["selected"]
    )
"""
    export_cells[0]["source"] = (
        export_source + export_verification
    ).splitlines(keepends=True)
    _replace_in_cells(notebook, "## V11 Contract", "## Point-model contract")
    _replace_in_cells(
        notebook,
        "## V11 Feature Coverage",
        "## Point-model feature coverage",
    )
    if station_id == "RJTT":
        _replace_required(
            notebook,
            """celsius_predictions = result.test_predictions.copy()
for column in ("actual_high_f", "predicted_high_f", "error_f"):
    if column in celsius_predictions:
        celsius_predictions[column.replace("_f", "_c")] = pd.to_numeric(celsius_predictions[column], errors="coerce") * 5.0 / 9.0
celsius_predictions.head()
""",
            """celsius_predictions = result.test_predictions.copy()
for column in ("actual_high_f", "predicted_high_f"):
    if column in celsius_predictions:
        celsius_predictions[column.replace("_f", "_c")] = (
            pd.to_numeric(celsius_predictions[column], errors="coerce") - 32.0
        ) * 5.0 / 9.0
if "error_f" in celsius_predictions:
    celsius_predictions["error_c"] = (
        pd.to_numeric(celsius_predictions["error_f"], errors="coerce") * 5.0 / 9.0
    )
celsius_predictions.head()
""",
            setting="tokyo_celsius_reporting_conversion",
        )
    _replace_in_cells(
        notebook,
        "## 2026 OOF Weather Brackets",
        "## 2026 Exploratory Holdout Weather Brackets",
    )
    _replace_in_cells(
        notebook,
        "## 2026 Monthly Metrics",
        "## 2026 Exploratory Holdout Monthly Metrics",
    )
    _replace_in_cells(notebook, 'period="oof_2026"', 'period="holdout_2026"')
    _remove_sections(
        notebook,
        {
            "## Version Comparison",
            "## Common-Date Comparison with Existing V11 Settlement",
        },
    )
    setup = "".join(notebook["cells"][1]["source"])
    marker = "EXPORT_MODEL_WEIGHTS = True\n"
    market_settings = (
        f'PROBABILITY_TARGET = "{config["probability_target"]}"\n'
        + f'PROBABILITY_OUTPUT_SUBDIR = "{config["probability_output_subdir"]}"\n'
        if config.get("probability_target") == "celsius_market_1c"
        else ""
    )
    probability_settings = (
        marker
        + "EXPORT_LIVE_MODEL_WEIGHTS = False\n"
        + "POINT_EVALUATION_TRAIN_YEARS = "
        + f'{tuple(config["point_evaluation_train_years"])!r}\n'
        + f'POINT_BUCKET_CONTRACT = "{config["point_bucket_contract"]}"\n'
        + "POINT_MAX_FEATURE_MISSING_FRACTION = "
        + f'{float(config["point_max_feature_missing_fraction"])!r}\n'
        + f'LIVE_POINT_MODEL_VERSION = "{config["point_live_model_version"]}"\n'
        + f'PROBABILITY_MODEL_VERSION = "{config["probability_model_version"]}"\n'
        + market_settings
        + f'PROBABILITY_FEATURE_PROFILE = "{config["probability_feature_profile"]}"\n'
        + f'PROBABILITY_FEATURE_COUNT = {int(config["probability_feature_count"])}\n'
        + f'PROBABILITY_PROVIDERS = {tuple(config["probability_providers"])!r}\n'
        + "PROBABILITY_DEVELOPMENT_YEARS = "
        + f'{tuple(int(year) for year in config["probability_development_years"])!r}\n'
        + "PROBABILITY_FORWARD_VALIDATION_YEARS = "
        + f'{tuple(int(year) for year in config["probability_forward_validation_years"])!r}\n'
        + f'PROBABILITY_HOLDOUT_YEAR = {int(config["probability_holdout_year"])}\n'
    )
    if marker not in setup:
        raise ValueError("base notebook does not expose the model-export setting")
    notebook["cells"][1]["source"] = setup.replace(
        marker, probability_settings
    ).splitlines(keepends=True)
    notebook["cells"][0] = _markdown(
        f"# Station Training Baseline — {station_id}: {config['station_name']}\n\n"
        "**Status: active baseline.** This is the complete per-station workflow: "
        f"the **{config['point_workflow_label']}** point pipeline, followed in "
        "this same notebook by "
        f"**{config['probability_model_label']}**, the pure cumulative-threshold "
        "ordinal probability model. Versioned source notebooks remain "
        "reference-only; new station work starts here.\n"
    )

    export_index = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if "export_station_model_weights(" in "".join(cell.get("source", []))
    )
    source_pipeline = (
        "notebooks/station_training_baseline/"
        f'{Path(config["notebook_path"]).parent.as_posix()}'
    )
    live_cells = [
        _markdown(
            """## Optional live-production point bundle

The evaluation bundle above is frozen before the exploratory holdout and is the
only point bundle used by probability training and holdout reporting. A live
production refit may use all completed actuals, including completed holdout-year
dates, but it has a distinct version and cannot claim holdout performance as
out-of-sample evidence. Keep this export disabled until the source is committed;
then create the immutable release record in a separate promotion review.
"""
        ),
        _code(
            f"""live_exported_weights = None
if EXPORT_LIVE_MODEL_WEIGHTS:
    live_exported_weights = export_station_model_weights(
        project_root=PROJECT_ROOT,
        station_id=STATION_ID,
        city_id=CITY_ID if "CITY_ID" in globals() else None,
        artifact_dir=config.resolved_output_dir(),
        model_version=LIVE_POINT_MODEL_VERSION,
        train_years=None,
        timing_mode=config.timing_mode,
        providers=tuple(config.providers),
        feature_version=config.effective_feature_version,
        training_profile=config.effective_training_profile,
        optuna_metric=config.effective_optuna_metric,
        target_mode=config.effective_target_mode,
        target_source=config.effective_target_source,
        base_model_methods=tuple(config.effective_base_model_methods),
        stack_enabled=config.stack_enabled,
        max_feature_missing_fraction=config.effective_max_feature_missing_fraction,
        bucket_contract=POINT_BUCKET_CONTRACT,
        source_pipeline="{source_pipeline}",
    )
    assert live_exported_weights.bundle_path != exported_weights.bundle_path
    assert LIVE_POINT_MODEL_VERSION != MODEL_VERSION
    print(
        "Live bundle exported as an unreleased candidate. "
        "Create a clean-checkout release record before promotion."
    )
else:
    print("Live-production export disabled; evaluation bundle remains frozen.")
"""
        ),
    ]
    notebook["cells"][export_index + 1 : export_index + 1] = live_cells

    train_index = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if "result = run_station_year_split_experiment(config)" in "".join(
            cell.get("source", [])
        )
    )
    notebook["cells"][train_index + 1 : train_index + 1] = _point_bucket_cells(
        config
    )


def _point_bucket_cells(config: dict[str, Any]) -> list[dict[str, Any]]:
    contract = config["point_bucket_contract"]
    if contract == "polymarket_half_up_1c":
        description = "nearest whole Celsius degree using half-up rounding"
    elif contract == "floor_1c":
        description = "whole Celsius interval using floor rounding"
    else:
        description = "two-degree Fahrenheit bracket after half-up degree rounding"
    return [
        _markdown(
            f"""## Point-model market-bucket hit rate

This score belongs to the continuous point model, not the ordinal probability
model. The configured market contract is `{contract}`: {description}.
The forward score is honest chronological evidence. The holdout score is shown
separately and remains exploratory.
"""
        ),
        _code(
            """from src.calibration.temperature_buckets import (
    point_bucket_metrics,
    point_bucket_predictions,
)
from src.calibration.v19_bucket import crossfit_ridge_predictions

point_forward_predictions = crossfit_ridge_predictions(
    result.validation_predictions,
    providers=PROBABILITY_PROVIDERS,
)
point_forward_predictions = point_forward_predictions.loc[
    point_forward_predictions["validation_year"].isin(
        PROBABILITY_FORWARD_VALIDATION_YEARS
    )
].copy()
assert not point_forward_predictions.empty
assert (
    point_forward_predictions["train_through_year"]
    < point_forward_predictions["validation_year"]
).all()

point_forward_bucket_predictions = point_bucket_predictions(
    point_forward_predictions,
    POINT_BUCKET_CONTRACT,
)
point_forward_bucket_metrics = point_bucket_metrics(
    point_forward_predictions,
    POINT_BUCKET_CONTRACT,
)
point_forward_bucket_metrics["evaluation_status"] = "honest_forward"
point_forward_bucket_metrics
"""
        ),
        _code(
            """point_holdout_predictions = result.test_predictions.loc[
    result.test_predictions["method"].eq("ridge_stack"),
    ["contract_date", "actual_high_f", "predicted_high_f"],
].copy()
assert not point_holdout_predictions.empty

point_holdout_bucket_predictions = point_bucket_predictions(
    point_holdout_predictions,
    POINT_BUCKET_CONTRACT,
)
point_holdout_bucket_metrics = point_bucket_metrics(
    point_holdout_predictions,
    POINT_BUCKET_CONTRACT,
)
point_holdout_bucket_metrics["evaluation_status"] = "exploratory_holdout"
point_holdout_bucket_metrics
"""
        ),
        _code(
            """point_bucket_output_dir = config.resolved_output_dir() / "point_bucket_evaluation"
point_bucket_output_dir.mkdir(parents=True, exist_ok=True)
point_forward_bucket_predictions.to_csv(
    point_bucket_output_dir / f"{STATION_ID}_forward_predictions.csv", index=False
)
point_forward_bucket_metrics.to_csv(
    point_bucket_output_dir / f"{STATION_ID}_forward_metrics.csv", index=False
)
point_holdout_bucket_predictions.to_csv(
    point_bucket_output_dir / f"{STATION_ID}_{PROBABILITY_HOLDOUT_YEAR}_holdout_predictions.csv",
    index=False,
)
point_holdout_bucket_metrics.to_csv(
    point_bucket_output_dir / f"{STATION_ID}_{PROBABILITY_HOLDOUT_YEAR}_holdout_metrics.csv",
    index=False,
)
point_bucket_output_dir
"""
        ),
    ]


def _celsius_market_probability_cells(config: dict[str, Any]) -> list[dict[str, Any]]:
    holdout_year = int(config["probability_holdout_year"])
    market_city = str(config["artifact_subdir"])
    validation_years = [
        int(year) for year in config["probability_forward_validation_years"]
    ]
    return [
        _markdown(
            f"""## {config['probability_model_label']} — market-aligned correction

This stage replaces {market_city}'s historical integer-Fahrenheit/2°F probability
target with the actual {market_city} Polymarket whole-1°C market contract. The point
model remains Fahrenheit-native.

- `point_prediction_c = (point_prediction_f - 32) * 5 / 9`;
- `point_bucket_c = floor(point_prediction_c + 0.5)`;
- `actual_bucket_c = floor(actual_high_c + 0.5)`;
- `offset_c = actual_bucket_c - point_bucket_c`;
- ordered classes: `<=-3, -2, -1, 0, +1, +2, >=+3` °C, chosen from pre-2026
  support with open tails;
- exact market probabilities are `point_bucket_c + exact_offset_c` and sum to
  one on every row;
- all confidence thresholds and the tail-ambiguity rule are selected on the
  {validation_years} forward-validation rows only;
- {holdout_year} remains exploratory and cannot select the model or policy.

The target uses native settlement `actual_high_c` when the normalized settlement
source provides it. Only if native Celsius is unavailable does it convert the
matching Wunderground `actual_high_f`; `iem_daily_high_c` remains diagnostic and
is never substituted as the settlement target. This matches {market_city}
Polymarket's integer Celsius buckets without approximating the
old 2°F distribution.
"""
        ),
        _code(
            """import json

from src.calibration.celsius_market_probability import (
    OFFSET_LABELS_C,
    TARGET_CONTRACT,
    build_celsius_probability_frame,
    celsius_calibration_table,
    celsius_probability_metrics,
    evaluate_celsius_probability_holdout,
    export_celsius_probability_bundle,
    fit_celsius_probability_system,
    sha256_file as celsius_sha256_file,
)
from src.calibration.bucket_probability import probability_feature_names
from src.calibration.v19_bucket import crossfit_ridge_predictions
"""
        ),
        _markdown("### Celsius feature and target contract\n"),
        _code(
            """celsius_feature_names = probability_feature_names(
    include_peak_features=False,
    feature_profile=PROBABILITY_FEATURE_PROFILE,
)
celsius_feature_contract = pd.DataFrame(
    {"position": range(1, len(celsius_feature_names) + 1), "feature": celsius_feature_names}
)
celsius_target_contract = {
    "market": "__MARKET_CITY__ Polymarket whole 1C integer buckets",
    "rounding": "round_half_up(value) = floor(value + 0.5)",
    "target": TARGET_CONTRACT,
    "actual_celsius_source_priority": [
        "actual_high_c",
        "settlement_high_c",
        "actual_high_f_converted_to_c",
    ],
    "excluded_diagnostic_source": "iem_daily_high_c",
    "ordered_offset_classes_c": list(OFFSET_LABELS_C),
    "tail_contract": "training-supported exact offsets within <=-3 and >=+3",
    "feature_profile": PROBABILITY_FEATURE_PROFILE,
    "feature_count": len(celsius_feature_names),
    "providers": list(PROBABILITY_PROVIDERS),
}
assert len(celsius_feature_names) == PROBABILITY_FEATURE_COUNT
celsius_target_contract
""".replace("__MARKET_CITY__", market_city)
        ),
        _markdown(f"### Fit with chronological {validation_years} outer validation\n"),
        _code(
            """point_forward_predictions = crossfit_ridge_predictions(
    result.validation_predictions,
    providers=PROBABILITY_PROVIDERS,
)
assert not point_forward_predictions.empty
assert (
    point_forward_predictions["train_through_year"]
    < point_forward_predictions["validation_year"]
).all()

celsius_training_frame = build_celsius_probability_frame(
    result.features,
    point_forward_predictions,
    result.validation_predictions,
    include_peak_features=False,
    feature_profile=PROBABILITY_FEATURE_PROFILE,
)
assert celsius_training_frame["actual_high_c_source"].isin(
    {"actual_high_c", "settlement_high_c", "actual_high_f_converted_to_c"}
).all()
assert not celsius_training_frame["actual_high_c_source"].eq(
    "iem_daily_high_c"
).any()
celsius_bundle, celsius_forward_predictions, celsius_tuning = (
    fit_celsius_probability_system(
        celsius_training_frame,
        station_id=STATION_ID,
        point_model_version=MODEL_VERSION,
        point_bundle_sha256=celsius_sha256_file(exported_weights.bundle_path),
        feature_profile=PROBABILITY_FEATURE_PROFILE,
        model_version=PROBABILITY_MODEL_VERSION,
        development_years=PROBABILITY_DEVELOPMENT_YEARS,
        forward_validation_years=PROBABILITY_FORWARD_VALIDATION_YEARS,
    )
)
assert celsius_bundle["selected_family"] == "celsius_offset_ordinal_logistic"
assert celsius_bundle["training_cutoff"] < f"{PROBABILITY_HOLDOUT_YEAR}-01-01"
celsius_probability_metrics(celsius_forward_predictions)
"""
        ),
        _markdown("### Verify chronology and exact Celsius market probabilities\n"),
        _code(
            """celsius_forward_dates = pd.to_datetime(celsius_forward_predictions["contract_date"])
assert set(celsius_forward_predictions["validation_year"]) == set(
    PROBABILITY_FORWARD_VALIDATION_YEARS
)
assert (
    pd.to_datetime(celsius_forward_predictions["model_training_cutoff"])
    < celsius_forward_dates
).all()
assert (
    pd.to_datetime(celsius_forward_predictions["calibration_training_cutoff"])
    < pd.to_datetime(celsius_forward_predictions["calibration_validation_start"])
).all()
assert (
    pd.to_datetime(celsius_forward_predictions["calibration_validation_cutoff"])
    < celsius_forward_dates
).all()
for column in ("celsius_offset_probabilities", "market_bucket_probabilities_c"):
    assert celsius_forward_predictions[column].map(
        lambda probabilities: np.isclose(sum(probabilities.values()), 1.0, atol=1e-10)
    ).all()
assert celsius_forward_predictions.apply(
    lambda row: int(row["recommended_bucket_c"])
    == int(max(row["market_bucket_probabilities_c"], key=row["market_bucket_probabilities_c"].get)),
    axis=1,
).all()
assert celsius_bundle["policy_selection_data"] == "pre-2026 forward validation only"
celsius_bundle["decision_thresholds"]
"""
        ),
        _markdown(f"### Evaluate the frozen model on exploratory {holdout_year}\n"),
        _code(
            """holdout_point_predictions = result.test_predictions.loc[
    result.test_predictions["method"].eq("ridge_stack"),
    ["contract_date", "actual_high_f", "predicted_high_f"],
].copy()
assert not holdout_point_predictions.empty

celsius_holdout_predictions, celsius_holdout_metrics, celsius_holdout_calibration = (
    evaluate_celsius_probability_holdout(
        result.features,
        holdout_point_predictions,
        result.test_predictions,
        celsius_bundle,
        holdout_year=PROBABILITY_HOLDOUT_YEAR,
    )
)
assert not celsius_holdout_predictions.empty
assert pd.to_datetime(celsius_holdout_predictions["contract_date"]).dt.year.eq(
    PROBABILITY_HOLDOUT_YEAR
).all()
for column in ("celsius_offset_probabilities", "market_bucket_probabilities_c"):
    assert celsius_holdout_predictions[column].map(
        lambda probabilities: np.isclose(sum(probabilities.values()), 1.0, atol=1e-10)
    ).all()
celsius_bundle["holdout_metrics"] = celsius_holdout_metrics.iloc[0].to_dict()
celsius_bundle["holdout_status"] = "exploratory_previously_inspected_shadow_only"
celsius_holdout_metrics
"""
        ),
        _markdown("### Export the isolated Celsius research artifacts\n"),
        _code(
            f"""celsius_output_dir = config.resolved_output_dir() / PROBABILITY_OUTPUT_SUBDIR
celsius_output_dir.mkdir(parents=True, exist_ok=True)

def _serialize_probability_columns(frame):
    output = frame.copy()
    for column in ("celsius_offset_probabilities", "market_bucket_probabilities_c"):
        output[column] = output[column].map(lambda value: json.dumps(value, sort_keys=True))
    return output

celsius_artifact_paths = []
forward_predictions_path = celsius_output_dir / f"{{STATION_ID}}_forward_validation_predictions.csv"
forward_metrics_path = celsius_output_dir / f"{{STATION_ID}}_forward_validation_metrics.csv"
holdout_predictions_path = celsius_output_dir / f"{{STATION_ID}}_{{PROBABILITY_HOLDOUT_YEAR}}_holdout_predictions.csv"
holdout_metrics_path = celsius_output_dir / f"{{STATION_ID}}_{{PROBABILITY_HOLDOUT_YEAR}}_holdout_metrics.csv"
forward_calibration_path = celsius_output_dir / f"{{STATION_ID}}_forward_validation_calibration.csv"
holdout_calibration_path = celsius_output_dir / f"{{STATION_ID}}_{{PROBABILITY_HOLDOUT_YEAR}}_holdout_calibration.csv"
tuning_path = celsius_output_dir / f"{{STATION_ID}}_pre_2026_tuning.csv"
feature_contract_path = celsius_output_dir / f"{{STATION_ID}}_celsius_feature_contract.csv"
target_contract_path = celsius_output_dir / f"{{STATION_ID}}_celsius_target_contract.json"

_serialize_probability_columns(celsius_forward_predictions).to_csv(forward_predictions_path, index=False)
celsius_probability_metrics(celsius_forward_predictions).to_csv(forward_metrics_path, index=False)
_serialize_probability_columns(celsius_holdout_predictions).to_csv(holdout_predictions_path, index=False)
celsius_holdout_metrics.to_csv(holdout_metrics_path, index=False)
celsius_calibration_table(celsius_forward_predictions).to_csv(forward_calibration_path, index=False)
celsius_holdout_calibration.to_csv(holdout_calibration_path, index=False)
celsius_tuning.to_csv(tuning_path, index=False)
celsius_feature_contract.to_csv(feature_contract_path, index=False)
target_contract_path.write_text(json.dumps(celsius_target_contract, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
celsius_artifact_paths.extend([
    forward_predictions_path, forward_metrics_path, holdout_predictions_path,
    holdout_metrics_path, forward_calibration_path, holdout_calibration_path,
    tuning_path, feature_contract_path, target_contract_path,
])

celsius_bundle_path, celsius_manifest_path = export_celsius_probability_bundle(
    celsius_bundle,
    celsius_output_dir / "model_weights",
    source_identity={{
        "pipeline": "station_training_baseline",
        "notebook": "notebooks/station_training_baseline/{config['notebook_path']}",
        "point_workflow": "{config['point_workflow_label']}",
        "probability_setup": "{config['probability_model_label']}",
    }},
    artifact_paths=celsius_artifact_paths,
)
celsius_manifest = json.loads(celsius_manifest_path.read_text(encoding="utf-8"))
assert celsius_manifest["point_bundle_sha256"] == celsius_sha256_file(exported_weights.bundle_path)
assert celsius_manifest["artifact_integrity"]["bundle_sha256"] == celsius_sha256_file(celsius_bundle_path)
for path in celsius_artifact_paths:
    assert celsius_manifest["artifact_integrity"]["artifact_sha256"][path.name] == celsius_sha256_file(path)
{{
    "point_bundle": exported_weights.bundle_path,
    "point_manifest": exported_weights.manifest_path,
    "celsius_probability_bundle": celsius_bundle_path,
    "celsius_probability_manifest": celsius_manifest_path,
    "output_dir": celsius_output_dir,
}}
"""
        ),
    ]


def _probability_cells(config: dict[str, Any]) -> list[dict[str, Any]]:
    if config.get("probability_target") == "celsius_market_1c":
        return _celsius_market_probability_cells(config)
    station_id = config["station_id"]
    development_years = [
        int(year) for year in config["probability_development_years"]
    ]
    validation_years = [
        int(year) for year in config["probability_forward_validation_years"]
    ]
    return [
        _markdown(
            f"""## {config["probability_model_label"]} — part of this station run

This is the station's verified **pure ordinal** setup, not a separate experiment:

- target: rounded actual degree minus rounded point-model degree;
- ordered classes: `≤-4, -3, -2, -1, 0, +1, +2, +3, ≥+4`;
- eight cumulative logistic regressions with median imputation and scaling;
- forced family `ordinal_logistic`;
- learned-model blend weight `1.0` (no empirical probability blend);
- development years: `{development_years}`;
- honest forward-validation years: `{validation_years}`, each trained only on
  earlier development years;
- the last 90 days of each outer training period tune regularization,
  class weighting, and temperature;
- final probability artifact fitted on all available development rows;
- {int(config["probability_holdout_year"])} is an exploratory holdout,
  **not** an out-of-fold training fold.

Preprocessing is fitted independently inside each chronological training fold.
Continuous features are standardized after median imputation. The verified
implementation does not apply skew transforms or feature
winsorization/clipping.

Degree probabilities are aggregated into the actual 2°F market buckets after
prediction, so bucket boundaries do not need to be centered on the point degree.
The ordinal output remains shadow/research-only until fresh promotion gates pass.
"""
        ),
        _code(
            """import json

from src.calibration.bucket_probability import (
    build_probability_frame,
    default_candidate_specs,
    evaluate_probability_holdout,
    export_probability_bundle,
    fit_probability_system,
    probability_feature_names,
    probability_metrics,
    sha256_file,
)
from src.calibration.v19_bucket import crossfit_ridge_predictions
"""
        ),
        _markdown("### Exact ordinal-regression feature contract\n"),
        _code(
            """ordinal_feature_contract = pd.DataFrame(
    {
        "position": range(
            1,
            len(
                probability_feature_names(
                    include_peak_features=False,
                    feature_profile=PROBABILITY_FEATURE_PROFILE,
                )
            )
            + 1,
        ),
        "feature": probability_feature_names(
            include_peak_features=False,
            feature_profile=PROBABILITY_FEATURE_PROFILE,
        ),
    }
)
ordinal_feature_contract
"""
        ),
        _markdown(
            f"### Fit with chronological {validation_years} outer validation\n"
        ),
        _code(
            """point_forward_predictions = crossfit_ridge_predictions(
    result.validation_predictions,
    providers=PROBABILITY_PROVIDERS,
)
assert not point_forward_predictions.empty
assert (
    point_forward_predictions["train_through_year"]
    < point_forward_predictions["validation_year"]
).all()

ordinal_training_frame = build_probability_frame(
    result.features,
    point_forward_predictions,
    result.validation_predictions,
    include_peak_features=False,
    feature_profile=PROBABILITY_FEATURE_PROFILE,
)
ordinal_candidate_specs = [
    spec
    for spec in default_candidate_specs()
    if spec.family in {"empirical", "ordinal_logistic"}
]
ordinal_bundle, ordinal_forward_predictions, ordinal_tuning = (
    fit_probability_system(
        ordinal_training_frame,
        station_id=STATION_ID,
        point_model_version=MODEL_VERSION,
        point_bundle_sha256=sha256_file(exported_weights.bundle_path),
        include_peak_features=False,
        feature_profile=PROBABILITY_FEATURE_PROFILE,
        model_version=PROBABILITY_MODEL_VERSION,
        candidate_specs=ordinal_candidate_specs,
        forced_family="ordinal_logistic",
        blend_weights=(1.0,),
        development_years=PROBABILITY_DEVELOPMENT_YEARS,
        forward_validation_years=PROBABILITY_FORWARD_VALIDATION_YEARS,
    )
)
assert ordinal_bundle["selected_family"] == "ordinal_logistic"
assert ordinal_bundle["blend_weight"] == 1.0
probability_metrics(ordinal_forward_predictions)
"""
        ),
        _markdown("### Verify chronology and the frozen probability contract\n"),
        _code(
            """ordinal_forward_dates = pd.to_datetime(
    ordinal_forward_predictions["contract_date"]
)
assert set(ordinal_forward_predictions["validation_year"]) == set(
    PROBABILITY_FORWARD_VALIDATION_YEARS
)
assert (
    pd.to_datetime(ordinal_forward_predictions["model_training_cutoff"])
    < ordinal_forward_dates
).all()
assert (
    pd.to_datetime(
        ordinal_forward_predictions["calibration_training_cutoff"]
    )
    < pd.to_datetime(
        ordinal_forward_predictions["calibration_validation_start"]
    )
).all()
assert (
    pd.to_datetime(
        ordinal_forward_predictions["calibration_validation_cutoff"]
    )
    < ordinal_forward_dates
).all()
assert ordinal_bundle["selected_family"] == "ordinal_logistic"
assert ordinal_bundle["family_selection_mode"] == "forced"
assert ordinal_bundle["blend_weight"] == 1.0
assert ordinal_bundle["feature_profile"] == PROBABILITY_FEATURE_PROFILE
assert len(ordinal_bundle["feature_names"]) == PROBABILITY_FEATURE_COUNT
assert not any("peak" in name.lower() for name in ordinal_bundle["feature_names"])
{
    "development_years": list(PROBABILITY_DEVELOPMENT_YEARS),
    "forward_validation_years": sorted(
        ordinal_forward_predictions["validation_year"].unique().tolist()
    ),
    "final_training_start": ordinal_bundle["training_start"],
    "final_training_cutoff": ordinal_bundle["training_cutoff"],
}
"""
        ),
        _markdown("### Evaluate the frozen ordinal model on the 2026 holdout\n"),
        _code(
            """holdout_point_predictions = result.test_predictions.loc[
    result.test_predictions["method"].eq("ridge_stack"),
    ["contract_date", "actual_high_f", "predicted_high_f"],
].copy()
assert not holdout_point_predictions.empty

ordinal_holdout_predictions, ordinal_holdout_metrics = (
    evaluate_probability_holdout(
        result.features,
        holdout_point_predictions,
        result.test_predictions,
        ordinal_bundle,
        holdout_year=PROBABILITY_HOLDOUT_YEAR,
    )
)
assert not ordinal_holdout_predictions.empty
assert not ordinal_holdout_metrics.empty
for probability_column in (
    "offset_probabilities",
    "degree_probabilities",
    "bucket_probabilities",
):
    assert ordinal_holdout_predictions[probability_column].map(
        lambda probabilities: np.isclose(
            sum(float(value) for value in probabilities.values()),
            1.0,
            atol=1e-10,
        )
    ).all()

ordinal_bundle["holdout_metrics"] = ordinal_holdout_metrics.iloc[0].to_dict()
ordinal_bundle["holdout_status"] = "exploratory"
ordinal_bundle["historical_acceptance"] = {
    "passed": False,
    "reasons": ["fresh_shadow_data_required"],
    "holdout_status": "exploratory_previously_inspected",
}
ordinal_holdout_metrics
"""
        ),
        _markdown("### Export point and probability outputs together\n"),
        _code(
            f"""probability_output_dir = config.resolved_output_dir() / "ordinal_probability"
probability_output_dir.mkdir(parents=True, exist_ok=True)

serializable_forward = ordinal_forward_predictions.copy()
serializable_forward["offset_probabilities"] = (
    serializable_forward["offset_probabilities"].map(
        lambda value: json.dumps(value, sort_keys=True)
    )
)
serializable_forward.to_csv(
    probability_output_dir / f"{{STATION_ID}}_forward_probability_predictions.csv",
    index=False,
)
ordinal_tuning.to_csv(
    probability_output_dir / f"{{STATION_ID}}_probability_tuning.csv",
    index=False,
)
probability_metrics(ordinal_forward_predictions).to_csv(
    probability_output_dir / f"{{STATION_ID}}_forward_probability_metrics.csv",
    index=False,
)

serializable_holdout = ordinal_holdout_predictions.copy()
for column in (
    "offset_probabilities",
    "degree_probabilities",
    "bucket_probabilities",
):
    serializable_holdout[column] = serializable_holdout[column].map(
        lambda value: json.dumps(value, sort_keys=True)
    )
serializable_holdout.to_csv(
    probability_output_dir
    / f"{{STATION_ID}}_{{PROBABILITY_HOLDOUT_YEAR}}_probability_holdout_predictions.csv",
    index=False,
)
ordinal_holdout_metrics.to_csv(
    probability_output_dir
    / f"{{STATION_ID}}_{{PROBABILITY_HOLDOUT_YEAR}}_probability_holdout_metrics.csv",
    index=False,
)
ordinal_feature_contract.to_csv(
    probability_output_dir / f"{{STATION_ID}}_ordinal_feature_contract.csv",
    index=False,
)

ordinal_bundle_path, ordinal_manifest_path = export_probability_bundle(
    ordinal_bundle,
    probability_output_dir / "model_weights",
    source_identity={{
        "pipeline": "station_training_baseline",
        "notebook": "notebooks/station_training_baseline/{config['notebook_path']}",
        "point_workflow": "{config['point_workflow_label']}",
        "probability_setup": "{config['probability_model_label']}",
    }},
)
ordinal_manifest = json.loads(ordinal_manifest_path.read_text(encoding="utf-8"))
assert ordinal_manifest["point_bundle_sha256"] == sha256_file(
    exported_weights.bundle_path
)
assert ordinal_manifest["artifact_integrity"]["bundle_sha256"] == sha256_file(
    ordinal_bundle_path
)
{{
    "point_bundle": exported_weights.bundle_path,
    "point_manifest": exported_weights.manifest_path,
    "ordinal_bundle": ordinal_bundle_path,
    "ordinal_manifest": ordinal_manifest_path,
    "output_dir": probability_output_dir,
}}
"""
        ),
    ]


def _challenger_cells(config: dict[str, Any]) -> list[dict[str, Any]]:
    if not config.get("ordinal_challenger_enabled", False):
        return []
    station_id = config["station_id"]
    challenger_version = config.get("ordinal_challenger_version")
    if station_id != "KDAL" or challenger_version != "kdal_ordinal_challenger_v1":
        raise ValueError(
            "the current three-arm ordinal challenger is verified only for KDAL"
        )
    return [
        _markdown(
            """## Required three-arm ordinal challenger export

This stage is part of the KDAL training run and freezes exactly three probability
arms in contract order:

1. best blended independent ordinal model (`model_weight < 1.0`);
2. best shared-slope ordinal model; and
3. best pure independent ordinal model (`model_weight = 1.0`).

Hyperparameters and feature ablations are selected only inside chronological
pre-2026 training folds. The already-inspected 2026 data remains exploratory.
Each arm exports a loadable `.joblib` weight bundle plus a JSON manifest whose
SHA-256 is checked below. These models remain shadow-only and never override the
V20 point-model bucket.
"""
        ),
        _code(
            """from scripts.run_kdal_ordinal_challenger_v1 import run_challenger
from src.calibration.kdal_ordinal_challenger import (
    FROZEN_CANDIDATE_ROLES,
)
"""
        ),
        _markdown(
            "### Train, evaluate, and export all three challenger weight bundles\n"
        ),
        _code(
            """challenger_run = run_challenger()
challenger_comparison = challenger_run["comparison"].copy()

assert tuple(challenger_comparison["candidate_role"]) == (
    FROZEN_CANDIDATE_ROLES
)
assert len(challenger_comparison) == 3
assert challenger_comparison.iloc[0]["model_weight"] < 1.0
assert (
    challenger_comparison.iloc[1]["family"]
    == "shared_slope_ordinal_logistic"
)
assert challenger_comparison.iloc[2]["model_weight"] == 1.0
assert len(challenger_run["bundle_paths"]) == 3
assert len(challenger_run["manifest_paths"]) == 3

for challenger_bundle_path, challenger_manifest_path in zip(
    challenger_run["bundle_paths"],
    challenger_run["manifest_paths"],
):
    assert challenger_bundle_path.is_file()
    assert challenger_manifest_path.is_file()
    challenger_manifest = json.loads(
        challenger_manifest_path.read_text(encoding="utf-8")
    )
    assert challenger_manifest["point_bundle_sha256"] == sha256_file(
        exported_weights.bundle_path
    )
    assert (
        challenger_manifest["artifact_integrity"]["bundle_sha256"]
        == sha256_file(challenger_bundle_path)
    )

challenger_comparison[
    [
        "candidate_role",
        "candidate_name",
        "family",
        "feature_set",
        "feature_count",
        "c",
        "temperature",
        "model_weight",
        "prior_strength",
        "bucket_log_loss",
        "holdout_bucket_log_loss",
        "manifest_path",
    ]
]
"""
        ),
        _markdown("### Exported challenger artifacts\n"),
        _code(
            """pd.DataFrame(
    {
        "candidate_role": FROZEN_CANDIDATE_ROLES,
        "weight_bundle": [
            str(path) for path in challenger_run["bundle_paths"]
        ],
        "manifest": [
            str(path) for path in challenger_run["manifest_paths"]
        ],
    }
)
"""
        ),
    ]


def build_notebook(config: dict[str, Any]) -> dict[str, Any]:
    notebook = _load_base_notebook(config)
    _configure_point_workflow(notebook, config)
    notebook["cells"].extend(_probability_cells(config))
    notebook["cells"].extend(_challenger_cells(config))
    notebook.setdefault("metadata", {})
    notebook["metadata"]["station_training_baseline"] = {
        "station_id": config["station_id"],
        "point_model_version": config["point_model_version"],
        "point_evaluation_train_years": list(
            config["point_evaluation_train_years"]
        ),
        "point_bucket_contract": config["point_bucket_contract"],
        "point_max_feature_missing_fraction": float(
            config["point_max_feature_missing_fraction"]
        ),
        "point_live_model_version": config["point_live_model_version"],
        "point_live_export_default": False,
        "probability_model_label": config["probability_model_label"],
        "probability_model_version": config["probability_model_version"],
        "probability_target": config.get("probability_target", "fahrenheit_2f"),
        "probability_output_subdir": config.get(
            "probability_output_subdir", "ordinal_probability"
        ),
        "probability_family": "ordinal_logistic",
        "probability_blend_weight": 1.0,
        "probability_preprocessing": "median_imputer_then_standard_scaler_per_fold",
        "probability_feature_profile": config["probability_feature_profile"],
        "probability_feature_count": int(config["probability_feature_count"]),
        "probability_providers": list(config["probability_providers"]),
        "probability_development_years": [
            int(year) for year in config["probability_development_years"]
        ],
        "probability_forward_validation_years": [
            int(year)
            for year in config["probability_forward_validation_years"]
        ],
        "probability_holdout_year": int(config["probability_holdout_year"]),
        "probability_holdout_status": "exploratory",
        "ordinal_challenger_enabled": bool(
            config.get("ordinal_challenger_enabled", False)
        ),
        "ordinal_challenger_version": config.get(
            "ordinal_challenger_version"
        ),
        "ordinal_challenger_roles": (
            [
                "blended_ordinal",
                "shared_slope_ordinal",
                "pure_ordinal",
            ]
            if config.get("ordinal_challenger_enabled", False)
            else []
        ),
        "ordinal_challenger_exports_model_weights": bool(
            config.get("ordinal_challenger_enabled", False)
        ),
        "status": "active_baseline",
    }
    if config.get("probability_target") != "celsius_market_1c":
        notebook["metadata"]["station_training_baseline"].pop(
            "probability_target", None
        )
        notebook["metadata"]["station_training_baseline"].pop(
            "probability_output_subdir", None
        )
    return notebook


def _preserve_existing_notebook_state(
    generated: dict[str, Any], existing: dict[str, Any]
) -> dict[str, Any]:
    """Carry execution state for source-identical cells across regeneration."""
    generated_cells = generated.get("cells", [])
    existing_cells = existing.get("cells", [])
    existing_by_source: dict[tuple[str, tuple[str, ...]], list[dict[str, Any]]] = {}
    for existing_cell in existing_cells:
        signature = (
            str(existing_cell.get("cell_type", "")),
            tuple(existing_cell.get("source", [])),
        )
        existing_by_source.setdefault(signature, []).append(existing_cell)
    for generated_cell in generated_cells:
        signature = (
            str(generated_cell.get("cell_type", "")),
            tuple(generated_cell.get("source", [])),
        )
        matches = existing_by_source.get(signature, [])
        if not matches:
            continue
        existing_cell = matches.pop(0)
        generated_cell["metadata"] = dict(existing_cell.get("metadata", {}))
        if generated_cell.get("cell_type") == "code":
            generated_cell["execution_count"] = existing_cell.get("execution_count")
            generated_cell["outputs"] = list(existing_cell.get("outputs", []))

    metadata = dict(existing.get("metadata", {}))
    metadata["station_training_baseline"] = generated["metadata"][
        "station_training_baseline"
    ]
    generated["metadata"] = metadata
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate one self-contained Station Training Baseline notebook."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=BASELINE_ROOT / "configs" / "KDAL.json",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = _load_config(config_path)
    notebook = build_notebook(config)
    output = BASELINE_ROOT / config["notebook_path"]
    if output.is_file():
        existing = json.loads(output.read_text(encoding="utf-8"))
        notebook = _preserve_existing_notebook_state(notebook, existing)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(notebook, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
