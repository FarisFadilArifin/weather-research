from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


STATION_ID = "RJTT"
MODEL_VERSION = "station_high_regressor_baseline_tokyo_no_peak_stack"
FEATURE_VERSION = "v20_asia_no_peak"
FEATURE_PIPELINE = "station_stacking_v20_asia_no_peak"
TIMING_MODE = "asia_same_day_11am_live_safe"
PROVIDERS = ("gfs", "gefs", "jma_msm")
TARGET_SOURCE = "wunderground_only"
PREDICTION_UNIT = "celsius"
RUNTIME_CONTRACT_SHA256 = (
    "178006146855e2685d81fb3b9ce40c5475ae8f472aa81ac1335d7a2b493c5f33"
)
REFERENCE_ROW_COUNT = 5
EXPECTED_REPLAY = {
    "entries": 96,
    "net_pnl_usd": 372.1667330085244,
    "roi_on_gross_notional": 0.9691842005430322,
    "max_drawdown_usd": 18.94453765586033,
}


@dataclass(frozen=True)
class SourceIdentity:
    git_commit: str
    git_dirty: bool


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_identity(project_root: Path) -> SourceIdentity:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return SourceIdentity(git_commit=commit, git_dirty=bool(status))


def validate_contract(bundle: dict[str, Any], manifest: dict[str, Any]) -> None:
    expected = {
        "station_id": STATION_ID,
        "model_version": MODEL_VERSION,
        "feature_version": FEATURE_VERSION,
        "timing_mode": TIMING_MODE,
        "target_source": TARGET_SOURCE,
    }
    for key, value in expected.items():
        if bundle.get(key) != value:
            raise ValueError(f"bundle_{key}_mismatch")
    if tuple(bundle.get("providers") or ()) != PROVIDERS:
        raise ValueError("bundle_providers_mismatch")
    contract = manifest.get("model_contract") or {}
    if contract.get("feature_version") != FEATURE_VERSION:
        raise ValueError("manifest_feature_version_mismatch")
    if tuple(contract.get("providers") or ()) != PROVIDERS:
        raise ValueError("manifest_providers_mismatch")
    if manifest.get("station_id") != STATION_ID:
        raise ValueError("manifest_station_id_mismatch")
    if not bundle.get("feature_names") or not bundle.get("base_models"):
        raise ValueError("bundle_missing_models_or_features")
    if bundle.get("stack_model") is None:
        raise ValueError("bundle_missing_stack_model")


def predict_high_f(bundle: dict[str, Any], row: Any) -> float:
    import pandas as pd

    feature_names = list(bundle["feature_names"])
    frame = pd.DataFrame([{name: row.get(name) for name in feature_names}], columns=feature_names)
    observed_high = float(row["observed_high_temp_through_as_of_f"])
    base_predictions: dict[str, float] = {}
    for name in ("xgboost", "lightgbm", "catboost"):
        remaining = float(bundle["base_models"][name].predict(frame)[0])
        base_predictions[f"{name}_predicted_high_f"] = max(observed_high, observed_high + remaining)
    stack_features = list(bundle["stack_features"])
    stack_row = pd.DataFrame(
        [{name: base_predictions[name] for name in stack_features}], columns=stack_features
    )
    result = float(bundle["stack_model"].predict(stack_row)[0])
    if not math.isfinite(result):
        raise ValueError("non_finite_prediction")
    return result


def fixed_reference_rows(features_csv: Path, bundle: dict[str, Any]) -> list[Any]:
    import pandas as pd

    frame = pd.read_csv(features_csv, low_memory=False)
    # The trained pipelines natively handle numeric NaNs. Reference rows only
    # require the observation anchor and categorical inputs used by CatBoost.
    required = [
        "observed_high_temp_through_as_of_f",
        *list(bundle.get("categorical_features") or []),
    ]
    usable = frame.dropna(subset=required).sort_values("contract_date").reset_index(drop=True)
    if len(usable) < REFERENCE_ROW_COUNT:
        raise ValueError("insufficient_complete_reference_rows")
    indices = [round(index * (len(usable) - 1) / (REFERENCE_ROW_COUNT - 1)) for index in range(REFERENCE_ROW_COUNT)]
    return [usable.iloc[index] for index in indices]


def assert_prediction_parity(
    source_bundle: dict[str, Any], promoted_bundle: dict[str, Any], features_csv: Path
) -> list[dict[str, Any]]:
    rows = fixed_reference_rows(features_csv, source_bundle)
    evidence: list[dict[str, Any]] = []
    for row in rows:
        source = predict_high_f(source_bundle, row)
        promoted = predict_high_f(promoted_bundle, row)
        if not math.isclose(source, promoted, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("prediction_parity_mismatch")
        evidence.append(
            {
                "contract_date": str(row["contract_date"]),
                "source_predicted_high_f": source,
                "promoted_predicted_high_f": promoted,
                "absolute_difference_f": abs(source - promoted),
            }
        )
    return evidence


def validate_replay(summary_path: Path) -> dict[str, float | int]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    method = payload.get("method") or {}
    if method.get("station") != "RJTT/Tokyo":
        raise ValueError("replay_station_mismatch")
    if method.get("selector") != "point_bucket_c" or method.get("entry_policy") != "no_filter":
        raise ValueError("replay_strategy_mismatch")
    if not math.isclose(float(method.get("cost_cap", -1)), 0.47, abs_tol=1e-12):
        raise ValueError("replay_cost_cap_mismatch")
    result = payload.get("flat_4") or {}
    for key, expected in EXPECTED_REPLAY.items():
        actual = result.get(key)
        if isinstance(expected, int):
            if actual != expected:
                raise ValueError(f"replay_{key}_mismatch")
        elif actual is None or not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"replay_{key}_mismatch")
    return {key: result[key] for key in EXPECTED_REPLAY}
