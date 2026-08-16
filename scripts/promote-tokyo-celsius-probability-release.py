#!/usr/bin/env python3
"""Promote an exact-point Tokyo probability candidate with explicit replay evidence."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_STATION = "RJTT"
EXPECTED_ARTIFACT_TYPE = "station_celsius_market_probability_model"
EXPECTED_POINT_HASH = "2a30f116c188e4199950911523cdbe4cdb680a0e9cb8361092e23fd374c07d70"
EXPECTED_POINT_MANIFEST_HASH = "6ad5eab351b263de177f77bb67842c5115ee6e2f1c593cf2084ae90e12894e30"
EXPECTED_SELECTOR = "recommended_bucket_c"
EXPECTED_POLICY = "point_probability_ge_0.25"
EXPECTED_CAP = 0.55


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def clean_source_identity() -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()
    if dirty:
        raise ValueError("probability promotion requires a clean source checkout")
    return {
        "git_commit": commit,
        "git_dirty": False,
        "pipeline": "scripts/promote-tokyo-celsius-probability-release.py",
    }


def validate_candidate(
    bundle: Mapping[str, Any], manifest: Mapping[str, Any], bundle_path: Path
) -> None:
    checks = (
        (bundle.get("artifact_type") == EXPECTED_ARTIFACT_TYPE, "bundle_type_mismatch"),
        (manifest.get("artifact_type") == EXPECTED_ARTIFACT_TYPE, "manifest_type_mismatch"),
        (str(bundle.get("station_id") or "").upper() == EXPECTED_STATION, "bundle_station_mismatch"),
        (manifest.get("model_version") == bundle.get("model_version"), "model_version_mismatch"),
        ((manifest.get("artifact_integrity") or {}).get("bundle_sha256") == sha256_file(bundle_path), "bundle_hash_mismatch"),
        (bundle.get("point_bundle_sha256") == EXPECTED_POINT_HASH, "point_bundle_mismatch"),
        (manifest.get("point_bundle_sha256") == EXPECTED_POINT_HASH, "manifest_point_bundle_mismatch"),
        (bundle.get("point_manifest_sha256") == EXPECTED_POINT_MANIFEST_HASH, "point_manifest_mismatch"),
        (manifest.get("point_manifest_sha256") == EXPECTED_POINT_MANIFEST_HASH, "manifest_point_manifest_mismatch"),
        (bundle.get("selection_excludes_holdout") is True, "holdout_selection_mismatch"),
        (str(bundle.get("training_cutoff") or "") < "2026-01-01", "training_cutoff_not_pre_2026"),
        (not bool((manifest.get("historical_acceptance") or {}).get("passed")), "candidate_already_accepted"),
    )
    for passed, reason in checks:
        if not passed:
            raise ValueError(reason)
    source = manifest.get("source_identity") or {}
    if source.get("git_dirty") is not False or len(str(source.get("git_commit") or "")) != 40:
        raise ValueError("candidate_source_identity_invalid")


def accepted_replay_row(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8", newline="") as handle:
        matches = [
            row
            for row in csv.DictReader(handle)
            if row.get("selector") == EXPECTED_SELECTOR
            and row.get("policy") == EXPECTED_POLICY
            and abs(float(row.get("cost_cap") or "nan") - EXPECTED_CAP) < 1e-12
        ]
    if len(matches) != 1:
        raise ValueError("exact_55c_replay_row_missing_or_ambiguous")
    row = matches[0]
    checks = (
        (int(row["entries"]) >= 50, "insufficient_replay_entries"),
        (int(row["active_months"]) >= 5, "insufficient_active_months"),
        (int(row["positive_months"]) == int(row["active_months"]), "replay_has_nonpositive_month"),
        (float(row["pnl_ex_top_five_wins_usd"]) > 0.0, "replay_not_robust_to_top_five"),
        (row.get("robust_eligible", "").strip().lower() == "true", "replay_not_robust_eligible"),
    )
    for passed, reason in checks:
        if not passed:
            raise ValueError(reason)
    return {
        "selector": EXPECTED_SELECTOR,
        "policy": EXPECTED_POLICY,
        "cost_cap": EXPECTED_CAP,
        "entries": int(row["entries"]),
        "wins": int(row["wins"]),
        "hit_rate": float(row["hit_rate"]),
        "net_pnl_usd": float(row["net_pnl_usd"]),
        "pnl_ex_top_five_wins_usd": float(row["pnl_ex_top_five_wins_usd"]),
        "max_drawdown_usd": float(row["max_drawdown_usd"]),
        "active_months": int(row["active_months"]),
        "positive_months": int(row["positive_months"]),
    }


def promote_bundle(
    source: Mapping[str, Any], *, model_version: str, approval: Mapping[str, Any]
) -> dict[str, Any]:
    output = copy.deepcopy(dict(source))
    output["model_version"] = model_version
    output["historical_acceptance"] = {
        "passed": True,
        "reason": "operator_approved_after_exact_point_55c_replay",
        "limitations": [
            "2026 serving replay uses a point model fitted through 2026-07-25",
            "2026 replay was previously inspected and is not fresh out-of-sample evidence",
            "live activation is an explicitly approved production hypothesis",
        ],
    }
    output["production_approval"] = copy.deepcopy(dict(approval))
    output["activation_status"] = "operator_approved_for_live"
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--serving-predictions", type=Path, required=True)
    parser.add_argument("--reference-source-manifest", type=Path, required=True)
    parser.add_argument("--backtest-summary", type=Path, required=True)
    parser.add_argument("--backtest-cap-summary", type=Path, required=True)
    parser.add_argument("--strategy-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--approved-by", required=True)
    parser.add_argument("--approved-at", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_bundle = joblib.load(args.source_bundle)
    source_manifest = load_json(args.source_manifest)
    validate_candidate(source_bundle, source_manifest, args.source_bundle)
    reference_manifest = load_json(args.reference_source_manifest)
    prediction_hash = sha256_file(args.serving_predictions)
    reference_prediction_hash = (
        (reference_manifest.get("artifact_integrity") or {})
        .get("artifact_sha256", {})
        .get(args.serving_predictions.name)
    )
    if prediction_hash != reference_prediction_hash:
        raise ValueError("serving_predictions_do_not_match_replayed_candidate")
    backtest = load_json(args.backtest_summary)
    if Path(str((backtest.get("settings") or {}).get("prediction_source") or "")).name != args.serving_predictions.name:
        raise ValueError("backtest_prediction_source_mismatch")
    replay = accepted_replay_row(args.backtest_cap_summary)
    source_identity = clean_source_identity()
    approval = {
        "approved_by": args.approved_by,
        "approved_at": args.approved_at,
        "approval_scope": "RJTT recommended bucket with point probability >=0.25 and 0.55 maximum entry cost",
        "replay": replay,
        "evidence_sha256": {
            "serving_predictions": prediction_hash,
            "reference_source_manifest": sha256_file(args.reference_source_manifest),
            "backtest_summary": sha256_file(args.backtest_summary),
            "backtest_cap_summary": sha256_file(args.backtest_cap_summary),
            "strategy_report": sha256_file(args.strategy_report),
        },
        "source_candidate_bundle_sha256": sha256_file(args.source_bundle),
        "source_candidate_manifest_sha256": sha256_file(args.source_manifest),
        "fitting_performed": False,
        "threshold_selection_performed": False,
    }
    output_bundle = promote_bundle(source_bundle, model_version=args.model_version, approval=approval)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    stem = f"{EXPECTED_STATION}_{args.model_version}"
    bundle_path = output / f"{stem}.joblib"
    manifest_path = output / f"{stem}.json"
    joblib.dump(output_bundle, bundle_path)
    manifest = {key: copy.deepcopy(value) for key, value in output_bundle.items() if key != "model_state"}
    manifest["source_identity"] = source_identity
    manifest["artifact_integrity"] = {
        "bundle_sha256": sha256_file(bundle_path),
        "evidence_sha256": approval["evidence_sha256"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "bundlePath": str(bundle_path),
        "bundleSha256": sha256_file(bundle_path),
        "manifestPath": str(manifest_path),
        "manifestSha256": sha256_file(manifest_path),
        "pointBundleSha256": output_bundle["point_bundle_sha256"],
        "pointManifestSha256": output_bundle["point_manifest_sha256"],
        "historicalAcceptance": output_bundle["historical_acceptance"],
        "replay": replay,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
