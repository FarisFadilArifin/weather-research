from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.calibration.sdk_pipeline import date_range, resolve_contract_end
from src.calibration.station_stacking import (
    StationStackingConfig,
    _fit_feature_columns,
    build_station_wide_dataset,
    feature_columns,
)


STATION = "KDAL"
TIMING_MODE = "same_day_11am_live_safe"
PROVIDERS = ("gfs", "hrrr", "nbm")
FEATURE_VERSION = "v11_settlement_fix_temp"
REQUIRED_TAIL_COLUMNS = (
    "actual_high_f",
    "observed_temp_at_as_of_f",
    "observed_high_temp_through_as_of_f",
    "gfs_high_f",
    "hrrr_high_f",
    "nbm_high_f",
    "v11sf_forecast_temp_11am_mean_f",
    "v11sf_forecast_temp_11am_minus_observed_f",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize and audit the KDAL V20 no-peak 11 AM feature tail"
    )
    parser.add_argument("--project-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--minimum-tail-coverage", type=float, default=1.0)
    return parser.parse_args()


def materialize(project_root: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    os.environ["WEATHER_RESEARCH_INCLUDE_DIRECT_NBM"] = "1"
    config = StationStackingConfig(
        station_id=STATION,
        project_root=project_root,
        timing_mode=TIMING_MODE,
        providers=PROVIDERS,
        feature_version=FEATURE_VERSION,
        training_profile="v20_aligned",
        target_mode="remaining_warmup",
        target_source="wunderground_only",
        max_feature_missing_fraction=0.03,
        output_dir=output_dir,
    )
    features = build_station_wide_dataset(
        project_root=project_root,
        station_id=STATION,
        timing_mode=TIMING_MODE,
        providers=PROVIDERS,
        feature_version=config.effective_feature_version,
        target_source=config.effective_target_source,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_dir / "KDAL_features.csv", index=False)

    categorical, numeric = feature_columns(features, config)
    years = pd.to_numeric(features.get("year"), errors="coerce")
    train = features.loc[years.between(2021, 2025)].copy()
    retained_categorical, retained_numeric = _fit_feature_columns(
        train,
        categorical,
        numeric,
        max_missing_fraction=0.03,
    )
    retained = set(retained_categorical) | set(retained_numeric)
    inventory = pd.DataFrame(
        [
            {
                "feature": column,
                "kind": "categorical" if column in categorical else "numeric",
                "coverage_pct": float(features[column].notna().mean() * 100),
                "retained_3pct_gate": column in retained,
            }
            for column in [*categorical, *numeric]
        ]
    )
    inventory.to_csv(output_dir / "KDAL_feature_columns.csv", index=False)
    return features, inventory


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = (
        args.output_dir
        or project_root / "data" / "calibration" / "station_stacking_v20_kdal_no_peak"
    ).resolve()
    audit_dir = output_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    end = resolve_contract_end(args.end_date)
    expected_dates = date_range(args.start_date, end)
    features, inventory = materialize(project_root, output_dir)
    work = features.copy()
    work["contract_date"] = work["contract_date"].astype(str).str[:10]
    tail = work.loc[work["contract_date"].isin(expected_dates)].copy()

    issues: list[dict[str, object]] = []
    present_dates = set(tail["contract_date"])
    for contract_date in sorted(set(expected_dates) - present_dates):
        issues.append({"contract_date": contract_date, "feature": "*", "issue": "missing_feature_row"})
    duplicates = tail.duplicated("contract_date", keep=False)
    for contract_date in sorted(tail.loc[duplicates, "contract_date"].unique()):
        issues.append({"contract_date": contract_date, "feature": "*", "issue": "duplicate_feature_row"})
    for column in REQUIRED_TAIL_COLUMNS:
        if column not in tail:
            for contract_date in expected_dates:
                issues.append({"contract_date": contract_date, "feature": column, "issue": "missing_column"})
            continue
        missing = tail.loc[tail[column].isna(), "contract_date"]
        for contract_date in missing:
            issues.append({"contract_date": contract_date, "feature": column, "issue": "missing_value"})

    expected_count = len(expected_dates)
    coverage = float(len(present_dates) / expected_count) if expected_count else 1.0
    issues_frame = pd.DataFrame(issues, columns=["contract_date", "feature", "issue"])
    issues_frame.to_csv(audit_dir / "tail_unresolved_rows.csv", index=False)
    tail.to_csv(audit_dir / "tail_features.csv", index=False)
    result = {
        "station_id": STATION,
        "timing_mode": TIMING_MODE,
        "feature_version": FEATURE_VERSION,
        "start_date": args.start_date,
        "end_date": end.isoformat(),
        "expected_tail_rows": expected_count,
        "materialized_rows": len(features),
        "materialized_min_date": str(features["contract_date"].min()) if not features.empty else None,
        "materialized_max_date": str(features["contract_date"].max()) if not features.empty else None,
        "present_tail_rows": len(present_dates),
        "tail_coverage": coverage,
        "feature_inventory_rows": len(inventory),
        "issue_count": len(issues_frame),
        "passed": bool(
            coverage >= args.minimum_tail_coverage
            and len(issues_frame) == 0
            and (not expected_dates or str(features["contract_date"].max()) >= expected_dates[-1])
        ),
    }
    (audit_dir / "tail_audit_result.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
