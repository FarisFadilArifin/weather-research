from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.calibration.sdk_pipeline import (
    DIRECT_NBM_FILE,
    LIVE_SAFE_DECISION_DELAY_MINUTES,
    LIVE_SAFE_MODEL_LAG_MINUTES,
    SDK_NWP_FILE,
    TIMING_MODE_SAME_DAY_1PM_LIVE_SAFE,
    choose_direct_nbm_latest_live_safe_cycle,
    choose_same_day_1pm_live_safe_cycle,
    date_range,
    resolve_contract_end,
)
from src.calibration.station_stacking import (
    StationStackingConfig,
    _fit_feature_columns,
    build_station_wide_dataset,
    feature_columns,
)
from src.current_observations import CURRENT_OBSERVATIONS_1PM_FILE


TIMING_MODE = TIMING_MODE_SAME_DAY_1PM_LIVE_SAFE
STATION = "KDAL"
TIMEZONE = "America/Chicago"
PROVIDERS = ("gfs", "hrrr", "nbm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit and materialize the KDAL 1 PM live-safe pull")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--forecast-cache-dir", type=Path, required=True)
    parser.add_argument("--observation-cache-dir", type=Path, required=True)
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default="latest-complete")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--materialize-features", action="store_true")
    parser.add_argument("--minimum-success-rate", type=float, default=0.97)
    return parser.parse_args()


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()


def _expected_cycle(model: str, contract_date: str) -> tuple[datetime, tuple[int, ...], datetime, datetime]:
    if model == "nbm":
        cycle, fxx, as_of, _, end = choose_direct_nbm_latest_live_safe_cycle(contract_date, TIMEZONE, TIMING_MODE)
    else:
        cycle, fxx, as_of, _, end = choose_same_day_1pm_live_safe_cycle(model, contract_date, TIMEZONE)
    if cycle is None:
        raise AssertionError(f"No expected {model} cycle for {contract_date}")
    return cycle, fxx, as_of, end


def audit_forecasts(frame: pd.DataFrame, dates: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    issues: list[dict[str, object]] = []
    expected = pd.MultiIndex.from_product([[STATION], dates, PROVIDERS], names=["station_id", "contract_date", "provider"])
    if frame.empty:
        missing = expected.to_frame(index=False)
        missing["issue"] = "missing_row"
        return pd.DataFrame(), missing
    work = frame.copy()
    work["station_id"] = work["station_id"].astype(str).str.upper()
    work["provider"] = work["provider"].astype(str).str.lower()
    work["contract_date"] = work["contract_date"].astype(str).str[:10]
    work = work.loc[
        work["station_id"].eq(STATION)
        & work["timing_mode"].astype(str).eq(TIMING_MODE)
        & work["contract_date"].isin(dates)
        & work["provider"].isin(PROVIDERS)
    ].copy()
    duplicates = work.duplicated(["station_id", "contract_date", "provider"], keep=False)
    for row in work.loc[duplicates].itertuples(index=False):
        issues.append({"station_id": STATION, "contract_date": row.contract_date, "provider": row.provider, "issue": "duplicate_key"})
    work = work.drop_duplicates(["station_id", "contract_date", "provider"], keep="last")
    present = pd.MultiIndex.from_frame(work[["station_id", "contract_date", "provider"]])
    for station_id, contract_date, provider in expected.difference(present):
        issues.append({"station_id": station_id, "contract_date": contract_date, "provider": provider, "issue": "missing_row"})

    for row in work.itertuples(index=False):
        key = {"station_id": STATION, "contract_date": row.contract_date, "provider": row.provider}
        if str(getattr(row, "fetch_status", "")).lower() != "ok":
            issues.append({**key, "issue": "fetch_not_ok", "detail": str(getattr(row, "unavailable_reason", ""))})
            continue
        expected_cycle, expected_fxx, expected_as_of, expected_end = _expected_cycle(row.provider, row.contract_date)
        issued = pd.Timestamp(row.issued_at).to_pydatetime().astimezone(UTC)
        as_of = pd.Timestamp(row.forecast_as_of).to_pydatetime().astimezone(UTC)
        window_start = pd.Timestamp(row.forecast_window_start).to_pydatetime().astimezone(UTC)
        window_end = pd.Timestamp(row.forecast_window_end).to_pydatetime().astimezone(UTC)
        decision = expected_as_of + timedelta(minutes=LIVE_SAFE_DECISION_DELAY_MINUTES)
        available = issued + timedelta(minutes=LIVE_SAFE_MODEL_LAG_MINUTES[row.provider])
        checks = {
            "wrong_cycle": issued != expected_cycle,
            "wrong_as_of": as_of != expected_as_of,
            "wrong_window_start": window_start != expected_as_of,
            "wrong_window_end": window_end != expected_end,
            "cycle_not_live_safe": available > decision,
            "wrong_fxx_min": int(float(row.forecast_hour_min)) != min(expected_fxx),
            "wrong_fxx_max": int(float(row.forecast_hour_max)) != max(expected_fxx),
        }
        high = pd.to_numeric(pd.Series([getattr(row, "raw_forecast_high_f", pd.NA)]), errors="coerce").iloc[0]
        if pd.isna(high) or not -100 <= float(high) <= 150:
            checks["implausible_or_missing_high"] = True
        for issue, failed in checks.items():
            if failed:
                issues.append({**key, "issue": issue})

    summary_rows = []
    for provider in PROVIDERS:
        subset = work.loc[work["provider"].eq(provider)]
        ok = subset.get("fetch_status", pd.Series(index=subset.index, dtype="object")).astype(str).str.lower().eq("ok")
        summary_rows.append(
            {
                "provider": provider,
                "expected_rows": len(dates),
                "present_rows": len(subset),
                "ok_rows": int(ok.sum()),
                "success_rate": float(ok.sum() / len(dates)) if dates else 1.0,
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(issues)


def audit_observations(frame: pd.DataFrame, dates: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    issues: list[dict[str, object]] = []
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame({"contract_date": dates, "issue": "missing_observation_row"})
    work = frame.copy()
    work["station_id"] = work["station_id"].astype(str).str.upper()
    work["contract_date"] = work["contract_date"].astype(str).str[:10]
    work = work.loc[
        work["station_id"].eq(STATION)
        & work["timing_mode"].astype(str).eq(TIMING_MODE)
        & work["contract_date"].isin(dates)
    ].copy()
    duplicates = work.duplicated(["station_id", "contract_date", "timing_mode"], keep=False)
    for row in work.loc[duplicates].itertuples(index=False):
        issues.append({"contract_date": row.contract_date, "issue": "duplicate_observation_key"})
    work = work.drop_duplicates(["station_id", "contract_date", "timing_mode"], keep="last")
    present = set(work["contract_date"])
    for contract_date in sorted(set(dates) - present):
        issues.append({"contract_date": contract_date, "issue": "missing_observation_row"})
    tz = ZoneInfo(TIMEZONE)
    for row in work.itertuples(index=False):
        if str(getattr(row, "observed_fetch_status", "")).lower() != "ok":
            issues.append({"contract_date": row.contract_date, "issue": "observation_not_ok", "detail": str(getattr(row, "observed_unavailable_reason", ""))})
            continue
        observed = pd.Timestamp(row.observed_as_of_time_utc).to_pydatetime().astimezone(tz)
        local_day = date.fromisoformat(row.contract_date)
        start = datetime(local_day.year, local_day.month, local_day.day, 12, 50, tzinfo=tz)
        end = datetime(local_day.year, local_day.month, local_day.day, 13, 10, tzinfo=tz)
        if not start <= observed <= end:
            issues.append({"contract_date": row.contract_date, "issue": "observation_outside_1250_1310"})
        temp = pd.to_numeric(pd.Series([getattr(row, "observed_temp_at_as_of_f", pd.NA)]), errors="coerce").iloc[0]
        high = pd.to_numeric(pd.Series([getattr(row, "observed_high_temp_through_as_of_f", pd.NA)]), errors="coerce").iloc[0]
        if pd.notna(temp) and pd.notna(high) and float(high) < float(temp):
            issues.append({"contract_date": row.contract_date, "issue": "high_so_far_below_current_temp"})
    ok = work.get("observed_fetch_status", pd.Series(index=work.index, dtype="object")).astype(str).str.lower().eq("ok")
    summary = pd.DataFrame([{"provider": "observations", "expected_rows": len(dates), "present_rows": len(work), "ok_rows": int(ok.sum()), "success_rate": float(ok.sum() / len(dates)) if dates else 1.0}])
    return summary, pd.DataFrame(issues)


def materialize_features(project_root: Path, output_dir: Path) -> pd.DataFrame:
    config = StationStackingConfig(
        station_id=STATION,
        project_root=project_root,
        timing_mode=TIMING_MODE,
        providers=PROVIDERS,
        feature_version="v20_kdal_1pm_no_peak",
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
    features["remaining_warmup_after_1pm_f"] = (
        pd.to_numeric(features.get("actual_high_f"), errors="coerce")
        - pd.to_numeric(features.get("observed_high_temp_through_as_of_f"), errors="coerce")
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
    inventory = pd.DataFrame(
        [
            {"feature": column, "kind": "categorical", "retained_3pct_gate": column in retained_categorical}
            for column in categorical
        ]
        + [
            {"feature": column, "kind": "numeric", "retained_3pct_gate": column in retained_numeric}
            for column in numeric
        ]
    )
    inventory["coverage_pct"] = inventory["feature"].map(features.notna().mean().mul(100).to_dict())
    inventory.to_csv(output_dir / "KDAL_feature_columns.csv", index=False)
    return features


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = (args.output_dir or project_root / "data/calibration/station_stacking_v20_kdal_1pm_no_peak").resolve()
    audit_dir = output_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    end = resolve_contract_end(args.end_date)
    dates = date_range(args.start_date, end)
    sdk = _read(args.forecast_cache_dir / SDK_NWP_FILE)
    direct_nbm = _read(args.forecast_cache_dir / DIRECT_NBM_FILE)
    forecasts = pd.concat([sdk, direct_nbm], ignore_index=True, sort=False)
    observations = _read(args.observation_cache_dir / CURRENT_OBSERVATIONS_1PM_FILE)
    forecast_summary, forecast_issues = audit_forecasts(forecasts, dates)
    observation_summary, observation_issues = audit_observations(observations, dates)
    summary = pd.concat([forecast_summary, observation_summary], ignore_index=True)
    issues = pd.concat([forecast_issues, observation_issues], ignore_index=True, sort=False)
    summary.to_csv(audit_dir / "coverage_summary.csv", index=False)
    issues.to_csv(audit_dir / "unresolved_rows.csv", index=False)
    feature_rows = None
    if args.materialize_features:
        feature_rows = len(materialize_features(project_root, output_dir))
    non_blocking_issues = {"fetch_not_ok", "observation_not_ok"}
    blocking_issue_count = (
        int((~issues["issue"].isin(non_blocking_issues)).sum())
        if not issues.empty and "issue" in issues
        else 0
    )
    result = {
        "timing_mode": TIMING_MODE,
        "start_date": args.start_date,
        "end_date": end.isoformat(),
        "coverage": summary.to_dict(orient="records"),
        "issue_count": len(issues),
        "blocking_issue_count": blocking_issue_count,
        "feature_rows": feature_rows,
        "passed": bool(
            blocking_issue_count == 0
            and not summary.empty
            and summary["success_rate"].ge(args.minimum_success_rate).all()
        ),
    }
    (audit_dir / "audit_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
