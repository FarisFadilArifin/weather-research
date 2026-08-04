from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from src.asia_11am import CITY_PROFILES, run_live
from src.calibration.asia_station_stacking import build_asia_station_wide_dataset


ARTIFACT_TYPE = "weather_bot_tokyo_feature_artifact_v1"
FEATURE_VERSION = "v20_asia_no_peak"
FEATURE_LINEAGE = "station_stacking_v20_asia_no_peak"
TIMING_MODE = "asia_same_day_11am_live_safe"
PROVIDERS = ("gfs", "gefs", "jma_msm")
TARGET_FIELDS = {
    "actual_high_f",
    "actual_high_c",
    "settlement_high_f",
    "settlement_high_c",
    "remaining_warmup_f",
    "remaining_warmup_from_observed_high_so_far_f",
}
REQUIRED_FIELDS = (
    "gfs_high_f",
    "gefs_high_f",
    "jma_msm_high_f",
    "gfs_forecast_temp_at_as_of_f",
    "gefs_forecast_temp_at_as_of_f",
    "jma_msm_forecast_temp_at_as_of_f",
    "observed_temp_at_as_of_f",
    "observed_high_temp_through_as_of_f",
    "observed_as_of_age_minutes",
)


def clean_value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    return value


def git_commit(project_root: Path) -> str:
    declared = os.environ.get("WEATHER_RESEARCH_SOURCE_COMMIT", "").strip()
    if declared:
        commit = declared
    elif (project_root / ".source-commit").is_file():
        commit = (project_root / ".source-commit").read_text(encoding="utf-8").strip()
    else:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    if len(commit) != 40 or any(ch not in "0123456789abcdef" for ch in commit.lower()):
        raise ValueError("invalid_source_commit")
    return commit


def build_payload(
    frame: pd.DataFrame, contract_date: date, *, source_commit: str, generated_at: datetime
) -> dict[str, Any]:
    selected = frame.loc[
        frame["contract_date"].astype(str).eq(contract_date.isoformat())
    ]
    if len(selected) != 1:
        raise ValueError("expected_exactly_one_tokyo_feature_row")
    row = selected.iloc[0]
    inputs = {
        str(name): clean_value(value)
        for name, value in row.items()
        if str(name) not in TARGET_FIELDS
    }
    missing = [
        name
        for name in REQUIRED_FIELDS
        if not isinstance(inputs.get(name), (int, float))
        or isinstance(inputs.get(name), bool)
        or not math.isfinite(float(inputs[name]))
    ]
    if missing:
        raise ValueError("missing_required_live_features:" + ",".join(missing))
    observed = datetime.fromisoformat(str(inputs.get("observed_as_of_time_local") or ""))
    if observed.date() != contract_date or not (
        (observed.hour == 10 and observed.minute >= 40)
        or (observed.hour == 11 and observed.minute == 0)
    ):
        raise ValueError("observation_outside_1040_1100_local_window")
    return {
        "artifactType": ARTIFACT_TYPE,
        "stationId": "RJTT",
        "contractDate": contract_date.isoformat(),
        "featureVersion": FEATURE_VERSION,
        "featureLineage": FEATURE_LINEAGE,
        "timingMode": TIMING_MODE,
        "providers": list(PROVIDERS),
        "predictionTemperatureUnit": "celsius",
        "pointInTimeSafe": True,
        "observationSource": inputs.get("observed_source"),
        "generatedAtUtc": generated_at.astimezone(UTC).isoformat(),
        "acquisitionSourceCommit": source_commit,
        "featureInputs": inputs,
    }


def publish(payload: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"RJTT_{payload['contractDate']}.json"
    destination = output_dir / filename
    raw = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    digest = hashlib.sha256(raw).hexdigest()
    with tempfile.TemporaryDirectory(dir=output_dir) as temporary:
        root = Path(temporary)
        artifact = root / filename
        sidecar = root / f"{filename}.sha256"
        artifact.write_bytes(raw)
        sidecar.write_text(f"{digest}  {filename}\n", encoding="utf-8")
        os.replace(artifact, destination)
        os.replace(sidecar, destination.with_suffix(".json.sha256"))
    return destination, destination.with_suffix(".json.sha256")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish the live-safe RJTT feature artifact")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contract-date", type=date.fromisoformat)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--skip-live-pull", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    current = datetime.now(UTC)
    day = args.contract_date or current.astimezone(ZoneInfo("Asia/Tokyo")).date()
    if not args.skip_live_pull:
        result = run_live(
            args.data_root,
            [CITY_PROFILES["tokyo"]],
            contract_date=day,
            workers=max(1, args.workers),
            now=current,
        )
        if result.get("status") != "complete":
            raise SystemExit("Tokyo live acquisition is incomplete")
    frame = build_asia_station_wide_dataset(
        args.data_root,
        "tokyo",
        feature_version=FEATURE_VERSION,
        providers=PROVIDERS,
    )
    payload = build_payload(
        frame, day, source_commit=git_commit(project_root), generated_at=current
    )
    artifact, sidecar = publish(payload, args.output_dir)
    print(json.dumps({"status": "ok", "artifact": str(artifact), "sidecar": str(sidecar)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
