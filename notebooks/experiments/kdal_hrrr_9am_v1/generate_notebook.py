from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any


EXPERIMENT_ROOT = Path(__file__).resolve().parent
NOTEBOOKS_ROOT = EXPERIMENT_ROOT.parents[1]
PROJECT_ROOT = NOTEBOOKS_ROOT.parent
CONFIG_PATH = EXPERIMENT_ROOT / "config.json"
OUTPUT_PATH = EXPERIMENT_ROOT / "train_KDAL.ipynb"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "hrrr_cache_path",
        "data_project_root",
        "hrrr_validation_summary_path",
        "hrrr_data_manifest_path",
        "hrrr_timing_mode",
        "hrrr_expected_sha256",
        "hrrr_expected_rows",
        "hrrr_date_range",
        "hrrr_local_timezone",
        "observation_cutoff_local",
        "prediction_decision_time_local",
        "research_artifact_subdir",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError("experiment config is missing: " + ", ".join(missing))
    if config["hrrr_timing_mode"] != "same_day_11am_hrrr_9am_cycle_v1":
        raise ValueError("the experiment requires the exact 09:00-local HRRR timing mode")
    if config["observation_cutoff_local"] != "11:00":
        raise ValueError("the KDAL observation cutoff must remain 11:00 local")
    if config["prediction_decision_time_local"] != "11:15":
        raise ValueError("the KDAL prediction/decision time must remain 11:15 local")
    return config


def _code(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def _markdown(source: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def _replace_in_cells(notebook: dict[str, Any], old: str, new: str) -> int:
    count = 0
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        count += source.count(old)
        cell["source"] = source.replace(old, new).splitlines(keepends=True)
    return count


def _provider_override_cell(config: dict[str, Any]) -> dict[str, Any]:
    contract = {
        key: config[key]
        for key in (
            "hrrr_cache_path",
            "hrrr_validation_summary_path",
            "hrrr_data_manifest_path",
            "hrrr_timing_mode",
            "hrrr_expected_sha256",
            "hrrr_expected_rows",
            "hrrr_date_range",
            "hrrr_local_timezone",
            "observation_cutoff_local",
            "prediction_decision_time_local",
        )
    }
    contract_json = json.dumps(contract, indent=2)
    return _code(
        f'''# Provider-specific HRRR source override. GFS, NBM, and observations retain
# the baseline same_day_11am_live_safe loader and 11:00-local snapshot.
import hashlib
import json
from zoneinfo import ZoneInfo

import src.calibration.station_stacking as _station_stacking

HRRR_9AM_CONTRACT = {contract_json}
HRRR_9AM_CACHE_PATH = Path(HRRR_9AM_CONTRACT["hrrr_cache_path"])
HRRR_9AM_VALIDATION_SUMMARY_PATH = Path(HRRR_9AM_CONTRACT["hrrr_validation_summary_path"])
HRRR_9AM_DATA_MANIFEST_PATH = Path(HRRR_9AM_CONTRACT["hrrr_data_manifest_path"])
HRRR_9AM_TIMING_MODE = HRRR_9AM_CONTRACT["hrrr_timing_mode"]


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_validated_exact_9am_hrrr():
    required_paths = (
        HRRR_9AM_CACHE_PATH,
        HRRR_9AM_VALIDATION_SUMMARY_PATH,
        HRRR_9AM_DATA_MANIFEST_PATH,
    )
    missing_paths = [str(path) for path in required_paths if not path.is_file()]
    assert not missing_paths, f"Missing exact-09:00 HRRR inputs: {{missing_paths}}"

    summary = json.loads(HRRR_9AM_VALIDATION_SUMMARY_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(HRRR_9AM_DATA_MANIFEST_PATH.read_text(encoding="utf-8"))
    expected_rows = int(HRRR_9AM_CONTRACT["hrrr_expected_rows"])
    expected_dates = tuple(HRRR_9AM_CONTRACT["hrrr_date_range"])
    expected_sha = HRRR_9AM_CONTRACT["hrrr_expected_sha256"]
    assert summary["timing_mode"] == HRRR_9AM_TIMING_MODE
    assert summary["rows"] == summary["ok_rows"] == expected_rows
    assert summary["date_range"] == list(expected_dates)
    assert not summary["missing_dates"]
    assert not summary["issue_timestamp_error_dates"]
    assert not summary["leakage_error_dates"]
    assert not summary["forecast_window_error_dates"]
    assert manifest == {{"sha256": expected_sha, "rows": expected_rows, "timing_mode": HRRR_9AM_TIMING_MODE}}
    assert _sha256(HRRR_9AM_CACHE_PATH) == expected_sha

    frame = pd.read_csv(HRRR_9AM_CACHE_PATH, low_memory=False)
    assert len(frame) == expected_rows
    assert frame["contract_date"].astype(str).nunique() == expected_rows
    assert set(frame["provider"].astype(str).str.lower()) == {{"hrrr"}}
    assert set(frame["model"].astype(str).str.lower()) == {{"hrrr"}}
    assert set(frame["timing_mode"].astype(str)) == {{HRRR_9AM_TIMING_MODE}}
    assert set(frame["fetch_status"].astype(str).str.lower()) == {{"ok"}}
    assert tuple(frame["contract_date"].astype(str).agg(["min", "max"])) == expected_dates
    assert set(pd.to_numeric(frame["forecast_hour_min"], errors="raise")) == {{2}}
    assert set(pd.to_numeric(frame["forecast_hour_max"], errors="raise")) == {{14}}

    local_zone = ZoneInfo(HRRR_9AM_CONTRACT["hrrr_local_timezone"])
    issued_local = pd.to_datetime(frame["issued_at"], utc=True, errors="raise").dt.tz_convert(local_zone)
    as_of_local = pd.to_datetime(frame["forecast_as_of"], utc=True, errors="raise").dt.tz_convert(local_zone)
    contract_dates = pd.to_datetime(frame["contract_date"], errors="raise").dt.date
    assert (issued_local.dt.date == contract_dates).all()
    assert (issued_local.dt.hour == 9).all() and (issued_local.dt.minute == 0).all()
    utc_offsets = issued_local.map(lambda value: value.utcoffset().total_seconds() / 3600)
    assert set(utc_offsets) == {{-6.0, -5.0}}
    issue_hours_utc = pd.to_datetime(frame["issued_at"], utc=True, errors="raise").dt.hour
    assert set(issue_hours_utc) == {{14, 15}}
    assert (as_of_local.dt.date == contract_dates).all()
    assert (as_of_local.dt.hour == 11).all() and (as_of_local.dt.minute == 0).all()
    decision_local = pd.to_datetime(
        frame["contract_date"].astype(str) + " 11:15", errors="raise"
    ).dt.tz_localize(local_zone)
    assert (issued_local <= decision_local).all()
    assert (as_of_local <= decision_local).all()

    for column in _station_stacking.FORECAST_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    frame = frame[_station_stacking.FORECAST_COLUMNS].copy()
    frame["source_cache_dir"] = HRRR_9AM_CACHE_PATH.parent.name
    frame["source_cache_mtime"] = HRRR_9AM_CACHE_PATH.stat().st_mtime
    return frame, summary, manifest


_baseline_forecast_loader = _station_stacking.load_same_day_provider_forecasts


def _load_forecasts_with_exact_9am_hrrr(project_root=".", timing_mode="same_day_11am", providers=("gfs", "hrrr")):
    baseline = _baseline_forecast_loader(project_root, timing_mode=timing_mode, providers=providers)
    requested = tuple(str(provider).lower() for provider in providers)
    if "hrrr" not in requested:
        return baseline
    exact_hrrr, _, _ = _load_validated_exact_9am_hrrr()
    exact_hrrr = exact_hrrr.loc[exact_hrrr["station_id"].astype(str).str.upper().eq("KDAL")].copy()
    baseline = baseline.loc[baseline["provider"].astype(str).str.lower().ne("hrrr")].copy()
    combined = pd.concat([baseline, exact_hrrr], ignore_index=True, sort=False)
    hrrr_rows = combined.loc[combined["provider"].astype(str).str.lower().eq("hrrr")]
    assert len(hrrr_rows) == int(HRRR_9AM_CONTRACT["hrrr_expected_rows"])
    assert set(hrrr_rows["timing_mode"].astype(str)) == {{HRRR_9AM_TIMING_MODE}}
    assert not hrrr_rows.duplicated(["station_id", "provider", "contract_date"]).any()
    return combined


_station_stacking.load_same_day_provider_forecasts = _load_forecasts_with_exact_9am_hrrr
'''
    )


def _readiness_cells() -> list[dict[str, Any]]:
    return [
        _markdown(
            """## Exact-09:00 HRRR data readiness and chronology gate

This experiment changes only the HRRR source contract. The cache must contain
one valid KDAL row per date from 2021-01-01 through 2026-08-08, issued at exact
09:00 `America/Chicago` (14Z in CDT and 15Z in CST), using f02-f14. GFS, NBM,
Wunderground labels, and the live-safe observation snapshot remain inherited
from the KDAL V20 no-peak full-refit lineage. Observations stop at 11:00 local;
prediction and decision remain 11:15 local.
"""
        ),
        _code(
            """exact_9am_hrrr, hrrr_validation_summary, hrrr_data_manifest = _load_validated_exact_9am_hrrr()
hrrr_readiness = pd.DataFrame(
    [{
        "timing_mode": HRRR_9AM_TIMING_MODE,
        "rows": len(exact_9am_hrrr),
        "unique_dates": exact_9am_hrrr["contract_date"].nunique(),
        "first_date": exact_9am_hrrr["contract_date"].min(),
        "last_date": exact_9am_hrrr["contract_date"].max(),
        "issue_hours_utc": sorted(pd.to_datetime(exact_9am_hrrr["issued_at"], utc=True).dt.hour.unique().tolist()),
        "forecast_hour_min": int(pd.to_numeric(exact_9am_hrrr["forecast_hour_min"]).min()),
        "forecast_hour_max": int(pd.to_numeric(exact_9am_hrrr["forecast_hour_max"]).max()),
        "sha256": hrrr_data_manifest["sha256"],
        "observation_cutoff_local": HRRR_9AM_CONTRACT["observation_cutoff_local"],
        "prediction_decision_time_local": HRRR_9AM_CONTRACT["prediction_decision_time_local"],
    }]
)
hrrr_readiness
"""
        ),
    ]


def _strengthen_provider_readiness_gate(notebook: dict[str, Any]) -> None:
    matches = [
        cell
        for cell in notebook["cells"]
        if "availability = provider_availability(" in "".join(cell.get("source", []))
    ]
    if len(matches) != 1:
        raise ValueError("expected exactly one provider-availability cell")
    source = "".join(matches[0]["source"]).rstrip()
    source += '''

station_availability = availability.loc[availability["station_id"].eq(STATION_ID)].copy()
assert set(station_availability["provider"].astype(str)) == set(PROVIDERS), (
    "Full training readiness failed: the inherited 11 AM GFS/NBM caches and "
    "the exact-09:00 HRRR cache must all be available for KDAL."
)
observation_rows = _station_stacking.load_current_observation_features(
    DATA_PROJECT_ROOT,
    station_id=STATION_ID,
    timing_mode=TIMING_MODE,
)
assert not observation_rows.empty, (
    "Full training readiness failed: no inherited 11:00-local KDAL observation cache was found."
)

station_availability
'''
    matches[0]["source"] = source.splitlines(keepends=True)


def build_notebook(config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = dict(_load_config() if config is None else config)
    baseline_generator = _load_module(
        NOTEBOOKS_ROOT / "station_training_baseline" / "generate_station_notebook.py",
        "kdal_hrrr_9am_station_baseline_generator",
    )
    station_config = baseline_generator._load_config(CONFIG_PATH.resolve())
    notebook = baseline_generator.build_notebook(station_config)

    setup_source = "".join(notebook["cells"][1]["source"])
    setup_marker = "PROBABILITY_HOLDOUT_YEAR = 2026\nPROJECT_ROOT\n"
    if setup_marker not in setup_source:
        raise ValueError("could not configure the experiment data project root")
    setup_source = setup_source.replace(
        setup_marker,
        "PROBABILITY_HOLDOUT_YEAR = 2026\n"
        f'DATA_PROJECT_ROOT = Path({config["data_project_root"]!r}).resolve()\n'
        "assert (DATA_PROJECT_ROOT / \"data\" / \"processed\" / \"actual_highs.csv\").is_file()\n"
        "PROJECT_ROOT\n",
    )
    notebook["cells"][1]["source"] = setup_source.splitlines(keepends=True)

    availability_cells = [
        cell
        for cell in notebook["cells"]
        if "availability = provider_availability(" in "".join(cell.get("source", []))
    ]
    if len(availability_cells) != 1:
        raise ValueError("expected exactly one provider-availability cell")
    availability_source = "".join(availability_cells[0]["source"])
    availability_source = availability_source.replace(
        "availability = provider_availability(\n    PROJECT_ROOT,",
        "availability = provider_availability(\n    DATA_PROJECT_ROOT,",
    )
    availability_cells[0]["source"] = availability_source.splitlines(keepends=True)

    config_cells = [
        cell
        for cell in notebook["cells"]
        if "config = StationStackingConfig(" in "".join(cell.get("source", []))
    ]
    if len(config_cells) != 1:
        raise ValueError("expected exactly one point-training config cell")
    config_source = "".join(config_cells[0]["source"])
    if "    project_root=PROJECT_ROOT,\n" not in config_source:
        raise ValueError("could not isolate the training data root")
    config_cells[0]["source"] = config_source.replace(
        "    project_root=PROJECT_ROOT,\n",
        "    project_root=DATA_PROJECT_ROOT,\n",
        1,
    ).splitlines(keepends=True)

    old_artifact = (
        'PROJECT_ROOT / "data" / "calibration" / "station_training_baseline" / '
        f'"{config["artifact_subdir"]}"'
    )
    new_artifact = (
        'PROJECT_ROOT / "data" / "calibration" / "experiments" / '
        f'"{config["research_artifact_subdir"]}"'
    )
    if _replace_in_cells(notebook, old_artifact, new_artifact) == 0:
        raise ValueError("could not isolate the experiment artifact path")
    generated_source = "notebooks/station_training_baseline/experiments/kdal_hrrr_9am_v1"
    canonical_source = "notebooks/experiments/kdal_hrrr_9am_v1"
    if _replace_in_cells(notebook, generated_source, canonical_source) == 0:
        raise ValueError("could not set the canonical experiment source identity")

    notebook["cells"][0] = _markdown(
        """# KDAL Research Training — Exact-09:00-Local HRRR V1

**Status: isolated research experiment; not production and not deployed.** This
generator-backed notebook retains the active KDAL V20 no-peak/full-refit point
and pure-ordinal lineage. Its sole provider change is replacement of HRRR with
the completed exact-09:00 `America/Chicago` candidate cache. The 11:00-local
observation cutoff and 11:15-local prediction/decision contract are unchanged.
All outputs are confined to
`data/calibration/experiments/kdal_hrrr_9am_v1/`.
"""
    )
    notebook["cells"][3:3] = [_provider_override_cell(config), *_readiness_cells()]
    _strengthen_provider_readiness_gate(notebook)
    metadata = notebook.setdefault("metadata", {}).setdefault("station_training_baseline", {})
    metadata.update(
        {
            "status": "research_only",
            "source_pipeline": canonical_source,
            "artifact_root": "data/calibration/experiments/kdal_hrrr_9am_v1",
            "hrrr_timing_mode": config["hrrr_timing_mode"],
            "hrrr_cache_sha256": config["hrrr_expected_sha256"],
            "hrrr_expected_rows": int(config["hrrr_expected_rows"]),
            "observation_cutoff_local": config["observation_cutoff_local"],
            "prediction_decision_time_local": config["prediction_decision_time_local"],
            "production_export": False,
            "deployed": False,
        }
    )
    return notebook


def main() -> None:
    notebook = build_notebook()
    OUTPUT_PATH.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
