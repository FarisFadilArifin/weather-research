from __future__ import annotations

import importlib.util
import json
from pathlib import Path


STATION_ID = "KDAL"
TIMING_MODE = "same_day_1pm_live_safe"
FEATURE_VERSION = "v20_kdal_1pm_no_peak"
TARGET_MODE = "remaining_warmup"


def _load_v20_no_peak_generator():
    source = (
        Path(__file__).resolve().parents[1]
        / "station_stacking_v20_kdal_no_peak"
        / "generate_station_notebook.py"
    )
    spec = importlib.util.spec_from_file_location("station_stacking_v20_no_peak_generator", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load the V20 no-peak notebook generator from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _replace_all(text: str, replacements: list[tuple[str, str]]) -> str:
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def _notebook() -> dict:
    notebook = _load_v20_no_peak_generator()._notebook()
    replacements = [
        ("KDAL Station Stacking V20 No Peak", "KDAL Station Stacking V20 1 PM No Peak"),
        ("same_day_11am_live_safe", TIMING_MODE),
        ("station_high_regressor_v20_kdal_no_peak_stack", "station_high_regressor_v20_kdal_1pm_no_peak_stack"),
        ('feature_version="v11_settlement_fix_temp"', f'feature_version="{FEATURE_VERSION}"'),
        ('"station_stacking_v20_kdal_no_peak"', '"station_stacking_v20_kdal_1pm_no_peak"'),
        ("notebooks/experiments/station_stacking_v20_kdal_no_peak", "notebooks/experiments/station_stacking_v20_kdal_1pm_no_peak"),
        ("V11_SETTLEMENT_FIX_TEMP_FEATURE_COLUMNS", "V20_KDAL_1PM_TEMP_FEATURE_COLUMNS"),
        ("    V11_DROPPED_FEATURE_COLUMNS,\n", ""),
        ("    V11_FEATURE_COLUMNS,\n", ""),
        ("v11sf_forecast_temp_11am_provider_count", "v13sf_forecast_temp_1pm_provider_count"),
        ("v11sf_forecast_temp_11am_minus_observed_f", "v13sf_forecast_temp_1pm_minus_observed_f"),
        ('f"{STATION_ID}_11am_feature_coverage.csv"', 'f"{STATION_ID}_1pm_feature_coverage.csv"'),
        ("## Common-Date Comparison with Existing V11 Settlement", "## Common-Date Comparison with V20 11 AM No Peak"),
        ("station_stacking_v11_settlement", "station_stacking_v20_kdal_no_peak"),
        ('f"{STATION_ID}_v11_common_date_comparison.csv"', 'f"{STATION_ID}_1pm_vs_11am_common_date_comparison.csv"'),
    ]
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        source = _replace_all(source, replacements)
        cell["source"] = source.splitlines(keepends=True)

    notebook["cells"][0]["source"] = [
        "# KDAL Station Stacking V20 1 PM No Peak\n",
        "\n",
        "KDAL-only V20-aligned experiment using the audited 1 PM live-safe forecast and observation data. "
        "It uses the dedicated 1 PM temperature-alignment feature contract, predicts remaining warmup after "
        "the observed 1 PM high-so-far, excludes the HRRR/NBM peak-timing feature family, keeps "
        "Wunderground-only targets and the 3% train-fold missingness gate, and exports the fitted model bundle "
        "after a successful full run.\n",
    ]

    notebook["cells"][3]["source"] = [
        "## 1 PM Data and Feature Contract\n",
        "\n",
        "`timing_mode=\"same_day_1pm_live_safe\"` selects only cycles available by the 1:15 PM local "
        "decision cutoff. `feature_version=\"v20_kdal_1pm_no_peak\"` uses the 1 PM temperature-alignment "
        "features and excludes both the old 11 AM alignment family and the V20 peak-timing family. The model "
        "target is remaining warmup above the observed high through 1 PM; reported predictions are converted "
        "back to final daily highs by the shared pipeline.\n",
    ]

    # Replace the inherited generic V11 feature-list cells with a focused 1 PM contract view.
    for index, cell in enumerate(notebook["cells"]):
        source = "".join(cell.get("source", []))
        if source.strip() == "V11_FEATURE_COLUMNS, sorted(V11_DROPPED_FEATURE_COLUMNS)":
            cell["source"] = ["V20_KDAL_1PM_TEMP_FEATURE_COLUMNS\n"]
        elif source.startswith("v11_feature_coverage = ("):
            cell["source"] = [
                "one_pm_feature_coverage = (\n",
                "    result.features[V20_KDAL_1PM_TEMP_FEATURE_COLUMNS]\n",
                "    .notna().mean().mul(100).sort_values(ascending=False)\n",
                "    .rename(\"coverage_pct\").reset_index().rename(columns={\"index\": \"feature\"})\n",
                ")\n",
                "one_pm_feature_coverage\n",
            ]
        elif source.strip() == 'result.feature_columns.loc[result.feature_columns["feature"].isin(V11_FEATURE_COLUMNS)]':
            cell["source"] = [
                'result.feature_columns.loc[result.feature_columns["feature"].isin(V20_KDAL_1PM_TEMP_FEATURE_COLUMNS)]\n'
            ]
        elif source.startswith("dropped_present = result.feature_columns.loc["):
            cell["source"] = [
                "old_11am_alignment_features = {\n",
                '    feature for feature in result.feature_columns["feature"]\n',
                '    if feature.startswith("v11sf_forecast_temp_11am")\n',
                "}\n",
                'assert not old_11am_alignment_features, f"11 AM alignment features leaked into 1 PM model: {sorted(old_11am_alignment_features)}"\n',
                'print("No 11 AM alignment features selected.")\n',
            ]
        if cell.get("cell_type") == "markdown" and "## V11 Feature Coverage" in source:
            cell["source"] = ["## 1 PM Feature Coverage\n"]

    # Add an explicit, executable contract audit before model fitting.
    config_index = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if "config = StationStackingConfig(" in "".join(cell.get("source", []))
    )
    audit_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "audit_path = PROJECT_ROOT / \"data\" / \"calibration\" / \"station_stacking_v20_kdal_1pm_no_peak\" / \"audit\" / \"audit_result.json\"\n",
            "if not audit_path.exists():\n",
            "    raise FileNotFoundError(f\"Missing 1 PM pull audit: {audit_path}\")\n",
            "pull_audit = pd.read_json(audit_path, typ=\"series\")\n",
            "assert pull_audit[\"timing_mode\"] == TIMING_MODE, pull_audit.to_dict()\n",
            "assert bool(pull_audit[\"passed\"]), pull_audit.to_dict()\n",
            "assert int(pull_audit[\"blocking_issue_count\"]) == 0, pull_audit.to_dict()\n",
            "pull_audit\n",
        ],
    }
    notebook["cells"].insert(config_index, audit_cell)
    return notebook


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "stacking_KDAL_v20_1pm_no_peak.ipynb"
    path.write_text(json.dumps(_notebook(), indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
