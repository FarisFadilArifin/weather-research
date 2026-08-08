from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_source_generator():
    source = (
        Path(__file__).resolve().parents[1]
        / "station_stacking_v20_kdal_1pm_no_peak"
        / "generate_station_notebook.py"
    )
    spec = importlib.util.spec_from_file_location("v20_kdal_1pm_generator", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load source generator: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def notebook() -> dict:
    value = _load_source_generator()._notebook()
    replacements = (
        ("KDAL Station Stacking V20 1 PM No Peak", "KDAL V23 1 PM Bucket-Loss Challenger"),
        ('OPTUNA_METRIC = "mae_f"', 'OPTUNA_METRIC = "bucket_log_loss"'),
        (
            'MODEL_VERSION = "station_high_regressor_v20_kdal_1pm_no_peak_stack"',
            'MODEL_VERSION = "station_high_regressor_v23_kdal_1pm_bucket_loss_stack"',
        ),
        (
            '"station_stacking_v20_kdal_1pm_no_peak"',
            '"station_stacking_v23_kdal_1pm_bucket_loss"',
        ),
        (
            "notebooks/experiments/station_stacking_v20_kdal_1pm_no_peak",
            "notebooks/experiments/station_stacking_v23_kdal_1pm_bucket_loss",
        ),
    )
    for cell in value["cells"]:
        source = "".join(cell.get("source", []))
        for old, new in replacements:
            source = source.replace(old, new)
        cell["source"] = source.splitlines(keepends=True)
    for cell in value["cells"]:
        source = "".join(cell.get("source", []))
        if "audit_path = PROJECT_ROOT" in source:
            source = source.replace(
                '"station_stacking_v23_kdal_1pm_bucket_loss" / "audit"',
                '"station_stacking_v20_kdal_1pm_no_peak" / "audit"',
            )
            cell["source"] = source.splitlines(keepends=True)
    for cell in value["cells"]:
        source = "".join(cell.get("source", []))
        if "config = StationStackingConfig(" in source:
            source = source.replace(
                '    hyperparameter_space="wide",\n',
                '    hyperparameter_space="wide",\n'
                '    catboost_max_iterations=1500,\n'
                '    catboost_max_depth=8,\n'
                '    catboost_max_border_count=128,\n',
            )
            cell["source"] = source.splitlines(keepends=True)
    value["cells"][0]["source"] = [
        "# KDAL V23 1 PM Bucket-Loss Challenger\n",
        "\n",
        "A clean challenger to V20 using the identical audited 1 PM live-safe feature, target, fold, "
        "missingness, and no-peak contracts. The model-selection objective changes: base and stack "
        "hyperparameters are selected by two-degree bucket log loss instead of MAE, with bounded "
        "CatBoost complexity documented below. The 2026 split "
        "remains the final holdout and Wunderground alignment features are intentionally excluded.\n",
    ]
    value["cells"].extend(
        [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Forward-only selective prediction\n",
                    "\n",
                    "Train a separate win classifier from honest V23 point OOF predictions. Its "
                    "confidence threshold is frozen from 2024-2025 forward predictions before the "
                    "2026 report is read. No WU source-alignment features are added.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import json\n",
                    "import subprocess\n",
                    "SELECTOR_DIR = PROJECT_ROOT / 'data/calibration/station_stacking_v23_kdal_1pm_bucket_loss_selector'\n",
                    "SELECTOR_VERSION = 'station_bucket_win_selector_v23_kdal_1pm_bucket_loss'\n",
                    "selector_manifest_path = SELECTOR_DIR / 'model_weights' / f'KDAL_{SELECTOR_VERSION}.json'\n",
                    "point_bundle_path = config.resolved_output_dir() / 'model_weights' / f'KDAL_{MODEL_VERSION}.joblib'\n",
                    "from src.calibration.bucket_probability import sha256_file\n",
                    "current_point_sha = sha256_file(point_bundle_path)\n",
                    "selector_is_current = False\n",
                    "if selector_manifest_path.exists():\n",
                    "    existing_selector_manifest = json.loads(selector_manifest_path.read_text(encoding='utf-8'))\n",
                    "    selector_is_current = existing_selector_manifest.get('point_bundle_sha256') == current_point_sha\n",
                    "if not selector_is_current:\n",
                    "    subprocess.run([\n",
                    "        sys.executable, str(PROJECT_ROOT / 'scripts/train-win-classifier.py'),\n",
                    "        '--station', 'KDAL', '--pipeline-dir', str(config.resolved_output_dir()),\n",
                    "        '--point-bundle', str(point_bundle_path),\n",
                    "        '--point-model-version', MODEL_VERSION, '--feature-profile', 'kdal_1pm',\n",
                    "        '--model-version', SELECTOR_VERSION, '--output-dir', str(SELECTOR_DIR),\n",
                    "    ], cwd=PROJECT_ROOT, check=True)\n",
                    "else:\n",
                    "    print('Using selector artifact bound to the current point bundle.')\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Complete mismatch audit and frozen holdout report\n"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "subprocess.run([sys.executable, str(PROJECT_ROOT / 'scripts/audit_v23_kdal_1pm_bucket_loss.py')], cwd=PROJECT_ROOT, check=True)\n",
                    "selector_audit = json.loads((SELECTOR_DIR / 'audit/audit_result.json').read_text(encoding='utf-8'))\n",
                    "assert selector_audit['passed']\n",
                    "selector_audit\n",
                ],
            },
        ]
    )
    return value


def main() -> None:
    path = Path(__file__).resolve().parent / "v23_kdal_1pm_bucket_loss.ipynb"
    path.write_text(json.dumps(notebook(), indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
