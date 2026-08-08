from __future__ import annotations

import json
from pathlib import Path


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


def notebook() -> dict:
    cells = [
        md(
            "# KDAL V22 1 PM Direct Bucket Challenger\n\n"
            "Two-stage direct bucket correction: predict whether the V20 1 PM point bucket is wrong, then classify "
            "the actual bucket as lower, same, or upper. Overrides require stable positive lift in both forward years.\n"
        ),
        code(
            "from pathlib import Path\nimport json\nimport subprocess\nimport pandas as pd\n\n"
            "PROJECT_ROOT = Path.cwd().resolve()\n"
            "while not (PROJECT_ROOT / 'src/calibration/bucket_correction.py').exists():\n"
            "    if PROJECT_ROOT.parent == PROJECT_ROOT: raise RuntimeError('Project root not found')\n"
            "    PROJECT_ROOT = PROJECT_ROOT.parent\n"
            "PYTHON = PROJECT_ROOT / '.venv/Scripts/python.exe'\n"
            "POINT_DIR = PROJECT_ROOT / 'data/calibration/station_stacking_v20_kdal_1pm_no_peak'\n"
            "OUTPUT_DIR = PROJECT_ROOT / 'data/calibration/station_stacking_v22_kdal_1pm_direct_bucket'\n"
            "POINT_VERSION = 'station_high_regressor_v20_kdal_1pm_no_peak_stack'\n"
            "BUCKET_VERSION = 'station_bucket_v22_kdal_1pm_direct'\n"
            "RUN_TRAINING = False\nPROJECT_ROOT\n"
        ),
        md("## Verify immutable point contract\n"),
        code(
            "point_manifest = json.loads((POINT_DIR / 'model_weights' / f'KDAL_{POINT_VERSION}.json').read_text(encoding='utf-8'))\n"
            "contract = point_manifest['model_contract']\n"
            "assert contract['timing_mode'] == 'same_day_1pm_live_safe'\n"
            "assert contract['feature_version'] == 'v20_kdal_1pm_no_peak'\n"
            "assert contract['target_mode'] == 'remaining_warmup'\ncontract\n"
        ),
        md("## Train direct lower/same/upper challenger\n"),
        code(
            "manifest_path = OUTPUT_DIR / 'model_weights' / f'KDAL_{BUCKET_VERSION}.json'\n"
            "if RUN_TRAINING or not manifest_path.exists():\n"
            "    subprocess.run([\n"
            "        str(PYTHON), str(PROJECT_ROOT / 'scripts/train-bucket-correction.py'),\n"
            "        '--station', 'KDAL', '--pipeline-dir', str(POINT_DIR),\n"
            "        '--point-bundle', str(POINT_DIR / 'model_weights' / f'KDAL_{POINT_VERSION}.joblib'),\n"
            "        '--point-model-version', POINT_VERSION, '--model-version', BUCKET_VERSION,\n"
            "        '--feature-profile', 'kdal_1pm', '--output-dir', str(OUTPUT_DIR),\n"
            "    ], cwd=PROJECT_ROOT, check=True)\n"
            "else:\n    print('Using existing V22 artifacts; set RUN_TRAINING=True to retrain.')\n"
        ),
        md("## Full mismatch audit\n"),
        code(
            "subprocess.run([str(PYTHON), str(PROJECT_ROOT / 'scripts/audit_v22_kdal_1pm_direct_bucket.py')], cwd=PROJECT_ROOT, check=True)\n"
            "audit = json.loads((OUTPUT_DIR / 'audit/audit_result.json').read_text(encoding='utf-8'))\n"
            "assert audit['passed']\naudit\n"
        ),
        md("## Forward and 2026 results\n"),
        code(
            "forward = pd.read_csv(OUTPUT_DIR / 'KDAL_forward_bucket_correction_metrics.csv')\n"
            "holdout = pd.read_csv(OUTPUT_DIR / 'KDAL_2026_bucket_correction_holdout_metrics.csv')\n"
            "display(forward)\ndisplay(holdout)\n"
        ),
        md("## Promotion decision\n"),
        code(
            "manifest = json.loads(manifest_path.read_text(encoding='utf-8'))\n"
            "acceptance = manifest['historical_acceptance']\n"
            "if acceptance['passed']:\n    print('Historical gates passed; shadow evaluation is next.')\n"
            "else:\n    print('RESEARCH-ONLY:', ', '.join(acceptance['reasons']))\nacceptance\n"
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": ".venv", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.14"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    path = Path(__file__).resolve().parent / "v22_kdal_1pm_direct_bucket.ipynb"
    path.write_text(json.dumps(notebook(), indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
