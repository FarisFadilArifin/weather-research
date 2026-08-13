from __future__ import annotations

import json
from pathlib import Path


def _markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def _code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def notebook() -> dict:
    cells = [
        _markdown(
            "# KDAL V21 1 PM Bucket Challenger\n\n"
            "Bucket-first probability challenger layered on the immutable V20 KDAL 1 PM no-peak point model. "
            "It uses strict nested forward folds, a dedicated `kdal_1pm` feature profile, and an untouched 2026 report. "
            "A successful training run is not automatically promotable; the historical acceptance gates must also pass.\n"
        ),
        _code(
            "from pathlib import Path\n"
            "import json\n"
            "import subprocess\n"
            "import sys\n"
            "import pandas as pd\n\n"
            "PROJECT_ROOT = Path.cwd().resolve()\n"
            "while not (PROJECT_ROOT / 'src' / 'calibration' / 'bucket_probability.py').exists():\n"
            "    if PROJECT_ROOT.parent == PROJECT_ROOT:\n"
            "        raise RuntimeError('Could not locate weather-research project root')\n"
            "    PROJECT_ROOT = PROJECT_ROOT.parent\n"
            "PYTHON = PROJECT_ROOT / '.venv' / 'Scripts' / 'python.exe'\n"
            "POINT_DIR = PROJECT_ROOT / 'data/calibration/station_stacking_v20_kdal_1pm_no_peak'\n"
            "OUTPUT_DIR = PROJECT_ROOT / 'data/calibration/station_stacking_v21_kdal_1pm_bucket'\n"
            "POINT_VERSION = 'station_high_regressor_v20_kdal_1pm_no_peak_stack'\n"
            "BUCKET_VERSION = 'station_bucket_v21_kdal_1pm'\n"
            "RUN_TRAINING = False\n"
            "PROJECT_ROOT\n"
        ),
        _markdown("## Source contract\n"),
        _code(
            "point_manifest_path = POINT_DIR / 'model_weights' / f'KDAL_{POINT_VERSION}.json'\n"
            "point_manifest = json.loads(point_manifest_path.read_text(encoding='utf-8'))\n"
            "contract = point_manifest['model_contract']\n"
            "assert contract['timing_mode'] == 'same_day_1pm_live_safe'\n"
            "assert contract['feature_version'] == 'v20_kdal_1pm_no_peak'\n"
            "assert contract['target_mode'] == 'remaining_warmup'\n"
            "assert contract['target_source'] == 'wunderground_only'\n"
            "contract\n"
        ),
        _markdown("## Train the challenger\n"),
        _code(
            "bucket_manifest_path = OUTPUT_DIR / 'model_weights' / f'KDAL_{BUCKET_VERSION}.json'\n"
            "if RUN_TRAINING or not bucket_manifest_path.exists():\n"
            "    command = [\n"
            "        str(PYTHON), str(PROJECT_ROOT / 'scripts/train-bucket-probability.py'),\n"
            "        '--station', 'KDAL', '--pipeline-dir', str(POINT_DIR),\n"
            "        '--point-bundle', str(POINT_DIR / 'model_weights' / f'KDAL_{POINT_VERSION}.joblib'),\n"
            "        '--point-model-version', POINT_VERSION, '--model-version', BUCKET_VERSION,\n"
            "        '--feature-profile', 'kdal_1pm', '--output-dir', str(OUTPUT_DIR),\n"
            "    ]\n"
            "    subprocess.run(command, cwd=PROJECT_ROOT, check=True)\n"
            "else:\n"
            "    print('Using existing v21 artifacts; set RUN_TRAINING=True to retrain.')\n"
        ),
        _markdown("## End-to-end mismatch audit\n"),
        _code(
            "subprocess.run(\n"
            "    [str(PYTHON), str(PROJECT_ROOT / 'scripts/audit_v21_kdal_1pm_bucket.py')],\n"
            "    cwd=PROJECT_ROOT, check=True,\n"
            ")\n"
            "audit = json.loads((OUTPUT_DIR / 'audit/audit_result.json').read_text(encoding='utf-8'))\n"
            "assert audit['passed']\n"
            "audit\n"
        ),
        _markdown("## Forward and holdout results\n"),
        _code(
            "forward = pd.read_csv(OUTPUT_DIR / 'KDAL_forward_probability_metrics.csv')\n"
            "holdout = pd.read_csv(OUTPUT_DIR / 'KDAL_2026_probability_holdout_metrics.csv')\n"
            "profile = pd.read_csv(OUTPUT_DIR / 'KDAL_probability_feature_profile_comparison.csv')\n"
            "display(profile)\n"
            "display(forward)\n"
            "display(holdout)\n"
        ),
        _markdown("## Promotion decision\n"),
        _code(
            "manifest = json.loads(bucket_manifest_path.read_text(encoding='utf-8'))\n"
            "acceptance = manifest['historical_acceptance']\n"
            "if acceptance['passed']:\n"
            "    print('Historical gates passed; shadow evaluation is the next step.')\n"
            "else:\n"
            "    print('RESEARCH-ONLY: promotion gates failed:', ', '.join(acceptance['reasons']))\n"
            "acceptance\n"
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
    path = Path(__file__).resolve().parent / "v21_kdal_1pm_bucket.ipynb"
    path.write_text(json.dumps(notebook(), indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
