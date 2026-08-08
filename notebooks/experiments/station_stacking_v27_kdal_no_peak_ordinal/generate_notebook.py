from __future__ import annotations

import json
from pathlib import Path


def _markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def _code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def _notebook() -> dict:
    return {
        "cells": [
            _markdown(
                """# KDAL V27: V20 No-Peak Ordinal Distribution

This research-only challenger freezes the KDAL V20 no-peak point model and
fits cumulative-threshold ordinal logistic distributions over rounded-degree
residual offsets. It compares a pure ordinal arm with an ordinal/empirical
blend. The already inspected 2026 period is exploratory only.
"""
            ),
            _code(
                """from pathlib import Path
import json
import subprocess
import sys

import pandas as pd
from IPython.display import display

PROJECT_ROOT = Path.cwd().resolve()
while not (PROJECT_ROOT / "src" / "calibration" / "bucket_probability.py").exists():
    if PROJECT_ROOT.parent == PROJECT_ROOT:
        raise RuntimeError("Could not find project root")
    PROJECT_ROOT = PROJECT_ROOT.parent

OUTPUT_DIR = PROJECT_ROOT / "data" / "calibration" / "station_stacking_v27_kdal_no_peak_ordinal"
"""
            ),
            _markdown("## Train both frozen ordinal arms and run the audit\n"),
            _code(
                """subprocess.run(
    [sys.executable, str(PROJECT_ROOT / "scripts" / "run_v27_kdal_no_peak_ordinal.py")],
    cwd=PROJECT_ROOT,
    check=True,
)
"""
            ),
            _markdown("## Forward and exploratory comparison\n"),
            _code(
                """comparison = pd.read_csv(OUTPUT_DIR / "KDAL_comparison.csv")
year_metrics = pd.read_csv(OUTPUT_DIR / "KDAL_forward_year_metrics.csv")
display(comparison)
display(year_metrics.sort_values(["validation_year", "ranked_probability_score"]))
"""
            ),
            _markdown("## Integrity and promotion status\n"),
            _code(
                """summary = json.loads((OUTPUT_DIR / "KDAL_summary.json").read_text())
audit = json.loads((OUTPUT_DIR / "audit" / "audit_result.json").read_text())
print(json.dumps(summary, indent=2))
print(f"Audit: {audit['passed_count']}/{audit['check_count']} checks passed")
assert audit["passed"]
assert not summary["promotion_approved"]
"""
            ),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": ".venv (Python 3)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    output = Path(__file__).resolve().parent / "stacking_KDAL_v27_ordinal.ipynb"
    output.write_text(json.dumps(_notebook(), indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
