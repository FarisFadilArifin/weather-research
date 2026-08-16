from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "promote-tokyo-celsius-probability-release.py"
SPEC = importlib.util.spec_from_file_location("promote_tokyo_probability", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_replay_acceptance_requires_exact_live_rule(tmp_path: Path) -> None:
    path = tmp_path / "caps.csv"
    fields = [
        "selector", "policy", "cost_cap", "entries", "wins", "hit_rate",
        "net_pnl_usd", "pnl_ex_top_five_wins_usd", "max_drawdown_usd",
        "active_months", "positive_months", "robust_eligible",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "selector": "recommended_bucket_c",
            "policy": "point_probability_ge_0.25",
            "cost_cap": "0.55",
            "entries": "92", "wins": "62", "hit_rate": "0.67",
            "net_pnl_usd": "504", "pnl_ex_top_five_wins_usd": "262",
            "max_drawdown_usd": "12.44", "active_months": "5",
            "positive_months": "5", "robust_eligible": "True",
        })

    row = MODULE.accepted_replay_row(path)

    assert row["cost_cap"] == 0.55
    assert row["entries"] == 92


def test_promotion_preserves_model_state_and_records_limitations() -> None:
    source = {"model_version": "candidate", "model_state": {"weights": [1, 2, 3]}}

    promoted = MODULE.promote_bundle(
        source,
        model_version="approved",
        approval={"approved_by": "operator"},
    )

    assert promoted["model_state"] == source["model_state"]
    assert promoted["model_version"] == "approved"
    assert promoted["historical_acceptance"]["passed"] is True
    assert "not fresh out-of-sample" in " ".join(promoted["historical_acceptance"]["limitations"])
