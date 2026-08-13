from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "notebooks" / "experiments" / "station_expert_ensemble_v1"


def _generator():
    path = EXPERIMENT / "generate_notebooks.py"
    spec = importlib.util.spec_from_file_location("station_expert_ensemble_generator", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_all_station_notebooks_generate_with_contract_order() -> None:
    module = _generator()
    expected = {
        "KDAL": ("KDAL", ("gfs", "hrrr", "nbm"), "polymarket_half_up_2f"),
        "Seoul": ("RKSI", ("gfs", "gefs", "jma_msm"), "polymarket_half_up_1c"),
        "Tokyo": ("RJTT", ("gfs", "gefs", "jma_msm"), "polymarket_half_up_1c"),
    }
    for name, (station, providers, bucket) in expected.items():
        config = json.loads((EXPERIMENT / "configs" / f"{name}.json").read_text(encoding="utf-8"))
        notebook = module.build_notebook(config)
        source = "\n".join(str(cell["source"]) for cell in notebook["cells"])
        assert notebook["metadata"]["station_expert_ensemble"]["station_id"] == station
        assert tuple(notebook["metadata"]["station_expert_ensemble"]["providers"]) == providers
        assert bucket in source
        assert source.index("Strictly forward expert training") < source.index("Four-way simplex blend") < source.index("Research-only point bundle export") < source.index("Linked ordinal probability model")
        assert "OPTUNA_TRIALS = 30" in source
        assert "OPTUNA_STARTUP_TRIALS = 15" in source
        assert "ENABLE_LIVE_REFIT = False" in source
        assert "station_expert_ensemble_v1" in source
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse(str(cell["source"]))


def test_kdal_ports_three_profile_aware_challenger_arms_only() -> None:
    module = _generator()
    kdal = json.loads((EXPERIMENT / "configs" / "KDAL.json").read_text(encoding="utf-8"))
    source = "\n".join(str(cell["source"]) for cell in module.build_notebook(kdal)["cells"])
    assert "market_core_21" in source and "compact_29" in source and "full_61" in source
    assert "frozen_candidate_rows" in source
    assert "apply_no_override_policy" in source
