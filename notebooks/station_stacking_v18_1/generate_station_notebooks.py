import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path


TARGET_STATIONS = ("KATL", "KDAL", "KMIA")


@dataclass(frozen=True)
class Variant:
    name: str
    feature_version: str
    model_version: str
    description: str


VARIANTS = (
    Variant(
        name="nbm",
        feature_version="v18_1_nbm",
        model_version="station_high_regressor_v18_1_nbm_settlement_stack",
        description="adds only the coverage-gated NBM hourly-curve shard features",
    ),
    Variant(
        name="rap",
        feature_version="v18_1_rap",
        model_version="station_high_regressor_v18_1_rap_physics_settlement_stack",
        description="adds only the coverage-gated RAP physics and station-specific physics shard features",
    ),
)


def _load_v18_generator():
    source = Path(__file__).resolve().parents[1] / "station_stacking_v18" / "generate_station_notebooks.py"
    spec = importlib.util.spec_from_file_location("station_stacking_v18_generator", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load v18 notebook generator from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _notebook(station_id: str, variant: Variant) -> dict:
    v18 = _load_v18_generator()
    notebook = v18._notebook(station_id)
    label = f"v18.1 {variant.name.upper()}"
    output_dir = f"station_stacking_{variant.feature_version}"
    replacements = [
        ("Station Stacking v18 Wunderground Physics", f"Station Stacking {label} Wunderground Ablation"),
        (
            "This v18 run keeps the v11 remaining-warmup ridge-stack backbone, requires Wunderground station-history labels, adds NBM hourly curve and HRRR/RAP physics shard features, and writes isolated artifacts to `data/calibration/station_stacking_v18`.",
            f"This {label} ablation keeps the v11 remaining-warmup ridge-stack backbone, requires Wunderground station-history labels, and {variant.description}. It writes isolated artifacts to `data/calibration/{output_dir}`.",
        ),
        (
            '`feature_version="v18"` adds coverage-gated NBM hourly curve and HRRR/RAP physics features while selecting by validation MAE.',
            f'`feature_version="{variant.feature_version}"` {variant.description} while selecting by validation MAE.',
        ),
        ("station_high_regressor_v18_nbm_hrrr_physics_settlement_stack", variant.model_version),
        ('feature_version="v18"', f'feature_version="{variant.feature_version}"'),
        (
            'output_dir=PROJECT_ROOT / "data" / "calibration" / "station_stacking_v18"',
            f'output_dir=PROJECT_ROOT / "data" / "calibration" / "{output_dir}"',
        ),
        ('source_pipeline="notebooks/station_stacking_v18"', 'source_pipeline="notebooks/station_stacking_v18_1"'),
    ]
    for cell in notebook["cells"]:
        cell["source"] = [_replace_all(line, replacements) for line in cell.get("source", [])]
    return notebook


def _replace_all(text: str, replacements: list[tuple[str, str]]) -> str:
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    for variant in VARIANTS:
        for station in TARGET_STATIONS:
            notebook = _notebook(station, variant)
            path = out_dir / f"stacking_{station}_{variant.feature_version}.ipynb"
            path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
            print(path)


if __name__ == "__main__":
    main()
