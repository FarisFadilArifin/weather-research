from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def test_notebooks_use_only_canonical_roots() -> None:
    directory_names = {
        path.name
        for path in NOTEBOOKS.iterdir()
        if path.is_dir() and path.name != "__pycache__"
    }
    assert directory_names == {"experiments", "station_training_baseline"}

    for notebook_path in NOTEBOOKS.rglob("*.ipynb"):
        relative_path = notebook_path.relative_to(NOTEBOOKS)
        assert relative_path.parts[0] in {
            "experiments",
            "station_training_baseline",
        }, relative_path


def test_repository_documentation_uses_canonical_layout() -> None:
    assert (ROOT / "AGENTS.md").is_file()
    assert (ROOT / "README.md").is_file()
    assert (ROOT / "docs" / "README.md").is_file()
    assert (NOTEBOOKS / "experiments" / "README.md").is_file()

    expected_doc_directories = {
        "architecture",
        "data",
        "getting-started",
        "handoffs",
        "modeling",
        "notebooks",
        "operations",
        "station-training",
    }
    actual_doc_directories = {
        path.name for path in (ROOT / "docs").iterdir() if path.is_dir()
    }
    assert actual_doc_directories == expected_doc_directories
