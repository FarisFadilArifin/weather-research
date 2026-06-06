# Quickstart

This is the shortest path for a new collaborator to clone the repo, install dependencies, and understand what can be reproduced locally.

## 1. Create the environment

```powershell
cd D:\dev\weather-research
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

The project currently uses the `src` Python package name, so commands are run as `python -m src.<module>`.

## 2. Verify the code

```powershell
python -m pytest
```

Passing tests confirm the parsing, feature, time-rule, fetch, and station-stacking helpers are importable in your local environment.

## 3. Know what is not in Git

Large research artifacts are intentionally ignored:

- `data/`
- `logs/`
- `outputs/`
- `.venv/`
- GRIB/cache files

That means a fresh clone has code, config, notebooks, and docs, but not the local historical weather caches or generated model artifacts. See [DATA.md](DATA.md) for the expected data layout.

## 4. Main workflow

The current main workflow is station-stacking v2:

1. Backfill current observations and HRRR/GFS forecast rows.
2. Run a station notebook in `notebooks/station_stacking_v2/`.
3. Review artifacts in `data/calibration/station_stacking_v2/`.
4. Export deployable model bundles with:

```powershell
python -m src.export_station_stacking_v2_models --project-root .
```

The older station-stacking and 6 AM calibration paths are kept for research comparison, but v2 is the current deployment direction.

## 5. Start-here files

- [README.md](README.md): full project overview and command reference.
- [DATA.md](DATA.md): local data expectations and ignored artifacts.
- [notebooks/README.md](notebooks/README.md): which notebooks are current.
- [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md): detailed project assumptions and guardrails.
