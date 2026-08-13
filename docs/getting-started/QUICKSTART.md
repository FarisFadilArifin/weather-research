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

That means a fresh clone has code, config, notebooks, and docs, but not the local historical weather caches or generated model artifacts. See [DATA.md](../data/DATA.md) for the expected data layout.

## 4. Main workflow

The active workflow is the non-versioned Station Training Baseline:

1. Backfill current observations and HRRR/GFS forecast rows.
2. Read `docs/station-training/SOP.md`.
3. Run one station notebook, starting with
   `notebooks/station_training_baseline/stations/KDAL/train_KDAL.ipynb`.
4. Review point and ordinal outputs together under
   `data/calibration/station_training_baseline/{STATION}/`.

The notebook exports its point and probability artifacts together. Versioned
station-stacking and 6 AM calibration paths remain historical research.

## 5. Start-here files

- [README.md](../../README.md): full project overview and command reference.
- [DATA.md](../data/DATA.md): local data expectations and ignored artifacts.
- [Notebook catalog](../notebooks/README.md): which notebooks are current.
- [Station Training SOP](../station-training/SOP.md): create,
  train, validate, and export one station.
- [PROJECT_CONTEXT.md](../architecture/PROJECT_CONTEXT.md): detailed project assumptions and guardrails.
