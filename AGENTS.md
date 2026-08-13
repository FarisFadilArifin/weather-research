# Repository Agent Instructions

These instructions apply to the entire repository.

## Canonical structure

- `notebooks/station_training_baseline/` contains the active, generator-backed
  station training notebooks, configs, and generator.
- `notebooks/experiments/` contains every historical or research notebook,
  together with its generator and experiment-local README/AUDIT files.
- `docs/` contains project documentation. Start at `docs/README.md`.
- `data/`, `outputs/`, `logs/`, and `tmp/` contain local or generated artifacts
  and are not source-of-truth locations.

Do not create a new top-level notebook directory. New research belongs under
`notebooks/experiments/{experiment_name}/`. New active station work belongs
under `notebooks/station_training_baseline/` and must follow the Station
Training SOP.

## Notebook workflow

- Treat notebook generators and station configs as the source of truth.
- Make structural or source-cell changes in the generator/config first, then
  regenerate the `.ipynb`.
- Preserve saved outputs and user metadata unless regeneration is explicitly
  required by the task.
- Keep experiment generators beside the notebooks they generate.
- Use repository-relative `source_pipeline` and source-identity paths that match
  the canonical filesystem location.
- Do not use an experimental notebook as a new active baseline. Merge accepted
  behavior into the non-versioned station baseline through its generator.

## Station training requirements

Follow:

- `docs/station-training/NOTEBOOK_STANDARD.md`
- `docs/station-training/SOP.md`
- `docs/station-training/RESEARCH_NOTEBOOK_GUIDE.md` for any work under
  `notebooks/experiments/`
- `docs/station-training/ORDINAL_MODEL_2.md`

Keep point and probability chronology strictly forward. Fit preprocessing,
selection, calibration, and policy thresholds inside the applicable training
fold. Previously inspected holdouts remain exploratory and cannot be relabeled
as out-of-fold promotion evidence.

Every exported model bundle must have a matching manifest, point-model
dependency hash where applicable, and verified bundle SHA-256. Probability
models remain shadow-only unless a separate promotion review approves them.

## Editing and safety

- Preserve unrelated tracked and untracked user changes.
- Never rewrite or delete ignored historical artifacts under `data/` as part of
  a source refactor.
- Do not move or delete historical notebooks without an explicit repository
  organization request.
- Keep project-level documentation in `docs/`; keep experiment-specific notes
  beside their experiment.
- Update all code, test, notebook JSON, manifest-source, and Markdown references
  when canonical paths change.

## Validation

For notebook or station-training changes:

```powershell
python -m pytest tests\test_station_training_baseline.py
python -m pytest tests\test_bucket_probability.py
```

Also run the relevant experiment generator tests and model-export manifest
tests. Before handoff, parse changed notebooks as JSON, compile ordinary Python
cells where possible, search for stale paths, check Markdown links, and run
`git diff --check`.
