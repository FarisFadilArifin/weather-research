# Research Notebook Guide for the Current Station Pipeline

This guide is the required procedure for creating a research notebook that
uses the current station-training pipeline. It applies to agents and humans
working under `notebooks/experiments/`. For changes to an active station
baseline, follow [SOP.md](SOP.md) instead.

The central rule is simple: a research notebook may compare or extend the
current pipeline, but it must not silently become a second source of truth for
an active station. The checked-in generator, config, shared Python modules, and
generated active notebook remain authoritative.

## 1. Decide whether the work is research or baseline maintenance

Use an experiment when the work introduces or evaluates any of the following:

- a new feature family, provider, cutoff, target transform, base learner,
  stacker, probability family, calibration method, or policy;
- an ablation, alternative hyperparameter space, robustness study, or
  diagnostic;
- a station not yet accepted into the active baseline; or
- a comparison whose result is not yet approved for the live pipeline.

Use the active baseline workflow when the behavior is already accepted and the
task is to repair, regenerate, or extend
`notebooks/station_training_baseline/stations/{STATION}/train_{STATION}.ipynb`.

Never create another top-level notebook directory. Research belongs at:

```text
notebooks/experiments/{experiment_name}/
```

Accepted research is merged into the non-versioned station baseline through
its generator and config. Do not promote an experiment merely by renaming or
copying its `.ipynb` into the baseline directory.

## 2. Read the contracts before editing

Before creating files, read:

1. repository `AGENTS.md`;
2. [NOTEBOOK_STANDARD.md](NOTEBOOK_STANDARD.md);
3. [SOP.md](SOP.md);
4. [TEMPERATURE_UNITS_AND_BUCKETS.md](TEMPERATURE_UNITS_AND_BUCKETS.md);
5. [ORDINAL_MODEL_2.md](ORDINAL_MODEL_2.md) if probabilities are involved;
6. the closest active station config and generated notebook; and
7. the closest experiment's `README.md`, generator, tests, and `AUDIT.md` when
   one exists.

For production parity or release lineage, also read
[LIVE_MODEL_NOTEBOOK_LINEAGE.md](LIVE_MODEL_NOTEBOOK_LINEAGE.md) and
[RELEASE_PROVENANCE.md](RELEASE_PROVENANCE.md).

## 3. Use a production-aligned research environment

The production point-model releases use CPython 3.12.3 and the direct package
versions in `requirements-ml-runtime.txt`. Production itself is Linux x86-64
with a fully hash-pinned transitive lock. A Windows venv can match the Python
and direct package contract but is not byte-identical to production.

The local production-aligned Windows environment is:

```text
D:\dev\weather-research\.venv-production-312\Scripts\python.exe
```

In VS Code select:

```text
Weather Research (Production-aligned Python 3.12.3)
```

To rebuild that environment without modifying `.venv`:

```powershell
uv venv .venv-production-312 --python 3.12.3 --seed
uv pip install `
  --python .venv-production-312\Scripts\python.exe `
  --requirement requirements-ml-runtime.txt
uv pip install `
  --python .venv-production-312\Scripts\python.exe `
  --constraint requirements-ml-runtime.txt `
  --editable ".[dev]" `
  ipykernel
.venv-production-312\Scripts\python.exe -m ipykernel install `
  --user `
  --name weather-research-production-312 `
  --display-name "Weather Research (Production-aligned Python 3.12.3)"
```

Verify the interpreter before an expensive run:

```powershell
.venv-production-312\Scripts\python.exe -c `
  "import sys; assert sys.version_info[:3] == (3, 12, 3); print(sys.version)"
.venv-production-312\Scripts\python.exe -m pip check
```

Do not claim exact Linux production reproduction from a Windows notebook run.
Use the locked Linux runtime for cross-platform release verification.

## 4. Understand the current pipeline topology

The active station notebook is assembled from three source layers:

```text
station-specific point generator
    + notebooks/station_training_baseline/configs/{STATION}.json
    + notebooks/station_training_baseline/generate_station_notebook.py
    -> notebooks/station_training_baseline/stations/{STATION}/train_{STATION}.ipynb
```

The generated notebook imports shared implementations from `src/calibration/`
and `src/export_station_stacking_v2_models.py`. Comparing only notebook bytes is
not enough; research must record the source commit and every shared module it
depends on.

Current production reference paths are:

| Station | Active notebook | Point providers | Frozen live feature contract |
| --- | --- | --- | ---: |
| KDAL | `stations/KDAL/train_KDAL.ipynb` | GFS, HRRR, NBM | 198 ordered features |
| RJTT | `stations/Tokyo/train_Tokyo.ipynb` | GFS, GEFS, JMA MSM | 293 ordered features |

Use KDAL as the reference for the Dallas-style two-Fahrenheit-degree pipeline.
Use Tokyo or Seoul as the reference for the Asia local-11-AM, whole-Celsius
market pipeline. Provider sets, timezones, unit rules, settlement sources, and
feature availability are station contracts, not interchangeable defaults.

## 5. Create the experiment directory

Use a descriptive lowercase name with an explicit version:

```text
notebooks/experiments/{station}_{hypothesis}_v1/
```

Minimum contents:

```text
notebooks/experiments/{experiment_name}/
|-- README.md
|-- generate_notebook.py
|-- config.json                 # when station/config driven
|-- train_{STATION}.ipynb
`-- AUDIT.md                    # required once results are reviewed
```

For a multi-station experiment, use `configs/{STATION}.json` and generate one
notebook per station. Never combine multiple stations into a single training
notebook.

The experiment `README.md` must state:

- hypothesis and decision being tested;
- parent active notebook and source commit;
- station, timezone, cutoff, providers, settlement source, and native unit;
- target and named market-bucket contract;
- training, forward-validation, and exploratory-holdout periods;
- exact changed behavior relative to the parent;
- output directory and Optuna database name;
- expected artifacts and tests;
- promotion status, initially `research_only`; and
- known limitations, including previously inspected holdouts.

## 6. Keep the generator as the source of truth

Create or modify the experiment generator first. The `.ipynb` is a generated
artifact and must not contain structural changes that are absent from its
generator.

Prefer one of these patterns:

1. import the closest current builder and make narrow, asserted replacements;
2. call shared pipeline functions directly from generated cells; or
3. extract generally useful behavior into a tested shared Python module, then
   call it from both baseline and experiment generators.

Fail closed when an expected replacement token or source cell is missing. A
generator that silently produces a partially patched notebook is unacceptable.

Use repository-relative identity strings:

```text
source_pipeline=notebooks/experiments/{experiment_name}
```

Do not store a developer-specific absolute path in a source manifest. Runtime
paths may resolve absolutely, but recorded source identity must be portable.

Preserve saved outputs and user metadata unless the experiment explicitly
requires regeneration. After generation, parse the notebook as JSON and compare
its source cells with a second generation to detect nondeterminism.

## 7. Isolate artifacts from the active baseline

An experiment must never overwrite active station artifacts. Use:

```text
data/calibration/experiments/{experiment_name}/{STATION}/
```

or an equally explicit experiment-local directory already established by the
parent experiment. Keep model versions experiment-specific.

At minimum isolate:

- feature frames;
- validation and holdout predictions;
- metrics and selected hyperparameters;
- Optuna SQLite storage;
- model weights and manifests; and
- probability/calibration artifacts.

`data/` is ignored and is not source of truth. A Git merge does not transfer
SQLite databases, feature CSVs, fitted bundles, or other generated data. Record
their paths and SHA-256 values in `AUDIT.md` when they matter to reproducibility.

## 8. Preserve and reuse Optuna state safely

Set a stable, experiment-specific SQLite name in config rather than relying on
an implicit default:

```json
{
  "optuna_storage_name": "{STATION}_{experiment_name}.sqlite3",
  "optuna_verbose": true
}
```

The current point workflow uses separate named studies for XGBoost, LightGBM,
CatBoost, and the Ridge stack. Study names also encode station, feature version,
target mode, training profile, method, metric, and search-space identity. Do not
reuse a database when any of those contracts change unless the study name also
changes and the old studies remain unambiguous.

The pipeline counts every finished Optuna state, including pruned trials,
toward the configured target. When a study already has the target number of
finished trials, `_remaining_optuna_trials` returns zero and the notebook reuses
the stored best parameters without retuning.

Before copying or replacing a database:

1. stop active notebook kernels using that database;
2. check for matching `-wal` or `-shm` sidecars;
3. make a timestamped backup of the destination;
4. copy the database as a binary file, never through Git;
5. compare SHA-256 with the source;
6. run `PRAGMA quick_check`;
7. list expected studies and count finished trials; and
8. retain the backup until the notebook completes successfully.

For the completed Tokyo/RJTT production-aligned tuning state, the canonical
local path is:

```text
data/calibration/station_training_baseline/Tokyo/
  RJTT_optuna_no_fullday_high.sqlite3
```

Its current expected SHA-256 is:

```text
95eedbd7b033e31a06c6345f9f6ce5b43cc48c5c08ff75064093f98163bd8f09
```

It contains 30 finished trials for each of the four expected studies. Do not
delete it or change the storage/study identity merely to obtain a fresh run.
If the research question requires a new search, use a new experiment database
and explain why the old parameters are not reusable.

## 9. Enforce data, feature, and unit contracts

Before training, write down and assert:

- station and settlement source;
- local timezone and inference cutoff;
- permitted provider cycles and observation timestamps;
- native target unit and any fallback conversion;
- named rounding and bucket contract;
- target-equivalent columns excluded from features; and
- required versus optional live inputs.

Every feature must be historically available by the simulated inference time.
Final daily highs, post-cutoff observations, revised settlement data, and
same-day actual-derived summaries are forbidden inputs.

For Tokyo/Asia optional-field aggregation, convert only temperature and
dewpoint dimensions between Celsius and Fahrenheit. Humidity, cloud cover,
precipitation, wind speed, and wind direction remain in their native units.
Validate physical bounds so impossible values fail instead of silently entering
the model.

Apply the point-feature missingness gate to raw/coerced training values before
imputation. The active maximum is `0.03`. Recompute eligibility inside every
training fold and again on the exact evaluation-refit population.

## 10. Keep chronology and evidence scopes separate

At every learned stage, training data must precede validation data:

```text
raw live-safe rows
    -> fold-local feature eligibility and preprocessing
    -> base-model training
    -> honest forward base predictions
    -> fold-local stack training
    -> honest forward point predictions
    -> probability training and calibration
```

Never train the stack on predictions made by base models that saw the same
targets. Never train probabilities from in-sample point predictions. Fit
imputation, scaling, feature selection, hyperparameters, calibration, blend
weights, and policy thresholds inside the applicable chronological history.

Keep these scopes distinct in outputs and prose:

| Scope | Meaning |
| --- | --- |
| Honest forward validation | Every prediction is trained only on earlier dates |
| Exploratory holdout | A frozen model is scored on a period that has now been inspected |
| Evaluation refit | Frozen pre-holdout bundle used for reproducible evidence |
| Live refit | All completed eligible actuals, for inference after separate review |
| In-sample refit | Never valid performance evidence |

Do not tune on 2026 and later call it untouched. Once inspected, a holdout
remains exploratory.

## 11. Freeze evaluation features before a live refit

The evaluation bundle establishes the exact ordered point-feature contract. A
live refit may use all completed eligible actuals, but it must:

- reuse the evaluation manifest's exact ordered feature list;
- reject every frozen feature exceeding the 3% missingness gate on live-refit
  rows;
- never add a newly dense feature;
- record the evaluation bundle and ordered-feature hashes;
- use a distinct live model identity; and
- avoid claiming the evaluation model's holdout metrics.

Live export is disabled by default. It requires:

```powershell
$env:STATION_TRAINING_EXPORT_LIVE_MODEL_WEIGHTS = "1"
```

Do not set that variable for ordinary research runs. Exporting a live candidate
does not authorize deployment.

## 12. Execute the experiment

Run static tests before expensive training. Then execute from a clean kernel
with the repository root as the working directory.

Example:

```powershell
$python = "D:\dev\weather-research\.venv-production-312\Scripts\python.exe"

& $python -m pytest tests\test_station_training_baseline.py
& $python -m pytest tests\test_bucket_probability.py
& $python -m pytest tests\test_temperature_buckets.py
& $python -m pytest tests\test_{experiment_name}_notebook.py

& $python scripts\execute_notebook_cells.py `
  notebooks\experiments\{experiment_name}\train_{STATION}.ipynb
```

Do not enable live export during the exploratory run. Do not skip failed cells,
manually inject state, or run later cells against artifacts from an earlier
configuration.

## 13. Review outputs before accepting a result

Review at least:

- source coverage, unavailable reasons, and row loss;
- selected and rejected features for every fold;
- final feature count, ordered feature hash, and worst selected missingness;
- Optuna study identity, finished-trial count, best parameters, and best value;
- continuous MAE, RMSE, bias, and large misses;
- exact market-bucket metrics using the native contract;
- monthly/seasonal stability and provider availability;
- honest-forward and exploratory-holdout results in separate tables;
- model bundle and manifest SHA-256 values; and
- probability sums, point-bundle dependency hash, and shadow-only status.

Do not accept a result solely because aggregate MAE or bucket accuracy improved.
Investigate leakage, unit mistakes, sparse features, narrow seasonal gains, and
changes in the scored population first.

## 14. Write the audit and handoff

After results are reviewed, add or update `AUDIT.md` beside the experiment. It
must include:

- source commit and dirty/clean status;
- interpreter, OS, direct dependency contract, and relevant lock identity;
- input artifact paths and hashes;
- notebook and generator paths;
- experiment config and Optuna database hash;
- chronology and row counts;
- feature count and ordered-feature hash;
- metrics by evidence scope;
- exported artifact paths and hashes;
- comparison with the active baseline;
- failure analysis and limitations; and
- disposition: reject, continue research, or propose baseline integration.

If proposing integration, list the exact generator, config, shared-code, test,
and documentation changes required. Integration requires a separate baseline
change and full SOP validation.

## 15. Final validation checklist

- [ ] Work is under `notebooks/experiments/{experiment_name}/`.
- [ ] `README.md`, generator, config, notebook, test, and reviewed `AUDIT.md`
      exist as applicable.
- [ ] The parent baseline path and source commit are recorded.
- [ ] Generator changes precede generated notebook changes.
- [ ] Experiment artifacts and Optuna storage cannot overwrite baseline files.
- [ ] Python is 3.12.3 and all direct runtime pins match.
- [ ] Providers, cutoff, timezone, units, settlement source, and bucket contract
      are station-correct.
- [ ] Target leakage and post-cutoff data are excluded.
- [ ] Missingness is fold-local, pre-imputation, and no weaker than 3%.
- [ ] Stack and probability inputs are honest chronological predictions.
- [ ] Evaluation, exploratory holdout, live refit, and in-sample scopes are
      labeled separately.
- [ ] Optuna storage is intact and existing completed studies are reused when
      the search contract is unchanged.
- [ ] Live export remains disabled for research.
- [ ] Probability artifacts remain shadow-only without explicit promotion.
- [ ] Notebook JSON parses, ordinary Python cells compile, and regeneration is
      deterministic at the source-cell level.
- [ ] Relevant tests pass and `git diff --check` is clean.
- [ ] No generated `data/`, `tmp/`, environment, or model-weight files are
      staged for Git.

## Common mistakes

| Mistake | Consequence | Correct action |
| --- | --- | --- |
| Hand-edit the `.ipynb` only | Regeneration deletes the change | Edit the generator/config first |
| Copy an experiment into the baseline | Creates an unreviewed second source of truth | Integrate accepted behavior through the active generator |
| Reuse a baseline artifact directory | Overwrites trusted evidence | Use an experiment-isolated output root |
| Delete Optuna SQLite to “start clean” | Repeats expensive tuning and loses provenance | Reuse it when the search identity is unchanged; otherwise use a new named DB |
| Assume Git transferred an SQLite database | The checkout silently uses an old or missing study | Copy and verify ignored artifacts explicitly |
| Compare notebook hashes only | Shared pipeline differences are missed | Record and compare imported shared modules and source commit |
| Run with the old Python 3.14 venv | Does not match release interpreter | Use `.venv-production-312` |
| Convert humidity or wind as temperature | Corrupts Asia features | Convert only temperature/dewpoint dimensions |
| Select features after imputation | Hides missingness | Gate raw/coerced fold-training values first |
| Retune or calibrate on the holdout | Invalidates evidence | Keep the inspected period exploratory |
| Enable live export during research | Produces a release-shaped artifact without approval | Leave the environment switch unset |
