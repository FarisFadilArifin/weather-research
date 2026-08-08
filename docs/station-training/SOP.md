# Station Training Notebook SOP

Use this procedure to create, retrain, or change a station notebook. The
structural and artifact requirements are defined in `NOTEBOOK_STANDARD.md`.

## 1. Define the station contract

Copy the closest verified config—KDAL for the Dallas-style pipeline or
Seoul/Tokyo for the Asia 11 AM pipeline—to `configs/{STATION}.json` and set:

- station ID and human-readable name;
- station-specific source generator and builder argument;
- source model, output, and pipeline replacement tokens;
- point-workflow label and output notebook path;
- isolated station artifact directory;
- point- and probability-model versions;
- verified probability feature profile; and
- probability providers, feature count, development years, and forward
  validation years; and
- exploratory holdout year.

Do not copy KDAL providers, observation timing, timezone, target source, feature
availability, or market rules without validating them for the new station.

`ordinal_challenger_enabled` is currently a KDAL-only option. A new station must
leave it false or absent until its three-arm feature and chronology contracts are
implemented and tested.

## 2. Audit data availability before coding

Record and verify:

- inference cutoff and timezone;
- settlement target and rounding rule;
- provider issue times and forecast availability;
- observation timestamps and maximum acceptable age;
- features available by the cutoff;
- training years, forward-validation years, and holdout year; and
- station-isolated input/output paths.

Reject features that use the final daily high, revised data, or observations
arriving after inference time.

## 3. Change generator-backed sources

Make structural changes in:

1. the station-specific point-notebook generator;
2. `generate_station_notebook.py`; and/or
3. the station config.

Do not make a notebook-only change that will disappear on regeneration.

Generate the notebook:

```powershell
python notebooks\station_training_baseline\generate_station_notebook.py `
  --config notebooks\station_training_baseline\configs\{STATION}.json
```

For KDAL:

```powershell
python notebooks\station_training_baseline\generate_station_notebook.py `
  --config notebooks\station_training_baseline\configs\KDAL.json
```

For the current Asia baselines:

```powershell
python notebooks\station_training_baseline\generate_station_notebook.py `
  --config notebooks\station_training_baseline\configs\Seoul.json

python notebooks\station_training_baseline\generate_station_notebook.py `
  --config notebooks\station_training_baseline\configs\Tokyo.json
```

## 4. Run static contract tests

Run the generator and challenger tests before the expensive training run:

```powershell
python -m pytest `
  tests\test_station_training_baseline.py `
  tests\test_kdal_ordinal_challenger.py
```

For a new station, add equivalent tests for notebook identity, stage ordering,
chronology, output paths, and artifact-integrity assertions.

## 5. Execute the complete notebook

Start from a clean kernel and run every cell in order. For a non-interactive
check:

```powershell
python scripts\execute_notebook_cells.py `
  notebooks\station_training_baseline\stations\{STATION}\train_{STATION}.ipynb
```

Do not skip the point-model export before probability training. KDAL's
three-arm challenger reloads the just-exported baseline feature, prediction, and
point-weight artifacts so the command-line runner and notebook share one
implementation.

The default point export is the evaluation bundle and must pass explicit
`train_years` ending before the holdout plus the configured feature-missingness
gate. Probability and challenger exports bind to this frozen bundle. The
optional live-production export uses a separate model version, may include the
latest completed actuals, and stays disabled until a clean source commit and
separate release-provenance review are available.

## 6. Review training evidence

Confirm:

- the point model uses only live-safe features;
- forward point predictions precede ordinal training;
- validation years are trained only on earlier years;
- inner tuning/calibration is chronological;
- offset, degree, and bucket probabilities each sum to one;
- feature counts match the selected feature set;
- the 2026 data is labeled exploratory, not OOF;
- no probability arm overrides the point bucket; and
- warnings, missingness, row loss, and coverage changes are explained.

For KDAL, the challenger comparison must contain exactly:

```text
blended_ordinal
shared_slope_ordinal
pure_ordinal
```

## 7. Verify exported artifacts

Under `data/calibration/station_training_baseline/{STATION}/`, check:

- point prediction/metric CSVs;
- point `.joblib` and JSON manifest;
- pure ordinal prediction/metric CSVs;
- pure ordinal `.joblib` and JSON manifest; and
- enabled challenger prediction CSVs, comparison tables, `.joblib` bundles,
  JSON manifests, and summary.

For KDAL the challenger directory is:

```text
data/calibration/station_training_baseline/KDAL/ordinal_challenger_v1/
```

The notebook must finish only after confirming every expected weight and
manifest exists, each bundle SHA-256 matches its manifest, and each probability
manifest is bound to the exported point-bundle SHA-256.

## 8. Check reproducibility

Regenerate the notebook once more and rerun:

```powershell
python -m pytest tests\test_station_training_baseline.py
```

Unexpected source/cell differences mean the generated notebook and its source
are out of sync. Resolve that before review.

## 9. Promotion and historical-work rules

Training and export do not authorize production promotion. Previously inspected
2026 results are exploratory. Keep all probability arms shadow-only until fresh
station-specific evidence passes a separate promotion review.

Before an approved candidate can be handed off, create and verify an immutable
release record as described in `RELEASE_PROVENANCE.md`. The registry requires a
clean source commit and hashes the exact dataset, features, generated notebook,
export, model, ordinary manifest, and runtime inputs. It rejects dirty or
mismatched inputs and must never overwrite a mutable bundle.

`git_dirty: true` in an ordinary model manifest means tracked source content at
training time was not fully represented by the recorded commit. The bundle can
be used for local research, but it is not reproducible from that commit and is
not eligible for an immutable release record. Commit the intended source first,
then retrain from the clean checkout; do not suppress or rewrite the dirty flag.

Put temporary comparisons in an explicit experiment directory. Preserve
historical versioned notebooks as evidence, and merge accepted behavior into the
non-versioned station baseline through its generator.
