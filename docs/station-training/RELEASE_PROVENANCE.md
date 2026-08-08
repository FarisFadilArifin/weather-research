# Station Training Release Provenance

Station-training exports remain research artifacts unless a separate promotion
review approves them. A promotion candidate must have an immutable release
record created with `src.calibration.release_provenance`; this is deliberately
separate from mutable model bundles and their ordinary export manifests.

## What a release record binds

A record is created only from a clean Git checkout and records SHA-256 hashes
for the dataset, feature frame, generated notebook, exported evaluation file,
model bundle, ordinary model manifest, and runtime inputs (for example
`pyproject.toml` and `uv.lock`). It also records the source commit, provider
contract, required chronological history, and feature-missingness gate.

`build_release_manifest(...)` fails closed when any required input is absent,
outside the repository, or the source checkout is dirty. `write_release_manifest`
creates a new file with exclusive creation and will never overwrite an existing
record. `load_and_verify_release_manifest(...)` rechecks the clean commit and
every recorded hash before a release can be consumed; callers should pass the
station's expected provider contract as well.

No release record is supplied for the current baseline. The repository's active
restructure is dirty, so recording or promoting a mutable input would be
incorrect. Create the record only after the intended source and training inputs
are committed, all required evidence is present, and the separate promotion
review authorizes the candidate.

## Required evidence shape

The artifact keys are exactly:

- `dataset`, `features`, `notebook`, `export`, `model`, and `model_manifest`;
- at least one named runtime file;
- a provider contract containing the ordered provider list and station-specific
  cutoff/timezone fields as applicable;
- sorted training and forward-validation years, a later holdout year, and an
  explicit `selection_excludes_holdout: true`; and
- observed fractions for each gated feature that meet the declared maximum
  missingness threshold.

The record is an audit gate, not a model exporter. It must not be used to write,
replace, or promote a model bundle.
