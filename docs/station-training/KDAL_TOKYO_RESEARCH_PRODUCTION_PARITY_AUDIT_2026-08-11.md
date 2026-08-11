# KDAL and Tokyo research/production parity audit

Audit date: 2026-08-11

## Result

Tokyo is already feature- and artifact-identical between research and
production. KDAL is not: the evaluated research bundle has 198 ordered features,
while the released live-refit bundle has 230. Both routes already use the same
production runtime-package contract.

The refit exporter is patched so an optional live refit must consume the exact
ordered feature list from its evaluation manifest. It revalidates those frozen
features on all completed refit rows and fails closed if one is absent or exceeds
the 3% missingness limit. Newly dense features cannot enter the live model.

## Evidence

| Contract | Research | Released production | Assessment |
|---|---:|---:|---|
| KDAL ordered point features | 198 | 230 | Mismatch; production replacement required after clean retrain/release |
| KDAL refit rows | 1,447 evaluation rows | 1,635 all-available rows | Expected population difference |
| KDAL source feature CSV SHA-256 | `917ceefdb2484e5932bc682f2c04eeb5b44f7162169b0edccacaeb2ef86250f9` | Same immutable seed | Identical source bytes |
| Tokyo ordered point features | 293 | 293 | Exact order parity |
| Tokyo bundle SHA-256 | `9095e1105f105e576ca6a48e6a9b129d430f3d8be089ca029cf9ae56fea0c8c3` | Same | Exact bundle-byte parity |
| Tokyo source feature CSV SHA-256 | `111d9a70850f7576c56087354de34e2d5d2862575c9b8f8a05b6edc1809f0661` | Same release source | Identical source bytes |
| Direct runtime-package pins | 14 | 14 | Exact, zero differences |

The released KDAL bundle is
`cb9a485fb25b5de5e6993150f4e122b365f3241b338df3f9498c81dcb696850d`.
The frozen 198-feature evaluation bundle is
`5d7ea0bf7bf9d8a84d15fb66b75f76f487b50fa9e11c90eb34fd752950a30d43`.
These hashes are expected to differ because a valid live refit uses more
completed training rows even while preserving the same feature contract.

All 198 evaluated KDAL features are a subset of the released 230-feature bundle.
The 32 live-only features were admitted by the older release's refit-selection
implementation. Re-evaluating the same immutable source CSV with the current
clean source rejects all 32 above the 3% threshold even without freezing the
list. They are mainly wind/ceiling observations, since-11-AM change features,
heat-index/wind-chill fields, and v4/v8 precipitation, cloud, wind, and dewpoint
interactions. This was source/refit selection drift, not a different source CSV.

## Frozen-refit viability check

The patched selector was run against the real KDAL all-available modeling frame:

- refit rows: 1,635;
- requested research features: 198;
- selected live-refit features: 198;
- exact ordered-list match: yes;
- ordered feature SHA-256:
  `098ec973a8ec9f8dfcf165cebb24c9ed738741e2b3b0ba70951d0888f2b6e16f`;
- worst selected-feature training missingness: 2.1407%; and
- selected features above the 3% gate: 0.

Therefore KDAL can be refit on all completed actuals as a genuine 198-feature
model without weakening the missingness guard.

## Runtime-package contract

Research and production now declare the same exact direct versions in
`requirements-ml-runtime.txt`: pandas 3.0.5, numpy 2.5.1, requests 2.34.2,
xarray 2026.7.0, cfgrib 0.9.15.1, eccodes 2.47.0, ecmwflibs 0.7.0,
scikit-learn 1.9.0, joblib 1.5.3, xgboost 3.3.0, lightgbm 4.7.0,
catboost 1.2.10, mostlyrightmd 1.17.0, and mostlyrightmd-weather 1.17.0.

Production additionally fixes CPython 3.12.3 and a fully hash-pinned Linux
x86-64 transitive lock. Research's project metadata now requires Python 3.12 or
newer and exactly pins the 14 direct packages above.

## Meaning of live input density

Live input density is the proportion of a model's manifest features that have a
real, finite value in one same-day inference row:

```text
input density = non-missing manifest features / manifest feature count
```

For the audited KDAL live row, the released 230-feature model received 161
non-missing values and 69 missing values: 70.0% density and 30.0% missingness.
This does not mean the historical training dataset has only 70% of its rows. It
means that one live prediction row is sparse; optional missing values are passed
as NaN and handled by fitted preprocessing, while hard-required provider and
observation fields still fail closed.

Feature-count parity and input density are separate controls. Freezing KDAL at
198 guarantees research/live dimensional parity. The readiness audit must still
report same-day non-missing counts and missing groups for that 198-feature
contract after release.

## Release status

This patch does not alter the active production artifact or trading mode. KDAL
will remain on the released 230-feature bundle until the research changes are
committed, the live refit is exported from a clean checkout, the new bundle and
manifest pass cross-runtime parity and route validation, and the production
release registry is updated through the normal deployment review. Tokyo needs no
artifact replacement for feature or package parity.
