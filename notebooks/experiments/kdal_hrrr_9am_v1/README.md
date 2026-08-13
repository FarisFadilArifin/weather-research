# KDAL Exact-09:00-Local HRRR V1

This is an isolated research fork of the active KDAL station-training notebook.
It retains the KDAL V20 no-peak/full-refit lineage and changes only the HRRR
provider source from the baseline 11 AM live-safe selection to the completed
exact-09:00 `America/Chicago` candidate cache.

## Contracts

- Notebook: `train_KDAL.ipynb`
- Generator/config source of truth: `generate_notebook.py` and `config.json`
- HRRR timing mode: `same_day_11am_hrrr_9am_cycle_v1`
- HRRR issue: exact 09:00 local, mapping to 14Z in CDT and 15Z in CST
- HRRR forecast window: f02-f14
- Candidate coverage: 2021-01-01 through 2026-08-08, 2,046 valid unique rows
- Observation cutoff: unchanged at 11:00 local
- Prediction/decision time: unchanged at 11:15 local
- Research artifacts: `data/calibration/experiments/kdal_hrrr_9am_v1/`
- Read-only historical data root: `D:/dev/weather-research/`

GFS, NBM, Wunderground settlement labels, feature engineering, V20 folds,
model settings, the 3% fold-owned missingness gate, bucket contract, and
pure-ordinal chronology remain inherited from the active KDAL baseline.

The provider override is intentionally HRRR-only. The global station timing
mode stays `same_day_11am_live_safe`, so the loader cannot substitute 09:00,
10:00, or 11:00 HRRR rows from other caches and cannot shift the observation
snapshot or the GFS/NBM contracts.

The ignored historical inputs remain in the canonical `D:/dev/weather-research`
checkout because Git worktrees do not carry them. The generated notebook reads
those inputs but writes its distinct research artifacts beneath the current
worktree's `data/calibration/experiments/kdal_hrrr_9am_v1/` directory.

The active three-arm challenger is omitted from this experiment because its
current runner is hard-coded to active-baseline artifact paths. The pure ordinal
stage remains in the notebook. Re-enabling the challenger would require a
separate parameterized, research-path-safe runner; it must not read or overwrite
the active KDAL artifacts.

## Generate and validate

```powershell
python notebooks\experiments\kdal_hrrr_9am_v1\generate_notebook.py
python -m pytest tests\test_kdal_hrrr_9am_v1_notebook.py
```

Do not execute training until the notebook's exact-HRRR readiness cell passes.
Execution writes only research artifacts. It does not authorize production
promotion, deployment, or any write to `polymarket-weather-prediction`.
