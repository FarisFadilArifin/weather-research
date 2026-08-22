# Station Training Baseline

The canonical baseline is a generator-backed, station-code workflow for four
stations. Each notebook trains exactly one XGBoost point model, keeps Gaussian
as the probability benchmark, exports four ordinal research candidates, and
evaluates a canonical three-member ordinal ensemble built from honest point
predictions.

| Station | Notebook | Config | Native market buckets |
|---|---|---|---|
| KDAL | `stations/KDAL/train_KDAL.ipynb` | `configs/KDAL.json` | 2°F |
| RJTT | `stations/RJTT/train_RJTT.ipynb` | `configs/RJTT.json` | 1°C |
| RKSI | `stations/RKSI/train_RKSI.ipynb` | `configs/RKSI.json` | 1°C |
| RKPK | `stations/RKPK/train_RKPK.ipynb` | `configs/RKPK.json` | 1°C |

The shared implementation is:

- `src/calibration/station_baseline.py`: point-training and artifact orchestration;
- `src/calibration/station_probability_models.py`: Gaussian and ordinal models;
- `generate_station_notebook.py`: deterministic station-code notebooks.

## Model contract

1. Build the point-in-time station feature frame at 11 AM local.
2. Tune one XGBoost regressor with 100 Optuna trials using
   `TPESampler(n_startup_trials=40)`.
3. Produce chronological XGBoost point predictions.
4. Fit a conditional Gaussian residual probability baseline.
5. Fit native-reference, blended, shared-slope, and pure ordinal candidates.
6. Require all blended/shared-slope/pure artifacts and evaluate their
   station-specific two-of-three confidence vote.
7. Aggregate selected-bucket probability with the median of those three voting
   members; the native reference remains non-voting.
8. Compare candidates and ensembles on market-bucket log loss, Brier score,
   calibration, agreement, gate coverage, and top-bucket accuracy.
9. Export frozen evaluation artifacts and separately named live-production
   candidates.

No LightGBM, CatBoost, Ridge stack, or other point ensemble is part of this
baseline. Every probability candidate depends only on chronological XGBoost
point predictions.

## Generate notebooks

```powershell
python notebooks\station_training_baseline\generate_station_notebook.py --all
```

Run one notebook with:

```powershell
python scripts\execute_notebook_cells.py `
  notebooks\station_training_baseline\stations\RJTT\train_RJTT.ipynb
```

Generated data remains under
`data/calibration/station_training_baseline/{STATION}/` and is not source of
truth. Production-candidate artifacts from a dirty worktree remain unapproved;
they require a separate clean-commit promotion review before deployment.

Each run also writes `{STATION}_trading_backtest_input.csv`, containing the
per-family market distributions, point/top buckets, ensemble votes, and
approval state. The trading engine joins this file to historical market mids;
the training notebook does not invent prices or calculate P&L without that
external market history.
