# KDAL V26 XGBoost + Extra Trees + RBF-SVR

V26 keeps the previously tuned V20 XGBoost and V24 Extra Trees predictions fixed. It tunes only
an RBF-SVR with 30 Optuna trials across the same four expanding validation folds.

The three base predictions are combined with a non-negative simplex grid whose weights sum to one.
Selection minimizes equal-fold mean MAE, then worst-fold MAE. The grid step is 0.005 and permits
any model to receive zero weight.

Outputs are written to
`data/calibration/station_stacking_v26_kdal_xgb_extra_svr_blend`.

The 2026 results are exploratory because that period has already been inspected during prior model
development. Fresh shadow data is required for any promotion decision.
