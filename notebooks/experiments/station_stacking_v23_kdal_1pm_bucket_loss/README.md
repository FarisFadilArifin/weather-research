# KDAL V23 1 PM bucket-loss challenger

This notebook is an isolated challenger to the V20 KDAL 1 PM no-peak point model.
It preserves the same data and feature contracts while changing Optuna selection
from `mae_f` to `bucket_log_loss`. It does not add Wunderground source-alignment
features. The exported point bundle is consumed by the V23 win selector, whose
confidence threshold is frozen from chronological 2024-2025 predictions before
the 2026 holdout is evaluated.

To keep the full 30-trial CatBoost search computationally bounded, V23 uses the
pipeline's native caps of 1,500 iterations, depth 8, and 128 borders. XGBoost and
LightGBM retain the V20 wide search space.
