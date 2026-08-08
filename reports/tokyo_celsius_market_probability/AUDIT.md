# Tokyo 1°C Market Probability Audit

Status: **research complete; shadow-only; not approved for production promotion.**

## Settlement and target contract

Tokyo Polymarket rules resolve the RJTT daily high at Tokyo Haneda Airport
using Wunderground at whole degrees Celsius. A representative market states
both the station/Wunderground source and whole-°C precision:
[Highest temperature in Tokyo on May 5, 2026](https://polymarket.com/event/highest-temperature-in-tokyo-on-may-5-2026).
This validates the exact 1°C bucket contract, not the historical 2°F contract.

The source frame does not contain a settlement-equivalent raw Celsius target.
`iem_daily_high_c` belongs to a different diagnostic IEM source, so it is not
substituted for the Wunderground target. `actual_high_f` is converted exactly by
`(F - 32) * 5 / 9` and then rounded by `floor(C + 0.5)`. All 1486
available converted targets are integer Celsius to floating tolerance, which is
consistent with Wunderground's whole-Celsius source values.

Target: `point_bucket_c=round_half_up((point_prediction_f-32)*5/9); actual_bucket_c=round_half_up(actual_high_c); offset_c=actual_bucket_c-point_bucket_c`.

## Artifact identities

- Point model: `station_high_regressor_baseline_tokyo_no_peak_stack`
- Point bundle SHA-256: `fe43b7e68a2914db14978cdab6b8c1a86ba8ad1b259cf09d4d5e26fb6e3d7a67`
- Probability model: `station_bucket_baseline_tokyo_1c_market_ordinal`
- Probability bundle SHA-256: `42a739f6f1c304116373f2899d3202eac30b94322e1393c37d2d91b12d252ad4`
- Probability manifest point dependency: `fe43b7e68a2914db14978cdab6b8c1a86ba8ad1b259cf09d4d5e26fb6e3d7a67`
- Feature contract: `asia_no_peak`, 59 features, providers GFS/GEFS/JMA-MSM
- Probability learner: `celsius_offset_ordinal_logistic`, C=0.03, class_weight=None, temperature=1.0

The probability dependency hash equals the freshly exported point bundle hash.

## Chronology

| Stage | Period | Use |
|---|---|---|
| Honest 2025 outer-model training | through `2024-12-31` | Fit the 2025 forward fold |
| 2025 fold inner training | through `2024-10-02` | Candidate preprocessing/model fit |
| 2025 fold inner calibration | `2024-10-03` to `2024-12-31` | C/weight/temperature selection |
| Forward validation | `2025-01-01` to `2025-12-31` | Model-development metrics and policy thresholds |
| Final probability development | `2024-01-01` to `2025-12-31` (731 rows) | Frozen probability model |
| Final inner calibration | `2025-10-03` to `2025-12-31` | Final C/weight/temperature selection |
| Exploratory holdout | `2026-01-01` to `2026-07-25` | Metrics only; never model/policy selection |

The point bundle is the current production-style refit over
`2022-07-03` to
`2026-07-25`. Probability development
does not use its in-sample predictions: it uses honest expanding point-stack
predictions for 2024–2025 and the 2026 point holdout predictions. The point
bundle hash records the serving dependency.

## Ordered offset support and frozen policy

- Classes: `<=-3, -2, -1, 0, +1, +2, >=+3 Celsius degrees`
- Forward-fold exact support (tail policy fitted on the earlier 2024 history):
  `[-4, -3, -2, -1, 0, 1, 2, 3]`
- Final-model exact development support: -4°C through 3°C
- Low tail allocation: `{"-3": 0.7222222222222222, "-4": 0.2777777777777778}`
- High tail allocation: `{"3": 1.0}`
- Minimum top probability: 0.425
- Minimum top-two margin: 0.175
- Minimum switch advantage: 0.000
- Tail rule: reject when an open-tail class spans multiple exact market buckets
  and its mass is at least the top-two margin.

All thresholds and the tail rule were frozen from 2025 forward-validation data.
No 2026 ROI, P&L, model variant, or threshold influenced selection.

## Model-development metrics (2025 forward validation, n=365)

- Market bucket accuracy: 0.5342
- Nearest-Celsius point accuracy: 0.5260
- Market log loss: 1.2179
- Market Brier score: 0.6102
- Offset ranked probability score: 0.0665
- Offset calibration error: 0.0730
- Decision coverage: 0.5671 (207 rows)
- Decision accuracy: 0.6329
- Calibration table: `data/calibration/station_training_baseline/Tokyo/celsius_market_probability/RJTT_forward_validation_calibration.csv`

## Exploratory 2026 holdout metrics (n=206)

- Market bucket accuracy: 0.5437
- Nearest-Celsius point accuracy: 0.5631
- Market log loss: 1.1465
- Market Brier score: 0.6013
- Offset ranked probability score: 0.0653
- Offset calibration error: 0.0610
- Decision coverage: 0.6845 (141 rows)
- Decision accuracy: 0.6028
- Calibration table: `data/calibration/station_training_baseline/Tokyo/celsius_market_probability/RJTT_2026_holdout_calibration.csv`

## Required 2026 comparison

| Comparison | Unit | Accuracy | Log loss | Brier | Decision coverage | Decision accuracy |
|---|---:|---:|---:|---:|---:|---:|
| nearest_celsius_point_bucket | 1C | 0.5631 | — | — | 1.0000 | 0.5631 |
| old_mapped_fahrenheit_probabilities | mapped_to_1C_post_hoc | 0.5874 | 1.1139 | 0.5897 | 1.0000 | 0.5874 |
| old_native_fahrenheit_probability_decision | native_2F | 0.5194 | — | — | 0.5291 | 0.5780 |
| new_celsius_ordinal_probabilities | 1C | 0.5437 | 1.1465 | 0.6013 | 0.6845 | 0.6028 |

The old mapped-Fahrenheit distribution is diagnostic only: integer-Fahrenheit
degree probabilities were converted and aggregated to nearest whole Celsius
after prediction. It was not trained on Celsius offsets. The old native
`probability_decision` evaluates incompatible 2°F buckets and must not be used
as a Tokyo market filter. Its displayed accuracy is native 2°F accuracy, not
1°C accuracy, and is included only to document the historical policy.

On the already-inspected 2026 holdout, the new model has 0.5437 accuracy versus 0.5631 for the nearest-C point bucket and 0.5874 for the mapped legacy distribution. Its log loss is 1.1465 versus 1.1139 for mapped legacy, while its Brier score is 0.6013 versus 0.5897. This mixed, previously inspected evidence cannot be used for retuning and does not justify promotion.

## Point-model weights for later handoff

- XGBoost point-stack coefficient: 0.332843044577251
- LightGBM point-stack coefficient: 0.332469512060160
- CatBoost point-stack coefficient: 0.333286800465310
- Intercept: 0.0
- Ridge alpha: 54498.00722754935850

Use the point bundle itself rather than copying these coefficients in isolation;
it also contains the three fitted base models, preprocessing, feature lists,
and inference contract.

## Production handoff (future work only)

Do not modify `D:\dev\polymarket-weather-prediction` yet. A later reviewed
integration should:

1. Load the exact point bundle above and verify its SHA-256.
2. Load the exact Celsius probability bundle above and verify both its own hash
   and `point_bundle_sha256` dependency.
3. Supply the 59 `asia_no_peak` features and GFS/GEFS/JMA-MSM inputs at Tokyo's
   live-safe 11 AM cutoff.
4. Consume `point_bucket_c`, `recommended_bucket_c`,
   `recommended_bucket_probability_c`, `actual_bucket_probability_c`,
   `market_top_probability_c`, `market_top_two_margin_c`,
   `market_switch_advantage_c`, `market_tail_ambiguity_c`,
   `market_probability_decision`, `market_probability_decision_reason`,
   `celsius_offset_probabilities`, and `market_bucket_probabilities_c`.
5. Freeze the thresholds listed above. Never fall back to the old native
   Fahrenheit `probability_decision`.
6. Keep outputs shadow-only pending a separate promotion review on fresh,
   previously unseen Tokyo market outcomes.

## Limitations

- The target is an exact conversion of Wunderground-derived Fahrenheit values,
  because no same-source raw Celsius column is stored. Future ingestion should
  persist the raw Wunderground Celsius settlement value directly.
- Open-tail allocation is data-limited: low-tail exact support is
  `-4, -3` and high-tail exact support is `3`;
  multi-bucket tails can trigger ambiguity rejection.
- Only one 365-row forward-validation year is
  available for probability policy development.
- The 2026 holdout was previously inspected and is exploratory only.
- Market prices, liquidity, slippage, and ROI are outside this model audit.

## Artifact SHA-256

- `data\calibration\station_training_baseline\Tokyo\celsius_market_probability\RJTT_2026_holdout_calibration.csv` — `7c1c73f6ac3997641428ecbefceddc79ee0635b062740ed7e2d3447fe870bc1a`
- `data\calibration\station_training_baseline\Tokyo\celsius_market_probability\RJTT_2026_holdout_metrics.csv` — `12b12976aa13605ad66e5442bbd30b95b7be31f87fc0159728178d0e855cdeef`
- `data\calibration\station_training_baseline\Tokyo\celsius_market_probability\RJTT_2026_holdout_predictions.csv` — `6304139146cb6435b92430cc42596439ff00c9ee645de65c4b4bb8b10457ea43`
- `data\calibration\station_training_baseline\Tokyo\celsius_market_probability\RJTT_celsius_feature_contract.csv` — `0da3fd45cc2b6194f11fdac4786171bcab619b73ea0f65c5f79bc2f8fe0cd9fc`
- `data\calibration\station_training_baseline\Tokyo\celsius_market_probability\RJTT_celsius_target_contract.json` — `2e8639fb275de313ba2ba13d83eaf2bfe8d08fadc5c35915927ab5aa47460269`
- `data\calibration\station_training_baseline\Tokyo\celsius_market_probability\RJTT_forward_validation_calibration.csv` — `81b5a65495c76a1134f5060e48224cc7f90bb5184a27b03297c57218ab5075d9`
- `data\calibration\station_training_baseline\Tokyo\celsius_market_probability\RJTT_forward_validation_metrics.csv` — `a2bde85fbfe5a6f5e646b6a7ce31865d79c7f3711846696f9d8648bccf27e0b0`
- `data\calibration\station_training_baseline\Tokyo\celsius_market_probability\RJTT_forward_validation_predictions.csv` — `84ba434a5ffea114051237f947cc1d0397988fe0bc3f8d662c31ed1f1493ca56`
- `data\calibration\station_training_baseline\Tokyo\celsius_market_probability\RJTT_pre_2026_tuning.csv` — `2acd43515d2eb4be75d3509b38d2930c148d8c30892ebf2411e0c2cc3a30f9e1`
- `data\calibration\station_training_baseline\Tokyo\celsius_market_probability\model_weights\RJTT_station_bucket_baseline_tokyo_1c_market_ordinal.joblib` — `42a739f6f1c304116373f2899d3202eac30b94322e1393c37d2d91b12d252ad4`
- `data\calibration\station_training_baseline\Tokyo\celsius_market_probability\model_weights\RJTT_station_bucket_baseline_tokyo_1c_market_ordinal.json` — `baac92de85f9ec45a551d797c87e88732149e7f49ba077c39402164e030689e2`
- `data\calibration\station_training_baseline\Tokyo\model_weights\RJTT_station_high_regressor_baseline_tokyo_no_peak_stack.joblib` — `fe43b7e68a2914db14978cdab6b8c1a86ba8ad1b259cf09d4d5e26fb6e3d7a67`
- `data\calibration\station_training_baseline\Tokyo\model_weights\RJTT_station_high_regressor_baseline_tokyo_no_peak_stack.json` — `27c68a87b5063d28d0a586fdb8486e013851f0aa6a413f351658a265146e0a86`
