# Seoul 1°C Market Probability Audit

Status: **research complete; shadow-only; not approved for production promotion.**

## Settlement and target contract

Seoul Polymarket rules resolve the RKSI daily high at Incheon Intl Airport
using Wunderground at whole degrees Celsius. A representative market states
both the station/Wunderground source and whole-°C precision:
[Highest temperature in Seoul on May 6, 2026](https://polymarket.com/event/highest-temperature-in-seoul-on-may-6-2026/highest-temperature-in-seoul-on-may-6-2026-13c).
This validates the exact 1°C bucket contract, not the historical 2°F contract.

The source frame does not contain a settlement-equivalent raw Celsius target.
`iem_daily_high_c` belongs to a different diagnostic IEM source, so it is not
substituted for the Wunderground target. `actual_high_f` is converted exactly by
`(F - 32) * 5 / 9` and then rounded by `floor(C + 0.5)`. All 1486
available converted targets are integer Celsius to floating tolerance, which is
consistent with Wunderground's whole-Celsius source values.

Target: `point_bucket_c=round_half_up((point_prediction_f-32)*5/9); actual_bucket_c=round_half_up(actual_high_c); offset_c=actual_bucket_c-point_bucket_c`.

## Artifact identities

- Point model: `station_high_regressor_baseline_seoul_no_peak_stack`
- Point bundle SHA-256: `d01042e34d41d4e6f9e769c6dcb50dbe97cde2e10ffb640d50d06df43245581e`
- Probability model: `station_bucket_baseline_seoul_1c_market_ordinal`
- Probability bundle SHA-256: `129bd4ce8e5d7c8280a83c71c597cf65bd12bcc290e00d7afa9665ad29335189`
- Probability manifest point dependency: `d01042e34d41d4e6f9e769c6dcb50dbe97cde2e10ffb640d50d06df43245581e`
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
| Final probability development | `2024-01-01` to `2025-12-31` (730 rows) | Frozen probability model |
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
  `[-3, -2, -1, 0, 1, 2, 3]`
- Final-model exact development support: -3°C through 5°C
- Low tail allocation: `{"-3": 1.0}`
- High tail allocation: `{"3": 0.6842105263157895, "4": 0.15789473684210525, "5": 0.15789473684210525}`
- Minimum top probability: 0.400
- Minimum top-two margin: 0.025
- Minimum switch advantage: 0.175
- Tail rule: reject when an open-tail class spans multiple exact market buckets
  and its mass is at least the top-two margin.

All thresholds and the tail rule were frozen from 2025 forward-validation data.
No 2026 ROI, P&L, model variant, or threshold influenced selection.

## Model-development metrics (2025 forward validation, n=364)

- Market bucket accuracy: 0.4203
- Nearest-Celsius point accuracy: 0.4451
- Market log loss: 1.5466
- Market Brier score: 0.6934
- Offset ranked probability score: 0.0893
- Offset calibration error: 0.0486
- Decision coverage: 0.5824 (212 rows)
- Decision accuracy: 0.5189
- Calibration table: `data/calibration/station_training_baseline/Seoul/celsius_market_probability/RKSI_forward_validation_calibration.csv`

## Exploratory 2026 holdout metrics (n=203)

- Market bucket accuracy: 0.4138
- Nearest-Celsius point accuracy: 0.4187
- Market log loss: 1.3944
- Market Brier score: 0.6909
- Offset ranked probability score: 0.0859
- Offset calibration error: 0.0507
- Decision coverage: 0.5271 (107 rows)
- Decision accuracy: 0.4766
- Calibration table: `data/calibration/station_training_baseline/Seoul/celsius_market_probability/RKSI_2026_holdout_calibration.csv`

## Required 2026 comparison

| Comparison | Unit | Accuracy | Log loss | Brier | Decision coverage | Decision accuracy |
|---|---:|---:|---:|---:|---:|---:|
| nearest_celsius_point_bucket | 1C | 0.4187 | — | — | 1.0000 | 0.4187 |
| old_mapped_fahrenheit_probabilities | mapped_to_1C_post_hoc | 0.4138 | 1.6948 | 0.6704 | 1.0000 | 0.4138 |
| old_native_fahrenheit_probability_decision | native_2F | 0.3842 | — | — | 0.6404 | 0.4538 |
| new_celsius_ordinal_probabilities | 1C | 0.4138 | 1.3944 | 0.6909 | 0.5271 | 0.4766 |

The old mapped-Fahrenheit distribution is diagnostic only: integer-Fahrenheit
degree probabilities were converted and aggregated to nearest whole Celsius
after prediction. It was not trained on Celsius offsets. The old native
`probability_decision` evaluates incompatible 2°F buckets and must not be used
as a Seoul market filter. Its displayed accuracy is native 2°F accuracy, not
1°C accuracy, and is included only to document the historical policy.

On the already-inspected 2026 holdout, the new model has 0.4138 accuracy versus 0.4187 for the nearest-C point bucket and 0.4138 for the mapped legacy distribution. Its log loss is 1.3944 versus 1.6948 for mapped legacy, while its Brier score is 0.6909 versus 0.6704. This mixed, previously inspected evidence cannot be used for retuning and does not justify promotion.

## Point-model weights for later handoff

- XGBoost point-stack coefficient: 0.334502808187234
- LightGBM point-stack coefficient: 0.333255243451752
- CatBoost point-stack coefficient: 0.332841139931583
- Intercept: 0.0
- Ridge alpha: 22121.92165899044630

Use the point bundle itself rather than copying these coefficients in isolation;
it also contains the three fitted base models, preprocessing, feature lists,
and inference contract.

## Production handoff (future work only)

Do not modify `D:\dev\polymarket-weather-prediction` yet. A later reviewed
integration should:

1. Load the exact point bundle above and verify its SHA-256.
2. Load the exact Celsius probability bundle above and verify both its own hash
   and `point_bundle_sha256` dependency.
3. Supply the 59 `asia_no_peak` features and GFS/GEFS/JMA-MSM inputs at Seoul's
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
   previously unseen Seoul market outcomes.

## Limitations

- The target is an exact conversion of Wunderground-derived Fahrenheit values,
  because no same-source raw Celsius column is stored. Future ingestion should
  persist the raw Wunderground Celsius settlement value directly.
- Open-tail allocation is data-limited: low-tail exact support is
  `-3` and high-tail exact support is `3, 4, 5`;
  multi-bucket tails can trigger ambiguity rejection.
- Only one 364-row forward-validation year is
  available for probability policy development.
- The 2026 holdout was previously inspected and is exploratory only.
- Market prices, liquidity, slippage, and ROI are outside this model audit.

## Artifact SHA-256

- `data\calibration\station_training_baseline\Seoul\celsius_market_probability\RKSI_2026_holdout_calibration.csv` — `9fd180cf96a17af33073b77c4735d05d3ae6c05b54654f272e0b1d625ca8192d`
- `data\calibration\station_training_baseline\Seoul\celsius_market_probability\RKSI_2026_holdout_metrics.csv` — `f2fe9ae5ecec74c767bc3fbd8ef441aa816a5abadb63da5f0d784d5891dea30a`
- `data\calibration\station_training_baseline\Seoul\celsius_market_probability\RKSI_2026_holdout_predictions.csv` — `cbebb18385f19bcabaaa7ccff84ad6b9a1508fe48a5ec64f4bac7ea050935b42`
- `data\calibration\station_training_baseline\Seoul\celsius_market_probability\RKSI_celsius_feature_contract.csv` — `0da3fd45cc2b6194f11fdac4786171bcab619b73ea0f65c5f79bc2f8fe0cd9fc`
- `data\calibration\station_training_baseline\Seoul\celsius_market_probability\RKSI_celsius_target_contract.json` — `169c85fa159777c511fd8437eaa23515e06cf1457039d49dce559e78c4d2831a`
- `data\calibration\station_training_baseline\Seoul\celsius_market_probability\RKSI_forward_validation_calibration.csv` — `546f5a0e4d806e5760305a348a235f6e0af9aae90eb04bb3ed98686203ea4011`
- `data\calibration\station_training_baseline\Seoul\celsius_market_probability\RKSI_forward_validation_metrics.csv` — `fe6f10db167306f2f238fa399746ad4af03e79f1ee486c24b59cb2c6312eaf09`
- `data\calibration\station_training_baseline\Seoul\celsius_market_probability\RKSI_forward_validation_predictions.csv` — `7458f1ada8a77bdb513b96ceb8472ec361d5e50bef29fda1a62559a6072425be`
- `data\calibration\station_training_baseline\Seoul\celsius_market_probability\RKSI_pre_2026_tuning.csv` — `e1970c1dfecccbf96f05b83a5b22c92d77986e280473d9f3dff3551ee24af0d7`
- `data\calibration\station_training_baseline\Seoul\celsius_market_probability\model_weights\RKSI_station_bucket_baseline_seoul_1c_market_ordinal.joblib` — `129bd4ce8e5d7c8280a83c71c597cf65bd12bcc290e00d7afa9665ad29335189`
- `data\calibration\station_training_baseline\Seoul\celsius_market_probability\model_weights\RKSI_station_bucket_baseline_seoul_1c_market_ordinal.json` — `1fa904086e6ce9b31ba18bd6feba3f5eb8966e7ffedc42eaeb37511360f92067`
- `data\calibration\station_training_baseline\Seoul\model_weights\RKSI_station_high_regressor_baseline_seoul_no_peak_stack.joblib` — `d01042e34d41d4e6f9e769c6dcb50dbe97cde2e10ffb640d50d06df43245581e`
- `data\calibration\station_training_baseline\Seoul\model_weights\RKSI_station_high_regressor_baseline_seoul_no_peak_stack.json` — `a9b77b78b31cba41f71c18f93d1ee308b910d7be90b0eea5e73c590ef840e561`
