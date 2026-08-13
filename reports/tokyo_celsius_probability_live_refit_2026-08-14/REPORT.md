# Tokyo Celsius Probability Live-Point Refit — 2026-08-14

Status: **local shadow candidate; not production-approved; not deployed.**

## Artifact identity

- Serving point model: `station_high_regressor_live_tokyo_no_peak_stack_2026`
- Serving point SHA-256: `2a30f116c188e4199950911523cdbe4cdb680a0e9cb8361092e23fd374c07d70`
- Probability model: `station_bucket_live_tokyo_1c_market_ordinal_2026_r1`
- Probability SHA-256: `ac5c980c71cb7ab294a9f7aca3b28297cfd9d8cdbdae352c94026f66cb5c2b87`
- Probability manifest SHA-256: `079ab09dab39a03734b75016327940b6c4499dfd63392b8c1759aae54abbfa76`

This is a new fit, not a metadata-only adoption. Model and threshold selection use
honest 2024-2025 point-stack predictions. The resulting artifact binds directly to
the exact August 12 serving point bundle and uses that bundle to create the separate
2026 exploratory replay.

## Causal model-development evidence

The 2025 forward fold contains 365 predictions, each trained and calibrated only on
earlier data.

| Metric | Result |
|---|---:|
| Probability recommended-bucket accuracy | 50.68% |
| Point-bucket accuracy | 49.86% |
| Offset log loss | 1.2714 |
| Offset Brier score | 0.6297 |
| Decision coverage | 57.53% (210/365) |
| Decision accuracy | 60.48% |

## Exploratory 2026 serving-point replay

The replay contains 206 usable dates and excludes two dates with no required 11:00
observed high. The probability recommended bucket was correct on 71.36% of rows,
while the point bucket was correct on 82.04%. The frozen confidence policy selected
80 rows and was correct on 82.50% of them.

These numbers are not unseen validation: the serving point model was fitted through
2026-07-25. Its January-July predictions therefore contain in-sample information.
They may be used to test runtime compatibility and conduct an exploratory economic
replay, but not to approve production promotion.

## 2026 economic replay

Common assumptions: resolved Tokyo events from 2026-03-10 through 2026-07-25,
public one-minute Polymarket price history converted to a synthetic executable ask,
weather taker fees, flat $4 gross notional, and settlement hold. Historical order-book
depth and the 92c full-depth take-profit are unavailable and are not simulated.

| Strategy | Window | Cap | Entries | Wins | P&L | ROI | Max DD | Avg/month |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Current exact point bucket | 11:15-12:00 | 47c | 91 | 71 | +$719.78 | 197.74% | $12.38 | 18.2 |
| Probability recommended bucket, point probability >=25% | 11:15-12:00 | 50c | 85 | 56 | +$489.54 | 143.98% | $12.45 | 17.0 |
| Probability recommended bucket, point probability >=25% | 11:15 one-shot | 50c | 74 | 49 | +$477.89 | 161.45% | $8.46 | 14.8 |

The requested 11:15-12:00 probability rule was positive in every observed month:

| Month | Entries | Wins | P&L | ROI | Average entry |
|---|---:|---:|---:|---:|---:|
| March | 17 | 12 | +$203.25 | 298.89% | 30.4c |
| April | 18 | 14 | +$88.24 | 122.55% | 35.2c |
| May | 14 | 10 | +$65.10 | 116.24% | 33.7c |
| June | 17 | 10 | +$65.59 | 96.46% | 34.6c |
| July | 19 | 10 | +$67.37 | 88.64% | 33.9c |

Its P&L remains +$286.29 after removing the best month, +$306.81 after removing
the three largest wins, and +$247.92 after removing the five largest wins. The
47c-51c cap neighborhood stays profitable. Nevertheless, the current exact strategy
dominates it on trades, hit rate, P&L, and ROI in this same replay.

## Decision

Keep the new probability artifact in local shadow mode. Do not replace the current
exact-nearest route. Promotion requires genuinely fresh resolved outcomes after the
point model's 2026-07-25 training cutoff plus executable shadow fills; previously
inspected or in-sample 2026 results cannot satisfy that gate.
