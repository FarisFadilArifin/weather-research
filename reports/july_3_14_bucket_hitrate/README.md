# KATL/KDAL bucket hit rate: July 3–14, 2026

The comparison uses the model's top point-prediction bucket after Polymarket half-up whole-degree rounding and the repository's 2°F bucket mapping (for example, 92–93°F).

| Station | Model | Hits | Hit rate | MAE | Bias (actual − prediction) |
| --- | --- | ---: | ---: | ---: | ---: |
| KATL | v11 settlement | 4/12 | 33.3% | 1.539°F | -0.097°F |
| KATL | v20 | 4/12 | 33.3% | 1.473°F | +0.073°F |
| KDAL | v11 settlement | 5/12 | 41.7% | 1.649°F | +1.076°F |
| KDAL | v11 settlement fix | 3/12 | 25.0% | 1.824°F | +1.130°F |
| KDAL | v20 | 4/12 | 33.3% | 2.191°F | +2.096°F |

## Interpretation

- KATL is a bucket tie. V11 uniquely hit July 6, while v20 uniquely hit July 13. V20's point MAE was 0.065°F lower.
- KDAL v11 settlement had the best bucket result. Its five hits were July 3, 4, 8, 9, and 10.
- KDAL v11 settlement fix hit July 8–10. Its three hits were a strict subset of v11's hits.
- KDAL v20 hit July 3–5 and July 10. It had one fewer hit than v11 and a materially larger cool bias.

## Data and reconstruction notes

- Settlement labels are exact Wunderground airport-station daily highs fetched for KATL and KDAL for all 12 dates.
- The 11 AM observation snapshots were backfilled from IEM ASOS history for all 24 station-days.
- Direct point-in-time GFS archive rows were recovered for all 24 station-days.
- Direct point-in-time HRRR rows were recovered for 15 station-days. For the remaining nine, the high and 11 AM temperature came from the separately archived v20 HRRR hourly peak curve using the same live-safe cycle/cutoff rule.
- NBM highs and 11 AM temperatures came from the archived v20 13Z NBM peak curve, except where an existing direct NBM row was already present.
- These are retrospective reconstructions using the production-refit model artifacts; they are not claims about values logged by a live bot on those dates.

See `detail.csv` for every prediction, rounded bucket, and hit/miss. `summary.csv` contains the aggregate metrics.
