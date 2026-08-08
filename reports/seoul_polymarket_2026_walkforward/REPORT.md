# Seoul 2026 Polymarket Walk-Forward Backtest

Status: **historical economic backtest using public Polymarket data; research only.**

The joint grid contains 25 filter families and
3,109 parameter combinations. Every
eligible non-price filter is crossed with entry-price caps from 20¢ through 80¢.

## Best filter

The least-bad eligible walk-forward family at the base 1¢ execution penalty was `probability_provider_spread_edge_price_cap`.
Across the six out-of-sample months it made 80 trades at
47.9% coverage, won 40.0%, and
earned +183.88 USDC using a fixed 4 USDC risk per trade. Mean
net return was 57.5%; its one-sided 95% lower confidence bound
was -3.0%, profit factor was
1.96, and maximum drawdown was 33.14 USDC.

**No candidate had a non-negative 95% lower confidence bound.** Therefore this 2026 sample does
not establish a robustly profitable filter. The highest realized P&L belonged to
`probability_provider_spread_edge_price_cap` at +183.88 USDC, but its lower bound was
-3.0%; it is not a validated promotion candidate.

The result is also concentrated: the two largest winners, 2026-02-04 (+86.91 USDC), 2026-02-14 (+51.56 USDC), contributed
+138.46 USDC, or 75.3% of the family's total P&L.
This tail dependence is another reason not to treat the realized headline return as robust.

The final July rule was selected using June 1-30 only:
`{"max_price":0.8,"max_provider_spread":6.0,"min_edge":0.0}`. In July 1-25 it made 13 trades,
earned -4.10 USDC, and returned -7.9% per
trade.

## Walk-forward results

| Test month | Training cutoff | Frozen parameters | Trades | Win rate | Net P&L | Mean return |
|---|---:|---|---:|---:|---:|---:|
| february | 2026-01-31 | `{"max_price":0.8,"max_provider_spread":5.0,"min_edge":0.0}` | 16 | 50.0% | +128.11 | 200.2% |
| march | 2026-02-28 | `{"max_price":0.8,"max_provider_spread":3.0,"min_edge":0.0}` | 10 | 40.0% | +6.05 | 15.1% |
| april | 2026-03-30 | `{"max_price":0.8,"max_provider_spread":8.0,"min_edge":0.0}` | 16 | 31.2% | +37.17 | 58.1% |
| may | 2026-04-30 | `{"max_price":0.8,"max_provider_spread":6.0,"min_edge":0.0}` | 12 | 41.7% | +5.53 | 11.5% |
| june | 2026-05-31 | `{"max_price":0.8,"max_provider_spread":6.0,"min_edge":0.0}` | 13 | 46.2% | +11.13 | 21.4% |
| july | 2026-06-30 | `{"max_price":0.8,"max_provider_spread":6.0,"min_edge":0.0}` | 13 | 30.8% | -4.10 | -7.9% |

Each filter's thresholds were selected on the immediately preceding calendar month only. January
selected February, February selected March, and so on through June selecting July. Candidate
selection maximized the one-sided 95% lower confidence bound of net return with minimum trade-count
and coverage constraints. February through July were disjoint forward test folds.

## Median frozen filter

Taking the median of the six monthly parameter selections gives
`{"max_price":0.8,"max_provider_spread":6.0,"min_edge":0.0}`. Freezing that one
filter and applying it uniformly to every January-July market produced
99 trades, 40.4% wins,
+258.54 USDC P&L, 65.3%
mean return, and 33.14 USDC maximum drawdown.

| Month | Trades | Win rate | Fixed $4 P&L | KDAL-style sized P&L | High-risk trades |
|---|---:|---:|---:|---:|---:|
| january | 15 | 46.7% | +76.37 | +113.57 | 11 |
| february | 17 | 47.1% | +124.11 | +184.53 | 13 |
| march | 15 | 33.3% | +0.34 | -11.66 | 6 |
| april | 14 | 35.7% | +45.17 | +62.06 | 6 |
| may | 12 | 41.7% | +5.53 | -0.47 | 3 |
| june | 13 | 46.2% | +11.13 | +1.13 | 5 |
| july | 13 | 30.8% | -4.10 | -2.54 | 6 |

Only January 1 through July 25 can be reported. August-December 2026 markets were not available as
resolved historical outcomes at the backtest cutoff and are intentionally not fabricated.

This is a hindsight diagnostic, not a clean walk-forward estimate: later monthly selections help
define the median filter that is then applied to earlier months.

### Price-cap sensitivity

Holding the median provider-spread and edge thresholds fixed, the following table varies the
entry-price cap from 20¢ through 80¢:

| Max entry price | Trades | Win rate | Net P&L | 95% LCB | Max DD |
|---:|---:|---:|---:|---:|---:|
| 20% | 26 | 19.2% | +164.44 | -30.7% | 28.00 |
| 25% | 41 | 22.0% | +176.37 | -16.6% | 32.00 |
| 30% | 54 | 24.1% | +181.10 | -12.3% | 43.61 |
| 35% | 67 | 28.4% | +204.66 | -2.3% | 44.00 |
| 40% | 74 | 31.1% | +221.51 | 3.2% | 36.00 |
| 45% | 84 | 34.5% | +238.29 | 7.4% | 40.00 |
| 50% | 90 | 35.6% | +239.93 | 7.1% | 40.00 |
| 55% | 94 | 38.3% | +254.44 | 10.7% | 33.14 |
| 60% | 96 | 38.5% | +253.34 | 10.1% | 33.61 |
| 70% | 97 | 39.2% | +255.57 | 10.6% | 33.61 |
| 80% | 99 | 40.4% | +258.54 | 11.1% | 33.14 |

The same trades were also evaluated with the existing KDAL-style probability sizing rule: $4 base
risk, increasing to $6 when model edge is at least 0.15. This produced
+346.61 USDC with 48.16 USDC
maximum drawdown across 50 high-risk trades.

## Baselines

- Buy every point-model bucket: 167 trades,
  +48.32 USDC P&L, 7.2% mean return.
- Buy every ordinal recommended bucket: 167 trades,
  +103.36 USDC P&L, 15.5% mean return.
- Existing notebook confidence policy: 85 trades,
  -17.37 USDC P&L, -5.1% mean return.

| Rank | Family | Trades | Coverage | Win rate | Net P&L | Mean return | 95% LCB | Max DD |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `probability_provider_spread_edge_price_cap` | 80 | 47.9% | 40.0% | +183.88 | 57.5% | -3.0% | 33.14 |
| 2 | `probability_provider_spread_edge` | 82 | 49.1% | 40.2% | +179.92 | 54.9% | -4.2% | 33.14 |
| 3 | `probability_confidence_spread_edge_price_cap` | 75 | 44.9% | 40.0% | +171.27 | 57.1% | -6.7% | 33.14 |
| 4 | `probability_confidence_edge_price_cap` | 104 | 62.3% | 33.7% | +130.29 | 31.3% | -16.6% | 44.00 |
| 5 | `probability_price_edge` | 105 | 62.9% | 33.3% | +126.29 | 30.1% | -17.5% | 44.00 |
| 6 | `probability_edge_price_cap` | 110 | 65.9% | 32.7% | +118.19 | 26.9% | -18.7% | 46.60 |
| 7 | `probability_edge` | 108 | 64.7% | 33.3% | +118.33 | 27.4% | -18.9% | 46.60 |
| 8 | `probability_price_cap` | 138 | 82.6% | 33.3% | +96.76 | 17.5% | -19.5% | 78.16 |
| 9 | `point_edge_price_cap` | 103 | 61.7% | 35.0% | +109.72 | 26.6% | -19.9% | 53.70 |
| 10 | `point_edge` | 109 | 65.3% | 36.7% | +102.47 | 23.5% | -20.5% | 60.95 |

## Polymarket pricing and settlement contract

- Markets: 200 Seoul daily-high events matched to RKSI model predictions from
  2026-01-01 through 2026-07-25.
- Entry time: 11:15 Asia/Seoul. The reference is the last public CLOB price-history point at or
  before entry, with median age 48 seconds.
- Execution: reference price plus 1¢, held to binary settlement, fixed 4 USDC risk per trade.
- Fees: each Gamma market's own `feesEnabled` and `feeSchedule` fields; 112 matched dates
  had fees enabled for the point action. Fee per share is
  `rate * (p * (1-p)) ** exponent`.
- Settlement: Gamma's resolved YES outcome. Notebook Wunderground settlement mapping mismatched
  17 events; those rows remain in the economic backtest because Gamma is the
  authoritative Polymarket settlement source.
- Historical CLOB price history is not a historical ask book. The added execution penalty is a
  conservative proxy, and the sensitivity table shows how the result moves at wider penalties.

## Execution sensitivity

| Execution penalty | Winner family | Trades | Net P&L | Mean return | 95% LCB |
|---:|---|---:|---:|---:|---:|
| 0¢ | `probability_provider_spread_edge_price_cap` | 83 | +243.73 | 73.4% | 2.2% |
| 1¢ | `probability_provider_spread_edge_price_cap` | 80 | +183.88 | 57.5% | -3.0% |
| 2¢ | `probability_provider_spread_edge` | 78 | +128.58 | 41.2% | -11.9% |
| 3¢ | `point_edge_price_cap` | 99 | +58.90 | 14.9% | -22.2% |

## Limitations

- Public price history does not reconstruct historical depth, executable ask size, queue position,
  partial fills, or rejected FOK orders.
- Results recycle a fixed 4 USDC risk and do not model overlapping capital or wallet limits.
- Multiple filter families were compared. Walk-forward testing reduces leakage but does not remove
  multiple-testing risk from a single January-July market regime.
- Treat the July fold as the cleanest evidence and keep any rule shadow-only until it survives new
  Seoul markets with captured bid/ask depth.
