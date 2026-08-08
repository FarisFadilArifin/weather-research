# Dallas and Tokyo 2026 Polymarket Buy-NO Backtest

Status: **historical economic research using public Polymarket NO-token price history; not a live strategy.**

The sweep tests 23,520 combinations per station: maximum model YES probability
(5%-40%), NO price caps (55%-95%), minimum model NO edge (0%-20%), provider-spread caps,
confidence-gate on/off, open-tail exclusion on/off, and bucket scope. At most one NO trade is chosen
per event. Entry is 11:15 local using the last public CLOB price at or before entry plus 1¢,
market-specific fees, and settlement hold.

## Dallas

### Clean monthly walk-forward

| Test month | Selected parameters from prior month | Trades | Win rate | Fixed $4 P&L |
|---|---|---:|---:|---:|
| 2026-02 | `{"bucket_scope":"all","exclude_open_tail":true,"max_no_price":0.9,"max_provider_spread_f":8.0,"max_yes_probability":0.2,"min_no_edge":0.05,"require_confidence_gate":true}` | 4 | 100.0% | +3.84 |
| 2026-03 | `{"bucket_scope":"all","exclude_open_tail":true,"max_no_price":0.9,"max_provider_spread_f":100.0,"max_yes_probability":0.25,"min_no_edge":0.05,"require_confidence_gate":false}` | 19 | 84.2% | +22.59 |
| 2026-04 | `{"bucket_scope":"all","exclude_open_tail":true,"max_no_price":0.95,"max_provider_spread_f":8.0,"max_yes_probability":0.4,"min_no_edge":0.2,"require_confidence_gate":false}` | 2 | 100.0% | +9.05 |
| 2026-05 | `{"bucket_scope":"all","exclude_open_tail":false,"max_no_price":0.95,"max_provider_spread_f":8.0,"max_yes_probability":0.4,"min_no_edge":0.1,"require_confidence_gate":false}` | 7 | 85.7% | +8.94 |
| 2026-06 | `{"bucket_scope":"all","exclude_open_tail":true,"max_no_price":0.9,"max_provider_spread_f":100.0,"max_yes_probability":0.4,"min_no_edge":0.0,"require_confidence_gate":false}` | 16 | 75.0% | +3.16 |
| 2026-07 | `{"bucket_scope":"all","exclude_open_tail":true,"max_no_price":0.95,"max_provider_spread_f":100.0,"max_yes_probability":0.3,"min_no_edge":0.075,"require_confidence_gate":false}` | 11 | 81.8% | +7.01 |

Combined: 59 trades, 83.1% win rate,
+54.59 USDC fixed-$4 P&L, 9.6% one-sided
95% mean-return lower bound, and 14.04 USDC maximum drawdown.

### Frozen median rule

`{"bucket_scope":"all","exclude_open_tail":true,"max_no_price":0.9,"max_provider_spread_f":8.0,"max_yes_probability":0.3,"min_no_edge":0.05,"require_confidence_gate":false}`

| Month | Trades | Win rate | Fixed $4 P&L | $4/$6 sized P&L |
|---|---:|---:|---:|---:|
| 2026-01 | 21 | 81.0% | +33.31 | +49.09 |
| 2026-02 | 7 | 85.7% | +2.84 | +1.94 |
| 2026-03 | 15 | 86.7% | +23.64 | +36.46 |
| 2026-04 | 9 | 100.0% | +16.11 | +21.87 |
| 2026-05 | 8 | 87.5% | +9.61 | +13.56 |
| 2026-06 | 13 | 76.9% | +3.89 | +8.98 |
| 2026-07 | 11 | 81.8% | +7.01 | +8.83 |

Combined: 84 trades, 84.5% win rate,
+96.41 USDC fixed-$4 P&L and +140.73 USDC
under $4/$6 edge sizing. This frozen-rule table is a hindsight diagnostic; the clean
walk-forward table above is the robustness estimate.

Holding the other frozen parameters fixed, the low-probability threshold sensitivity is:

| Maximum model YES probability | Trades | Win rate | Fixed $4 P&L | 95% LCB |
|---:|---:|---:|---:|---:|
| 5% | 23 | 95.7% | +28.78 | 15.4% |
| 10% | 54 | 90.7% | +45.33 | 10.4% |
| 15% | 68 | 86.8% | +57.83 | 10.1% |
| 20% | 74 | 85.1% | +58.34 | 8.7% |
| 25% | 77 | 84.4% | +71.69 | 11.5% |
| 30% | 84 | 84.5% | +96.41 | 16.5% |
| 35% | 85 | 82.4% | +89.68 | 13.8% |
| 40% | 89 | 80.9% | +90.13 | 12.6% |

## Tokyo

### Clean monthly walk-forward

| Test month | Selected parameters from prior month | Trades | Win rate | Fixed $4 P&L |
|---|---|---:|---:|---:|
| 2026-04 | `{"bucket_scope":"all","exclude_open_tail":true,"max_no_price":0.95,"max_provider_spread_f":8.0,"max_yes_probability":0.05,"min_no_edge":0.05,"require_confidence_gate":false}` | 5 | 60.0% | -2.48 |
| 2026-05 | `{"bucket_scope":"all","exclude_open_tail":true,"max_no_price":0.95,"max_provider_spread_f":8.0,"max_yes_probability":0.4,"min_no_edge":0.025,"require_confidence_gate":true}` | 9 | 66.7% | +7.91 |
| 2026-06 | `{"bucket_scope":"all","exclude_open_tail":false,"max_no_price":0.95,"max_provider_spread_f":8.0,"max_yes_probability":0.1,"min_no_edge":0.075,"require_confidence_gate":false}` | 2 | 50.0% | -2.04 |
| 2026-07 | `{"bucket_scope":"all","exclude_open_tail":true,"max_no_price":0.95,"max_provider_spread_f":100.0,"max_yes_probability":0.05,"min_no_edge":0.025,"require_confidence_gate":false}` | 12 | 91.7% | +2.95 |

Combined: 28 trades, 75.0% win rate,
+6.34 USDC fixed-$4 P&L, -17.6% one-sided
95% mean-return lower bound, and 12.00 USDC maximum drawdown.

### Frozen median rule

`{"bucket_scope":"all","exclude_open_tail":true,"max_no_price":0.95,"max_provider_spread_f":8.0,"max_yes_probability":0.05,"min_no_edge":0.025,"require_confidence_gate":false}`

| Month | Trades | Win rate | Fixed $4 P&L | $4/$6 sized P&L |
|---|---:|---:|---:|---:|
| 2026-03 | 9 | 100.0% | +8.48 | +11.87 |
| 2026-04 | 7 | 71.4% | -1.93 | -1.33 |
| 2026-05 | 7 | 100.0% | +4.20 | +4.67 |
| 2026-06 | 7 | 100.0% | +3.54 | +4.52 |
| 2026-07 | 8 | 87.5% | -0.05 | +0.35 |

Combined: 38 trades, 92.1% win rate,
+14.24 USDC fixed-$4 P&L and +20.09 USDC
under $4/$6 edge sizing. This frozen-rule table is a hindsight diagnostic; the clean
walk-forward table above is the robustness estimate.

Holding the other frozen parameters fixed, the low-probability threshold sensitivity is:

| Maximum model YES probability | Trades | Win rate | Fixed $4 P&L | 95% LCB |
|---:|---:|---:|---:|---:|
| 5% | 38 | 92.1% | +14.24 | -0.2% |
| 10% | 43 | 86.0% | +9.27 | -6.1% |
| 15% | 53 | 81.1% | +10.36 | -8.0% |
| 20% | 63 | 76.2% | +27.00 | -7.0% |
| 25% | 69 | 72.5% | +43.61 | -2.6% |
| 30% | 81 | 71.6% | +64.41 | 2.8% |
| 35% | 82 | 69.5% | +58.65 | 0.6% |
| 40% | 83 | 68.7% | +54.65 | -0.8% |


## Limitations

- Public price history is not historical executable ask depth; the 1¢ penalty is only a proxy.
- January has no earlier 2026 month and therefore appears only in the hindsight frozen-rule table.
- Results cover only dates for which the current research prediction artifacts and resolved markets overlap.
- Thousands of parameter combinations create multiple-testing risk. Do not promote without fresh shadow evidence.
