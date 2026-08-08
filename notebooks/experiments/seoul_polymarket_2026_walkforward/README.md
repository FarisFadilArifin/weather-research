# Seoul 2026 Polymarket Walk-Forward Backtest

This experiment joins the active `train_Seoul.ipynb` 2026 point and ordinal
predictions to resolved Seoul daily-high Polymarket events and public CLOB price
history at 11:15 Asia/Seoul.

It compares price caps, model-edge gates, confidence thresholds, provider-spread
limits, and combinations using one-month rolling walk-forward validation:
January selects February, February selects March, and so on. A
trade buys one YES outcome, holds to settlement, includes each market's own fee
schedule, and risks a fixed 4 USDC. The base execution model adds 1¢ to the last
price-history observation at or before the decision time.

Run from the repository root:

```powershell
python notebooks\experiments\seoul_polymarket_2026_walkforward\run_backtest.py
```

Use `--refresh-events` or `--refresh-prices` to replace the local public-API
cache. Generated reports are written to
`reports/seoul_polymarket_2026_walkforward/`.
