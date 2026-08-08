# Dallas and Tokyo Buy-NO Walk-Forward

This experiment economically backtests one NO trade per daily-high event. It sweeps low model-YES
probability thresholds, executable NO price caps, minimum NO edge, provider-spread gates,
confidence gates, open-tail handling, and bucket scope. Each test month uses parameters selected
only from the immediately preceding calendar month.

Run from the repository root:

```powershell
python notebooks\experiments\dallas_tokyo_no_walkforward\run_backtest.py --workers 16
```

Public Gamma events and CLOB price history are cached under
`data/polymarket/dallas_tokyo_no_walkforward/`. Reports are written to
`reports/dallas_tokyo_no_walkforward/`.
