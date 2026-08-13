# KDAL Ordinal Challenger V1

This experiment improves Ordinal Probabilities Model 2. Its verified three-arm
training and export runner is invoked by the active KDAL Station Training
Baseline notebook so notebook and command-line runs share one implementation.

It searches:

- empirical/ordinal weights `0.25`, `0.50`, `0.75`, and `1.00`;
- the full 59-feature contract against 27- and 21-feature ablations;
- independent cumulative logits against a lower-variance shared-slope
  cumulative-logit model;
- candidate selection by market-bucket log loss, with RPS as the secondary
  ordered-distribution metric; and
- confidence-only policies whose thresholds are selected inside a nested
  chronological fold.

All candidates retain the V20 point-model bucket. Probability outputs may mark a
date as shadow-trade/no-trade, but they cannot override the point bucket.

Historical selection uses only pre-2026 chronological folds. The inspected 2026
period is diagnostic only. The frozen output contract selects exactly:

1. the best blended independent ordinal arm;
2. the best shared-slope ordinal arm; and
3. the best pure independent ordinal arm.

Frozen candidates require fresh shadow data beginning 2026-07-31 before
promotion can be considered.

Run:

```powershell
.\.venv\Scripts\python.exe scripts\run_kdal_ordinal_challenger_v1.py
```

Outputs are written to:

```text
data/calibration/station_training_baseline/KDAL/ordinal_challenger_v1/
```
