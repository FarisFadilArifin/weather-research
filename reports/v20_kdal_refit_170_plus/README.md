# KDAL v20 no-peak: refit after day 170

The live model trained through June 21, 2026 (170 eligible 2026 rows) was
compared with an expanding refit on each subsequent eligible enriched row.
For every forecast date, the expanding arm used only earlier settled rows.

| Arm | Days | Bucket hits | Hit rate | MAE | RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen live model 170 | 18 | 10 | 55.6% | 1.191 deg F | 1.525 deg F |
| Daily expanding refit | 18 | 10 | 55.6% | 1.107 deg F | 1.539 deg F |

The expanding versions tested were model 170 through model 187. All 18
bucket outcomes were identical between the arms: the same ten dates hit and
the same eight dates missed. The observed bucket-hit gain was therefore zero.

Point accuracy was mixed. Daily refitting reduced MAE by 0.085 deg F but
increased RMSE by 0.015 deg F and increased actual-minus-prediction bias from
+0.074 deg F to +0.162 deg F.

The current enrichment file ends on July 29. Only 18 post-cutoff rows satisfy
the original v20 production modeling contract: July 1, July 2, July 12, and
July 15-29. Missing provider inputs prevent the intervening calendar dates
from becoming valid additional training rows.

See `detail.csv` for every model version, date, prediction, bucket, and hit.
`result.json` contains the paired comparison and runtime versions.
