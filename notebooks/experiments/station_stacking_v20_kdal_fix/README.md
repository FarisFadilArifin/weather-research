# V20 KDAL NBM-Physics Fix

This isolated KDAL experiment removes HRRR temperature-curve inputs, retains NBM temperature timing plus HRRR solar/cloud/precipitation physics, and evaluates an OOF monthly residual correction.

The July 1, 2026 provisional run rejected the patch:

| Candidate | 2026 MAE (F) | Status |
|---|---:|---|
| V11 Settlement Fix | 1.300 | Keep |
| Original V20 | 1.338 | Do not promote |
| KDAL fix, uncalibrated ridge | 1.372 | Reject |
| KDAL fix, OOF calibrated ridge | 1.359 | Reject |

The calibrated patch was worse than V11 Fix by 0.059 F on 170 common dates and won only 69 of 170 days. Model export remains disabled. The result indicates that removing the HRRR temperature curve wholesale is not the correct KDAL fix.
