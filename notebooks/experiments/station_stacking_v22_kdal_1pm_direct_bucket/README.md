# KDAL V22 1 PM Direct Bucket Challenger

V22 is a research-only two-stage direct bucket correction model layered on the immutable V20 KDAL 1 PM point model.

It predicts:

1. whether the point bucket is wrong; and
2. whether the actual bucket is lower, the same, or upper.

The selected override policy must demonstrate positive lift with minimum switch counts in both 2024 and 2025. The point bucket is physically floored by the observed high through 1 PM, and lower overrides below that floor are prohibited.

Regenerate the notebook:

```powershell
.\.venv\Scripts\python.exe notebooks\experiments\station_stacking_v22_kdal_1pm_direct_bucket\generate_notebook.py
```

Audit the completed artifacts:

```powershell
.\.venv\Scripts\python.exe scripts\audit_v22_kdal_1pm_direct_bucket.py
```

Do not promote the exported artifact unless `historical_acceptance.passed` is true.
