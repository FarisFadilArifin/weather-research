# KDAL V27 V20 No-Peak Ordinal Distribution

V27 is an isolated probability challenger layered on the immutable KDAL V20
same-day 11 AM no-peak point model.

Contract:

- point model: `station_high_regressor_v20_kdal_no_peak_stack`;
- feature profile: `common_no_peak`;
- target: ordered rounded-degree residual offset;
- classes: `<=-4, -3, -2, -1, 0, +1, +2, +3, >=+4`;
- primary arm: pure cumulative-threshold ordinal logistic distribution;
- comparison arm: the same ordinal family with an empirically calibrated blend;
- development: strict forward 2024 and 2025 folds;
- 2026: exploratory because it has already been inspected;
- promotion: prohibited until fresh shadow data confirms the frozen artifact.

Run the complete experiment and audit:

```powershell
.\.venv\Scripts\python.exe scripts\run_v27_kdal_no_peak_ordinal.py
```

Regenerate the notebook:

```powershell
.\.venv\Scripts\python.exe notebooks\experiments\station_stacking_v27_kdal_no_peak_ordinal\generate_notebook.py
```
