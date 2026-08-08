# Directional Residual Audit V1

This audit tests fixed forecast-context subgroups for consistent residual direction and, separately,
for profitable alternative-bucket utility. Development selection uses 2023-2025 only; 2026 is an
untouched confirmation set. All significance tests receive Benjamini-Hochberg correction.

```powershell
python notebooks\experiments\directional_residual_audit_v1\generate_notebook.py
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 `
  notebooks\experiments\directional_residual_audit_v1\directional_residual_audit_v1.ipynb
```
