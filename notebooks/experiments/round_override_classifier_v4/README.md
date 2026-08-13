# Cost-Sensitive Half-Up Override V4

V4 fits separate recovery and damage probability models using only actionable floor/ceil rows.
It combines them through a damage-penalized expected-utility rule and enables overrides only after
the rule passes three earlier chronological policy folds. Continuous bucket-probability features
use shifted rolling residuals, so the current row's outcome is never used as an input.

```powershell
python notebooks\experiments\round_override_classifier_v4\generate_notebook.py
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 `
  notebooks\experiments\round_override_classifier_v4\half_up_utility_override_v4.ipynb
```
