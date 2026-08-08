# Half-Up Override Classifier V3

V3 keeps nearest half-up rounding by default and predicts whether switching to the opposite
floor/ceil bucket recovers settlement. All rows receive a binary label; same-bucket floor/ceil rows
are non-actionable class `0`. Training, calibration, threshold selection, and outer validation use
separate chronological windows.

```powershell
.\.venv\Scripts\python.exe notebooks\experiments\round_override_classifier_v3\generate_notebook.py
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 `
  notebooks\experiments\round_override_classifier_v3\half_up_override_classifier_v3.ipynb
```
