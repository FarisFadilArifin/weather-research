# V20 Binary Floor-or-Ceil Classifier

`floor_ceil_classifier_v1.ipynb` implements the all-row binary task:

- `0`: use `floor(regression prediction)`;
- `1`: use `ceil(regression prediction)`.

There are no boundary-distance windows, abstentions, or tuned action thresholds. Every honest point
prediction receives one class at the fixed probability cutoff of `0.5`. KATL uses V20 peak-timing
features and KDAL uses V20 no-peak. Candidate hyperparameters are selected on a strictly earlier
inner window, Platt calibration uses a later prior window, and 2024/2025 are outer evaluations.

Regenerate and execute with:

```powershell
.\.venv\Scripts\python.exe notebooks\experiments\round_direction_classifier_v1\generate_notebook.py
python -m jupyter nbconvert `
  --to notebook `
  --execute `
  --inplace `
  --ExecutePreprocessor.timeout=900 `
  notebooks\experiments\round_direction_classifier_v1\floor_ceil_classifier_v1.ipynb
```

Research outputs are written below `data/calibration/station_round_direction_classifier`.
