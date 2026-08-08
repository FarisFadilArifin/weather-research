# KDAL V20 1 PM No-Peak Experiment

This notebook is the 1 PM counterpart of `station_stacking_v20_kdal_no_peak`.

- timing mode: `same_day_1pm_live_safe`
- feature version: `v20_kdal_1pm_no_peak`
- target mode: remaining warmup after the observed high through 1 PM
- target source: Wunderground only
- training profile: V20 aligned
- peak-timing features: excluded
- output directory: `data/calibration/station_stacking_v20_kdal_1pm_no_peak`

The notebook requires the pull audit at
`data/calibration/station_stacking_v20_kdal_1pm_no_peak/audit/audit_result.json` to pass before model fitting.

Regenerate it with:

```powershell
.\.venv\Scripts\python.exe notebooks\experiments\station_stacking_v20_kdal_1pm_no_peak\generate_station_notebook.py
```
