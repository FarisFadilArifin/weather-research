# Asia V20 No-Peak Notebooks

These notebooks adapt the KDAL V20-aligned station-stacking workflow to the
existing Tokyo and Seoul 11 AM parquet data contract.

- `stacking_Tokyo_v20_no_peak.ipynb` trains station `RJTT`.
- `stacking_Seoul_v20_no_peak.ipynb` trains station `RKSI`.
- `generate_city_notebooks.py` regenerates both notebooks without executing cells.

The model uses prior-day 18Z GFS, GEFS ensemble summaries, JMA MSM previous-day
forecasts, live-safe 11 AM observations, and Wunderground-only highs. Values
are modeled in Fahrenheit and reported in Celsius as a secondary diagnostic.

Run the generator from the repository root:

```powershell
python notebooks\station_stacking_v20_asia_no_peak\generate_city_notebooks.py
```
