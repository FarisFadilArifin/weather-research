# Weather Market Calibration Guide

This guide explains how agents should calibrate NWS/NBM high-temperature forecasts for the station-calendar backtest:

- Stations: KDAL, KATL, KAUS, KMIA, KSEA
- Provider/model: NWS / NBM
- Trading/entry timing: around 5 AM local station time
- Calibration table: `data/outputs/day_of_morning_monthly_calibration_by_station.csv`

## Core Rule

Always calibrate by station and month.

Do not use city-center weather. Do not mix stations. Use the exact airport station:

- KDAL: Dallas Love Field
- KATL: Atlanta/Hartsfield-Jackson Intl
- KAUS: Austin/Bergstrom Intl
- KMIA: Miami Intl
- KSEA: Seattle-Tacoma Intl

For a 5 AM local entry, use the `0h` / day-of morning calibration.

Formula:

```text
calibrated_high_f = nbm_forecast_high_f + calibration_add_f
```

Where:

```text
calibration_add_f = historical mean error for station + month + 0h horizon
error_f = actual_high_f - forecast_high_f
```

Positive calibration means actual highs historically ran warmer than NBM.

Negative calibration means actual highs historically ran cooler than NBM.

## Day-Of Morning Calibration Table

Use this table for quick manual calibration. Values are degrees Fahrenheit to add to the raw NBM forecast high.

```text
Month   KATL   KAUS   KDAL   KMIA   KSEA
Jan    +1.72  -1.07  +1.37  +1.05  +1.24
Feb    +0.90  -0.30  +0.69  +0.95  +0.78
Mar    +1.43  -1.02  +0.38  +1.19  +1.10
Apr    -0.24  -1.96  +0.01  +1.45  +0.75
May    -0.24  -1.53  +0.01  +1.20  +0.00
Jun    -0.76  -2.54  +0.06  +0.60  +0.03
Jul    +0.67  -2.32  +0.61  +1.30  +0.50
Aug    -0.17  -1.14  +0.75  +0.42  +0.09
Sep    +0.64  -1.67  +0.78  +0.89  +0.34
Oct    +0.60  -0.79  +1.07  +0.94  +0.44
Nov    +1.24  -1.27  +1.12  +0.65  +1.59
Dec    +2.22  -0.83  +1.43  +0.61  +1.43
```

## Agent Procedure

1. Identify the station.

Use station code, not city text. If a market title says Dallas but resolution is KDAL, use KDAL.

2. Identify the local target date.

Use the airport local timezone. Month is based on the target date at the airport, not UTC.

3. Pull the raw NBM forecast high available around 5 AM local time.

This should correspond to the project's `0h` / day-of morning window. Do not use forecasts issued after the intended decision time.

4. Look up `calibration_add_f`.

Use:

```text
data/outputs/day_of_morning_monthly_calibration_by_station.csv
```

Filter:

```text
station_code == target station
month == local target month
```

5. Calculate calibrated high.

```text
calibrated_high_f = nbm_forecast_high_f + calibration_add_f
```

6. Estimate uncertainty.

Use `error_std_f` or MAE from the same row in the calibration CSV. Avoid treating the calibrated forecast as exact.

7. Convert to bucket probabilities.

Use `src.bucket_probs.bucket_probabilities` with:

```text
forecast_high_f = nbm_forecast_high_f
error_mean_f = calibration_add_f
error_std_f = error_std_f
buckets = market bucket labels
```

8. Apply a no-trade filter.

Avoid or reduce confidence when:

- The calibrated high is close to a bucket boundary.
- `error_std_f` is high.
- The station is KAUS, especially spring/summer, unless the edge is very large.
- The month historically has high MAE, especially January, February, November, December, or weak months in the station-specific table.
- The current weather setup is unusual versus the historical sample.

## Worked Examples

### KAUS In July

Raw NBM high:

```text
96.0F
```

July KAUS calibration:

```text
-2.32F
```

Calibrated high:

```text
96.0 + (-2.32) = 93.68F
```

Interpretation: KAUS summer NBM historically runs too warm, so pull the forecast down.

### KMIA In July

Raw NBM high:

```text
91.0F
```

July KMIA calibration:

```text
+1.30F
```

Calibrated high:

```text
91.0 + 1.30 = 92.30F
```

Interpretation: KMIA NBM historically runs cool in July, so push the forecast up.

## Station Notes

KMIA is the most predictable station in this backtest. Its day-of morning MAE is about 1.96F.

KSEA and KDAL are usable research candidates, with moderate error.

KATL is usable but noisier than KDAL/KSEA.

KAUS is the highest-risk station. It has the highest MAE and error volatility, and the calibration is strongly negative in many months.

## Important Guardrails

This guide estimates probabilities; it is not an automatic trading system.

Do not calibrate from future data. When doing historical simulation, only use calibration values that would have been known before the simulated trade date. The current table is valid as a research summary over the full 2022-2026 dataset, but strict walk-forward testing should recompute calibration using only prior dates.

Do not use active-market actuals before the local day is complete and observations are finalized.

Do not assume Polymarket city names imply stations. Use resolution station mapping only.

Do not use Open-Meteo here for the NWS/NBM calibration workflow.

