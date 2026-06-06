# Forecast Feature Pull

This file defines the forecast-side features for the 11 AM calibration dataset.

## Scope

- Stations: `KATL`, `KAUS`, `KORD`, `KDAL`, `KHOU`, `KLAX`, `KMIA`, `KLGA`, `KSEA`
- Active deployment providers/models:
  - HRRR via Mostly Right `forecast_nwp(model="hrrr")`
  - GFS via Mostly Right `forecast_nwp(model="gfs")`
  - NBM is currently excluded from the main calibration dataset because SDK NBM has many unavailable historical station/date rows.
- Timing mode: `same_day_11am`
- Forecast snapshot: 11:00 AM local station time
- Forecast window: valid times `>= 11:00 AM local` and `< next local midnight`
- Target high forecast: max forecast temperature over the forecast window

## Leakage Contract

- Forecast features must come only from the selected forecast cycle and valid times available at the 11 AM snapshot.
- No observed actual high, post-11 AM observation, or same-day final actual-derived value can be used as a forecast feature.
- HRRR and GFS lineage must remain separate for the v2 deployment model.
- Wide station-stacking experiment notebooks may keep provider-prefixed NBM columns for research, but the exported v2 deployment model uses GFS and HRRR only.
- NBM must not be labeled as SDK NBM in the final dataset when it comes from direct NOAA archives.

## Feature Columns

In long calibration samples, these are stored without a provider prefix.

In wide station notebooks, they become provider-prefixed columns:

- `gfs_<feature>`
- `hrrr_<feature>`
- `nbm_<feature>`

The station-stacking v2 deployment bundles use GFS and HRRR features only. NBM-prefixed columns can exist in research artifacts but should not be required by `station_high_regressor_v2`.

Exception: `raw_forecast_high_f` becomes:

- `gfs_high_f`
- `hrrr_high_f`
- `nbm_high_f`

| Feature | Unit | Meaning | HRRR SDK | GFS SDK | Active use |
|---|---:|---|---|---|---|
| `raw_forecast_high_f` | deg F | Forecast high from 11 AM through end of local day | yes | yes | yes |
| `dewpoint_mean_f` | deg F | Mean forecast dew point over remaining day | yes | yes | yes |
| `humidity_mean` | percent | Mean forecast relative humidity over remaining day | yes | yes | yes |
| `wind_speed_mean` | mph | Mean forecast wind speed over remaining day | yes | yes | yes |
| `wind_speed_max` | mph | Max forecast wind speed over remaining day | yes | yes | yes |
| `wind_direction_mean` | degrees | Mean forecast wind direction over remaining day | partial | near-complete | no |
| `wind_gust_max` | mph | Max forecast wind gust over remaining day | partial | near-complete | no |
| `precip_amount` | mm | Total forecast precipitation over remaining day | partial | near-complete | no |

Experimental cache-only fields, not used by `calibration_samples.csv`, ML training, or station-stacking until coverage is proven across all three providers:

| Feature | Unit | Meaning | Current status |
|---|---:|---|---|
| `pressure_mslp_mean` | Pa | Mean sea-level pressure over remaining day | HRRR/GFS partial; unavailable in current NBM archive probe |
| `pressure_surface_mean` | Pa | Mean surface pressure over remaining day | HRRR/GFS partial; unavailable in current NBM archive probe |
| `cloud_cover_mean` | percent | Mean forecast cloud cover over remaining day | SDK HRRR/GFS blank; direct NBM archive supports inventory but needs coverage proof |
| `cloud_cover_max` | percent | Max forecast cloud cover over remaining day | SDK HRRR/GFS blank; direct NBM archive supports inventory but needs coverage proof |
| `visibility_mean` | m or provider-native | Mean forecast visibility over remaining day | SDK HRRR/GFS blank; direct NBM archive supports inventory but needs coverage proof |
| `ceiling_min` | m or provider-native | Minimum forecast ceiling over remaining day | SDK HRRR/GFS blank; direct NBM archive supports inventory but needs coverage proof |

Forecast snapshot fields at exactly 11 AM, such as `forecast_temp_at_as_of_f`, `dewpoint_at_as_of_f`, `humidity_at_as_of`, `wind_speed_at_as_of`, and `wind_direction_at_as_of`, may still exist in raw caches from older pulls. They are intentionally excluded from `calibration_samples.csv`, ML features, and provider-wide station-stacking outputs.

Direct NOAA NBM core rows may initially contain only `raw_forecast_high_f`. Re-run the same shard with `--include-weather-features` after the core NBM rows finish; the command revisits only rows without `weather_features_included=true`.

Mostly Right HRRR/GFS rows can also be revisited with `--include-weather-features`; this skips unavailable rows and only re-fetches OK rows that predate the feature checkpoint flag.

SDK HRRR/GFS enrichment commands:

```powershell
python -m src.backfill_mostlyright_sdk_nwp --sdk-cache-dir data/calibration/sdk_11am_hrrr_2021_2022 --stations $stations --models hrrr --timing-mode same_day_11am --start-date 2021-01-01 --end-date 2022-12-31 --include-weather-features --fxx-workers 3
python -m src.backfill_mostlyright_sdk_nwp --sdk-cache-dir data/calibration/sdk_11am_hrrr_2023_2024 --stations $stations --models hrrr --timing-mode same_day_11am --start-date 2023-01-01 --end-date 2024-12-31 --include-weather-features --fxx-workers 3
python -m src.backfill_mostlyright_sdk_nwp --sdk-cache-dir data/calibration/sdk_11am_hrrr_2025_2026 --stations $stations --models hrrr --timing-mode same_day_11am --start-date 2025-01-01 --end-date latest-complete --include-weather-features --fxx-workers 3

python -m src.backfill_mostlyright_sdk_nwp --sdk-cache-dir data/calibration/sdk_11am_gfs_2021_2022 --stations $stations --models gfs --timing-mode same_day_11am --start-date 2021-01-01 --end-date 2022-12-31 --include-weather-features --fxx-workers 3
python -m src.backfill_mostlyright_sdk_nwp --sdk-cache-dir data/calibration/sdk_11am_gfs_2023_2024 --stations $stations --models gfs --timing-mode same_day_11am --start-date 2023-01-01 --end-date 2024-12-31 --include-weather-features --fxx-workers 3
python -m src.backfill_mostlyright_sdk_nwp --sdk-cache-dir data/calibration/sdk_11am_gfs_2025_2026 --stations $stations --models gfs --timing-mode same_day_11am --start-date 2025-01-01 --end-date latest-complete --include-weather-features --fxx-workers 3
```

## Metadata Columns

These columns document forecast lineage and timing:

| Column | Meaning |
|---|---|
| `provider` | `hrrr`, `gfs`, or `nbm` |
| `model` | Model name, normally same as provider |
| `source_label` | Human-readable source and timing label |
| `timing_mode` | `same_day_11am` for this dataset |
| `cycle_selection_policy` | Rule used to select the model cycle |
| `forecast_as_of` | Forecast snapshot timestamp |
| `issued_at` | Selected model cycle timestamp |
| `forecast_window_start` | Start of valid-time window used for the high |
| `forecast_window_end` | End of valid-time window used for the high |
| `forecast_hour_min` | First forecast hour used |
| `forecast_hour_max` | Last forecast hour used |
| `forecast_hour_count_requested` | Number of forecast hours requested from the provider path |
| `forecast_hour_count_returned` | Number of requested forecast hours returned after filtering |
| `forecast_hour_missing` | Comma-separated missing forecast hours, when any per-hour SDK request failed |
| `forecast_hour_completeness` | Returned/requested forecast-hour ratio |
| `forecast_hour_fetch_status` | `ok`, `partial`, or `unavailable` at the forecast-hour level |
| `grid_dist_km_mean` | Mean station-to-grid distance, if provided |
| `data_source` | Source lineage |
| `source_file_or_url` | Source path/URL/SDK call description |
| `fetch_status` | `ok` or `unavailable` |
| `unavailable_reason` | Reason a row could not be fetched |

## Current NBM Decision

NBM is excluded from the active main calibration dataset for now. SDK NBM finished date-wise, but produced many unavailable station/date rows, and the SDK NBM path does not provide the same wind speed/direction feature set as HRRR/GFS.

Direct NOAA NBM remains available for experiments. A direct-vs-SDK overlap check found identical cycle/window selection and nearly identical daily forecast highs:

- Overlap checked: `3,826` OK station/date rows.
- Cycle/window fields matched exactly.
- Mean absolute high difference: `0.008°F`.
- Median high difference: `0.000°F`.
- Max high difference: `1.08°F`.

The remaining direct NOAA NBM cache is resumed from checkpointed shard directories:

```powershell
python -m src.backfill_direct_nbm --sdk-cache-dir data/calibration/direct_nbm_2021_2022 --stations $stations --start-date 2021-01-01 --end-date 2022-12-31
python -m src.backfill_direct_nbm --sdk-cache-dir data/calibration/direct_nbm_2023_2024 --stations $stations --start-date 2023-01-01 --end-date 2024-12-31
python -m src.backfill_direct_nbm --sdk-cache-dir data/calibration/direct_nbm_2025_2026 --stations $stations --start-date 2025-01-01 --end-date latest-complete
```

Each shard writes `direct_nbm_0h_cache.csv` and resumes by skipping completed `(station_id, contract_date, provider, model, timing_mode)` keys.

Feature enrichment commands:

```powershell
python -m src.backfill_direct_nbm --sdk-cache-dir data/calibration/direct_nbm_2021_2022 --stations $stations --start-date 2021-01-01 --end-date 2022-12-31 --include-weather-features
python -m src.backfill_direct_nbm --sdk-cache-dir data/calibration/direct_nbm_2023_2024 --stations $stations --start-date 2023-01-01 --end-date 2024-12-31 --include-weather-features
python -m src.backfill_direct_nbm --sdk-cache-dir data/calibration/direct_nbm_2025_2026 --stations $stations --start-date 2025-01-01 --end-date latest-complete --include-weather-features
```

Do not run the core and enrichment commands against the same shard at the same time because both update the same CSV checkpoint.
