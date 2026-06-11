# Current Observation Feature Pull

This file defines the actual observed conditions to pull at the 11 AM forecast snapshot.

These are not forecast features. They are actual station observations from the deployment observation window around the 11:00 AM local forecast snapshot.

## Scope

- Stations: `KATL`, `KAUS`, `KORD`, `KDAL`, `KHOU`, `KLAX`, `KMIA`, `KLGA`, `KSEA`
- Observation snapshot: 11:00 AM local station time
- Observation selection rule for deployment: use the latest observation with timestamp from `10:50 AM local` through `11:10 AM local`
- Staleness tracking: always record the exact observation time and age in minutes

## Leakage Contract

- Observation timestamp must satisfy `10:50 AM local <= observed_at <= 11:10 AM local`.
- Do not use observations after 11:10 AM local.
- Do not use observations before 10:50 AM local for deployment inference.
- Do not use `actual_high_f` or any same-day final actual-derived summary as an input feature.
- Daily high target remains `actual_high_f`.
- These features are operationally valid only because they would have been known by the bot decision time after the 11:10 AM observation-window close.

## Deployment Timing Rule

The model does not require an observation timestamped exactly 11:00 AM. Use the latest station observation in this deployment window:

```text
10:50 AM local <= observed_at <= 11:10 AM local
```

Because METAR reports can arrive a few minutes after their actual observation timestamp, separate the observation window from the bot run:

```text
observation_window_start = 10:50 AM local
observation_window_end = 11:10 AM local
bot_decision_time = 11:10 AM local or 11:15 AM local
```

If `received_at` is available, also require:

```text
received_at <= bot_decision_time
```

If `received_at` is unavailable, run inference with the small delay and record `observed_as_of_time_local`, `observed_as_of_time_utc`, and either a signed `observed_as_of_age_minutes` or a clearer `observed_offset_minutes_from_11am`. Reject or mark the prediction unavailable if no valid observation exists inside the 10:50-11:10 window.

This widens the current-observation contract relative to the older `<= 11:00 AM` cache rule. Retrain the station-stacking v2 artifacts and re-export model weights with this same observation rule before relying on post-11:00 observations in production.

## Source Rule

Preferred source:

- Mostly Right SDK timestamped station observations.

Implemented SDK path:

- `mostlyright.weather.obs.obs()` is still useful for daily observation aggregates.
- For current-observation features, the project uses the SDK's raw observation fetch path so each row keeps an `observed_at` timestamp and raw METAR-derived fields.
- The cache is SDK-only and records `observed_data_source=mostlyright_sdk_raw_observations`.

Fallback source, only if approved:

- IEM/ASOS/METAR station observations.

Do not silently mix in outside observations unless the research direction explicitly allows it.

## Feature Columns

| Feature | Unit | Meaning | Priority |
|---|---:|---|---|
| `observed_temp_at_as_of_f` | deg F | Actual observed temperature at or before 11 AM local | required |
| `observed_high_temp_through_as_of_f` | deg F | Highest observed station temperature from local midnight through the selected observation as-of time | required |
| `observed_dewpoint_at_as_of_f` | deg F | Actual observed dew point at or before 11 AM local | required |
| `observed_humidity_at_as_of` | percent | Actual relative humidity at or before 11 AM local | required if available or derivable |
| `observed_wind_speed_at_as_of` | mph | Actual wind speed at or before 11 AM local | required |
| `observed_wind_direction_at_as_of` | degrees | Actual wind direction at or before 11 AM local | required |
| `observed_wind_gust_at_as_of` | mph | Actual gust at or before 11 AM local | optional, station-dependent |
| `observed_pressure_at_as_of` | hPa or inHg | Actual pressure at or before 11 AM local | useful |
| `observed_pressure_source` | text | Whether pressure came from sea-level pressure or altimeter conversion | useful |
| `observed_altimeter_inhg_at_as_of` | inHg | Raw altimeter setting at or before 11 AM local | useful |
| `observed_sea_level_pressure_mb_at_as_of` | mb/hPa | Raw sea-level pressure at or before 11 AM local | useful |
| `observed_visibility_at_as_of` | miles | Actual visibility at or before 11 AM local | useful |
| `observed_ceiling_at_as_of` | ft | Lowest broken/overcast ceiling at or before 11 AM local | useful |
| `observed_cloud_cover_at_as_of` | category or percent | Actual cloud cover derived from METAR cloud layers | useful |
| `observed_weather_code_at_as_of` | text/category | Present-weather code such as rain, fog, thunder, mist | useful |
| `observed_precip_recent_at_as_of` | inches or mm | Recent observed precipitation, if reported | optional, sparse |
| `observed_snow_depth_at_as_of` | inches | Snow depth at or before 11 AM local, if reported | optional, sparse |
| `observed_temp_change_last_1h_f` | deg F | Selected observation temperature minus the latest same-day temperature at or before one hour earlier | v6 |
| `observed_temp_change_last_3h_f` | deg F | Selected observation temperature minus the latest same-day temperature at or before three hours earlier | v6 |
| `observed_morning_warmup_rate_f_per_hour` | deg F/hour | Warming rate from the 9 AM local anchor to the selected observation time | v6 |
| `observed_high_so_far_change_since_9am_f` | deg F | Change in same-day high-so-far from the 9 AM local anchor to the selected observation time | v6 |
| `observed_as_of_time_local` | timestamp | Local timestamp of the observation used | required |
| `observed_as_of_time_utc` | timestamp | UTC timestamp of the observation used | required |
| `observed_as_of_age_minutes` | minutes | Signed minutes between 11 AM local and the observation used, if this legacy name is retained | required |
| `observed_source` | text | Observation source lineage | required |
| `observed_observation_type` | text | Raw observation type, usually METAR/SPECI where reported | useful |
| `observed_qc_field` | text | Raw SDK QC field, if provided | useful |
| `observed_raw_metar` | text | Raw METAR string used for the selected observation | useful for audit |
| `observed_data_source` | text | Project data-source lineage, currently `mostlyright_sdk_raw_observations` | required |
| `observed_fetch_status` | text | `ok` or `unavailable` | required |
| `observed_unavailable_reason` | text | Reason observation features could not be fetched | required when unavailable |

Feature classes for the bot:

| Field class | Columns | Type | Bot use |
|---|---|---|---|
| Live numeric observation features | `observed_temp_at_as_of_f`, `observed_high_temp_through_as_of_f`, temperature trends, dew point, humidity, wind, pressure, visibility, ceiling, cloud cover, recent precip, snow depth, age/offset | numeric | May be model inputs if present in the exported bundle's `feature_names` |
| Live categorical observation features | `observed_pressure_source`, `observed_weather_code_at_as_of`, `observed_observation_type`, derived precip-intensity labels | string/categorical | May be model inputs if present in the exported bundle's `feature_names` |
| Point-in-time lineage | `observed_as_of_time_local`, `observed_as_of_time_utc`, `observed_source`, `observed_raw_metar`, `observed_data_source`, `observed_fetch_status`, `observed_unavailable_reason` | timestamp/string | Required for validation and logging; not ordinary numeric model features |

`observed_high_temp_through_as_of_f` and the v6 trend features are point-in-time safe because they use only same-day observations at or before the selected observation time. They are not final daily actual-derived summaries. For live inference, the final prediction should normally be at least the high-so-far value because the day has already reached that temperature.

The v6 trend columns are model-input candidates, not guaranteed trained features. The station-stacking trainer only keeps numeric inputs with at least one non-null value in the fitting data; historical caches with all-null trend columns will omit those fields from `data/calibration/station_stacking_v6/{STATION}_feature_columns.csv`.

## Derived Observation Features

These can be computed after the raw current-observation fields exist:

| Feature | Meaning |
|---|---|
| `observed_dewpoint_depression_f` | `observed_temp_at_as_of_f - observed_dewpoint_at_as_of_f` |
| `observed_high_temp_minus_temp_at_as_of_f` | `observed_high_temp_through_as_of_f - observed_temp_at_as_of_f` |
| `observed_heat_index_at_as_of_f` | Heat index from observed temp and humidity, when warm enough |
| `observed_wind_chill_at_as_of_f` | Wind chill from observed temp and wind, when cold enough |
| `observed_wind_dir_sin` | Circular encoding of observed wind direction |
| `observed_wind_dir_cos` | Circular encoding of observed wind direction |
| `observed_is_raining_at_as_of` | Derived from present-weather code or recent precip |
| `observed_is_drizzle_at_as_of` | Derived from METAR drizzle codes such as `DZ`/`FZDZ` |
| `observed_is_snowing_at_as_of` | Derived from METAR snow codes such as `SN`/`SHSN` |
| `observed_is_fog_or_mist_at_as_of` | Derived from present-weather code and/or visibility |
| `observed_is_thunder_at_as_of` | Derived from present-weather code |
| `observed_precip_intensity_code` | `0=none`, `1=light`, `2=moderate`, `3=heavy`, derived from METAR intensity signs and recent precip |
| `observed_precip_intensity` | Text label for `observed_precip_intensity_code` |

## Planned Cache

Current cache file:

```text
data/calibration/sdk_current_obs_2021_2026/sdk_current_observations_11am.csv
```

Recommended key:

```text
station_id, contract_date, timing_mode
```

For this project, `timing_mode` should be:

```text
same_day_11am
```

Rows are checkpointed and resumable. Transient SDK/network failures are left pending for a future resume; true unavailable station/date rows are written as `unavailable` and can be retried intentionally with `--retry-unavailable`.

Recommended command:

```powershell
python -m src.backfill_mostlyright_current_observations --sdk-cache-dir data/calibration/sdk_current_obs_2021_2026 --stations $stations --start-date 2021-01-01 --end-date latest-complete --chunk-days 31 --retry-unavailable --request-retries 5 --retry-sleep-seconds 90 --sleep-between-chunks 8
```

## Modeling Use

These columns may be used as ML features because they are known by the post-window bot decision time. For station-stacking v2 deployment, they should be shared across the same station/date row used by HRRR and GFS.

They must not replace the target:

```text
calibration_bias_f = actual_high_f - raw_forecast_high_f
```

Calibration remains additive:

```text
calibrated_high_f = raw_forecast_high_f + predicted_calibration_bias_f
```
