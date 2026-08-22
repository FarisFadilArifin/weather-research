# Tokyo 11 AM Forecast and METAR Live-Alignment Contract

## Purpose

This report defines the point-in-time data contract for Tokyo Haneda Airport
(`RJTT`) station-high inference. Its purpose is to make a live production row
use the same forecast versions and observation cutoff as the research data.

This contract applies to the active Tokyo `asia_no_peak` model using GFS, GEFS,
JMA MSM, and RJTT METAR features. It does not define the final settlement high,
which remains a post-event Wunderground label and must never enter live
inference.

## Production decision

The no-retraining production contract is:

1. Run collection no earlier than **11:10 JST / 02:10 UTC**.
2. Use the **previous calendar day 18Z** run for both GFS and GEFS.
3. Use Open-Meteo JMA MSM **`previous_day1`**, not the latest JMA run, as the
   model input.
4. Use the latest RJTT METAR whose observation timestamp is at or before
   **11:00 JST / 02:00 UTC** and no more than 60 minutes old.
5. Fail closed when a required source does not satisfy its version, timing, or
   completeness checks. Do not silently substitute a newer model run.

This preserves the trained feature distribution without requiring retraining.
The newest JMA forecast may be collected as shadow data, but it must not replace
`previous_day1` in the production model row.

## Clock and daily timeline

Tokyo uses Japan Standard Time (`Asia/Tokyo`), which is UTC+9 throughout the
year and has no daylight-saving transition.

| Event for contract date D | JST | UTC |
|---|---:|---:|
| GFS/GEFS initialization | D 03:00 | D-1 18:00 |
| Forecast and observation cutoff | D 11:00 | D 02:00 |
| Earliest production collection | D 11:10 | D 02:10 |
| End of modeled remaining-day window | D 23:00 | D 14:00 |

The ten-minute delay is an ingestion allowance, not permission to use
observations after 11:00. Every feature retains the 11:00 information boundary.

## Research-to-live source matrix

| Input | Research version | Required live version | Alignment status |
|---|---|---|---|
| GFS | NOAA AWS, D-1 18Z | NOAA AWS, exact D-1 18Z | Aligned when cycle is pinned |
| GEFS | NOAA AWS, D-1 18Z, `c00` + `p01`-`p30` | Same cycle and 31 members | Aligned when cycle and members are pinned |
| JMA MSM | Open-Meteo `*_previous_day1` | Open-Meteo `*_previous_day1` | Aligned under this report's contract |
| RJTT METAR | IEM global METAR archive | Prefer the same IEM record; AWC is allowed only after record-equivalence checks | Timestamp semantics aligned; distributor may differ |

The repository's current `run_live` implementation does **not** yet implement
the JMA row in this matrix: it writes `jma_msm_latest_at_collection`. Production
must replace that input with `jma_msm_previous_day1` or stop with an alignment
error. Feeding the latest JMA values to the existing trained model is a
forecast-vintage shift. The current METAR collector also uses AWC first without
cross-checking IEM, so strict distributor alignment requires the additional
equivalence gate specified below.

## Provider contracts

### GFS deterministic

- Model cycle: exactly **D-1 18:00 UTC** (`D 03:00 JST`).
- Required forecast hours: `f008` through `f020`, inclusive.
- Valid window: **D 02:00-14:00 UTC / D 11:00-23:00 JST**.
- Expected rows: 13 hourly rows.
- Repository lineage: `gfs_noaa_aws_previous_day_18z`.
- Point extraction: bilinear at `35.553, 139.781`.

The production fetcher must not choose the latest GFS run dynamically. NOAA
identifies GFS runs by cycle and forecast hour; operational cycles include 00,
06, 12, and 18 UTC. The trained Tokyo contract deliberately pins 18Z. See the
[NOAA NOMADS cycle and forecast-hour documentation](https://nomads.ncep.noaa.gov/info.php?page=opendap_grib_migration).

Required acceptance checks:

- `issued_at_utc == D-1T18:00:00Z` on every row;
- `forecast_as_of_utc == DT02:00:00Z`;
- exactly one row for each `f008`-`f020`;
- all required fields are finite and every row has `fetch_status == "ok"`;
- source URL identifies the expected date, 18Z cycle, and forecast hour.

### GEFS ensemble

- Model cycle: exactly **D-1 18:00 UTC** (`D 03:00 JST`).
- Members: control `c00` plus perturbed `p01`-`p30` (31 total).
- Two-metre temperature forecast hours: `f006`, `f009`, `f012`, `f015`,
  `f018`, and `f021`.
- Three-hour TMAX forecast hours: `f009`, `f012`, `f015`, `f018`, and `f021`.
- Expected raw rows: 31 x 6 = 186.
- Repository lineage: `gefs_noaa_aws_previous_day_18z`.

The five TMAX intervals end at 12:00, 15:00, 18:00, 21:00, and 00:00 JST.
Together they cover 09:00 JST through 00:00 JST the next day. This is slightly
wider than the nominal 11:00-23:00 window, but it is intentional and identical
in research and live production. The daily provider high is the maximum TMAX
per member; the model uses ensemble summaries including the median high.

NOAA documents GEFS cycles and member names (`c00`, `p01`-`p30`) in its
[GEFS product inventory](https://www.nco.ncep.noaa.gov/pmb/products/gens/).

Required acceptance checks:

- exact D-1 18Z cycle on every row;
- all 31 expected members are present exactly once at every required hour;
- all five TMAX intervals are present for every member;
- the ensemble summary reports 31 usable member highs;
- no fallback to a different cycle, reduced-member ensemble, or ensemble mean
  product.

### JMA MSM

#### Version used by research

Research requests Open-Meteo variables with the `*_previous_day1` suffix for
local hours 11:00-23:00 JST. Open-Meteo defines `previous_day1` as the value
predicted approximately 24 hours before each valid time. It is a fixed
lead-time series, not a named initialization cycle. See the
[Open-Meteo Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api).

Consequently, the version identity is:

```text
provider: jma_msm
lineage: jma_msm_previous_day1
availability_basis: open_meteo_previous_day1_variable
valid_local_hours: 11..23
```

`issued_at_utc` is unavailable and must remain null. Production must not invent
an initialization timestamp for this source.

#### Required live behavior without retraining

At or after 11:10 JST, request the same JMA MSM `*_previous_day1` fields for
contract date D and retain only local hours 11-23. Require exactly 13 rows and
the same variable set used in training.

The ordinary Open-Meteo JMA endpoint supplies the latest ingested model run.
JMA MSM updates every three hours, so that result is normally much fresher than
the research proxy. Open-Meteo documents the model cadence in its
[JMA API documentation](https://open-meteo.com/en/docs/jma-api). The newer data
may be more accurate, but it is not the trained version.

If latest JMA is also collected, store it separately with:

```text
lineage: jma_msm_latest_at_collection_shadow
retrieved_at_utc: <actual retrieval time>
model_input: false
```

Never merge shadow values into the production feature row. A future promotion
can compare paired `previous_day1` and latest-at-11:10 values before authorizing
retraining or a version change.

### RJTT METAR observation

#### Historical selection

Historical observations come from the IEM global ASOS/AWOS/METAR archive. For
each contract date, the pipeline selects the latest same-day RJTT observation
whose report timestamp is at or before 11:00 JST. It then builds current
conditions and the observed high-so-far using only records through that selected
timestamp.

The selected METAR is valid only when:

```text
observed_at_utc <= D 02:00 UTC
0 <= (D 02:00 UTC - observed_at_utc) <= 60 minutes
station_id == RJTT
observation_type == METAR
```

The IEM archive is an as-is collection with limited archive-side quality
control; its variables and source description are documented by
[Iowa Environmental Mesonet](https://mesonet.agron.iastate.edu/request/download.phtml?network=GN__ASOS).

#### Live selection

The current collector queries the Aviation Weather Center first and falls back
to IEM. AWC's current METAR cache is updated once per minute according to its
[Data API documentation](https://connect.aviationweather.gov/data/api/).

For strictest source alignment, production should prefer IEM because it is the
historical distributor. AWC may be used for operational resilience only when
normalization verifies the same underlying RJTT report. Distributor equivalence
requires:

- identical station and observation timestamp;
- identical normalized temperature, dew point, wind, pressure, visibility,
  weather, and cloud values used by the model;
- retention of the raw METAR text and a checksum;
- the same pre-11:00 and maximum-age rules.

If AWC and IEM disagree on a model feature, use the IEM record when available;
otherwise fail closed for strict inference and retain the disagreement for
audit. Never use a post-11:00 special report merely because collection occurs at
11:10.

## Required production row metadata

Store enough metadata to reproduce every input decision. At minimum:

| Field | Requirement |
|---|---|
| `station_id` | `RJTT` |
| `contract_date` | Tokyo local calendar date |
| `timezone` | `Asia/Tokyo` |
| `feature_cutoff_local` | `D 11:00:00+09:00` |
| `feature_cutoff_utc` | `D 02:00:00Z` |
| `collection_not_before_utc` | `D 02:10:00Z` |
| `gfs_cycle_utc` | `D-1 18:00:00Z` |
| `gefs_cycle_utc` | `D-1 18:00:00Z` |
| `jma_lineage` | `jma_msm_previous_day1` |
| `jma_availability_basis` | `open_meteo_previous_day1_variable` |
| `metar_observed_at_utc` | Selected report time, no later than 02:00Z |
| `metar_source` | `iem_asos_global_metar` or audited AWC equivalent |
| `retrieved_at_utc` | Actual retrieval time for every source |
| `source_url` | Exact upstream file or request URL |
| `source_checksum` | SHA-256 of the raw response or file subset |
| `timing_mode` | `asia_same_day_11am_live_safe` |
| `alignment_status` | `aligned` only after every gate passes |

The model bundle and prediction log should also record the point-model version,
feature version, ordered feature names, and bundle SHA-256.

## Fail-closed production gate

A row is eligible for model inference only if all of the following are true:

- current time is at or after 02:10 UTC on contract date D;
- GFS is complete and pinned to D-1 18Z;
- GEFS is complete with D-1 18Z and all 31 members;
- JMA has exactly 13 `previous_day1` rows for local hours 11-23;
- the selected RJTT METAR is pre-cutoff and at most 60 minutes old;
- provider highs and required current-observation values are finite;
- no source identity, station, timezone, unit, or forecast-hour mismatch exists;
- checksums and retrieval timestamps have been persisted.

On failure, return `predictionStatus = "unavailable"` with explicit reason codes,
for example:

```text
gfs_wrong_cycle
gefs_missing_member
jma_wrong_lineage
jma_incomplete_hours
metar_post_cutoff
metar_too_old
metar_distributor_disagreement
source_checksum_missing
```

Do not repair a failed row with a newer provider run, another airport, city
weather, reanalysis, or the final daily high.

## Deployment checklist

Before enabling Tokyo live predictions:

- [ ] Production explicitly requests JMA MSM `previous_day1`.
- [ ] `jma_msm_latest_at_collection` is shadow-only and isolated from features.
- [ ] GFS and GEFS selectors are fixed to D-1 18Z.
- [ ] GEFS requires all 31 members and all expected TMAX intervals.
- [ ] METAR selection uses report time, not API retrieval time.
- [ ] No observation after 11:00 JST can enter any feature or high-so-far value.
- [ ] METAR age is between 0 and 60 minutes at the 11:00 cutoff.
- [ ] Units match the builder contract before feature engineering.
- [ ] Raw payload checksums, normalized lineage, and timestamps are logged.
- [ ] A dry-run prediction demonstrates `alignment_status = "aligned"`.

## Repository implementation references

- Tokyo profile and cutoff constants: `src/asia_11am.py`
- GFS/GEFS cycle and valid-time construction: `src/asia_11am.py`
- JMA historical and live normalization: `src/asia_11am.py`
- METAR historical/live normalization: `src/asia_11am.py`
- Observation timestamp selection: `src/current_observations.py`
- Tokyo provider feature aggregation: `src/calibration/asia_station_stacking.py`
- Active station configuration:
  `notebooks/station_training_baseline/configs/RJTT.json`
