# Seoul/Incheon 11 AM Live Feature Worker

## Scope

This contract defines the immutable live feature publisher for Incheon
International Airport (`RKSI`). It is station-scoped and does not replace or
modify the Tokyo/Haneda (`RJTT`) publisher or its archives.

## Runtime identity

- Station and city: `RKSI`, `seoul`
- Timezone: `Asia/Seoul` (UTC+9)
- Prediction unit: Celsius
- Feature version: `v20_asia_no_peak`
- Feature lineage: `station_stacking_v20_asia_no_peak`
- Providers: GFS, GEFS, and JMA MSM
- Observation population: RKSI IEM global METAR records at or before 11:00
  Asia/Seoul
- Collection window: no earlier than 11:10 Asia/Seoul

The worker calls the shared Asia acquisition and feature builder with only the
`seoul` profile. It publishes `RKSI_<contract-date>.json` plus a SHA-256
sidecar through an atomic `current` symlink.

## Fail-closed gates

Publication requires all of the following:

- alignment metadata identifies RKSI, `Asia/Seoul`, the requested local date,
  and `asia_same_day_11am_live_safe`;
- GFS and GEFS use the prior calendar day's 18Z cycle;
- JMA uses the `jma_msm_previous_day1` lineage;
- the selected IEM RKSI METAR is no later than 11:00 and no more than 60
  minutes old at the cutoff;
- every provider has non-empty source URLs and a valid SHA-256 provenance
  checksum;
- the v20 Asia provider-high, forecast-at-cutoff, current-observation,
  humidity, visibility, precipitation, and weather-code inputs are present;
- no settlement target is included in `featureInputs`;
- the worker archive manifest exactly enumerates every runtime file and every
  file matches its recorded size and checksum;
- `WEATHER_RESEARCH_SEOUL_WORKER_ARCHIVE_SHA256` pins the deployed archive.

An archive is built by `scripts/build_seoul_worker_archive.py`. The builder
rejects a dirty worktree by default and creates a deterministic archive with a
`.source-commit` marker and `WORKER-MANIFEST.json`. Deployment must extract
that archive without modifying its payload; the publisher independently checks
the manifest before emitting a live artifact.

The checked-in IEM population contract is
`config/seoul_iem_asos_observation_contract.json`.
