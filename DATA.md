# Data And Generated Artifacts

The repo is code-first. Most data and generated artifacts are local-only because they can be large, slow to regenerate, or tied to API/cache availability.

## Ignored Directories

These directories are intentionally not committed:

- `data/`: raw inputs, calibration caches, station-stacking outputs, model bundles.
- `logs/`: backfill and enrichment logs.
- `outputs/`: ad hoc tables, plots, screenshots, and report exports.
- `.venv/`: local Python environment.

## Expected Data Layout

The active station-stacking v2 workflow reads and writes under:

```text
data/
  calibration/
    sdk_current_obs_2021_2026/
      sdk_current_observations_11am.csv
    sdk_11am_gfs_*/
      sdk_nwp_0h_cache.csv
      sdk_station_registry.csv
    sdk_11am_hrrr_*/
      sdk_nwp_0h_cache.csv
      sdk_station_registry.csv
    station_stacking_v2/
      {STATION}_features.csv
      {STATION}_year_split_*csv
      model_weights/
```

Legacy calibration artifacts, if generated, also live in `data/calibration/`.

## Fresh Clone Expectations

A fresh clone can install dependencies, import the package, and run most unit tests. It cannot reproduce historical station results until the required local data caches are populated.

Use the commands in [README.md](README.md) to backfill current observations and HRRR/GFS rows before running station-stacking v2 notebooks.

## Sharing Results

For collaborators, prefer sharing compact derived artifacts instead of raw caches:

- CSV summaries from `data/calibration/station_stacking_v2/`.
- Model manifests and bundles from `data/calibration/station_stacking_v2/model_weights/`.
- Screenshot-ready tables or plots from `outputs/`.

If a generated artifact should become part of the public repo, move it to a tracked docs or examples folder and keep the file small.
