# Weather Research Documentation

Project documentation is organized by purpose:

| Area | Entry point |
|---|---|
| Getting started | [Quickstart](getting-started/QUICKSTART.md) |
| Architecture and project assumptions | [Project context](architecture/PROJECT_CONTEXT.md) |
| Data and feature contracts | [Data documentation](data/) |
| Tokyo 11 AM live data alignment | [Forecast and METAR contract](data/TOKYO_11AM_LIVE_ALIGNMENT.md) |
| Modeling and calibration | [Modeling documentation](modeling/) |
| Notebook catalog | [Notebook index](notebooks/README.md) |
| Station training standard and SOP | [Station training](station-training/README.md) |
| Operations | [Operations documentation](operations/) |
| Historical handoffs | [Handoffs](handoffs/) |

The root `README.md` remains the repository entry point. Experiment-specific
documentation stays with its notebook under `notebooks/experiments/`, and
report-specific documentation stays under `reports/`.
