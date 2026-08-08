# Notebook Experiments

This directory contains historical and research notebooks. Each experiment
keeps its notebook, generator, and local README/AUDIT files together.

These notebooks are evidence and comparison paths, not active station training
entry points. Active station workflows live under:

```text
notebooks/station_training_baseline/
```

To change an experiment notebook, edit its generator first when one exists and
regenerate the notebook without discarding unrelated saved outputs or metadata.
Accepted behavior should be integrated into the active baseline through the
station-training generator and config.

See the [notebook catalog](../../docs/notebooks/README.md) and the
[active station-training workflow](../../docs/station-training/README.md).
