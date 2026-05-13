# DataKit Data Sources

This module contains all data source implementations for loading various types of experimental data. Data sources are organized by category and support versioning for backward compatibility and feature evolution.

## Organization

The data sources are organized into the following categories:

### Camera Data Sources (`camera/`)
- **MesoMetadataSource** (`mesoscope.py`) - Loads mesoscope camera metadata JSON files
- **PupilMetadataSource** (`pupil.py`) - Loads pupil camera metadata JSON files
- **Suite2pV2** (`suite2p.py`) - Loads Suite2p fluorescence outputs with nidaq-aligned timeline

### Behavioral Data Sources (`behavior/`)
- **TreadmillSource** (`treadmill.py`) - Loads treadmill speed and distance data with advanced alignment
- **DataqueueSource** (`dataqueue.py`) - Loads dataqueue synchronization data
- **WheelEncoder** (`wheel.py`) - Loads wheel encoder clicks/speed and aligns to dataqueue
- **Psychopy** (`psychopy.py`) - Loads Psychopy trial windows as interval data

### Analysis Data Sources (`analysis/`)
- **MesoMeanSource** (`mesoscope.py`) - Loads mesoscope mean fluorescence traces
- **MesoMapSource** (`mesomap.py`) - Loads mesomap traces and optional mask/region metadata
- **PupilDLCSource** (`pupil.py`) - Loads DeepLabCut pupil analysis results

### Session Data Sources (`session/`)
- **SessionConfigSource** (`config.py`) - Loads session configuration parameters
- **SessionNotesSource** (`notes.py`) - Loads timestamped session notes
- **SessionTimestampsSource** (`timestamps.py`) - Loads per-device start/stop timestamps

## Registry and Naming

Data sources are registered explicitly in `datakit/sources/__init__.py`. The registry
maps each source `tag` (e.g., `pupil_dlc`) to its class. Dataset column labels can
optionally use logical name overrides (see `settings.dataset.logical_name_overrides`).

## Adding New Data Sources

Define a subclass of `DataSource` (or `TimeseriesSource`, `TableSource`, or
`IntervalSeriesSource`) with a handful of class attributes. You do not need an
`__init__`. The `load()` method must return a `LoadedStream`, which you can get
by returning `(timeline, payload, meta)` from the appropriate `build_*` method.

To add a new data source:

1. Choose the appropriate category directory
2. Create a new Python file
3. Implement a class inheriting from `DataSource`
4. Set the required class variables:
   - `tag`: Unique identifier (e.g., "new_sensor")
   - `patterns`: File patterns to match (e.g., `("**/*_sensor.csv",)`)
   - `camera_tag`: Camera binding if applicable (or `None`)
    - `is_timeseries` (optional): Defaults to `True` for timeseries sources
    - `flatten_payload` (optional): Set `False` to keep a structured `raw_output`
5. Implement `build_timeseries()`, `build_table()`, or `build_intervals()`
6. Add the import and registry entry in `datakit/sources/__init__.py`

### Example New Data Source

```python
import numpy as np
import pandas as pd
from pathlib import Path

from datakit.sources.register import TimeseriesSource

class NewSensorSource(TimeseriesSource):
    """Load new sensor data."""
    tag = "new_sensor"
    patterns = ("**/*_sensor.csv",)
    camera_tag = None
    
    def build_timeseries(self, path: Path) -> tuple[np.ndarray, pd.DataFrame, dict]:
        # Implementation here
        ...
```

## Sharing a `tag` Across Multiple Sources (Additive Columns)

Two or more `DataSource` classes can share the same `tag` attribute so that
their outputs merge into the same column block on the materialized DataFrame.
This is useful when several files contribute different features that
conceptually belong to one logical source.

**Mechanics**
- The **registry key** in `SOURCE_REGISTRY` must be unique per class — it
  drives file discovery and produces one inventory column per key.
- The class **`tag` attribute** is what `materialize()` uses as the top-level
  column name on the output DataFrame.
- When two classes share `tag`, their flattened payloads land under the same
  `(tag, *)` block and sit side-by-side on each row.

**Example — `MesoMeanSource` and `MesoDFFSource`** both declare `tag = "meso"`
but match different file patterns, and are registered under distinct keys:

```python
# datakit/sources/analysis/mesoscope.py
class MesoMeanSource(TimeseriesSource):
    tag = "meso"
    patterns = ("**/*_meso-mean-trace.csv",)
    # build_timeseries returns a DataFrame with column "Mean"

class MesoDFFSource(TimeseriesSource):
    tag = "meso"
    patterns = ("**/*_meso-collapse_first-trace.csv",)
    # build_timeseries returns a DataFrame with column "dF_F"
```

```python
# datakit/sources/__init__.py
SOURCE_REGISTRY = {
    ...
    "meso_mean": MesoMeanSource,   # unique registry key
    "meso_dff":  MesoDFFSource,    # unique registry key
    ...
}
```

After materialization, each row exposes both features under the shared `meso`
block:

```text
("meso", "Mean")    ("meso", "dF_F")
```

**Caveats**
- Feature names within the shared block must not collide; if two sub-sources
  emit the same column name, the later writer wins for that cell.
- Each sub-source is still discovered, loaded, and error-reported
  independently under its own registry key (visible in `Dataset.sources`,
  `inspect_sources()`, and `validate()` output).
- Dependencies declared via `requires = (...)` reference **registry keys**,
  not the shared `tag`.

