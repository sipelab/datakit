# Dataset DataFrame Guide

This guide documents the Pandas structure produced by `ExperimentStore.materialize()` (and by helper utilities such as `build_default_dataset`). It explains how the resulting `dataset` DataFrame is organized and how to navigate it from user code.

## Overall layout

- **Type**: `pandas.DataFrame`
- **Row index**: Pandas `MultiIndex` with the levels `Subject`, `Session`, `Task`. Each row corresponds to a unique experiment task within a subject/session pair.
- **Columns**: Pandas `MultiIndex` with the levels `Source` and `Feature`. The top level identifies the logical data source (e.g. `suite2p`, `psychopy`); the second level encodes the specific measurement or payload exported by that source.

Because the columns form a multi-level index, you can select all features for a given source using either column indexing or the attribute shorthand:

```python
# Using column selection
suite2p_block = dataset.loc[:, "suite2p"]

# Using attribute access (pandas exposes top-level column labels as attributes)
suite2p_block = dataset.suite2p
```

Both approaches return another DataFrame where the columns correspond to the features emitted by the source.

## Source naming

Dataset column labels use logical names that can differ from source tags. The
default mapping is defined in `settings.dataset.logical_name_overrides`:

- `dataqueue` -> `time`
- `wheel` -> `encoder`
- `pupil_dlc` -> `pupil`
- `meso_metadata` -> `meso_meta`
- `pupil_metadata` -> `pupil_meta`

## Index expectations

The `dataset.index` always carries three named levels (`Subject`, `Session`, `Task`). Typical access patterns include:

```python
# All tasks for a specific subject
subject_rows = dataset.loc["ACUTEVIS06"]

# Single subject/session pair
session_rows = dataset.xs(("ACUTEVIS06", "ses-01"))

# Exact subject/session/task triple
row = dataset.loc[("ACUTEVIS06", "ses-01", "task-gratings")]
```

## Source feature overview

The table below lists the primary sources produced by `DEFAULT_SOURCES` together with the columns created for each source. The "Sample value type" column reflects the Python type you receive when inspecting one row of the DataFrame.

| Source | Feature | Sample value type |
| --- | --- | --- |
| `meso_mean` | `Slice`, `Mean`, `time_elapsed_s`, `dF_F` | `numpy.ndarray` |
| `mesomap` | ROI columns from the traces CSV, plus `time_elapsed_s` | `numpy.ndarray` |
| `timestamps` | `device_id`, `started`, `stopped`, `start_s`, `stop_s` | `numpy.ndarray` |
| `time` | `queue_elapsed`, `packet_ts`, `device_ts`, `device_id`, `payload`, `time_elapsed_s`, `device_elapsed_s` (if available) | `numpy.ndarray` |
| `treadmill` | `timestamp`, `distance_mm`, `speed_mm`, `time_elapsed_s` (plus passthrough columns) | `numpy.ndarray` |
| `encoder` | `time_elapsed_s`, `time_reference_s`, `click_delta`, `click_position`, `speed_mm`, `distance_mm`, `time_absolute` (optional) | `numpy.ndarray` |
| `notes` | `raw_output` | `pandas.DataFrame` |
| `session_config` | Parameter columns from the configuration CSV | scalar values (`str`, `float`, `int`, etc.) |
| `meso_meta` | `raw_output` | `pandas.DataFrame` |
| `pupil_meta` | `raw_output` | `pandas.DataFrame` |
| `pupil` | `pupil_diameter_mm`, `time_elapsed_s` | `numpy.ndarray` |
| `suite2p` | `cell_identifier` | `numpy.ndarray` (nROIs × 2 columns: accepted flag & original ROI index) |
|  | `stat` | `list` of ROI metadata dicts |
|  | `ops` | `dict` (Suite2p ops configuration) |
|  | `plane_directory` | `str` |
|  | `cell_mask` | `numpy.ndarray` (boolean mask of accepted cells) |
|  | `roi_fluorescence` | `numpy.ndarray` (cells × frames) |
|  | `neuropil_fluorescence` | `numpy.ndarray` (cells × frames) |
|  | `deltaf_f` | `numpy.ndarray` (dF/F per cell) |
|  | `time_native_s` | `numpy.ndarray` (native frame times) |
|  | `time_elapsed_s` | `numpy.ndarray` (aligned frame times) |

> **Note:** Most timeseries sources include `time_elapsed_s` seconds since acquisition start. Interval sources (like `timestamps`) use `start_s`/`stop_s`. Some sources add `time_absolute` when wall-clock timestamps are available.

## Metadata columns

Every loader can emit metadata via the `meta` dictionary. During materialization,
metadata is added as extra columns under the same `Source` using two patterns:

- `meta.<key>` columns for individual metadata keys
- `meta` column containing a dict of all metadata for that source

Session-scoped metadata (e.g., from `mesomap` or `session_config`) is stored once
per subject/session instead of once per task.

## Working with feature values

Columns that contain arrays or lists (e.g. `suite2p.roi_fluorescence`) store numpy arrays directly in the DataFrame cell. Use standard numpy operations to manipulate them:

```python
row = dataset.loc[("ACUTEVIS06", "ses-01", "task-gratings")]
trace_matrix = row["suite2p", "roi_fluorescence"]  # ndarray shape: (n_cells, n_frames)
mean_trace = trace_matrix.mean(axis=0)
```

Structured sources (those that set `structured=True` when being registered) retain their original pandas object under a `raw_output` feature. For example:

```python
session_cfg = dataset["session_config", "raw_output"].loc[("ACUTEVIS06", "ses-01", "task-gratings")]
# session_cfg is a pandas.DataFrame with the experiment configuration.
```

## Discovering available features at runtime

Use the pandas metadata to introspect the DataFrame dynamically:

```python
sources = dataset.columns.get_level_values("Source").unique()
for source in sources:
    block = dataset[source]
    print(source, list(block.columns))
```

This pattern is useful when custom experiments register additional data sources or when you load datasets built with an extended source list.

## Summary

- Rows are keyed by `(Subject, Session, Task)`.
- Columns are grouped by source, then feature.
- Array-valued features store numpy ndarrays per row; structured outputs remain as nested pandas objects under `raw_output`.
- Attribute access (`dataset.suite2p`, `dataset.psychopy`) is a convenient way to grab all columns for a source before working with individual features.

Armed with the structure and access patterns above, users can explore the dataset object interactively, slice individual sources, and feed the underlying arrays into downstream analysis pipelines.
