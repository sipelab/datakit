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
| `timestamps` | `raw_output` | `pandas.DataFrame` (frame metadata) |
| `dataqueue` | `queue_elapsed` | `numpy.ndarray` |
|  | `packet_ts` | `numpy.ndarray` |
|  | `device_ts` | `numpy.ndarray` |
|  | `device_id` | `numpy.ndarray` |
|  | `payload` | `numpy.ndarray` |
| `encoder` | `time_s` | `numpy.ndarray` |
|  | `time_raw_s` | `numpy.ndarray` |
|  | `click_delta` | `numpy.ndarray` |
|  | `click_position` | `numpy.ndarray` |
|  | `speed_mm` | `numpy.ndarray` |
|  | `distance_mm` | `numpy.ndarray` |
|  | `time_basis` | `numpy.ndarray` |
| `notes` | `timestamp` | `numpy.ndarray` (UTC timestamps) |
|  | `note` | `numpy.ndarray` (strings) |
| `psychopy` | `('trials', '...')` columns | `list` of scalar values (one per trial) |
|  | `('epochs', '...')` columns | `list` of timestamps (epoch on/off pairs) |
|  | `('analysis', '...')` columns | `list` of derived timing offsets |
| `pupil_dlc` | `pupil_diameter_mm` | `numpy.ndarray` |
| `pupil_meta` | `runner_time_ms`, `exposure_ms`, `mda_event`, `Camera`, `ElapsedTime-ms`, `Height`, `ImageNumber`, `TimeReceivedByCore`, `Width` | `numpy.ndarray` |
| `session_config` | `raw_output` | `pandas.DataFrame` |
| `suite2p` | `cell_identifier` | `numpy.ndarray` (nROIs × 2 columns: accepted flag & original ROI index) |
|  | `spike_rate` | `numpy.ndarray` (cells × frames) |
|  | `stat` | `list` of ROI metadata dicts |
|  | `ops` | `dict` (Suite2p ops configuration) |
|  | `plane_directory` | `str` |
|  | `cell_mask` | `numpy.ndarray` (boolean mask of accepted cells) |
|  | `roi_fluorescence` | `numpy.ndarray` (cells × frames) |
|  | `neuropil_fluorescence` | `numpy.ndarray` (cells × frames) |
|  | `deltaf_f` | `numpy.ndarray` (ΔF/F per cell) |
|  | `interp_deltaf_f` | `numpy.ndarray` (ΔF/F resampled to 10 Hz) |
|  | `smoothed_dff` | `numpy.ndarray` (moving-average ΔF/F) |
|  | `interpolation_method` | `str` (`"cubic_spline"`, `"linear"`, or `"unavailable"`) |
|  | `mean_fluo_dff` | `numpy.ndarray` (population-average ΔF/F trace) |
|  | `peaks_prominence` | `numpy.ndarray` (indices of prominent peaks) |
|  | `num_peaks_prominence` | `int` |
|  | `time_native_s` | `numpy.ndarray` (native frame times) |
|  | `time_elapsed_s` | `numpy.ndarray` (10 Hz timeline) |

> **Note:** All timeseries sources expose a `time_elapsed_s` feature representing seconds since acquisition start. Optional absolute timestamps appear under `time_absolute` when the origin device records wall-clock time.

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
