#%%
# ─── Imports and Configuration ─────────────────────────────────────────────────────
"""Step-by-step helpers for exercising datakit discovery and loading."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is importable even when running cells out of context
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import matplotlib.pyplot as plt

import datakit
from datakit import Dataset, LoadedStream
from datakit import explore
from datakit.sources import SOURCE_REGISTRY

# Display pandas objects more readably while debugging
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 180)

#%%
# ─── Pipeline-Local Source Selection ───────────────────────────────────────────
PIPELINE_TAGS = (
    "mesomap",
    "timestamps",
    "dataqueue",
    "treadmill",
    "wheel",
    "notes",
    "session_config",
    "meso_metadata",
    "pupil_metadata",
    "pupil_dlc",
    "psychopy",
    "suite2p",
)

#%%

# ─── Quick-Test Helpers ───────────────────────────────────────────────────────────
# Use the built-in convenience methods on Dataset:
#   ds.head(3)                         # first 3 rows (replaces slice_inventory)
#   ds.select(subject, session, task)  # single entry (replaces select_inventory_entry)
#   ds.include(subject=..., session=..., task=..., source=...)  # general filter
#   ds.exclude(...)                    # inverse of include


#%%
# ─── Test a DataSource loader ───────────────────────────────────────────────────

source_tag = "psychopy"
experiment_root = Path(r"F:\251215_ETOH_RO1").resolve()
experiment_dataset = Dataset.from_directory(experiment_root, include_task_level=True)
inventory = experiment_dataset.inventory

# Get a file path from the inventory with the source_tag
# If you do not want to generate the inventory first, just use the filepath directly.
entry = Path(inventory[source_tag].iloc[0]).resolve()

# `entry` is a Path to a file for the given source_tag
loaded = datakit.load_path(source_tag, entry)

print(type(loaded))
print(loaded)

#%%
# ─── Load a single datasource from a path and plot its trace ───────────────
source_tag = "treadmill"
entry_path = Path(r"D:\jgronemeyer\240324_HFSA\data\sub-STREHAB02\ses-04\beh\20250327_165423_sub-STREHAB02_ses-04_task-widefield_treadmill_data.csv").resolve()

loaded_trace = datakit.load_path(source_tag, entry_path)
if not isinstance(loaded_trace, LoadedStream):
    raise TypeError("Expected LoadedStream from datakit.load_path")

trace = loaded_trace.value
if not isinstance(trace, pd.DataFrame):
    raise TypeError("Expected DataFrame payload from treadmill source")


time = trace["time_elapsed_s"]
speed = trace["speed_mm"]
distance = trace["distance_mm"]

plt.figure(figsize=(10, 4))
plt.plot(time, distance)
plt.xlabel("Time (s)")
plt.title(f"{source_tag} distance")
plt.ylabel("Distance (mm)")
plt.tight_layout()
plt.show()

#%%
# ─── Build a dataset ───────────────────────────────────────────────────

# One-shot: discover + materialize.
acutevis_root = Path(r"G:\Projects\ACUTEVIS").resolve()
materialized = datakit.load(acutevis_root, sources=PIPELINE_TAGS, progress=True)

# When you need to filter rows before materializing, use the lazy (meaning deferred) form:
#   ds = Dataset.from_directory(acutevis_root, sources=PIPELINE_TAGS)
#   ds = ds.select("ACUTEVIS06", "ses-02", "task-movies")  # single entry
#   ds = ds.include(subject=["ACUTEVIS06", "ACUTEVIS07"])  # multi-row filter
#   ds = ds.head(3)                                          # first 3 rows
#   materialized = ds.materialize(progress=True)

#%%
# ─── Explore: inspect dataset structure ───────────────────────────────
# `datakit.explore` accepts a Dataset, a materialized DataFrame, or a path
# to a directory / .pkl / .h5 file. It prints a formatted summary (uses
# `rich` if installed, otherwise plain text) and returns a report object
# for programmatic access.

# Pre-load: inventory overview from a discovered Dataset
acutevis_dataset = Dataset.from_directory(acutevis_root, sources=PIPELINE_TAGS)
inventory_report = explore(acutevis_dataset)

# Post-load: structure, dtypes, and coverage of a materialized DataFrame
materialized_report = explore(materialized)

# Or point directly at a directory or saved artifact
# explore(acutevis_root)
# explore(etoH_pickle_path)

#%%
# ─── Save dataset to disk ─────────────────────────────────────────────────────

# `Dataset.save` materializes and writes to disk in one step.
# Pickle by default; pass format="hdf5" or use a .h5/.hdf5 suffix for HDF5.
etoH_root = Path(r"F:\251215_ETOH_RO1").resolve()
etoH_dataset = Dataset.from_directory(etoH_root, sources=PIPELINE_TAGS)

etoH_pickle_path = etoH_root / "processed" / "260204_dataset_mvp.pkl"
etoH_dataset.save(etoH_pickle_path, progress=True)
print(f"ETOH dataset pickled to: {etoH_pickle_path}")

# Optionally also write HDF5
etoH_hdf_path = etoH_pickle_path.with_suffix(".h5")
etoH_dataset.save(etoH_hdf_path, format="hdf5", progress=True)
print(f"ETOH dataset stored at: {etoH_hdf_path}")

# Load pickle back into memory
materialized = pd.read_pickle(etoH_pickle_path)
print("Loaded dataset from pickle with shape", materialized.shape)


#%%
# ─── List Available Data Sources ─────────────────────────────────────────────────────
overview_df = pd.DataFrame(
    [{"tag": tag} for tag in sorted(SOURCE_REGISTRY.keys())]
).sort_values("tag")
print(overview_df)



#%%
# ─── Load and Merge ─────────────────────────────────────────────────────
experiments = [r'E:\jgronemeyer\250921_HFSA', r'D:\jgronemeyer\250627_HFSA']  # r'D:\jgronemeyer\240324_HFSA',

# `Dataset.from_directory` accepts a sequence of roots and concatenates them.
experiment_paths = [Path(p) for p in experiments]

merged_dataset = Dataset.from_directory(experiment_paths, sources=PIPELINE_TAGS).head(1)
inventory_report = explore(merged_dataset)

# Per-source coverage report on the merged inventory.
coverage = datakit.inspect_sources(merged_dataset, sources=PIPELINE_TAGS)
print(coverage)

materialized = merged_dataset.materialize(progress=True)
#materialized.to_pickle('260319_HFSA-full.pkl')

# %%
# ─── Validation report ───────────────────────────────────────────────────
# `validate` runs every (cell, source) and reports status without raising.
report = merged_dataset.validate(progress=True)
print(report.head())
# %%
