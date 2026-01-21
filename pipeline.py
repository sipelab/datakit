#%%
# ─── Imports and Configuration ─────────────────────────────────────────────────────
"""Step-by-step helpers for exercising datakit discovery and loading."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

# Ensure the project root is importable even when running cells out of context
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from datakit.config import settings
#from datakit.discover import DataManifest
from datakit.experiment import ExperimentData
from datakit.loader import (
    DEFAULT_SOURCES,
    ExperimentStore,
    build_default_dataset,
    launch_dataset_shell,
)
from datakit.sources.register import DataSource
from datakit.sources.analysis.pupil import PupilDLCSource
import numpy as np
import matplotlib.pyplot as plt

# Display pandas objects more readably while debugging
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 180)



# ─── Quick-Test Helper ───────────────────────────────────────────────────────────
"""
Return a small slice of the inventory for quick tests.
"""
def slice_inventory(frame: pd.DataFrame, entries: Any = 3) -> pd.DataFrame:
    if isinstance(entries, int):
        return frame.iloc[:entries].copy()
    return frame.loc[list(entries)].copy()


#%%
# ─── Test a DataSource loader ───────────────────────────────────────────────────

source_tag = "psychopy"
experiment_root = Path(r"F:\251215_ETOH_RO1").resolve()
experiment = ExperimentData(experiment_root, include_task_level=True)
inventory = experiment.data

# Get a file path from the inventory with the source_tag
# If you do not want to generate the inventory first, just simply to the filepath directly
entry = Path(inventory[source_tag].iloc[0]).resolve()

# All DataSource subclasses are registered and created via a factory method using the tag
loader = DataSource.create_loader(source_tag)

# `entry` is a Path to a file for the given source_tag
loaded = loader.load(entry)

print(type(loaded))
print(loaded)

#%%
# ─── Build a dataset ───────────────────────────────────────────────────

# Build dataset for F:\251215_ETOH_RO1
etoH_root = Path(r"F:\251205_ETOH_RO1").resolve()
etoH_experiment = ExperimentData(etoH_root, include_task_level=True)
sliced_inventory = slice_inventory(etoH_experiment.data)

store = ExperimentStore(sliced_inventory)

for src in DEFAULT_SOURCES:
    if src.tag in sliced_inventory.columns:
        store.register_series(src.logical_name, sliced_inventory[src.tag], src.loader, structured=src.structured)

dataset = store.materialize()

#%%
#%%
# ─── Build Dataset Function and Save dataset to disk ─────────────────────────────────────────────────────

# As opposed to the steps above, the build_default_dataset function
# handles the entire experiment inventory and dataset building in one step.
# This saves to an HDF5 file on disk and returns the path to that file.
etoH_dataset_path = build_default_dataset(etoH_root)
print(f"ETOH dataset stored at: {etoH_dataset_path}")

# Load HDF5 into pandas and save as pickle
etoH_loaded = pd.read_hdf(etoH_dataset_path)
etoH_pickle_path = etoH_dataset_path.with_suffix(".pkl")
etoH_loaded.to_pickle(etoH_pickle_path)
print(f"ETOH dataset pickled to: {etoH_pickle_path}")

# Load pickle into memory; `dataset` variable
dataset = pd.read_pickle(r"F:\251215_ETOH_RO1\processed\260116_dataset_mvp.pkl")
print("Loaded dataset from pickle with shape", dataset.shape)


#%%
# ─── List Available Data Sources ─────────────────────────────────────────────────────
"""
Display an overview of registered data source tags and their versions.
"""
registry = DataSource.get_registered_sources()
overview = []
for tag, versions in registry.items():
    overview.append({
        "tag": tag,
        "versions": sorted(versions.keys()),
        "latest": DataSource.get_latest_version(tag),
    })
overview_df = pd.DataFrame(overview).sort_values("tag")
print(overview_df)



#%%
# ─── Load and Merge ─────────────────────────────────────────────────────
experiments = [r'D:\jgronemeyer\240324_HFSA', r'E:\jgronemeyer\250921_HFSA', r'D:\jgronemeyer\250627_HFSA']
INSPECT_SOURCES = ("dataqueue", "timestamps")

def load_inventory(path: Path) -> pd.DataFrame:
    experiment = ExperimentData(path, include_task_level=True)
    return experiment.data


def inspect_sources(frame: pd.DataFrame, label: str) -> None:
    for source in INSPECT_SOURCES:
        if source not in frame.columns:
            print(f"[inspect] {label}: missing '{source}' column entirely")
            continue
        series = frame[source]
        present = series.notna().sum()
        total = len(series)
        print(f"[inspect] {label}: '{source}' populated for {present}/{total} rows")
        if present == 0:
            sample_index = list(series.index[:3])
            print(f"           first few index entries without '{source}': {sample_index}")


def merge_inventories(paths) -> pd.DataFrame:
    frames = []
    for raw in paths:
        resolved = raw.expanduser().resolve()
        if not resolved.exists() or not resolved.is_dir():
            raise FileNotFoundError(f"Experiment directory missing: {resolved}")
        frame = load_inventory(resolved)
        inspect_sources(frame, resolved.name)
        frames.append(frame)
        print(f"Loaded {resolved} -> shape {frame.shape}")
    return pd.concat(frames, sort=False)


experiment_paths = [Path(p) for p in experiments]
merged = merge_inventories(experiment_paths)
merged_store = ExperimentStore(merged)

merged_missing: list[str] = []
for src in DEFAULT_SOURCES:
    if src.tag not in merged.columns:
        merged_missing.append(src.tag)
        continue
    merged_store.register_series(src.logical_name, merged[src.tag], loader=src.loader, structured=src.structured)

if merged_missing:
    print("Merged inventory missing sources:", sorted(merged_missing))
else:
    print("Merged inventory includes all default sources.")


dataset = merged_store.materialize()


# %%
