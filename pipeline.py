#%%
# ─── Imports and Configuration ─────────────────────────────────────────────────────
"""Step-by-step helpers for exercising datakit discovery and loading."""

from __future__ import annotations

# ─── standard python libraries ─────────────────────────────────────────────────────
import sys
from pathlib import Path
from typing import Any, Optional

# Ensure the project root is importable even when running cells out of context
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ─── third-party imports ─────────────────────────────────────────────────────
import pandas as pd


# ─── relative module imports ─────────────────────────────────────────────────────
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

# Display pandas objects more readably while debugging
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 180)

# Root of the experiment to inspect. Update this path for other runs.
EXPERIMENT_ROOT = (CURRENT_DIR / Path(r'E:\jgronemeyer\250921_HFSA')).resolve()
print(f"Using experiment root: {EXPERIMENT_ROOT}")
roots = [r'E:\jgronemeyer\250921_HFSA',
         r'D:\jgronemeyer\240324_HFSA',
         r'D:\jgronemeyer\250627_HFSA']
#%%
# ─── Tweak Pupil DLC Patterns ─────────────────────────────────────────────────────
"""
Modify the search patterns for Pupil DLC sources before running discovery.
"""
from datakit.sources import MesoMetadataSource, PupilMetadataSource
# add a custom glob for prototype exports
meso_patterns = ("**/*_mesoscope.ome.tiff_frame_metadata.json",
                  "**/*_meso.ome.tiff_frame_metadata.json")
pupil_patterns = ("**/*_pupil.mp4_frame_metadata.json",
                  "**/*_pupil.ome.tiff_frame_metadata.json")
MesoMetadataSource.patterns = meso_patterns
PupilMetadataSource.patterns = pupil_patterns



#%%
# ─── Build Experiment Inventory ─────────────────────────────────────────────────────
"""
Create an inventory of experiment data with MultiIndex on Subject/Session/Task.
"""
experiment = ExperimentData(roots, include_task_level=True)
inventory = experiment.data.copy()
print(inventory.head())


#%%
# ─── Prepare Experiment Store ─────────────────────────────────────────────────────
"""
Set up a minimal experiment store with default data sources registered.
"""
store = ExperimentStore(inventory)


#%%
# ─── Materialize Dataset ─────────────────────────────────────────────────────
"""
Generate the dataset DataFrame from the experiment store.
"""
from datakit.loader import ExperimentStore, DEFAULT_SOURCES

inventory = ExperimentData(roots, include_task_level=True).data
store = ExperimentStore(inventory)

for src in DEFAULT_SOURCES:
    if src.tag in inventory.columns:
        store.register_series(src.logical_name, inventory[src.tag], src.loader, structured=src.structured)

dataset = store.materialize()
#%%
print("Dataset summary:")
print("Columns:", dataset.columns.tolist())
print("Data types:\n", dataset.dtypes)
print("\nStructure:")
dataset.info(verbose=False)
print("\nTop rows:")
print(dataset.head())
#%%
# ─── Inspect metadata ─────────────────────────────────────────────────────
meta = store.meta_frame              # normalized per-stream metadata
session_info = store.session_attrs   # dict keyed by (subject, session)
tbases = store.time_basis            # per-source basis strings
print("Treadmill time basis:", store.time_basis.get("treadmill", "<missing>"))


#%%
# ─── Explore Metadata and Session Attributes ────────────────────────────────────
"""
Inspect per-source metadata, session-level attributes from config, and time bases.
"""
print("\nMeta rows (first 10):")
print(store.meta_frame.head(10))

print("\nSession attributes (first 5 entries):")
for key, attrs in list(store.session_attrs.items())[:5]:
    print(key, attrs)

print("\nExperiment attributes:")
print(store.experiment_attrs)

print("\nTime basis per source:")
print(store.time_basis)


#%%
# ─── Metadata / Time Basis Checks ─────────────────────────────────────
"""
Sanity-check that metadata columns match config and every recorded time basis is well-formed.
"""
expected_meta_columns = list(settings.dataset.meta_columns)
missing_meta_columns = [col for col in expected_meta_columns if col not in store.meta_frame.columns]
if missing_meta_columns:
    raise AssertionError(f"Missing metadata columns: {missing_meta_columns}")

source_col = expected_meta_columns[3] if len(expected_meta_columns) > 3 else "Source"
if source_col in store.meta_frame.columns and not store.meta_frame.empty:
    null_sources = store.meta_frame[source_col].isna().sum()
    print(f"{null_sources} metadata rows missing {source_col} tag")
else:
    print("No metadata rows to validate against source tags.")

invalid_time_bases = {src: basis for src, basis in store.time_basis.items() if not isinstance(basis, str) or not basis}
if invalid_time_bases:
    raise AssertionError(f"Invalid time basis entries: {invalid_time_bases}")

print("Metadata/time-basis checks passed.")


#%%
# ─── Persist Dataset to Disk ─────────────────────────────────────────────────────
"""
Save the dataset to an HDF file on disk for later use.
"""
output_path: Optional[Path] = None  # Set to a Path to customise the output location
hdf_path = build_default_dataset(EXPERIMENT_ROOT, output_path=output_path)
print(f"Dataset stored at: {hdf_path}")


#%%
# ─── Explore Single Data Source Entry ─────────────────────────────────────────────────────
"""
Load and examine a specific entry from a chosen data source.
"""
source_tag = "psychopy"
ENTRY_INDEX = settings.debug.default_entry_index

entries = manifest.for_tag("psychopy")

entry_path = entries[0].path
print(f"Loading {source_tag} entry at index {ENTRY_INDEX}: {entry_path}")

source = DataSource.create_loader("psychopy")
loaded = source.load(entry_path)
print(type(loaded))
print(loaded)


#%%
# ─── Inspect Loaded Payload ─────────────────────────────────────────────────────
"""
Examine the payload from the loaded data stream if available.
"""
if hasattr(loaded, "payload"):
    payload = loaded.payload
    print(f"Payload kind: {payload.kind}")
    if hasattr(payload, "data_view"):
        print(payload.data_view)
else:
    print("Loaded object does not expose a payload attribute.")


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
# ─── Settings Override Checks ─────────────────────────────────────
"""
Ensure runtime overrides (e.g., toggling DatasetLayout.desired_tags) take effect.
"""
original_tags = settings.dataset.desired_tags
custom_tag = "custom_debug_tag"
test_tags = tuple(list(original_tags) + [custom_tag])

settings.dataset.desired_tags = test_tags
assert settings.dataset.desired_tags == test_tags
print("Updated desired_tags:", settings.dataset.desired_tags)

# Revert to avoid polluting later cells.
settings.dataset.desired_tags = original_tags
print("Restored desired_tags:", settings.dataset.desired_tags)


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

#%%
dataset = merged_store.materialize()
# %%
