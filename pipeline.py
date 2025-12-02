#%%
# ─── Imports and Configuration ─────────────────────────────────────────────────────
"""Step-by-step helpers for exercising datakit discovery and loading."""

from __future__ import annotations

# ─── standard python libraries ─────────────────────────────────────────────────────
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

# Ensure the project root is importable even when running cells out of context
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ─── third-party imports ─────────────────────────────────────────────────────
import pandas as pd


# ─── relative module imports ─────────────────────────────────────────────────────
from datakit.config import settings
from datakit.datamodel import Manifest
from datakit.discover import discover_manifest
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
EXPERIMENT_ROOT = (CURRENT_DIR / Path(*settings.debug.sample_experiment_rel)).resolve()
print(f"Using experiment root: {EXPERIMENT_ROOT}")


def _manifest_summary(manifest: Manifest) -> pd.DataFrame:
    if not manifest.entries:
        return pd.DataFrame(columns=["tag", "files", "processed", "data"])
    stats: dict[str, dict[str, int]] = {}
    for entry in manifest.entries:
        tag_stats = stats.setdefault(entry.tag, {"files": 0, "processed": 0, "data": 0})
        tag_stats["files"] += 1
        tag_stats[entry.origin] = tag_stats.get(entry.origin, 0) + 1
    rows = []
    for tag in sorted(stats.keys()):
        tag_stats = stats[tag]
        rows.append({
            "tag": tag,
            "files": tag_stats["files"],
            "processed": tag_stats.get("processed", 0),
            "data": tag_stats.get("data", 0),
        })
    return pd.DataFrame(rows)


def _entries_for_tag(manifest: Manifest, tag: str):
    return [entry for entry in manifest.entries if entry.tag == tag]


def _require_tags(manifest: Manifest, *tags: str) -> None:
    available = {entry.tag for entry in manifest.entries}
    missing = sorted(tag for tag in tags if tag not in available)
    if missing:
        raise ValueError(f"Missing required manifest tags: {', '.join(missing)}")

#%%
# ─── Tweak Pupil DLC Patterns ─────────────────────────────────────────────────────
"""
Modify the search patterns for Pupil DLC sources before running discovery.
"""
# add a custom glob for prototype exports
custom_pattern = ("**/*_full.pickle",)
PupilDLCSource.patterns = custom_pattern
print("Updated Pupil DLC patterns:", PupilDLCSource.patterns)

from datakit.sources import Suite2p
Suite2p.required_files = ("Fneu.npy", "spks.npy", "iscell.npy")


#%%
# ─── Discover Files and Build Manifest ─────────────────────────────────────────────────────
"""
Discover data files in the experiment root and construct a data manifest.
"""
manifest = discover_manifest(EXPERIMENT_ROOT)
manifest_summary_df = _manifest_summary(manifest)
if manifest_summary_df.empty:
    print("Manifest summary: <no files discovered>")
else:
    print("Manifest summary:\n", manifest_summary_df)


#%%
# ─── Inspect Manifest as DataFrame ─────────────────────────────────────────────────────
"""
Convert manifest entries to a pandas DataFrame for inspection.
"""
manifest_df = pd.DataFrame([asdict(entry) for entry in manifest.entries])
print(manifest_df.head())


#%%
# ─── Require Specific Data Source Tags ─────────────────────────────────────────────────────
"""
Ensure required data source tags are present in the manifest, raising an error if missing.
"""
# required_tags = ("psychopy", "suite2p")
# _require_tags(manifest, *required_tags)
# print("Required tags available:", required_tags)


#%%
# ─── Build Experiment Inventory ─────────────────────────────────────────────────────
"""
Create an inventory of experiment data with MultiIndex on Subject/Session/Task.
"""
experiment = ExperimentData(EXPERIMENT_ROOT, include_task_level=True)
inventory = experiment.data.copy()
print(inventory.head())


#%%
# ─── Prepare Experiment Store ─────────────────────────────────────────────────────
"""
Set up a minimal experiment store with default data sources registered.
"""
store = ExperimentStore(inventory)
missing_sources: list[str] = []
for src in DEFAULT_SOURCES:
    if src.tag not in inventory.columns:
        missing_sources.append(src.tag)
        continue
    store.register_series(src.logical_name, inventory[src.tag], loader=src.loader, structured=src.structured)

if missing_sources:
    print("Skipping sources missing from inventory:", sorted(missing_sources))
else:
    print("All default sources registered.")


#%%
# ─── Materialize Dataset ─────────────────────────────────────────────────────
"""
Generate the dataset DataFrame from the experiment store.
"""
dataset = store.materialize()
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

entries = _entries_for_tag(manifest, source_tag)

entry = entries[ENTRY_INDEX]
entry_path = (EXPERIMENT_ROOT / entry.path).resolve()
print(f"Loading {source_tag} entry at index {ENTRY_INDEX}: {entry_path}")

source = DataSource.create_loader(source_tag)
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
