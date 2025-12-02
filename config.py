"""Centralized configuration for ``datakit`` infrastructure.

The file gathers all tweakable defaults (registry behaviour, dataset layout,
timeline parsing, debug helpers, etc.) into a single, human-readable module so
new contributors can understand *which* knobs exist without spelunking through
call sites.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class RegistryDefaults:
    """Base behaviour for the :class:`datakit.sources.register.DataSource` registry.

    These knobs rarely change at runtime, but documenting them here makes it
    obvious what metadata ``DataSource`` subclasses are expected to emit.
    """

    default_version: str = "1.0"
    auto_register: bool = True
    meta_camera_key: str = "camera_tag"
    meta_timeseries_key: str = "is_timeseries"
    meta_source_key: str = "source_tag"


@dataclass
class DatasetLayout:
    """Shape and naming expectations for materialized experiment datasets."""

    index_names: Tuple[str, str, str] = ("Subject", "Session", "Task")
    desired_tags: Tuple[str, ...] = (
        "meso_mean",
        "timestamps",
        "dataqueue",
        "psychopy",
        "wheel",
        "notes",
        "session_config",
        "meso_metadata",
        "pupil_metadata",
        "dlc_pupil",
        "suite2p",
    )
    logical_name_overrides: Dict[str, str] = field(default_factory=lambda: {
        "meso_metadata": "meso_meta",
        "pupil_metadata": "pupil_meta",
        "dlc_pupil": "pupil",
        "wheel": "encoder",
    })
    meta_columns: Tuple[str, ...] = ("Subject", "Session", "Task", "Source", "Key", "Value", "dtype")
    scope_key: str = "scope"
    session_scope: str = "session"
    experiment_scope: str = "experiment"
    time_basis_key: str = "time_basis"


@dataclass
class TimelineDefaults:
    """Options that describe how we discover and parse timeline CSV files."""

    dataqueue_glob: str = "*_dataqueue.csv"
    queue_column: str = "queue_elapsed"


@dataclass
class DebugDefaults:
    """Hard-coded fixture shortcuts used across debugging notebooks/scripts."""

    sample_experiment_rel: Tuple[str, ...] = ("tests", "sample_experiment3")
    default_entry_index: int = 0


@dataclass
class Settings:
    """Container aggregating all configuration namespaces for callers."""

    registry: RegistryDefaults = field(default_factory=RegistryDefaults)
    dataset: DatasetLayout = field(default_factory=DatasetLayout)
    timeline: TimelineDefaults = field(default_factory=TimelineDefaults)
    debug: DebugDefaults = field(default_factory=DebugDefaults)


settings = Settings()


__all__ = ["settings"]
