"""User-facing wrapper that ties discovery, manifests, and inventory data together."""

from __future__ import annotations

from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, cast

import pandas as pd

from .datamodel import Manifest
from .discover import discover_manifest
from .inventory import entries_to_inventory

# Ensure all DataSource subclasses register themselves
from . import sources as _data_sources  # noqa: F401


PathLike = Path | str
ManifestSource = Manifest | PathLike | Sequence[PathLike]


def _coerce_manifest(source: ManifestSource) -> Manifest:
    if isinstance(source, Manifest):
        return source

    if isinstance(source, (str, Path)):
        return discover_manifest(Path(source))

    if isinstance(source, SequenceABC):
        if isinstance(source, (str, bytes, bytearray)):
            return discover_manifest(Path(source))

        normalized: list[Path] = []
        for index, entry in enumerate(source):
            if isinstance(entry, Manifest):
                raise TypeError("Manifest objects are not supported inside source sequences")
            if not isinstance(entry, (str, Path)):
                raise TypeError(
                    f"Unsupported entry type at position {index}: {type(entry).__name__}"
                )
            normalized.append(Path(entry))

        if not normalized:
            raise ValueError("source sequence must contain at least one path")

        if len(normalized) == 1:
            return discover_manifest(normalized[0])

        return _combine_manifests(normalized)

    raise TypeError(f"Unsupported source type: {type(source).__name__}")


def _combine_manifests(paths: Sequence[Path]) -> Manifest:
    manifests = [discover_manifest(path) for path in paths]
    combined_entries = []

    for manifest in manifests:
        for entry in manifest.entries:
            absolute_path = (manifest.root / entry.path).resolve()
            combined_entries.append(replace(entry, path=absolute_path.as_posix()))

    if not combined_entries:
        joined = ", ".join(str(path) for path in paths)
        raise ValueError(f"No files discovered across directories: {joined}")

    combined_entries.sort(key=lambda entry: (entry.tag, entry.path))
    return Manifest(root=Path("."), entries=combined_entries)


@dataclass(frozen=True)
class ExperimentMetadata:
    """Convenience bundle returned when callers need both manifest and inventory."""

    manifest: Manifest
    inventory: pd.DataFrame


class ExperimentData:
    """High-level façade around discovery manifests and inventory DataFrames.

    Point it at an experiment directory (or a pre-built :class:`Manifest`) to
    run discovery, keep the resulting manifest handy, and expose a clean
    ``pandas.DataFrame`` listing every source path indexed by
    ``(Subject, Session, Task)``.
    """

    def __init__(
        self,
        source: ManifestSource,
        *,
        prefer_processed: bool = True,
        absolute_paths: bool = True,
        include_task_level: bool | None = None,
    ) -> None:
        manifest = _coerce_manifest(source)

        inventory = entries_to_inventory(
            manifest.entries,
            root=manifest.root,
            prefer_processed=prefer_processed,
            absolute_paths=absolute_paths,
            include_task_level=include_task_level,
        )

        self._manifest = manifest
        self._inventory = inventory
        self._prefer_processed = prefer_processed
        self._absolute_paths = absolute_paths
        self._include_task_level = include_task_level

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------
    @property
    def manifest(self) -> Manifest:
        return self._manifest

    @property
    def data(self) -> pd.DataFrame:
        return self._inventory.copy()

    @property
    def inventory(self) -> pd.DataFrame:
        return self.data

    @property
    def subjects(self) -> Sequence[str]:
        return sorted(self._inventory.index.get_level_values(0).unique().tolist())

    @property
    def sessions(self) -> Sequence[str]:
        if self._inventory.index.nlevels < 2:
            return []
        return sorted(self._inventory.index.get_level_values(1).unique().tolist())

    @property
    def has_task_level(self) -> bool:
        return self._inventory.index.nlevels == 3

    def select(
        self,
        *,
        subjects: Optional[Iterable[str]] = None,
        sessions: Optional[Iterable[str]] = None,
        tasks: Optional[Iterable[str]] = None,
    ) -> "ExperimentData":
        filtered_manifest = self._manifest.select(subjects=subjects, sessions=sessions, tasks=tasks)
        return ExperimentData(
            filtered_manifest,
            prefer_processed=self._prefer_processed,
            absolute_paths=self._absolute_paths,
            include_task_level=self._include_task_level,
        )

    def get_paths(self, subject: str, session: str, *, task: str | None = None) -> dict[str, str]:
        if self.has_task_level:
            if task is None:
                raise KeyError("task must be provided when inventory includes a task level")
            index_tuple = (subject, session, task)
        else:
            index_tuple = (subject, session)
        row = self._inventory.loc[cast(tuple[Any, ...], index_tuple), :]
        cleaned = row.dropna()
        return {str(key): str(value) for key, value in cleaned.items()}

    def to_metadata(self) -> ExperimentMetadata:
        return ExperimentMetadata(manifest=self._manifest, inventory=self.data)

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (
            f"ExperimentData(subjects={len(self.subjects)}, sessions={len(self.sessions)}, "
            f"has_task_level={self.has_task_level})"
        )

__all__ = ["ExperimentData", "ExperimentMetadata"]
