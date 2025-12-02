"""Utility primitives for declaring and discovering data sources.

The registry keeps a lightweight mapping of ``tag → SourceSpec`` so the loader
pipeline can look up which files to scan and which callable should deserialize
them.  Higher-level helpers in :mod:`datakit.loader` rely on this module to
bootstrap default sources as well as user-provided extensions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Iterable,
    Iterator,
    Optional,
    Tuple,
)

import numpy as np

from datakit.config import settings
from ..datamodel import LoadedStream

LoaderFn = Callable[[Path | str], object]

@dataclass(frozen=True)
class SourceSpec:
    """Description of a registered source.

    Attributes
    ----------
    tag:
        Short identifier that other components reference (e.g. ``"wheel"``).
    patterns:
        File glob patterns that tell the discovery step where to look.
    loader:
        Callable that receives a :class:`pathlib.Path` and returns the parsed
        payload, typically a :class:`pandas.DataFrame` or numpy array.
    structured:
        Flag used by legacy tooling to differentiate table outputs from raw
        binary blobs.  The loader does not interpret this flag directly but it
        is surfaced in metadata for dashboards.
    """

    tag: str
    patterns: Tuple[str, ...]
    loader: LoaderFn
    structured: bool = False


_REGISTRY: Dict[str, SourceSpec] = {}


def register_source(spec: SourceSpec) -> SourceSpec:
    """Insert a fully constructed :class:`SourceSpec` into the registry."""

    _REGISTRY[spec.tag] = spec
    return spec


def register(
    tag: str,
    *,
    patterns: Iterable[str],
    loader: LoaderFn,
    structured: bool = False,
) -> SourceSpec:
    """Convenience helper to register a source using primitive arguments."""

    spec = SourceSpec(tag=tag, patterns=tuple(patterns), loader=loader, structured=structured)
    return register_source(spec)


def get_source(tag: str) -> SourceSpec:
    """Return the registered spec for ``tag`` or raise ``KeyError``."""

    return _REGISTRY[tag]


def iter_sources() -> Iterator[SourceSpec]:
    """Iterate over all registered sources in insertion order."""

    return iter(_REGISTRY.values())


def list_sources() -> Tuple[SourceSpec, ...]:
    """Return a tuple snapshot of the registry contents."""

    return tuple(_REGISTRY.values())


class DataSource:
    """Shared contract for all file-backed loaders in ``datakit.sources``.

    Subclasses only need to declare a ``tag`` and override :meth:`load`.  When
    :meth:`load` returns a :class:`datakit.datamodel.LoadedStream`, the higher-
    level pipeline gains access to both the parsed payload and metadata that
    describes how the file relates to the experiment timeline.  The class also
    manages automatic registration (respecting :mod:`datakit.config` settings)
    and offers helper utilities for building typed streams with consistent
    metadata decoration.
    """
    
    REGISTRY: ClassVar[Dict[str, Dict[str, type["DataSource"]]]] = {}
    
    # Subclasses must define these class variables
    tag: ClassVar[str]                              # unique tag, e.g. "meso_metadata"
    patterns: ClassVar[Iterable[str]]               # e.g. ("**/*_mesoscope.ome.tiff_frame_metadata.json",)
    camera_tag: ClassVar[str | None] = None         # provider or binding camera tag
    version: ClassVar[str] = settings.registry.default_version
    is_timeseries: ClassVar[bool] = True            # whether this source produces a time-indexed stream
    flatten_payload: ClassVar[bool] = True          # True → break into scalar/array columns

    def __init_subclass__(cls) -> None:
        """Register subclasses automatically with version support."""
        if not settings.registry.auto_register:
            return
        tag = getattr(cls, "tag", None)
        patterns = getattr(cls, "patterns", None)
        if not tag or not patterns:
            return

        DataSource.REGISTRY.setdefault(tag, {})[cls.version] = cls
        spec = cls._as_source_spec()
        if spec is not None:
            register_source(spec)

    def load(self, path: Path) -> LoadedStream:
        """Load data from the given path. Subclasses must implement this method."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement load() method")

    # ------------------------------------------------------------------
    # Registration helpers
    # ------------------------------------------------------------------
    @classmethod
    def _patterns_tuple(cls) -> Tuple[str, ...]:
        patterns = getattr(cls, "patterns", ())
        if isinstance(patterns, tuple):
            return patterns
        return tuple(patterns)

    @classmethod
    def _build_loader(cls) -> LoaderFn:
        def _loader(path: Path | str) -> object:
            instance = cls()
            return instance.load(Path(path))

        return _loader

    @classmethod
    def _as_source_spec(cls) -> Optional[SourceSpec]:
        if not getattr(cls, "tag", None):
            return None
        patterns = cls._patterns_tuple()
        if not patterns:
            return None
        loader = cls._build_loader()
        structured = not getattr(cls, "flatten_payload", True)
        return SourceSpec(tag=cls.tag, patterns=patterns, loader=loader, structured=structured)

    # ------------------------------------------------------------------
    # Helper utilities for subclasses
    # ------------------------------------------------------------------
    def _create_stream(
        self,
        tag: str,
        t: Iterable[float],
        value: Any,
        meta: Optional[Dict[str, Any]] = None,
    ) -> LoadedStream:
        """Utility to build a :class:`LoadedStream` with standard metadata decoration.

        Subclasses that need to customize the metadata can pass in a ``meta`` dictionary.
        This helper will ensure that fundamental fields like ``camera_tag`` and
        ``is_timeseries`` are present, cast the timeline to ``float64``, and remember the
        most recent stream for follow-up helpers that need it (e.g. frame index queries).
        """

        meta_dict: Dict[str, Any] = dict(meta or {})

        if self.camera_tag is not None:
            meta_dict.setdefault(settings.registry.meta_camera_key, self.camera_tag)
        meta_dict.setdefault(settings.registry.meta_timeseries_key, self.is_timeseries)
        meta_dict.setdefault(settings.registry.meta_source_key, tag)

        timeline = np.asarray(list(t) if not isinstance(t, np.ndarray) else t, dtype=np.float64)

        stream = LoadedStream(tag=tag, t=timeline, value=value, meta=meta_dict)
        setattr(self, "_last_loaded_stream", stream)
        return stream
    
    @classmethod
    def get_registered_sources(cls) -> Dict[str, Dict[str, type["DataSource"]]]:
        """Get all registered data source classes with versions."""
        return cls.REGISTRY.copy()
    
    @classmethod
    def get_available_versions(cls, tag: str) -> list[str]:
        """Get available versions for a given tag."""
        if tag not in cls.REGISTRY:
            return []
        return list(cls.REGISTRY[tag].keys())
    
    @classmethod
    def get_latest_version(cls, tag: str) -> str:
        """Get the latest version for a given tag (highest version number)."""
        versions = cls.get_available_versions(tag)
        if not versions:
            raise ValueError(f"No versions found for tag: {tag}")
        # Sort versions as strings for now (could implement semantic versioning later)
        return sorted(versions)[-1]
    
    @classmethod
    def create_loader(cls, tag: str, version: str | None = None) -> "DataSource":
        """Create a loader instance for the given tag and version.
        
        Args:
            tag: The data source tag
            version: Specific version to use. If None, uses latest version.
        """
        if tag not in cls.REGISTRY:
            raise ValueError(f"No loader registered for tag: {tag}")
        
        if version is None:
            version = cls.get_latest_version(tag)
        
        if version not in cls.REGISTRY[tag]:
            available = list(cls.REGISTRY[tag].keys())
            raise ValueError(f"Version {version} not found for tag {tag}. Available versions: {available}")
        
        return cls.REGISTRY[tag][version]()