"""Thin wrappers around the MultiIndex dataset produced by ``ExperimentStore``.

Exploratory analysis often needs higher-level affordances than a raw
``pandas.DataFrame``. The utilities here expose source-aware views, cached
metadata helpers, and attribute accessors that make notebooks and scripts more
readable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Tuple

import pandas as pd

from datakit.config import settings


class _SourceView:
    """Readable access shim for a single logical source inside the dataset."""

    def __init__(self, dataset: "MaterializedDataset", name: str) -> None:
        self._dataset = dataset
        self.name = name

    @property
    def frame(self) -> pd.DataFrame:
        columns = self._dataset.frame.columns
        if isinstance(columns, pd.MultiIndex):
            try:
                subset = self._dataset.frame.xs(self.name, axis=1, level=0, drop_level=True)
            except (KeyError, ValueError):
                mask = columns.get_level_values(0) == self.name
                subset = self._dataset.frame.loc[:, mask]
                subset.columns = subset.columns.droplevel(0)
            if isinstance(subset, pd.Series):
                subset = subset.to_frame()
            return subset
        filtered = self._dataset.frame.filter(like=self.name, axis=1)
        if isinstance(filtered, pd.Series):
            filtered = filtered.to_frame()
        return filtered

    @property
    def meta(self) -> pd.DataFrame:
        return self._dataset.meta_for(self.name)

    @property
    def time_basis(self) -> str | None:
        return self._dataset.time_basis.get(self.name)

    def __repr__(self) -> str:  # pragma: no cover - repr sugar
        return f"SourceView(name={self.name!r}, time_basis={self.time_basis!r})"


class _SourceAccessor:
    def __init__(self, dataset: "MaterializedDataset") -> None:
        self._dataset = dataset

    def __getattr__(self, name: str) -> _SourceView:
        if name in self._dataset.source_names:
            return _SourceView(self._dataset, name)
        raise AttributeError(f"Unknown source '{name}'")

    def __getitem__(self, name: str) -> _SourceView:
        if name not in self._dataset.source_names:
            raise KeyError(name)
        return _SourceView(self._dataset, name)

    def __iter__(self) -> Iterable[str]:
        return iter(self._dataset.source_names)

    def keys(self) -> Iterable[str]:
        return self._dataset.source_names


@dataclass
class MaterializedDataset:
    """User-friendly companion to the DataFrame returned by ``ExperimentStore``.

    Besides the wide ``frame`` of measurement values, it keeps normalized
    metadata, cached session-level attributes, and helper accessors that mirror
    familiar pandas idioms while remaining source-aware.
    """

    frame: pd.DataFrame
    meta_frame: pd.DataFrame
    session_attrs: Dict[Tuple[str, str], Dict[str, Any]]
    experiment_attrs: Dict[str, Any]
    time_basis: Dict[str, str]
    _meta_cache: Dict[str, pd.DataFrame] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sources", _SourceAccessor(self))
        if not isinstance(self.meta_frame, pd.DataFrame):
            object.__setattr__(self, "meta_frame", pd.DataFrame(columns=self._meta_columns()))
        else:
            missing = [col for col in self._meta_columns() if col not in self.meta_frame.columns]
            if missing:
                for col in missing:
                    self.meta_frame[col] = pd.NA
                self.meta_frame = self.meta_frame[self._meta_columns()]

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _meta_columns() -> list[str]:
        return list(settings.dataset.meta_columns)

    @property
    def source_names(self) -> Tuple[str, ...]:
        if not isinstance(self.frame.columns, pd.MultiIndex):
            return tuple(sorted({str(col) for col in self.frame.columns}))
        return tuple(sorted(self.frame.columns.get_level_values(0).unique()))

    def meta_for(
        self,
        source: str,
        *,
        subject: str | None = None,
        session: str | None = None,
        task: str | None = None,
    ) -> pd.DataFrame:
        meta_columns = list(settings.dataset.meta_columns)
        source_col = meta_columns[3] if len(meta_columns) > 3 else "Source"
        if source_col not in self.meta_frame.columns:
            return pd.DataFrame(columns=self._meta_columns())
        if source not in self._meta_cache:
            subset = self.meta_frame[self.meta_frame[source_col] == source]
            self._meta_cache[source] = subset
        result = self._meta_cache[source]
        mask = pd.Series(True, index=result.index)
        index_names = list(settings.dataset.index_names)
        if not index_names:
            raise ValueError("dataset.index_names must define at least one label")
        subject_col = index_names[0]
        session_col = index_names[1] if len(index_names) > 1 else "Session"
        task_col = index_names[2] if len(index_names) > 2 else "Task"
        if subject is not None and subject_col in result.columns:
            mask &= result[subject_col] == subject
        if session is not None and session_col in result.columns:
            mask &= result[session_col] == session
        if task is not None and task_col in result.columns:
            mask &= result[task_col] == task
        return result.loc[mask].copy()

    def session_value(self, subject: str, session: str, key: str) -> Any:
        return self.session_attrs.get((subject, session), {}).get(key)

    def experiment_value(self, key: str) -> Any:
        return self.experiment_attrs.get(key)

    # ------------------------------------------------------------------
    # DataFrame delegation
    # ------------------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        if name in {"frame", "meta_frame", "session_attrs", "experiment_attrs", "time_basis", "sources"}:
            return object.__getattribute__(self, name)
        if name in self.experiment_attrs:
            return self.experiment_attrs[name]
        if name in self.source_names:
            return _SourceView(self, name)
        attr = getattr(self.frame, name)
        if callable(attr):
            def _wrapper(*args: Any, **kwargs: Any) -> Any:
                return attr(*args, **kwargs)
            return _wrapper
        return attr

    def __getitem__(self, key: Any) -> Any:
        return self.frame.__getitem__(key)

    def __len__(self) -> int:
        return len(self.frame)

    def __iter__(self):  # pragma: no cover - delegation
        return iter(self.frame)

    def head(self, n: int = 5) -> pd.DataFrame:
        return self.frame.head(n)

    def tail(self, n: int = 5) -> pd.DataFrame:
        return self.frame.tail(n)

    def to_frame(self) -> pd.DataFrame:
        return self.frame.copy()

    def __repr__(self) -> str:  # pragma: no cover - repr sugar
        shape = self.frame.shape
        sources = ", ".join(self.source_names)
        return f"MaterializedDataset(shape={shape}, sources=[{sources}])"