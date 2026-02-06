"""Helpers for aligning source streams onto a shared time base."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Tuple

import numpy as np
import pandas as pd

TimeKey = Tuple[str, str]


@dataclass(frozen=True)
class AlignmentResult:
    """Aligned values plus optional per-source offsets."""

    time_axis: np.ndarray
    values: Dict[str, np.ndarray]
    offsets: Dict[str, float]


def _dedupe_sorted(t_source: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if t_source.size == 0:
        return t_source, values
    _, unique_idx = np.unique(t_source, return_index=True)
    return t_source[unique_idx], values[unique_idx]


def _maybe_shift_timebase(
    t_source: np.ndarray,
    time_axis: np.ndarray,
    *,
    threshold_s: float,
) -> tuple[np.ndarray, float]:
    if t_source.size == 0 or time_axis.size == 0:
        return t_source, 0.0
    offset = float(t_source[0] - time_axis[0])
    if abs(offset) >= threshold_s:
        return t_source - offset, offset
    return t_source, 0.0


def _interp_trace(
    t_source: np.ndarray,
    values: np.ndarray,
    time_axis: np.ndarray,
    *,
    method: str,
) -> np.ndarray:
    if method == "previous":
        indices = np.searchsorted(t_source, time_axis, side="right") - 1
        indices = np.clip(indices, 0, len(values) - 1)
        return values[indices]
    return np.interp(time_axis, t_source, values)


def align_row(
    entry: pd.Series,
    sources: Mapping[str, str],
    *,
    time_key: TimeKey = ("time", "master_elapsed_s"),
    time_column: str = "time_elapsed_s",
    methods: Optional[Mapping[str, str]] = None,
    default_method: str = "linear",
    align_offset: bool = False,
    offset_threshold_s: float = 0.5,
    keep_missing: bool = False,
) -> AlignmentResult:
    """Align multiple source features onto a shared time axis.

    Parameters
    ----------
    entry
        Dataset row (MultiIndex columns).
    sources
        Mapping of source tag -> feature column name.
    time_key
        Column key that defines the shared time axis.
    time_column
        Column name for per-source time arrays.
    methods
        Optional per-source interpolation methods ("linear" or "previous").
    default_method
        Fallback method for sources not in ``methods``.
    align_offset
        If True, shift per-source time bases by a constant offset to match
        the shared time axis (based on first sample).
    offset_threshold_s
        Minimum absolute offset (seconds) to apply when ``align_offset`` is True.
    keep_missing
        If True, include missing sources with empty arrays.
    """

    if time_key not in entry.index:
        raise KeyError(f"Missing shared time axis column {time_key}")

    time_axis = np.asarray(entry[time_key], dtype=float)
    method_map = dict(methods or {})

    aligned: Dict[str, np.ndarray] = {}
    offsets: Dict[str, float] = {}

    for source, feature in sources.items():
        time_idx = (source, time_column)
        value_idx = (source, feature)
        if time_idx not in entry.index or value_idx not in entry.index:
            if keep_missing:
                aligned[source] = np.array([], dtype=float)
            continue

        t_source = np.asarray(entry[time_idx], dtype=float)
        values = np.asarray(entry[value_idx], dtype=float)
        if t_source.size == 0 or values.size == 0:
            if keep_missing:
                aligned[source] = np.array([], dtype=float)
            continue
        if t_source.size != values.size:
            raise ValueError(f"{source} time/value arrays mismatch")

        order = np.argsort(t_source)
        t_source = t_source[order]
        values = values[order]
        t_source, values = _dedupe_sorted(t_source, values)

        if align_offset:
            t_source, offset = _maybe_shift_timebase(
                t_source,
                time_axis,
                threshold_s=offset_threshold_s,
            )
            offsets[source] = offset

        method = method_map.get(source, default_method)
        aligned[source] = _interp_trace(t_source, values, time_axis, method=method)

    return AlignmentResult(time_axis=time_axis, values=aligned, offsets=offsets)
