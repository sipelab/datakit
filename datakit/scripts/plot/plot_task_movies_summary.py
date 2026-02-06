"""Plot task-movies traces for all subject/session/task entries.

The dataset can be a pickle created via pandas.DataFrame.to_pickle, an HDF5
export created via pandas.DataFrame.to_hdf, or an in-memory MultiIndex DataFrame.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure the package root is importable when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_PARENT = PROJECT_ROOT.parent
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from datakit.config import settings


def _load_dataset(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    if suffix in {".h5", ".hdf5"}:
        return pd.read_hdf(path)
    raise ValueError(f"Unsupported dataset file type: {path}")


def _task_level_name(frame: pd.DataFrame) -> str:
    if isinstance(frame.index, pd.MultiIndex):
        if frame.index.names and len(frame.index.names) >= 3:
            return frame.index.names[2]
    return settings.dataset.index_names[2]


def _select_task_rows(frame: pd.DataFrame, task_filter: str, match: str) -> pd.DataFrame:
    if not isinstance(frame.index, pd.MultiIndex):
        raise ValueError("Dataset must have a MultiIndex index.")

    level_name = _task_level_name(frame)
    tasks = frame.index.get_level_values(level_name)
    tasks = tasks.astype(str)

    if match == "exact":
        mask = tasks == task_filter
    else:
        mask = tasks.str.contains(task_filter, regex=False, na=False)
    return frame.loc[mask].copy()


def _coerce_trace(array: np.ndarray) -> np.ndarray:
    if array.ndim == 1:
        return array
    if array.ndim == 2:
        return np.nanmean(array, axis=0)
    raise ValueError(f"Unsupported trace array shape {array.shape}")


def _coerce_time(time: np.ndarray, length: int) -> np.ndarray:
    if time.ndim != 1:
        time = time.reshape(-1)
    if len(time) >= length:
        return time[:length]
    return np.arange(length, dtype=float)


def _to_windows(start: Iterable[float], stop: Iterable[float]) -> list[tuple[float, float]]:
    windows = []
    for s, e in zip(start, stop):
        if np.isfinite(s) and np.isfinite(e) and e > s:
            windows.append((float(s), float(e)))
    return windows


def _windows_from_columns(row, start_key: tuple[str, str], stop_key: tuple[str, str]) -> list[tuple[float, float]]:
    if start_key not in row.index or stop_key not in row.index:
        return []
    starts = np.atleast_1d(np.asarray(row[start_key], dtype=float))
    stops = np.atleast_1d(np.asarray(row[stop_key], dtype=float))
    return _to_windows(starts, stops)


def _windows_from_list(row, key: tuple[str, str]) -> list[tuple[float, float]]:
    if key not in row.index:
        return []
    data = row[key]
    if not isinstance(data, Sequence):
        return []
    windows = []
    for item in data:
        if isinstance(item, Sequence) and len(item) >= 2:
            s, e = item[0], item[1]
            if np.isfinite(s) and np.isfinite(e) and e > s:
                windows.append((float(s), float(e)))
    return windows


def _plot_rows(
    frame: pd.DataFrame,
    *,
    source: str,
    trace_key: str,
    time_key: str,
    title: str | None,
    max_plots: int | None,
    overlay_psychopy: bool,
) -> None:
    if frame.empty:
        raise ValueError("No rows matched the task filter.")

    rows = frame
    if max_plots is not None:
        rows = rows.iloc[:max_plots]

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, max(3, 2.5 * n_rows)), sharex=False)
    if n_rows == 1:
        axes = [axes]

    for ax, (index, row) in zip(axes, rows.iterrows()):
        trace_col = (source, trace_key)
        time_col = (source, time_key)
        if trace_col not in rows.columns:
            raise KeyError(f"Missing column {trace_col} in dataset.")
        if time_col not in rows.columns:
            raise KeyError(f"Missing column {time_col} in dataset.")

        trace = np.asarray(row[trace_col], dtype=float)
        time = np.asarray(row[time_col], dtype=float)
        trace = _coerce_trace(trace)
        time = _coerce_time(time, len(trace))

        ax.plot(time, trace, color="#1f77b4", linewidth=1.0)

        if overlay_psychopy:
            gratings = _windows_from_columns(
                row,
                ("psychopy", "gratings_window_grating_start"),
                ("psychopy", "gratings_window_grating_stop"),
            )
            if not gratings:
                gratings = _windows_from_list(row, ("psychopy", "gratings_gratings_windows"))

            gray = _windows_from_columns(
                row,
                ("psychopy", "gratings_window_gray_start"),
                ("psychopy", "gratings_window_gray_stop"),
            )
            if not gray:
                gray = _windows_from_list(row, ("psychopy", "gratings_gray_windows"))

            for idx, (start, stop) in enumerate(gratings):
                label = "grating" if idx == 0 else None
                ax.axvspan(start, stop, color="#ff7f0e", alpha=0.2, label=label)

            for idx, (start, stop) in enumerate(gray):
                label = "gray" if idx == 0 else None
                ax.axvspan(start, stop, color="#7f7f7f", alpha=0.15, label=label)
        if isinstance(index, Iterable):
            label = " | ".join(str(item) for item in index)
        else:
            label = str(index)
        ax.set_title(label)
        ax.set_ylabel(trace_key)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    if title:
        fig.suptitle(title)
    if overlay_psychopy:
        axes[0].legend(loc="upper right")
    fig.tight_layout()
    plt.show()


def plot_task_movies(
    dataset: pd.DataFrame | str | Path,
    *,
    source: str = "suite2p",
    trace_key: str = "smoothed_dff",
    time_key: str = "time_elapsed_s",
    task_filter: str = "task-movies",
    match: str = "contains",
    overlay_psychopy: bool = True,
    title: str | None = None,
    max_plots: int | None = None,
) -> None:
    """Plot traces for every row matching the task filter."""
    if isinstance(dataset, (str, Path)):
        dataset = _load_dataset(Path(dataset))
    rows = _select_task_rows(dataset, task_filter, match)
    _plot_rows(
        rows,
        source=source,
        trace_key=trace_key,
        time_key=time_key,
        title=title,
        max_plots=max_plots,
        overlay_psychopy=overlay_psychopy,
    )
