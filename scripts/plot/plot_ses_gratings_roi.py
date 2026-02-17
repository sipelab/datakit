"""Plot task-movies traces for all subject/session/task entries.

The dataset can be a pickle created via pandas.DataFrame.to_pickle, an HDF5
export created via pandas.DataFrame.to_hdf, or an in-memory MultiIndex DataFrame.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datakit.config import settings
from datakit.experiment import ExperimentData
from datakit.loader import ExperimentStore


def _task_level_name(frame: pd.DataFrame) -> str:
    if isinstance(frame.index, pd.MultiIndex) and frame.index.names:
        name = frame.index.names[2]
        if name is not None:
            return str(name)
    return settings.dataset.index_names[2]


def _select_task_rows(frame: pd.DataFrame, task_filter: str, match: str) -> pd.DataFrame:
    tasks = frame.index.get_level_values(_task_level_name(frame)).astype(str)
    if match == "exact":
        mask = tasks == task_filter
    else:
        mask = tasks.str.contains(task_filter, regex=False, na=False)
    return frame.loc[mask].copy()


def _coerce_trace(array: np.ndarray, roi_index: int | None) -> np.ndarray:
    if array.ndim == 1:
        return array
    if roi_index is None:
        return np.nanmean(array, axis=0)
    return array[roi_index]


def _coerce_time(time: np.ndarray, length: int) -> np.ndarray:
    time = time.reshape(-1)
    if len(time) >= length:
        return time[:length]
    return np.arange(length, dtype=float)


def _windows_from_columns(row, start_key: tuple[str, str], stop_key: tuple[str, str]) -> list[tuple[float, float]]:
    starts = np.atleast_1d(np.asarray(row[start_key], dtype=float))
    stops = np.atleast_1d(np.asarray(row[stop_key], dtype=float))
    return [
        (float(s), float(e))
        for s, e in zip(starts, stops)
        if np.isfinite(s) and np.isfinite(e) and e > s
    ]


def _windows_from_list(row, key: tuple[str, str]) -> list[tuple[float, float]]:
    windows = []
    for item in row[key]:
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
    roi_index: int | None,
    title: str | None,
    max_plots: int | None,
    overlay_psychopy: bool,
) -> None:
    rows = frame.iloc[:max_plots] if max_plots is not None else frame
    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, max(3, 2.5 * n_rows)), sharex=False)
    if n_rows == 1:
        axes = [axes]

    trace_col = (source, trace_key)
    time_col = (source, time_key)

    for ax, (index, row) in zip(axes, rows.iterrows()):
        trace = np.asarray(row[trace_col], dtype=float)
        time = np.asarray(row[time_col], dtype=float)
        trace = _coerce_trace(trace, roi_index)
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
                ax.axvspan(
                    start,
                    stop,
                    color="#ff7f0e",
                    alpha=0.2,
                    label="grating" if idx == 0 else None,
                )

            for idx, (start, stop) in enumerate(gray):
                ax.axvspan(
                    start,
                    stop,
                    color="#7f7f7f",
                    alpha=0.15,
                    label="gray" if idx == 0 else None,
                )

        label = " | ".join(str(item) for item in index) if isinstance(index, tuple) else str(index)
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


def plot_psychopy_roi_overview(
    dataset: pd.DataFrame,
    *,
    source: str = "suite2p",
    trace_key: str = "smoothed_dff",
    time_key: str = "time_native_s",
    roi_index: int | None = None,
    task_filter: str = "task-movies",
    match: str = "contains",
    overlay_psychopy: bool = True,
    title: str | None = None,
    max_plots: int | None = None,
) -> None:
    """Plot traces for every row matching the task filter."""
    rows = _select_task_rows(dataset, task_filter, match)
    _plot_rows(
        rows,
        source=source,
        trace_key=trace_key,
        time_key=time_key,
        roi_index=roi_index,
        title=title,
        max_plots=max_plots,
        overlay_psychopy=overlay_psychopy,
    )


# ─── Pipeline-Local Source Selection ───────────────────────────────────────────
PIPELINE_TAGS = (
    "psychopy",
    "suite2p",
)


# ─── Plot Defaults (global params) ────────────────────────────────────────────
PLOT_SOURCE = "suite2p"
PLOT_TRACE_KEY = "deltaf_f"
PLOT_TIME_KEY = "time_native_s"
PLOT_ROI_INDEX: int | None = None
PLOT_TASK_FILTER = "task-gratings"
PLOT_MATCH = "contains"
PLOT_OVERLAY_PSYCHOPY = True
PLOT_TITLE: str | None = None
PLOT_INVENTORY_SLICE: slice | None = slice(30, 40)


# ─── Quick-Test Helper ─────────────────────────────────────────────────────────
def slice_inventory(frame: pd.DataFrame, entries: int | slice | tuple | list = 3) -> pd.DataFrame:
    """Return a small slice of the inventory for quick tests."""
    if isinstance(entries, int):
        return frame.iloc[:entries].copy()
    if isinstance(entries, slice):
        return frame.iloc[entries].copy()
    if isinstance(entries, tuple) and len(entries) == 2:
        start, stop = entries
        return frame.iloc[slice(start, stop)].copy()
    return frame.loc[list(entries)].copy()


def select_inventory_entry(frame: pd.DataFrame, subject: str, session: str, task: str) -> pd.DataFrame:
    """Select a single (Subject, Session, Task) entry from a MultiIndex inventory."""
    return frame.loc[[(subject, session, task)]].copy()


SUBJECT = "ACUTEVIS10"
SESSION = "ses-03"
TASK = "task-gratings"

etoH_root = Path(r"G:\Projects\ACUTEVIS").resolve()
etoH_experiment = ExperimentData(etoH_root, include_task_level=True)
sliced_inventory = slice_inventory(etoH_experiment.data, PLOT_INVENTORY_SLICE)
# sliced_inventory = select_inventory_entry(
#     etoH_experiment.data,
#     subject=SUBJECT,
#     session=SESSION,
#     task=TASK,
# )
store = ExperimentStore(sliced_inventory)
store.register_sources(PIPELINE_TAGS)

dataset = store.materialize(progress=True)


def _print_meta(row: pd.Series, key: tuple[str, str], title: str) -> None:
    print(f"\n{title}:")
    for k, v in row[key].items():
        print(f"{k}: {v}")


for (subject, session, task), group in dataset.groupby(level=[0, 1, 2]):
    if task != "task-gratings":
        continue

    print(f"\n{'=' * 60}")
    print(f"Subject: {subject} | Session: {session} | Task: {task}")
    print(f"{'=' * 60}")

    row = group.iloc[0]
    _print_meta(row, ("psychopy", "meta"), "PsychoPy Meta")
    _print_meta(row, ("suite2p", "meta"), "Suite2p Meta")

plot_psychopy_roi_overview(
    dataset,
    source=PLOT_SOURCE,
    trace_key=PLOT_TRACE_KEY,
    time_key=PLOT_TIME_KEY,
    roi_index=3,
    task_filter=PLOT_TASK_FILTER,
    match=PLOT_MATCH,
    overlay_psychopy=PLOT_OVERLAY_PSYCHOPY,
    title=PLOT_TITLE,
    max_plots=None,
)