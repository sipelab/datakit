"""Plot suite2p ROI traces with pupil and wheel encoder overlays.

This mirrors the gratings ROI plot helper while adding the pupil_diameter_mm
trace from the pupil_dlc source and the speed_mm trace from the wheel encoder.
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
        return frame.index.names[2]
    return settings.dataset.index_names[2]


def _select_task_rows(frame: pd.DataFrame, task_filter: str, match: str) -> pd.DataFrame:
    tasks = frame.index.get_level_values(_task_level_name(frame)).astype(str)
    if match == "exact":
        mask = tasks == task_filter
    else:
        mask = tasks.str.contains(task_filter, regex=False, na=False)
    return frame.loc[mask].copy()


def _available_sources(frame: pd.DataFrame) -> set[str]:
    if isinstance(frame.columns, pd.MultiIndex) and frame.columns.nlevels >= 2:
        return set(frame.columns.get_level_values(0))
    return set()


def _resolve_source(frame: pd.DataFrame, preferred: str, fallbacks: Sequence[str]) -> str:
    available = _available_sources(frame)
    if preferred in available:
        return preferred
    for fallback in fallbacks:
        if fallback in available:
            return fallback
    return preferred


def _require_column(frame: pd.DataFrame, column: tuple[str, str]) -> None:
    if column not in frame.columns:
        available = sorted({col for col in frame.columns if col[0] == column[0]})
        raise KeyError(
            f"Missing column {column}. Available under '{column[0]}': {available}"
        )


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


def _plot_windows(ax, row, *, label_once: bool) -> None:
    gratings = _windows_from_columns(
        row,
        (PSYCHOPY_SOURCE, "gratings_window_grating_start"),
        (PSYCHOPY_SOURCE, "gratings_window_grating_stop"),
    )
    if not gratings:
        gratings = _windows_from_list(row, (PSYCHOPY_SOURCE, "gratings_gratings_windows"))

    gray = _windows_from_columns(
        row,
        (PSYCHOPY_SOURCE, "gratings_window_gray_start"),
        (PSYCHOPY_SOURCE, "gratings_window_gray_stop"),
    )
    if not gray:
        gray = _windows_from_list(row, (PSYCHOPY_SOURCE, "gratings_gray_windows"))

    for idx, (start, stop) in enumerate(gratings):
        ax.axvspan(
            start,
            stop,
            color="#ff7f0e",
            alpha=0.2,
            label="grating" if label_once and idx == 0 else None,
        )

    for idx, (start, stop) in enumerate(gray):
        ax.axvspan(
            start,
            stop,
            color="#7f7f7f",
            alpha=0.15,
            label="gray" if label_once and idx == 0 else None,
        )


def _plot_rows(
    frame: pd.DataFrame,
    *,
    suite2p_source: str,
    suite2p_trace_key: str,
    suite2p_time_key: str,
    pupil_source: str,
    pupil_trace_key: str,
    pupil_time_key: str,
    encoder_source: str,
    encoder_trace_key: str,
    encoder_time_key: str,
    roi_index: int | None,
    title: str | None,
    max_plots: int | None,
    overlay_psychopy: bool,
) -> None:
    rows = frame.iloc[:max_plots] if max_plots is not None else frame
    n_rows = len(rows)
    n_axes = max(1, n_rows * 3)
    fig, axes = plt.subplots(n_axes, 1, figsize=(12, max(6, 2.2 * n_axes)), sharex=False)
    if n_axes == 1:
        axes = [axes]

    suite2p_col = (suite2p_source, suite2p_trace_key)
    suite2p_time_col = (suite2p_source, suite2p_time_key)
    pupil_col = (pupil_source, pupil_trace_key)
    pupil_time_col = (pupil_source, pupil_time_key)
    encoder_col = (encoder_source, encoder_trace_key)
    encoder_time_col = (encoder_source, encoder_time_key)

    _require_column(rows, suite2p_col)
    _require_column(rows, suite2p_time_col)
    _require_column(rows, pupil_col)
    _require_column(rows, pupil_time_col)
    _require_column(rows, encoder_col)
    _require_column(rows, encoder_time_col)

    for idx, (index, row) in enumerate(rows.iterrows()):
        ax_suite2p = axes[idx * 3]
        ax_pupil = axes[idx * 3 + 1]
        ax_encoder = axes[idx * 3 + 2]

        suite2p_trace = np.asarray(row[suite2p_col], dtype=float)
        suite2p_time = np.asarray(row[suite2p_time_col], dtype=float)
        suite2p_trace = _coerce_trace(suite2p_trace, roi_index)
        suite2p_time = _coerce_time(suite2p_time, len(suite2p_trace))
        ax_suite2p.plot(suite2p_time, suite2p_trace, color="#1f77b4", linewidth=1.0)
        ax_suite2p.set_ylabel(suite2p_trace_key)
        ax_suite2p.grid(True, alpha=0.3)

        pupil_trace = np.asarray(row[pupil_col], dtype=float)
        pupil_time = np.asarray(row[pupil_time_col], dtype=float)
        pupil_time = _coerce_time(pupil_time, len(pupil_trace))
        ax_pupil.plot(pupil_time, pupil_trace, color="#2ca02c", linewidth=1.0)
        ax_pupil.set_ylabel(pupil_trace_key)
        ax_pupil.grid(True, alpha=0.3)

        encoder_trace = np.asarray(row[encoder_col], dtype=float)
        encoder_time = np.asarray(row[encoder_time_col], dtype=float)
        encoder_time = _coerce_time(encoder_time, len(encoder_trace))
        ax_encoder.plot(encoder_time, encoder_trace, color="#d62728", linewidth=1.0)
        ax_encoder.set_ylabel(encoder_trace_key)
        ax_encoder.grid(True, alpha=0.3)

        if overlay_psychopy:
            _plot_windows(ax_suite2p, row, label_once=True)
            _plot_windows(ax_pupil, row, label_once=False)
            _plot_windows(ax_encoder, row, label_once=False)

        label = " | ".join(str(item) for item in index) if isinstance(index, tuple) else str(index)
        ax_suite2p.set_title(label)
        ax_encoder.set_xlabel("Time (s)")

    if title:
        fig.suptitle(title)
    if overlay_psychopy:
        axes[0].legend(loc="upper right")
    fig.tight_layout()
    plt.show()


def plot_psychopy_roi_pupil_encoder_overview(
    dataset: pd.DataFrame,
    *,
    suite2p_source: str = "suite2p",
    suite2p_trace_key: str = "smoothed_dff",
    suite2p_time_key: str = "time_native_s",
    pupil_source: str = "pupil",
    pupil_trace_key: str = "pupil_diameter_mm",
    pupil_time_key: str = "time_elapsed_s",
    encoder_source: str = "encoder",
    encoder_trace_key: str = "speed_mm",
    encoder_time_key: str = "time_elapsed_s",
    roi_index: int | None = None,
    task_filter: str = "task-movies",
    match: str = "contains",
    overlay_psychopy: bool = True,
    title: str | None = None,
    max_plots: int | None = None,
) -> None:
    """Plot suite2p ROI traces with pupil + encoder traces for matching tasks."""
    suite2p_source = _resolve_source(dataset, suite2p_source, ())
    pupil_source = _resolve_source(dataset, pupil_source, ("pupil_dlc",))
    encoder_source = _resolve_source(dataset, encoder_source, ("wheel",))

    rows = _select_task_rows(dataset, task_filter, match)
    _plot_rows(
        rows,
        suite2p_source=suite2p_source,
        suite2p_trace_key=suite2p_trace_key,
        suite2p_time_key=suite2p_time_key,
        pupil_source=pupil_source,
        pupil_trace_key=pupil_trace_key,
        pupil_time_key=pupil_time_key,
        encoder_source=encoder_source,
        encoder_trace_key=encoder_trace_key,
        encoder_time_key=encoder_time_key,
        roi_index=roi_index,
        title=title,
        max_plots=max_plots,
        overlay_psychopy=overlay_psychopy,
    )


# ─── Pipeline-Local Source Selection ───────────────────────────────────────────
PIPELINE_TAGS = (
    "psychopy",
    "suite2p",
    "pupil_dlc",
    "wheel",
)

# ─── Plot Defaults (global params) ────────────────────────────────────────────
PLOT_SUITE2P_SOURCE = "suite2p"
PLOT_SUITE2P_TRACE_KEY = "deltaf_f"
PLOT_SUITE2P_TIME_KEY = "time_native_s"
PLOT_ROI_INDEX: int | None = None
PLOT_TASK_FILTER = "task-gratings"
PLOT_MATCH = "contains"
PLOT_OVERLAY_PSYCHOPY = True
PLOT_TITLE: str | None = None
PLOT_INVENTORY_SLICE: slice | None = slice(30, 40)

PLOT_PUPIL_SOURCE = "pupil"
PLOT_PUPIL_TRACE_KEY = "pupil_diameter_mm"
PLOT_PUPIL_TIME_KEY = "time_elapsed_s"

PLOT_ENCODER_SOURCE = "encoder"
PLOT_ENCODER_TRACE_KEY = "speed_mm"
PLOT_ENCODER_TIME_KEY = "time_elapsed_s"

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


PSYCHOPY_SOURCE = "psychopy"
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

plot_psychopy_roi_pupil_encoder_overview(
    dataset,
    suite2p_source=PLOT_SUITE2P_SOURCE,
    suite2p_trace_key=PLOT_SUITE2P_TRACE_KEY,
    suite2p_time_key=PLOT_SUITE2P_TIME_KEY,
    pupil_source=PLOT_PUPIL_SOURCE,
    pupil_trace_key=PLOT_PUPIL_TRACE_KEY,
    pupil_time_key=PLOT_PUPIL_TIME_KEY,
    encoder_source=PLOT_ENCODER_SOURCE,
    encoder_trace_key=PLOT_ENCODER_TRACE_KEY,
    encoder_time_key=PLOT_ENCODER_TIME_KEY,
    roi_index=3,
    task_filter=PLOT_TASK_FILTER,
    match=PLOT_MATCH,
    overlay_psychopy=PLOT_OVERLAY_PSYCHOPY,
    title=PLOT_TITLE,
    max_plots=None,
)
