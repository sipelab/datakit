"""Overlay pupil diameter, wheel speed, and Suite2p ROI for gratings sessions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datakit.config import settings
from datakit.experiment import ExperimentData
from datakit.loader import ExperimentStore

PSYCHOPY_SOURCE = "psychopy"


def _task_level_name(frame: pd.DataFrame) -> str:
    if isinstance(frame.index, pd.MultiIndex) and frame.index.names:
        name = frame.index.names[2]
        if name is not None:
            return str(name)
    return settings.dataset.index_names[2]


def _select_task_rows(frame: pd.DataFrame, task_filter: str, match: str) -> pd.DataFrame:
    tasks = frame.index.get_level_values(_task_level_name(frame)).astype(str)
    if match == "exact":
        return frame.loc[tasks == task_filter].copy()
    return frame.loc[tasks.str.contains(task_filter, regex=False, na=False)].copy()


def _windows_from_columns(row, start_key: tuple[str, str], stop_key: tuple[str, str]) -> list[tuple[float, float]]:
    starts = np.atleast_1d(np.asarray(row[start_key], dtype=float))
    stops = np.atleast_1d(np.asarray(row[stop_key], dtype=float))
    return [
        (float(s), float(e))
        for s, e in zip(starts, stops)
        if np.isfinite(s) and np.isfinite(e) and e > s
    ]


def _windows_from_list(row, key: tuple[str, str]) -> list[tuple[float, float]]:
    return [
        (float(item[0]), float(item[1]))
        for item in row[key]
        if isinstance(item, (list, tuple))
        and len(item) >= 2
        and np.isfinite(item[0])
        and np.isfinite(item[1])
        and item[1] > item[0]
    ]


def _plot_windows(ax, row, *, label_once: bool) -> None:
    gratings = _windows_from_columns(
        row,
        (PSYCHOPY_SOURCE, "gratings_window_grating_start"),
        (PSYCHOPY_SOURCE, "gratings_window_grating_stop"),
    ) or _windows_from_list(row, (PSYCHOPY_SOURCE, "gratings_gratings_windows"))

    gray = _windows_from_columns(
        row,
        (PSYCHOPY_SOURCE, "gratings_window_gray_start"),
        (PSYCHOPY_SOURCE, "gratings_window_gray_stop"),
    ) or _windows_from_list(row, (PSYCHOPY_SOURCE, "gratings_gray_windows"))

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


def plot_pupil_encoder_overlay_overview(
    dataset: pd.DataFrame,
    *,
    pupil_source: str,
    pupil_trace_key: str,
    pupil_time_key: str,
    encoder_source: str,
    encoder_trace_key: str,
    encoder_time_key: str,
    roi_source: str,
    roi_trace_key: str,
    roi_time_key: str,
    task_filter: str,
    match: str,
    overlay_psychopy: bool,
    title: str | None,
    max_plots: int | None,
) -> None:
    """Plot pupil + wheel encoder + Suite2p ROI overlay for matching tasks."""
    rows = _select_task_rows(dataset, task_filter, match)
    rows = rows.iloc[:max_plots] if max_plots is not None else rows
    n_rows = len(rows)

    fig, axes = plt.subplots(n_rows, 1, figsize=(12, max(4, 3.0 * n_rows)), sharex=False)
    if n_rows == 1:
        axes = [axes]

    pupil_col = (pupil_source, pupil_trace_key)
    pupil_time_col = (pupil_source, pupil_time_key)
    encoder_col = (encoder_source, encoder_trace_key)
    encoder_time_col = (encoder_source, encoder_time_key)
    roi_col = (roi_source, roi_trace_key)
    roi_time_col = (roi_source, roi_time_key)

    for idx, (index, row) in enumerate(rows.iterrows()):
        ax = axes[idx]

        pupil_trace = np.asarray(row[pupil_col], dtype=float)
        pupil_time = np.asarray(row[pupil_time_col], dtype=float).reshape(-1)
        encoder_trace = np.asarray(row[encoder_col], dtype=float)
        encoder_time = np.asarray(row[encoder_time_col], dtype=float).reshape(-1)
        roi_trace = np.asarray(row[roi_col][3], dtype=float)
        roi_time = np.asarray(row[roi_time_col], dtype=float).reshape(-1)

        ax.plot(pupil_time, pupil_trace, color="#2ca02c", label="pupil (mm)")
        ax.plot(roi_time, roi_trace, color="#1f77b4", label="suite2p ROI")
        ax.set_ylabel(pupil_trace_key, color="#2ca02c")
        ax.tick_params(axis="y", labelcolor="#2ca02c")
        ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()
        ax2.plot(encoder_time, encoder_trace, color="#d62728", label="speed (mm/s)")
        ax2.set_ylabel(encoder_trace_key, color="#d62728")
        ax2.tick_params(axis="y", labelcolor="#d62728")

        if overlay_psychopy:
            _plot_windows(ax, row, label_once=(idx == 0))

        label = " | ".join(str(item) for item in index) if isinstance(index, tuple) else str(index)
        ax.set_title(label)
        ax.set_xlabel("Time (s)")

        if idx == 0:
            ax.legend(loc="upper right")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    plt.show()


PIPELINE_TAGS = (
    "psychopy",
    "pupil_dlc",
    "pupil_metadata",
    "wheel",
    "suite2p",
)

PLOT_TASK_FILTER = "task-gratings"
PLOT_MATCH = "contains"
PLOT_OVERLAY_PSYCHOPY = True
PLOT_TITLE: str | None = None
PLOT_INVENTORY_SLICE: slice | None = slice(10, 40)

PLOT_PUPIL_SOURCE = "pupil"
PLOT_PUPIL_TRACE_KEY = "pupil_diameter_mm"
PLOT_PUPIL_TIME_KEY = "time_elapsed_s"

PLOT_ENCODER_SOURCE = "encoder"
PLOT_ENCODER_TRACE_KEY = "speed_mm"
PLOT_ENCODER_TIME_KEY = "time_elapsed_s"

PLOT_ROI_SOURCE = "suite2p"
PLOT_ROI_TRACE_KEY = "deltaf_f"
PLOT_ROI_TIME_KEY = "time_elapsed_s"

etoH_root = Path(r"G:\Projects\ACUTEVIS").resolve()
etoH_experiment = ExperimentData(etoH_root, include_task_level=True)
sliced_inventory = etoH_experiment.data.iloc[PLOT_INVENTORY_SLICE].copy()

store = ExperimentStore(sliced_inventory)
store.register_sources(PIPELINE_TAGS)

dataset = store.materialize(progress=True)

plot_pupil_encoder_overlay_overview(
    dataset,
    pupil_source=PLOT_PUPIL_SOURCE,
    pupil_trace_key=PLOT_PUPIL_TRACE_KEY,
    pupil_time_key=PLOT_PUPIL_TIME_KEY,
    encoder_source=PLOT_ENCODER_SOURCE,
    encoder_trace_key=PLOT_ENCODER_TRACE_KEY,
    encoder_time_key=PLOT_ENCODER_TIME_KEY,
    roi_source=PLOT_ROI_SOURCE,
    roi_trace_key=PLOT_ROI_TRACE_KEY,
    roi_time_key=PLOT_ROI_TIME_KEY,
    task_filter=PLOT_TASK_FILTER,
    match=PLOT_MATCH,
    overlay_psychopy=PLOT_OVERLAY_PSYCHOPY,
    title=PLOT_TITLE,
    max_plots=None,
)
