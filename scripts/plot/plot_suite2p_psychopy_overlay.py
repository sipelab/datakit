"""Plot Suite2p ROI trace with Psychopy grating/gray overlays.

Usage:
  python scripts/plot/plot_suite2p_psychopy_overlay.py \
    --experiment-dir <path> --subject <sub> --session <ses> --task <task> \
    --roi-index 0 --trace smoothed_dff

If --experiment-dir is omitted, the debug sample experiment is used.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

# Ensure the package root is importable when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_PARENT = PROJECT_ROOT.parent
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from datakit.config import settings
from datakit.experiment import ExperimentData
from datakit.loader import ExperimentStore


TRACE_TIME_MAP = {
    "roi_fluorescence": "time_native_s",
    "neuropil_fluorescence": "time_native_s",
    "deltaf_f": "time_native_s",
    "interp_deltaf_f": "time_elapsed_s",
    "smoothed_dff": "time_elapsed_s",
}


def _select_inventory_entry(frame, subject: str, session: str, task: str):
    key = (subject, session, task)
    return frame.loc[[key]].copy()


def _require_column(frame, column: tuple[str, str]) -> None:
    if column not in frame.columns:
        available = sorted({col for col in frame.columns if col[0] == column[0]})
        raise KeyError(f"Missing column {column}. Available under '{column[0]}': {available}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Suite2p ROI trace with Psychopy overlays.")
    parser.add_argument("--experiment-dir", type=Path, default=None)
    parser.add_argument("--subject", type=str, default=None)
    parser.add_argument("--session", type=str, default=None)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--roi-index", type=int, default=0)
    parser.add_argument(
        "--trace",
        type=str,
        default="smoothed_dff",
        choices=sorted(TRACE_TIME_MAP.keys()),
        help="Suite2p trace to plot.",
    )
    parser.add_argument("--title", type=str, default=None)
    return parser.parse_args()


def _to_windows(start: Iterable[float], stop: Iterable[float]) -> list[tuple[float, float]]:
    windows = []
    for s, e in zip(start, stop):
        if np.isfinite(s) and np.isfinite(e) and e > s:
            windows.append((float(s), float(e)))
    return windows


def _windows_from_columns(row, start_key: tuple[str, str], stop_key: tuple[str, str]) -> list[tuple[float, float]]:
    if start_key not in row.index or stop_key not in row.index:
        return []
    starts = np.asarray(row[start_key], dtype=float)
    stops = np.asarray(row[stop_key], dtype=float)
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


def _resolve_selection(experiment: ExperimentData, args: argparse.Namespace) -> tuple[str, str, str]:
    inventory = experiment.data
    if inventory.empty:
        raise ValueError("Experiment inventory is empty")

    if args.subject and args.session and args.task:
        return args.subject, args.session, args.task

    default_key = inventory.index[0]
    subject = args.subject or default_key[0]
    session = args.session or default_key[1]
    task = args.task or default_key[2]
    return subject, session, task


def main() -> None:
    args = _parse_args()

    if args.experiment_dir is None:
        experiment_root = PROJECT_ROOT.joinpath(*settings.debug.sample_experiment_rel)
    else:
        experiment_root = args.experiment_dir

    experiment_root = experiment_root.resolve()

    experiment = ExperimentData(experiment_root, include_task_level=True)
    subject, session, task = _resolve_selection(experiment, args)

    inventory = _select_inventory_entry(experiment.data, subject, session, task)
    store = ExperimentStore(inventory)
    missing = store.register_sources(("suite2p", "psychopy"))
    if missing:
        raise ValueError(f"Missing sources in inventory: {missing}")

    dataset = store.materialize()
    if dataset.empty:
        raise ValueError("Materialized dataset is empty")

    row = dataset.iloc[0]

    trace_key = ("suite2p", args.trace)
    time_key = ("suite2p", TRACE_TIME_MAP[args.trace])
    _require_column(dataset, trace_key)
    _require_column(dataset, time_key)

    trace = np.asarray(row[trace_key], dtype=float)
    time = np.asarray(row[time_key], dtype=float)

    if trace.ndim != 2:
        raise ValueError(f"Expected 2D ROI traces for {trace_key}, got shape {trace.shape}")
    if args.roi_index < 0 or args.roi_index >= trace.shape[0]:
        raise IndexError(f"ROI index {args.roi_index} out of range for {trace.shape[0]} ROIs")

    roi_trace = trace[args.roi_index]

    min_len = min(len(time), len(roi_trace))
    if min_len == 0:
        raise ValueError("Trace or time array is empty")
    time = time[:min_len]
    roi_trace = roi_trace[:min_len]

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

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(time, roi_trace, color="#1f77b4", linewidth=1.0, label=f"ROI {args.roi_index}")

    for idx, (start, stop) in enumerate(gratings):
        label = "grating" if idx == 0 else None
        ax.axvspan(start, stop, color="#ff7f0e", alpha=0.2, label=label)

    for idx, (start, stop) in enumerate(gray):
        label = "gray" if idx == 0 else None
        ax.axvspan(start, stop, color="#7f7f7f", alpha=0.15, label=label)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(args.trace)
    ax.grid(True, alpha=0.3)

    if args.title:
        title = args.title
    else:
        title = f"Suite2p {args.trace} | ROI {args.roi_index}\n{subject} | {session} | {task}"
    ax.set_title(title)
    ax.legend(loc="upper right")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
