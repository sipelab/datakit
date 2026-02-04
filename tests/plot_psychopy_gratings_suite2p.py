"""Plot a Suite2p ROI trace with Psychopy gratings shading (sample_experiment3)."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

REPO_PARENT = Path(__file__).resolve().parents[2]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from datakit.config import settings
from datakit.experiment import ExperimentData
from datakit.loader import ExperimentStore


def _sample_root() -> Path:
    return Path(__file__).resolve().parents[1].joinpath(*settings.debug.sample_experiment_rel)


def _select_gratings_inventory(inventory):
    tasks = inventory.index.get_level_values("Task")
    mask = tasks == "task-gratings"
    if mask.any():
        return inventory.loc[mask].copy()
    return inventory.iloc[:1].copy()


def _first_available_row(dataset):
    if dataset.empty:
        raise RuntimeError("Dataset is empty after filtering.")
    return dataset.iloc[0]


def _get_suite2p_trace(row):
    candidates = ["smoothed_dff", "interp_deltaf_f", "deltaf_f", "roi_fluorescence"]
    for name in candidates:
        key = ("suite2p", name)
        if key in row.index:
            values = row[key]
            if isinstance(values, np.ndarray) and values.ndim >= 1:
                return values, name
    raise KeyError("No Suite2p trace found in dataset row.")


def _get_suite2p_time(row, trace_name: str, n_frames: int) -> np.ndarray:
    time_key = ("suite2p", "time_elapsed_s")
    
    # Always try time_elapsed_s first (works for both v1.0 and v2.0)
    if time_key in row.index:
        time = row[time_key]
        if isinstance(time, np.ndarray) and time.size == n_frames:
            return time.astype(np.float64)

    return np.arange(n_frames, dtype=np.float64)


def _get_psychopy_windows(row) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    gratings_key = ("psychopy", "gratings_gratings_windows")
    gray_key = ("psychopy", "gratings_gray_windows")

    gratings_windows = row[gratings_key] if gratings_key in row.index else []
    gray_windows = row[gray_key] if gray_key in row.index else []

    return gratings_windows, gray_windows


def main() -> None:
    root = _sample_root()
    experiment = ExperimentData(root, include_task_level=True)
    inventory = _select_gratings_inventory(experiment.data)

    store = ExperimentStore(inventory)
    store.register_source("suite2p", inventory["suite2p"], version="2.1")
    store.register_sources(("psychopy", "dataqueue"))
    dataset = store.materialize()

    row = _first_available_row(dataset)

    trace, trace_name = _get_suite2p_trace(row)
    if trace.ndim == 2:
        trace = trace[3]
    n_frames = int(trace.shape[-1])
    time = _get_suite2p_time(row, trace_name, n_frames)

    gratings_windows, gray_windows = _get_psychopy_windows(row)
    if not gratings_windows and not gray_windows:
        print("No gratings or gray-screen times found in psychopy data.")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time, trace, color="black", linewidth=1.0, label=f"Suite2p {trace_name}")

    for start, stop in gratings_windows:
        ax.axvspan(start, stop, color="#f59e0b", alpha=0.25, linewidth=0, label="gratings" if start == gratings_windows[0][0] else None)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("dF/F")
    ax.set_title("Suite2p ROI with Psychopy gratings shading")
    ax.legend(loc="upper right")

    output_path = Path("logs") / "psychopy_suite2p_gratings.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    #fig.savefig(output_path, dpi=150)
    print(f"Saved plot to: {output_path.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()
