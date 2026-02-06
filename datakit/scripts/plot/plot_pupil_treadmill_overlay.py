"""Test script: load treadmill + pupil and plot both traces.

Usage:
  python scripts/plot/plot_pupil_treadmill_overlay.py --experiment-dir <path> --row 0

If --experiment-dir is omitted, the sample experiment fixture is used.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Ensure the package root is importable when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_PARENT = PROJECT_ROOT.parent
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from datakit.experiment import ExperimentData
from datakit.loader import ExperimentStore


def _select_inventory_entry(frame, subject: str, session: str, task: str):
    key = (subject, session, task)
    return frame.loc[[key]].copy()


def _require_column(frame, column: tuple[str, str]) -> None:
    if column not in frame.columns:
        available = sorted({col for col in frame.columns if col[0] == column[0]})
        raise KeyError(
            f"Missing column {column}. Available under '{column[0]}': {available}"
        )


def main() -> None:
    experiment_root = Path(r"F:\251215_ETOH_RO1").resolve()
    subject = "GS28"
    session = "ses-01"
    task = "task-spont"
    experiment = ExperimentData(experiment_root, include_task_level=True)
    inventory = _select_inventory_entry(experiment.data, subject, session, task)

    store = ExperimentStore(inventory)
    missing = store.register_sources(("treadmill", "pupil_dlc"), versions={"treadmill": "2.1"})
    if missing:
        raise ValueError(f"Missing sources in inventory: {missing}")

    dataset = store.materialize()
    if dataset.empty:
        raise ValueError("Materialized dataset is empty")

    row_index = 0
    if row_index < 0 or row_index >= len(dataset):
        raise IndexError(f"Row index {row_index} is out of range for dataset length {len(dataset)}")

    _require_column(dataset, ("treadmill", "time_elapsed_s"))
    _require_column(dataset, ("treadmill", "speed_mm"))
    _require_column(dataset, ("pupil", "time_elapsed_s"))
    _require_column(dataset, ("pupil", "pupil_diameter_mm"))

    row = dataset.iloc[row_index]
    treadmill_t = np.asarray(row[("treadmill", "time_elapsed_s")], dtype=float)
    treadmill_speed = np.asarray(row[("treadmill", "speed_mm")], dtype=float)
    pupil_t = np.asarray(row[("pupil", "time_elapsed_s")], dtype=float)
    pupil_diameter = np.asarray(row[("pupil", "pupil_diameter_mm")], dtype=float)

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(treadmill_t, treadmill_speed, color="#1f77b4", label="Locomotion speed (mm/s)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Speed (mm/s)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(pupil_t, pupil_diameter, color="#ff7f0e", label="Pupil diameter (mm)", alpha=0.85)
    ax2.set_ylabel("Pupil diameter (mm)", color="#ff7f0e")
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")

    index_label = " | ".join(map(str, dataset.index[row_index]))
    fig.suptitle(f"Pupil + treadmill traces\n{index_label}")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
