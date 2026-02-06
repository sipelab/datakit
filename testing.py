from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from datakit.experiment import ExperimentData
from datakit.loader import ExperimentStore

import numpy as np
import matplotlib.pyplot as plt


# ─── Pipeline-Local Source Selection ───────────────────────────────────────────
PIPELINE_TAGS = (
    "mesomap",
    "timestamps",
    "dataqueue",
    "treadmill",
    "wheel",
    "notes",
    "session_config",
    "meso_metadata",
    "pupil_metadata",
    "pupil_dlc",
    "psychopy",
    "suite2p",
)

PIPELINE_VERSIONS = {"treadmill": "3.0", "psychopy": "3.1", "suite2p": "2.1"}



# ─── Quick-Test Helper ───────────────────────────────────────────────────────────
def slice_inventory(frame: pd.DataFrame, entries: Any = 3) -> pd.DataFrame:
	"""
	Return a small slice of the inventory for quick tests.
	"""
	if isinstance(entries, int):
		return frame.iloc[:entries].copy()
	return frame.loc[list(entries)].copy()


def select_inventory_entry(frame: pd.DataFrame, subject: str, session: str, task: str) -> pd.DataFrame:
	"""
	Select a single (Subject, Session, Task) entry from a MultiIndex inventory.
	"""
	key = (subject, session, task)
	return frame.loc[[key]].copy()


etoH_root = Path(r"G:\Projects\ACUTEVIS").resolve()
etoH_experiment = ExperimentData(etoH_root, include_task_level=True)
#sliced_inventory = etoH_experiment.data
sliced_inventory = select_inventory_entry(etoH_experiment.data, 
                                          subject="ACUTEVIS06", 
                                          session="ses-02", 
                                          task="task-gratings")
store = ExperimentStore(sliced_inventory)
store.register_sources(PIPELINE_TAGS, versions=PIPELINE_VERSIONS)

dataset = store.materialize(progress=True)

print(dataset.psychopy.meta['ACUTEVIS06']['ses-02']['task-gratings'])