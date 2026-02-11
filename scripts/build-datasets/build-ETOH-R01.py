from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from datakit.experiment import ExperimentData
from datakit.loader import ExperimentStore

import numpy as np
from IPython import embed
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
)

def slice_inventory(frame: pd.DataFrame, entries: Any = 3) -> pd.DataFrame:
	"""
	Return a small slice of the inventory for quick tests.
	"""
	if isinstance(entries, int):
		return frame.iloc[:entries].copy()
	return frame.loc[list(entries)].copy()

# Build dataset for F:\251215_ETOH_RO1
etoH_root = Path(r"F:\251215_ETOH_RO1").resolve()
etoH_experiment = ExperimentData(etoH_root, include_task_level=True)
# sliced_inventory = etoH_experiment.data.drop(index=[('GS26','ses-00','task-widefield'),
#                                                     ('GS27', 'ses-11', 'task-violet'),])
# #                                                    ('GS29', 'ses-04', 'task-movies')])
#sliced_inventory = slice_inventory(sliced_inventory, entries=3)
store = ExperimentStore(etoH_experiment.data)
store.register_sources(PIPELINE_TAGS)

dataset = store.materialize(progress=True)

#embed()

dataset.to_pickle(etoH_root / "pickle jar" / "260211_ETOH_dataset.pkl")