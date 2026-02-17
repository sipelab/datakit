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


# Build dataset for F:\251215_ETOH_RO1
etoH_root = Path(r"G:\Projects\ACUTEVIS").resolve()
etoH_experiment = ExperimentData(etoH_root, include_task_level=True)
sliced_inventory = etoH_experiment.data.drop(index=[('ACUTEVIS15','ses-02','task-gratings')])
store = ExperimentStore(sliced_inventory)
store.register_sources(PIPELINE_TAGS)

dataset = store.materialize(progress=True)

dataset.to_pickle(etoH_root / "picklejar" / "260211_ACUTEVIS_dataset.pkl")