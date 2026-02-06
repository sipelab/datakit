import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(r"c:\dev").resolve()))

from datakit.experiment import ExperimentData
from datakit.loader import ExperimentStore
from datakit.config import settings

root = Path(r"c:\dev\datakit").joinpath(*settings.debug.sample_experiment_rel)
experiment = ExperimentData(root, include_task_level=True)
inventory = experiment.data
store = ExperimentStore(inventory)
store.register_sources(("psychopy", "dataqueue"))
dataset = store.materialize()
row = dataset.iloc[0]

keys = [
    ("psychopy", "gratings_gratings_windows"),
    ("psychopy", "gratings_gray_windows"),
]
print("keys", [k for k in keys if k in row.index])
for key in keys:
    if key in row.index:
        value = row[key]
        print(key, type(value))
        if isinstance(value, (list, tuple)):
            print("length", len(value))
            if len(value) > 0:
                first = value[0]
                print("first", type(first), first)
                if isinstance(first, (list, tuple, np.ndarray)):
                    print("first_len", len(first))
        else:
            print("value", value)
