from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = REPO_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from datakit.config import settings
from datakit.sources.analysis.mesomap import MesoMapSource


TRACE_NAME = "20251215_142459_sub-GS26_ses-01_task-spont_mesoscope.ome_traces.csv"
TRACE_PATH = REPO_ROOT / "tests" / "mesomap" / TRACE_NAME


def test_mesomap_loader_parses_traces_and_metadata():
    loader = MesoMapSource()
    stream = loader.load(TRACE_PATH)

    df = stream.data
    assert len(stream.t) == len(df) == 58799
    assert np.array_equal(stream.t, df[loader.frame_column].to_numpy(dtype=np.float64))

    assert stream.meta["scope"] == settings.dataset.session_scope
    assert stream.meta["time_basis"] == loader.time_basis

    assert stream.meta["n_rois"] == 44
    assert not stream.meta["roi_missing_region_metadata"]
    mapping = stream.meta["roi_to_mask_label"]
    assert len(mapping) == 44
    for col in df.columns:
        if col == loader.frame_column:
            continue
        assert col in mapping

    assert stream.meta["mask_missing"] is False
    assert stream.meta["mask_shape"] == (512, 512)
    assert stream.meta["region_count"] == 64
