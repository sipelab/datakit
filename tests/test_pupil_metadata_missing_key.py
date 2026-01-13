from pathlib import Path
import json
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = REPO_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from datakit.sources.camera.pupil import PupilMetadataSource


def test_pupil_metadata_falls_back_when_key_missing(tmp_path: Path):
    # Craft JSON without the expected 'p0' key but with a single alternative entry.
    payload = {
        "p1": [
            {
                "camera_device": "pupil_cam",
                "camera_metadata": {"ElapsedTime-ms": 0.0},
                "ElapsedTime-ms": 0.0,
            },
            {
                "camera_device": "pupil_cam",
                "camera_metadata": {"ElapsedTime-ms": 10.0},
                "ElapsedTime-ms": 10.0,
            },
        ]
    }
    json_path = tmp_path / "sample_pupil.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    loader = PupilMetadataSource()
    stream = loader.load(json_path)

    assert stream.meta["json_entry_key"] == "p1"
    assert len(stream.t) == 2
    assert isinstance(stream.data, pd.DataFrame)
