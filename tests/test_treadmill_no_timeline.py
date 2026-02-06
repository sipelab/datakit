from pathlib import Path
import sys
import pandas as pd

# Ensure package importable
ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from datakit.sources.behavior.treadmill import TreadmillSource


def test_treadmill_load_without_timeline(tmp_path: Path):
    csv_path = tmp_path / "sample_treadmill.csv"
    df = pd.DataFrame({
        "timestamp": [0, 1_000_000, 2_000_000],
        "distance_mm": [0.0, 1.0, 2.0],
        "speed_mm": [0.0, 1.0, 1.0],
    })
    df.to_csv(csv_path, index=False)

    loader = TreadmillSource()
    stream = loader.load(csv_path)

    assert stream.tag == "treadmill"
    assert len(stream.t) == 3
    assert "distance_mm" in stream.data.columns
    assert "time_elapsed_s" in stream.data.columns
