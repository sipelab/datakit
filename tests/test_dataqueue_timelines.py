from pathlib import Path

import numpy as np
import pandas as pd

from datakit.sources.behavior.dataqueue import DataqueueSource


def _write_dataqueue(tmp_path: Path) -> Path:
    rows = pd.DataFrame(
        {
            "queue_elapsed": [0.0, 0.1, 0.2, 0.3, 0.4],
            "device_id": ["Dhyana", "Dhyana", "encoder", "Dhyana", "encoder"],
            "device_ts": [1_000.0, 1_100.0, 50.0, 1_200.0, 60.0],
            "payload": ["a", "b", "c", "d", "e"],
        }
    )
    path = tmp_path / "sample_dataqueue.csv"
    rows.to_csv(path, index=False)
    return path


def test_dataqueue_builds_master_aligned_timelines(tmp_path: Path) -> None:
    path = _write_dataqueue(tmp_path)
    stream = DataqueueSource().load(path)

    assert stream.tag == "dataqueue"
    summary = stream.value
    assert isinstance(summary, pd.DataFrame)
    assert len(summary) == 1

    row = summary.iloc[0]
    master = row["master_elapsed_s"]
    device_elapsed = row["device_elapsed"]
    device_ts = row["device_ts"]
    device_aligned = row["device_aligned_abs"]

    assert set(device_elapsed.keys()) == {"Dhyana", "encoder"}
    assert set(device_ts.keys()) == {"Dhyana", "encoder"}
    # The master timeline matches the Dhyana rows
    np.testing.assert_allclose(master, np.array([0.0, 0.1, 0.3]))

    encoder_elapsed = device_elapsed["encoder"]
    assert encoder_elapsed.size == 2
    assert np.all(np.diff(encoder_elapsed) >= 0)

    assert "Dhyana" in device_aligned
    assert device_aligned["Dhyana"].size == master.size

    dhyana_rate = row["device_sample_rate_hz"]["Dhyana"]
    assert dhyana_rate > 0
