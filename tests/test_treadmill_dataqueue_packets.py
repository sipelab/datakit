from pathlib import Path

import pandas as pd

from datakit.sources.behavior.treadmill import TreadmillSource


def _write_treadmill_csv(path: Path) -> None:
    df = pd.DataFrame(
        {
            "timestamp": [100, 200, 300],
            "distance": [0.0, 1.0, 2.0],
            "speed": [0.0, 10.0, 10.0],
        }
    )
    df.to_csv(path, index=False)


def _write_dataqueue_csv(path: Path) -> None:
    df = pd.DataFrame(
        {
            "queue_elapsed": [0.0, 0.1, 0.2],
            "device_id": ["treadmill", "treadmill", "treadmill"],
            "device_ts": ["2025-01-01 00:00:00", "2025-01-01 00:00:00.1", "2025-01-01 00:00:00.2"],
            "payload": [
                "EncoderData(timestamp=100, distance=0.000 mm, speed=0.000 mm/s)",
                "EncoderData(timestamp=200, distance=1.000 mm, speed=10.000 mm/s)",
                "EncoderData(timestamp=300, distance=2.000 mm, speed=10.000 mm/s)",
            ],
        }
    )
    df.to_csv(path, index=False)


def test_treadmill_includes_dataqueue_packets(tmp_path: Path) -> None:
    beh_dir = tmp_path / "beh"
    beh_dir.mkdir(parents=True, exist_ok=True)

    treadmill_path = beh_dir / "sample_treadmill.csv"
    dataqueue_path = beh_dir / "sample_dataqueue.csv"

    _write_treadmill_csv(treadmill_path)
    _write_dataqueue_csv(dataqueue_path)

    stream = TreadmillSource().load(treadmill_path)
    out = stream.value

    assert "dataqueue_packet" in out.columns
    packet = out["dataqueue_packet"].iloc[0]
    assert isinstance(packet, dict)
    assert packet["timestamp"].size == 3
    assert packet["distance_mm"].size == 3
