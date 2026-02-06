"""Session timing markers data source.

Transforms a CSV of per-device ``started``/``stopped`` timestamps into a static
stream that downstream tooling can inspect alongside other metadata sources.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from datakit.datamodel import LoadedStream
from datakit.sources.register import DataSource


class SessionTimestampsSource(DataSource):
    """Load per-device start/stop timestamps and expose them as metadata."""

    tag = "timestamps"
    patterns = ("**/*_timestamps.csv",)
    camera_tag = None
    version = "1.0"
    is_timeseries = False

    required_columns = ("device_id", "started", "stopped")

    def load(self, path: Path) -> LoadedStream:
        frame = pd.read_csv(path)

        missing = [col for col in self.required_columns if col not in frame.columns]
        if missing:
            raise ValueError(f"Timestamps CSV missing required columns {missing}: {path}")

        normalized = frame.copy()
        for column in ("started", "stopped"):
            normalized[column] = pd.to_datetime(normalized[column], errors="coerce")

        # Represent as a static stream; single sample with metadata table
        t = np.zeros(len(normalized), dtype=np.float64)

        meta = {
            "source_file": str(path),
            "devices": normalized["device_id"].dropna().astype(str).tolist(),
        }

        return self._create_stream(self.tag, t, normalized, meta=meta)


__all__ = ["SessionTimestampsSource"]
