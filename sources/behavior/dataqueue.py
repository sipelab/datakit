"""Dataqueue synchronization data source.

This loader ingests the central ``*_dataqueue.csv`` log produced by the rig and
returns a stream describing every message with absolute and relative timing.
It is commonly used to align per-device clocks (treadmill, cameras, nidaq) by
extracting anchor pairs for downstream timeline fitting.
"""

from typing import Optional
import pandas as pd
import numpy as np
from pathlib import Path

from datakit.sources.register import DataSource
from datakit.datamodel import LoadedStream


class DataqueueSource(DataSource):
    """Load the ubiquitous dataqueue CSV and expose alignment metadata."""
    tag = "dataqueue"
    patterns = ("**/*_dataqueue.csv",)
    camera_tag = None
    version = "1.0"
    time_column = "queue_elapsed"
    device_id_column = "device_id"
    device_timestamp_column = "device_ts"
    optional_time_columns = ("queue_elapsed",)
    payload_column = "payload"
    
    def load(self, path: Path) -> LoadedStream:
        """Read ``*_dataqueue.csv`` and derive alignment helpers."""
        df = pd.read_csv(path, low_memory=False)

        if self.time_column not in df.columns:
            raise ValueError(f"Dataqueue file missing required time column '{self.time_column}': {path}")

        queue_elapsed = pd.to_numeric(df[self.time_column], errors="coerce")
        queue_elapsed = queue_elapsed.to_numpy(dtype=np.float64)
        if queue_elapsed.size == 0:
            raise ValueError("Dataqueue has no timing samples")

        t = queue_elapsed - queue_elapsed[0]
        df = df.copy()
        df["time_elapsed_s"] = t
        absolute = self._coerce_device_ts(df.get(self.device_timestamp_column))
        if absolute is not None:
            df["time_absolute"] = absolute
        
        return LoadedStream(
            tag=self.tag,
            t=t.astype(np.float64),
            value=df,
            meta={
                "source_file": str(path),
                "n_entries": len(df),
                "anchors": self._extract_device_anchors(df, queue_elapsed[0]),
                "device_catalog": self._build_device_catalog(df),
                "device_ids": self._list_device_ids(df),
            }
        )

    @staticmethod
    def _coerce_device_ts(series: Optional[pd.Series]) -> Optional[pd.Series]:
        """Best-effort parse of per-device timestamps into ISO strings."""
        if series is None:
            return None
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
        if parsed.isna().all():
            return None
        return parsed.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    @classmethod
    def _extract_device_anchors(cls, df: pd.DataFrame, queue_start: float) -> dict[str, list[tuple[float, float]]]:
        """Extract device-specific anchor pairs from a dataqueue dataframe."""
        anchors: dict[str, list[tuple[float, float]]] = {}

        if (
            cls.device_id_column not in df.columns
            or cls.device_timestamp_column not in df.columns
            or cls.time_column not in df.columns
        ):
            return anchors

        for device_id, group in df.groupby(cls.device_id_column):
            device_ts = group[cls.device_timestamp_column]

            try:
                master_times = np.asarray(group[cls.time_column], dtype=float) - float(queue_start)
            except Exception:
                continue

            if len(master_times) == 0:
                continue

            try:
                # Prefer timestamp parsing for string-based device clocks
                device_dt = pd.to_datetime(device_ts, errors='raise')
                device_rel = (device_dt - device_dt.iloc[0]).dt.total_seconds().values
            except Exception:
                try:
                    device_numeric = np.asarray(device_ts, dtype=float)
                    device_rel = device_numeric - device_numeric[0]
                except Exception:
                    continue

            if len(device_rel) != len(master_times):
                length = min(len(device_rel), len(master_times))
                device_rel = device_rel[:length]
                master_times = master_times[:length]

            anchors[str(device_id)] = list(zip(master_times.astype(float), device_rel.astype(float)))

        return anchors

    @classmethod
    def _list_device_ids(cls, df: pd.DataFrame) -> list[str]:
        """Return the unique device identifiers present in the queue."""
        if cls.device_id_column not in df.columns:
            return []
        ids = df[cls.device_id_column].dropna().astype(str)
        return list(dict.fromkeys(ids))

    @classmethod
    def _build_device_catalog(cls, df: pd.DataFrame) -> dict[str, dict[str, object]]:
        """Summarize rows/columns per device for quick UX previews."""
        catalog: dict[str, dict[str, object]] = {}

        if cls.device_id_column not in df.columns:
            return catalog

        for device_id, group in df.groupby(cls.device_id_column):
            entry: dict[str, object] = {
                "rows": int(len(group)),
                "columns": list(group.columns),
            }
            if cls.time_column in group.columns:
                queue = pd.to_numeric(group[cls.time_column], errors='coerce').dropna()
                if not queue.empty:
                    entry['queue_start'] = float(queue.iloc[0])
                    entry['queue_stop'] = float(queue.iloc[-1])
            catalog[str(device_id)] = entry

        return catalog