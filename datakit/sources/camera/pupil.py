"""Pupil camera metadata data source.

Handles the JSON metadata files written next to pupil MP4 recordings, ensuring
the timeline is derived from the most reliable timestamp column available.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

from datakit.sources.register import DataSource
from datakit.datamodel import LoadedStream


class PupilMetadataSource(DataSource):
    """Load pupil camera metadata JSON files with clock normalization."""
    tag = "pupil_metadata"
    patterns = ("**/*_pupil.mp4_frame_metadata.json",)
    camera_tag = "pupil_metadata"
    version = "1.0"
    flatten_payload = False
    timeline_columns = ("time_elapsed_s", "ElapsedTime-ms", "runner_time_ms")
    json_entry_key = "p0"
    metadata_column = "camera_metadata"
    timestamp_preference = ("TimeReceivedByCore", "ElapsedTime-ms", "runner_time_ms")
    millisecond_columns = {"ElapsedTime-ms", "runner_time_ms"}
    drop_columns = (
        "camera_metadata",
        "property_values",
        "version",
        "format",
        "camera_device",
        "pixel_size_um",
        "images_remaining_in_buffer",
        "PixelType",
        "hardware_triggered",
    )
    device_column = "camera_device"
    
    def load(self, path: Path) -> LoadedStream:
        """Flatten camera metadata JSON into a time-indexed dataframe."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        p0_data = data.get(self.json_entry_key)
        entry_key_used = self.json_entry_key

        if p0_data is None:
            # Fallback: use the first available entry if the expected key is missing.
            if isinstance(data, dict) and data:
                entry_key_used, p0_data = next(iter(data.items()))
            else:
                raise KeyError(
                    f"Missing expected entry '{self.json_entry_key}' in {path}; found keys: {list(data.keys()) if isinstance(data, dict) else 'n/a'}"
                )

        if p0_data is None:
            raise ValueError(f"No usable metadata entries found in {path}")

        df = pd.DataFrame(p0_data)

        device_id = None
        if self.device_column in df.columns and len(df):
            device_id = str(df[self.device_column].iloc[0])

        camera_metadata_df = pd.json_normalize(df[self.metadata_column].tolist())
        # Avoid collisions when nested metadata repeats top-level columns.
        non_overlapping = [col for col in camera_metadata_df.columns if col not in df.columns]
        df = df.join(camera_metadata_df[non_overlapping])

        existing_columns = [col for col in self.drop_columns if col in df.columns]
        df.drop(columns=existing_columns, inplace=True)

        t = None
        absolute: pd.Series | None = None
        for column in self.timestamp_preference:
            if column not in df.columns:
                continue
            if column == self.timestamp_preference[0]:
                timestamps = pd.to_datetime(df[column], errors="coerce", utc=True)
                valid = timestamps.dropna()
                if not valid.empty:
                    origin = valid.iloc[0]
                    t = (timestamps - origin).dt.total_seconds().to_numpy(dtype=np.float64)
                    absolute = timestamps.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            else:
                scale = 1000.0 if column in self.millisecond_columns else 1.0
                values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=np.float64)
                if values.size:
                    t = values - values[0]
            if t is not None:
                break

        if t is None:
            raise ValueError(f"No recognized timestamp column found in {path}")

        df = df.copy()
        df["time_elapsed_s"] = t
        if absolute is not None:
            df["time_absolute"] = absolute

        return LoadedStream(
            tag=self.tag,
            t=t.astype(np.float64),
            value=df,
            meta={
                "source_file": str(path),
                "n_frames": len(df),
                "device_id": device_id,
                "json_entry_key": entry_key_used,
            }
        )