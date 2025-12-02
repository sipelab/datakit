"""Mesoscope camera metadata data source.

Loads the JSON emitted alongside mesoscope OME-TIFF sequences, flattens nested
metadata, and produces a :class:`~datakit.datamodel.LoadedStream` keyed to the
preferred timestamp column.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

from datakit.sources.register import DataSource
from datakit.datamodel import LoadedStream


class MesoMetadataSource(DataSource):
    """Load mesoscope camera metadata JSON files with consistent timestamps."""
    tag = "meso_metadata"
    patterns = ("**/*_mesoscope.ome.tiff_frame_metadata.json",)
    camera_tag = "meso_metadata"
    version = "1.0"
    flatten_payload = False
    json_entry_key = "p0"
    metadata_column = "camera_metadata"
    timestamp_preference = ("TimeReceivedByCore", "ElapsedTime-ms", "runner_time_ms")
    millisecond_columns = {"ElapsedTime-ms", "runner_time_ms"}
    drop_columns = (
        "camera_metadata",
        "property_values",
        "version",
        "format",
        "ROI-X-start",
        "ROI-Y-start",
        "mda_event",
        "Height",
        "Width",
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
        
        p0_data = data[self.json_entry_key]
        df = pd.DataFrame(p0_data)

        device_id = None
        if self.device_column in df.columns and len(df):
            device_id = str(df[self.device_column].iloc[0])

        camera_metadata_df = pd.json_normalize(df[self.metadata_column].tolist())
        df = df.join(camera_metadata_df)

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
                "device_id": device_id
            }
        )