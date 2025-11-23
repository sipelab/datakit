"""Default source definitions for the simplified loader pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import numpy as np

from datakit.config import settings
from datakit.sources.register import register, SourceSpec


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_session_notes(path: Path) -> pd.DataFrame:
    rows: list[tuple[pd.Timestamp, str]] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if ":" not in line:
            continue
        timestamp_text, note_text = line.split(":", 1)
        timestamp = pd.to_datetime(timestamp_text.strip())
        rows.append((timestamp, note_text.strip()))
    if not rows:
        return pd.DataFrame(columns=["note"], index=pd.DatetimeIndex([], name="timestamp"))
    frame = pd.DataFrame(rows, columns=["timestamp", "note"])  # type: ignore[arg-type]
    frame.set_index("timestamp", inplace=True)
    return frame


def load_session_config(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    index_name = frame.columns[0]
    frame.set_index(index_name, inplace=True)
    return frame.T


def load_camera_metadata(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text())
    frames = pd.DataFrame(payload["p0"])
    metadata = pd.json_normalize(frames["camera_metadata"].tolist())
    frames = pd.concat([frames.drop(columns=["camera_metadata"]), metadata], axis=1)
    return frames


def load_dlc_pickle(path: Path) -> pd.DataFrame:
    data = pd.read_pickle(path)
    return pd.DataFrame(data)


def load_numpy_array(path: Path) -> np.ndarray:
    return np.load(path)


_LOADER_MAP = {
    "load_csv": load_csv,
    "load_session_notes": load_session_notes,
    "load_session_config": load_session_config,
    "load_camera_metadata": load_camera_metadata,
    "load_dlc_pickle": load_dlc_pickle,
    "load_numpy_array": load_numpy_array,
}


REGISTERED_SOURCES: dict[str, SourceSpec] = {}
for tag, patterns, loader_name, structured in settings.sources.entries:
    loader = _LOADER_MAP[loader_name]
    REGISTERED_SOURCES[tag] = register(
        tag=tag,
        patterns=patterns,
        loader=loader,
        structured=structured,
    )

DEFAULT_SOURCES: tuple[SourceSpec, ...] = tuple(
    REGISTERED_SOURCES[tag]
    for tag in settings.dataset.desired_tags
    if tag in REGISTERED_SOURCES
)

__all__ = [
    "DEFAULT_SOURCES",
    "REGISTERED_SOURCES",
    "load_camera_metadata",
    "load_csv",
    "load_dlc_pickle",
    "load_session_config",
    "load_session_notes",
    "load_numpy_array",
]
