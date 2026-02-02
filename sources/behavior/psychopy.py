"""Psychopy behavioral event data source."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

from datakit.sources.register import DataSource
from datakit.datamodel import LoadedStream, StreamPayload
from datakit.timeline import GlobalTimeline


_TASK_PATTERN = re.compile(r"task[-_]?([A-Za-z0-9]+)", re.IGNORECASE)
_TASK_ALIASES = {
    "gratings": "task-gratings",
    "movies": "task-movies",
}

def _infer_task(path: Path) -> str | None:
    """Infer a known task label from the file path."""
    name = path.as_posix().lower()
    match = _TASK_PATTERN.search(name)
    if match:
        return _TASK_ALIASES.get(match.group(1).lower())
    for key, label in _TASK_ALIASES.items():
        if key in name:
            return label
    return None


def _prefix_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Add a task-specific prefix to every column name."""
    renamed = {col: f"{prefix}_{col}" for col in df.columns}
    return df.rename(columns=renamed)


def _select_columns(raw: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return only the available columns, or an empty frame if none are present."""
    available = [col for col in columns if col in raw.columns]
    if not available:
        return pd.DataFrame(index=raw.index)
    return raw.loc[:, available].copy()


def _align_to_dataqueue(t: np.ndarray, directory: Path) -> tuple[np.ndarray, dict]:
    """Align a local timeline to the dataqueue clock when available.

    The alignment is intentionally simple: offset the local time vector by the
    first available dataqueue queue_elapsed sample. This yields a timeline that
    lives in the same time basis as other dataqueue-aligned streams.
    """
    timeline = GlobalTimeline.for_directory(directory)
    if timeline is None:
        return t, {}

    queue = timeline.queue_series()
    if queue.empty:
        return t, {"dataqueue_file": str(timeline.source_path)}

    offset = float(queue.iloc[0])
    aligned = t + offset
    meta = {
        "time_basis": "dataqueue",
        "dataqueue_file": str(timeline.source_path),
        "alignment_method": "dataqueue_start_offset",
        "dataqueue_start": offset,
    }
    return aligned, meta


def _extract_gratings_trials(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a task-gratings trial metadata table.

    TODO: replace the column list with the true Psychopy fields you care about.
    """
    columns: list[str] = [
        "trials.thisN",
        "display_gratings.started",
        "stim_grayScreen.started",
        "display_gratings.stopped",
    ]
    return _select_columns(raw, columns)


def _extract_movies_trials(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a task-movies trial metadata table.

    TODO: replace the column list with the true Psychopy fields you care about.
    """
    columns: list[str] = [
        "trials2.thisN",
        "trials.thisIndex",
        "description",
        "display_mov.started",
        "natural_mov.started",
        "display_mov.stopped",
        "grey.started",
        "grey.stopped",
    ]
    return _select_columns(raw, columns)


class Psychopy(DataSource):
    """Load the raw Psychopy CSV with task-specific trial metadata."""

    tag = "psychopy"
    patterns = ("**/*_psychopy.csv",)
    version = "3.0"

    def load(self, path: Path) -> LoadedStream:
        # 1) Read the raw CSV.
        raw = pd.read_csv(path, low_memory=False)
        # 2) Infer task and extract a task-specific metadata table.
        task = _infer_task(path)
        extractors = {
            "task-gratings": ("gratings", _extract_gratings_trials),
            "task-movies": ("movies", _extract_movies_trials),
        }
        extractor = extractors.get(task)
        if extractor is None:
            payload = StreamPayload.table(raw)
            task_branch = "raw"
        else:
            prefix, func = extractor
            trial_table = func(raw)
            payload = StreamPayload.table(_prefix_columns(trial_table, prefix))
            task_branch = prefix

        # 3) Provide a 15 Hz time vector (seconds) for row-aligned data.
        timeline = np.arange(len(raw), dtype=np.float64) / 15.0215
        # 4) Optionally map the timeline onto the dataqueue clock.
        timeline, alignment_meta = _align_to_dataqueue(timeline, path.parent)

        # 5) Persist useful diagnostics for downstream inspection.
        metrics = {
            "source_file": str(path),
            "n_rows": int(len(raw)),
            "columns": tuple(str(col) for col in raw.columns),
            "task_inferred": task,
            "task_branch": task_branch,
        }
        metrics.update(alignment_meta)
        
        return LoadedStream(tag=self.tag, t=timeline, value=payload, meta=metrics)
    