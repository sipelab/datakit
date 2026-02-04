"""Psychopy behavioral event data source.

Alignment:

psychopy files have an expStart column that indicates start of the experiment in absolute clock datetime.
Psychopy files have a Custom_Trigger routine; a spacebar press then triggers a sync pulse to the DAQ system.
Thus, the true start is the `expStart` + the `key_resp_2.rt` (the key_resp_2.rt is column in the psychopy file, with 1 value, but that value may not be on the first row)

true end can be calculated using the `expStart` + the elapsed value from the last row of the column that precedes `Subject ID` and 

The dataqueue file collects nidaq TTL pulses. The first pulse (payload=1) has a clock time and relative time (queue_elapsed). We can use this to align the psychopy timeline to the dataqueue timeline.
The experimental window, for sample_experiment3, is between the first nidaq and last nidaq.

The psychopy experiment likely runs longer than the nidaq pulse window, so we can trim the psychopy timeline to be within the nidaq pulse window.

Additional details:
    - The loader derives its primary timeline from event start columns (e.g. display_*.started).
    - All time-like columns are aligned by subtracting the keypress RT so time zero is the trigger.
    - For task-gratings, rows are taken from trial metadata and extended to include the final stopped event.
    - When dataqueue is available, the timeline is trimmed to the nidaq window but remains zero-based.
"""


from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
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
_TIME_SUFFIXES = (".started", ".stopped", ".rt", ".t")


def _parse_exp_start(value: Any) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("h", ":", 1)
    text = re.sub(r":(\d{2})\.(\d{2}\.\d+)", r":\1:\2", text)
    parsed = pd.to_datetime(text, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return parsed

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


def _resolve_keypress_rt(raw: pd.DataFrame) -> tuple[Optional[float], Optional[str]]:
    candidates = [col for col in raw.columns if re.match(r"key_resp.*\.rt$", str(col))]
    for col in candidates:
        series = pd.to_numeric(raw[col], errors="coerce")
        if series.notna().any():
            return float(series.dropna().iloc[0]), str(col)
    return None, None


def _resolve_row_time(raw: pd.DataFrame) -> tuple[np.ndarray, str, dict]:
    """Extract timeline from event start times, preferring display_*.started columns."""
    # Try display event columns first
    display_candidates = [c for c in raw.columns if c.startswith('display_') and c.endswith('.started')]
    for col in display_candidates:
        series = pd.to_numeric(raw[col], errors="coerce")
        if series.notna().sum() >= 2:
            filled = series.interpolate(limit_direction="both")
            
            # Extend timeline to capture final .stopped events
            time_cols = [c for c in raw.columns if any(c.endswith(suf) for suf in _TIME_SUFFIXES)]
            max_times = [pd.to_numeric(raw[c], errors="coerce").max() for c in time_cols]
            max_time = max([t for t in max_times if pd.notna(t)], default=filled.max())
            
            # If there are events after the last filled time, extend with the max time
            if max_time > filled.max():
                last_valid_idx = filled.last_valid_index()
                if last_valid_idx is not None and last_valid_idx < len(filled) - 1:
                    # Extend all remaining NaN values with max_time
                    filled.loc[last_valid_idx + 1:] = max_time
            
            meta = {
                "row_time_column": col,
                "row_time_missing": int(series.isna().sum()),
                "row_time_extended_to": float(max_time),
            }
            return filled.to_numpy(dtype=np.float64), col, meta

    # Fallback to 15Hz estimate
    fallback = np.arange(len(raw), dtype=np.float64) / 15.0215
    meta = {
        "row_time_column": "fallback_15hz",
        "row_time_missing": int(len(raw)),
    }
    return fallback, "fallback_15hz", meta


def _dataqueue_pulse_window(directory: Path) -> tuple[Optional[float], Optional[float], dict]:
    timeline = GlobalTimeline.for_directory(directory)
    if timeline is None:
        return None, None, {}

    frame = timeline.dataframe()
    if frame.empty:
        return None, None, {"dataqueue_file": str(timeline.source_path)}

    queue_col = "queue_elapsed"
    queue = pd.to_numeric(frame.get(queue_col), errors="coerce")
    if queue.isna().all():
        return None, None, {"dataqueue_file": str(timeline.source_path)}

    queue_start = float(queue.dropna().iloc[0])

    mask = pd.Series(True, index=frame.index)
    if "device_id" in frame.columns:
        mask &= frame["device_id"].astype(str).str.contains("nidaq", case=False, na=False)
    if "payload" in frame.columns:
        mask &= pd.to_numeric(frame["payload"], errors="coerce") == 1

    pulses = queue.loc[mask].dropna()
    if pulses.empty:
        pulses = queue.dropna()

    if pulses.empty:
        return None, None, {"dataqueue_file": str(timeline.source_path)}

    start = float(pulses.iloc[0])
    end = float(pulses.iloc[-1])
    start_rel = start - queue_start
    end_rel = end - queue_start
    meta = {
        "dataqueue_file": str(timeline.source_path),
        "dataqueue_pulses": int(pulses.size),
        "dataqueue_queue_start": queue_start,
        "dataqueue_start": start_rel,
        "dataqueue_end": end_rel,
    }
    return start_rel, end_rel, meta


def _align_time_values(values: np.ndarray, key_rt: Optional[float], offset: Optional[float]) -> np.ndarray:
    aligned = values.astype(np.float64, copy=True)
    if key_rt is not None:
        aligned = aligned - float(key_rt)
    if offset is not None:
        aligned = aligned - float(offset)
    return aligned


def _align_trial_times(
    frame: pd.DataFrame,
    *,
    key_rt: Optional[float],
    offset: Optional[float],
) -> tuple[pd.DataFrame, list[str]]:
    aligned = frame.copy()
    aligned_cols: list[str] = []
    for col in aligned.columns:
        name = str(col)
        if not (name.lower().endswith(_TIME_SUFFIXES) or name.startswith("window_")):
            continue
        series = pd.to_numeric(aligned[col], errors="coerce")
        if series.notna().any():
            aligned[col] = _align_time_values(series.to_numpy(dtype=np.float64), key_rt, offset)
            aligned_cols.append(name)
    return aligned, aligned_cols


def _build_window_list(frame: pd.DataFrame, start_col: str, stop_col: str) -> list[tuple[float, float]]:
    if start_col not in frame.columns or stop_col not in frame.columns:
        return []
    starts = pd.to_numeric(frame[start_col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    stops = pd.to_numeric(frame[stop_col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    windows = [
        (float(start), float(stop))
        for start, stop in zip(starts, stops)
        if np.isfinite(start) and np.isfinite(stop) and stop > start
    ]
    return windows


def _extract_gratings_trials(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a task-gratings trial metadata table.

    Uses display_gratings.started as the trial time, not thisRow.t.
    """
    columns: list[str] = [
        "trials.thisN",
        "display_gratings.started",
        "display_gratings.stopped",
        "stim_grayScreen.started",
        "stim_grayScreen.stopped",
        "stim_grating.started",
        "stim_grating.stopped",
    ]
    selected = _select_columns(raw, columns)
    
    # Keep only rows with valid trial numbers
    has_trial = selected.get('trials.thisN', pd.Series(dtype=float)).notna()
    result = selected[has_trial].copy()

    grating_windows: list[tuple[float, float]] = []
    gray_windows: list[tuple[float, float]] = []

    if not result.empty:
        start_candidates = [
            col for col in ["stim_grating.started", "display_gratings.started"] if col in result.columns
        ]
        stop_candidates = [
            col
            for col in [
                "stim_grating.stopped",
                "display_gratings.stopped",
            ]
            if col in result.columns
        ]

        if start_candidates:
            start_frame = result.loc[:, start_candidates]
            grating_start = start_frame.bfill(axis=1).iloc[:, 0]
        else:
            grating_start = pd.Series(np.nan, index=result.index)

        if stop_candidates:
            stop_frame = result.loc[:, stop_candidates]
            grating_stop = stop_frame.bfill(axis=1).iloc[:, 0]
        else:
            grating_stop = pd.Series(np.nan, index=result.index)

        gray_start = result.get("stim_grayScreen.started", pd.Series(np.nan, index=result.index))

        if start_candidates:
            next_grating = grating_start.shift(-1)
        else:
            next_grating = pd.Series(np.nan, index=result.index)
        gray_stop_fallback = result.get("display_gratings.stopped", pd.Series(np.nan, index=result.index))
        gray_stop = next_grating.where(next_grating.notna(), gray_stop_fallback)

        result["window_grating_start"] = pd.to_numeric(grating_start, errors="coerce")
        result["window_grating_stop"] = pd.to_numeric(grating_stop, errors="coerce")
        result["window_gray_start"] = pd.to_numeric(gray_start, errors="coerce")
        result["window_gray_stop"] = pd.to_numeric(gray_stop, errors="coerce")

        grating_windows = list(
            zip(
                result["window_grating_start"].to_numpy(dtype=np.float64, copy=False),
                result["window_grating_stop"].to_numpy(dtype=np.float64, copy=False),
            )
        )
        grating_windows = [
            (float(start), float(stop))
            for start, stop in grating_windows
            if np.isfinite(start) and np.isfinite(stop) and stop > start
        ]

        gray_windows = list(
            zip(
                result["window_gray_start"].to_numpy(dtype=np.float64, copy=False),
                result["window_gray_stop"].to_numpy(dtype=np.float64, copy=False),
            )
        )
        gray_windows = [
            (float(start), float(stop))
            for start, stop in gray_windows
            if np.isfinite(start) and np.isfinite(stop) and stop > start
        ]

        result["gratings_windows"] = [grating_windows] * len(result)
        result["gray_windows"] = [gray_windows] * len(result)
    
    # Add a final row to capture the end of the last stopped event
    if len(result) > 0 and 'display_gratings.started' in result.columns:
        trial_start = pd.to_numeric(result['display_gratings.started'], errors='coerce')
        stopped_cols = [c for c in columns if c.endswith('.stopped')]
        
        # Find max time across all stopped columns
        max_time = trial_start.max()
        for col in stopped_cols:
            if col in result.columns:
                col_vals = pd.to_numeric(result[col], errors='coerce')
                col_max = col_vals.max()
                if pd.notna(col_max) and col_max > max_time:
                    max_time = col_max
        
        # If the last stopped event is after the last trial start, append a row
        last_start = trial_start.iloc[-1]
        if pd.notna(max_time) and max_time > last_start:
            # Create a final row with just the time extension
            final_row: dict[str, object] = {col: np.nan for col in result.columns}
            final_row["display_gratings.started"] = max_time
            final_row["window_grating_start"] = max_time
            if "gratings_windows" in result.columns:
                final_row["gratings_windows"] = grating_windows
            if "gray_windows" in result.columns:
                final_row["gray_windows"] = gray_windows
            result = pd.concat([result, pd.DataFrame([final_row])], ignore_index=True)
    
    return result


def _extract_movies_trials(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a task-movies trial metadata table.

    TODO: replace the column list with the true Psychopy fields you care about.
    """
    columns: list[str] = [
        "trials_2.thisN",
        "trials_2.thisIndex",
        "trials2.thisN",
        "trials.thisIndex",
        "description",
        "thisRow.t",
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
        extractor = extractors.get(task or "")
        if extractor is None:
            trial_table = raw
            task_branch = "raw"
        else:
            prefix, func = extractor
            trial_table = func(raw)
            task_branch = prefix

        # 3) Build a row-level timeline using event start times.
        # Use trial_table if it has display_*.started, otherwise use raw
        time_source = trial_table if any(c.startswith('display_') and c.endswith('.started') 
                                        for c in trial_table.columns) else raw
        row_time, row_col, row_meta = _resolve_row_time(time_source)
        key_rt, key_col = _resolve_keypress_rt(raw)

        exp_start_raw = raw.get("expStart")
        exp_start = None
        if exp_start_raw is not None:
            exp_start = _parse_exp_start(exp_start_raw.dropna().iloc[0]) if exp_start_raw.dropna().any() else None

        dq_start, dq_end, dq_meta = _dataqueue_pulse_window(path.parent)
        offset_s = dq_start if dq_start is not None else None
        aligned_t = _align_time_values(row_time, key_rt, offset_s)

        # 4) Align all time-like columns in the trial table (no offset, start from 0).
        aligned_table, aligned_cols = _align_trial_times(
            trial_table,
            key_rt=key_rt,
            offset=offset_s,
        )

        # 5) Trim to nidaq window when available (convert to zero-based window).
        trim_mask = np.isfinite(aligned_t)
        if dq_start is not None and dq_end is not None:
            window_duration = dq_end - dq_start
            trim_mask &= (aligned_t >= 0) & (aligned_t <= window_duration)

        trimmed_rows = int(trim_mask.sum())
        if 0 < trimmed_rows < len(aligned_t):
            aligned_t = aligned_t[trim_mask]
            aligned_table = aligned_table.loc[trim_mask].reset_index(drop=True)

        if task_branch == "gratings":
            gratings_windows = _build_window_list(
                aligned_table,
                "window_grating_start",
                "window_grating_stop",
            )
            gray_windows = _build_window_list(
                aligned_table,
                "window_gray_start",
                "window_gray_stop",
            )
            if "window_grating_start" in aligned_table.columns:
                aligned_table["gratings_windows"] = [gratings_windows] * len(aligned_table)
            if "window_gray_start" in aligned_table.columns:
                aligned_table["gray_windows"] = [gray_windows] * len(aligned_table)

        payload = StreamPayload.table(_prefix_columns(aligned_table, task_branch))

        # 6) Persist useful diagnostics for downstream inspection.
        metrics = {
            "source_file": str(path),
            "n_rows": int(len(raw)),
            "columns": tuple(str(col) for col in raw.columns),
            "task_inferred": task,
            "task_branch": task_branch,
            "row_time_column": row_col,
            "key_resp_rt": key_rt,
            "key_resp_column": key_col,
            "aligned_columns": tuple(aligned_cols),
            "trimmed_rows": int(trimmed_rows),
            "exp_start": exp_start.isoformat() if exp_start is not None else None,
            "start_offset_s": offset_s,
        }
        metrics.update(row_meta)
        metrics.update(dq_meta)

        if dq_start is not None:
            metrics["time_basis"] = "psychopy_zero_based"
            metrics["alignment_method"] = "keypress_relative"
        else:
            metrics["time_basis"] = "psychopy_native"

        return LoadedStream(tag=self.tag, t=aligned_t, value=payload, meta=metrics)
    

