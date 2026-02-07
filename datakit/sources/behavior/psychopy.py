"""Psychopy behavioral event data source.

Explicit assumptions:
- Psychopy CSV contains at least one display_*.started column with 2+ numeric rows.
- A single keypress RT column exists (key_resp*.rt) and is used to align time zero.
- If a dataqueue timeline exists, it must include nidaq payload==1 pulses.

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


class Psychopy(DataSource):
    """Load the raw Psychopy CSV with task-specific trial metadata."""

    tag = "psychopy"
    patterns = ("**/*_psychopy.csv",)
    version = "3.1"
    task_pattern = _TASK_PATTERN
    task_aliases = _TASK_ALIASES
    time_suffixes = _TIME_SUFFIXES
    keypress_rt_pattern = re.compile(r"key_resp.*\.rt$")
    time_source_prefix = "display_"
    time_source_suffix = ".started"
    gratings_columns = (
        "trials.thisN",
        "stim_grayScreen.started",
        "stim_grayScreen.stopped",
        "stim_grating.started",
        "stim_grating.stopped",
        "display_gratings.stopped",
    )
    movies_columns = (
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
    )
    dataqueue_elapsed_column = "queue_elapsed"
    dataqueue_device_column = "device_id"
    dataqueue_device_match = "nidaq"
    dataqueue_payload_column = "payload"
    dataqueue_payload_value = 1
    
    
    def load(self, path: Path) -> LoadedStream:
        # 1) Read raw psychopy CSV.
        raw = pd.read_csv(path, low_memory=False)

        # 2) Infer task and choose trial table.
        task = self._infer_task(path)
        if task == "task-gratings":
            trial_table = self._extract_gratings(raw)
            task_branch = "gratings"
        elif task == "task-movies":
            trial_table = self._extract_movies(raw)
            task_branch = "movies"
        else:
            trial_table = raw
            task_branch = "raw"

        # 3) Resolve timeline and keypress offset.
        time_source = trial_table if any(
            c.startswith(self.time_source_prefix) and c.endswith(self.time_source_suffix) for c in trial_table.columns
        ) else raw
        row_time, row_col, row_meta = self._resolve_row_time(time_source)
        key_rt, key_col = self._resolve_keypress_rt(raw)

        # 4) Parse experiment start time (optional).
        exp_start = None
        exp_start_raw = raw.get("expStart")
        if exp_start_raw is not None and exp_start_raw.dropna().any():
            exp_start = self._parse_exp_start(exp_start_raw.dropna().iloc[0])

        # 5) Apply dataqueue offset and align times.
        dq_start, dq_meta = self._dataqueue_offset(path.parent, task)
        aligned_t = self._align_time(row_time, key_rt, dq_start)
        aligned_table = self._align_table(trial_table, key_rt, dq_start)

        # 6) For gratings, build windows once from the aligned table.
        if task_branch == "gratings":
            if "window_grating_start" in aligned_table.columns:
                aligned_table["gratings_windows"] = [
                    self._build_window_list(aligned_table, "window_grating_start", "window_grating_stop")
                ] * len(aligned_table)
            if "window_gray_start" in aligned_table.columns:
                aligned_table["gray_windows"] = [
                    self._build_window_list(aligned_table, "window_gray_start", "window_gray_stop")
                ] * len(aligned_table)

        # 7) Package payload and metrics.
        payload = StreamPayload.table(
            aligned_table.rename(columns={c: f"{task_branch}_{c}" for c in aligned_table.columns})
        )
        metrics = {
            "source_file": str(path),
            "n_rows": int(len(raw)),
            "task_inferred": task,
            "task_branch": task_branch,
            "row_time_column": row_col,
            "key_resp_rt": key_rt,
            "key_resp_column": key_col,
            "trimmed_rows": int(len(aligned_t)),
            "exp_start": exp_start.isoformat() if exp_start is not None else None,
            "start_offset_s": dq_start,
        }
        metrics.update(row_meta)
        metrics.update(dq_meta)
        metrics["time_basis"] = "psychopy_zero_based" if dq_start is not None else "psychopy_native"
        metrics["alignment_method"] = "keypress_relative"

        return LoadedStream(tag=self.tag, t=aligned_t, value=payload, meta=metrics)

    def _parse_exp_start(self, value: Any) -> Optional[pd.Timestamp]:
        text = str(value).strip()
        text = text.replace("h", ":", 1)
        text = re.sub(r":(\d{2})\.(\d{2}\.\d+)", r":\1:\2", text)
        parsed = pd.to_datetime(text, errors="coerce", utc=True)
        return None if pd.isna(parsed) else parsed

    def _infer_task(self, file_path: Path) -> str | None:
        name = file_path.as_posix().lower()
        match = self.task_pattern.search(name)
        if match:
            return self.task_aliases.get(match.group(1).lower())
        for key, label in self.task_aliases.items():
            if key in name:
                return label
        return None

    def _select_columns(self, frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        available = [col for col in columns if col in frame.columns]
        if not available:
            raise ValueError("Psychopy required columns are missing from the CSV.")
        return frame.loc[:, available].copy()

    def _coalesce_column(self, frame: pd.DataFrame, columns: list[str]) -> pd.Series:
        available = [col for col in columns if col in frame.columns]
        if not available:
            raise ValueError(f"Psychopy required columns missing: {columns}")
        return frame.loc[:, available].bfill(axis=1).iloc[:, 0]

    def _resolve_keypress_rt(self, frame: pd.DataFrame) -> tuple[float, str]:
        candidates = [col for col in frame.columns if self.keypress_rt_pattern.match(str(col))]
        for col in candidates:
            series = pd.to_numeric(frame[col], errors="coerce")
            if series.notna().any():
                return float(series.dropna().iloc[0]), str(col)
        raise ValueError("Psychopy keypress RT column not found (key_resp*.rt).")

    def _resolve_row_time(self, frame: pd.DataFrame) -> tuple[np.ndarray, str, dict]:
        display_candidates = [
            c for c in frame.columns if c.startswith(self.time_source_prefix) and c.endswith(self.time_source_suffix)
        ]
        for col in display_candidates:
            series = pd.to_numeric(frame[col], errors="coerce")
            if series.notna().sum() >= 2:
                filled = series.interpolate(limit_direction="both")
                time_cols = [c for c in frame.columns if any(c.endswith(suf) for suf in self.time_suffixes)]
                max_times = [pd.to_numeric(frame[c], errors="coerce").max() for c in time_cols]
                max_time = max([t for t in max_times if pd.notna(t)], default=filled.max())
                if max_time > filled.max():
                    last_valid_idx = filled.last_valid_index()
                    if last_valid_idx is not None and last_valid_idx < len(filled) - 1:
                        filled.loc[last_valid_idx + 1:] = max_time
                meta = {
                    "row_time_column": col,
                    "row_time_missing": int(series.isna().sum()),
                    "row_time_extended_to": float(max_time),
                }
                return filled.to_numpy(dtype=np.float64), col, meta
        raise ValueError("Psychopy requires a display_*.started column with at least 2 numeric values.")

    def _dataqueue_offset(self, directory: Path, task: Optional[str]) -> tuple[Optional[float], dict]:
        #TODO: If a dataqueue does not have nidaq payload==1 but may start at nidaq==2,
        # treat the first pulse as the start and log a warning. 
        candidates = sorted(directory.glob(f"*{task}*_dataqueue.csv"))
        dq_path = candidates[0]
        frame = GlobalTimeline._load_dataqueue(dq_path)
        timeline = GlobalTimeline(directory.resolve(), dq_path, frame)
        queue_all = timeline.queue_series()

        queue_start = float(queue_all.iloc[0])
        device_slice = timeline.slice(self.dataqueue_device_match)
        abs_start = device_slice.packet_absolute().to_numpy(dtype=str)
        pulses = pd.to_numeric(device_slice.rows.get(self.dataqueue_elapsed_column), errors="coerce")
        # if self.dataqueue_payload_column in device_slice.rows.columns:
        #     payload = pd.to_numeric(device_slice.rows[self.dataqueue_payload_column], errors="coerce")
        #     pulses = pulses.loc[payload == self.dataqueue_payload_value]
        pulses = pulses.dropna()
        if pulses.empty:
            raise ValueError(f"Dataqueue timeline exists {timeline.source_path} but contains no nidaq payload==1 pulses.")

        start_rel = float(pulses.iloc[0]) - 0
        meta = {
            "dataqueue_file": str(timeline.source_path),
            "dataqueue_pulses": int(pulses.size),
            "dataqueue_queue_start": queue_start,
            "abs_start_time": abs_start[0],
            "dataqueue_start": start_rel,
            "dataqueue_end": None,
        }
        return start_rel, meta

    def _align_time(self, values: np.ndarray, key_rt: float, offset: Optional[float]) -> np.ndarray:
        aligned = values.astype(np.float64, copy=True) - float(key_rt)
        # if offset is not None:
        #     aligned = aligned - float(offset)
        return aligned

    def _align_table(self, frame: pd.DataFrame, key_rt: float, offset: Optional[float]) -> pd.DataFrame:
        aligned = frame.copy()
        for col in aligned.columns:
            name = str(col)
            if not (name.lower().endswith(self.time_suffixes) or name.startswith("window_")):
                continue
            series = pd.to_numeric(aligned[col], errors="coerce")
            aligned[col] = self._align_time(series.to_numpy(dtype=np.float64), key_rt, offset)
        return aligned.dropna(axis=1, how="all")

    def _build_window_list(self, frame: pd.DataFrame, start_col: str, stop_col: str) -> list[tuple[float, float]]:
        if start_col not in frame.columns or stop_col not in frame.columns:
            return []
        starts = pd.to_numeric(frame[start_col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        stops = pd.to_numeric(frame[stop_col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        return [
            (float(start), float(stop))
            for start, stop in zip(starts, stops)
            if np.isfinite(start) and np.isfinite(stop) and stop > start
        ]

    def _extract_gratings(self, frame: pd.DataFrame) -> pd.DataFrame:
        selected = self._select_columns(frame, list(self.gratings_columns))
        result = selected.loc[selected.get("trials.thisN").notna()].copy()
        if result.empty:
            raise ValueError("Psychopy gratings trials are empty.")

        gray_start = self._coalesce_column(result, ["stim_grayScreen.started"])
        gray_stop = self._coalesce_column(result, ["stim_grating.started"])
        grating_start = self._coalesce_column(result, ["stim_grating.started"])
        grating_stop = self._coalesce_column(result, ["display_gratings.stopped"])

        result["window_grating_start"] = pd.to_numeric(grating_start, errors="coerce")
        result["window_grating_stop"] = pd.to_numeric(grating_stop, errors="coerce")
        result["window_gray_start"] = pd.to_numeric(gray_start, errors="coerce")
        result["window_gray_stop"] = pd.to_numeric(gray_stop, errors="coerce")

        return result

    def _extract_movies(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self._select_columns(frame, list(self.movies_columns))

    
