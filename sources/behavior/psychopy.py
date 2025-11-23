"""Psychopy behavioral event data source."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from datakit.sources.register import DataSource
from datakit.datamodel import LoadedStream, StreamPayload
from datakit.timeline import GlobalTimeline


class Psychopy(DataSource):
    """Load Psychopy trial tables emitted by the HFSA rig.

    The CSV exported by Psychopy interleaves rig metadata, stimulus timing, and
    response measurements. This loader extracts valid trial rows, casts the time
    columns to floating point seconds, and assembles a compact table capturing
    the key stimulus epochs needed for downstream analysis.
    """

    tag = "psychopy"
    patterns = ("**/*_psychopy.csv",)
    version = "2.0"

    # Canonical columns describing trial structure
    _timeline_cols: tuple[str, ...] = (
        "thisRow.t",
        "display_gratings.started",
        "stim_grayScreen.started",
        "stim_grating.started",
        "stim_grayScreen.stopped",
        "stim_grating.stopped",
        "display_gratings.stopped",
    )
    keypress_start_column = "key_resp_2.started"
    keypress_rt_column = "key_resp_2.rt"
    fallback_start_columns: tuple[tuple[str, str], ...] = (
        ("key_resp_2.started", "keyboard_component_started"),
        ("Custom_Trigger.started", "custom_trigger_started"),
        ("Custom_Trigger.stopped", "custom_trigger_stopped"),
    )
    experiment_start_priority: tuple[str, ...] = (
        "keyboard_press",
        "keyboard_component_started",
        "custom_trigger_started",
        "custom_trigger_stopped",
    )
    default_stim_duration_s: float = 2.0
    default_gray_extension_s: float = 3.0
    timeline_alignment_columns: tuple[str, ...] = (
        "time_elapsed_s",
        "grating_display_on_s",
        "gray_on_s",
        "stim_on_s",
        "gray_off_s",
        "stim_off_s",
        "grating_display_off_s",
    )
    timeline_filter_keyword: str = "nidaq"
    timeline_filter_fallback_label: str = "all"
    timeline_scale_tolerance: float = 0.005

    def load(self, path: Path) -> LoadedStream:
        raw = pd.read_csv(path)
        trials = self._extract_trials(raw)

        experiment_zero_s, zero_source = self._estimate_experiment_start(raw)
        timeline_obj = GlobalTimeline.for_directory(path.parent)

        if trials.empty:
            raise ValueError(f"Psychopy file does not contain trial rows: {path}")

        trial_table = self._build_trial_table(
            trials,
            path.parent,
            experiment_zero_s=experiment_zero_s,
            experiment_zero_source=zero_source,
            timeline=timeline_obj,
        )
        timeline_values = trial_table["time_elapsed_s"].to_numpy(dtype=np.float64)

        payload_fields = self._build_payload_fields(trial_table)
        payload_table = self._build_payload_table(payload_fields)
        payload = StreamPayload.table(payload_table)

        subject = self._unique_value(raw.get("Subject ID"))
        session = self._unique_value(raw.get("Session ID"))
        exp_name = self._unique_value(raw.get("expName"))
        exp_start = self._unique_value(raw.get("expStart"))

        metrics = {
            "source_file": str(path),
            "n_trials": int(len(trial_table)),
            "subject_id": subject,
            "session_id": session,
            "experiment_name": exp_name,
            "experiment_start": exp_start,
            "mean_gray_duration_s": trial_table["gray_duration_s"].mean(),
            "mean_stim_duration_s": trial_table["stim_duration_s"].mean(),
        }

        alignment_meta = trial_table.attrs.get("clock_alignment")
        if alignment_meta:
            metrics.update(alignment_meta)

        timeline_origin = trial_table.attrs.get("timeline_origin")
        if isinstance(timeline_origin, dict):
            metrics.update({
                "timeline_origin_s": timeline_origin.get("offset_s"),
                "timeline_origin_source": timeline_origin.get("source"),
            })

        return LoadedStream(
            tag=self.tag,
            t=timeline_values,
            value=payload,
            meta=metrics,
        )

    # ------------------------------------------------------------------
    # Trial parsing helpers
    # ------------------------------------------------------------------
    def _extract_trials(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Restrict the CSV to rows that encode real trial data."""

        source = (
            raw["trials.thisN"]
            if "trials.thisN" in raw.columns
            else pd.Series(np.nan, index=raw.index, dtype=float)
        )
        trial_idx = pd.to_numeric(source, errors="coerce")
        selection = raw.loc[trial_idx.notna()].copy()

        # Preserve the integer trial order for downstream grouping
        selection["trial_index"] = trial_idx.loc[selection.index].astype(int)
        return selection.reset_index(drop=True)

    def _estimate_experiment_start(self, raw: pd.DataFrame) -> Tuple[Optional[float], Optional[str]]:
        """Estimate the absolute start of the experiment in seconds.

        Psychopy records multiple timing markers around experiment start. We
        prefer the actual key press that triggered the acquisition and fall back
        to other stimulus components when it is unavailable.
        """

        def _coerce(name: str) -> Optional[pd.Series]:
            if name not in raw.columns:
                return None
            return pd.to_numeric(raw[name], errors="coerce")

        candidates: dict[str, float] = {}

        key_start = _coerce(self.keypress_start_column)
        key_rt = _coerce(self.keypress_rt_column)
        if key_start is not None and key_rt is not None:
            key_press = key_start + key_rt
            value = self._first_finite(key_press)
            if value is not None:
                candidates["keyboard_press"] = value if "keyboard_press" not in candidates else min(candidates["keyboard_press"], value)

        for column, label in self.fallback_start_columns:
            value = self._first_finite(_coerce(column))
            if value is not None:
                if label in candidates:
                    candidates[label] = min(candidates[label], value)
                else:
                    candidates[label] = value

        if not candidates:
            return None, None

        for label in self.experiment_start_priority:
            if label in candidates:
                return float(candidates[label]), label

        label, value = min(candidates.items(), key=lambda item: item[1])
        return float(value), label

    def _build_trial_table(
        self,
        trials: pd.DataFrame,
        directory: Optional[Path] = None,
        *,
        experiment_zero_s: Optional[float] = None,
        experiment_zero_source: Optional[str] = None,
        timeline: Optional[GlobalTimeline] = None,
    ) -> pd.DataFrame:
        """Cast time columns and derive convenience measures."""

        time_cols: Iterable[str] = [col for col in self._timeline_cols if col in trials.columns]
        for col in time_cols:
            trials[col] = pd.to_numeric(trials[col], errors="coerce")

        trial_start = trials.get("thisRow.t")
        if trial_start is None or trial_start.isna().all():
            # Fallback to grating onset when thisRow.t is unavailable
            trial_start = trials.get("display_gratings.started")

        if trial_start is None or trial_start.isna().all():
            raise ValueError("Unable to determine trial start times from Psychopy CSV")

        trial_start = trial_start.ffill().bfill()

        table = pd.DataFrame({
            "trial_index": trials["trial_index"].to_numpy(dtype=np.int64),
            "time_elapsed_s": trial_start.to_numpy(dtype=np.float64),
            "grating_display_on_s": trials.get("display_gratings.started"),
            "gray_on_s": trials.get("stim_grayScreen.started"),
            "stim_on_s": trials.get("stim_grating.started"),
            "gray_off_s": trials.get("stim_grayScreen.stopped"),
            "stim_off_s": trials.get("stim_grating.stopped"),
            "grating_display_off_s": trials.get("display_gratings.stopped"),
            "response_key": trials.get("key_resp_2.keys"),
            "response_rt_s": self._coerce_numeric(trials.get("key_resp_2.rt"), trials.index),
            "response_duration_s": self._coerce_numeric(trials.get("key_resp_2.duration"), trials.index),
            "notes": trials.get("notes"),
        })

        for col in (
            "grating_display_on_s",
            "gray_on_s",
            "stim_on_s",
            "gray_off_s",
            "stim_off_s",
            "grating_display_off_s",
        ):
            if col in table.columns:
                table[col] = pd.to_numeric(table[col], errors="coerce")

        # Replace NaNs that stem from incomplete trials with sensible defaults
        table.sort_values("trial_index", inplace=True)
        table.reset_index(drop=True, inplace=True)

        # Ensure the timeline starts at zero using the acquisition trigger when available
        baseline_source: Optional[str] = None
        baseline: Optional[float] = None

        if experiment_zero_s is not None and np.isfinite(experiment_zero_s):
            baseline = float(experiment_zero_s)
            baseline_source = experiment_zero_source or "keyboard_trigger"

        if baseline is None and len(table):
            candidate = float(pd.to_numeric(table["time_elapsed_s"], errors="coerce").min())
            if np.isfinite(candidate):
                baseline = candidate
                baseline_source = "first_trial_start"

        if baseline is None and len(table):
            baseline = float(table["time_elapsed_s"].iloc[0])
            baseline_source = baseline_source or "first_trial_start"

        if baseline is None:
            baseline = 0.0
            baseline_source = baseline_source or "implicit_zero"

        if len(table):
            table["time_elapsed_s"] = table["time_elapsed_s"] - baseline
            for col in (
                "grating_display_on_s",
                "gray_on_s",
                "stim_on_s",
                "gray_off_s",
                "stim_off_s",
                "grating_display_off_s",
            ):
                if col in table.columns:
                    table[col] = table[col] - baseline

        table.attrs["timeline_origin"] = {
            "offset_s": float(baseline),
            "source": baseline_source,
        }

        # Infer missing stop times from available measurements
        def _nanmedian_positive(values: np.ndarray) -> float:
            finite = values[np.isfinite(values)]
            finite = finite[finite > 0]
            if finite.size:
                return float(np.nanmedian(finite))
            return float("nan")

        stim_on = table["stim_on_s"].to_numpy(dtype=float)
        stim_off = table["stim_off_s"].to_numpy(dtype=float)
        gray_on = table["gray_on_s"].to_numpy(dtype=float)
        gray_off = table["gray_off_s"].to_numpy(dtype=float)
        disp_off = table["grating_display_off_s"].to_numpy(dtype=float)

        default_stim_duration = _nanmedian_positive(stim_off - stim_on)
        if not np.isfinite(default_stim_duration):
            default_stim_duration = _nanmedian_positive(disp_off - stim_on)
        if not np.isfinite(default_stim_duration):
            default_stim_duration = self.default_stim_duration_s

        default_gray_duration = _nanmedian_positive(gray_off - gray_on)
        if not np.isfinite(default_gray_duration):
            default_gray_duration = _nanmedian_positive(disp_off - gray_on)
        if not np.isfinite(default_gray_duration):
            default_gray_duration = default_stim_duration + self.default_gray_extension_s

        stim_off_series = table["stim_off_s"]
        gray_off_series = table["gray_off_s"]

        # First, borrow available measurements from related columns
        if "grating_display_off_s" in table.columns:
            table.loc[stim_off_series.isna() & table["grating_display_off_s"].notna(), "stim_off_s"] = table.loc[stim_off_series.isna() & table["grating_display_off_s"].notna(), "grating_display_off_s"]
            table.loc[gray_off_series.isna() & table["grating_display_off_s"].notna(), "gray_off_s"] = table.loc[gray_off_series.isna() & table["grating_display_off_s"].notna(), "grating_display_off_s"]

        stim_off_series = table["stim_off_s"]
        gray_off_series = table["gray_off_s"]

        # Next, copy gray/stim stops between each other when one is present
        table.loc[stim_off_series.isna() & gray_off_series.notna(), "stim_off_s"] = table.loc[stim_off_series.isna() & gray_off_series.notna(), "gray_off_s"]
        table.loc[gray_off_series.isna() & table["stim_off_s"].notna(), "gray_off_s"] = table.loc[gray_off_series.isna() & table["stim_off_s"].notna(), "stim_off_s"]

        stim_off_series = table["stim_off_s"]
        gray_off_series = table["gray_off_s"]

        # Finally, fall back to adding the typical durations when still missing
        if stim_off_series.isna().any():
            offsets = table.loc[stim_off_series.isna(), "stim_on_s"] + default_stim_duration
            table.loc[stim_off_series.isna() & table["stim_on_s"].notna(), "stim_off_s"] = offsets

        if gray_off_series.isna().any():
            offsets = table.loc[gray_off_series.isna(), "gray_on_s"] + default_gray_duration
            table.loc[gray_off_series.isna() & table["gray_on_s"].notna(), "gray_off_s"] = offsets

        # Scale timeline to the dataqueue-derived master clock when possible
        alignment_meta = self._scale_to_master_clock(table, timeline)
        if alignment_meta:
            table.attrs["clock_alignment"] = alignment_meta

        # Recompute durations now that stop times are complete
        table["gray_duration_s"] = table["gray_off_s"] - table["gray_on_s"]
        table["stim_duration_s"] = table["stim_off_s"] - table["stim_on_s"]

        return table

    # ------------------------------------------------------------------
    # Structured payload helpers
    # ------------------------------------------------------------------
    def _build_payload_fields(self, table: pd.DataFrame) -> Dict[str, Any]:
        def _get_array(series_name: str, dtype: type[np.floating] | None = np.float64) -> np.ndarray:
            series = table.get(series_name)
            if series is None:
                return np.full(table.shape[0], np.nan, dtype=float)
            arr = pd.to_numeric(series, errors="coerce").to_numpy(copy=False)
            if dtype is not None:
                return arr.astype(dtype, copy=False)
            return arr

        def _store_epoch(prefix: str, onset: np.ndarray, offset: np.ndarray, payload: Dict[str, Any]) -> None:
            payload[f"{prefix}_on"] = onset.tolist()
            payload[f"{prefix}_off"] = offset.tolist()

        payload: Dict[str, Any] = {}

        trial_index = table["trial_index"].to_numpy(dtype=np.int64)
        trial_start = table["time_elapsed_s"].to_numpy(dtype=np.float64)
        gray_on = _get_array("gray_on_s")
        gray_off = _get_array("gray_off_s")
        stim_on = _get_array("stim_on_s")
        stim_off = _get_array("stim_off_s")
        display_on = _get_array("grating_display_on_s")
        display_off = _get_array("grating_display_off_s")

        gray_duration = _get_array("gray_duration_s")
        stim_duration = _get_array("stim_duration_s")
        isi = np.concatenate([[np.nan], trial_start[1:] - (stim_off[:-1])]) if len(trial_start) > 1 else np.array([np.nan])

        payload.update({
            "trial_index": trial_index.tolist(),
            "time_elapsed_s": trial_start.tolist(),
            "gray_on_s": gray_on.tolist(),
            "gray_off_s": gray_off.tolist(),
            "stim_on_s": stim_on.tolist(),
            "stim_off_s": stim_off.tolist(),
            "display_on_s": display_on.tolist(),
            "display_off_s": display_off.tolist(),
            "gray_duration_s": gray_duration.tolist(),
            "stim_duration_s": stim_duration.tolist(),
            "inter_stim_interval_s": isi.tolist(),
        })

        _store_epoch("epochs_gray_screen", gray_on, gray_off, payload)
        _store_epoch("epochs_stimulus", stim_on, stim_off, payload)
        _store_epoch("epochs_grating_display", display_on, display_off, payload)

        relative_gray_on = gray_on - trial_start
        relative_gray_off = gray_off - trial_start
        relative_stim_on = stim_on - trial_start
        relative_stim_off = stim_off - trial_start
        relative_display_on = display_on - trial_start
        relative_display_off = display_off - trial_start

        _store_epoch("epochs_relative_gray_screen", relative_gray_on, relative_gray_off, payload)
        _store_epoch("epochs_relative_stimulus", relative_stim_on, relative_stim_off, payload)
        _store_epoch("epochs_relative_grating_display", relative_display_on, relative_display_off, payload)

        payload.update({
            "analysis_stim_on_offset_s": relative_stim_on.tolist(),
            "analysis_stim_off_offset_s": relative_stim_off.tolist(),
            "analysis_gray_on_offset_s": relative_gray_on.tolist(),
            "analysis_gray_off_offset_s": relative_gray_off.tolist(),
        })

        return payload

    def _build_payload_table(self, payload_fields: Dict[str, Any]) -> pd.DataFrame:
        if not payload_fields:
            return pd.DataFrame(index=pd.Index([0], name="record"))

        data = {column: [value] for column, value in payload_fields.items()}
        frame = pd.DataFrame(data)
        frame.index = pd.Index([0], name="record")
        return frame

    @staticmethod
    def _unique_value(series: Optional[pd.Series]) -> Optional[str]:
        """Return a single string value if the series contains one."""

        if series is None:
            return None
        values = [str(v) for v in series.dropna().unique() if str(v).strip()]
        return values[0] if values else None

    @staticmethod
    def _coerce_numeric(series: Optional[pd.Series], index: pd.Index) -> pd.Series:
        """Convert an optional series to floats while preserving the target index."""

        if series is None:
            return pd.Series(np.nan, index=index, dtype=float)
        return pd.to_numeric(series, errors="coerce").reindex(index)

    @staticmethod
    def _first_finite(values: Optional[pd.Series]) -> Optional[float]:
        """Return the earliest finite value from a numeric series."""

        if values is None:
            return None
        arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float, copy=False)
        if arr.size == 0:
            return None
        mask = np.isfinite(arr)
        if not mask.any():
            return None
        arr = arr[mask]
        positive_mask = arr >= 0
        if positive_mask.any():
            arr = arr[positive_mask]
        if arr.size == 0:
            return None
        return float(arr.min())

    def _scale_to_master_clock(
        self,
        table: pd.DataFrame,
        timeline: Optional[GlobalTimeline],
    ) -> Optional[Dict[str, Any]]:
        if timeline is None:
            return None

        total_rows = int(len(timeline.dataframe()))

        nidaq_series = timeline.queue_series(
            lambda ids: ids.str.contains(self.timeline_filter_keyword, regex=False, na=False)
        )
        filter_label = self.timeline_filter_keyword
        if nidaq_series.empty:
            nidaq_series = timeline.queue_series()
            filter_label = self.timeline_filter_fallback_label

        if nidaq_series.empty:
            return None

        dq_rel = nidaq_series - float(nidaq_series.iloc[0])
        dq_span = float(dq_rel.iloc[-1])
        if not np.isfinite(dq_span) or dq_span <= 0:
            return None

        max_time = 0.0
        min_time = 0.0
        if len(table):
            min_series = pd.to_numeric(table["time_elapsed_s"], errors="coerce")
            if min_series.notna().any():
                min_time = float(min_series.min())

        for col in self.timeline_alignment_columns:
            if col in table:
                series = pd.to_numeric(table[col], errors="coerce")
                if series.notna().any():
                    candidate = float(series.max())
                    if np.isfinite(candidate):
                        max_time = max(max_time, candidate)

        if not np.isfinite(max_time) or max_time <= min_time:
            return None

        original_span = max_time - min_time
        if original_span <= 0:
            return None

        scale = dq_span / original_span
        if abs(scale - 1.0) <= self.timeline_scale_tolerance:
            return None

        for col in self.timeline_alignment_columns:
            if col in table:
                table[col] = pd.to_numeric(table[col], errors="coerce") * scale

        return {
            "time_basis": "dataqueue",
            "timeline_scale_factor": float(scale),
            "psychopy_span_s": float(original_span),
            "dataqueue_span_s": float(dq_span),
            "dataqueue_file": str(timeline.source_path),
            "dataqueue_rows_used": int(len(nidaq_series)),
            "dataqueue_rows_total": total_rows,
            "dataqueue_device_filter": filter_label,
        }