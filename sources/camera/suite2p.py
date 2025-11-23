"""Suite2p mesoscope plane data source aligned to the nidaq timeline.

The loader outputs a table payload with single-level columns such as::

    roi_fluorescence        → np.ndarray[n_rois, frames]
    deltaf_f                → np.ndarray[n_cells, frames]
    interpolation_method    → str
    mean_fluo_dff           → np.ndarray[frames]
    time_elapsed_s         → np.ndarray[new_frames]

This mirrors the historical pandas workflow while remaining friendly to the
standardized HDF5 writer/reader pipeline.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from datakit import logger
from datakit.sources.register import DataSource
from datakit.datamodel import LoadedStream, StreamPayload

try:  # pragma: no cover - optional dependency guard
    from scipy.interpolate import CubicSpline
    from scipy.signal import find_peaks
except Exception:  # pragma: no cover - graceful degradation when scipy missing
    CubicSpline = None
    find_peaks = None


class Suite2p(DataSource):
    """Load Suite2p plane outputs (F, Fneu, spks, stat, ops, iscell).

    The loader locates the accompanying dataqueue for the session to obtain a
    nidaq-driven frame timeline. When those anchors are unavailable, it falls back
    to the frame rate stored inside ``ops.npy``.
    """

    tag = "suite2p"
    patterns = ("**/*_suite2p/**/F.npy",)
    version = "1.0"

    required_files: Tuple[str, ...] = ("Fneu.npy", "spks.npy", "iscell.npy")
    optional_files: Tuple[str, ...] = ("ops.npy", "stat.npy")

    COLUMN_MAP: Dict[Tuple[str, str], str] = {
        ("raw", "cell_identifier"): "cell_identifier",
        ("raw", "spike_rate"): "spike_rate",
        ("raw", "stat"): "stat",
        ("raw", "ops"): "ops",
        ("raw", "plane_directory"): "plane_directory",
        ("raw", "cell_mask"): "cell_mask",
        ("processed", "roi_fluorescence"): "roi_fluorescence",
        ("processed", "neuropil_fluorescence"): "neuropil_fluorescence",
        ("processed", "deltaf_f"): "deltaf_f",
        ("processed", "interp_deltaf_f"): "interp_deltaf_f",
        ("processed", "smoothed_dff"): "smoothed_dff",
        ("processed", "interpolation_method"): "interpolation_method",
        ("analysis", "mean_deltaf_f"): "mean_fluo_dff",
        ("analysis", "peaks_prominence"): "peaks_prominence",
        ("analysis", "num_peaks_prominence"): "num_peaks_prominence",
        ("toolkit", "time_native_s"): "time_native_s",
        ("toolkit", "time_elapsed_s"): "time_elapsed_s",
    }

    NEUROPIL_SCALE = 0.3
    BASELINE_PERCENTILE = 3.0
    TARGET_RATE_HZ = 10.0
    SMOOTHING_KERNEL = 3
    PEAK_PROMINENCE = 0.3
    MIN_TIMELINE_POINTS = 2
    MIN_SPLINE_POINTS = 4
    MIN_INTERPOLATED_SAMPLES = 2

    def load(self, path: Path) -> LoadedStream:
        plane_dir = path.parent
        session_dir = self._find_session_dir(plane_dir)

        f_traces = Suite2p._ensure_float32(self._load_array(path))
        ancillary, optional_missing = self._load_companions(plane_dir, f_traces.shape)

        n_cells, n_frames = f_traces.shape
        timeline, timing_meta = self._frame_timeline(
            plane_dir, ancillary["ops"], n_frames, session_dir
        )

        cell_mask = ancillary["iscell"][:, 0].astype(bool)
        accepted_cells = int(cell_mask.sum())

        frame_rate = self._infer_frame_rate(ancillary["ops"])

        sections, processing_meta = self._process_traces(
            raw_traces=f_traces,
            neuropil_traces=ancillary["Fneu"],
            cell_mask=cell_mask,
            timeline=timeline,
            frame_rate=frame_rate,
        )

        raw_section = {
            "cell_identifier": ancillary["iscell"],
            "spike_rate": Suite2p._ensure_float32(ancillary["spks"]),
            "stat": ancillary["stat"],
            "ops": ancillary["ops"],
            "plane_directory": str(plane_dir),
            "cell_mask": cell_mask,
        }

        sections = {"raw": raw_section, **sections}

        table = self._build_payload_table(sections)
        payload = StreamPayload.table(table)

        meta = {
            "source_file": str(path),
            "plane": plane_dir.name,
            "session_dir": str(session_dir) if session_dir else None,
            "n_rois": int(n_cells),
            "n_cells": accepted_cells,
            "n_frames": int(n_frames),
            "frame_rate_hz": frame_rate,
        }
        meta.update(timing_meta)
        meta.update(processing_meta)
        if optional_missing:
            meta["missing_optional_files"] = tuple(optional_missing)

        return LoadedStream(
            tag=self.tag,
            t=timeline,
            value=payload,
            meta=meta,
        )

    # ------------------------------------------------------------------
    # Processing helpers
    # ------------------------------------------------------------------
    def _process_traces(
        self,
        *,
        raw_traces: np.ndarray,
        neuropil_traces: np.ndarray,
        cell_mask: np.ndarray,
        timeline: np.ndarray,
        frame_rate: Optional[float],
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
        timeline = np.asarray(timeline, dtype=np.float64)

        if raw_traces.size == 0 or neuropil_traces.size == 0:
            sections = self._empty_sections(timeline)
            return sections, {
                "processing_status": "no_data",
                "interpolation_rate_hz": self.TARGET_RATE_HZ,
                "neuropil_scale": self.NEUROPIL_SCALE,
                "baseline_percentile": self.BASELINE_PERCENTILE,
                "smoothing_kernel": self.SMOOTHING_KERNEL,
                "peak_prominence": self.PEAK_PROMINENCE,
            }

        # Ensure timeline aligns with traces
        if timeline.shape[0] != raw_traces.shape[1]:
            logger.warning(
                "Suite2p: timeline length mismatch",
                extra={
                    "phase": "suite2p_processing",
                    "timeline": timeline.shape,
                    "frames": raw_traces.shape,
                },
            )
            timeline = np.arange(raw_traces.shape[1], dtype=np.float64) / (frame_rate or self.TARGET_RATE_HZ)

        roi_fluo = raw_traces[cell_mask].astype(np.float32, copy=False)
        neuropil_fluo = neuropil_traces[cell_mask].astype(np.float32, copy=False)

        filtered_meta = {
            "total_rois": int(raw_traces.shape[0]),
            "filtered_cells": int(roi_fluo.shape[0]),
        }

        if roi_fluo.size == 0:
            sections = self._empty_sections(timeline)
            sections["processed"]["roi_fluorescence"] = roi_fluo
            sections["processed"]["neuropil_fluorescence"] = neuropil_fluo
            meta = {
                **filtered_meta,
                "processing_status": "no_cells",
                "interpolation_rate_hz": self.TARGET_RATE_HZ,
                "neuropil_scale": self.NEUROPIL_SCALE,
                "baseline_percentile": self.BASELINE_PERCENTILE,
                "smoothing_kernel": self.SMOOTHING_KERNEL,
                "peak_prominence": self.PEAK_PROMINENCE,
            }
            return sections, meta

        baseline = np.percentile(
            roi_fluo,
            self.BASELINE_PERCENTILE,
            axis=1,
            keepdims=True,
        ).astype(np.float32, copy=False)
        baseline_mask = np.abs(baseline) > np.finfo(np.float64).eps
        deltaf_f = np.full_like(roi_fluo, np.nan, dtype=np.float32)
        np.divide(
            roi_fluo - baseline,
            baseline,
            out=deltaf_f,
            where=baseline_mask,
        )

        interp_result = self._resample_traces(
            traces=roi_fluo,
            timeline=timeline,
            target_rate=self.TARGET_RATE_HZ,
        )

        interpolated = Suite2p._ensure_float32(interp_result["values"])
        interp_times = Suite2p._ensure_float32(interp_result["timeline"])
        interpolation_method = interp_result["method"]
        interpolation_status = interp_result["status"]

        interp_baseline = np.percentile(
            interpolated,
            self.BASELINE_PERCENTILE,
            axis=1,
            keepdims=True,
        ).astype(np.float32, copy=False) if interpolated.size else np.empty((roi_fluo.shape[0], 0), dtype=np.float32)

        interp_baseline_mask = np.abs(interp_baseline) > np.finfo(np.float64).eps
        interp_deltaf_f = np.full_like(interpolated, np.nan, dtype=np.float32)
        if interpolated.size:
            np.divide(
                interpolated - interp_baseline,
                interp_baseline,
                out=interp_deltaf_f,
                where=interp_baseline_mask,
            )

        smoothed = self._moving_average(interp_deltaf_f, self.SMOOTHING_KERNEL)

        if interp_deltaf_f.size:
            mean_deltaf = np.nanmean(interp_deltaf_f, axis=0).astype(np.float32, copy=False)
        else:
            mean_deltaf = np.empty((0,), dtype=np.float32)

        if mean_deltaf.size and find_peaks is not None:
            peaks, _ = find_peaks(mean_deltaf, prominence=self.PEAK_PROMINENCE)
        else:
            peaks = np.empty((0,), dtype=int)

        sections = {
            "processed": {
                "roi_fluorescence": roi_fluo,
                "neuropil_fluorescence": neuropil_fluo,
                "deltaf_f": deltaf_f,
                "interp_deltaf_f": interp_deltaf_f,
                "smoothed_dff": smoothed,
                "interpolation_method": interpolation_method,
            },
            "analysis": {
                "mean_deltaf_f": mean_deltaf,
                "peaks_prominence": peaks.astype(np.int32, copy=False),
                "num_peaks_prominence": int(peaks.size),
            },
            "toolkit": {
                "time_native_s": Suite2p._ensure_float32(timeline),
                "time_elapsed_s": interp_times,
            },
        }

        meta = {
            **filtered_meta,
            "processing_status": interpolation_status,
            "interpolation_rate_hz": self.TARGET_RATE_HZ,
            "interpolation_method": interpolation_method,
            "neuropil_scale": self.NEUROPIL_SCALE,
            "baseline_percentile": self.BASELINE_PERCENTILE,
            "smoothing_kernel": self.SMOOTHING_KERNEL,
            "peak_prominence": self.PEAK_PROMINENCE,
            "interpolated_frames": int(interpolated.shape[1]) if interpolated.ndim == 2 else 0,
        }

        if baseline.size:
            meta["baseline_percentile_mean"] = float(np.nanmean(baseline))
        if interp_baseline.size:
            meta["interp_baseline_percentile_mean"] = float(np.nanmean(interp_baseline))

        return sections, meta

    def _empty_sections(self, timeline: np.ndarray) -> Dict[str, Dict[str, Any]]:
        frames = int(timeline.size)
        processed = {
            "roi_fluorescence": np.empty((0, frames), dtype=np.float32),
            "neuropil_fluorescence": np.empty((0, frames), dtype=np.float32),
            "deltaf_f": np.empty((0, frames), dtype=np.float32),
            "interp_deltaf_f": np.empty((0, 0), dtype=np.float32),
            "smoothed_dff": np.empty((0, 0), dtype=np.float32),
            "interpolation_method": "unavailable",
        }
        analysis = {
            "mean_deltaf_f": np.empty((0,), dtype=np.float32),
            "peaks_prominence": np.empty((0,), dtype=int),
            "num_peaks_prominence": 0,
        }
        toolkit = {
            "time_native_s": Suite2p._ensure_float32(timeline),
            "time_elapsed_s": np.empty((0,), dtype=np.float32),
        }
        return {
            "processed": processed,
            "analysis": analysis,
            "toolkit": toolkit,
        }

    def _build_payload_table(self, sections: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        flat_data: Dict[str, list[Any]] = {}

        for section, items in sections.items():
            if not items:
                continue

            for field, value in items.items():
                column_name = self.COLUMN_MAP.get((section, field))
                if column_name is None:
                    raise KeyError(f"Suite2p column mapping missing for section '{section}' field '{field}'")

                row_labels: list[str] | None = None
                data_value = value

                if isinstance(value, tuple) and len(value) == 2:
                    candidate, labels = value
                    if isinstance(labels, Iterable) and not isinstance(labels, (str, bytes)):
                        row_labels = []
                        for idx, lbl in enumerate(labels):
                            label_text = str(lbl).strip().replace(" ", "_")
                            row_labels.append(label_text or f"component_{idx}")
                        data_value = candidate
                    else:
                        data_value = candidate

                coerced = self._coerce_table_value(data_value)

                if row_labels:
                    components = self._expand_components(coerced, len(row_labels))
                    for idx, label in enumerate(row_labels):
                        flat_data[f"{column_name}_{label}"] = [components[idx]]
                else:
                    flat_data[column_name] = [coerced]

        frame = pd.DataFrame(flat_data)
        frame.index = pd.Index([0], name="record")
        return frame

    @staticmethod
    def _coerce_table_value(value: Any) -> Any:
        if isinstance(value, pd.Series):
            return Suite2p._coerce_table_value(value.to_numpy(copy=False))
        if isinstance(value, pd.Index):
            return Suite2p._coerce_table_value(value.to_numpy(copy=False))
        if isinstance(value, np.ndarray):
            if value.dtype == object:
                return [Suite2p._coerce_table_value(item) for item in value.tolist()]
            return Suite2p._ensure_numeric_dtype(value)
        if isinstance(value, (list, tuple)):
            if not value:
                return np.asarray(value)
            try:
                array = np.asarray(value)
            except Exception:
                array = None
            if array is not None and array.dtype != object:
                return Suite2p._ensure_numeric_dtype(array)
            return [Suite2p._coerce_table_value(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    @staticmethod
    def _expand_components(value: Any, expected: int) -> list[Any]:
        if isinstance(value, list) and len(value) >= expected:
            return value
        if isinstance(value, list) and value:
            first = value[0]
            return [first for _ in range(expected)]
        if isinstance(value, np.ndarray):
            if value.ndim >= 1 and value.shape[0] >= expected:
                return [value[idx] for idx in range(expected)]
            return [value for _ in range(expected)]
        return [value for _ in range(expected)]

    @staticmethod
    def _ensure_numeric_dtype(array: np.ndarray) -> np.ndarray:
        if array.dtype.kind in {"i", "u"}:
            return array.astype(np.int32, copy=False)
        if array.dtype.kind in {"f"} and array.dtype != np.float32:
            return array.astype(np.float32, copy=False)
        return array

    @staticmethod
    def _ensure_float32(array: np.ndarray | Iterable[float]) -> np.ndarray:
        arr = np.asarray(array)
        if arr.dtype == np.float32:
            return arr
        return arr.astype(np.float32, copy=False)

    @staticmethod
    def _cast_nested_numeric(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: Suite2p._cast_nested_numeric(val) for key, val in value.items()}
        if isinstance(value, list):
            return [Suite2p._cast_nested_numeric(val) for val in value]
        if isinstance(value, tuple):
            return tuple(Suite2p._cast_nested_numeric(val) for val in value)
        if isinstance(value, np.ndarray):
            if value.dtype == object:
                return [Suite2p._cast_nested_numeric(item) for item in value.tolist()]
            return Suite2p._ensure_numeric_dtype(value)
        if isinstance(value, np.generic):
            kind = value.dtype.kind
            if kind == "f" and value.dtype != np.float32:
                return np.float32(value)
            if kind in {"i", "u"} and value.itemsize > np.dtype(np.int32).itemsize:
                return np.int32(value)
            return value.item()
        return value

    def _resample_traces(
        self,
        *,
        traces: np.ndarray,
        timeline: np.ndarray,
        target_rate: float,
    ) -> Dict[str, Any]:
        traces = np.asarray(traces, dtype=np.float64)
        timeline = np.asarray(timeline, dtype=np.float64)

        n_rois = traces.shape[0]
        empty_result = {
            "values": np.empty((n_rois, 0), dtype=np.float64),
            "timeline": np.empty((0,), dtype=np.float64),
            "method": "unavailable",
            "status": "insufficient_timeline",
        }

        if timeline.ndim != 1 or timeline.size != traces.shape[1]:
            logger.warning(
                "Suite2p: timeline shape mismatch",
                extra={"phase": "suite2p_processing", "timeline": timeline.shape, "frames": traces.shape},
            )
            return empty_result

        if timeline.size < self.MIN_TIMELINE_POINTS:
            return empty_result

        if not np.all(np.isfinite(timeline)):
            return {**empty_result, "status": "invalid_timeline"}

        normalized_time = timeline - timeline[0]
        duration = normalized_time[-1]

        if duration <= 0 or not np.isfinite(duration):
            logger.warning(
                "Suite2p: invalid duration for interpolation",
                extra={"phase": "suite2p_processing", "duration": float(duration)},
            )
            return {**empty_result, "status": "invalid_duration"}

        if not np.isfinite(target_rate) or target_rate <= 0:
            return {
                "values": traces,
                "timeline": normalized_time,
                "method": "native",
                "status": "native_rate",
            }

        step = 1.0 / target_rate
        n_samples = max(int(np.floor(duration / step)) + 1, self.MIN_INTERPOLATED_SAMPLES)
        new_timeline = np.linspace(0.0, step * (n_samples - 1), n_samples, dtype=np.float64)

        try:
            if CubicSpline is not None and normalized_time.size >= self.MIN_SPLINE_POINTS:
                spline = CubicSpline(normalized_time, traces, axis=1)
                interpolated = spline(new_timeline)
                method = "cubic_spline"
            else:
                interpolated = np.vstack([
                    np.interp(new_timeline, normalized_time, row) for row in traces
                ])
                method = "linear"
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Suite2p: interpolation failed",
                extra={
                    "phase": "suite2p_processing",
                    "error": repr(exc),
                    "method": "cubic_spline" if CubicSpline is not None else "linear",
                },
            )
            return {**empty_result, "status": "interpolation_failed"}

        return {
            "values": interpolated,
            "timeline": new_timeline,
            "method": method,
            "status": "ok",
        }

    @staticmethod
    def _moving_average(data: np.ndarray, window: int) -> np.ndarray:
        if data.size == 0 or window <= 1:
            return data

        kernel = np.ones(window, dtype=np.float64) / float(window)
        smoothed = np.empty_like(data)
        for idx, row in enumerate(data):
            smoothed[idx] = np.convolve(row, kernel, mode="same")
        return Suite2p._ensure_float32(smoothed)

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load_array(path: Path) -> np.ndarray:
        arr = np.load(path, allow_pickle=False)
        if not isinstance(arr, np.ndarray):
            raise ValueError(f"Expected numpy array in {path}, got {type(arr)}")
        return arr

    def _load_companions(self, plane_dir: Path, f_shape: Tuple[int, int]) -> Tuple[Dict[str, Any], List[str]]:
        required = tuple(self.required_files)
        optional = tuple(getattr(self, "optional_files", ()))

        missing = [name for name in required if not (plane_dir / name).exists()]
        if missing:
            raise FileNotFoundError(f"Suite2p plane missing companion files {missing} in {plane_dir}")

        companions: Dict[str, Any] = {}
        companions["Fneu"] = Suite2p._ensure_float32(self._load_array(plane_dir / "Fneu.npy"))
        companions["spks"] = Suite2p._ensure_float32(self._load_array(plane_dir / "spks.npy"))
        companions["iscell"] = Suite2p._ensure_numeric_dtype(
            np.load(plane_dir / "iscell.npy", allow_pickle=False)
        )

        expected_cells, expected_frames = f_shape
        for key in ("Fneu", "spks"):
            arr = companions[key]
            if arr.shape[0] != expected_cells or arr.shape[1] != expected_frames:
                raise ValueError(
                    f"Suite2p file {key}.npy shape {arr.shape} does not match F.npy shape {f_shape}"
                )
        if companions["iscell"].shape[0] != expected_cells:
            raise ValueError(
                f"Suite2p file iscell.npy first dimension {companions['iscell'].shape[0]}"
                f" does not match number of ROIs {expected_cells}"
            )

        missing_optional: List[str] = []

        stat_file = "stat.npy"
        expect_stat = stat_file in required or stat_file in optional
        stat_path = plane_dir / stat_file
        if expect_stat and stat_path.exists():
            stat_raw = np.load(stat_path, allow_pickle=True)
            stat_payload = stat_raw.tolist() if hasattr(stat_raw, "tolist") else stat_raw
            companions["stat"] = Suite2p._cast_nested_numeric(stat_payload)
        else:
            if expect_stat and not stat_path.exists():
                missing_optional.append(stat_file)
            companions["stat"] = []

        ops_file = "ops.npy"
        expect_ops = ops_file in required or ops_file in optional
        ops_path = plane_dir / ops_file
        if expect_ops and ops_path.exists():
            ops_raw = np.load(ops_path, allow_pickle=True)
            try:
                ops_payload = ops_raw.item()
            except (ValueError, AttributeError):
                ops_payload = ops_raw
            companions["ops"] = Suite2p._cast_nested_numeric(ops_payload)
        else:
            if expect_ops and not ops_path.exists():
                missing_optional.append(ops_file)
            companions["ops"] = {}

        return companions, missing_optional

    # ------------------------------------------------------------------
    # Timeline helpers
    # ------------------------------------------------------------------
    def _frame_timeline(
        self,
        plane_dir: Path,
        ops: Any,
        n_frames: int,
        session_dir: Optional[Path],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        dq_path = self._find_dataqueue(session_dir) if session_dir else None

        if dq_path:
            dq_result = self._timeline_from_dataqueue(dq_path, n_frames)
            if dq_result is not None:
                timeline, dq_meta = dq_result
                meta = {
                    "time_basis": "dataqueue",
                    "dataqueue_file": str(dq_path),
                }
                if dq_meta:
                    meta.update(dq_meta)
                return timeline, meta

        fallback = self._timeline_from_ops(ops, n_frames)
        return fallback, {
            "time_basis": "ops_framerate",
            "dataqueue_file": str(dq_path) if dq_path else None,
        }

    def _timeline_from_dataqueue(
        self,
        dq_path: Path,
        n_frames: int,
    ) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        try:
            df = pd.read_csv(dq_path, usecols=["queue_elapsed", "device_id", "payload"], low_memory=False)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(
                "Suite2p: unable to read dataqueue",
                extra={"phase": "suite2p_timeline", "dataqueue": str(dq_path), "error": repr(exc)},
            )
            return None

        mask = df["device_id"].astype(str).str.contains("nidaq", case=False, na=False)
        nidaq = df.loc[mask].copy()
        nidaq["queue_elapsed"] = pd.to_numeric(nidaq["queue_elapsed"], errors="coerce")
        nidaq.dropna(subset=["queue_elapsed"], inplace=True)

        if nidaq.empty:
            return None

        times = nidaq["queue_elapsed"].to_numpy(dtype=np.float64)
        n_anchors = int(times.size)
        if n_anchors >= n_frames:
            if n_anchors == n_frames:
                trimmed = times
                method = "direct"
            else:
                anchor_index = np.arange(n_anchors, dtype=np.float64)
                frame_index = np.linspace(0.0, float(n_anchors - 1), num=n_frames, dtype=np.float64)
                trimmed = np.interp(frame_index, anchor_index, times)
                method = "resampled"

            nidaq_meta = {
                "dataqueue_anchors": n_anchors,
                "dataqueue_alignment": method,
            }
            return trimmed, nidaq_meta

        # If there are fewer nidaq entries than frames, we cannot anchor reliably.
        logger.warning(
            "Suite2p: insufficient nidaq anchors",
            extra={
                "phase": "suite2p_timeline",
                "dataqueue": str(dq_path),
                "anchors": len(times),
                "frames": n_frames,
            },
        )
        return None

    def _timeline_from_ops(self, ops: Any, n_frames: int) -> np.ndarray:
        frame_rate = self._infer_frame_rate(ops)
        if frame_rate and frame_rate > 0:
            return np.arange(n_frames, dtype=np.float64) / frame_rate
        return np.arange(n_frames, dtype=np.float64)

    @staticmethod
    def _infer_frame_rate(ops: Any) -> Optional[float]:
        if isinstance(ops, dict):
            for key in ("fs", "framerate", "frame_rate"):
                value = ops.get(key)
                if value is None:
                    continue
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
        elif isinstance(ops, (list, tuple)):
            for item in ops:
                rate = Suite2p._infer_frame_rate(item)
                if rate:
                    return rate
        elif isinstance(ops, np.ndarray):
            if ops.ndim == 0:
                return Suite2p._infer_frame_rate(ops.item())
            for item in ops.flat:
                rate = Suite2p._infer_frame_rate(item)
                if rate:
                    return rate
        return None

    # ------------------------------------------------------------------
    # Session + dataqueue discovery
    # ------------------------------------------------------------------
    def _find_session_dir(self, plane_dir: Path) -> Optional[Path]:
        current = plane_dir
        while current != current.parent:
            if (current / "beh").is_dir() and (current / "func").is_dir():
                return current
            current = current.parent
        return None

    def _find_dataqueue(self, session_dir: Path) -> Optional[Path]:
        candidates = sorted((session_dir / "beh").glob("*_dataqueue.csv"))
        return candidates[0] if candidates else None