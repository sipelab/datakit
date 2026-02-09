"""Minimal treadmill loader aligned to dataqueue window (MVP)."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

from datakit.config import settings
from datakit.datamodel import LoadedStream
from datakit.sources.register import DataSource
from datakit.timeline import GlobalTimeline
from ..._utils._logger import get_logger

logger = get_logger(__name__)


class TreadmillSource(DataSource):
    tag = "treadmill"
    patterns = ("**/*_treadmill.csv", "**/*_treadmill_data.csv")
    camera_tag = None
    version = "3.0"
    timeline_columns = ("time_elapsed_s",)

    timestamp_column = "timestamp"
    distance_columns = ("distance_mm", "distance")
    speed_columns = ("speed_mm", "speed_mm_s", "speed")
    output_distance_column = "distance_mm"
    output_speed_column = "speed_mm"

    dataqueue_glob = settings.timeline.dataqueue_glob
    dataqueue_queue_column = settings.timeline.queue_column
    dataqueue_device_id_column = "device_id"
    dataqueue_payload_column = "payload"

    dataqueue_payload_prefix = "EncoderData"
    timeline_master_patterns = ("dhyana", "mesoscope")
    dataqueue_search_dirs = ("", "beh")

    _encoder_ts_re = re.compile(r"timestamp\s*=\s*(\d+)", re.IGNORECASE)

    def load(self, path: Path) -> LoadedStream:
        df = pd.read_csv(path)
        dist = next((c for c in self.distance_columns if c in df.columns), None)
        speed = next((c for c in self.speed_columns if c in df.columns), None)
        if self.timestamp_column not in df.columns or dist is None or speed is None:
            raise ValueError(f"Expected timestamp/distance/speed columns in {path}")
        df = df.rename(columns={dist: self.output_distance_column, speed: self.output_speed_column})

        dq_path = self._find_dataqueue(path)
        if not dq_path:
            raise ValueError(f"No dataqueue found for treadmill file: {path}")

        try:
            return self._load_with_dataqueue(df, path, dq_path)
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "TreadmillSource: dataqueue alignment failed (%s)",
                repr(exc),
                extra={"phase": "treadmill_dataqueue", "path": str(path), "dataqueue_path": str(dq_path)},
            )
            raise

    # --- alignment --------------------------------

    def _load_with_dataqueue(self, df: pd.DataFrame, path: Path, dq_path: Path) -> LoadedStream:
        aligned, meta = self.extract_treadmill_aligned(df, dq_path)

        return LoadedStream(
            tag=self.tag,
            t=aligned["time_elapsed_s"].to_numpy(dtype=np.float64),
            value=aligned,
            meta={"source_file": str(path), "dataqueue_file": str(dq_path), "n_samples": len(aligned), **meta},
        )

    def extract_treadmill_aligned(self, treadmill_df: pd.DataFrame, dq_path: Path) -> tuple[pd.DataFrame, dict]:
        t0, t1 = self._window(dq_path)
        duration = float(t1 - t0)
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError(f"Invalid camera window duration from dataqueue: start={t0}, end={t1}")

        enc = self._encoder_rows(dq_path)
        a, b = self._fit(enc["encoder_ts"].to_numpy(), enc["queue_elapsed"].to_numpy())

        ts = pd.to_numeric(treadmill_df[self.timestamp_column], errors="coerce").to_numpy(dtype=np.float64)
        time_s = a * ts + b - t0

        mask = np.isfinite(time_s) & (time_s >= 0.0) & (time_s <= duration)
        if not np.any(mask):
            raise ValueError("No treadmill samples fall inside the Dhyana window")

        aligned = treadmill_df.loc[mask].copy()
        aligned["time_elapsed_s"] = time_s[mask]
        aligned.sort_values("time_elapsed_s", inplace=True)
        aligned.reset_index(drop=True, inplace=True)

        return aligned, {
            "source_method": "dataqueue_align_mvp",
            "experiment_window": {"start": float(t0), "end": float(t1)},
            "alignment": {"a": float(a), "b": float(b)},
        }

    # --- dataqueue utilities -----------------------------------------------------

    def _find_dataqueue(self, treadmill_path: Path) -> Path | None:
        if treadmill_path.name and "_treadmill" in treadmill_path.name:
            candidate = treadmill_path.with_name(treadmill_path.name.replace("_treadmill", "_dataqueue"))
            if candidate.exists():
                return candidate
        base = treadmill_path.parent
        for rel in self.dataqueue_search_dirs:
            root = base if rel == "" else base.parent / rel
            if not root.exists():
                continue
            found = sorted(root.glob(self.dataqueue_glob))
            if found:
                return found[0]
        return None

    def _window(self, dq_path: Path) -> tuple[float, float]:
        dq = pd.read_csv(
            dq_path,
            usecols=[self.dataqueue_queue_column, self.dataqueue_device_id_column],
            low_memory=False,
        )
        device = dq.get(self.dataqueue_device_id_column, pd.Series(dtype=str)).astype(str)

        mask = np.zeros(len(dq), dtype=bool)
        for pattern in self.timeline_master_patterns:
            mask |= device.str.contains(pattern, case=False, na=False, regex=False).to_numpy()

        rows = pd.to_numeric(dq.loc[mask, self.dataqueue_queue_column], errors="coerce").dropna()
        if len(rows) < 2:
            raise ValueError("Could not find >=2 master camera rows in dataqueue")
        return float(rows.iloc[0]), float(rows.iloc[-1])

    def _encoder_rows(self, dq_path: Path) -> pd.DataFrame:
        dq = pd.read_csv(
            dq_path,
            usecols=[self.dataqueue_queue_column, self.dataqueue_payload_column],
            low_memory=False,
        )
        payload = dq[self.dataqueue_payload_column].astype(str)

        # strict parse only from EncoderData payloads with timestamp=...
        mask = payload.str.contains(self.dataqueue_payload_prefix, na=False)
        if not np.any(mask):
            raise ValueError("No EncoderData payloads found in dataqueue")

        enc = dq.loc[mask, [self.dataqueue_queue_column, self.dataqueue_payload_column]].copy()
        enc["encoder_ts"] = enc[self.dataqueue_payload_column].apply(self._parse_encoder_ts)

        enc = enc.dropna(subset=["encoder_ts", self.dataqueue_queue_column])
        enc = enc.rename(columns={self.dataqueue_queue_column: "queue_elapsed"})
        enc["queue_elapsed"] = pd.to_numeric(enc["queue_elapsed"], errors="coerce")
        enc = enc.dropna(subset=["queue_elapsed"])

        if len(enc) < 2:
            raise ValueError("Insufficient encoder samples for alignment")
        return enc

    def _parse_encoder_ts(self, payload: str) -> int | None:
        m = self._encoder_ts_re.search(payload or "")
        return int(m.group(1)) if m else None

    def _fit(self, encoder_ts: np.ndarray, queue_elapsed: np.ndarray) -> tuple[float, float]:
        x = np.asarray(encoder_ts, dtype=np.float64)
        y = np.asarray(queue_elapsed, dtype=np.float64)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if x.size < 2:
            raise ValueError("Insufficient encoder samples for alignment")

        # Centered affine fit (stable for large microsecond timestamps)
        x0 = float(x.mean())
        y0 = float(y.mean())
        xc = x - x0
        yc = y - y0
        denom = float(np.dot(xc, xc))
        if denom <= 0:
            raise ValueError("Degenerate encoder timestamps for alignment")

        a = float(np.dot(xc, yc) / denom)
        b = float(y0 - a * x0)

        if not np.isfinite(a) or not np.isfinite(b):
            raise ValueError("Encoder alignment fit returned non-finite coefficients")
        return a, b



class TreadmillSourceV2(DataSource):
    """Load treadmill behavioral data with proper wraparound handling and time alignment."""
    tag = "treadmill"
    patterns = ("**/*_treadmill.csv", "**/*_treadmill_data.csv")
    camera_tag = None  # Not bound to camera
    version = "2.0"  # Enhanced version with improved alignment
    timeline_columns = ("time_elapsed_s",)
    # Treat any pause longer than this gap as a true break in the data.
    smoothing_gap_threshold_s = 0.5
    # Resample the smoothed trace to this many evenly spaced points per second.
    smoothing_interpolate_hz = 20
    # Size of the moving window used for gentle smoothing.
    smoothing_window_length = 5
    # Order of the fitted curve within each smoothing window.
    smoothing_polyorder = 2
    # Median filter window used to knock out sharp spikes before smoothing.
    smoothing_median_size = 3
    # Minimum number of samples required to produce a smoothed trace.
    smoothing_min_points = 10
    # How many raw samples we keep before switching to a downsampled preview.
    smoothing_simplify_threshold = 1000
    # Target number of samples when producing a simplified plot.
    smoothing_downsample_target = 500
    # Small cushion around gaps where we force the speed down to zero.
    smoothing_zero_pad_offset_s = 0.05

    # Column name preferences
    timestamp_column = "timestamp"
    distance_columns = ("distance_mm", "distance")
    speed_columns = ("speed_mm", "speed_mm_s", "speed")
    output_distance_column = "distance_mm"
    output_speed_column = "speed_mm"

    # Minimum absolute speed to count the animal as moving.
    locomotion_speed_thresh = 2.0
    # Maximum pause duration (in ms) that still belongs to the same locomotion bout.
    locomotion_min_pause_ms = 1000
    # Shortest bout length (in ms) we consider meaningful movement.
    locomotion_min_bout_duration_ms = 2000
    # Distance (in mm) below which a bout is flagged as a tiny “micro” movement.
    locomotion_micro_distance_thresh = 60

    # Counter wrap value used by the encoder hardware (32-bit microsecond timer).
    encoder_rollover_value = 2 ** 32
    # Fallback scale converting seconds back to encoder ticks when timing fits poorly.
    queue_time_fallback_scale = 1e6
    # Smallest slope we accept when fitting the encoder alignment line.
    alignment_min_slope = 1e-12
    # Shortest spacing (in seconds) allowed between samples when padding.
    min_sample_dt_s = 0.01
    # Tiny tolerance used when comparing sample timing.
    sample_dt_epsilon = 1e-6
    # Tiny tolerance used when extending padding around the experiment window.
    padding_time_epsilon = 1e-6
    # Convenience conversion factors for timestamp math.
    seconds_to_microseconds = 1e6
    seconds_to_milliseconds = 1e3
    milliseconds_to_microseconds = 1e3
    absolute_device_id = "treadmill"
    
    def _fix_32bit_wraparound_ms(self, arr_ms):
        """Fix 32-bit wraparound in millisecond timestamps.
        
        Based on reference implementation for proper wraparound handling.
        """
        arr = np.asarray(arr_ms, dtype=np.int64)
        if arr.size == 0:
            return arr.astype(np.int64)
        
        wrap_value = 2**32
        corrected = arr.copy().astype(np.int64)
        offset = 0
        
        for i in range(1, len(arr)):
            if arr[i] < arr[i-1]:
                offset += wrap_value
            corrected[i] = arr[i] + offset
            
        return corrected
    
    def _parse_encoder_row(self, payload_str):
        """Parse encoder data from dataqueue payload string.
        
        Example: EncoderData(timestamp=3385473622, distance=-217.800 mm, speed=-6.250 mm/s)
        """
        match = re.search(r"timestamp\s*=\s*(\d+)", payload_str)
        if match:
            return {'encoder_ts_ms': int(match.group(1))}
        return {}

    def _encoder_timestamp_from_queue(self, queue_elapsed_s, a: float, b: float):
        """Convert queue time (seconds) back into the encoder's native counter units."""

        qe = np.asarray(queue_elapsed_s, dtype=np.float64)
        if not np.isfinite(a) or abs(a) < self.alignment_min_slope:
            mapped = qe * self.queue_time_fallback_scale
        else:
            mapped = (qe - b) / a
        if np.isscalar(queue_elapsed_s):
            return float(mapped)
        return mapped

    def _pad_to_experiment_window(
        self,
        proc: pd.DataFrame,
        t0: float,
        t1: float,
        a: float,
        b: float,
        sample_dt: float,
    ) -> pd.DataFrame:
        """Pad the treadmill stream so it spans the full experiment window."""

        duration = max(0.0, t1 - t0)
        if duration <= 0:
            return proc

        frames: list[pd.DataFrame] = [proc]
        first_time = float(proc["time_s"].iloc[0]) if not proc.empty else duration
        last_time = float(proc["time_s"].iloc[-1]) if not proc.empty else 0.0

        if proc.empty:
            base_distance = 0.0
        else:
            base_distance = float(proc["distance_mm"].iloc[-1])

        epsilon = self.padding_time_epsilon

        def _make_padding(times: np.ndarray, distance: float) -> pd.DataFrame:
            if times.size == 0:
                return pd.DataFrame(columns=proc.columns)
            qe = t0 + times
            raw_ts = np.asarray(self._encoder_timestamp_from_queue(qe, a, b), dtype=np.float64)
            return pd.DataFrame(
                {
                    "queue_elapsed_s": qe,
                    "time_s": times,
                    "distance_mm": np.full(times.shape, distance, dtype=float),
                    "speed_mm_s": np.zeros_like(times, dtype=float),
                    "raw_timestamp": raw_ts,
                    "_padding": True,
                }
            )

        if first_time > 0.0:
            start_times = np.arange(0.0, first_time, sample_dt, dtype=float)
            start_times = start_times[start_times < first_time - epsilon]
            if start_times.size == 0 or start_times[0] > epsilon:
                start_times = np.insert(start_times, 0, 0.0)
            frames.append(_make_padding(start_times, distance=0.0))

        if last_time < duration:
            end_times = np.arange(last_time + sample_dt, duration + sample_dt, sample_dt, dtype=float)
            end_times = end_times[end_times <= duration + epsilon]
            if end_times.size == 0 or end_times[-1] < duration - epsilon:
                end_times = np.append(end_times, duration)
            frames.append(_make_padding(end_times, distance=base_distance))

        padded = pd.concat(frames, ignore_index=True, copy=False)
        padded.sort_values("time_s", inplace=True)
        padded.reset_index(drop=True, inplace=True)
        return padded

    def _experiment_window(self, timeline: GlobalTimeline, dq: pd.DataFrame) -> tuple[float, float]:
        if "device_id" in dq.columns:
            device_series = dq["device_id"].astype(str)
            meso_mask = (
                device_series.str.contains("dhyana", case=False, na=False, regex=False)
                | device_series.str.contains("mesoscope", case=False, na=False, regex=False)
            )
            meso_rows = dq.loc[meso_mask, "queue_elapsed"].dropna()
            if len(meso_rows) >= 2:
                return float(meso_rows.iloc[0]), float(meso_rows.iloc[-1])

        queue = timeline.queue_series()
        if queue.empty:
            raise ValueError("GlobalTimeline queue series is empty")
        return float(queue.iloc[0]), float(queue.iloc[-1])

    def _select_encoder_rows(self, timeline: GlobalTimeline, dq: pd.DataFrame) -> pd.DataFrame:
        encoder_rows = timeline.slice(
            lambda ids: ids.str.contains("encoder", case=False, na=False, regex=False)
        ).rows
        if encoder_rows.empty and "payload" in dq.columns:
            encoder_rows = dq.loc[dq["payload"].astype(str).str.contains("EncoderData", na=False)].copy()
        else:
            encoder_rows = encoder_rows.copy()

        if encoder_rows.empty:
            raise ValueError("No encoder samples available in dataqueue for treadmill alignment")
        if "payload" not in encoder_rows.columns:
            raise ValueError("Encoder rows missing payload column for timestamp extraction")

        cols = [col for col in ("queue_elapsed", "payload") if col in encoder_rows.columns]
        if "queue_elapsed" not in cols:
            raise ValueError("Encoder rows missing queue_elapsed column")
        return encoder_rows[cols].copy()

    def _decode_encoder_rows(self, encoder_rows: pd.DataFrame) -> pd.DataFrame:
        rows = encoder_rows.copy()
        rows["encoder_ts_ms"] = rows["payload"].astype(str).apply(
            lambda s: self._parse_encoder_row(s).get("encoder_ts_ms", np.nan)
        )
        rows = rows.dropna(subset=["encoder_ts_ms"])
        if len(rows) == 0:
            raise ValueError("No encoder timestamps could be extracted from dataqueue payloads")
        return rows

    def _fit_encoder_alignment(self, encoder_rows: pd.DataFrame) -> tuple[float, float]:
        enc_ts = encoder_rows["encoder_ts_ms"].astype(np.int64).values
        enc_ts_corr = GlobalTimeline.fix_32bit_wraparound(enc_ts)
        enc_qe = pd.to_numeric(encoder_rows["queue_elapsed"], errors="coerce").astype(float).values
        X = np.vstack([enc_ts_corr.astype(np.float64), np.ones_like(enc_ts_corr, dtype=np.float64)]).T
        a, b = np.linalg.lstsq(X, enc_qe.astype(np.float64), rcond=None)[0]
        return float(a), float(b)

    def _build_aligned_frame(
        self,
        tdf: pd.DataFrame,
        a: float,
        b: float,
        t0: float,
    ) -> pd.DataFrame:
        ts_ms = pd.to_numeric(tdf["timestamp"], errors="coerce").astype("Int64").astype("float").fillna(0).astype(np.int64)
        ts_ms_corr = self._fix_32bit_wraparound_ms(ts_ms).astype(np.float64)
        qe_est = a * ts_ms_corr + b

        return pd.DataFrame(
            {
                "queue_elapsed_s": qe_est,
                "time_s": qe_est - t0,
                "distance_mm": pd.to_numeric(tdf["distance_mm"], errors="coerce").values,
                "speed_mm_s": pd.to_numeric(tdf["speed_mm"], errors="coerce").values,
                "raw_timestamp": ts_ms_corr,
            }
        )
    
    def _create_smooth_speed_trace(self, ts_s, speed, simplify=False):
        """Create a smooth speed trace with proper gap handling and noise reduction.
        
        This applies optimal preprocessing parameters for speed smoothing.
        """
        from scipy.signal import savgol_filter
        from scipy.ndimage import median_filter
        
        # Preprocessing config
        gap_threshold = self.smoothing_gap_threshold_s
        interpolate_resolution = self.smoothing_interpolate_hz
        
        # Step 1: Identify and handle large gaps
        dt = np.diff(ts_s)
        large_gaps = np.where(dt > gap_threshold)[0]
        
        # Step 2: Apply configured smoothing
        speed_filtered = speed.copy()
        if len(speed) > self.smoothing_window_length:
            # Use median filter first to remove outliers, then Savitzky-Golay
            window_length = self.smoothing_window_length
            if window_length % 2 == 0:
                window_length = max(3, window_length - 1)
            polyorder = self.smoothing_polyorder
            
            # Ensure window length is valid
            window_length = min(window_length, len(speed) if len(speed) % 2 == 1 else len(speed) - 1)
            if window_length >= 3:
                speed_filtered = median_filter(speed_filtered, size=self.smoothing_median_size)
                if window_length >= polyorder + 1:
                    speed_filtered = savgol_filter(speed_filtered, window_length, polyorder)
        
        if simplify and len(ts_s) > self.smoothing_simplify_threshold:
            # For simplified processing: downsample but preserve key features
            downsample_factor = max(2, len(ts_s) // self.smoothing_downsample_target)
            indices = np.arange(0, len(ts_s), downsample_factor)
            ts_plot = ts_s[indices]
            speed_plot = speed_filtered[indices]
            return ts_plot, speed_plot
        
        # Step 3: Create interpolated grid with configured resolution
        session_duration = ts_s[-1] - ts_s[0]
        target_resolution = interpolate_resolution
        n_points = int(session_duration * target_resolution)
        if n_points < self.smoothing_min_points:  # Minimum number of points
            n_points = len(ts_s)
        time_grid = np.linspace(ts_s[0], ts_s[-1], n_points)
        
        # Step 4: Interpolate with gap-aware zero insertion
        speed_interpolated = np.interp(time_grid, ts_s, speed_filtered)
        
        # Step 5: Set to zero where we're far from actual measurements
        for i, t in enumerate(time_grid):
            distances = np.abs(ts_s - t)
            closest_dist = np.min(distances)
            if closest_dist > gap_threshold:
                speed_interpolated[i] = 0
        
        # Step 6: Add explicit zeros at gap boundaries for cleaner lines
        final_times = []
        final_speeds = []
        
        for i in range(len(time_grid)):
            final_times.append(time_grid[i])
            final_speeds.append(speed_interpolated[i])
            
            # If next point is far away, add zero points at boundaries
            if i < len(time_grid) - 1:
                time_gap = time_grid[i+1] - time_grid[i]
                if time_gap > 2/target_resolution:  # Larger than expected
                    # Add zero point slightly after current point
                    final_times.append(time_grid[i] + self.smoothing_zero_pad_offset_s)
                    final_speeds.append(0)
                    # Add zero point slightly before next point
                    final_times.append(time_grid[i+1] - self.smoothing_zero_pad_offset_s)
                    final_speeds.append(0)
        
        return np.array(final_times), np.array(final_speeds)

    def load(self, path: Path) -> LoadedStream:
        timeline = None
        timeline_factory = getattr(GlobalTimeline, "for_directory", None)
        if callable(timeline_factory):
            try:
                timeline = timeline_factory(path.parent)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "TreadmillSource: failed to build timeline, falling back to CSV",
                    extra={"phase": "treadmill_timeline", "path": str(path), "error": repr(exc)},
                )

        logger.debug(
            "TreadmillSource.load",
            extra={
                "phase": "treadmill_load",
                "path": str(path),
                "timeline_source": str(getattr(timeline, "source_path", None)) if timeline else None,
            },
        )

        if timeline is not None:
            try:
                return self._load_with_timeline(path, timeline)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "TreadmillSource: timeline alignment failed, falling back to CSV",
                    extra={
                        "phase": "treadmill_timeline",
                        "path": str(path),
                        "timeline_source": str(getattr(timeline, "source_path", None)),
                        "error": repr(exc),
                    },
                )

        # Fallback: Use direct CSV parsing with wraparound handling
        return self._load_from_csv(path)

    def _load_with_timeline(self, treadmill_path: Path, timeline: GlobalTimeline) -> LoadedStream:
        """Load treadmill data aligned to the global timeline with minimal ceremony."""

        logger.debug(
            "TreadmillSource: using global timeline alignment",
            extra={
                "phase": "treadmill_timeline",
                "treadmill_path": str(treadmill_path),
                "timeline_source": str(timeline.source_path),
            },
        )

        tdf = pd.read_csv(treadmill_path)
        expected_cols = {"timestamp", "distance_mm", "speed_mm"}
        if not expected_cols.issubset(tdf.columns):
            raise ValueError(f"Expected columns {expected_cols} not found in {treadmill_path}")

        dq = timeline.dataframe()
        t0, t1 = self._experiment_window(timeline, dq)
        experiment_duration = max(0.0, t1 - t0)

        encoder_rows = self._select_encoder_rows(timeline, dq)
        encoder_rows = self._decode_encoder_rows(encoder_rows)
        a, b = self._fit_encoder_alignment(encoder_rows)

        proc = self._build_aligned_frame(tdf, a, b, t0)
        in_window_mask = (proc["time_s"] >= 0.0) & (proc["time_s"] <= experiment_duration)
        proc = proc.loc[in_window_mask].reset_index(drop=True)
        if len(proc) == 0:
            raise ValueError("No treadmill data found within experiment window")

        proc = proc.sort_values("time_s", kind="mergesort").reset_index(drop=True)
        if np.isfinite(proc["distance_mm"].iloc[0]):
            proc["distance_mm"] = proc["distance_mm"] - proc["distance_mm"].iloc[0]

        time_values = proc["time_s"].values.astype(np.float64)
        mono_mask = np.r_[True, np.diff(time_values) > 0]
        proc = proc.loc[mono_mask].reset_index(drop=True)

        observed_first_time = float(proc["time_s"].iloc[0])
        observed_last_time = float(proc["time_s"].iloc[-1])
        missing_start = float(max(0.0, observed_first_time))
        missing_end = float(max(0.0, experiment_duration - observed_last_time))

        valid_dt = np.diff(proc["time_s"].values.astype(np.float64))
        valid_dt = valid_dt[valid_dt > self.sample_dt_epsilon]
        if valid_dt.size:
            sample_dt = float(np.median(valid_dt))
        else:
            sample_dt = experiment_duration if experiment_duration > 0 else 1.0
        min_dt = self.min_sample_dt_s
        sample_dt = float(max(min_dt, min(sample_dt, experiment_duration if experiment_duration > 0 else sample_dt)))

        proc["_padding"] = False
        proc = self._pad_to_experiment_window(proc, t0, t1, a, b, sample_dt)

        t = proc["time_s"].values.astype(np.float64)
        speed_values = proc["speed_mm_s"].values.astype(np.float64)
        distance_values = proc["distance_mm"].values.astype(np.float64)

        timestamps_us = (t * self.seconds_to_microseconds).astype(np.int64)
        locomotion_bouts = self.detect_locomotion_bouts(timestamps_us, speed_values, distance_values)
        smooth_times, smooth_speeds = self._create_smooth_speed_trace(t, speed_values, simplify=False)

        df_output = pd.DataFrame(
            {
                "timestamp": proc["raw_timestamp"].astype(np.int64),
                "distance_mm": distance_values,
                "speed_mm": speed_values,
                "time_elapsed_s": t,
            }
        )

        elapsed_abs = timeline.absolute_for_device(self.absolute_device_id)
        if elapsed_abs:
            elapsed, absolute = elapsed_abs
            df_output["time_absolute"] = np.interp(t, elapsed, np.arange(len(absolute))).astype(int)
            df_output["time_absolute"] = df_output["time_absolute"].map(lambda i: absolute[i] if 0 <= i < len(absolute) else None)

        padded_samples = int(proc["_padding"].sum())
        observed_samples = int(len(proc) - padded_samples)

        logger.debug(
            "TreadmillSource: aligned treadmill stream",
            extra={
                "path": str(treadmill_path),
                "samples": int(len(proc)),
                "padding_samples": padded_samples,
                "cadence": sample_dt,
            },
        )

        return LoadedStream(
            tag=self.tag,
            t=t.astype(np.float64),
            value=df_output,
            meta={
                "source_file": str(treadmill_path),
                "dataqueue_file": str(timeline.source_path),
                "n_samples": len(proc),
                "source_method": "dataqueue_encoder_alignment_v2",
                "experiment_window": {"start": t0, "end": t1},
                "alignment": {"a": float(a), "b": float(b)},
                "padding": {
                    "prepended_seconds": float(missing_start),
                    "appended_seconds": float(missing_end),
                    "added_samples": padded_samples,
                    "observed_samples": observed_samples,
                    "cadence_seconds": float(sample_dt),
                },
                "locomotion_bouts": locomotion_bouts,
                "smooth_speed": {
                    "times": smooth_times,
                    "values": smooth_speeds,
                }
            }
        )
    
    def _load_from_csv(self, path: Path) -> LoadedStream:
        """Fallback: Load treadmill data directly from CSV with wraparound handling.

        Accepts both legacy columns (distance/speed) and standardized
        distance_mm/speed_mm naming; output is always distance_mm/speed_mm.
        """
        df = pd.read_csv(path)

        timestamp_col = self.timestamp_column
        distance_col = next((c for c in self.distance_columns if c in df.columns), None)
        speed_col = next((c for c in self.speed_columns if c in df.columns), None)

        if timestamp_col not in df.columns or distance_col is None or speed_col is None:
            raise ValueError(
                f"Expected columns including '{timestamp_col}' and distance/speed variants not found in {path}"
            )

        df_unwrapped = self._unwrap_treadmill_data(df, distance_col=distance_col, speed_col=speed_col)
        timestamps_us = df_unwrapped[timestamp_col].values
        t = (timestamps_us - timestamps_us[0]) / self.seconds_to_microseconds

        df_unwrapped = df_unwrapped.rename(
            columns={distance_col: self.output_distance_column, speed_col: self.output_speed_column}
        )
        df_unwrapped["time_elapsed_s"] = t

        locomotion_bouts = self.detect_locomotion_bouts(
            timestamps_us,
            df_unwrapped[self.output_speed_column].values,
            df_unwrapped[self.output_distance_column].values,
        )
        smooth_times, smooth_speeds = self._create_smooth_speed_trace(
            t, df_unwrapped[self.output_speed_column].values
        )

        return LoadedStream(
            tag=self.tag,
            t=t.astype(np.float64),
            value=df_unwrapped,
            meta={
                "source_file": str(path),
                "n_samples": len(df_unwrapped),
                "source_method": "csv_with_wraparound_correction",
                "locomotion_bouts": locomotion_bouts,
                "smooth_speed": {"times": smooth_times, "values": smooth_speeds},
            },
        )
    
    def _unwrap_treadmill_data(self, df, *, distance_col: str, speed_col: str):
        if df.empty:
            return df

        timestamps = df['timestamp'].to_numpy(dtype=np.int64, copy=True)
        timestamps = self._unwrap_teensy_timestamps(timestamps)
        timestamps = self._fix_32bit_wraparound_ms(timestamps)

        result = df.copy().reset_index(drop=True)
        result['timestamp'] = timestamps
        if distance_col in result.columns:
            result[distance_col] = result[distance_col] - result[distance_col].iloc[0]
        return result
    
    def _unwrap_teensy_timestamps(self, timestamps_us, rollover_val=None):
        """
        Standard Teensy micros() unwrapping for negative diffs.
        """
        if rollover_val is None:
            rollover_val = self.encoder_rollover_value
        timestamps_us = np.asarray(timestamps_us)
        diffs = np.diff(timestamps_us)
        rollover_indices = np.where(diffs < 0)[0]
        
        corrected = timestamps_us.copy()
        
        for i in rollover_indices:
            corrected[i + 1:] += rollover_val
            
        return corrected

    def detect_locomotion_bouts(
        self,
        ts_us,
        speed,
        distance,
        speed_thresh=None,
        min_pause_ms=None,
        min_bout_duration_ms=None,
        micro_distance_thresh=None,
    ):
        """
        Detects locomotion bouts from encoder speed data.
        
        Args:
            ts_us: timestamps in microseconds
            speed: speed array (mm/s)
            distance: cumulative distance (mm)
            speed_thresh: minimum speed to consider as movement (mm/s)
            min_pause_ms: maximum pause duration within a bout (ms)
            min_bout_duration_ms: minimum duration for a valid bout (ms)
            micro_distance_thresh: threshold for micro-movements (mm)
        
        Returns:
            list of dicts per bout with start_time, end_time, duration, distance, is_micro
        """
        ts_us = np.asarray(ts_us)
        speed = np.asarray(speed)
        distance = np.asarray(distance)

        if speed_thresh is None:
            speed_thresh = self.locomotion_speed_thresh
        if min_pause_ms is None:
            min_pause_ms = self.locomotion_min_pause_ms
        if min_bout_duration_ms is None:
            min_bout_duration_ms = self.locomotion_min_bout_duration_ms
        if micro_distance_thresh is None:
            micro_distance_thresh = self.locomotion_micro_distance_thresh

        is_moving = np.abs(speed) > speed_thresh
        bouts_idx = []
        i = 0
        while i < len(is_moving):
            if is_moving[i]:
                start = i
                # allow short pauses < min_pause_ms
                while i < len(is_moving) and (
                    is_moving[i] or
                    (i < len(is_moving) - 1 and (ts_us[i + 1] - ts_us[i] < min_pause_ms * self.milliseconds_to_microseconds))
                ):
                    i += 1
                end = i
                bouts_idx.append((start, end))
            else:
                i += 1

        bouts = []
        for start, end in bouts_idx:
            t0 = ts_us[start] / self.seconds_to_microseconds
            t1 = ts_us[end - 1] / self.seconds_to_microseconds
            dur = t1 - t0
            if dur * self.seconds_to_milliseconds < min_bout_duration_ms:
                continue
            dist = distance[end - 1] - distance[start]
            is_micro = dist < micro_distance_thresh
            bouts.append(dict(
                start_time=t0,
                end_time=t1,
                duration=dur,
                distance=dist,
                is_micro=is_micro
            ))
        return bouts
    
    def interpolate_speed_to_timebase(self, treadmill_time_s, treadmill_speed, target_time_s):
        """
        Interpolates treadmill speed to target timebase using forward-fill method.
        
        Args:
            treadmill_time_s: treadmill timestamps in seconds
            treadmill_speed: speed values (mm/s)
            target_time_s: target timebase to interpolate to (e.g., mesoscope frames)
        
        Returns:
            interpolated speed array aligned to target timebase
        """
        # Convert to pandas series for easy interpolation
        speed_series = pd.Series(treadmill_speed, index=treadmill_time_s)
        
        # Use forward-fill: maintain speed values between samples, fill with 0 when no prior data
        interp_series = speed_series.reindex(target_time_s, method='ffill', fill_value=0)
        
        return interp_series.values


# Create a legacy v1.0 version for backward compatibility
class TreadmillSourceV1(DataSource):
    """Legacy version 1.0 of TreadmillSource for backward compatibility."""
    tag = "treadmill"
    patterns = ("**/*_treadmill.csv", "**/*_treadmill_data.csv")
    camera_tag = None  # Not bound to camera
    version = "1.0"
    timeline_columns = ("time_elapsed_s", "timestamp")
    
    def load(self, path: Path) -> LoadedStream:
        """Simple version without advanced alignment features."""
        df = pd.read_csv(path)
        
        expected_cols = {'timestamp', 'distance', 'speed'}
        if not expected_cols.issubset(df.columns):
            raise ValueError(f"Expected columns {expected_cols} not found in {path}")
        
        # Convert timestamps to seconds relative to start
        timestamps_us = df['timestamp'].values
        t = (timestamps_us - timestamps_us[0]) / 1e6
        
        # Rename columns to match expected format
        df_output = df.rename(columns={
            'distance': 'distance_mm',
            'speed': 'speed_mm'
        })
        
        return LoadedStream(
            tag=self.tag,
            t=t.astype(np.float64),
            value=df_output,
            meta={
                "source_file": str(path), 
                "n_samples": len(df_output),
                "source_method": "simple_csv_v1",
                "version": "1.0"
            }
        )