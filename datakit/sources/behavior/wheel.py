"""Wheel encoder behavioral data source with dda synchronized timeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..._utils._logger import get_logger

from datakit.sources.register import DataSource
from datakit.datamodel import LoadedStream
from datakit.timeline import GlobalTimeline

logger = get_logger(__name__)


@dataclass(frozen=True)
class _WheelSummary:
    """Aggregate metrics extracted from a wheel run."""

    duration_s: float
    total_distance_mm: float
    total_click_delta: int
    start_ts: Optional[str]
    stop_ts: Optional[str]


class WheelEncoder(DataSource):
    """Load wheel encoder streams recorded alongside nidaq pulses.

    The raw CSV emitted by the behavioral rig contains incremental click counts,
    elapsed time in seconds, and instantaneous speed estimates. The loader
    converts this information into a strictly increasing timeline anchored to
    acquisition start, computes cumulative distance, and exposes rich metadata
    for downstream alignment against the nidaq-driven master clock.
    """

    tag = "wheel"
    patterns = ("**/*_wheel.csv",)
    camera_tag = None  # Not bound to camera
    version = "2.0"
    timeline_columns = ("time_elapsed_s", "time_reference_s")

    required_columns = ("Clicks", "Time", "Speed")
    time_column = "Time"
    click_column = "Clicks"
    speed_column = "Speed"
    cumulative_column = "click_delta"
    anchor_filter_pattern = "encoder"
    queue_elapsed_column = "queue_elapsed"
    dataqueue_payload_column = "payload"
    alignment_min_points = 2
    alignment_time_basis_dataqueue = "dataqueue"
    alignment_time_basis_wheel = "wheel_clock"
    alignment_poly_degree = 1
    distance_integration_method = "trapezoid"
    absolute_device_id = "encoder"

    def load(self, path: Path) -> LoadedStream:
        raw = pd.read_csv(path)

        if not set(self.required_columns).issubset(raw.columns):
            raise ValueError(
                f"Wheel file is missing required columns {self.required_columns}: {path}"
            )

        df = self._prepare_frame(raw)
        df_aligned, alignment_meta = self._align_to_dataqueue(df, path.parent)
        summary = self._summarize(df_aligned, raw)
        df_aligned = df_aligned.rename(columns={"time_s": "time_elapsed_s", "time_raw_s": "time_reference_s"})

        return LoadedStream(
            tag=self.tag,
            t=df_aligned["time_elapsed_s"].to_numpy(dtype=np.float64),
            value=df_aligned,
            meta={
                "source_file": str(path),
                "n_samples": int(len(df_aligned)),
                "start_time": summary.start_ts,
                "stop_time": summary.stop_ts,
                "duration_s": summary.duration_s,
                "total_distance_mm": summary.total_distance_mm,
                "total_click_delta": summary.total_click_delta,
                "source_method": "wheel_csv_v2",
                **alignment_meta,
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prepare_frame(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Type cast and derive the canonical wheel dataframe."""

        frame = pd.DataFrame()
        frame["time_s"] = pd.to_numeric(raw[self.time_column], errors="coerce")
        frame["click_delta"] = pd.to_numeric(raw[self.click_column], errors="coerce")
        frame["speed_mm"] = pd.to_numeric(raw[self.speed_column], errors="coerce")

        frame = frame.dropna(subset=["time_s"]).sort_values("time_s").reset_index(drop=True)

        if frame.empty:
            frame = pd.DataFrame(
            {
                "time_s": [0.0, 0.0],
                "click_delta": [0, 0],
                "speed_mm": [0.0, 0.0],
            }
            )

        # Ensure speed/clicks missing entries become zeros
        frame["click_delta"] = frame["click_delta"].fillna(0)
        frame["speed_mm"] = frame["speed_mm"].fillna(0.0)

        # Re-zero the timeline relative to the first available sample
        frame["time_raw_s"] = frame["time_s"].to_numpy(dtype=np.float64)
        t = frame["time_raw_s"].copy()
        t0 = float(t[0])
        t -= t0
        frame["time_s"] = t

        # Derive cumulative click position for quick sanity checks
        frame["click_position"] = np.cumsum(frame["click_delta"].to_numpy(dtype=np.int64))

        # Integrate speed to distance using trapezoidal rule
        speed = frame["speed_mm"].to_numpy(dtype=np.float64)
        distance = np.zeros_like(speed)
        if speed.size > 1:
            dt = np.diff(t)
            trapezoids = 0.5 * (speed[:-1] + speed[1:]) * dt
            distance[1:] = np.cumsum(trapezoids)
        frame["distance_mm"] = distance

        return frame[["time_s", "time_raw_s", "click_delta", "click_position", "speed_mm", "distance_mm"]]

    def _summarize(self, frame: pd.DataFrame, raw: pd.DataFrame) -> _WheelSummary:
        """Build aggregate metadata for diagnostics and alignment hints."""

        duration = float(frame["time_s"].iloc[-1]) if len(frame) else 0.0
        total_distance = float(frame["distance_mm"].iloc[-1]) if len(frame) else 0.0
        total_clicks = int(frame["click_delta"].sum())

        start_ts = self._extract_timestamp(raw.get("Started"))
        stop_ts = self._extract_timestamp(raw.get("Stopped"))

        return _WheelSummary(
            duration_s=duration,
            total_distance_mm=total_distance,
            total_click_delta=total_clicks,
            start_ts=start_ts,
            stop_ts=stop_ts,
        )

    # ------------------------------------------------------------------
    # Alignment helpers
    # ------------------------------------------------------------------
    def _align_to_dataqueue(self, frame: pd.DataFrame, directory: Path) -> tuple[pd.DataFrame, dict]:
        """Map wheel timestamps onto the nidaq master clock via dataqueue anchors."""

        timeline = GlobalTimeline.for_directory(directory)
        if timeline is None:
            return frame, {"time_basis": self.alignment_time_basis_wheel}

        encoder_slice = timeline.slice(
            lambda ids: ids.str.contains(self.anchor_filter_pattern, case=False, na=False, regex=False)
        )
        anchors = encoder_slice.rows

        if anchors.empty or self.dataqueue_payload_column not in anchors.columns:
            return frame, {
                "time_basis": self.alignment_time_basis_wheel,
                "dataqueue_file": str(timeline.source_path),
            }

        anchors = anchors[[self.queue_elapsed_column, self.dataqueue_payload_column]].copy()
        anchors[self.queue_elapsed_column] = pd.to_numeric(anchors[self.queue_elapsed_column], errors="coerce")
        anchors["click_delta"] = pd.to_numeric(anchors[self.dataqueue_payload_column], errors="coerce")
        anchors.dropna(subset=[self.queue_elapsed_column, "click_delta"], inplace=True)

        if anchors.empty:
            return frame, {
                "time_basis": self.alignment_time_basis_wheel,
                "dataqueue_file": str(timeline.source_path),
            }

        anchors["click_delta"] = anchors["click_delta"].astype(int)
        anchors["click_position"] = anchors["click_delta"].cumsum()
        anchors["queue_rel"] = anchors[self.queue_elapsed_column] - anchors[self.queue_elapsed_column].iloc[0]

        wheel_points = frame.loc[frame["click_delta"] != 0, ["click_position", "time_s"]].copy()
        queue_points = anchors.loc[anchors["click_delta"] != 0, ["click_position", "queue_rel"]].copy()

        if wheel_points.empty or queue_points.empty:
            return frame, {
                "time_basis": self.alignment_time_basis_wheel,
                "dataqueue_file": str(timeline.source_path),
            }

        wheel_points["click_position"] = (
            wheel_points["click_position"] - wheel_points["click_position"].iloc[0]
        )
        queue_points["click_position"] = (
            queue_points["click_position"] - queue_points["click_position"].iloc[0]
        )
        wheel_points["time_s"] = wheel_points["time_s"] - wheel_points["time_s"].iloc[0]

        wheel_points = wheel_points.drop_duplicates("click_position")
        queue_points = queue_points.drop_duplicates("click_position")

        merged = wheel_points.merge(queue_points, on="click_position", how="inner")

        if len(merged) < self.alignment_min_points:
            logger.warning(
                "WheelEncoder: insufficient anchors for alignment",
                extra={
                    "phase": "wheel_align",
                    "dataqueue": str(timeline.source_path),
                    "anchors": len(merged),
                },
            )
            return frame, {
                "time_basis": self.alignment_time_basis_wheel,
                "dataqueue_file": str(timeline.source_path),
                "alignment_anchors": len(merged),
            }

        x = merged["time_s"].to_numpy(dtype=np.float64)
        y = merged["queue_rel"].to_numpy(dtype=np.float64)
        slope, intercept = np.polyfit(x, y, self.alignment_poly_degree)

        predicted = slope * x + intercept
        residual = y - predicted
        ss_res = float(np.sum(residual ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2)) if len(y) > 1 else 0.0
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        aligned = frame.copy()
        aligned["time_s"] = slope * aligned["time_s"] + intercept

        elapsed_abs = timeline.absolute_for_device(self.absolute_device_id)
        if elapsed_abs:
            elapsed, absolute = elapsed_abs
            aligned_t = aligned["time_s"].to_numpy(dtype=np.float64)
            aligned["time_absolute"] = np.interp(aligned_t, elapsed, np.arange(len(absolute))).astype(int)
            aligned["time_absolute"] = aligned["time_absolute"].map(
                lambda i: absolute[i] if 0 <= i < len(absolute) else None
            )

        return aligned, {
            "time_basis": self.alignment_time_basis_dataqueue,
            "dataqueue_file": str(timeline.source_path),
            "alignment_coefficients": {"a": float(slope), "b": float(intercept)},
            "alignment_r2": float(r2),
            "alignment_anchors": int(len(merged)),
            "queue_elapsed_start": float(anchors[self.queue_elapsed_column].iloc[0]),
        }

    @staticmethod
    def _extract_timestamp(series: Optional[pd.Series]) -> Optional[str]:
        """Return an ISO8601 string if the provided column encodes a timestamp."""

        if series is None:
            return None

        first = series.dropna().astype(str).head(1)
        if first.empty:
            return None

        try:
            ts = pd.to_datetime(first.iloc[0], utc=False, errors="raise")
        except (TypeError, ValueError):
            return None

        return ts.isoformat()