"""Helpers for diagnosing treadmill alignment and interpolation artifacts."""

from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_time_gaps(time_s: np.ndarray, gap_threshold_s: float = 0.5) -> pd.DataFrame:
    """Return a dataframe of time gaps larger than the threshold."""
    t = np.asarray(time_s, dtype=np.float64)
    if t.size < 2:
        return pd.DataFrame(columns=["start_s", "end_s", "duration_s", "index_start", "index_end"])

    diffs = np.diff(t)
    gap_idx = np.where(diffs > gap_threshold_s)[0]
    rows = []
    for idx in gap_idx:
        start = float(t[idx])
        end = float(t[idx + 1])
        rows.append(
            {
                "index_start": int(idx),
                "index_end": int(idx + 1),
                "start_s": start,
                "end_s": end,
                "duration_s": float(end - start),
            }
        )
    return pd.DataFrame(rows)


def summarize_interpolation_artifacts(
    t_meso: np.ndarray,
    t_tread: np.ndarray,
    speed: np.ndarray,
    gap_threshold_s: float = 0.5,
) -> pd.DataFrame:
    """Identify meso samples that fall inside large treadmill gaps."""
    t_m = np.asarray(t_meso, dtype=np.float64)
    t_t = np.asarray(t_tread, dtype=np.float64)
    spd = np.asarray(speed, dtype=np.float64)

    gaps = summarize_time_gaps(t_t, gap_threshold_s=gap_threshold_s)
    if gaps.empty:
        return gaps

    rows = []
    for _, gap in gaps.iterrows():
        start = float(gap["start_s"])
        end = float(gap["end_s"])
        in_gap = (t_m > start) & (t_m < end)
        n_meso = int(np.count_nonzero(in_gap))
        idx_start = int(gap["index_start"])
        idx_end = int(gap["index_end"])
        speed_start = float(spd[idx_start]) if idx_start < spd.size else float("nan")
        speed_end = float(spd[idx_end]) if idx_end < spd.size else float("nan")
        rows.append(
            {
                "start_s": start,
                "end_s": end,
                "duration_s": float(end - start),
                "meso_points_in_gap": n_meso,
                "speed_before_gap": speed_start,
                "speed_after_gap": speed_end,
            }
        )
    return pd.DataFrame(rows)


def debug_treadmill_alignment(
    x,
    *,
    gap_threshold_s: float = 0.5,
    show_plot: bool = False,
):
    """Inspect treadmill gaps and where linear interpolation will occur.

    Parameters
    ----------
    x : pandas.Series
        A row from the dataset MultiIndex selection, e.g. dataset.loc[(SUBJECT, SESSION, TASK)].
    gap_threshold_s : float
        Minimum gap duration to flag.
    show_plot : bool
        If True, plots interpolated speed and highlights gap spans.
    """
    t_meso = np.asarray(x[("time", "master_elapsed_s")], dtype=np.float64)
    t_tread = np.asarray(x[("treadmill", "time_elapsed_s")], dtype=np.float64)
    speed = np.asarray(x[("treadmill", "speed_mm")], dtype=np.float64)

    gap_df = summarize_time_gaps(t_tread, gap_threshold_s=gap_threshold_s)
    artifact_df = summarize_interpolation_artifacts(
        t_meso, t_tread, speed, gap_threshold_s=gap_threshold_s
    )

    if show_plot:
        import matplotlib.pyplot as plt

        interp = np.interp(t_meso, t_tread, speed)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(t_meso, interp, color="tab:blue", linewidth=1.0, label="treadmill interp")
        for _, row in artifact_df.iterrows():
            ax.axvspan(row["start_s"], row["end_s"], color="tab:red", alpha=0.15)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Speed (mm)")
        ax.set_title("Treadmill interpolation with gap spans")
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.show()

    return {
        "gap_summary": gap_df,
        "artifact_summary": artifact_df,
    }
