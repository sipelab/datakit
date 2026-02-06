"""Quick treadmill replot without rebuilding datasets.

Usage:
    python scripts/replot_treadmill_debug.py \
        --treadmill <treadmill_csv> \
        --dataqueue <dataqueue_csv>
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _parse_encoder_ts(payload: str) -> int | None:
    match = re.search(r"timestamp\s*=\s*(\d+)", payload)
    if not match:
        return None
    return int(match.group(1))


def _dhyana_window(dq: pd.DataFrame) -> tuple[float, float]:
    device_series = dq["device_id"].astype(str)
    mask = (
        device_series.str.contains("dhyana", case=False, na=False, regex=False)
        | device_series.str.contains("mesoscope", case=False, na=False, regex=False)
    )
    rows = dq.loc[mask, "queue_elapsed"].dropna()
    if len(rows) < 2:
        raise ValueError("No Dhyana/mesoscope rows found in dataqueue.")
    return float(rows.iloc[0]), float(rows.iloc[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Replot treadmill vs dataqueue window")
    parser.add_argument("--treadmill", required=True, type=Path)
    parser.add_argument("--dataqueue", required=True, type=Path)
    args = parser.parse_args()

    tdf = pd.read_csv(args.treadmill)
    dq = pd.read_csv(args.dataqueue)

    # Dhyana window from dataqueue
    t0, t1 = _dhyana_window(dq)

    # Encoder anchors from dataqueue
    enc = dq.loc[dq["payload"].astype(str).str.contains("EncoderData", na=False)].copy()
    enc["encoder_ts"] = enc["payload"].astype(str).apply(_parse_encoder_ts)
    enc.dropna(subset=["encoder_ts"], inplace=True)
    enc_ts = enc["encoder_ts"].astype(np.int64).to_numpy()
    enc_qe = pd.to_numeric(enc["queue_elapsed"], errors="coerce").to_numpy(dtype=np.float64)

    # Fit affine: qe = a * encoder_ts + b
    X = np.vstack([enc_ts.astype(np.float64), np.ones_like(enc_ts, dtype=np.float64)]).T
    a, b = np.linalg.lstsq(X, enc_qe, rcond=None)[0]

    # Align treadmill CSV timestamps to queue elapsed
    ts = pd.to_numeric(tdf["timestamp"], errors="coerce").astype(np.int64).to_numpy()
    qe_est = a * ts + b
    time_s = qe_est - t0

    # Clip to Dhyana window
    duration = max(0.0, t1 - t0)
    mask = (time_s >= 0.0) & (time_s <= duration)

    speed = pd.to_numeric(tdf.get("speed_mm", tdf.get("speed", np.nan)), errors="coerce").to_numpy(dtype=np.float64)

    plt.figure(figsize=(10, 4))
    plt.scatter(time_s[mask], speed[mask], s=6, alpha=0.7)
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (mm/s)")
    plt.title(f"Treadmill speed (window {duration:.1f}s)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
