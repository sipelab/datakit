"""Hover viewer for the bundled mesomap fixture.

Run:
    python -m scripts.mesomap_fixture_hover

This uses the test fixture in tests/mesomap to load the traces, mask, and
regions table, then opens an interactive matplotlib window. Hovering over the
mask shows the region label.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datakit.sources.analysis.mesomap import MesoMapMaskSource, MesoMapRegionsSource

def load_fixture_paths() -> tuple[Path, Path, Path]:
    root = Path(__file__).resolve().parents[1] / "tests" / "mesomap"
    trace = next(root.glob("*_mesoscope.ome_traces.csv"))
    mask = trace.with_suffix(".mask.npy")
    regions = trace.with_suffix(".regions.csv")
    return trace, mask, regions


def load_mask_and_regions(mask_path: Path, regions_path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    mask_loader = MesoMapMaskSource()
    regions_loader = MesoMapRegionsSource()
    mask_stream = mask_loader.load(mask_path)
    regions_stream = regions_loader.load(regions_path)
    return mask_stream.data, regions_stream.data


def build_region_lookup(regions_df: pd.DataFrame) -> Dict[int, str]:
    lookup: Dict[int, str] = {}
    for _, row in regions_df.iterrows():
        label = int(row["mask_label"])
        acronym = str(row.get("acronym", ""))
        name = str(row.get("name", ""))
        hemi = str(row.get("hemisphere", ""))
        parts = [p for p in (acronym, name, hemi) if p]
        lookup[label] = " | ".join(parts) if parts else f"label {label}"
    lookup.setdefault(0, "background")
    return lookup


def show_hover(mask: np.ndarray, regions_df: pd.DataFrame, title: str) -> None:
    lookup = build_region_lookup(regions_df)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(mask, cmap="tab20", interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")

    text = ax.text(
        0.01,
        0.99,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    def on_move(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            text.set_text("")
            fig.canvas.draw_idle()
            return
        x = int(event.xdata)
        y = int(event.ydata)
        if y < 0 or x < 0 or y >= mask.shape[0] or x >= mask.shape[1]:
            text.set_text("")
            fig.canvas.draw_idle()
            return
        label = int(mask[y, x])
        desc = lookup.get(label, f"label {label}")
        text.set_text(f"{desc}\n(x={x}, y={y})")
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    plt.show()


def main() -> None:
    trace_path, mask_path, regions_path = load_fixture_paths()
    mask, regions = load_mask_and_regions(mask_path, regions_path)
    title = f"Mesomap fixture mask | trace: {trace_path.name}"
    show_hover(mask, regions, title)


if __name__ == "__main__":
    main()
