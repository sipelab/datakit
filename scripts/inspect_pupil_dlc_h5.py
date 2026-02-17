"""Inspect a DeepLabCut HDF5 output and confirm access to pupil columns."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


def _print(msg: str) -> None:
    print(msg, flush=True)


def _summarize_columns(df: pd.DataFrame, limit: int = 20) -> None:
    _print(f"DataFrame shape: {df.shape}")
    _print("Columns preview:")
    cols = list(df.columns)
    for col in cols[:limit]:
        _print(f"  - {col}")
    if len(cols) > limit:
        _print(f"  ... ({len(cols) - limit} more)")


def _find_scorers(cols: Iterable) -> list[str]:
    scorers: set[str] = set()
    for col in cols:
        if isinstance(col, tuple) and len(col) >= 1:
            scorers.add(str(col[0]))
    return sorted(scorers)


def _find_bodyparts(cols: Iterable) -> list[str]:
    bodyparts: set[str] = set()
    for col in cols:
        if isinstance(col, tuple) and len(col) >= 2:
            bodyparts.add(str(col[1]))
    return sorted(bodyparts)


def _find_coords(cols: Iterable) -> list[str]:
    coords: set[str] = set()
    for col in cols:
        if isinstance(col, tuple) and len(col) >= 3:
            coords.add(str(col[2]))
    return sorted(coords)


def _load_first_key(path: Path) -> pd.DataFrame:
    with pd.HDFStore(path, mode="r") as store:
        keys = store.keys()
        if not keys:
            raise ValueError("No keys found in HDF5 file")
        _print("HDF5 keys:")
        for key in keys:
            _print(f"  - {key}")
        key = keys[0]
        _print(f"Loading key: {key}")
        return store.get(key)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Path to DLC .h5 file")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    path = args.path
    if not path.exists():
        raise FileNotFoundError(path)

    df = _load_first_key(path)
    _summarize_columns(df)

    scorers = _find_scorers(df.columns)
    bodyparts = _find_bodyparts(df.columns)
    coords = _find_coords(df.columns)

    if scorers:
        _print(f"Scorers: {scorers}")
    if bodyparts:
        _print(f"Bodyparts: {bodyparts}")
    if coords:
        _print(f"Coords: {coords}")

    if isinstance(df.columns, pd.MultiIndex) and scorers and bodyparts and coords:
        scorer = scorers[0]
        _print(f"Sample extraction using scorer: {scorer}")
        sub = df[scorer]
        _print(f"Subframe shape: {sub.shape}")
        _print("Subframe columns preview:")
        for col in list(sub.columns)[:10]:
            _print(f"  - {col}")

        if "likelihood" in coords:
            likelihood = sub.xs("likelihood", level=1, axis=1, drop_level=False)
            _print(f"Likelihood columns: {likelihood.columns.tolist()[:10]}")


if __name__ == "__main__":
    main()
