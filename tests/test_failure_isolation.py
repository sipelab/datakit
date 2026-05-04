"""Tests for materialize/validate failure isolation and reporting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from datakit import Dataset


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------


def test_validate_returns_dataframe(dataset):
    v = dataset.validate()
    assert isinstance(v, pd.DataFrame)


def test_validate_columns(dataset):
    v = dataset.validate()
    expected = {"Subject", "Session", "Task", "Source",
                "status", "path", "error_type", "message", "traceback"}
    assert expected.issubset(set(v.columns))


def test_validate_one_row_per_cell_source(dataset):
    v = dataset.validate()
    assert len(v) == len(dataset.inventory) * len(dataset.sources)


def test_validate_status_values(dataset):
    v = dataset.validate()
    assert set(v["status"].unique()).issubset({"ok", "missing", "error"})


def test_validate_does_not_raise_on_broken_path(dataset, tmp_path):
    """A nonexistent path should be reported as 'error' rather than raised."""
    inv = dataset.inventory
    bad = tmp_path / "does_not_exist.csv"
    inv.iloc[0, inv.columns.get_loc("dataqueue")] = str(bad)
    ds = Dataset(inv, sources=dataset.sources, roots=dataset.roots)
    v = ds.validate()
    dq_row = v[v["Source"] == "dataqueue"].iloc[0]
    assert dq_row["status"] == "error"
    assert dq_row["error_type"]
    assert dq_row["message"]
    assert dq_row["traceback"]


def test_validate_missing_path_status(dataset):
    inv = dataset.inventory
    inv.iloc[0, inv.columns.get_loc("treadmill")] = np.nan
    ds = Dataset(inv, sources=dataset.sources, roots=dataset.roots)
    v = ds.validate()
    treadmill_row = v[v["Source"] == "treadmill"].iloc[0]
    assert treadmill_row["status"] == "missing"
    assert treadmill_row["path"] is None


# ---------------------------------------------------------------------------
# Materialize: strict=True
# ---------------------------------------------------------------------------


def test_materialize_strict_true_raises_on_error(dataset, tmp_path):
    inv = dataset.inventory
    inv.iloc[0, inv.columns.get_loc("dataqueue")] = str(tmp_path / "missing.csv")
    ds = Dataset(inv, sources=dataset.sources, roots=dataset.roots)
    with pytest.raises(RuntimeError) as exc:
        ds.materialize(strict=True)
    msg = str(exc.value)
    assert "dataqueue" in msg
    assert "STREHAB07" in msg


def test_materialize_strict_true_default(dataset, tmp_path):
    inv = dataset.inventory
    inv.iloc[0, inv.columns.get_loc("dataqueue")] = str(tmp_path / "missing.csv")
    ds = Dataset(inv, sources=dataset.sources, roots=dataset.roots)
    with pytest.raises(RuntimeError):
        ds.materialize()  # default strict=True


# ---------------------------------------------------------------------------
# Materialize: strict=False
# ---------------------------------------------------------------------------


def test_materialize_strict_false_continues(dataset, tmp_path):
    inv = dataset.inventory
    inv.iloc[0, inv.columns.get_loc("dataqueue")] = str(tmp_path / "missing.csv")
    ds = Dataset(inv, sources=dataset.sources, roots=dataset.roots)
    df = ds.materialize(strict=False)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(inv)


def test_materialize_strict_false_returns_errors(dataset, tmp_path):
    inv = dataset.inventory
    inv.iloc[0, inv.columns.get_loc("dataqueue")] = str(tmp_path / "missing.csv")
    ds = Dataset(inv, sources=dataset.sources, roots=dataset.roots)
    df, errs = ds.materialize(strict=False, return_errors=True)
    assert isinstance(errs, pd.DataFrame)
    err_rows = errs[errs["status"] == "error"]
    assert len(err_rows) >= 1
    assert (err_rows["Source"] == "dataqueue").any()


def test_materialize_no_errors_clean_run(dataset):
    df, errs = dataset.materialize(strict=False, return_errors=True)
    assert len(errs[errs["status"] == "error"]) == 0
