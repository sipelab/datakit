"""Tests for materialize output structure, flattening, and persistence."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from datakit import Dataset
from datakit.core import _flatten, _series_to_cell
from datakit.datamodel import LoadedStream


# ---------------------------------------------------------------------------
# Materialize structural assertions
# ---------------------------------------------------------------------------


def test_materialize_returns_dataframe(dataset):
    df = dataset.materialize(strict=True)
    assert isinstance(df, pd.DataFrame)


def test_materialize_preserves_inventory_index(dataset):
    df = dataset.materialize(strict=True)
    assert df.index.equals(dataset.inventory.index)
    assert df.index.names == ["Subject", "Session", "Task"]


def test_materialize_columns_are_source_feature_multiindex(dataset):
    df = dataset.materialize(strict=True)
    assert isinstance(df.columns, pd.MultiIndex)
    assert df.columns.names == ["Source", "Feature"]


def test_materialize_includes_loaded_sources(dataset):
    df = dataset.materialize(strict=True)
    sources = set(df.columns.get_level_values(0).unique())
    assert "dataqueue" in sources
    assert "treadmill" in sources


def test_materialize_meta_columns_prefixed(dataset):
    df = dataset.materialize(strict=True)
    features = df.columns.get_level_values(1).unique().tolist()
    assert "meta" in features


def test_materialize_empty_inventory_returns_empty_frame(sample_experiment_1):
    ds = Dataset.from_directory(sample_experiment_1)
    empty_inv = ds.inventory.iloc[0:0]
    empty = Dataset(empty_inv, sources=ds.sources, roots=ds.roots)
    df = empty.materialize(strict=True)
    assert df.empty
    assert df.columns.names == ["Source", "Feature"]


# ---------------------------------------------------------------------------
# Flatten unit tests
# ---------------------------------------------------------------------------


def test_series_to_cell_collapses_constant():
    s = pd.Series([3, 3, 3])
    assert _series_to_cell(s) == 3


def test_series_to_cell_returns_scalar_for_single_value():
    s = pd.Series([42])
    assert _series_to_cell(s) == 42


def test_series_to_cell_returns_array_for_varying():
    s = pd.Series([1, 2, 3])
    out = _series_to_cell(s)
    assert isinstance(out, np.ndarray)
    assert out.tolist() == [1, 2, 3]


def test_series_to_cell_empty():
    assert pd.isna(_series_to_cell(pd.Series([], dtype=float)))


def _stream(value, *, meta=None):
    return LoadedStream(tag="probe", t=np.array([0.0]), value=value, meta=meta or {})


def test_flatten_dataframe_payload_one_cell_per_column():
    frame = pd.DataFrame({"a": [1, 2, 3], "b": [9, 9, 9]})
    cells = _flatten(_stream(frame))
    assert ("probe", "a") in cells
    assert ("probe", "b") in cells
    assert cells[("probe", "b")] == 9


def test_flatten_array_payload():
    cells = _flatten(_stream(np.array([1.0, 2.0])))
    assert ("probe", "values") in cells
    assert isinstance(cells[("probe", "values")], np.ndarray)


def test_flatten_dict_payload_one_cell_per_key():
    cells = _flatten(_stream({"alpha": 1, "beta": 2}))
    assert cells[("probe", "alpha")] == 1
    assert cells[("probe", "beta")] == 2


def test_flatten_scalar_payload():
    cells = _flatten(_stream(42))
    assert cells[("probe", "value")] == 42


def test_flatten_includes_meta_columns():
    cells = _flatten(_stream(np.array([1.0]), meta={"sample_rate": 1000}))
    assert ("probe", "meta") in cells
    assert cells[("probe", "meta")] == {"sample_rate": 1000}


def test_flatten_skips_session_scoped_meta():
    cells = _flatten(_stream(np.array([1.0]), meta={"scope": "session", "k": "v"}))
    assert all(f != "meta" for (_, f) in cells.keys())


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_save_pickle_roundtrip(dataset, tmp_path):
    out = tmp_path / "ds.pkl"
    dataset.save(out)
    assert out.exists()
    df = pd.read_pickle(out)
    assert isinstance(df, pd.DataFrame)
    assert isinstance(df.columns, pd.MultiIndex)


def test_save_default_format_is_pickle(dataset, tmp_path):
    out = tmp_path / "no_extension"
    dataset.save(out)
    df = pd.read_pickle(out)
    assert isinstance(df, pd.DataFrame)


def test_save_explicit_pickle_format(dataset, tmp_path):
    out = tmp_path / "data.bin"
    dataset.save(out, format="pickle")
    assert out.exists()


def test_save_unsupported_format_raises(dataset, tmp_path):
    with pytest.raises(ValueError):
        dataset.save(tmp_path / "x", format="parquet")


def test_save_hdf5_extension_inferred(dataset, tmp_path):
    pytest.importorskip("tables")
    out = tmp_path / "ds.h5"
    dataset.save(out)
    df = pd.read_hdf(out, key="dataset")
    assert isinstance(df, pd.DataFrame)
