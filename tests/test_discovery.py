"""Tests for `Dataset.from_directory` discovery and inventory shape."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from datakit import Dataset


def test_from_directory_returns_dataset(dataset):
    assert isinstance(dataset, Dataset)
    assert len(dataset.inventory) == 1


def test_inventory_has_expected_index_levels(dataset):
    inv = dataset.inventory
    assert isinstance(inv.index, pd.MultiIndex)
    assert inv.index.names == ["Subject", "Session", "Task"]
    assert dataset.has_task_level


def test_inventory_columns_are_source_tags(dataset):
    inv = dataset.inventory
    expected_subset = {
        "dataqueue", "meso_mean", "meso_metadata",
        "notes", "pupil_metadata", "session_config",
        "timestamps", "treadmill",
    }
    assert expected_subset.issubset(set(inv.columns))


def test_subjects_sessions(dataset):
    assert dataset.subjects == ["STREHAB07"]
    assert dataset.sessions == ["ses-05"]


def test_sources_property_subset_of_columns(dataset):
    assert set(dataset.sources).issubset(set(dataset.inventory.columns))


def test_roots_recorded(dataset, sample_experiment_1):
    assert len(dataset.roots) == 1
    assert dataset.roots[0].resolve() == sample_experiment_1.resolve()


def test_multi_directory_concatenates(multi_dataset):
    assert len(multi_dataset.inventory) == 2
    assert len(multi_dataset.roots) == 2


def test_multi_directory_path_or_sequence_equivalent(sample_experiment_1):
    a = Dataset.from_directory(sample_experiment_1)
    b = Dataset.from_directory([sample_experiment_1])
    assert len(a.inventory) == len(b.inventory)


def test_unknown_source_tag_rejected(sample_experiment_1):
    with pytest.raises(KeyError):
        Dataset.from_directory(sample_experiment_1, sources=["nope_not_real"])


def test_explicit_sources_filter(sample_experiment_1):
    ds = Dataset.from_directory(sample_experiment_1, sources=["dataqueue"])
    assert ds.sources == ("dataqueue",)


def test_repr_does_not_raise(dataset):
    assert "Dataset" in repr(dataset)


def test_inventory_returns_copy(dataset):
    inv = dataset.inventory
    inv.iloc[0, 0] = "MUTATED"
    assert dataset.inventory.iloc[0, 0] != "MUTATED"


def test_invalid_inventory_rejected():
    with pytest.raises(TypeError):
        Dataset("not a frame")  # type: ignore[arg-type]


def test_non_multiindex_inventory_rejected():
    with pytest.raises(ValueError):
        Dataset(pd.DataFrame({"dataqueue": ["x"]}))
