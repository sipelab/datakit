"""Tests for `Dataset.include` and `Dataset.exclude` filtering semantics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def test_include_subject_keeps_only_matching_rows(multi_dataset):
    out = multi_dataset.include(subject="STREHAB07")
    levels = out.inventory.index.get_level_values(0).unique().tolist()
    assert levels == ["STREHAB07"]


def test_include_unknown_subject_returns_empty(multi_dataset):
    out = multi_dataset.include(subject="NOPE")
    assert len(out.inventory) == 0


def test_include_source_restricts_active_sources(dataset):
    out = dataset.include(source=["dataqueue", "treadmill"])
    assert set(out.sources) == {"dataqueue", "treadmill"}


def test_include_source_string_or_list_equivalent(dataset):
    a = dataset.include(source="dataqueue")
    b = dataset.include(source=["dataqueue"])
    assert a.sources == b.sources == ("dataqueue",)


def test_include_combines_kwargs_with_AND(multi_dataset):
    out = multi_dataset.include(subject="STREHAB07", session="ses-05")
    assert len(out.inventory) == len(multi_dataset.inventory)
    out = multi_dataset.include(subject="STREHAB07", session="DOES_NOT_EXIST")
    assert len(out.inventory) == 0


def test_include_returns_new_dataset(dataset):
    out = dataset.include(source="dataqueue")
    assert out is not dataset
    assert dataset.sources != out.sources


def test_exclude_source_drops_globally(dataset):
    n_before = len(dataset.sources)
    out = dataset.exclude(source="treadmill")
    assert "treadmill" not in out.sources
    assert len(out.sources) == n_before - 1


def test_exclude_subject_drops_rows(multi_dataset):
    out = multi_dataset.exclude(subject="STREHAB07")
    assert len(out.inventory) == 0


def test_exclude_combined_row_and_source_blanks_cells(multi_dataset):
    """exclude(subject=X, source=Y) keeps rows but NaNs Y for those rows."""
    out = multi_dataset.exclude(subject="STREHAB07", source="treadmill")
    # Rows preserved (since source kwarg given)
    assert len(out.inventory) == len(multi_dataset.inventory)
    # treadmill column still present
    assert "treadmill" in out.inventory.columns
    # but all values for STREHAB07 are NaN
    sub = out.inventory.xs("STREHAB07", level=0)["treadmill"]
    assert sub.isna().all()


def test_chain_include_then_exclude(dataset):
    out = dataset.include(source=["dataqueue", "treadmill"]).exclude(source="treadmill")
    assert out.sources == ("dataqueue",)


def test_include_preserves_roots(dataset):
    out = dataset.include(source="dataqueue")
    assert out.roots == dataset.roots
