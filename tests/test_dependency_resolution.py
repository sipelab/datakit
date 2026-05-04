"""Tests for source dependency declarations and topological loading order."""

from __future__ import annotations

import pandas as pd
import pytest

from datakit import Dataset
from datakit.core import _toposort
from datakit.sources import SOURCE_REGISTRY


def test_treadmill_declares_dataqueue_dependency():
    cls_ = SOURCE_REGISTRY["treadmill"]
    assert "dataqueue" in (getattr(cls_, "requires", ()) or ())


def test_toposort_orders_dependency_before_dependent():
    order = _toposort(["treadmill", "dataqueue"])
    assert order.index("dataqueue") < order.index("treadmill")


def test_toposort_handles_only_independent_sources():
    order = _toposort(["dataqueue", "session_config", "notes"])
    assert set(order) == {"dataqueue", "session_config", "notes"}


def test_toposort_skips_deps_outside_active_set():
    """If a tag's dep isn't active, it's tolerated (will surface as None)."""
    order = _toposort(["treadmill"])
    assert order == ["treadmill"]


def test_toposort_detects_cycle(monkeypatch):
    class FakeA:
        requires = ("fakeB",)

    class FakeB:
        requires = ("fakeA",)

    monkeypatch.setitem(SOURCE_REGISTRY, "fakeA", FakeA)
    monkeypatch.setitem(SOURCE_REGISTRY, "fakeB", FakeB)
    with pytest.raises(RuntimeError, match="[Cc]ycle"):
        _toposort(["fakeA", "fakeB"])


def test_dataqueue_loaded_streams_passed_to_dependents(dataset):
    """Materialize end-to-end and verify treadmill saw a dataqueue dependency.

    If the dependency wiring were broken, treadmill would either fail or
    produce empty/no aligned columns. We assert the cell has treadmill data.
    """
    df = dataset.materialize(strict=True)
    assert "treadmill" in df.columns.get_level_values(0)
    treadmill_cols = df.xs("treadmill", level=0, axis=1)
    # at least one non-NaN cell
    assert treadmill_cols.notna().any().any()


def test_load_context_populates_legacy_fields(dataset):
    """Verify LoadContext gets dataqueue convenience fields when dq is loaded.

    We do this by injecting a probe source into the registry that captures
    its context; treadmill et al. depend on those fields via __post_init__
    in source code, so we just need to confirm the bridge is wired.
    """
    from datakit.core import _make_context
    from datakit.datamodel import LoadedStream
    import numpy as np

    fake_dq_frame = pd.DataFrame({
        "device_id": ["mesoscope", "mesoscope"],
        "queue_elapsed": [1.0, 2.0],
    })
    dq_stream = LoadedStream(
        tag="dataqueue",
        t=np.array([1.0, 2.0]),
        value=fake_dq_frame,
        meta={"foo": "bar"},
    )
    ctx = _make_context(
        idx=("S1", "ses-01", "task-x"),
        inventory_row={},
        deps={"dataqueue": dq_stream},
    )
    assert ctx.subject == "S1"
    assert ctx.session == "ses-01"
    assert ctx.task == "task-x"
    assert ctx.dataqueue_frame is fake_dq_frame
    assert ctx.dataqueue_meta == {"foo": "bar"}
    assert ctx.master_timeline is not None
    assert ctx.master_timeline.tolist() == [1.0, 2.0]
    assert ctx.experiment_window == (1.0, 2.0)


def test_missing_dependency_yields_none_in_context():
    from datakit.core import _make_context

    ctx = _make_context(
        idx=("S1", "ses-01", "task-x"),
        inventory_row={},
        deps={"dataqueue": None},
    )
    assert ctx.dependencies["dataqueue"] is None
    assert ctx.master_timeline is None
    assert ctx.dataqueue_frame is None
    assert ctx.experiment_window is None
