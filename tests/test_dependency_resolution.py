"""Tests for source dependency declarations and topological loading order."""

from __future__ import annotations

import numpy as np
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


def test_treadmill_fallback_works_without_dataqueue(sample_experiment_2):
    ds = Dataset.from_directory(sample_experiment_2, sources=["treadmill"])
    df = ds.materialize(strict=True)

    treadmill_cols = df.xs("treadmill", level=0, axis=1)
    elapsed = np.asarray(treadmill_cols.iloc[0]["time_elapsed_s"], dtype=np.float64)

    assert elapsed.size > 0
    assert np.isfinite(elapsed).all()
    assert np.all(np.diff(elapsed) > 0)

    meta = treadmill_cols.iloc[0]["meta"]
    assert meta["source_method"] == "treadmill_csv_fallback"


def test_treadmill_fallback_with_missing_dataqueue_source(sample_experiment_2):
    ds = Dataset.from_directory(sample_experiment_2, sources=["dataqueue", "treadmill"])
    df = ds.materialize(strict=True)

    treadmill_cols = df.xs("treadmill", level=0, axis=1)
    elapsed = np.asarray(treadmill_cols.iloc[0]["time_elapsed_s"], dtype=np.float64)
    assert elapsed.size > 0
    assert np.all(np.diff(elapsed) > 0)


def test_pupil_dlc_fallback_without_dataqueue_or_metadata(monkeypatch):
    from pathlib import Path

    from datakit.sources.analysis.pupil import PupilDLCSource
    from datakit.sources.register import LoadContext

    source = PupilDLCSource()
    analyzed = pd.DataFrame({"pupil_diameter_mm": [0.9, 1.0, 1.1, 1.0]})
    monkeypatch.setattr(source, "_analyze_pupil_h5", lambda _path: analyzed.copy())

    context = LoadContext(
        subject="STREHAB02",
        session="ses-10",
        task="task-widefield",
        inventory_row={},
        dependencies={"dataqueue": None, "pupil_metadata": None},
    )

    t, frame, meta = source.build_timeseries(Path("dummy_pupil_dlc.h5"), context=context)

    expected = np.arange(len(analyzed), dtype=np.float64) / float(source.default_frame_rate_hz)
    assert np.allclose(t, expected)
    assert np.allclose(frame["time_elapsed_s"].to_numpy(dtype=np.float64), expected)
    assert meta["time_basis"] == "assumed_frame_rate"
    assert meta["assumed_frame_rate_hz"] == float(source.default_frame_rate_hz)
    assert meta["pupil_metadata_file"] is None
