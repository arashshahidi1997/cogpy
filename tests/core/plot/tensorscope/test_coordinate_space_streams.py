"""Tests for CoordinateSpace stream integration (v2.4)."""

from __future__ import annotations

import pytest


def test_coordinate_space_create_stream():
    hv = pytest.importorskip("holoviews")
    hv.extension("bokeh")

    from cogpy.core.plot.tensorscope.transforms.base import CoordinateSpace

    space = CoordinateSpace("spatial", dims={"AP", "ML"})
    stream = space.create_stream()

    assert stream is not None
    assert hasattr(stream, "AP")
    assert hasattr(stream, "ML")


def test_coordinate_space_stream_bidirectional_sync():
    hv = pytest.importorskip("holoviews")
    hv.extension("bokeh")

    from cogpy.core.plot.tensorscope.transforms.base import CoordinateSpace

    space = CoordinateSpace("spatial", dims={"AP", "ML"})
    stream = space.create_stream()

    space.set_selection("AP", 3)
    assert stream.AP == 3

    stream.event(ML=5)
    assert space.get_selection("ML") == 5


def test_multiple_coordinate_spaces():
    hv = pytest.importorskip("holoviews")
    hv.extension("bokeh")

    from cogpy.core.plot.tensorscope.transforms.base import CoordinateSpace

    spatial = CoordinateSpace("spatial", dims={"AP", "ML"})
    temporal = CoordinateSpace("temporal", dims={"time"})
    spectral = CoordinateSpace("spectral", dims={"freq"})

    spatial_stream = spatial.create_stream()
    temporal_stream = temporal.create_stream()
    spectral_stream = spectral.create_stream()

    spatial.set_selection("AP", 3)
    temporal.set_selection("time", 5.0)
    spectral.set_selection("freq", 40.0)

    assert spatial_stream.AP == 3
    assert temporal_stream.time == 5.0
    assert spectral_stream.freq == 40.0

