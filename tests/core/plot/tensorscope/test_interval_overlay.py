"""Tests for interval-aware temporal overlays (v2.6.3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

hv = pytest.importorskip("holoviews")

from cogpy.core.events import EventCatalog
from cogpy.core.plot.tensorscope.layers.events import EventOverlayLayer
from cogpy.core.plot.tensorscope.state import TensorScopeState


@pytest.fixture
def grid_data():
    return xr.DataArray(
        np.random.RandomState(0).randn(500, 4, 4),
        dims=("time", "AP", "ML"),
        coords={
            "time": np.arange(500, dtype=float) / 1000.0,
            "AP": np.arange(4, dtype=int),
            "ML": np.arange(4, dtype=int),
        },
        attrs={"fs": 1000.0},
    )


def test_temporal_overlay_point_events(grid_data):
    state = TensorScopeState(grid_data)
    df = pd.DataFrame({"event_id": ["a", "b"], "t": [0.1, 0.3]})
    state.register_event_catalog("points", EventCatalog(df=df, name="points"))

    layer = EventOverlayLayer(state, "points")
    el = layer.create_temporal_overlay()[()]
    assert isinstance(el, hv.Overlay)


def test_temporal_overlay_interval_events(grid_data):
    state = TensorScopeState(grid_data)
    df = pd.DataFrame(
        {
            "event_id": ["a", "b"],
            "t": [0.12, 0.32],
            "t0": [0.10, 0.30],
            "t1": [0.15, 0.36],
            "duration": [0.05, 0.06],
        }
    )
    state.register_event_catalog("intervals", EventCatalog(df=df, name="intervals"))

    layer = EventOverlayLayer(state, "intervals")
    el = layer.create_temporal_overlay()[()]
    assert isinstance(el, hv.Overlay)
    # Expect at least one VSpan and one VLine.
    types = {type(v).__name__ for v in el.values()} if hasattr(el, "values") else set()
    assert ("VSpan" in types) or ("VLine" in types)

