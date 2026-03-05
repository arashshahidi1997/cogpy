"""Tests for TensorScope EventOverlayLayer (v2.6.2)."""

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
        np.random.RandomState(0).randn(1000, 8, 8),
        dims=("time", "AP", "ML"),
        coords={
            "time": np.arange(1000, dtype=float) / 1000.0,
            "AP": np.arange(8, dtype=int),
            "ML": np.arange(8, dtype=int),
        },
        attrs={"fs": 1000.0},
    )


@pytest.fixture
def event_catalog():
    df = pd.DataFrame(
        {
            "event_id": [f"e{i}" for i in range(20)],
            "t": np.linspace(0.05, 0.95, 20),
            "AP": np.random.RandomState(1).randint(0, 8, size=20),
            "ML": np.random.RandomState(2).randint(0, 8, size=20),
            "freq": np.random.RandomState(3).uniform(20.0, 80.0, size=20),
            "value": np.random.RandomState(4).uniform(0.0, 1.0, size=20),
        }
    )
    return EventCatalog(df=df, name="test_events")


def test_event_overlay_layer_creates_dynamicmaps(grid_data, event_catalog):
    state = TensorScopeState(grid_data)
    state.register_event_catalog("bursts", event_catalog)

    layer = EventOverlayLayer(state, "bursts")
    spatial = layer.create_spatial_overlay()
    temporal = layer.create_temporal_overlay()
    spectro = layer.create_spectrogram_overlay()

    assert isinstance(spatial, hv.DynamicMap)
    assert isinstance(temporal, hv.DynamicMap)
    assert isinstance(spectro, hv.DynamicMap)


def test_event_overlay_layer_renders_elements(grid_data, event_catalog):
    state = TensorScopeState(grid_data)
    state.register_event_catalog("bursts", event_catalog)

    layer = EventOverlayLayer(state, "bursts")
    el_spatial = layer.create_spatial_overlay()[()]
    el_temporal = layer.create_temporal_overlay()[()]
    el_spectro = layer.create_spectrogram_overlay()[()]

    assert isinstance(el_spatial, hv.Points)
    assert isinstance(el_temporal, hv.Overlay)
    assert isinstance(el_spectro, hv.Points)

