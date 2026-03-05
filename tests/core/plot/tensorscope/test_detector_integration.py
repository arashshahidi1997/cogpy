"""Tests for detector integration helpers on TensorScopeState (v2.6.2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from cogpy.core.events import EventCatalog
from cogpy.core.plot.tensorscope.state import TensorScopeState


class DummyDetector:
    def detect(self, data, **_kwargs):
        df = pd.DataFrame({"event_id": ["a", "b", "c"], "t": [0.1, 0.5, 0.9]})
        return EventCatalog(df=df, name="dummy")


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


def test_run_detector_registers_event_stream(grid_data):
    state = TensorScopeState(grid_data)
    detector = DummyDetector()

    catalog = state.run_detector(detector, event_type="dummy_events")
    assert isinstance(catalog, EventCatalog)
    assert len(catalog) == 3

    stream = state.event_registry.get("dummy_events")
    assert stream is not None
    assert len(stream) == 3


def test_register_event_catalog_style_dict(grid_data):
    state = TensorScopeState(grid_data)
    df = pd.DataFrame({"event_id": ["a"], "t": [0.2]})
    catalog = EventCatalog(df=df, name="x")

    state.register_event_catalog("styled", catalog, style={"color": "#00ff00", "alpha": 0.5})
    stream = state.event_registry.get("styled")
    assert stream is not None
    assert stream.style.color == "#00ff00"
    assert float(stream.style.alpha) == 0.5

