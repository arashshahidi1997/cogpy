"""Tests for the event_explorer preset module (v2.6.2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

hv = pytest.importorskip("holoviews")

from cogpy.core.events import EventCatalog
from cogpy.core.plot.tensorscope.modules import ModuleRegistry
from cogpy.core.plot.tensorscope.modules.event_explorer import MODULE
from cogpy.core.plot.tensorscope.state import TensorScopeState


@pytest.fixture
def grid_data():
    return xr.DataArray(
        np.random.RandomState(0).randn(200, 4, 4),
        dims=("time", "AP", "ML"),
        coords={
            "time": np.arange(200, dtype=float) / 1000.0,
            "AP": np.arange(4, dtype=int),
            "ML": np.arange(4, dtype=int),
        },
        attrs={"fs": 1000.0},
    )


def test_module_registry_contains_event_explorer():
    reg = ModuleRegistry()
    assert "event_explorer" in reg.list()


def test_event_explorer_activates_with_events(grid_data):
    state = TensorScopeState(grid_data)
    df = pd.DataFrame(
        {
            "event_id": [f"e{i}" for i in range(5)],
            "t": np.linspace(0.02, 0.18, 5),
            "AP": [0, 1, 2, 1, 0],
            "ML": [0, 1, 2, 1, 0],
            "freq": [30, 40, 50, 45, 35],
        }
    )
    state.register_event_catalog("bursts", EventCatalog(df=df, name="bursts"))

    layout = MODULE.activate(state)
    assert layout is not None
    # HoloViews layout-like
    assert hasattr(layout, "opts")

