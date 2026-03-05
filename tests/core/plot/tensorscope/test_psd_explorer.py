"""Tests for PSD explorer module (v2.8.0)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

hv = pytest.importorskip("holoviews")

from cogpy.core.plot.tensorscope.modules import ModuleRegistry
from cogpy.core.plot.tensorscope.modules.psd_explorer import PSDExplorerModule
from cogpy.core.plot.tensorscope.state import TensorScopeState


@pytest.fixture
def grid_data():
    fs = 100.0
    t = np.arange(0.0, 10.0, 1.0 / fs, dtype=float)
    y = np.random.RandomState(0).randn(t.size, 8, 8)
    return xr.DataArray(
        y,
        dims=("time", "AP", "ML"),
        coords={"time": t, "AP": np.arange(8), "ML": np.arange(8)},
        attrs={"fs": fs},
    )


def test_registry_contains_psd_explorer():
    reg = ModuleRegistry()
    assert "psd_explorer" in reg.list()


def test_psd_explorer_activate(grid_data):
    state = TensorScopeState(grid_data)
    module = PSDExplorerModule()
    layout = module.activate(state)
    assert layout is not None
    import panel as pn

    assert isinstance(layout, pn.viewable.Viewable)
