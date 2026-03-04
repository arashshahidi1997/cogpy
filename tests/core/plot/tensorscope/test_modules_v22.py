"""Tests for TensorScope v2.2 module system."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def grid_data():
    xr = pytest.importorskip("xarray")

    rng = np.random.RandomState(0)
    data = rng.randn(100, 4, 5)
    return xr.DataArray(
        data,
        dims=("time", "AP", "ML"),
        coords={"time": np.arange(100) / 1000.0, "AP": np.arange(4), "ML": np.arange(5)},
        attrs={"fs": 1000.0},
    )


def test_module_registry_builtin():
    from cogpy.core.plot.tensorscope.modules import ModuleRegistry

    registry = ModuleRegistry()
    names = registry.list()
    assert "basic" in names
    assert "comparison" in names


def test_module_activate_returns_layout(grid_data):
    hv = pytest.importorskip("holoviews")

    from cogpy.core.plot.tensorscope import TensorScopeState
    from cogpy.core.plot.tensorscope.modules import ModuleRegistry

    hv.extension("bokeh")

    state = TensorScopeState(grid_data)
    mod = ModuleRegistry().get("basic")
    assert mod is not None
    layout = mod.activate(state)
    assert isinstance(layout, hv.Layout)

