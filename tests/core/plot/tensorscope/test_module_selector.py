"""Tests for TensorScope v2.3 Module Selector UI."""

from __future__ import annotations

import numpy as np
import pytest

from cogpy.core.plot.tensorscope import TensorScopeState
from cogpy.core.plot.tensorscope.modules import ModuleRegistry


@pytest.fixture
def grid_data():
    xr = pytest.importorskip("xarray")

    rng = np.random.RandomState(0)
    data = rng.randn(200, 8, 8)
    return xr.DataArray(
        data,
        dims=("time", "AP", "ML"),
        coords={"time": np.arange(200) / 1000.0, "AP": np.arange(8), "ML": np.arange(8)},
        attrs={"fs": 1000.0},
    )


def test_module_selector_creation(grid_data):
    pytest.importorskip("panel")

    from cogpy.core.plot.tensorscope.ui import ModuleSelectorLayer

    state = TensorScopeState(grid_data)
    registry = ModuleRegistry()
    selector = ModuleSelectorLayer(state, registry)

    assert selector.state is state
    assert selector.registry is registry


def test_module_selector_has_modules(grid_data):
    pytest.importorskip("panel")

    from cogpy.core.plot.tensorscope.ui import ModuleSelectorLayer

    state = TensorScopeState(grid_data)
    registry = ModuleRegistry()
    selector = ModuleSelectorLayer(state, registry)

    options = selector.module_selector.options
    assert len(options) >= 2

