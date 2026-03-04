"""Tests for TensorScope v2.3 View Builder UI."""

from __future__ import annotations

import numpy as np
import pytest

from cogpy.core.plot.tensorscope import TensorScopeState


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


def test_view_builder_creation(grid_data):
    pytest.importorskip("panel")

    from cogpy.core.plot.tensorscope.ui import ViewBuilderLayer

    state = TensorScopeState(grid_data)
    builder = ViewBuilderLayer(state)

    assert builder.state is state
    assert builder.available_dims == ["time", "AP", "ML"]


def test_view_builder_build_spec(grid_data):
    pytest.importorskip("panel")

    from cogpy.core.plot.tensorscope.ui import ViewBuilderLayer

    state = TensorScopeState(grid_data)
    builder = ViewBuilderLayer(state)

    builder.kdims_selector.value = ["ML", "AP"]
    builder.controls_selector.value = ["time"]
    builder.view_type.value = "Image"
    builder.colormap.value = "RdBu_r"

    spec = builder.build_spec()
    assert spec.kdims == ["ML", "AP"]
    assert spec.controls == ["time"]
    assert spec.view_type == "Image"
    assert spec.colormap == "RdBu_r"


def test_view_builder_panel(grid_data):
    pytest.importorskip("panel")

    from cogpy.core.plot.tensorscope.ui import ViewBuilderLayer

    state = TensorScopeState(grid_data)
    builder = ViewBuilderLayer(state)

    assert builder.panel() is not None

