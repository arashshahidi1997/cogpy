"""Tests for TensorScope v2.2 ViewFactory."""

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


def test_infer_view_type():
    from cogpy.core.plot.tensorscope.view_factory import infer_view_type

    assert infer_view_type(["AP", "ML"]) == "Image"
    assert infer_view_type(["time"]) == "Curve"
    assert infer_view_type(["time", "AP", "ML"]) == "Generic"


def test_viewfactory_spatial_dynamicmap(grid_data):
    hv = pytest.importorskip("holoviews")

    from cogpy.core.plot.tensorscope import TensorScopeState
    from cogpy.core.plot.tensorscope.view_factory import ViewFactory
    from cogpy.core.plot.tensorscope.view_spec import ViewSpec

    hv.extension("bokeh")

    state = TensorScopeState(grid_data)
    view = ViewFactory.create(ViewSpec(kdims=["AP", "ML"], controls=["time"]), state)
    assert isinstance(view, hv.DynamicMap)


def test_viewfactory_temporal_dynamicmap(grid_data):
    hv = pytest.importorskip("holoviews")

    from cogpy.core.plot.tensorscope import TensorScopeState
    from cogpy.core.plot.tensorscope.view_factory import ViewFactory
    from cogpy.core.plot.tensorscope.view_spec import ViewSpec

    hv.extension("bokeh")

    state = TensorScopeState(grid_data)
    view = ViewFactory.create(ViewSpec(kdims=["time"], controls=["AP", "ML"]), state)
    assert isinstance(view, hv.DynamicMap)


def test_viewfactory_with_operation(grid_data):
    hv = pytest.importorskip("holoviews")

    from cogpy.core.plot.tensorscope import TensorScopeState
    from cogpy.core.plot.tensorscope.view_factory import ViewFactory
    from cogpy.core.plot.tensorscope.view_spec import ViewSpec

    hv.extension("bokeh")

    state = TensorScopeState(grid_data)
    view = ViewFactory.create(
        ViewSpec(kdims=["AP", "ML"], controls=["time"], operation=np.abs, symmetric_clim=True),
        state,
    )
    assert isinstance(view, hv.DynamicMap)

