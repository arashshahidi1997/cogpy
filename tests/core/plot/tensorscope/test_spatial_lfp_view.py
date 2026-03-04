"""Tests for SpatialLFPView."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("panel")
pytest.importorskip("holoviews")

from cogpy.core.plot.tensorscope import TensorScopeState
from cogpy.core.plot.tensorscope.views.spatial_lfp import SpatialLFPView


@pytest.fixture
def grid_signal():
    data = xr.DataArray(
        np.random.randn(1000, 8, 8),
        dims=("time", "AP", "ML"),
        coords={"time": np.arange(1000) / 1000.0, "AP": np.arange(8), "ML": np.arange(8)},
        attrs={"fs": 1000.0},
    )
    return data


def test_spatial_lfp_view_creation(grid_signal):
    state = TensorScopeState(grid_signal)
    signal_id = state.signal_registry.list()[0]

    view = SpatialLFPView(state, signal_id, selected_time_source="cursor")

    assert view.selected_time_source == "cursor"
    assert view.colormap == "RdBu_r"
    assert view.symmetric_limits is True


def test_spatial_lfp_cursor_mode(grid_signal):
    state = TensorScopeState(grid_signal)
    signal_id = state.signal_registry.list()[0]

    view = SpatialLFPView(state, signal_id, selected_time_source="cursor")
    state.time_hair.t = 0.5

    assert view.panel() is not None


def test_spatial_lfp_selected_mode(grid_signal):
    state = TensorScopeState(grid_signal)
    signal_id = state.signal_registry.list()[0]

    view = SpatialLFPView(state, signal_id, selected_time_source="selected")
    state.selected_time = 0.5

    assert view.panel() is not None


def test_spatial_lfp_independent_mode(grid_signal):
    state = TensorScopeState(grid_signal)
    signal_id = state.signal_registry.list()[0]

    view = SpatialLFPView(
        state,
        signal_id,
        selected_time_source="independent",
        independent_time=0.5,
    )

    assert view.time_slider is not None
    assert float(view.time_slider.value) == 0.5


def test_spatial_selection_updates_marker(grid_signal):
    state = TensorScopeState(grid_signal)
    signal_id = state.signal_registry.list()[0]

    view = SpatialLFPView(state, signal_id, selected_time_source="cursor")
    state.time_hair.t = 0.5

    state.spatial_space.set_selection("AP", 3)
    state.spatial_space.set_selection("ML", 5)

    assert view.panel() is not None


def test_view_duplication(grid_signal):
    state = TensorScopeState(grid_signal)
    signal_id = state.signal_registry.list()[0]

    view1 = SpatialLFPView(
        state,
        signal_id,
        selected_time_source="independent",
        independent_time=0.5,
    )

    view2 = view1.duplicate()

    assert view2.selected_time_source == "independent"
    assert view2.signal_id == signal_id
    assert view2.view_id != view1.view_id


def test_coordinate_space_initialization(grid_signal):
    state = TensorScopeState(grid_signal)

    assert state.spatial_space is not None
    assert "AP" in state.spatial_space.dims
    assert "ML" in state.spatial_space.dims
    assert state.spatial_space.get_selection("AP") == 4
    assert state.spatial_space.get_selection("ML") == 4

