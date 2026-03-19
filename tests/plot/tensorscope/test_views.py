"""Tests for TensorScope v3 views."""

import holoviews as hv
import numpy as np
import pytest
import xarray as xr

from cogpy.tensorscope.state import SelectionState, TensorNode
from cogpy.tensorscope.views import (
    PSDAverageView,
    PSDSpatialView,
    SpatialMapView,
    TimeseriesView,
    get_available_views,
)


@pytest.fixture
def signal_tensor():
    """Create signal tensor (time, AP, ML)."""
    rng = np.random.default_rng(0)
    return xr.DataArray(
        rng.standard_normal((100, 8, 8)),
        dims=("time", "AP", "ML"),
        coords={
            "time": np.linspace(0, 10, 100),
            "AP": np.arange(8),
            "ML": np.arange(8),
        },
    )


@pytest.fixture
def psd_tensor():
    """Create PSD tensor (freq, AP, ML)."""
    rng = np.random.default_rng(0)
    return xr.DataArray(
        rng.standard_normal((129, 8, 8)),
        dims=("freq", "AP", "ML"),
        coords={
            "freq": np.linspace(0, 150, 129),
            "AP": np.arange(8),
            "ML": np.arange(8),
        },
    )


@pytest.fixture
def selection():
    """Create selection state."""
    sel = SelectionState()
    sel.time = 5.0
    sel.freq = 40.0
    sel.ap = 4
    sel.ml = 4
    return sel


def test_view_registry_signal(signal_tensor):
    """Test view discovery for signal tensor."""
    node = TensorNode(name="signal", data=signal_tensor)

    views = get_available_views(node)

    assert len(views) >= 2
    assert TimeseriesView in views
    assert SpatialMapView in views


def test_view_registry_psd(psd_tensor):
    """Test view discovery for PSD tensor."""
    node = TensorNode(name="psd", data=psd_tensor, transform="psd")

    views = get_available_views(node)

    assert len(views) >= 2
    assert PSDAverageView in views
    assert PSDSpatialView in views


def test_timeseries_view_render(signal_tensor, selection):
    """Test timeseries view rendering."""
    view = TimeseriesView()
    rendered = view.render(signal_tensor, selection)
    assert isinstance(rendered, hv.Overlay)


def test_spatial_map_view_render(signal_tensor, selection):
    """Test spatial map view rendering."""
    view = SpatialMapView()
    rendered = view.render(signal_tensor, selection)
    assert isinstance(rendered, hv.Image)


def test_psd_average_view_render(psd_tensor, selection):
    """Test PSD average view rendering."""
    view = PSDAverageView()
    rendered = view.render(psd_tensor, selection)
    assert isinstance(rendered, hv.Overlay)


def test_psd_spatial_view_render(psd_tensor, selection):
    """Test PSD spatial view rendering."""
    view = PSDSpatialView()
    rendered = view.render(psd_tensor, selection)
    assert isinstance(rendered, hv.Image)


def test_view_does_not_mutate_tensor(signal_tensor, selection):
    """Test that views don't mutate tensors."""
    original_data = signal_tensor.copy()

    view = SpatialMapView()
    view.render(signal_tensor, selection)

    assert np.allclose(signal_tensor.values, original_data.values)


def test_view_does_not_mutate_selection(signal_tensor, selection):
    """Test that views don't mutate selection."""
    original_time = selection.time
    original_ap = selection.ap

    view = SpatialMapView()
    view.render(signal_tensor, selection)

    assert selection.time == original_time
    assert selection.ap == original_ap

