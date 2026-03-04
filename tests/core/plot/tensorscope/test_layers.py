"""Tests for TensorScope layers."""

from __future__ import annotations

import panel as pn
import pytest

from cogpy.core.plot.tensorscope import TensorScopeState
from cogpy.core.plot.tensorscope.layers import (
    ChannelSelectorLayer,
    ProcessingControlsLayer,
    SpatialMapLayer,
    SpectrogramLayer,
    TimeseriesLayer,
    TimeNavigatorLayer,
)


def test_timeseries_layer_creation(small_ieeg):
    state = TensorScopeState(small_ieeg)
    layer = TimeseriesLayer(state)

    assert layer.layer_id == "timeseries"
    assert layer.state is state
    assert isinstance(layer.panel(), pn.viewable.Viewable)


def test_timeseries_layer_selection_update(small_ieeg):
    state = TensorScopeState(small_ieeg)
    layer = TimeseriesLayer(state)

    state.channel_grid.select_cell(1, 1)
    state.channel_grid.select_cell(2, 2)

    assert layer.panel() is not None


def test_spatial_layer_creation(small_ieeg):
    state = TensorScopeState(small_ieeg)
    layer = SpatialMapLayer(state, mode="rms", window_s=0.1)

    assert layer.layer_id == "spatial_map"
    assert layer.title == "Spatial RMS"
    assert isinstance(layer.panel(), pn.viewable.Viewable)


def test_spectrogram_layer_switching(small_ieeg):
    state = TensorScopeState(small_ieeg)

    # Register a synthetic spectrogram modality and verify the layer updates
    # without crashing when active modality changes.
    np = pytest.importorskip("numpy")
    xr = pytest.importorskip("xarray")

    spec = xr.DataArray(
        np.random.randn(50, 20, 8, 8),
        dims=("time", "freq", "AP", "ML"),
        coords={
            "time": np.linspace(0, 1, 50),
            "freq": np.linspace(1, 100, 20),
            "AP": np.arange(8),
            "ML": np.arange(8),
        },
    )
    from cogpy.core.plot.tensorscope.data.modalities import SpectrogramModality

    state.register_modality("spectrogram", SpectrogramModality(spec))

    layer = SpectrogramLayer(state)
    assert isinstance(layer.panel(), pn.viewable.Viewable)

    state.set_active_modality("spectrogram")
    assert state.active_modality == "spectrogram"


def test_channel_selector_layer(small_ieeg):
    state = TensorScopeState(small_ieeg)
    layer = ChannelSelectorLayer(state)
    assert isinstance(layer.panel(), pn.viewable.Viewable)


def test_processing_controls_layer(small_ieeg):
    state = TensorScopeState(small_ieeg)
    layer = ProcessingControlsLayer(state)
    assert isinstance(layer.panel(), pn.viewable.Viewable)


def test_time_navigator_layer(small_ieeg):
    state = TensorScopeState(small_ieeg)
    layer = TimeNavigatorLayer(state)
    assert isinstance(layer.panel(), pn.viewable.Viewable)


def test_layer_dispose(small_ieeg):
    state = TensorScopeState(small_ieeg)
    layer = TimeseriesLayer(state)

    assert len(layer._watchers) > 0
    layer.dispose()
    assert len(layer._watchers) == 0


def test_multiple_layers_same_state(small_ieeg):
    state = TensorScopeState(small_ieeg)

    layer1 = TimeseriesLayer(state)
    layer2 = SpatialMapLayer(state)
    layer3 = ChannelSelectorLayer(state)

    assert layer1.panel() is not None
    assert layer2.panel() is not None
    assert layer3.panel() is not None

    layer1.dispose()
    layer2.dispose()
    layer3.dispose()


def test_layer_lifecycle_100_cycles_lightweight(small_ieeg):
    state = TensorScopeState(small_ieeg)

    for _ in range(100):
        l1 = ChannelSelectorLayer(state)
        l2 = ProcessingControlsLayer(state)
        l3 = TimeNavigatorLayer(state)
        l1.dispose()
        l2.dispose()
        l3.dispose()
