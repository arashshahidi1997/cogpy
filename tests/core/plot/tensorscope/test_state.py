"""Tests for TensorScopeState."""

from __future__ import annotations

def test_state_initialization(small_ieeg):
    """Test TensorScopeState initialization with valid data."""

    from cogpy.core.plot.tensorscope import TensorScopeState

    state = TensorScopeState(small_ieeg)

    assert state.time_hair is not None
    assert state.time_window is not None
    assert state.channel_grid is not None
    assert state.processing is not None

    assert state.data_registry is not None
    assert state.event_registry is not None

    assert "grid_lfp" in state.data_registry.list()


def test_state_delegation_current_time(small_ieeg):
    """Test current_time delegates to TimeHair."""
    from cogpy.core.plot.tensorscope import TensorScopeState

    state = TensorScopeState(small_ieeg)

    state.current_time = 5.3
    assert state.time_hair.t == 5.3
    assert state.current_time == 5.3


def test_state_delegation_selection(small_ieeg):
    """Test selection delegates to ChannelGrid."""
    from cogpy.core.plot.tensorscope import TensorScopeState

    state = TensorScopeState(small_ieeg)

    state.channel_grid.select_cell(2, 3)
    assert (2, 3) in state.selected_channels


def test_state_serialization_roundtrip(small_ieeg):
    """Test state can be serialized and restored."""
    from cogpy.core.plot.tensorscope import TensorScopeState

    state = TensorScopeState(small_ieeg)
    state.current_time = 5.3
    state.channel_grid.select_cell(1, 1)
    state.channel_grid.select_cell(2, 2)

    state_dict = state.to_dict()

    assert state_dict["version"] == "1.0"
    assert state_dict["current_time"] == 5.3
    assert len(state_dict["selected_channels"]) == 2

    state2 = TensorScopeState.from_dict(state_dict, data_resolver=lambda: small_ieeg)

    assert state2.current_time == 5.3
    assert len(state2.selected_channels) == 2
    assert (1, 1) in state2.selected_channels
    assert (2, 2) in state2.selected_channels


def test_state_signal_registry(small_ieeg):
    """Test state initializes signal registry."""
    from cogpy.core.plot.tensorscope import TensorScopeState

    state = TensorScopeState(small_ieeg)

    assert state.signal_registry is not None
    assert len(state.signal_registry.list()) == 1

    active = state.signal_registry.get_active()
    assert active is not None
    assert active.name == "Raw LFP"
    assert active.metadata.get("is_base") is True


def test_state_create_derived_signal(small_ieeg):
    """Test state.create_derived_signal()."""
    from cogpy.core.plot.tensorscope import TensorScopeState

    state = TensorScopeState(small_ieeg)

    base_id = state.signal_registry.list()[0]
    hg_id = state.create_derived_signal(
        base_id,
        "High Gamma",
        {"bandpass_on": True, "bandpass_lo": 70.0, "bandpass_hi": 150.0},
    )

    hg_signal = state.signal_registry.get(hg_id)
    assert hg_signal is not None
    assert hg_signal.name == "High Gamma"
    assert hg_signal.processing.bandpass_on is True
    assert float(hg_signal.processing.bandpass_lo) == 70.0
    assert float(hg_signal.processing.bandpass_hi) == 150.0


def test_state_selected_time(small_ieeg):
    """Test selected_time feature."""
    from cogpy.core.plot.tensorscope import TensorScopeState

    state = TensorScopeState(small_ieeg)

    state.time_hair.t = 5.0
    state.set_selected_time_from_cursor()
    assert state.selected_time == 5.0

    state.time_hair.t = 7.0
    assert state.selected_time == 5.0


def test_state_legacy_processing_accessor(small_ieeg):
    """Test legacy state.processing accessor."""
    from cogpy.core.plot.tensorscope import TensorScopeState

    state = TensorScopeState(small_ieeg)

    proc = state.processing
    assert proc is not None
    assert proc is state.signal_registry.get_active().processing


def test_state_rejects_invalid_data():
    """Test state validates data schema on init."""
    import numpy as np
    import pytest
    import xarray as xr

    from cogpy.core.plot.tensorscope import TensorScopeState
    from cogpy.core.plot.tensorscope.schema import SchemaError

    bad_data = xr.DataArray(
        np.random.randn(100, 8),
        dims=("time", "channel"),
    )

    with pytest.raises(SchemaError, match="must have dimensions"):
        TensorScopeState(bad_data)


def test_state_normalizes_dimension_order():
    """Test state corrects dimension order."""
    import numpy as np
    import xarray as xr

    from cogpy.core.plot.tensorscope import TensorScopeState

    data = xr.DataArray(
        np.random.randn(100, 8, 8),
        dims=("time", "ML", "AP"),
        coords={
            "time": np.arange(100),
            "ML": np.arange(8),
            "AP": np.arange(8),
        },
    )

    state = TensorScopeState(data)

    modality = state.data_registry.get("grid_lfp")
    assert modality.data.dims == ("time", "AP", "ML")
