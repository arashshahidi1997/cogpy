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
