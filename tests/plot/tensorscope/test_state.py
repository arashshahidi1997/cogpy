"""Tests for TensorScope v3 state."""

import numpy as np
import pytest
import xarray as xr

from cogpy.tensorscope.state import (
    SelectionState,
    TensorNode,
    TensorRegistry,
    TensorScopeState,
)


@pytest.fixture
def sample_signal():
    """Create sample signal tensor (time, AP, ML)."""
    rng = np.random.default_rng(0)
    return xr.DataArray(
        rng.standard_normal((100, 8, 8)),
        dims=("time", "AP", "ML"),
        coords={
            "time": np.arange(100) / 10.0,
            "AP": np.arange(8),
            "ML": np.arange(8),
        },
    )


def test_tensor_node_creation(sample_signal):
    """Test TensorNode creation."""
    node = TensorNode(
        name="signal",
        data=sample_signal,
        source=None,
        transform="signal",
        params={},
    )

    assert node.name == "signal"
    assert node.dims == ("time", "AP", "ML")
    assert node.shape == (100, 8, 8)
    assert node.source is None


def test_tensor_node_lineage(sample_signal):
    """Test lineage strings."""
    signal_node = TensorNode(name="signal", data=sample_signal)
    assert signal_node.lineage_str() == "signal (original)"

    psd_node = TensorNode(
        name="psd",
        data=sample_signal,
        source="signal",
        transform="psd",
        params={"nperseg": 256},
    )
    assert "psd" in psd_node.lineage_str()
    assert "signal" in psd_node.lineage_str()


def test_tensor_registry(sample_signal):
    """Test tensor registry operations."""
    registry = TensorRegistry()

    node = TensorNode(name="signal", data=sample_signal)
    registry.add(node)

    assert "signal" in registry
    assert len(registry) == 1
    assert registry.list() == ["signal"]

    retrieved = registry.get("signal")
    assert retrieved.name == "signal"


def test_tensor_registry_duplicate_error(sample_signal):
    """Test error on duplicate names."""
    registry = TensorRegistry()

    node = TensorNode(name="signal", data=sample_signal)
    registry.add(node)

    with pytest.raises(ValueError, match="already registered"):
        registry.add(node)


def test_tensor_registry_missing_error():
    """Test error on missing tensor."""
    registry = TensorRegistry()

    with pytest.raises(KeyError, match="not found"):
        registry.get("nonexistent")


def test_selection_state():
    """Test SelectionState."""
    selection = SelectionState()

    assert selection.time == 0.0
    assert selection.freq == 0.0
    assert selection.ap == 0
    assert selection.ml == 0

    selection.time = 5.0
    assert selection.time == 5.0

    selection.update(freq=40.0, ap=8, ml=6)
    assert selection.freq == 40.0
    assert selection.ap == 8
    assert selection.ml == 6


def test_tensorscope_state(sample_signal):
    """Test TensorScopeState."""
    state = TensorScopeState()

    node = TensorNode(name="signal", data=sample_signal)
    state.tensors.add(node)

    state.set_active_tensor("signal")
    assert state.active_tensor == "signal"

    active = state.get_active_node()
    assert active.name == "signal"

    state.update_selection(time=7.5, ap=8)
    assert state.selection.time == 7.5
    assert state.selection.ap == 8


def test_tensorscope_state_invalid_active():
    """Test error setting invalid active tensor."""
    state = TensorScopeState()

    with pytest.raises(ValueError, match="not in registry"):
        state.set_active_tensor("nonexistent")


def test_selection_state_reactivity():
    """Test SelectionState param reactivity."""
    selection = SelectionState()

    changes = []

    def on_change(event):
        changes.append(event.new)

    selection.param.watch(on_change, "time")
    selection.time = 5.0

    assert len(changes) == 1
    assert changes[0] == 5.0

