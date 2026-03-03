"""
Tests for TensorScopeState (Phase 1 implementation).

Phase 0: Stub tests (skipped)
Phase 1: Implement these tests
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Phase 1 implementation pending")
def test_state_initialization(small_ieeg):
    """Test TensorScopeState initialization with valid data."""

    from cogpy.core.plot.tensorscope import TensorScopeState

    state = TensorScopeState(small_ieeg)

    assert hasattr(state, "time_hair")
    assert hasattr(state, "channel_grid")
    assert hasattr(state, "processing")


@pytest.mark.skip(reason="Phase 1 implementation pending")
def test_state_serialization(small_ieeg):
    """Test state can be serialized and restored."""

    from cogpy.core.plot.tensorscope import TensorScopeState

    state = TensorScopeState(small_ieeg)
    state.current_time = 5.3

    state_dict = state.to_dict()

    assert "current_time" in state_dict
    assert state_dict["current_time"] == 5.3

    state2 = TensorScopeState.from_dict(state_dict, data_resolver=lambda: small_ieeg)
    assert state2.current_time == 5.3

