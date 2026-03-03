"""
Tests for TensorLayer interface and concrete layers (Phase 2 implementation).

Phase 0: Stub tests (skipped)
Phase 2: Implement these tests
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Phase 2 implementation pending")
def test_layer_interface_contract(small_ieeg):
    """A TensorLayer should accept state and return a Panel viewable."""

    pytest.importorskip("panel")
    pytest.importorskip("param")

    from cogpy.core.plot.tensorscope import TensorScopeState

    state = TensorScopeState(small_ieeg)
    assert state is not None

