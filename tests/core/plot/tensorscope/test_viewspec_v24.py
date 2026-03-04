"""Tests for ViewSpec v2.4 enhancements."""

from __future__ import annotations

import pytest

from cogpy.core.plot.tensorscope.view_spec import ViewSpec


def test_viewspec_fixed_values():
    spec = ViewSpec(kdims=["AP", "ML"], controls=[], fixed_values={"time": 5.0})
    assert spec.fixed_values == {"time": 5.0}
    assert "time" not in spec.kdims
    assert "time" not in spec.controls


def test_viewspec_fixed_values_validation():
    with pytest.raises(ValueError, match="Fixed dimensions cannot also be in kdims/controls/iterate"):
        ViewSpec(kdims=["time"], controls=[], fixed_values={"time": 5.0})


def test_viewspec_coord_spaces():
    spec = ViewSpec(kdims=["AP", "ML"], controls=["time"], coord_spaces=["spatial", "temporal"])
    assert "spatial" in spec.coord_spaces
    assert "temporal" in spec.coord_spaces


def test_viewspec_v24_serialization():
    spec = ViewSpec(
        kdims=["AP", "ML"],
        controls=[],
        fixed_values={"time": 5.0},
        coord_spaces=["spatial"],
    )
    restored = ViewSpec.from_dict(spec.to_dict())
    assert restored.fixed_values == spec.fixed_values
    assert restored.coord_spaces == spec.coord_spaces

