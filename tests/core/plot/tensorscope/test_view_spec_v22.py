"""Tests for TensorScope v2.2 ViewSpec."""

from __future__ import annotations

import pytest

from cogpy.core.plot.tensorscope.view_spec import ViewSpec


def test_viewspec_creation_defaults():
    spec = ViewSpec(kdims=["AP", "ML"], controls=["time"])
    assert spec.kdims == ["AP", "ML"]
    assert spec.controls == ["time"]
    assert spec.view_type == "auto"
    assert spec.iterate == []


def test_viewspec_validation_overlap():
    with pytest.raises(ValueError, match="Dimensions cannot appear in multiple categories"):
        ViewSpec(kdims=["time"], controls=["time"])


def test_viewspec_roundtrip_dict():
    spec = ViewSpec(
        kdims=["AP", "ML"],
        controls=["time"],
        colormap="RdBu_r",
        symmetric_clim=True,
        clim=(-1.0, 1.0),
        title="Spatial",
    )

    restored = ViewSpec.from_dict(spec.to_dict())
    assert restored.to_dict() == spec.to_dict()


def test_viewspec_with_iterate():
    spec = ViewSpec(kdims=["AP", "ML"], controls=["time"], iterate=["signal"])
    assert spec.iterate == ["signal"]
