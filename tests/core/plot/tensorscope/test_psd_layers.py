"""Tests for PSD Explorer layers."""

from __future__ import annotations


def test_psd_layers_can_build_template(small_ieeg):
    import pytest

    pytest.importorskip("panel")
    pytest.importorskip("holoviews")

    from cogpy.core.plot.tensorscope import TensorScopeApp

    app = (
        TensorScopeApp(small_ieeg, title="TensorScope PSD")
        .with_layout("psd_explorer")
        .add_layer("timeseries")
        .add_layer("spatial_map")
        .add_layer("selector")
        .add_layer("processing")
        .add_layer("psd_settings")
        .add_layer("navigator")
        .add_layer("psd_explorer")
    )

    template = app.build()
    assert template is not None

