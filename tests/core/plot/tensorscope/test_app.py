"""Tests for TensorScopeApp."""

from __future__ import annotations

from cogpy.core.plot.tensorscope import TensorScopeApp


def test_app_creation(small_ieeg):
    app = TensorScopeApp(small_ieeg)
    assert app.state is not None
    assert app.layer_manager is not None
    assert app.layout_manager is not None


def test_app_add_layers(small_ieeg):
    app = TensorScopeApp(small_ieeg)
    app.add_layer("timeseries")
    app.add_layer("spatial_map")
    app.add_layer("selector")

    assert len(app.layer_manager.list_instances()) == 3
    assert len(app._panels) == 3


def test_app_builder_pattern(small_ieeg):
    app = (
        TensorScopeApp(small_ieeg)
        .with_layout("spatial_focus")
        .add_layer("timeseries")
        .add_layer("spatial_map")
    )

    assert app.layout_manager.current_preset == "spatial_focus"
    assert len(app.layer_manager.list_instances()) == 2


def test_app_build(small_ieeg):
    app = TensorScopeApp(small_ieeg)
    app.add_layer("timeseries")
    app.add_layer("spatial_map")
    template = app.build()
    assert template is not None


def test_app_session_serialization(small_ieeg):
    app = TensorScopeApp(small_ieeg)
    app.add_layer("timeseries")
    app.add_layer("spatial_map")
    app.state.current_time = 5.0

    session = app.to_session()
    assert session["version"] == "1.0"
    assert "state" in session
    assert "layout" in session
    assert len(session["layers"]) == 2

    app2 = TensorScopeApp.from_session(session, data_resolver=lambda: small_ieeg)
    assert app2.state.current_time == 5.0
    assert len(app2.layer_manager.list_instances()) == 2


def test_app_shutdown(small_ieeg):
    app = TensorScopeApp(small_ieeg)
    app.add_layer("timeseries")
    app.add_layer("spatial_map")
    app.shutdown()
    assert len(app.layer_manager.list_instances()) == 0

