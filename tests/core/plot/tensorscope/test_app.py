"""Tests for TensorScopeApp."""

from __future__ import annotations

import pandas as pd

from cogpy.core.plot.tensorscope import TensorScopeApp
from cogpy.core.plot.tensorscope.events import EventStream


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
    # Include an event layer to ensure session restore doesn't crash.
    df = pd.DataFrame({"event_id": [0, 1], "t": [1.0, 2.0], "label": ["burst", "ripple"]})
    app.state.register_events("bursts", EventStream("bursts", df))
    app.add_layer("event_table")
    app.state.current_time = 5.0

    session = app.to_session()
    assert session["version"] == "1.0"
    assert "state" in session
    assert "layout" in session
    assert len(session["layers"]) == 3

    app2 = TensorScopeApp.from_session(session, data_resolver=lambda: small_ieeg)
    assert app2.state.current_time == 5.0
    assert len(app2.layer_manager.list_instances()) == 3
    # Events restored sufficiently for default event_table layer.
    assert "bursts" in app2.state.event_registry.list()


def test_app_shutdown(small_ieeg):
    app = TensorScopeApp(small_ieeg)
    app.add_layer("timeseries")
    app.add_layer("spatial_map")
    app.shutdown()
    assert len(app.layer_manager.list_instances()) == 0
