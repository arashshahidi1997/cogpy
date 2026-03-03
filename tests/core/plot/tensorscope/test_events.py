"""Tests for event system."""

from __future__ import annotations

import pandas as pd

from cogpy.core.plot.tensorscope import TensorScopeState
from cogpy.core.plot.tensorscope.events import EventStream
from cogpy.core.plot.tensorscope.layers.events import EventTableLayer


def test_event_stream_creation():
    df = pd.DataFrame(
        {
            "event_id": [0, 1, 2],
            "t": [1.0, 2.5, 4.0],
            "label": ["burst", "ripple", "burst"],
        }
    )
    stream = EventStream("test_events", df)
    assert stream.name == "test_events"
    assert len(stream) == 3


def test_event_stream_window_query():
    df = pd.DataFrame({"event_id": [0, 1, 2, 3], "t": [1.0, 2.5, 4.0, 5.5]})
    stream = EventStream("test", df)
    events = stream.get_events_in_window(2.0, 4.5)
    assert len(events) == 2
    assert list(events["event_id"]) == [1, 2]


def test_event_navigation():
    df = pd.DataFrame({"event_id": [0, 1, 2], "t": [1.0, 3.0, 5.0]})
    stream = EventStream("test", df)
    next_ev = stream.get_next_event(2.0)
    assert int(next_ev["event_id"]) == 1
    assert float(next_ev["t"]) == 3.0
    prev_ev = stream.get_prev_event(4.0)
    assert int(prev_ev["event_id"]) == 1
    assert float(prev_ev["t"]) == 3.0


def test_event_registry(small_ieeg):
    state = TensorScopeState(small_ieeg)
    df = pd.DataFrame({"event_id": [0, 1], "t": [1.0, 2.0]})
    stream = EventStream("bursts", df)
    state.register_events("bursts", stream)
    assert "bursts" in state.event_registry.list()
    retrieved = state.event_registry.get("bursts")
    assert retrieved is stream


def test_event_table_layer(small_ieeg):
    state = TensorScopeState(small_ieeg)
    df = pd.DataFrame(
        {"event_id": [0, 1, 2], "t": [1.0, 2.0, 3.0], "label": ["burst", "ripple", "burst"]}
    )
    stream = EventStream("bursts", df)
    state.register_events("bursts", stream)
    layer = EventTableLayer(state, "bursts")
    assert layer.stream is stream
    assert layer.panel() is not None


def test_state_jump_to_event(small_ieeg):
    state = TensorScopeState(small_ieeg)
    df = pd.DataFrame({"event_id": [10, 20, 30], "t": [1.5, 3.2, 5.8]})
    stream = EventStream("bursts", df)
    state.register_events("bursts", stream)
    state.jump_to_event("bursts", 20)
    assert state.current_time == 3.2

