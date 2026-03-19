"""
Test Phase 4: Events System

Demonstrates:
- Creating EventStream from DataFrame
- Registering events with state
- EventTableLayer with navigation
- Event queries (next/prev/window)
- Jump to event functionality

Run with:
    conda run -n cogpy python code/lib/cogpy/examples/tensorscope/test_phase4_events.py
"""

import numpy as np
import pandas as pd

from cogpy.plot.tensorscope import TensorScopeState
from cogpy.plot.tensorscope.events import EventStream, EventStyle
from cogpy.plot.tensorscope.layers.events import EventTableLayer
from cogpy.datasets.entities import example_ieeg_grid

print("=" * 60)
print("TensorScope Phase 4: Events System Test")
print("=" * 60)

print("\n1. Loading data...")
data = example_ieeg_grid(mode="small")
state = TensorScopeState(data)
print("   ✅ State created")

print("\n2. Creating synthetic events...")
np.random.seed(42)
n_events = 20
event_times = np.sort(np.random.uniform(0, 10, n_events))
event_df = pd.DataFrame(
    {
        "event_id": range(n_events),
        "t": event_times,
        "label": np.random.choice(["burst", "ripple", "spindle"], n_events),
        "channel": np.random.randint(0, 64, n_events),
        "amplitude": np.random.uniform(2.0, 5.0, n_events),
    }
)
print(f"   Created {n_events} events:")
print(f"   - Time range: {event_times[0]:.2f}s - {event_times[-1]:.2f}s")
print(f"   - Types: {event_df['label'].value_counts().to_dict()}")
print("   ✅ Synthetic events created")

print("\n3. Creating EventStream...")
stream = EventStream(
    "test_events",
    event_df,
    style=EventStyle(color="#ff5555", marker="circle"),
)
print(f"   - Stream name: {stream.name}")
print(f"   - Event count: {len(stream)}")
print(f"   - Style: {stream.style.color}")
print("   ✅ EventStream created")

print("\n4. Registering events with state...")
state.register_events("test_events", stream)
registered = state.event_registry.list()
print(f"   - Registered streams: {registered}")
assert "test_events" in registered
print("   ✅ Events registered")

print("\n5. Testing event queries...")
events_in_window = stream.get_events_in_window(2.0, 5.0)
print(f"   - Events in window [2.0, 5.0]: {len(events_in_window)}")

state.current_time = 3.0
next_event = stream.get_next_event(state.current_time)
prev_event = stream.get_prev_event(state.current_time)
print(f"   - Current time: {state.current_time:.2f}s")
print(f"   - Next event: t={next_event['t']:.2f}s (ID={next_event['event_id']})")
print(f"   - Prev event: t={prev_event['t']:.2f}s (ID={prev_event['event_id']})")
print("   ✅ Event queries working")

print("\n6. Testing state navigation methods...")
initial_time = state.current_time
state.next_event("test_events")
after_next = state.current_time
print(f"   - After next_event(): {initial_time:.2f}s → {after_next:.2f}s")

state.prev_event("test_events")
after_prev = state.current_time
print(f"   - After prev_event(): {after_next:.2f}s → {after_prev:.2f}s")
print("   ✅ Navigation methods working")

print("\n7. Testing jump to event...")
state.jump_to_event("test_events", event_id=10)
print(f"   - Jumped to event 10: t={state.current_time:.2f}s")
expected_time = float(event_df.loc[event_df["event_id"] == 10, "t"].values[0])
assert abs(state.current_time - expected_time) < 0.001
print("   ✅ Jump to event working")

print("\n8. Creating EventTableLayer...")
table_layer = EventTableLayer(state, "test_events")
print(f"   - Layer ID: {table_layer.layer_id}")
print(f"   - Title: {table_layer.title}")
print(f"   - Panel: {table_layer.panel()}")
print("   ✅ EventTableLayer created")

print("\n9. Testing event serialization...")
stream_dict = stream.to_dict()
print(f"   - Serialized keys: {list(stream_dict.keys())}")
print(f"   - Event count: {stream_dict['n_events']}")
print(f"   - Time range: {stream_dict['time_range']}")
print("   ✅ Serialization working")

print("\n" + "=" * 60)
print("✅ Phase 4 Events System: ALL TESTS PASSED!")
print("=" * 60)
print("\nNext: Try the interactive event browser:")
print("  conda run -n cogpy panel serve code/lib/cogpy/examples/tensorscope/phase4_events.py --show")

