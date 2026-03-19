"""
Test Phase 6: Polish & Optimization - Complete System Test

Demonstrates:
- Full TensorScope functionality (Phases 1–5 integrated)
- Events system integration
- Session save/load round-trip
- Basic performance sanity checks

Run with:
    conda run -n cogpy python code/lib/cogpy/examples/tensorscope/test_phase6_complete.py
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from cogpy.datasets.entities import example_ieeg_grid
from cogpy.plot.tensorscope import TensorScopeApp
from cogpy.plot.tensorscope.events import EventStream


def _ms(s: float) -> float:
    return s * 1000.0


print("=" * 60)
print("TensorScope Phase 6: Complete System Test")
print("=" * 60)

# Load data
print("\n1. Loading data...")
start = time.perf_counter()
data = example_ieeg_grid(mode="large")
load_time = time.perf_counter() - start
print(f"   - Loaded: {data.dims}, shape={data.shape}")
print(f"   - Load time: {_ms(load_time):.1f}ms")
print("   ✅ Data loaded")

# Create app with core layers
print("\n2. Creating full TensorScope app...")
start = time.perf_counter()
app = (
    TensorScopeApp(data, title="TensorScope Phase 6")
    .with_layout("default")
    .add_layer("timeseries")
    .add_layer("spatial_map")
    .add_layer("selector")
    .add_layer("processing")
    .add_layer("navigator")
)
app_time = time.perf_counter() - start
print(f"   - Layers: {app.layer_manager.list_instances()}")
print(f"   - App creation time: {_ms(app_time):.1f}ms")
print("   ✅ App created")

# Add events (use stream name 'bursts' to match default event_table layer factory)
print("\n3. Adding event system...")
np.random.seed(0)
n_events = 50
event_times = np.sort(np.random.uniform(0, float(data.time.values[-1]), n_events))
events_df = pd.DataFrame(
    {
        "event_id": range(n_events),
        "t": event_times,
        "label": np.random.choice(["burst", "ripple"], n_events),
        "amplitude": np.random.uniform(2.0, 5.0, n_events),
    }
)
stream = EventStream("bursts", events_df)
app.state.register_events("bursts", stream)
app.add_layer("event_table")
print(f"   - Registered {len(stream)} events")
print("   ✅ Events integrated")

# Cursor performance (state update only; rendering may be async)
print("\n4. Testing cursor update performance...")
times = []
tmin = float(app.state.time_window.bounds[0]) if app.state.time_window is not None else 0.0
tmax = float(app.state.time_window.bounds[1]) if app.state.time_window is not None else 10.0
for t in np.linspace(tmin, tmax, 20):
    start = time.perf_counter()
    app.state.current_time = float(t)
    times.append(_ms(time.perf_counter() - start))

avg_cursor = float(np.mean(times))
max_cursor = float(np.max(times))
print(f"   - Cursor updates: avg={avg_cursor:.1f}ms, max={max_cursor:.1f}ms")
print("   ✅ Cursor update loop completed")

# Selection performance
print("\n5. Testing selection performance...")
start = time.perf_counter()
for ap, ml in [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]:
    app.state.channel_grid.select_cell(ap, ml)
select_time = time.perf_counter() - start
print(f"   - Selection (5 channels): {_ms(select_time):.1f}ms")
print("   ✅ Selection loop completed")

# Event navigation
print("\n6. Testing event navigation...")
start = time.perf_counter()
app.state.jump_to_event("bursts", 10)
nav_time = time.perf_counter() - start
print(f"   - Jump to event: {_ms(nav_time):.1f}ms")
print("   ✅ Event navigation completed")

# Session save/load
print("\n7. Testing session persistence...")
app.state.current_time = 5.0
app.state.channel_grid.select_cell(3, 3)

start = time.perf_counter()
session = app.to_session()
save_time = time.perf_counter() - start
print(f"   - Session save: {_ms(save_time):.1f}ms")

start = time.perf_counter()
app2 = TensorScopeApp.from_session(session, data_resolver=lambda: data)
restore_time = time.perf_counter() - start
print(f"   - Session load: {_ms(restore_time):.1f}ms")

assert app2.state.current_time == 5.0
assert (3, 3) in app2.state.selected_channels
print("   ✅ Session round-trip verified")

# Memory cleanup (optional)
print("\n8. Testing memory cleanup (optional)...")
try:
    import gc
    import psutil

    process = psutil.Process()
    gc.collect()
    mem_before = process.memory_info().rss / 1024 / 1024

    for _ in range(10):
        tmp = TensorScopeApp(data).add_layer("timeseries")
        tmp.shutdown()
        del tmp

    gc.collect()
    mem_after = process.memory_info().rss / 1024 / 1024
    mem_growth = float(mem_after - mem_before)
    print(f"   - Memory growth (10 cycles): {mem_growth:.1f}MB")
    print("   ✅ Memory cleanup check completed")
except Exception as e:  # noqa: BLE001
    print(f"   (skipped: {e})")

# Shutdown
print("\n9. Testing shutdown...")
layer_count = len(app.layer_manager.list_instances())
app.shutdown()
final_count = len(app.layer_manager.list_instances())
print(f"   - Layers before: {layer_count}")
print(f"   - Layers after: {final_count}")
assert final_count == 0
print("   ✅ Shutdown cleanup verified")

print("\n" + "=" * 60)
print("✅ TensorScope Phase 6: COMPLETE SYSTEM TEST PASSED")
print("=" * 60)
print("\nNext: Run the interactive demos in examples/tensorscope/")

