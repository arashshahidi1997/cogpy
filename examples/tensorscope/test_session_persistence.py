"""
Test TensorScope session save/load functionality.

Demonstrates:
- Creating a session with state
- Serializing to JSON file
- Loading from JSON file
- Verifying restored state matches original

Run with:
    conda run -n cogpy python code/lib/cogpy/examples/tensorscope/test_session_persistence.py
"""

import json
from pathlib import Path

from cogpy.core.plot.tensorscope import TensorScopeState
from cogpy.datasets.entities import example_ieeg_grid

print("=" * 60)
print("TensorScope Session Persistence Test")
print("=" * 60)

# Load data
print("\n1. Loading data...")
data = example_ieeg_grid(mode="small")
print(f"   ✅ Loaded: {data.dims}, shape={data.shape}")

# Create state and modify it
print("\n2. Creating session with custom state...")
state = TensorScopeState(data)
state.current_time = 7.5
state.time_window.set_window(5.0, 10.0)
state.channel_grid.select_cell(1, 2)
state.channel_grid.select_cell(3, 4)
state.channel_grid.select_cell(5, 6)

print("   Session state:")
print(f"   - Time: {state.current_time}s")
print(f"   - Window: {state.time_window.window}")
print(f"   - Selected channels: {state.selected_channels}")
print(f"   - Num selected: {len(state.selected_channels)}")

# Save to file
session_file = Path("/tmp/tensorscope_session.json")
session_data = {
    "data_path": 'example_ieeg_grid(mode="small")',  # In production: actual file path
    "state": state.to_dict(),
}

print(f"\n3. Saving to {session_file}...")
with open(session_file, "w") as f:
    json.dump(session_data, f, indent=2)

file_size = session_file.stat().st_size
print(f"   ✅ Saved! File size: {file_size:,} bytes")

# Load from file
print(f"\n4. Loading from {session_file}...")
with open(session_file, "r") as f:
    loaded = json.load(f)

print(f"   ✅ Loaded! Session version: {loaded['state']['version']}")

# Restore state
print("\n5. Restoring state...")
restored_state = TensorScopeState.from_dict(
    loaded["state"],
    data_resolver=lambda: data,  # In production: load from data_path
)

print("   Restored state:")
print(f"   - Time: {restored_state.current_time}s")
print(f"   - Window: {restored_state.time_window.window}")
print(f"   - Selected channels: {restored_state.selected_channels}")
print(f"   - Num selected: {len(restored_state.selected_channels)}")

# Verify match
print("\n6. Verifying restored state matches original...")
assert restored_state.current_time == state.current_time, "Time mismatch!"
assert restored_state.time_window.window == state.time_window.window, "Window mismatch!"
assert restored_state.selected_channels == state.selected_channels, "Selection mismatch!"
print("   ✅ All fields match!")

# Show session file contents
print("\n7. Session file contents:")
print("-" * 60)
print(json.dumps(loaded, indent=2)[:500] + "\n   ... (truncated)")
print("-" * 60)

print("\n" + "=" * 60)
print("✅ Session Persistence: WORKING PERFECTLY!")
print("=" * 60)
print(f"\nYou can manually inspect: {session_file}")
print("Try editing the JSON and reloading to test robustness!")

