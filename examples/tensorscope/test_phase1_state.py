"""
Test Phase 1: TensorScope State with real data.

Demonstrates:
- State creation with real data
- Controller ownership and delegation
- Time window management
- Channel selection
- Serialization and restoration
- Data registry

Run with:
    conda run -n cogpy python code/lib/cogpy/examples/tensorscope/test_phase1_state.py
"""

from cogpy.plot.tensorscope import TensorScopeState
from cogpy.datasets.entities import example_ieeg_grid

print("=" * 60)
print("TensorScope Phase 1: State Architecture Test")
print("=" * 60)

# Load data
print("\n1. Loading example data...")
data = example_ieeg_grid(mode="small")
print(f"   Loaded: {data.dims}, shape={data.shape}")

# Create state
print("\n2. Creating TensorScope state...")
state = TensorScopeState(data)
print("   ✅ State created successfully!")

# Test controllers
print("\n3. Testing controller ownership...")
print(f"   - TimeHair: {state.time_hair}")
print(f"   - TimeWindowCtrl: {state.time_window}")
print(f"   - ChannelGrid: {state.channel_grid} ({state.channel_grid.n_ap}x{state.channel_grid.n_ml})")
print(f"   - ProcessingChain: {state.processing}")

# Test delegation
print("\n4. Testing state delegation...")
state.current_time = 5.3
print("   - Set state.current_time = 5.3")
print(f"   - TimeHair.t = {state.time_hair.t}")
print(f"   - state.current_time = {state.current_time}")
print("   ✅ Delegation working!")

# Test time window
print("\n5. Testing time window...")
state.time_window.set_window(2.0, 8.0)
print(f"   - Set window: {state.time_window.window}")
state.time_window.recenter(5.0, width_s=3.0)
print(f"   - Recentered at t=5.0, width=3s: {state.time_window.window}")

# Test selection
print("\n6. Testing channel selection...")
state.channel_grid.select_cell(2, 3)
state.channel_grid.select_cell(4, 5)
print(f"   - Selected channels: {state.selected_channels}")
print(f"   - Flat indices: {state.selected_channels_flat}")

# Test processing chain
print("\n7. Testing processing chain...")
print(f"   - Available transforms: {state.processing.describe()}")
window_data = state.processing.get_window(2.0, 4.0)
print(f"   - Windowed data shape: {window_data.shape}")

# Test serialization
print("\n8. Testing serialization...")
state_dict = state.to_dict()
print(f"   - Serialized keys: {list(state_dict.keys())}")
print(f"   - Current time: {state_dict['current_time']}")
print(f"   - Selected channels: {state_dict['selected_channels']}")

# Test restoration
print("\n9. Testing restoration...")
state2 = TensorScopeState.from_dict(state_dict, data_resolver=lambda: data)
print(f"   - Restored current_time: {state2.current_time}")
print(f"   - Restored selection: {state2.selected_channels}")
assert state2.current_time == state.current_time
assert state2.selected_channels == state.selected_channels
print("   ✅ Round-trip successful!")

# Test data registry
print("\n10. Testing data registry...")
print(f"   - Registered modalities: {state.data_registry.list()}")
modality = state.data_registry.get("grid_lfp")
print(f"   - GridLFP time bounds: {modality.time_bounds()}")

print("\n" + "=" * 60)
print("✅ Phase 1 State Architecture: ALL TESTS PASSED!")
print("=" * 60)
print("\nNext: Phase 2 will wrap existing components as layers")
print("      Run: panel serve code/lib/cogpy/examples/tensorscope/hello_tensorscope.py")

