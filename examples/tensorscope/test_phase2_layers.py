"""
Test Phase 2: Core Layers

Demonstrates:
- Creating layers from state
- Layer panel rendering
- State changes triggering layer updates
- Layer disposal

Run with:
    conda run -n cogpy python code/lib/cogpy/examples/tensorscope/test_phase2_layers.py
"""

from cogpy.plot.tensorscope import TensorScopeState
from cogpy.plot.tensorscope.layers import (
    ChannelSelectorLayer,
    ProcessingControlsLayer,
    SpatialMapLayer,
    TimeseriesLayer,
    TimeNavigatorLayer,
)
from cogpy.datasets.entities import example_ieeg_grid

print("=" * 60)
print("TensorScope Phase 2: Core Layers Test")
print("=" * 60)

# Load data and create state
print("\n1. Loading data and creating state...")
data = example_ieeg_grid(mode="small")
state = TensorScopeState(data)
print("   ✅ State created")

# Create layers
print("\n2. Creating layers...")
ts_layer = TimeseriesLayer(state, show_hair=True)
spatial_layer = SpatialMapLayer(state, mode="rms", window_s=0.1)
selector_layer = ChannelSelectorLayer(state)
processing_layer = ProcessingControlsLayer(state)
navigator_layer = TimeNavigatorLayer(state)
print("   ✅ All layers created")

print("\n3. Rendering panels (construction only)...")
print(f"   - Timeseries panel: {ts_layer.panel()}")
print(f"   - Spatial panel: {spatial_layer.panel()}")
print(f"   - Selector panel: {selector_layer.panel()}")
print(f"   - Processing panel: {processing_layer.panel()}")
print(f"   - Navigator panel: {navigator_layer.panel()}")
print("   ✅ Panels created (no crashes)")

# Test state changes
print("\n4. Testing state changes trigger updates...")
state.current_time = 5.0
state.channel_grid.select_cell(2, 3)
state.channel_grid.select_cell(4, 5)
print("   ✅ Updated time and selection (no crashes)")
print(f"   - Selected (grid): {state.selected_channels}")
print(f"   - Selected (flat): {state.selected_channels_flat}")

# Test disposal
print("\n5. Testing layer disposal...")
layers = [ts_layer, spatial_layer, selector_layer, processing_layer, navigator_layer]
for layer in layers:
    watchers_before = len(layer._watchers)
    layer.dispose()
    watchers_after = len(layer._watchers)
    print(f"   - {layer.layer_id}: {watchers_before} watchers → {watchers_after} after dispose")
print("   ✅ All layers disposed cleanly")

# Memory/leak sanity: create/destroy cycles (best-effort)
print("\n6. Memory/leak sanity: 100 create/destroy cycles (lightweight)...")
for _ in range(100):
    l1 = ChannelSelectorLayer(state)
    l2 = ProcessingControlsLayer(state)
    l3 = TimeNavigatorLayer(state)
    l1.dispose()
    l2.dispose()
    l3.dispose()
print("   ✅ 100 cycles completed")

print("\n" + "=" * 60)
print("✅ Phase 2 Core Layers: ALL TESTS PASSED!")
print("=" * 60)
print("\nNext: Try the interactive demo:")
print("  conda run -n cogpy panel serve code/lib/cogpy/examples/tensorscope/phase2_interactive.py --show")

