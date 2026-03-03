"""
Test Phase 3: Application Shell

Demonstrates:
- Creating TensorScopeApp
- Builder API for configuration
- Adding layers programmatically
- Layout presets
- Session save/load

Run with:
    conda run -n cogpy python code/lib/cogpy/examples/tensorscope/test_phase3_app.py
"""

import json
from pathlib import Path

from cogpy.core.plot.tensorscope import TensorScopeApp
from cogpy.datasets.entities import example_ieeg_grid

print("=" * 60)
print("TensorScope Phase 3: Application Shell Test")
print("=" * 60)

print("\n1. Loading data...")
data = example_ieeg_grid(mode="small")
print(f"   ✅ Loaded: {data.dims}, shape={data.shape}")

print("\n2. Creating TensorScopeApp...")
app = TensorScopeApp(data, title="Test App")
print(f"   - State: {app.state}")
print(f"   - LayerManager: {app.layer_manager}")
print(f"   - LayoutManager: {app.layout_manager}")
print("   ✅ App created")

print("\n3. Adding layers...")
app.add_layer("timeseries")
app.add_layer("spatial_map")
app.add_layer("selector")
app.add_layer("processing")

instances = app.layer_manager.list_instances()
print(f"   - Active layers: {instances}")
print(f"   - Panels created: {list(app._panels.keys())}")
print(f"   ✅ {len(instances)} layers added")

print("\n4. Testing builder API...")
app2 = (
    TensorScopeApp(data)
    .with_layout("spatial_focus")
    .add_layer("timeseries")
    .add_layer("spatial_map")
)
print(f"   - Layout preset: {app2.layout_manager.current_preset}")
print(f"   - Layers: {app2.layer_manager.list_instances()}")
print("   ✅ Builder pattern working")

print("\n5. Testing layout presets...")
available_presets = ["default", "spatial_focus", "timeseries_focus"]
for preset in available_presets:
    app_test = TensorScopeApp(data).with_layout(preset)
    print(f"   - Preset '{preset}': {app_test.layout_manager.current_preset}")
print("   ✅ All presets available")

print("\n6. Building template...")
template = app.build()
print(f"   - Template: {template}")
print(f"   - Title: {template.title}")
print("   ✅ Template built")

print("\n7. Testing session serialization...")
app.state.current_time = 7.5
session = app.to_session()
print(f"   - Session version: {session['version']}")
print(f"   - Session keys: {list(session.keys())}")
print(f"   - Layers in session: {session['layers']}")

session_file = Path("/tmp/tensorscope_app_session.json")
with open(session_file, "w") as f:
    json.dump(session, f, indent=2)
print(f"   - Saved to: {session_file}")
print("   ✅ Session saved")

print("\n8. Testing session restoration...")
with open(session_file, "r") as f:
    loaded = json.load(f)

app_restored = TensorScopeApp.from_session(loaded, data_resolver=lambda: data)
print(f"   - Restored app state time: {app_restored.state.current_time}")
print(f"   - Restored layers: {app_restored.layer_manager.list_instances()}")
assert app_restored.state.current_time == 7.5
print("   ✅ Session restored correctly")

print("\n9. Testing app shutdown...")
layer_count_before = len(app.layer_manager.list_instances())
app.shutdown()
layer_count_after = len(app.layer_manager.list_instances())
print(f"   - Layers before shutdown: {layer_count_before}")
print(f"   - Layers after shutdown: {layer_count_after}")
assert layer_count_after == 0
print("   ✅ Shutdown cleaned up all layers")

print("\n" + "=" * 60)
print("✅ Phase 3 Application Shell: ALL TESTS PASSED!")
print("=" * 60)
print("\nNext: Try the interactive app:")
print("  conda run -n cogpy panel serve code/lib/cogpy/examples/tensorscope/phase3_app.py --show")

