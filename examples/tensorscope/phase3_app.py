"""
Phase 3 Complete Application Demo

Full TensorScope app with:
- Application shell (TensorScopeApp)
- All layers integrated
- Layout presets
- Real data visualization

Run with:
    conda run -n cogpy panel serve code/lib/cogpy/examples/tensorscope/phase3_app.py --show
"""

from cogpy.plot.tensorscope import TensorScopeApp
from cogpy.datasets.entities import example_ieeg_grid

print("Loading data...")
data = example_ieeg_grid(mode="small")

print("Creating TensorScope app...")
app = (
    TensorScopeApp(data, title="TensorScope v0.3 (Phase 3)")
    .with_layout("default")
    .add_layer("timeseries")
    .add_layer("spatial_map")
    .add_layer("selector")
    .add_layer("processing")
    .add_layer("navigator")
)

print("✅ App created with all layers!")
print(f"   Layers: {app.layer_manager.list_instances()}")
print(f"   Layout: {app.layout_manager.current_preset}")

app.servable()

print("✅ Phase 3 app ready!")
print("Try:")
print("  - Select channels in grid")
print("  - Adjust processing controls")
print("  - Use time navigator to play")

