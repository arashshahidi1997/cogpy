"""
PSD Explorer Demo (v2.8.0).

Runs the `psd_explorer` module on example grid data.

Run with:
    /storage/share/python/environments/Anaconda3/envs/cogpy/bin/python \\
        code/lib/cogpy/examples/tensorscope/psd_explorer_demo.py
"""

from __future__ import annotations

import holoviews as hv
import panel as pn

from cogpy.plot.tensorscope.modules import ModuleRegistry
from cogpy.plot.tensorscope.state import TensorScopeState
from cogpy.datasets.entities import example_ieeg_grid


def main():
    pn.extension()
    hv.extension("bokeh")
    
    print("Loading data...")
    data = example_ieeg_grid(mode="small")
    print(f"  Data shape: {data.shape}")
    
    print("Creating TensorScope state...")
    state = TensorScopeState(data)
    
    print("Loading PSD Explorer module...")
    reg = ModuleRegistry()
    mod = reg.get("psd_explorer")
    
    if mod is None:
        print("  ❌ Module not found!")
        layout = pn.pane.Markdown("""
        ## Error: psd_explorer module not found
        
        Make sure the module is registered in ModuleRegistry.
        """)
    else:
        print("  ✅ Module found, activating...")
        layout = mod.activate(state)
        print("  ✅ Layout created")
    
    template = pn.template.FastListTemplate(
        title="TensorScope v2.8.0: PSD Explorer Demo",
        main=[layout]  # ✅ No sizing_mode here
    )
    
    print("\n" + "="*60)
    print("✅ PSD Explorer Ready!")
    print("="*60)
    print("\nFeatures:")
    print("  • Stacked time traces")
    print("  • PSD heatmap (frequency × channel)")
    print("  • Average PSD (mean ± std)")
    print("  • Spatial PSD map (at selected freq)")
    print("  • Interactive filtering")
    print("\nControls:")
    print("  - Adjust time → updates PSD window")
    print("  - Select filter type → applies filter")
    print("  - Change frequency → updates spatial map")
    
    return template


if __name__ == "__main__":
    main().show()