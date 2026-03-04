"""
Phase 9: Orthoslicer - Time-Frequency-Spatial Exploration (v2.4).

Demonstrates:
- Spectrogram orthoslicer module (4 linked views)
- Multi-stream coordinate linking: spatial (AP/ML), temporal (time), spectral (freq)

Run with:
    conda run -n cogpy panel serve code/lib/cogpy/examples/tensorscope/phase9_orthoslicer.py --show
"""

from __future__ import annotations

import panel as pn

from cogpy.core.plot.tensorscope import TensorScopeState
from cogpy.core.plot.tensorscope.modules import ModuleRegistry
from cogpy.datasets.entities import example_ieeg_grid

pn.extension()

data = example_ieeg_grid(mode="small")
state = TensorScopeState(data)

state.time_hair.t = 5.0
try:
    state.spatial_space.set_selection("AP", 3)
    state.spatial_space.set_selection("ML", 5)
except Exception:  # noqa: BLE001
    pass

registry = ModuleRegistry()
orthoslicer = registry.get("orthoslicer")

layout = orthoslicer.activate(state) if orthoslicer is not None else pn.pane.Markdown("**No orthoslicer module**")

info = pn.pane.Markdown(
    "## Orthoslicer (v2.4)\n\n"
    "**Four linked views:**\n"
    "- Time profile (time @ freq + spatial)\n"
    "- Freq profile (freq @ time + spatial)\n"
    "- TF heatmap (time×freq @ spatial)\n"
    "- Spatial map (AP×ML @ time + freq)\n\n"
    "**Try:** click TF heatmap to set (time,freq); click Spatial map to set (AP,ML).",
    sizing_mode="stretch_width",
)

template = pn.template.FastGridTemplate(
    title="TensorScope Phase 9: Orthoslicer (v2.4)",
    theme="dark",
    sidebar_width=360,
    row_height=90,
)

template.sidebar.append(info)
template.main[0:12, 0:12] = pn.pane.HoloViews(layout, sizing_mode="stretch_both")

template.servable()

