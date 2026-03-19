"""
Phase 6: Spatial LFP Views (v2.1) with wide layout.

Demonstrates:
- SpatialLFPView (instantaneous voltage at selected time)
- Three time modes: cursor, selected, independent
- Side-by-side comparison
- Spatial selection linking (AP/ML indices)

Run with:
    conda run -n cogpy panel serve code/lib/cogpy/examples/tensorscope/phase6_spatial_views.py --show
"""

from __future__ import annotations

import panel as pn

from cogpy.datasets.entities import example_ieeg_grid
from cogpy.plot.tensorscope.views.spatial_lfp import SpatialLFPView

pn.extension()

data = example_ieeg_grid(mode="small")
from cogpy.plot.tensorscope import TensorScopeState

state = TensorScopeState(data)

base_signal_id = state.signal_registry.list()[0]
filt_id = state.create_derived_signal(
    base_signal_id,
    "Filtered (1-100Hz)",
    {"bandpass_on": True, "bandpass_lo": 1.0, "bandpass_hi": 100.0},
)

state.time_hair.t = 5.0
state.selected_time = 5.0

view_raw_cursor = SpatialLFPView(state, base_signal_id, selected_time_source="cursor")
view_filt_cursor = SpatialLFPView(state, filt_id, selected_time_source="cursor")
view_raw_t7 = SpatialLFPView(
    state,
    base_signal_id,
    selected_time_source="independent",
    independent_time=7.0,
)

info = pn.pane.Markdown(
    "## Spatial LFP Views Demo\n\n"
    "- Left: Raw @ cursor (follows playback)\n"
    "- Middle: Filtered @ cursor\n"
    "- Right: Raw @ independent time\n\n"
    "**Try:** click on a map to set (AP, ML) selection; yellow X shows selection in all views.",
    sizing_mode="stretch_width",
)

time_slider = pn.widgets.FloatSlider(
    name="Cursor Time",
    start=float(data.time.values[0]),
    end=float(data.time.values[-1]),
    value=5.0,
    step=0.01,
)


def _update_time(event):
    state.time_hair.t = float(event.new)


time_slider.param.watch(_update_time, "value")

template = pn.template.FastGridTemplate(
    title="Phase 6: Spatial LFP Views (v2.1)",
    theme="dark",
    sidebar_width=320,
    row_height=90,
)

template.sidebar.append(pn.Column(info, pn.layout.Divider(), time_slider))

template.main[0:1, 0:4] = pn.pane.Markdown("**Raw @ cursor**")
template.main[0:1, 4:8] = pn.pane.Markdown("**Filtered @ cursor**")
template.main[0:1, 8:12] = pn.pane.Markdown("**Raw @ independent time**")

template.main[1:6, 0:4] = view_raw_cursor.panel()
template.main[1:6, 4:8] = view_filt_cursor.panel()
template.main[1:6, 8:12] = view_raw_t7.panel()

template.servable()

