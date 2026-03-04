"""
Phase 7: View Composer (v2.2) - ViewSpec + ViewFactory + Modules.

Demonstrates:
- Declarative `ViewSpec`
- `ViewFactory.create()` -> HoloViews DynamicMap
- Preset modules via `ModuleRegistry`
- Linked cursor time (TimeHair) + linked spatial selection (CoordinateSpace)

Run with:
    conda run -n cogpy panel serve code/lib/cogpy/examples/tensorscope/phase7_view_composer.py --show
"""

from __future__ import annotations

import panel as pn

from cogpy.core.plot.tensorscope import TensorScopeState
from cogpy.core.plot.tensorscope.modules import ModuleRegistry
from cogpy.core.plot.tensorscope.view_factory import ViewFactory
from cogpy.core.plot.tensorscope.view_spec import ViewSpec
from cogpy.datasets.entities import example_ieeg_grid

pn.extension()

data = example_ieeg_grid(mode="small")
state = TensorScopeState(data)

state.time_hair.t = 5.0
state.selected_time = 5.0

# Manual ViewSpec creation.
spatial_spec = ViewSpec(
    kdims=["AP", "ML"],
    controls=["time"],
    colormap="RdBu_r",
    symmetric_clim=True,
    title="Spatial LFP (ViewSpec)",
)
temporal_spec = ViewSpec(
    kdims=["time"],
    controls=["AP", "ML"],
    title="Timeseries (ViewSpec)",
)

spatial_view = ViewFactory.create(spatial_spec, state)
temporal_view = ViewFactory.create(temporal_spec, state)

# Preset module activation.
registry = ModuleRegistry()
basic_module = registry.get("basic")
module_layout = basic_module.activate(state) if basic_module is not None else pn.pane.Markdown("**No module**")

info = pn.pane.Markdown(
    "## Phase 7: View Composer (v2.2)\n\n"
    "**Try:**\n"
    "- Drag the time slider → updates spatial + temporal views\n"
    "- Click the spatial map → updates (AP, ML) selection and timeseries\n\n"
    "**Notes:**\n"
    "- AP/ML are index selections (0..N-1)\n"
    "- Views are generated from declarative `ViewSpec`\n",
    sizing_mode="stretch_width",
)

time_slider = pn.widgets.FloatSlider(
    name="Cursor Time (s)",
    start=float(data.time.values[0]),
    end=float(data.time.values[-1]),
    value=5.0,
    step=0.01,
)


def _update_time(event):
    state.time_hair.t = float(event.new)


time_slider.param.watch(_update_time, "value")

template = pn.template.FastGridTemplate(
    title="TensorScope Phase 7: View Composer (v2.2)",
    theme="dark",
    sidebar_width=320,
    row_height=90,
)

template.sidebar.append(pn.Column(info, pn.layout.Divider(), time_slider))

template.main[0:1, 0:12] = pn.pane.Markdown("**Manual: ViewSpec + ViewFactory**")
template.main[1:6, 0:6] = pn.pane.HoloViews(spatial_view, sizing_mode="stretch_both")
template.main[1:6, 6:12] = pn.pane.HoloViews(temporal_view, sizing_mode="stretch_both")

template.main[6:7, 0:12] = pn.pane.Markdown("**Preset Module: basic**")
template.main[7:12, 0:12] = pn.pane.HoloViews(module_layout, sizing_mode="stretch_both")

template.servable()

