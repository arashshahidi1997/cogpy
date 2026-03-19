"""
Phase 2 Interactive Demo - Real Layers with Panel.

Shows actual visualization layers working together:
- Timeseries with real traces (MultichannelViewer)
- Spatial map updating with time cursor (GridFrameElement)
- Channel selector filtering traces
- Processing controls updating views
- Time navigator driving cursor

Run with:
    conda run -n cogpy panel serve code/lib/cogpy/examples/tensorscope/phase2_interactive.py --show
"""

from __future__ import annotations

import panel as pn

from cogpy.plot.theme import BG_PANEL, BLUE, TEXT
from cogpy.plot.tensorscope import TensorScopeState
from cogpy.plot.tensorscope.layers import (
    ChannelSelectorLayer,
    ProcessingControlsLayer,
    SpatialMapLayer,
    TimeseriesLayer,
    TimeNavigatorLayer,
)
from cogpy.datasets.entities import example_ieeg_grid

pn.extension("tabulator")

print("Loading data...")
data = example_ieeg_grid(mode="small")

print("Creating state and layers...")
state = TensorScopeState(data)
ts_layer = TimeseriesLayer(state, show_hair=True)
spatial_layer = SpatialMapLayer(state, mode="rms", window_s=0.1)
selector_layer = ChannelSelectorLayer(state)
processing_layer = ProcessingControlsLayer(state)
navigator_layer = TimeNavigatorLayer(state)

print("✅ All layers created!")


def build_app():
    info = pn.pane.Markdown(
        "# TensorScope Phase 2: Core Layers\n\n"
        "**All layers are now REAL visualization components.**\n\n"
        "Try:\n"
        "- Click grid to select channels → traces update\n"
        "- Use time player → spatial map updates\n"
        "- Toggle processing → views re-render\n\n",
        styles={"color": TEXT},
    )

    tmpl = pn.template.FastGridTemplate(
        title="TensorScope Phase 2 Demo",
        theme="dark",
        sidebar_width=360,
        sidebar=[
            info,
            selector_layer.panel(),
            processing_layer.panel(),
        ],
        row_height=80,
    )

    tmpl.main[0:5, 0:6] = pn.Card(
        spatial_layer.panel(),
        title=spatial_layer.title,
        header_background=BLUE,
        styles={"background": BG_PANEL},
        sizing_mode="stretch_both",
    )

    tmpl.main[0:5, 6:12] = pn.Card(
        navigator_layer.panel(),
        title=navigator_layer.title,
        header_background=BLUE,
        styles={"background": BG_PANEL},
        sizing_mode="stretch_both",
    )

    tmpl.main[5:10, 0:12] = pn.Card(
        ts_layer.panel(),
        title=ts_layer.title,
        header_background=BLUE,
        styles={"background": BG_PANEL},
        sizing_mode="stretch_both",
    )

    return tmpl


app = build_app()
app.servable()

print("✅ Phase 2 interactive demo ready!")

