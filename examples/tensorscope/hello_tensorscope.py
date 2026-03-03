"""
Hello TensorScope - Minimal working demo.

This demonstrates:
- Loading data with cogpy.datasets
- Importing TensorScope state/app scaffolding (Phase 0)
- Building a FastGridTemplate layout with placeholder cards
- Panel server deployment

Run with:
    panel serve code/lib/cogpy/examples/tensorscope/hello_tensorscope.py --show
"""

from __future__ import annotations

import panel as pn

from cogpy.core.plot.theme import BG, BG_PANEL, BLUE, TEAL, TEXT
from cogpy.datasets.entities import example_ieeg_grid

pn.extension("tabulator")

print("Loading example iEEG data...")
data = example_ieeg_grid(mode="small")
print(f"Loaded: {data.dims}, {data.sizes}")
print(f"Theme: BG={BG}, BG_PANEL={BG_PANEL}, TEXT={TEXT}, BLUE={BLUE}, TEAL={TEAL}")

# Phase 1: Create real state with controllers!
print("Phase 1: Creating TensorScope state with real controllers...")
from cogpy.core.plot.tensorscope import TensorScopeState

try:
    state = TensorScopeState(data)
    print("✅ State created successfully!")
    print("   - Controllers: TimeHair, TimeWindowCtrl, ChannelGrid, ProcessingChain")
    print(f"   - Time bounds: {state.time_window.bounds}")
    print(f"   - Grid size: {state.channel_grid.n_ap}x{state.channel_grid.n_ml}")
    state_created = True
except Exception as e:
    print(f"⚠️  Could not create state: {e}")
    state = None
    state_created = False

# Phase 2: Create real layers!
print("Phase 2: Creating visualization layers...")
try:
    if not state_created or state is None:
        raise RuntimeError("State not available")

    from cogpy.core.plot.tensorscope.layers import (
        ChannelSelectorLayer,
        ProcessingControlsLayer,
        SpatialMapLayer,
        TimeseriesLayer,
        TimeNavigatorLayer,
    )

    ts_layer = TimeseriesLayer(state, show_hair=True)
    spatial_layer = SpatialMapLayer(state, mode="rms", window_s=0.1)
    selector_layer = ChannelSelectorLayer(state)
    processing_layer = ProcessingControlsLayer(state)
    navigator_layer = TimeNavigatorLayer(state)

    print(f"✅ Layers created: {ts_layer.layer_id}, {spatial_layer.layer_id}")
    layers_created = True
except Exception as e:
    print(f"⚠️  Could not create layers: {e}")
    layers_created = False


def build_app() -> pn.template.base.BasicTemplate:
    """Build Phase 0 demo layout using FastGridTemplate."""

    spatial_placeholder = pn.Card(
        pn.Column(
            pn.pane.Markdown(
                "### Spatial Map Layer\n\n"
                "**Phase 2 will implement:**\n"
                "- GridFrameElement wrapper\n"
                "- Time-linked RMS/mean display\n"
                "- Colormap controls\n"
                "- Electrode grid overlay",
                styles={"color": TEXT, "padding": "10px 10px 0 10px"},
            ),
            pn.pane.HTML(
                "<div style=\"background:#2a2a3a; padding:16px; border-radius:6px; color:#cdd6f4;\">"
                "<b>Status:</b> placeholders only (Phase 0)"
                "</div>",
                sizing_mode="stretch_width",
                styles={"padding": "0 10px 10px 10px"},
            ),
            sizing_mode="stretch_both",
        ),
        title="Spatial View (Phase 2)",
        header_background=BLUE,
        styles={"background": BG_PANEL, "padding": "10px"},
        sizing_mode="stretch_both",
        min_height=350,
    )

    timeseries_placeholder = pn.Card(
        pn.Column(
            pn.pane.Markdown(
                "### Timeseries Layer\n\n"
                "**Phase 2 will implement:**\n"
                "- MultichannelViewer wrapper\n"
                "- Channel selection binding\n"
                "- Time cursor display\n"
                "- Windowed processing",
                styles={"color": TEXT, "padding": "10px 10px 0 10px"},
            ),
            pn.pane.HTML(
                "<div style=\"background:#2a2a3a; padding:16px; border-radius:6px; color:#cdd6f4;\">"
                "<b>Status:</b> placeholders only (Phase 0)"
                "</div>",
                sizing_mode="stretch_width",
                styles={"padding": "0 10px 10px 10px"},
            ),
            sizing_mode="stretch_both",
        ),
        title="Timeseries View (Phase 2)",
        header_background=BLUE,
        styles={"background": BG_PANEL, "padding": "10px"},
        sizing_mode="stretch_both",
        min_height=350,
    )

    controls_placeholder = pn.Card(
        pn.Column(
            pn.pane.Markdown("**Controls**\n\nPhase 2 layers:", styles={"color": TEXT}),
            pn.widgets.IntSlider(name="Placeholder slider", start=0, end=10),
            pn.widgets.Checkbox(name="Placeholder checkbox"),
        ),
        title="Controls (Phase 2)",
        header_background=TEAL,
        styles={"background": BG_PANEL, "padding": "10px"},
    )

    if layers_created:
        spatial_card = pn.Card(
            spatial_layer.panel(),
            title=spatial_layer.title,
            header_background=BLUE,
            styles={"background": BG_PANEL},
            sizing_mode="stretch_both",
            min_height=350,
        )
        timeseries_card = pn.Card(
            pn.Column(
                navigator_layer.panel(),
                ts_layer.panel(),
                sizing_mode="stretch_both",
            ),
            title=ts_layer.title,
            header_background=BLUE,
            styles={"background": BG_PANEL},
            sizing_mode="stretch_both",
            min_height=350,
        )
        sidebar_controls = pn.Column(
            selector_layer.panel(),
            processing_layer.panel(),
        )
    else:
        spatial_card = spatial_placeholder
        timeseries_card = timeseries_placeholder
        sidebar_controls = controls_placeholder

    sidebar_markdown = pn.pane.Markdown(
        "### Dataset Info\n\n"
        f"- Dims: `{data.dims}`\n"
        f"- Shape: `{data.shape}`\n"
        f"- Time: {float(data.time.values[0]):.2f}s - {float(data.time.values[-1]):.2f}s\n\n"
        "**Phase 2 Status:**\n"
        "- ✅ Package structure\n"
        "- ✅ State implementation\n"
        "- ✅ **Core layers complete!**\n"
        "- ⏳ Phase 3: Application shell\n\n",
        styles={"color": TEXT},
    )

    if state_created and state is not None:
        live_state_info = pn.pane.Markdown(
            "**Live State:**\n"
            f"- Current time: {state.current_time or 'None'}\n"
            f"- Time window: `{state.time_window.window}`\n"
            f"- Selected channels: {len(state.selected_channels)}\n"
            f"- Grid size: {state.channel_grid.n_ap}×{state.channel_grid.n_ml}\n",
            styles={
                "color": TEXT,
                "padding": "10px",
                "background": "#2a2a3a",
                "border-radius": "5px",
            },
        )
    else:
        live_state_info = pn.pane.Markdown(
            "*State not available*",
            styles={"color": "#888"},
        )

    sidebar = pn.Column(
        sidebar_markdown,
        live_state_info,
        sidebar_controls,
    )

    tmpl = pn.template.FastGridTemplate(
        title="TensorScope v0.0 (Phase 0 Demo)",
        theme="dark",
        sidebar_width=320,
        sidebar=[sidebar],
        row_height=80,
    )

    # Panel 1.8.8: FastGridTemplate.main is a GridSpec (no .append()).
    # Use grid assignment instead.
    tmpl.main[0:5, 0:12] = spatial_card
    tmpl.main[5:10, 0:12] = timeseries_card

    return tmpl


app = build_app()
app.servable()

print("✅ Phase 0 demo ready!")
