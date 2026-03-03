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

print("Phase 0: TensorScopeState exists but controllers are not wired yet (Phase 1).")


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

    sidebar = pn.Column(
        pn.pane.Markdown(
            "### Dataset Info\n\n"
            f"- Dims: `{data.dims}`\n"
            f"- Shape: `{data.shape}`\n"
            f"- Time: {float(data.time.values[0]):.2f}s - {float(data.time.values[-1]):.2f}s\n\n"
            "**Phase 0 Status:**\n"
            "- ✅ Package structure created\n"
            "- ✅ FastGridTemplate working\n"
            "- ⏳ Phase 1: State implementation\n"
            "- ⏳ Phase 2: Core layers",
            styles={"color": TEXT},
        ),
        controls_placeholder,
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
    tmpl.main[0:5, 0:12] = pn.Row(spatial_placeholder, sizing_mode="stretch_width")
    tmpl.main[5:10, 0:12] = pn.Row(timeseries_placeholder, sizing_mode="stretch_width")

    return tmpl


app = build_app()
app.servable()

print("✅ Phase 0 demo ready!")
