"""
Phase 8: View Builder & Module Selector UI (v2.3).

Demonstrates:
- Module selector UI for switching presets
- View builder UI for interactively composing ViewSpecs
- Advanced modules (montage, electrode panel)
- Cursor time slider driving v2.2/v2.3 views

Run with:
    conda run -n cogpy panel serve code/lib/cogpy/examples/tensorscope/phase8_view_builder.py --show
"""

from __future__ import annotations

import panel as pn

from cogpy.core.plot.tensorscope import ModuleSelectorLayer, TensorScopeState, ViewBuilderLayer
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
module_selector = ModuleSelectorLayer(state, registry)
view_builder = ViewBuilderLayer(state)

# Optional: auto-load default module into the main area.
module_selector._on_load()  # noqa: SLF001

time_slider = pn.widgets.FloatSlider(
    name="Cursor Time (s)",
    start=float(data.time.values[0]),
    end=float(data.time.values[-1]),
    value=float(state.time_hair.t or 0.0),
    step=0.01,
)


def _update_time(event):
    state.time_hair.t = float(event.new)


time_slider.param.watch(_update_time, "value")

template = pn.template.FastGridTemplate(
    title="TensorScope Phase 8: View Builder (v2.3)",
    theme="dark",
    sidebar_width=360,
    row_height=90,
)

template.sidebar.append(
    pn.Column(
        pn.pane.Markdown("# TensorScope v2.3"),
        pn.pane.Markdown(
            "- Use **Module Selector** to load presets\n"
            "- Use **View Builder** to preview a custom view\n"
            "- Drag **Cursor Time** to update linked views"
        ),
        pn.layout.Divider(),
        time_slider,
        pn.layout.Divider(),
        module_selector.panel(),
        pn.layout.Divider(),
        view_builder.panel(),
    )
)

template.main[0:12, 0:12] = module_selector.get_layout_container()

template.servable()

