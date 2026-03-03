"""
TensorScope Phase 1 Interactive Demo (script version).

This is a notebook-style script (with "cells" separated by headers) that
demonstrates the Phase 1 state architecture interactively using Panel.

Run with:
    conda run -n cogpy panel serve code/lib/cogpy/examples/tensorscope/phase1_interactive.py --show

Or, as a plain script:
    conda run -n cogpy python code/lib/cogpy/examples/tensorscope/phase1_interactive.py
"""

from __future__ import annotations

import json

import panel as pn

from cogpy.core.plot.channel_grid_widget import ChannelGridWidget
from cogpy.core.plot.tensorscope import TensorScopeState
from cogpy.datasets.entities import example_ieeg_grid

pn.extension()


# %% Cell 1: Load data
data = example_ieeg_grid(mode="small")
print(f"Loaded: {data.dims}, shape={data.shape}")
print(
    f"Time range: {float(data.time.values[0]):.2f}s - {float(data.time.values[-1]):.2f}s"
)


# %% Cell 2: Create state
state = TensorScopeState(data)
state.current_time = float(state.time_window.bounds[0])
print("State created!")
print("Controllers:")
print(f"  - TimeHair: {state.time_hair}")
print(f"  - TimeWindowCtrl: {state.time_window}")
print(f"  - ChannelGrid: {state.channel_grid}")
print(f"  - ProcessingChain: {state.processing}")


# %% Cell 3: Interactive time slider
time_slider = pn.widgets.FloatSlider(
    name="Current Time (s)",
    start=float(state.time_window.bounds[0]),
    end=float(state.time_window.bounds[1]),
    value=float(state.time_window.bounds[0]),
    step=0.1,
)


def _update_state_time(event):
    state.current_time = event.new


time_slider.param.watch(_update_state_time, "value")


@pn.depends(time_slider.param.value)
def _time_info(_t):
    ct = state.current_time
    ct_s = "None" if ct is None else f"{ct:.2f}s"
    return pn.pane.Markdown(
        f"**Current State:**\n"
        f"- Time: {ct_s}\n"
        f"- Window: {state.time_window.window}\n"
        f"- Selected: {len(state.selected_channels)} channels\n"
    )


time_nav = pn.Column(
    "## Time Navigation",
    time_slider,
    _time_info,
)


# %% Cell 4: Interactive channel selection
grid_widget = ChannelGridWidget.from_grid(state.channel_grid)


@pn.depends(state.channel_grid.param.selected)
def _selection_info(selected):
    return pn.pane.Markdown(
        f"**Selected:** {len(selected)} channels\n\n"
        f"Channels: {sorted(list(selected))}\n\n"
        f"Flat indices: {state.selected_channels_flat}"
    )


channel_select = pn.Column(
    "## Channel Selection",
    grid_widget.panel(),
    _selection_info,
)


# %% Cell 5: Serialization demo
serialization_header = pn.pane.Markdown("## Session Serialization")

state.current_time = 5.0
state.channel_grid.select_cell(2, 3)
state.channel_grid.select_cell(4, 5)

saved = state.to_dict()
saved_preview = pn.pane.Markdown(
    "Saved state (preview):\n\n```json\n"
    + json.dumps(saved, indent=2)[:300]
    + "\n... (truncated)\n```"
)

restored = TensorScopeState.from_dict(saved, data_resolver=lambda: data)
restored_preview = pn.pane.Markdown(
    f"Restored:\n\n- Time: `{restored.current_time}`\n- Selection: `{sorted(list(restored.selected_channels))}`"
)

serialization_demo = pn.Column(serialization_header, saved_preview, restored_preview)


# %% Cell 6: Time window operations
time_window_header = pn.pane.Markdown("## Time Window Operations")
state.time_window.set_window(2.0, 8.0)
state.time_window.recenter(5.0, width_s=3.0)
window_data = state.processing.get_window(
    state.time_window.window[0],
    state.time_window.window[1],
)
time_window_demo = pn.pane.Markdown(
    f"- Window: `{state.time_window.window}`\n"
    f"- Windowed data shape: `{window_data.shape}`\n"
)


# %% Cell 7: Complete dashboard
window_width = pn.widgets.FloatSlider(
    name="Window Width (s)",
    start=0.1,
    end=5.0,
    value=2.0,
    step=0.1,
)


def _recenter(_event=None):
    ct = state.current_time
    if ct is None:
        ct = float(state.time_window.bounds[0])
    state.time_window.recenter(ct, width_s=float(window_width.value))


window_width.param.watch(_recenter, "value")
_recenter()


@pn.depends(time_slider.param.value, window_width.param.value, state.channel_grid.param.selected)
def _live_state(_t, _w, _sel):
    ct = state.current_time
    ct_s = "None" if ct is None else f"{ct:.2f}s"
    return pn.pane.Markdown(
        f"### Live State\n\n"
        f"**Time:** {ct_s}\n\n"
        f"**Window:** `{state.time_window.window}`\n\n"
        f"**Selected:** {len(state.selected_channels)} channels\n\n"
        f"**Flat indices:** `{state.selected_channels_flat}`\n"
    )


dashboard = pn.Column(
    "# TensorScope Phase 1 Dashboard",
    pn.Row(pn.Column(time_slider, window_width, width=320), _live_state),
)


app = pn.Tabs(
    ("Dashboard", dashboard),
    ("Time + Selection", pn.Row(time_nav, channel_select)),
    ("Serialization", serialization_demo),
    ("Time Window", pn.Column(time_window_header, time_window_demo)),
)

app.servable()


if __name__ == "__main__":
    try:
        app.show()
    except Exception:
        # In some environments (headless / CI), .show() isn't appropriate.
        pass

