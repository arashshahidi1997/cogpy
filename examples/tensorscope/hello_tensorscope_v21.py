"""
Hello TensorScope - v2.1 demo (views + spatial linking + duplication).

This demonstrates:
- Loading data with cogpy.datasets
- TensorScopeApp (signal-centric state + SignalManagerLayer)
- v2.1 SpatialLFPView (instantaneous voltage maps)
- Shared spatial selection via state.spatial_space (tap-to-select + linked marker)
- PSD display computed on windows via cogpy.spectral.specx.psdx
- Simple view duplication (clone an independent SpatialLFPView panel)

Run with:
    conda run -n cogpy panel serve code/lib/cogpy/examples/tensorscope/hello_tensorscope_v21.py --show
"""

from __future__ import annotations

import panel as pn

from cogpy.datasets.entities import example_ieeg_grid

pn.extension("tabulator")

print("Loading example iEEG data...")
data = example_ieeg_grid(mode="small")
print(f"Loaded: {data.dims}, {data.sizes}")

from cogpy.plot.tensorscope import TensorScopeApp
from cogpy.plot.tensorscope.views.spatial_lfp import SpatialLFPView

print("Creating TensorScope app...")
app = (
    TensorScopeApp(data, title="TensorScope v2.1 Demo")
    .with_layout("default")
    .add_layer("timeseries")
    .add_layer("spatial_map")
    .add_layer("selector")
    .add_layer("signal_manager")
    .add_layer("processing")
    .add_layer("navigator")
)
state = app.state

# Ensure selected_time has a value for views that follow it.
try:
    state.selected_time = float(data.time.values[0])
except Exception:  # noqa: BLE001
    pass

# Create a derived, filtered signal to compare side-by-side.
base_id = state.signal_registry.list()[0]
filt_id = state.create_derived_signal(
    base_id,
    "Filtered (1-100Hz)",
    {"bandpass_on": True, "bandpass_lo": 1.0, "bandpass_hi": 100.0},
)

# v2.1 views
view_raw_cursor = SpatialLFPView(state, base_id, selected_time_source="cursor")
view_filt_cursor = SpatialLFPView(state, filt_id, selected_time_source="cursor")
view_raw_independent = SpatialLFPView(
    state, base_id, selected_time_source="independent", independent_time=7.0
)

extra_views = pn.Column(sizing_mode="stretch_width")


def _duplicate_independent(_event=None) -> None:
    # Clone the independent view at the current selected_time.
    t = getattr(state, "selected_time", None)
    if t is None:
        return
    v = view_raw_independent.duplicate(independent_time=float(t))
    extra_views.append(
        pn.Card(
            v.panel(),
            title=f"Spatial LFP (dup @ t={float(t):.2f}s)",
            collapsed=False,
            sizing_mode="stretch_width",
        )
    )


dup_btn = pn.widgets.Button(name="Duplicate independent view", button_type="primary")
dup_btn.on_click(_duplicate_independent)

use_cursor_btn = pn.widgets.Button(name="Set selected_time = cursor", button_type="success")
use_cursor_btn.on_click(lambda _e=None: state.set_selected_time_from_cursor())


def _psd_view(
    selected_time: float,
    span_s: float,
    method: str,
    bandwidth: float,
    nperseg: int,
):
    import holoviews as hv

    from cogpy.spectral.specx import psdx

    hv.extension("bokeh")

    sig = state.signal_registry.get_active() if state.signal_registry is not None else None
    if sig is None:
        return hv.Text(0, 0, "No active signal")

    t_mid = float(selected_time)
    span = float(span_s)
    t0, t1 = t_mid - span / 2.0, t_mid + span / 2.0

    channels = list(state.selected_channels_flat) if getattr(state, "selected_channels_flat", []) else None
    win = sig.get_window(t0, t1, channels=channels)

    reduce_dims = [d for d in win.dims if d != "time"]
    if reduce_dims:
        win = win.mean(dim=reduce_dims)

    psd = psdx(win, method=str(method), bandwidth=float(bandwidth), nperseg=int(nperseg))  # type: ignore[arg-type]
    return hv.Curve((psd["freq"].values, psd.values), kdims=["freq"], vdims=["power"]).opts(
        width=360,
        height=220,
        xlabel="Frequency (Hz)",
        ylabel="Power",
        tools=["hover"],
        line_width=2,
        title=f"PSD @ t={t_mid:.2f}s ({method})",
    )


psd_span = pn.widgets.FloatSlider(name="PSD window (s)", start=0.25, end=10.0, value=2.0, step=0.25)
psd_method = pn.widgets.Select(name="Method", options=["multitaper", "welch"], value="multitaper")
psd_bw = pn.widgets.FloatInput(name="Bandwidth (Hz)", value=4.0, step=0.5, width=140)
psd_nperseg = pn.widgets.IntInput(name="nperseg", value=512, step=64, width=140)

psd_plot = pn.bind(
    _psd_view,
    selected_time=state.param.selected_time,
    span_s=psd_span,
    method=psd_method,
    bandwidth=psd_bw,
    nperseg=psd_nperseg,
)

psd_card = pn.Card(
    pn.Column(
        pn.pane.Markdown(
            "### PSD (window → specx.psdx)\n\n"
            "Uses the **active signal** + its processing, then computes PSD on the processed window.",
        ),
        pn.Row(use_cursor_btn),
        psd_span,
        pn.Row(psd_method, psd_bw),
        psd_nperseg,
        pn.pane.HoloViews(psd_plot, sizing_mode="stretch_width"),
        sizing_mode="stretch_width",
    ),
    title="PSD",
    collapsed=False,
)

# Build our own wide layout template using app-generated cards.
template = pn.template.FastGridTemplate(
    title="TensorScope v2.1 Demo",
    theme="dark",
    sidebar_width=380,
    row_height=90,
)

template.sidebar.append(
    pn.Column(
        pn.pane.Markdown(
            "## v2.1 demo\n\n"
            "- SpatialLFPView panels are **instantaneous** AP×ML slices.\n"
            "- Click any spatial panel to set (AP, ML) selection (yellow X).\n"
            "- Left/middle follow cursor; right is independent.\n",
            sizing_mode="stretch_width",
        ),
        pn.Row(dup_btn),
        psd_card,
        pn.layout.Divider(),
        pn.pane.Markdown("### Duplicated views"),
        extra_views,
        sizing_mode="stretch_width",
    )
)

# Main: 2×4-ish grid.
template.main[0:6, 0:6] = app._panels["timeseries"]
template.main[0:6, 6:12] = app._panels["spatial_map"]

template.main[6:11, 0:4] = pn.Card(view_raw_cursor.panel(), title="Spatial LFP (raw @ cursor)", sizing_mode="stretch_both")
template.main[6:11, 4:8] = pn.Card(view_filt_cursor.panel(), title="Spatial LFP (filt @ cursor)", sizing_mode="stretch_both")
template.main[6:11, 8:12] = pn.Card(view_raw_independent.panel(), title="Spatial LFP (raw independent)", sizing_mode="stretch_both")

template.main[11:14, 0:4] = app._panels["signal_manager"]
template.main[11:14, 4:8] = app._panels["processing"]
template.main[11:14, 8:12] = app._panels["navigator"]

template.servable()

print("✅ TensorScope v2.1 demo ready!")

