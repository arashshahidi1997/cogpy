"""
Hello TensorScope - Phase 6 (signal-centric) demo.

This demonstrates:
- Loading data with cogpy.datasets
- TensorScopeApp (signal-centric state + SignalManagerLayer)
- Adding a PSD display (analysis on windows via cogpy.core.spectral.specx.psdx)
- Panel server deployment

Run with:
    panel serve code/lib/cogpy/examples/tensorscope/hello_tensorscope.py --show
"""

from __future__ import annotations

import panel as pn

from cogpy.datasets.entities import example_ieeg_grid

pn.extension("tabulator")

print("Loading example iEEG data...")
data = example_ieeg_grid(mode="small")
print(f"Loaded: {data.dims}, {data.sizes}")

print("Phase 6: Creating TensorScope application...")
from cogpy.core.plot.tensorscope import TensorScopeApp

app = (
    TensorScopeApp(data, title="TensorScope (Phase 6 Demo)")
    .with_layout("default")
    .add_layer("timeseries")
    .add_layer("spatial_map")
    .add_layer("selector")
    .add_layer("signal_manager")
    .add_layer("processing")
    .add_layer("navigator")
)
print(f"✅ App created with {len(app.layer_manager.list_instances())} layers")

state = app.state
try:
    state.selected_time = float(state.time_hair.t) if state.time_hair.t is not None else float(data.time.values[0])
except Exception:  # noqa: BLE001
    pass

def _psd_view(
    selected_time: float,
    span_s: float,
    method: str,
    bandwidth: float,
    nperseg: int,
):
    import holoviews as hv

    from cogpy.core.spectral.specx import psdx

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

    psd = psdx(
        win,
        method=str(method),  # type: ignore[arg-type]
        bandwidth=float(bandwidth),
        nperseg=int(nperseg),
    )

    f = psd["freq"].values
    y = psd.values

    return hv.Curve((f, y), kdims=["freq"], vdims=["power"]).opts(
        width=320,
        height=220,
        xlabel="Frequency (Hz)",
        ylabel="Power",
        tools=["hover"],
        line_width=2,
        title=f"PSD @ t={t_mid:.2f}s  (span={span:.2f}s, {method})",
    )


psd_span = pn.widgets.FloatSlider(name="PSD window (s)", start=0.25, end=10.0, value=2.0, step=0.25)
psd_method = pn.widgets.Select(name="Method", options=["multitaper", "welch"], value="multitaper")
psd_bw = pn.widgets.FloatInput(name="Bandwidth (Hz)", value=4.0, step=0.5, width=140)
psd_nperseg = pn.widgets.IntInput(name="nperseg", value=512, step=64, width=140)
use_cursor = pn.widgets.Button(name="Use cursor", button_type="primary", width=120)


def _on_use_cursor(_event=None):
    try:
        state.set_selected_time_from_cursor()
    except Exception:  # noqa: BLE001
        pass


use_cursor.on_click(_on_use_cursor)

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
        pn.Row(use_cursor),
        psd_span,
        pn.Row(psd_method, psd_bw),
        psd_nperseg,
        pn.pane.HoloViews(psd_plot, sizing_mode="stretch_width"),
        sizing_mode="stretch_width",
    ),
    title="PSD",
    collapsed=False,
)

demo = app.build()
try:
    demo.sidebar.append(psd_card)
except Exception:  # noqa: BLE001
    # Fallback: replace sidebar completely if append is unsupported.
    demo.sidebar = list(demo.sidebar) + [psd_card]

demo.servable()

print("✅ Demo ready!")
