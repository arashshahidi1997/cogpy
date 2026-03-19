"""Electrode panel module: all electrode timeseries in a GridSpace (v2.3)."""

from __future__ import annotations

from .base import ViewPresetModule

__all__ = ["MODULE"]


def _activate_electrode_panel(state):
    import holoviews as hv
    import numpy as np

    hv.extension("bokeh")

    reg = getattr(state, "signal_registry", None)
    sig = reg.get_active() if reg is not None else None
    if sig is None:
        return hv.Div("<b>No active signal</b>")

    data = sig.data
    if ("time" not in data.dims) or ("AP" not in data.dims) or ("ML" not in data.dims):
        return hv.Div("<b>Electrode panel requires grid data with dims: time, AP, ML</b>")

    traces = {}
    n_ap = int(data.sizes["AP"])
    n_ml = int(data.sizes["ML"])

    for ap in range(n_ap):
        for ml in range(n_ml):
            trace = data.isel(AP=int(ap), ML=int(ml))
            t_vals = np.asarray(trace["time"].values, dtype=float)
            y_vals = np.asarray(trace.values, dtype=float)

            curve = hv.Curve((t_vals, y_vals), kdims=["time"], vdims=["LFP (a.u.)"]).opts(
                width=150,
                height=100,
                show_legend=False,
                axiswise=True,
                xlabel="",
                ylabel="",
            )
            traces[(ap, ml)] = curve

    return hv.GridSpace(traces, kdims=["AP", "ML"]).opts(title="Electrode Panel")


MODULE = ViewPresetModule(
    name="electrode_panel",
    description="All electrode timeseries arranged by (AP, ML)",
    specs=[],
    layout="custom",
    activate_fn=_activate_electrode_panel,
)

