"""
Event Explorer module (v2.6.2).

This module provides a minimal, HoloViews-native event exploration layout:
- Spatial + temporal views (via ViewFactory) with event overlays
- Lightweight statistics panels (rate, spatial/frequency distributions)

Notes
-----
TensorScope's v2.2 module system returns HoloViews objects (not Panel layouts).
For richer table/navigation UIs, prefer the "layers" stack (EventTableLayer).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..layers.events import EventOverlayLayer
from ..view_factory import ViewFactory
from ..view_spec import ViewSpec
from .base import ViewPresetModule

__all__ = ["EventExplorerModule", "MODULE", "create_event_explorer_module"]


def _first_event_stream_name(state) -> str | None:
    reg = getattr(state, "event_registry", None)
    if reg is None:
        return None
    try:
        names = reg.list()
    except Exception:  # noqa: BLE001
        names = []
    return names[0] if names else None


def _event_rate_curve(df, *, time_col: str, bin_s: float = 0.5):
    import holoviews as hv

    hv.extension("bokeh")

    if df.empty:
        return hv.Curve([], kdims=["time"], vdims=["rate"])

    t = np.asarray(df[time_col].to_numpy(), dtype=float)
    t = t[np.isfinite(t)]
    if t.size == 0:
        return hv.Curve([], kdims=["time"], vdims=["rate"])

    t_min = float(np.nanmin(t))
    t_max = float(np.nanmax(t))
    if not np.isfinite(t_min) or not np.isfinite(t_max) or t_min == t_max:
        return hv.Curve([], kdims=["time"], vdims=["rate"])

    edges = np.arange(t_min, t_max + bin_s, bin_s, dtype=float)
    counts, _edges = np.histogram(t, bins=edges)
    centers = (_edges[:-1] + _edges[1:]) / 2.0
    rate = counts.astype(float) / float(bin_s)

    return hv.Curve((centers, rate), kdims=["time"], vdims=["rate"]).opts(
        width=520,
        height=180,
        tools=["hover"],
        xlabel="Time (s)",
        ylabel="Events/s",
        title="Event rate",
    )


def _spatial_heatmap(df):
    import holoviews as hv

    hv.extension("bokeh")

    if df.empty or ("AP" not in df.columns) or ("ML" not in df.columns):
        return hv.Div("<b>No spatial columns (AP/ML)</b>")

    counts = df.groupby(["AP", "ML"]).size().reset_index(name="count")
    return hv.HeatMap(counts, kdims=["ML", "AP"], vdims=["count"]).opts(
        cmap="Reds",
        colorbar=True,
        width=260,
        height=260,
        tools=["hover"],
        title="Spatial count",
    )


def _freq_hist(df):
    import holoviews as hv

    hv.extension("bokeh")

    if df.empty or ("freq" not in df.columns):
        return hv.Div("<b>No frequency column (freq)</b>")

    f = np.asarray(df["freq"].to_numpy(), dtype=float)
    f = f[np.isfinite(f)]
    if f.size == 0:
        return hv.Div("<b>No finite freq values</b>")

    hist = np.histogram(f, bins=30)
    return hv.Histogram(hist, kdims=["freq"], vdims=["count"]).opts(
        width=260,
        height=180,
        tools=["hover"],
        xlabel="Frequency (Hz)",
        ylabel="Count",
        title="Frequency",
    )


def create_event_explorer_module(*, stream_name: str | None = None) -> ViewPresetModule:
    """
    Create an Event Explorer preset module.

    Parameters
    ----------
    stream_name
        Which registered event stream to explore. If None, uses the first
        registered stream at activation time.
    """

    def _activate(state):
        import holoviews as hv

        hv.extension("bokeh")

        name = str(stream_name) if stream_name is not None else _first_event_stream_name(state)
        if name is None:
            return hv.Div("<b>No events registered</b>")

        stream = state.event_registry.get(name) if getattr(state, "event_registry", None) is not None else None
        if stream is None:
            return hv.Div(f"<b>Event stream not found</b><br>stream_name={name!r}")

        overlay = EventOverlayLayer(state, name)

        # Core views (ViewFactory currently supports spatial + temporal only).
        spatial_spec = ViewSpec(kdims=["ML", "AP"], controls=["time"], colormap="viridis")
        temporal_spec = ViewSpec(kdims=["time"], controls=["AP", "ML"])

        spatial_view = ViewFactory.create(spatial_spec, state) * overlay.create_spatial_overlay()
        temporal_view = ViewFactory.create(temporal_spec, state) * overlay.create_temporal_overlay()

        # Stats
        df = stream.df
        rate = _event_rate_curve(df, time_col=stream.time_col)
        spatial_dist = _spatial_heatmap(df)
        freq_dist = _freq_hist(df)

        header = hv.Div(
            "<b>Event Explorer</b><br>"
            f"stream={name!r} (n={len(stream)})"
        )

        top = hv.Layout([header, rate, spatial_dist, freq_dist]).cols(2)
        bottom = hv.Layout([spatial_view, temporal_view]).cols(2)
        return (top + bottom).cols(1)

    return ViewPresetModule(
        name="event_explorer",
        description="Explore events with overlays and summary stats",
        specs=[],
        layout="custom",
        activate_fn=_activate,
    )


@dataclass(frozen=True, slots=True)
class EventExplorerModule:
    """
    Convenience wrapper that mirrors the `ViewPresetModule` API.

    This is primarily for explicit construction in tests/examples.
    """

    stream_name: str | None = None

    def as_preset(self) -> ViewPresetModule:
        return create_event_explorer_module(stream_name=self.stream_name)

    def activate(self, state):
        return self.as_preset().activate(state)


MODULE = create_event_explorer_module()

