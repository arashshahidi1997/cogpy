"""
ieeg_viewer.py
==============
A self-contained Panel/HoloViews multichannel iEEG viewer.

Usage
-----
    from ieeg_viewer import ieeg_viewer
    import panel as pn

    # sig: xr.DataArray with dims (channel, time) or (time, channel)
    # time coordinate must be in seconds (float).
    # channel coordinate can be any hashable values (int, str, tuple, …).

    pn.extension()

    viewer = ieeg_viewer(sig)
    viewer.servable()        # inside `panel serve`
    pn.serve(viewer)         # programmatic serve

    # In a notebook:
    pn.extension()
    ieeg_viewer(sig)

Example
-------
>>> from cogpy.core.plot.ieeg_viewer import ieeg_viewer
>>> from cogpy.datasets.tensor import example_smooth_multichannel_sigx
>>> ieeg_view = ieeg_viewer(sig, initial_window_s=5, n_channels_default=8)
>>> ieeg_view.servable()
"""

from __future__ import annotations

import numpy as np
import holoviews as hv
from holoviews import streams
from holoviews.plotting.links import RangeToolLink
import panel as pn
import xarray as xr
from tsdownsample import MinMaxLTTBDownsampler

__all__ = ["ieeg_viewer"]

# Module-level downsampler singleton (stateless for our use, safe to share)
_ds = MinMaxLTTBDownsampler()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _downsample(
    t: np.ndarray, y: np.ndarray, n_out: int
) -> tuple[np.ndarray, np.ndarray]:
    """MinMaxLTTB downsample; pass through if already small enough."""
    if len(t) <= n_out:
        return t, y
    idx = _ds.downsample(t, y, n_out=n_out)
    return t[idx], y[idx]


def _find_indices(t_vals: np.ndarray, t0: float, t1: float) -> tuple[int, int]:
    i0 = int(np.searchsorted(t_vals, t0, side="left"))
    i1 = int(np.searchsorted(t_vals, t1, side="right"))
    return max(i0, 0), min(i1, len(t_vals))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ieeg_viewer(
    data: xr.DataArray,
    *,
    # Dimension names
    time_dim: str = "time",
    channel_dim: str = "channel",
    # Window settings
    initial_window_s: float = 10.0,
    min_window_s: float = 0.5,
    max_window_s: float = 120.0,
    # Rendering budgets
    detail_px: int = 2000,
    overview_px: int = 10_000,
    # Channel display
    n_channels_default: int = 8,
    max_channels: int = 16,
    offset_scale: float = 3.0,
    # Layout
    width: int = 1100,
    detail_height: int = 500,
    overview_height: int = 120,
    title: str | None = None,
) -> pn.viewable.Viewable:
    """
    Build an interactive multichannel iEEG viewer.

    Parameters
    ----------
    data : xr.DataArray
        Two-dimensional array with one time dimension and one channel
        dimension.  Dimension order does not matter.  The time coordinate
        must be numeric (seconds).  Channel coordinate values are used as
        display labels and can be any type (int, str, tuple, …).
        Dask-backed arrays are materialised once on first call.
    time_dim : str
        Name of the time dimension. Default ``"time"``.
    channel_dim : str
        Name of the channel dimension. Default ``"channel"``.
    initial_window_s : float
        Width of the detail window shown on first render, in seconds.
    min_window_s : float
        Lower bound of the window-width slider.
    max_window_s : float
        Upper bound of the window-width slider.
    detail_px : int
        Target number of Bokeh-rendered points *per channel* in the detail
        pane.  MinMaxLTTB preserves visual extrema within this budget.
    overview_px : int
        Points in the static overview strip (mean across all channels,
        computed once at startup).
    n_channels_default : int
        Number of channels selected by default (the first N).
    max_channels : int
        Maximum channels the MultiChoice widget allows simultaneously.
    offset_scale : float
        Vertical spacing between channel traces in z-score units.
    width : int
        Plot width in pixels.
    detail_height : int
        Detail pane height in pixels.
    overview_height : int
        Overview strip height in pixels.
    title : str | None
        Header text.  If None, auto-generated from ``data.name``.

    Returns
    -------
    pn.viewable.Viewable
        A Panel Column.  Call ``.servable()`` or pass to ``pn.serve()``.

    Examples
    --------
    >>> viewer = ieeg_viewer(sig, initial_window_s=5, n_channels_default=16)
    >>> viewer.servable()
    """
    # ------------------------------------------------------------------
    # 1. Validate dimensions
    # ------------------------------------------------------------------
    for dim, name in [(time_dim, "time_dim"), (channel_dim, "channel_dim")]:
        if dim not in data.dims:
            raise ValueError(
                f"{name}={dim!r} not found in data.dims={tuple(data.dims)}"
            )

    # ------------------------------------------------------------------
    # 2. Canonical shape (channel, time) → numpy, z-scored per channel
    # ------------------------------------------------------------------
    arr    = data.transpose(channel_dim, time_dim)
    t_vals = np.asarray(arr[time_dim].values,    dtype=np.float64)
    ch_vals = list(arr[channel_dim].values)       # original coordinate values
    n_ch   = len(ch_vals)

    # Materialise (triggers Dask compute exactly once)
    raw: np.ndarray = np.asarray(arr.values, dtype=np.float64)  # (n_ch, n_time)

    # Z-score so offset spacing is meaningful regardless of physical units
    means = raw.mean(axis=1, keepdims=True)
    stds  = raw.std(axis=1,  keepdims=True) + 1e-12
    sig_z = (raw - means) / stds                  # (n_ch, n_time)

    t0_full = float(t_vals[0])
    t1_full = float(t_vals[-1])

    # ------------------------------------------------------------------
    # 3. Channel label index  (label string ↔ row index in sig_z)
    # ------------------------------------------------------------------
    ch_labels   = [str(v) for v in ch_vals]
    label_to_ix = {lbl: i for i, lbl in enumerate(ch_labels)}
    default_sel = ch_labels[:min(n_channels_default, n_ch)]

    # ------------------------------------------------------------------
    # 4. Overview strip  (static, computed once)
    # ------------------------------------------------------------------
    mean_sig = sig_z.mean(axis=0)
    t_ov, y_ov = _downsample(t_vals, mean_sig, overview_px)

    overview_curve = hv.Curve(
        (t_ov, y_ov), kdims=time_dim, vdims="amp"
    ).opts(
        width=width,
        height=overview_height,
        color="#4a90d9",
        line_width=0.8,
        xlabel="",
        ylabel="",
        toolbar=None,
        default_tools=[],
        title="Overview  —  drag the box to navigate",
        yaxis=None,
    )

    # ------------------------------------------------------------------
    # 5. Widgets
    # ------------------------------------------------------------------
    ch_select = pn.widgets.MultiChoice(
        name="Channels",
        value=default_sel,
        options=ch_labels,
        max_items=max_channels,
        sizing_mode="stretch_width",
    )

    window_slider = pn.widgets.FloatSlider(
        name="Window (s)",
        value=initial_window_s,
        start=min_window_s,
        end=max_window_s,
        step=0.5,
        sizing_mode="fixed",
        width=260,
    )

    # ------------------------------------------------------------------
    # 6. Detail pane  (DynamicMap + RangeX stream)
    # ------------------------------------------------------------------
    range_stream = streams.RangeX(
        x_range=(t0_full, t0_full + initial_window_s)
    )

    def _build_detail(x_range):
        # Unpack time window from stream
        if x_range is None or x_range == (None, None):
            t0, t1 = t0_full, t0_full + initial_window_s
        else:
            t0 = max(float(x_range[0]), t0_full)
            t1 = min(float(x_range[1]), t1_full)

        i0, i1    = _find_indices(t_vals, t0, t1)
        t_win     = t_vals[i0:i1]
        labels    = ch_select.value or default_sel
        ch_ixs    = [label_to_ix[lbl] for lbl in labels]
        n_visible = len(ch_ixs)

        curves = []
        for rank, ch in enumerate(ch_ixs):
            y_win      = sig_z[ch, i0:i1]
            t_ds, y_ds = _downsample(t_win, y_win, detail_px)
            offset     = (n_visible - 1 - rank) * offset_scale
            curves.append(
                hv.Curve(
                    (t_ds, y_ds + offset),
                    kdims=time_dim,
                    vdims="amp",
                    label=ch_labels[ch],
                ).opts(color=hv.Cycle("Category20"), line_width=1)
            )

        if not curves:
            return hv.Overlay([hv.Curve([], kdims=time_dim, vdims="amp")])

        # Channel name y-ticks aligned to trace offsets
        yticks = [
            ((n_visible - 1 - rank) * offset_scale, ch_labels[ch])
            for rank, ch in enumerate(ch_ixs)
        ]

        return hv.Overlay(curves).opts(
            hv.opts.Overlay(
                width=width,
                height=detail_height,
                show_legend=False,
                toolbar="above",
                xlabel=f"{time_dim} (s)",
                ylabel="",
                yticks=yticks,
                title="Detail  —  MinMaxLTTB",
                framewise=True,   # recompute axis ranges every render
            )
        )

    detail_dmap = hv.DynamicMap(_build_detail, streams=[range_stream])

    # Channel selection change → retrigger the existing x_range
    ch_select.param.watch(
        lambda *_: range_stream.trigger([range_stream]), "value"
    )

    # Overview box drag → fires RangeX → _build_detail
    RangeToolLink(
        overview_curve,
        detail_dmap,
        axes=["x", "x"],
        boundsx=(t0_full, t0_full + initial_window_s),
    )

    # Window slider → keep center, resize box
    def _on_window(event):
        lo, hi = range_stream.x_range or (t0_full, t0_full + initial_window_s)
        center = (lo + hi) / 2
        half   = event.new / 2
        range_stream.event(x_range=(center - half, center + half))

    window_slider.param.watch(_on_window, "value")

    # ------------------------------------------------------------------
    # 7. Layout
    # ------------------------------------------------------------------
    header = title or (
        f"iEEG viewer  —  {data.name or 'signal'}  "
        f"({n_ch} channels,  {t1_full - t0_full:.1f} s)"
    )

    return pn.Column(
        pn.pane.Markdown(f"## {header}", styles={"color": "#cdd6f4"}),
        pn.Row(
            ch_select,
            window_slider,
            styles={
                "background": "#1e1e2e",
                "padding": "8px",
                "border-radius": "6px",
            },
        ),
        detail_dmap,
        overview_curve,
        styles={"background": "#181825", "padding": "16px"},
    )