"""Multichannel time-series viewer helpers (HoloViews/Panel).

This module provides a small "viewer-like" plot for (time, channel) signals.

Primary output (when available)
------------------------------
- HoloViews Overlay of stacked traces using ``subcoordinate_y=True``
- Linked minimap (z-scored image) via ``RangeToolLink``

Fallback
--------
- Matplotlib stacked traces

Notes
-----
This file was promoted from ``code/utils/plot/multichannel_timeseries.py`` so it
can be reused by CogPy plotting components (e.g. orthoslicers) without relying
on project-level ``utils`` imports. The original location keeps a small
compatibility shim.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def multichannel_timeseries_view(
    t_s: np.ndarray,
    x_tch: np.ndarray,
    *,
    title: str = "",
    channel_labels: list[str] | None = None,
    y_positions: np.ndarray | None = None,
    ylabel: str = "Channel (stacked)",
    boundsx: tuple[float, float] | None = None,
    boundsx_s: float = 5.0,
    width: int | None = None,
    overlay_height: int | None = None,
    minimap_height: int = 140,
    responsive: bool = True,
    return_parts: bool = False,
) -> Any:
    """
    Parameters
    ----------
    t_s
        Time vector in seconds, shape ``(time,)``.
    x_tch
        Signal array, shape ``(time, channel)``.
    title
        Title for the main plot.
    channel_labels
        Per-channel labels for hover and trace naming. Defaults to ``ch{i}``.
    y_positions
        Per-channel y coordinates for the minimap image (e.g., depth in mm).
        If omitted, uses ``0..nch-1``.
    ylabel
        Label used for the minimap y-dimension and matplotlib fallback axis.
    boundsx_s
        Initial visible x-window width (seconds) for the RangeTool selection.
    boundsx
        Initial x-range bounds ``(t0, t1)`` for the RangeTool selection.
        If provided, overrides ``boundsx_s``.
    return_parts
        If True, returns a dict containing ``layout``, ``overlay``, ``minimap``,
        and ``rtlink`` (RangeToolLink). Use this when you need to keep the link
        alive or attach additional streams.

    Returns
    -------
    Any
        HoloViews layout (preferred) or Matplotlib Figure (fallback).
    """
    t = np.asarray(t_s, dtype=float).reshape(-1)
    x = np.asarray(x_tch, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"Expected x_tch as (time, channel), got shape {x.shape}.")
    if t.size != x.shape[0]:
        raise ValueError(
            f"t_s length {t.size} does not match x_tch time dim {x.shape[0]}."
        )

    nch = int(x.shape[1])
    if channel_labels is None:
        channel_labels = [f"ch{i}" for i in range(nch)]
    if len(channel_labels) != nch:
        raise ValueError(f"Expected channel_labels length {nch}, got {len(channel_labels)}.")

    if y_positions is None:
        y_positions = np.arange(nch, dtype=float)
    else:
        y_positions = np.asarray(y_positions, dtype=float).reshape(-1)
        if y_positions.size != nch:
            raise ValueError(f"Expected y_positions length {nch}, got {y_positions.size}.")

    # Prefer interactive view; fall back cleanly if optional deps are missing.
    try:
        import pandas as pd
        import holoviews as hv
        from holoviews.operation.datashader import rasterize
        from holoviews.operation.normalization import subcoordinate_group_ranges
        from holoviews.plotting.links import RangeToolLink
        from scipy.stats import zscore

        hv.extension("bokeh")

        df = pd.DataFrame(x, index=t, columns=channel_labels)
        df.index.name = "Time (s)"

        hover_tooltips = [
            ("Channel", "$label"),
            ("Time", "@{Time (s)}"),
            ("Value", "@Value"),
        ]

        time_dim = hv.Dimension("Time (s)", unit="s")
        vdim = hv.Dimension("Value", unit="a.u.")
        y_dim = hv.Dimension(ylabel)

        curves = []
        for col in df.columns:
            ds = hv.Dataset((df.index.values, df[col].values), [time_dim, vdim])
            curve = hv.Curve(ds, time_dim, vdim, group="Signal", label=str(col)).opts(
                subcoordinate_y=True,
                color="black",
                line_width=1,
                hover_tooltips=hover_tooltips,
                line_alpha=0.8,
            )
            curves.append(curve)

        overlay = hv.Overlay(curves, "Channel").opts(
            xlabel=time_dim.pprint_label,
            ylabel=ylabel,
            show_legend=False,
            aspect=3,
            responsive=bool(responsive),
            title=title,
        )
        overlay = subcoordinate_group_ranges(overlay)
        if width is not None:
            overlay = overlay.opts(width=int(width))
        if overlay_height is not None:
            overlay = overlay.opts(height=int(overlay_height))

        z = zscore(x, axis=0, nan_policy="omit").T  # (channel, time)
        minimap = rasterize(hv.Image((t, y_positions, z), [time_dim, y_dim], vdim)).opts(
            cmap="RdBu_r",
            xlabel="",
            ylabel=ylabel,
            alpha=0.7,
            height=int(minimap_height),
            responsive=bool(responsive),
            toolbar="disable",
            cnorm="eq_hist",
        )
        if width is not None:
            minimap = minimap.opts(width=int(width))

        if boundsx is None:
            boundsx = (float(t[0]), float(min(t[0] + boundsx_s, t[-1])))
        else:
            boundsx = (float(boundsx[0]), float(boundsx[1]))

        rtlink = RangeToolLink(
            minimap,
            overlay,
            axes=["x"],
            boundsx=boundsx,
        )
        layout = (overlay + minimap).cols(1)
        if return_parts:
            return {"layout": layout, "overlay": overlay, "minimap": minimap, "rtlink": rtlink}
        return layout

    except Exception as e:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))
        scale = np.nanstd(x)
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        for i in range(nch):
            ax.plot(t, x[:, i] / scale + i, lw=0.5, color="black")
        ax.set_title(f"{title} (fallback: {type(e).__name__})")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        return fig
