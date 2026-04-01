"""Multichannel time-series viewer helpers (HoloViews/Panel).

This module provides a small "viewer-like" plot for (time, channel) signals.

Primary output (when available)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

import numpy as np
from cogpy.utils.imports import import_optional

hv = import_optional("holoviews")
pn = import_optional("panel")
param = import_optional("param")


hv.extension("bokeh")
pn.extension()

from typing import Any


def subgrid_indices(
    n_rows: int,
    n_cols: int,
    *,
    row0: int = 0,
    col0: int = 0,
    block_rows: int = 4,
    block_cols: int = 4,
    stride_row: int = 1,
    stride_col: int = 1,
) -> list[tuple[int, int]]:
    """
    Return a list of ``(row, col)`` indices for a strided subgrid selection.
    """
    n_rows = int(n_rows)
    n_cols = int(n_cols)
    out: list[tuple[int, int]] = []
    for i in range(int(block_rows)):
        r = int(row0) + i * int(stride_row)
        if r < 0 or r >= n_rows:
            continue
        for j in range(int(block_cols)):
            c = int(col0) + j * int(stride_col)
            if c < 0 or c >= n_cols:
                continue
            out.append((r, c))
    return out


def rc_to_flat(r: int, c: int, n_cols: int) -> int:
    return int(r) * int(n_cols) + int(c)


def flat_to_rc(idx: int, n_cols: int) -> tuple[int, int]:
    idx = int(idx)
    n_cols = int(n_cols)
    return (idx // n_cols, idx % n_cols)


def select_channels(x_tch: np.ndarray, selected: list[int] | np.ndarray) -> np.ndarray:
    """
    Slice ``x_tch`` by channel indices (keeps time dimension).
    """
    x = np.asarray(x_tch)
    sel = np.asarray(selected, dtype=int).reshape(-1)
    if x.ndim != 2:
        raise ValueError(f"Expected x_tch as (time, channel), got shape {x.shape}.")
    if sel.size == 0:
        return x[:, :0]
    return x[:, sel]


def multichannel_timeseries_view(
    t_s: np.ndarray,
    x_tch: np.ndarray,
    *,
    title: str = "",
    channel_labels: list[str] | None = None,
    channel_colors: list[str] | None = None,
    y_positions: np.ndarray | None = None,
    ylabel: str = "Channel (stacked)",
    boundsx: tuple[float, float] | None = None,
    boundsx_s: float = 5.0,
    width: int | None = None,
    overlay_height: int | None = None,
    minimap_height: int = 140,
    minimap_max_points: int | None = None,
    minimap_cnorm: str = "eq_hist",
    responsive: bool = True,
    return_parts: bool = False,
    max_channels_plot: int | None = None,
    downsample: bool = False,
    render: str = "auto",
    raster_dynamic: bool = False,
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
    channel_colors
        Optional per-channel colors (e.g. hex strings). If omitted, traces are black.
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
    minimap_max_points
        If provided, decimates the time axis used in the minimap to at most this
        many points. This reduces memory/latency for long recordings.
    minimap_cnorm
        Normalization mode for the minimap colormap (passed to Bokeh via HoloViews).
        Defaults to ``"eq_hist"``.
    max_channels_plot
        If provided, caps the number of plotted channels to preserve performance.
    downsample
        If True, applies ``holoviews.operation.downsample.downsample1d`` to each
        trace curve (helps for long time vectors).
    render
        Rendering mode for the main stacked trace view:
        - ``"vector"``: Bokeh line glyphs (hover supported; slower for large data)
        - ``"raster"``: Datashader rasterization of a single multi-line Path (fast; no per-trace hover)
        - ``"auto"``: chooses ``"raster"`` when ``downsample=True`` or when the
          number of points is large, otherwise uses ``"vector"``.
    raster_dynamic
        When ``render="raster"`` (or when rasterizing the minimap), controls whether
        datashader re-renders dynamically on pan/zoom. ``False`` is more stable for
        notebook/Panel rendering and for ``RangeToolLink`` reliability.
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

    if max_channels_plot is not None and nch > int(max_channels_plot):
        import warnings

        cap = int(max_channels_plot)
        warnings.warn(
            f"Capping plotted channels at {cap} (out of {nch}) to preserve performance."
        )
        nch = cap
        x = x[:, :nch]
        if channel_labels is not None:
            channel_labels = list(channel_labels[:nch])
        if channel_colors is not None:
            channel_colors = list(channel_colors[:nch])
        if y_positions is not None:
            y_positions = np.asarray(y_positions)[:nch]

    if channel_labels is None:
        channel_labels = [f"ch{i}" for i in range(nch)]
    if len(channel_labels) != nch:
        raise ValueError(
            f"Expected channel_labels length {nch}, got {len(channel_labels)}."
        )
    if channel_colors is not None and len(channel_colors) != nch:
        raise ValueError(
            f"Expected channel_colors length {nch}, got {len(channel_colors)}."
        )

    if y_positions is None:
        y_positions = np.arange(nch, dtype=float)
    else:
        y_positions = np.asarray(y_positions, dtype=float).reshape(-1)
        if y_positions.size != nch:
            raise ValueError(
                f"Expected y_positions length {nch}, got {y_positions.size}."
            )

    # Prefer interactive view; fall back cleanly if optional deps are missing.
    try:
        import holoviews as hv
        from holoviews.operation.datashader import rasterize
        from holoviews.operation.downsample import downsample1d
        from holoviews.operation.normalization import subcoordinate_group_ranges
        from holoviews.plotting.links import RangeToolLink

        hv.extension("bokeh")

        time_dim = hv.Dimension("Time", unit="s")
        vdim = hv.Dimension("Value", unit="a.u.")
        y_dim = hv.Dimension(ylabel)

        render_mode = str(render).lower()
        if render_mode not in {"auto", "vector", "raster"}:
            raise ValueError(
                f"render must be one of 'auto', 'vector', 'raster' (got {render!r})."
            )
        if render_mode == "auto":
            # Heuristic: downsample=True means "fast mode", prefer datashader.
            # Also rasterize very large point clouds to keep notebooks responsive.
            n_points = int(t.size) * int(nch)
            render_mode = (
                "raster" if bool(downsample) or n_points >= 2_000_000 else "vector"
            )

        def _zscore_nan(x_tc: np.ndarray) -> np.ndarray:
            mu = np.nanmean(x_tc, axis=0, keepdims=True)
            sigma = np.nanstd(x_tc, axis=0, keepdims=True)
            sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1.0)
            return (x_tc - mu) / sigma

        def _stack_for_raster(x_tc: np.ndarray) -> np.ndarray:
            # Stack channels as z-scored traces offset by channel index.
            z_tc = _zscore_nan(x_tc)
            return z_tc + np.arange(z_tc.shape[1], dtype=float)[None, :]

        if render_mode == "raster":
            y_tc = _stack_for_raster(x)
            paths = [np.column_stack([t, y_tc[:, i]]) for i in range(nch)]
            overlay: Any = rasterize(
                hv.Path(paths, kdims=[time_dim, y_dim]),
                dynamic=bool(raster_dynamic),
            ).opts(
                xlabel=time_dim.pprint_label,
                ylabel=ylabel,
                show_legend=False,
                aspect=3,
                responsive=bool(responsive),
                title=title,
            )
            if width is not None:
                overlay = overlay.opts(width=int(width))
            if overlay_height is not None:
                overlay = overlay.opts(height=int(overlay_height))
        else:
            curves = []
            for i in range(nch):
                curve = hv.Curve(
                    (t, x[:, i]),
                    kdims=[time_dim],
                    vdims=[vdim],
                    label=str(channel_labels[i]),
                )

                opts_curve: dict[str, Any] = dict(
                    subcoordinate_y=True,
                    color=(
                        "black" if channel_colors is None else str(channel_colors[i])
                    ),
                    line_width=1,
                    line_alpha=0.8,
                )
                if nch <= 16:
                    opts_curve["hover_tooltips"] = [
                        ("Channel", "$label"),
                        ("Time", "@Time"),
                        ("Value", "@Value"),
                    ]
                else:
                    opts_curve["tools"] = []

                curve = curve.opts(**opts_curve)
                if downsample:
                    curve = downsample1d(curve)
                curves.append(curve)

            # If we downsample each curve, they become DynamicMaps. Collate to avoid
            # nesting DynamicMaps within an Overlay (HoloViews recommends this).
            overlay = hv.Overlay(curves, "Channel")
            if any(isinstance(c, hv.DynamicMap) for c in curves):
                overlay = overlay.collate()

            overlay = overlay.opts(
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

        if minimap_max_points is not None:
            step = max(1, int(t.size) // int(minimap_max_points))
        else:
            step = 1
        t_mini = t[::step]
        x_mini = x[::step, :]
        z = _zscore_nan(x_mini).T  # (channel, time)

        minimap = rasterize(
            hv.Image((t_mini, y_positions, z), [time_dim, y_dim], vdim),
            dynamic=bool(raster_dynamic),
        ).opts(
            cmap="RdBu_r",
            xlabel="",
            ylabel=ylabel,
            alpha=0.7,
            height=int(minimap_height),
            responsive=bool(responsive),
            toolbar="disable",
            cnorm=str(minimap_cnorm),
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
            return {
                "layout": layout,
                "overlay": overlay,
                "minimap": minimap,
                "rtlink": rtlink,
            }
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


def multichannel_timeseries_lite(
    t_s: np.ndarray,
    x_tch: np.ndarray,
    *,
    title: str = "",
    channel_labels: list[str] | None = None,
    channel_colors: list[str] | None = None,
    y_positions: np.ndarray | None = None,
    ylabel: str = "Channel",
    boundsx: tuple[float, float] | None = None,
    boundsx_s: float = 5.0,
    width: int | None = None,
    overlay_height: int | None = None,
    minimap_height: int = 140,
    responsive: bool = True,
    return_parts: bool = False,
    max_channels_plot: int | None = 32,
    downsample: bool = True,
    minimap_max_points: int = 2000,
    minimap_cnorm: str = "eq_hist",
    as_pane: bool = True,
    render: str = "auto",
    raster_dynamic: bool = False,
) -> Any:
    """
    Lightweight multichannel time-series view (HoloViews-only).

    This is a convenience wrapper around :func:`multichannel_timeseries_view` with
    defaults tuned for notebook interactivity:
    - caps the number of plotted channels (``max_channels_plot``)
    - applies per-trace downsampling (``downsample=True``)
    - decimates the minimap time axis (``minimap_max_points``)

    Returns the same object types as :func:`multichannel_timeseries_view`.
    """
    out = multichannel_timeseries_view(
        t_s,
        x_tch,
        title=title,
        channel_labels=channel_labels,
        channel_colors=channel_colors,
        y_positions=y_positions,
        ylabel=ylabel,
        boundsx=boundsx,
        boundsx_s=boundsx_s,
        width=width,
        overlay_height=overlay_height,
        minimap_height=minimap_height,
        minimap_max_points=int(minimap_max_points),
        minimap_cnorm=minimap_cnorm,
        responsive=responsive,
        return_parts=return_parts,
        max_channels_plot=max_channels_plot,
        downsample=downsample,
        render=render,
        raster_dynamic=raster_dynamic,
    )

    if return_parts or not as_pane:
        return out

    # VSCode/Jupyter rendering can be brittle for HoloViews ``Layout`` objects.
    # Prefer returning a Panel Column of separate panes to avoid LayoutPlot issues.
    try:
        import panel as pn

        pn.extension()

        parts = multichannel_timeseries_view(
            t_s,
            x_tch,
            title=title,
            channel_labels=channel_labels,
            channel_colors=channel_colors,
            y_positions=y_positions,
            ylabel=ylabel,
            boundsx=boundsx,
            boundsx_s=boundsx_s,
            width=width,
            overlay_height=overlay_height,
            minimap_height=minimap_height,
            minimap_max_points=int(minimap_max_points),
            minimap_cnorm=minimap_cnorm,
            responsive=responsive,
            return_parts=True,
            max_channels_plot=max_channels_plot,
            downsample=downsample,
            render=render,
            raster_dynamic=raster_dynamic,
        )

        if not isinstance(parts, dict):
            return parts

        col = pn.Column(
            pn.pane.HoloViews(parts["overlay"], sizing_mode="stretch_width"),
            pn.pane.HoloViews(parts["minimap"], sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
        )
        # Keep the RangeToolLink alive (it is referenced via Python-side object graph).
        col._rtlink_holder = parts.get("rtlink")  # type: ignore[attr-defined]
        return col

    except Exception:
        # Fall back to returning the raw HoloViews object.
        return out
