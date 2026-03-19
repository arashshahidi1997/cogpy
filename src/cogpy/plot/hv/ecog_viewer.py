"""ECoG viewer and channel grid selector (HoloViews/Panel).

Extracted from ``multichannel_timeseries.py`` to separate the interactive
ECoG grid-selector application from the core timeseries rendering utilities.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr
import holoviews as hv
import panel as pn
from holoviews import opts, streams
import param

from .multichannel_timeseries import (
    subgrid_indices,
    rc_to_flat,
    flat_to_rc,
    select_channels,
    multichannel_timeseries_view,
)
from .xarray_hv import to_time_channel, normalize_coords_to_index

hv.extension("bokeh")
pn.extension()

class ChannelGridSelectorState(param.Parameterized):
    n_rows = param.Integer(default=1, bounds=(1, None), label="Grid rows")
    n_cols = param.Integer(default=1, bounds=(1, None), label="Grid cols")

    selected = param.List(default=[], item_type=int, label="Selected channels")
    metric = param.Array(default=None, allow_None=True, label="Metric")

    selection_mode = param.ObjectSelector(
        default="toggle",
        objects=["replace", "add", "toggle", "remove"],
        label="Selection mode",
    )

    row0 = param.Integer(default=0, bounds=(0, None), label="Row start")
    col0 = param.Integer(default=0, bounds=(0, None), label="Col start")
    block_rows = param.Integer(default=4, bounds=(1, None), label="Block rows")
    block_cols = param.Integer(default=4, bounds=(1, None), label="Block cols")
    stride_row = param.Integer(
        default=1,
        bounds=(1, None),
        label="Row period",
        doc="Period for row selection (1 selects every row, 2 selects every other row, etc.).",
    )
    stride_col = param.Integer(
        default=1,
        bounds=(1, None),
        label="Col period",
        doc="Period for col selection (1 selects every col, 2 selects every other col, etc.).",
    )

    sync_selected_to_subgrid = param.Boolean(
        default=False,
        label="Sync to subgrid",
        doc="If True, updating the subgrid replaces the selected channels.",
    )


class ChannelGridSelector:
    """
    Interactive channel grid selector for ECoG-style (row, col) layouts.

    This helper provides:
    - a grid "map" you can tap to toggle channels on/off
    - a rectangle (box) selection that can fill a subgrid selection
    - param state you can wire into other views (e.g. multichannel_timeseries_view)

    Notes
    -----
    - Stream sources are bound once to a stable *image DynamicMap* (not an Overlay).
    - The visual selection overlay is separate and passive, so it won't intercept taps.
    """

    def __init__(
        self,
        *,
        n_rows: int,
        n_cols: int,
        selected: list[int] | None = None,
        metric: np.ndarray | None = None,
        selection_mode: str = "toggle",
        width: int = 280,
        height: int = 280,
        invert_yaxis: bool = True,
        show_colorbar: bool = False,
        cmap: str = "gray",
    ):
        import holoviews as hv
        from holoviews import streams

        hv.extension("bokeh")

        self.state = ChannelGridSelectorState(
            n_rows=int(n_rows),
            n_cols=int(n_cols),
            selected=list(selected or []),
            selection_mode=str(selection_mode),
        )
        if metric is not None:
            self.state.metric = np.asarray(metric)

        self._hv = hv
        self._streams = streams
        self._width = int(width)
        self._height = int(height)
        self._invert_yaxis = bool(invert_yaxis)
        self._show_colorbar = bool(show_colorbar)
        self._cmap = str(cmap)

        # Stable image DM: only metric changes affect the rendered cells.
        self._metric_params = streams.Params(self.state, ["metric"])
        self._img_dm = hv.DynamicMap(self._grid_image, streams=[self._metric_params])

        # Streams bound ONCE to the stable image DM.
        self.tap = streams.Tap(source=self._img_dm)
        self.box = streams.BoundsXY(source=self._img_dm)

        self.tap.add_subscriber(self._on_tap)
        self.box.add_subscriber(self._on_box)

        # Passive overlays (selection + subgrid rectangle)
        sel_params = streams.Params(
            self.state,
            [
                "selected",
                "row0",
                "col0",
                "block_rows",
                "block_cols",
                "stride_row",
                "stride_col",
            ],
        )
        self._sel_dm = hv.DynamicMap(self._selection_overlay, streams=[sel_params])
        self._subgrid_dm = hv.DynamicMap(self._subgrid_overlay, streams=[sel_params])

        self.view = (self._img_dm * self._subgrid_dm * self._sel_dm).opts(
            width=self._width,
            height=self._height,
            invert_yaxis=self._invert_yaxis,
        )

    # ------------------------ public helpers ------------------------
    def selected_rc(self) -> list[tuple[int, int]]:
        return [flat_to_rc(i, int(self.state.n_cols)) for i in list(self.state.selected)]

    def set_selected_from_subgrid(self) -> None:
        yx = subgrid_indices(
            int(self.state.n_rows),
            int(self.state.n_cols),
            row0=int(self.state.row0),
            col0=int(self.state.col0),
            block_rows=int(self.state.block_rows),
            block_cols=int(self.state.block_cols),
            stride_row=int(self.state.stride_row),
            stride_col=int(self.state.stride_col),
        )
        self.state.selected = [rc_to_flat(r, c, int(self.state.n_cols)) for (r, c) in yx]

    def panel_controls(self):
        import panel as pn

        return pn.Param(
            self.state,
            parameters=[
                "selection_mode",
                "sync_selected_to_subgrid",
                "row0",
                "col0",
                "block_rows",
                "block_cols",
                "stride_row",
                "stride_col",
            ],
            show_name=False,
            width=320,
        )

    def clear(self) -> None:
        self.state.selected = []

    def set_metric(self, metric: np.ndarray | None) -> None:
        self.state.metric = None if metric is None else np.asarray(metric)

    # ------------------------ rendering ------------------------
    def _grid_image(self, metric=None, **_):
        hv = self._hv
        n_rows = int(self.state.n_rows)
        n_cols = int(self.state.n_cols)
        xs = np.arange(n_cols, dtype=float)
        ys = np.arange(n_rows, dtype=float)

        if metric is None:
            z = np.zeros((n_rows, n_cols), dtype=float)
        else:
            z = np.asarray(metric, dtype=float)
            if z.shape != (n_rows, n_cols):
                raise ValueError(f"metric must have shape {(n_rows, n_cols)}, got {z.shape}")

        img = hv.Image((xs, ys, z), kdims=["col", "row"], vdims=["metric"]).opts(
            cmap=self._cmap,
            colorbar=self._show_colorbar,
            toolbar="above",
            tools=["tap", "box_select", "reset"],
            active_tools=["tap"],
            xlabel="col",
            ylabel="row",
            title="Channels",
        )
        return img

    def _selection_overlay(self, selected=None, **_):
        hv = self._hv
        n_cols = int(self.state.n_cols)
        selected = list(selected or [])
        if len(selected) == 0:
            return hv.Rectangles([])

        rects = []
        for idx in selected:
            r, c = flat_to_rc(int(idx), n_cols)
            rects.append((c - 0.5, r - 0.5, c + 0.5, r + 0.5))

        return hv.Rectangles(rects, kdims=["x0", "y0", "x1", "y1"]).opts(
            fill_color="#808080",
            fill_alpha=0.35,
            line_alpha=0.0,
            tools=[],
        )

    def _subgrid_overlay(self, **_):
        hv = self._hv
        n_rows = int(self.state.n_rows)
        n_cols = int(self.state.n_cols)
        yx = subgrid_indices(
            n_rows,
            n_cols,
            row0=int(self.state.row0),
            col0=int(self.state.col0),
            block_rows=int(self.state.block_rows),
            block_cols=int(self.state.block_cols),
            stride_row=int(self.state.stride_row),
            stride_col=int(self.state.stride_col),
        )
        if len(yx) == 0:
            return hv.Rectangles([])
        rows = [r for r, _c in yx]
        cols = [c for _r, c in yx]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        # rectangle in data coords: left,bottom,right,top
        rect = hv.Rectangles([(c0 - 0.5, r0 - 0.5, c1 + 0.5, r1 + 0.5)], kdims=["x0", "y0", "x1", "y1"])
        return rect.opts(
            fill_alpha=0.0,
            line_color="#ffb000",
            line_width=2,
            tools=[],
        )

    def apply_periodic_template(
        self,
        *,
        period_row: int = 2,
        period_col: int = 2,
        row0: int = 0,
        col0: int = 0,
        sync_selected: bool | None = None,
    ) -> None:
        """
        Convenience helper: select a periodic subgrid pattern across the full grid.

        Example: ``period_row=2, period_col=2`` selects every other electrode.
        """
        pr = max(1, int(period_row))
        pc = max(1, int(period_col))
        r0 = int(np.clip(int(row0), 0, int(self.state.n_rows) - 1))
        c0 = int(np.clip(int(col0), 0, int(self.state.n_cols) - 1))

        self.state.row0 = r0
        self.state.col0 = c0
        self.state.stride_row = pr
        self.state.stride_col = pc
        self.state.block_rows = int(np.ceil((int(self.state.n_rows) - r0) / pr))
        self.state.block_cols = int(np.ceil((int(self.state.n_cols) - c0) / pc))

        if sync_selected is None:
            sync_selected = bool(self.state.sync_selected_to_subgrid)
        if bool(sync_selected):
            self.set_selected_from_subgrid()

    # ------------------------ interactions ------------------------
    def _snap_cell(self, x: float | None, y: float | None) -> tuple[int, int] | None:
        if x is None or y is None:
            return None
        n_rows = int(self.state.n_rows)
        n_cols = int(self.state.n_cols)
        c = int(np.clip(np.floor(float(x) + 0.5), 0, n_cols - 1))
        r = int(np.clip(np.floor(float(y) + 0.5), 0, n_rows - 1))
        return (r, c)

    def _apply_selection_mode(self, idx: int) -> None:
        mode = str(self.state.selection_mode)
        cur = list(map(int, self.state.selected))
        s = set(cur)
        if mode == "replace":
            s = {idx}
        elif mode == "add":
            s.add(idx)
        elif mode == "remove":
            s.discard(idx)
        else:  # toggle
            if idx in s:
                s.remove(idx)
            else:
                s.add(idx)
        self.state.selected = sorted(s)

    def _on_tap(self, x=None, y=None, **_):
        cell = self._snap_cell(x, y)
        if cell is None:
            return
        r, c = cell
        idx = rc_to_flat(r, c, int(self.state.n_cols))
        self._apply_selection_mode(idx)

    def _on_box(self, bounds=None, **_):
        if not bounds:
            return
        x0, y0, x1, y1 = bounds
        n_rows = int(self.state.n_rows)
        n_cols = int(self.state.n_cols)
        c0 = int(np.clip(np.floor(min(x0, x1) + 0.5), 0, n_cols - 1))
        c1 = int(np.clip(np.floor(max(x0, x1) + 0.5), 0, n_cols - 1))
        r0 = int(np.clip(np.floor(min(y0, y1) + 0.5), 0, n_rows - 1))
        r1 = int(np.clip(np.floor(max(y0, y1) + 0.5), 0, n_rows - 1))

        self.state.row0 = r0
        self.state.col0 = c0
        self.state.block_rows = max(1, r1 - r0 + 1)
        self.state.block_cols = max(1, c1 - c0 + 1)

        if bool(self.state.sync_selected_to_subgrid):
            self.set_selected_from_subgrid()


def sparkline_grid_view(
    t_s: np.ndarray,
    x_tyx: np.ndarray,
    *,
    t_window: tuple[float, float] | None = None,
    normalize: str = "zscore",
    clip: float = 3.0,
    amplitude: float = 0.42,
    padding: float = 0.08,
    width: int = 520,
    height: int = 520,
    rasterize: bool = True,
    line_width: float = 1.2,
    invert_yaxis: bool = True,
    title: str = "Trace snippets (grid)",
):
    """
    Render per-channel time snippets as sparklines arranged on a (row,col) grid.

    Parameters
    ----------
    t_s
        Time vector, shape ``(time,)``.
    x_tyx
        Signal array, shape ``(time, rows, cols)``.
    t_window
        Optional time bounds to display.
    rasterize
        If True, renders with datashader (much faster for large grids).
    """
    import holoviews as hv

    hv.extension("bokeh")

    t = np.asarray(t_s, dtype=float).reshape(-1)
    x = np.asarray(x_tyx, dtype=float)
    if x.ndim != 3:
        raise ValueError(f"Expected x_tyx as (time, rows, cols), got shape {x.shape}.")
    if x.shape[0] != t.size:
        raise ValueError(f"t_s length {t.size} does not match x_tyx time dim {x.shape[0]}.")

    n_rows, n_cols = int(x.shape[1]), int(x.shape[2])

    if t_window is not None:
        t0, t1 = map(float, t_window)
        m = (t >= min(t0, t1)) & (t <= max(t0, t1))
        if bool(m.any()):
            t = t[m]
            x = x[m, :, :]

    # Time normalized to [0,1] for within-cell placement
    if t.size < 2:
        tt = np.zeros_like(t)
    else:
        tt = (t - float(t[0])) / float(t[-1] - t[0])

    pad = float(np.clip(padding, 0.0, 0.45))
    amp = float(amplitude) * (1.0 - 2.0 * pad)

    # Normalize per-channel
    if normalize == "zscore":
        mu = np.nanmean(x, axis=0, keepdims=True)
        sd = np.nanstd(x, axis=0, keepdims=True)
        sd = np.where(sd == 0, 1.0, sd)
        xn = (x - mu) / sd
    elif normalize == "center":
        xn = x - np.nanmean(x, axis=0, keepdims=True)
    else:
        xn = x
    if clip is not None:
        xn = np.clip(xn, -float(clip), float(clip))
        if float(clip) > 0:
            xn = xn / float(clip)

    # Build Path segments, one per channel, offset into the grid.
    paths = []
    for r in range(n_rows):
        for c in range(n_cols):
            yy = (float(r) + 0.5) + amp * xn[:, r, c]
            xx = float(c) + pad + (1.0 - 2.0 * pad) * tt
            paths.append(np.column_stack([xx, yy]))

    el = hv.Path(paths, kdims=["col", "row"]).opts(
        color="black",
        line_width=float(line_width),
        alpha=0.85,
        width=int(width),
        height=int(height),
        title=str(title),
        show_grid=False,
        xlabel="col",
        ylabel="row",
        invert_yaxis=bool(invert_yaxis),
        tools=["tap", "pan", "wheel_zoom", "reset"],
    )

    if not bool(rasterize):
        return el

    from holoviews.operation.datashader import rasterize as _rasterize

    return _rasterize(el, width=int(width), height=int(height), line_width=float(line_width)).opts(
        cmap=["#000000"],
        colorbar=False,
        toolbar="above",
    )


def channel_grid_selector_demo(
    *,
    n_rows: int = 16,
    n_cols: int = 16,
    metric: np.ndarray | None = None,
):
    """
    Small Panel demo for :class:`ChannelGridSelector`.
    """
    import panel as pn
    import holoviews as hv

    pn.extension()
    hv.extension("bokeh")

    sel = ChannelGridSelector(n_rows=n_rows, n_cols=n_cols, metric=metric)

    selected_md = pn.pane.Markdown()

    def _render_selected(_event=None):
        rc = sel.selected_rc()
        selected_md.object = (
            f"**Selected:** {len(rc)} channels\n\n"
            + (", ".join([f"(r={r}, c={c})" for (r, c) in rc[:24]]) + (" …" if len(rc) > 24 else ""))
        )

    sel.state.param.watch(_render_selected, "selected")
    _render_selected()

    clear_btn = pn.widgets.Button(name="Clear", button_type="primary", width=90)
    clear_btn.on_click(lambda _e: sel.clear())

    return pn.Row(
        pn.Column(sel.view, pn.Row(clear_btn), selected_md),
        pn.Spacer(width=10),
        pn.Column(sel.panel_controls()),
    )


# ecog_viewer_lite.py
#
# Usage (in notebook):
#   from ecog_viewer_lite import ecog_viewer
#   view = ecog_viewer(sig_tyx)
#   view
#
# Usage (in app):
#   import panel as pn
#   from ecog_viewer_lite import ecog_viewer
#   servable = ecog_viewer(sig_tyx).servable()
#
# Notes:
# - No template, no pn.config global sizing defaults, minimal styling.
# - Responsive by default, safe in notebooks.

def ecog_viewer(
    sig_tyx: xr.DataArray,
    *,
    nsamples: int = 10_000,
    time_dim: str = "time",
    ap_dim: str = "AP",
    ml_dim: str = "ML",
    left_height: int = 420,
    trace_height: int = 180,
    right_height: int = 620,
    toolbar: str | None = "right",
    tools: list[str] | None = None,
    cmap_tc: str = "viridis",
    cmap_apml: str = "RdBu_r",
    symmetric_apml: bool = True,
    colorbar_tc: bool = True,
    colorbar_apml: bool = True,
) -> pn.viewable.Viewable:
    """
    Lightweight interactive ECoG viewer.

    Parameters
    ----------
    sig_tyx
        DataArray with dims like (time, AP, ML) or compatible.
    nsamples
        Number of time samples to include (slices from start).
    time_dim, ap_dim, ml_dim
        Dimension names (defaults match your schema).
    left_height, trace_height, right_height
        Height anchors for responsive plots.
    toolbar
        Bokeh toolbar position or None to hide.
    tools
        List of tools to enable (defaults to tap+nav).
    """

    if tools is None:
        tools = ["tap", "pan", "wheel_zoom", "box_zoom", "reset"]

    # --------------------------
    # Prep / normalize coords
    # --------------------------
    if time_dim not in sig_tyx.dims:
        raise ValueError(f"sig_tyx missing time_dim='{time_dim}'. dims={sig_tyx.dims}")
    if ap_dim not in sig_tyx.dims or ml_dim not in sig_tyx.dims:
        raise ValueError(f"sig_tyx missing ap_dim/ml_dim '{ap_dim}','{ml_dim}'. dims={sig_tyx.dims}")

    sub_sig = sig_tyx.isel({time_dim: slice(0, nsamples)})
    sub_sig = normalize_coords_to_index(sub_sig).compute()

    # time×ch representation with AP/ML coords per channel
    sub_sig_tc = to_time_channel(sub_sig)      # dims: (time, ch) with ch MultiIndex
    sub_sig_tc = sub_sig_tc.reset_index("ch")  # adds coords AP(ch), ML(ch)
    sub_sig_tc = sub_sig_tc.assign_coords(ch=np.arange(sub_sig_tc.sizes["ch"]))

    times = np.asarray(sub_sig_tc[time_dim].values, dtype=float)
    chs   = np.asarray(sub_sig_tc["ch"].values, dtype=int)

    ap_ch = np.asarray(sub_sig_tc[ap_dim].values, dtype=float)
    ml_ch = np.asarray(sub_sig_tc[ml_dim].values, dtype=float)

    def snap_time(t):
        t = float(t)
        return float(times[np.argmin(np.abs(times - t))])

    def snap_ch(c):
        c = int(round(float(c)))
        return int(chs[np.argmin(np.abs(chs - c))])

    def snap_ch_from_apml(ap, ml):
        ap = float(ap); ml = float(ml)
        d2 = (ap_ch - ap) ** 2 + (ml_ch - ml) ** 2
        return int(np.argmin(d2))

    def apml_for_ch(ch):
        ch = snap_ch(ch)
        return float(ap_ch[ch]), float(ml_ch[ch])

    # --------------------------
    # Shared state (streams)
    # --------------------------
    Time = streams.Stream.define("Time", time=float(times[0]))
    Chan = streams.Stream.define("Chan", ch=int(chs[0]))
    time_ctrl = Time()
    ch_ctrl   = Chan()

    # --------------------------
    # Overlays: crosshairs
    # --------------------------
    tline = hv.DynamicMap(
        lambda time: hv.VLine(time).opts(color="red", line_width=2, alpha=0.85),
        streams=[time_ctrl],
    )

    chline_tc = hv.DynamicMap(
        lambda ch: hv.HLine(snap_ch(ch)).opts(color="orange", line_width=2, alpha=0.85),
        streams=[ch_ctrl],
    )

    apml_cross = hv.DynamicMap(
        lambda ch: (hv.HLine(apml_for_ch(ch)[0]) * hv.VLine(apml_for_ch(ch)[1])).opts(
            opts.HLine(color="orange", line_width=2, alpha=0.85),
            opts.VLine(color="orange", line_width=2, alpha=0.85),
        ),
        streams=[ch_ctrl],
    )

    # --------------------------
    # Left: time×ch heatmap + trace
    # --------------------------
    ds = hv.Dataset(sub_sig_tc, kdims=[time_dim, "ch"], vdims=[sub_sig_tc.name or "val"])
    img = ds.to(hv.Image, kdims=[time_dim, "ch"]).opts(
        cmap=cmap_tc,
        colorbar=bool(colorbar_tc),
        tools=tools,
        active_tools=["tap"],
        toolbar=toolbar,
        responsive=True,
        height=int(left_height),
        xaxis=None,
        show_grid=False,
    )

    def curve_for_ch(ch):
        ch = snap_ch(ch)
        da = sub_sig_tc.sel(ch=ch)
        ap = float(da[ap_dim].values) if ap_dim in da.coords else np.nan
        ml = float(da[ml_dim].values) if ml_dim in da.coords else np.nan
        title = f"ch={ch} ({ap_dim}={ap:.2f}, {ml_dim}={ml:.2f})" if np.isfinite(ap) and np.isfinite(ml) else f"ch={ch}"
        return hv.Curve(da, kdims=[time_dim]).opts(title=title)

    curve = hv.DynamicMap(curve_for_ch, streams=[ch_ctrl]).opts(
        tools=tools,
        active_tools=["tap"],
        toolbar=toolbar,
        responsive=True,
        height=int(trace_height),
        show_grid=False,
    )

    # Tap handling
    tap_img   = streams.Tap(source=img)
    tap_curve = streams.Tap(source=curve)

    def set_from_img_tap(x=None, y=None, **_):
        if x is not None:
            time_ctrl.event(time=snap_time(x))
        if y is not None:
            ch_ctrl.event(ch=snap_ch(y))

    def set_time_from_curve_tap(x=None, **_):
        if x is None:
            return
        time_ctrl.event(time=snap_time(x))

    tap_img.add_subscriber(set_from_img_tap)
    tap_curve.add_subscriber(set_time_from_curve_tap)

    left = ((img * tline * chline_tc) + (curve * tline)).cols(1).opts(shared_axes=True)

    # --------------------------
    # Right: AP×ML frame at selected time
    # --------------------------
    # Normalize indices for AP/ML so hv.Image gets clean axes
    sub_sig_apml = normalize_coords_to_index(sub_sig, (ap_dim, ml_dim))

    def frame(time):
        t = snap_time(time)
        fr = sub_sig_apml.sel({time_dim: t}, method="nearest")
        return hv.Image(fr, kdims=[ml_dim, ap_dim]).opts(
            cmap=cmap_apml,
            symmetric=bool(symmetric_apml),
            colorbar=bool(colorbar_apml),
            tools=tools,
            active_tools=["tap"],
            toolbar=toolbar,
            responsive=True,
            height=int(right_height),
            aspect="equal",
            show_grid=False,
            title=f"{sub_sig_apml.name or 'signal'} @ t={t:.3f}",
        )

    gmovie = hv.DynamicMap(frame, streams=[time_ctrl])
    right = gmovie * apml_cross

    # Tap on AP×ML selects nearest channel (keeps time unchanged)
    tap_grid = streams.Tap(source=gmovie)

    def set_ch_from_grid_tap(x=None, y=None, **_):
        if x is None or y is None:
            return
        # kdims are (ML, AP): x=ML, y=AP
        ch_ctrl.event(ch=snap_ch_from_apml(ap=y, ml=x))

    tap_grid.add_subscriber(set_ch_from_grid_tap)

    # --------------------------
    # Small readout + layout
    # --------------------------
    readout = pn.pane.Markdown("", sizing_mode="stretch_width", margin=(0, 0, 10, 0))

    def update_readout(time, ch):
        ap, ml = apml_for_ch(ch)
        readout.object = (
            f"**Selection**  \n"
            f"- time: `{float(time):.4f}`  \n"
            f"- ch: `{int(ch)}`  \n"
            f"- {ap_dim}: `{ap:.2f}`  \n"
            f"- {ml_dim}: `{ml:.2f}`"
        )

    time_ctrl.add_subscriber(lambda time, **k: update_readout(time, ch_ctrl.contents["ch"]))
    ch_ctrl.add_subscriber(lambda ch, **k: update_readout(time_ctrl.contents["time"], ch))
    update_readout(time_ctrl.contents["time"], ch_ctrl.contents["ch"])

    left_card = pn.Card(
        pn.pane.HoloViews(left, sizing_mode="stretch_both"),
        title="Time × Channel",
        sizing_mode="stretch_both",
        collapsible=False,
    )
    right_card = pn.Card(
        readout,
        pn.pane.HoloViews(right, sizing_mode="stretch_both"),
        title="AP × ML",
        sizing_mode="stretch_both",
        collapsible=False,
    )

    # Responsive two-column layout (no templates, notebook-safe)
    layout = pn.Row(
        pn.Column(left_card, sizing_mode="stretch_both"),
        pn.Column(right_card, sizing_mode="stretch_both"),
        sizing_mode="stretch_both",
    )

    return layout


def lfp_overlay_curves(
    t_s: np.ndarray,
    lfp_tdepth: np.ndarray,
    depth_mm_grid: np.ndarray,
    *,
    stride: int = 4,
    wiggle_fraction_of_dz: float = 0.30,
):
    t_s = np.asarray(t_s, dtype=float).reshape(-1)
    d = np.asarray(depth_mm_grid, dtype=float).reshape(-1)
    if lfp_tdepth.shape != (t_s.size, d.size):
        raise ValueError(f"Expected lfp_tdepth {(t_s.size, d.size)}, got {lfp_tdepth.shape}")

    dz = float(np.nanmedian(np.diff(d))) if d.size > 2 else 0.02
    amp = float(np.nanpercentile(np.abs(lfp_tdepth), 95))
    scale = amp / (wiggle_fraction_of_dz * dz) if (amp > 0 and dz > 0) else 1.0

    time_dim = hv.Dimension("Time (s)", unit="s")
    dv_dim = hv.Dimension("DV (mm)", unit="mm")
    curves = []
    for i in range(0, lfp_tdepth.shape[1], int(stride)):
        y = d[i] + (lfp_tdepth[:, i] / scale)
        curves.append(hv.Curve((t_s, y), kdims=[time_dim], vdims=[dv_dim]).opts(color="black", line_width=0.6, alpha=0.8))
    return hv.Overlay(curves).opts(show_legend=False)


def main() -> None:
    channel_grid_selector_demo().show()


if __name__ == "__main__":
    main()

def ecog_viewer_hv(
    sig,
    *,
    time_dim: str = "time",
    ap_dim: str = "AP",
    ml_dim: str = "ML",
    n_rows: int | None = None,
    n_cols: int | None = None,
    metric: np.ndarray | None = None,
    max_channels_plot: int = 64,
    label_apml: bool = True,
    color_by_ap: bool = True,
    ap_cmap: str = "viridis",
    selector_width: int = 320,
    selector_height: int = 320,
    selector_kwargs: dict[str, Any] | None = None,
    timeseries_kwargs: dict[str, Any] | None = None,
    include_sparkline_grid: bool = False,
    sparkline_kwargs: dict[str, Any] | None = None,
    return_selector: bool = False,
):
    """
    Lightweight **HoloViews-only** ECoG viewer composed from utilities in this module.

    Improvements vs previous version
    -------------------------------
    - No "dead" blank axes panel: explicit placeholder text with axes hidden.
    - Never falls back to matplotlib: if interactive timeseries view fails, uses a
      HoloViews-native stacked-trace fallback (still interactive, consistent UI).
    - Optional RangeToolLink when available (kept alive via holder dict).
    """
    import holoviews as hv
    from holoviews import streams

    hv.extension("bokeh")

    # --------------------------
    # Input normalization
    # --------------------------
    try:
        import xarray as xr  # type: ignore
    except Exception:  # pragma: no cover
        xr = None

    if xr is not None and isinstance(sig, xr.DataArray):
        if "fs" not in sig.attrs:
            raise ValueError("xarray input requires `sig.attrs['fs']` (sampling rate).")
        fs = float(sig.attrs["fs"])

        if time_dim not in sig.dims or ap_dim not in sig.dims or ml_dim not in sig.dims:
            raise ValueError(
                f"xarray input must have dims ({time_dim!r}, {ap_dim!r}, {ml_dim!r}); got {list(sig.dims)}"
            )

        sig_tyx = sig.transpose(time_dim, ap_dim, ml_dim)
        x_arr = np.asarray(sig_tyx.values, dtype=float)
        n_time = int(sig_tyx.sizes[time_dim])
        ap_vals = np.asarray(sig_tyx[ap_dim].values)
        ml_vals = np.asarray(sig_tyx[ml_dim].values)

        tcoord = sig_tyx[time_dim].values
        try:
            t = np.asarray(tcoord, dtype=float).reshape(-1)
            if t.size != n_time:
                raise ValueError
        except Exception:
            t = (np.arange(n_time, dtype=float) / fs).reshape(-1)
    else:
        x_arr = np.asarray(sig, dtype=float)
        t = np.arange(int(x_arr.shape[0]), dtype=float)
        ap_vals = None
        ml_vals = None

    if x_arr.ndim not in (2, 3):
        raise ValueError(f"Expected x as (time,ch) or (time,rows,cols), got shape {x_arr.shape}.")
    if x_arr.shape[0] != t.size:
        raise ValueError(f"t_s length {t.size} does not match x time dim {x_arr.shape[0]}.")

    if x_arr.ndim == 3:
        rows, cols = int(x_arr.shape[1]), int(x_arr.shape[2])
        x_tch = x_arr.reshape(t.size, rows * cols)
        if metric is None:
            metric = np.sqrt(np.nanmean(x_arr**2, axis=0))
    else:
        if n_rows is None or n_cols is None:
            raise ValueError("n_rows and n_cols are required when x is provided as (time, ch).")
        rows, cols = int(n_rows), int(n_cols)
        if x_arr.shape[1] != rows * cols:
            raise ValueError(f"x has {x_arr.shape[1]} channels but grid is {rows}×{cols}={rows*cols}.")
        x_tch = x_arr

    selector_kwargs = {} if selector_kwargs is None else dict(selector_kwargs)
    timeseries_kwargs = {} if timeseries_kwargs is None else dict(timeseries_kwargs)
    sparkline_kwargs = {} if sparkline_kwargs is None else dict(sparkline_kwargs)

    # --------------------------
    # Selector
    # --------------------------
    selector = ChannelGridSelector(
        n_rows=rows,
        n_cols=cols,
        metric=metric,
        width=int(selector_width),
        height=int(selector_height),
        **selector_kwargs,
    )
    selector.state.sync_selected_to_subgrid = True

    selected_stream = streams.Params(selector.state, ["selected"])

    # Keep RangeToolLink alive (avoid GC killing link)
    _rtlink_holder: dict[str, Any] = {"rtlink": None}

    # --------------------------
    # Helpers: placeholders + HV-native fallback traces
    # --------------------------
    def _placeholder(msg: str, *, width: int = 900, height: int = 360):
        return hv.Text(0.5, 0.5, msg).opts(
            width=int(width),
            height=int(height),
            xaxis=None,
            yaxis=None,
            bgcolor="white",
            text_align="center",
            text_baseline="middle",
            text_font_size="12pt",
            show_frame=True,
        )

    def _hv_timeseries_fallback(
        t_s: np.ndarray,
        x_sel: np.ndarray,
        labels: list[str],
        *,
        title: str,
        width: int = 900,
        height: int = 360,
        minimap_height: int = 130,
        boundsx_s: float = 5.0,
    ):
        """
        HoloViews-only fallback:
        - stacked curves with simple offset
        - minimap image with optional RangeToolLink if available
        """
        t_s = np.asarray(t_s, dtype=float).reshape(-1)
        x_sel = np.asarray(x_sel, dtype=float)
        nch = int(x_sel.shape[1])

        # robust scale
        scale = np.nanstd(x_sel)
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0

        # stacked traces (offset by index)
        curves = []
        for i in range(nch):
            y = x_sel[:, i] / scale + i
            curves.append(
                hv.Curve((t_s, y), kdims=["Time (s)"], vdims=["Value"]).opts(
                    color="black",
                    line_width=1,
                    alpha=0.85,
                ).relabel(labels[i])
            )

        overlay = hv.Overlay(curves).opts(
            width=int(width),
            height=int(height),
            show_legend=False,
            xlabel="Time (s)",
            ylabel="Channel (stacked)",
            title=title,
        )

        # minimap (z-score per channel, shown as image: y=channel index)
        mu = np.nanmean(x_sel, axis=0, keepdims=True)
        sd = np.nanstd(x_sel, axis=0, keepdims=True)
        sd = np.where(sd == 0, 1.0, sd)
        z = ((x_sel - mu) / sd).T  # (ch, time)

        minimap = hv.Image((t_s, np.arange(nch), z), kdims=["Time (s)", "ch"], vdims=["z"]).opts(
            width=int(width),
            height=int(minimap_height),
            cmap="RdBu_r",
            colorbar=False,
            xlabel="",
            ylabel="",
            toolbar="disable",
            alpha=0.8,
        )

        # Try to link with RangeToolLink if available
        try:
            from holoviews.plotting.links import RangeToolLink  # type: ignore

            t0 = float(t_s[0])
            t1 = float(min(t_s[-1], t0 + float(boundsx_s)))
            _rtlink_holder["rtlink"] = RangeToolLink(minimap, overlay, axes=["x"], boundsx=(t0, t1))
        except Exception:
            _rtlink_holder["rtlink"] = None

        return (overlay + minimap).cols(1)

    # --------------------------
    # Main selected view DynamicMap
    # --------------------------
    def _selected_view(selected=None, **_):
        sel = list(selected or [])

        # No selection: explicit placeholder (no blank axes)
        if len(sel) == 0:
            return _placeholder("Tap cells in the grid to select channels.\n(Box-select sets subgrid.)")

        # Too many: protect performance
        if len(sel) > int(max_channels_plot):
            return _placeholder(f"Too many channels selected ({len(sel)}). Reduce selection ≤ {max_channels_plot}.")

        # Slice selected channels
        x_sel = select_channels(x_tch, sel)
        yx = [flat_to_rc(i, cols) for i in sel]
        if bool(label_apml) and ap_vals is not None and ml_vals is not None:
            def _fmt(v):
                try:
                    fv = float(v)
                    if abs(fv - round(fv)) < 1e-9:
                        return str(int(round(fv)))
                    return f"{fv:.3g}"
                except Exception:
                    return str(v)

            labels = [f"{ap_dim}={_fmt(ap_vals[int(r)])}, {ml_dim}={_fmt(ml_vals[int(c)])}" for (r, c) in yx]
            ap_for_sel = np.asarray([ap_vals[int(r)] for (r, _c) in yx], dtype=float)
        else:
            labels = [f"r{int(r):02d}c{int(c):02d}" for (r, c) in yx]
            ap_for_sel = None

        channel_colors = None
        if bool(color_by_ap) and ap_for_sel is not None:
            try:
                import matplotlib.cm as cm  # type: ignore
                import matplotlib.colors as mcolors  # type: ignore

                vmin = float(np.nanmin(ap_for_sel))
                vmax = float(np.nanmax(ap_for_sel))
                if np.isfinite(vmin) and np.isfinite(vmax) and vmin != vmax:
                    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                    cmap = cm.get_cmap(str(ap_cmap))
                    channel_colors = [mcolors.to_hex(cmap(norm(float(v)))) for v in ap_for_sel]
            except Exception:
                channel_colors = None

        # First attempt: your richer view (RangeTool + datashader + subcoordinate_y)
        # BUT: if it errors or returns a matplotlib Figure, we switch to HV fallback.
        try:
            parts = multichannel_timeseries_view(
                t,
                x_sel,
                title=f"Selected traces ({len(labels)} ch)",
                channel_labels=labels,
                channel_colors=channel_colors,
                return_parts=True,
                **timeseries_kwargs,
            )

            # Expected path: dict with hv layout + rtlink
            if isinstance(parts, dict) and "layout" in parts:
                _rtlink_holder["rtlink"] = parts.get("rtlink")
                return parts["layout"]

            # If it didn't return dict, don't accept matplotlib fallback
            return _hv_timeseries_fallback(
                t, x_sel, labels,
                title=f"Selected traces ({len(labels)} ch) (HV fallback: unexpected return)",
            )

        except Exception as exc:
            # HV-native fallback keeps the UI coherent
            return _hv_timeseries_fallback(
                t, x_sel, labels,
                title=f"Selected traces ({len(labels)} ch) (HV fallback: {type(exc).__name__})",
            )

    traces_dm = hv.DynamicMap(_selected_view, streams=[selected_stream])

    # --------------------------
    # Optional sparkline grid
    # --------------------------
    if not bool(include_sparkline_grid):
        layout = (selector.view + traces_dm).cols(2)
        return (layout, selector) if bool(return_selector) else layout

    def _spark(_selected=None, **_):
        if x_arr.ndim != 3:
            return _placeholder("Sparkline grid requires grid signals (time, rows, cols).", width=520, height=520)
        return sparkline_grid_view(t, x_arr, width=520, height=520, rasterize=True, **sparkline_kwargs)

    spark_dm = hv.DynamicMap(_spark, streams=[selected_stream])

    layout = hv.Layout([selector.view, traces_dm, spark_dm]).cols(2)
    return (layout, selector) if bool(return_selector) else layout
