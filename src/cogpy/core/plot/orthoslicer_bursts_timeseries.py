"""
OrthoSlicer with bursts + multichannel time-series (subgrid selection).

This module builds on :class:`cogpy.core.plot.orthoslicer_bursts.OrthoSlicerRangerBursts`
and adds a multichannel time-series panel that shows an interactively-selected
subset of channels from a 2D grid (e.g., 16×16 electrodes).

Design goals
------------
- Keep the orthoslicer (TZ + XY + bursts table) behavior unchanged.
- Add a fast time-series viewer:
  - stacked traces for a selected subgrid of channels
  - a minimap image (time × channel) with a RangeTool to pick ``t_window``

Channel model
-------------
Channels are treated as a 2D grid over the standardized spatial dims:
``y`` is the row axis and ``x`` is the column axis. A channel corresponds to
one (y, x) coordinate pair.

Example
-------
>>> import holoviews as hv
>>> import panel as pn
>>> from cogpy.datasets.tensor import make_flat_blob_dataset, detect_bursts_hmaxima
>>> from cogpy.core.plot.orthoslicer_bursts_timeseries import OrthoSlicerRangerBurstsTimeseries
>>>
>>> pn.extension()
>>> hv.extension("bokeh")
>>>
>>> da = make_flat_blob_dataset(duration=2.0, nt=80, n_peaks=5, seed=0)
>>> bursts = detect_bursts_hmaxima(da, h_quantile=0.99)
>>>
>>> dx = ("ml", hv.Dimension("x", label="Medial-Lateral", unit="mm"))
>>> dy = ("ap", hv.Dimension("y", label="Anterior-Posterior", unit="mm"))
>>> dt = ("time", hv.Dimension("t", label="Time", unit="s"))
>>> dz = ("freq", hv.Dimension("z", label="Frequency", unit="Hz"))
>>>
>>> slicer = OrthoSlicerRangerBurstsTimeseries(da, bursts=bursts, dt=dt, dz=dz, dy=dy, dx=dx)
>>> slicer.tz_logy = True
>>> slicer.panel_app().show()
"""

from __future__ import annotations

import numpy as np
import holoviews as hv
import panel as pn
import param
from holoviews.operation.datashader import rasterize
from holoviews.plotting.links import RangeToolLink
from holoviews import streams

from .orthoslicer_bursts import OrthoSlicerRangerBursts
from .orthoslicer_rangercopy import _clip_pair


hv.extension("bokeh")
pn.extension()


def _zscore(x: np.ndarray, axis: int = 0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x, axis=axis, keepdims=True)
    sd = np.nanstd(x, axis=axis, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    return (x - mu) / sd


def _downsample_xy(t: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if max_points is None or max_points <= 0:
        return t, y
    n = int(t.size)
    if n <= max_points:
        return t, y
    step = int(np.ceil(n / max_points))
    return t[::step], y[::step]


class OrthoSlicerRangerBurstsTimeseries(OrthoSlicerRangerBursts):
    """
    Extends :class:`~cogpy.core.plot.orthoslicer_bursts.OrthoSlicerRangerBursts`
    with a multichannel time-series panel.
    """

    grid_rows = param.Integer(default=16, bounds=(1, None), label="Grid rows")
    grid_cols = param.Integer(default=16, bounds=(1, None), label="Grid cols")

    block_rows = param.Integer(default=4, bounds=(1, None), label="Block rows")
    block_cols = param.Integer(default=4, bounds=(1, None), label="Block cols")
    stride_row = param.Integer(default=1, bounds=(1, None), label="Row stride")
    stride_col = param.Integer(default=1, bounds=(1, None), label="Col stride")
    row0 = param.Integer(default=0, bounds=(0, None), label="Row start")
    col0 = param.Integer(default=0, bounds=(0, None), label="Col start")

    max_trace_points = param.Integer(default=4000, bounds=(200, None), label="Max trace points")
    ts_height = param.Integer(default=360, bounds=(200, None), label="TS Height")
    ts_minimap_height = param.Integer(default=140, bounds=(80, None), label="TS Minimap Height")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Keep RangeToolLink objects alive
        self._ts_rtl = None
        self._ts_bx = None

        # Adapt grid defaults to the actual data if possible.
        if hasattr(self, "array") and "x" in self.array.dims and "y" in self.array.dims:
            self.grid_cols = int(self.array.sizes["x"])
            self.grid_rows = int(self.array.sizes["y"])

        # Ensure row0/col0 bounds align with grid size
        self.param["row0"].bounds = (0, max(0, self.grid_rows - 1))
        self.param["col0"].bounds = (0, max(0, self.grid_cols - 1))

    # ------------------------- channel selection -------------------------
    def _selected_yx_indices(self) -> list[tuple[int, int]]:
        nrows = int(min(self.grid_rows, self.array.sizes["y"]))
        ncols = int(min(self.grid_cols, self.array.sizes["x"]))

        out: list[tuple[int, int]] = []
        for i in range(int(self.block_rows)):
            r = int(self.row0 + i * self.stride_row)
            if r < 0 or r >= nrows:
                continue
            for j in range(int(self.block_cols)):
                c = int(self.col0 + j * self.stride_col)
                if c < 0 or c >= ncols:
                    continue
                out.append((r, c))
        return out

    def _selected_channel_labels(self, yx: list[tuple[int, int]]) -> list[str]:
        # row-major channel id for labeling
        return [f"r{r:02d}c{c:02d}" for r, c in yx]

    # ------------------------- timeseries view --------------------------
    @param.depends(
        "z",
        "t_window",
        "block_rows",
        "block_cols",
        "stride_row",
        "stride_col",
        "row0",
        "col0",
        "max_trace_points",
        "ts_height",
        "ts_minimap_height",
    )
    def view_timeseries(self):
        """
        Stacked time-series view for a selected subgrid at the current ``z``.
        """
        # Extract time series per channel at the current frequency slice.
        slab = self.array.sel(z=self.z, method="nearest")  # dims: (t, y, x)
        t = np.asarray(slab.t.values, dtype=float).reshape(-1)

        yx = self._selected_yx_indices()
        if len(yx) == 0:
            return hv.Text(0.5, 0.5, "No channels selected").opts(height=self.ts_height, width=self.tz_width)

        # shape (time, channel)
        traces = []
        for (r, c) in yx:
            v = np.asarray(slab.isel(y=r, x=c).values, dtype=float).reshape(-1)
            tt, vv = _downsample_xy(t, v, int(self.max_trace_points))
            traces.append(vv)
        x_tch = np.stack(traces, axis=1)

        labels = self._selected_channel_labels(yx)

        time_dim = hv.Dimension("Time (s)", unit="s")
        vdim = hv.Dimension("Value", unit="a.u.")
        ch_dim = hv.Dimension("Channel", unit="")

        curves = []
        for i, lab in enumerate(labels):
            ds = hv.Dataset((t, x_tch[:, i]), [time_dim, vdim])
            curve = hv.Curve(ds, time_dim, vdim, label=lab).opts(
                subcoordinate_y=True,
                color="black",
                line_width=1,
                line_alpha=0.7,
                tools=["hover"],
            )
            curves.append(curve)

        overlay = hv.Overlay(curves, "Channel").opts(
            xlabel=time_dim.pprint_label,
            ylabel="Channel (stacked)",
            show_legend=False,
            height=int(self.ts_height),
            width=int(self.tz_width),
            title=f"Time series @ {self._fmt_val('z', self.z)}",
        )

        # Minimap image (channel × time)
        zimg = _zscore(x_tch, axis=0).T  # (channel, time)
        y_positions = np.arange(zimg.shape[0], dtype=float)
        minimap = rasterize(
            hv.Image((t, y_positions, zimg), [time_dim, ch_dim], vdim)
        ).opts(
            cmap="RdBu_r",
            xlabel="",
            ylabel="",
            alpha=0.7,
            height=int(self.ts_minimap_height),
            width=int(self.tz_width),
            toolbar="disable",
            cnorm="eq_hist",
        )

        # Link minimap RangeTool to overlay x-range and keep references alive.
        tb = self.param["t_window"].bounds
        boundsx = _clip_pair(*self.t_window, tb)
        self._ts_rtl = RangeToolLink(minimap, overlay, axes=["x"], boundsx=boundsx)

        # Keep t_window in sync with the RangeTool selection itself (source plot),
        # not the target plot x-range. This avoids RangeToolLink <-> RangeX loops.
        self._ts_bx = streams.BoundsX(source=minimap, boundsx=boundsx)

        def _on_ts_bounds(boundsx_):
            if boundsx_:
                self.t_window = _clip_pair(float(boundsx_[0]), float(boundsx_[1]), tb)

        self._ts_bx.add_subscriber(lambda **kw: _on_ts_bounds(kw.get("boundsx")))

        return (overlay + minimap).cols(1)

    def timeseries_controls(self):
        return pn.Param(
            self,
            parameters=[
                "block_rows",
                "block_cols",
                "stride_row",
                "stride_col",
                "row0",
                "col0",
                "max_trace_points",
            ],
            show_name=False,
            width=int(self.xy_width),
        )

    def panel_app(self):
        """
        GUI-like Panel app with tabs and collapsible side panels.

        Tabs:
        - Slicer: TZ + XY views
        - Time Series: stacked traces + minimap RangeTool
        - Bursts: navigator + table

        Side panels:
        - Controls (collapsible)
        - Subgrid selection (collapsible)
        """
        # --- main panes ---
        tz_pane = pn.pane.HoloViews(
            self.view_tz,
            sizing_mode="stretch_width",
            height=int(self.tz_height + self.ranger_height + 80),
        )
        xy_pane = pn.pane.HoloViews(
            self.view_xy,
            sizing_mode="stretch_width",
            height=int(self.xy_height + 80),
        )
        slicer_tab = pn.Row(
            pn.Column(pn.pane.Markdown("### TZ"), tz_pane, sizing_mode="stretch_both"),
            pn.Column(pn.pane.Markdown("### XY"), xy_pane, sizing_mode="stretch_both"),
            sizing_mode="stretch_both",
            styles={"gap": "12px"},
        )

        ts_tab = pn.Column(
            pn.pane.Markdown("### Time Series (selected subgrid)"),
            pn.pane.HoloViews(self.view_timeseries, sizing_mode="stretch_width"),
            sizing_mode="stretch_both",
        )

        bursts_tab = pn.Column(
            pn.pane.Markdown("### Burst Navigator"),
            self.burst_nav(),
            pn.layout.Divider(),
            pn.pane.Markdown("### Bursts Table"),
            self.bursts_table(),
            sizing_mode="stretch_both",
        )

        tabs = pn.Tabs(
            ("Slicer", slicer_tab),
            ("Time Series", ts_tab),
            ("Bursts", bursts_tab),
            dynamic=True,
            sizing_mode="stretch_both",
        )

        # --- sidebar (collapsible cards) ---
        controls_card = pn.Card(
            super().controls_panel(),
            title="Controls",
            collapsible=True,
            collapsed=False,
            sizing_mode="stretch_width",
        )
        subgrid_card = pn.Card(
            pn.Column(
                pn.pane.Markdown("Select a subgrid of channels (y,x)."),
                self.timeseries_controls(),
                sizing_mode="stretch_width",
            ),
            title="Subgrid",
            collapsible=True,
            collapsed=False,
            sizing_mode="stretch_width",
        )
        bursts_card = pn.Card(
            pn.Column(
                pn.pane.Markdown("Burst controls (optional)"),
                pn.Param(self, parameters=["follow_burst"], show_name=False),
                sizing_mode="stretch_width",
            ),
            title="Bursts",
            collapsible=True,
            collapsed=True,
            sizing_mode="stretch_width",
        )

        sidebar = pn.Column(
            controls_card,
            subgrid_card,
            bursts_card,
            sizing_mode="fixed",
            width=380,
        )

        sidebar_toggle = pn.widgets.Toggle(
            name="Controls",
            value=True,
            button_type="primary",
            width=110,
        )
        sidebar_toggle.link(sidebar, value="visible")

        header = pn.Row(
            sidebar_toggle,
            pn.Spacer(),
            sizing_mode="stretch_width",
        )

        return pn.Column(
            header,
            pn.Row(sidebar, tabs, sizing_mode="stretch_both", styles={"gap": "12px"}),
            sizing_mode="stretch_both",
        )
