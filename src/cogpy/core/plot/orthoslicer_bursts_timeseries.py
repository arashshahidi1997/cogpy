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
from holoviews import streams

from .orthoslicer_bursts import OrthoSlicerRangerBursts
from .orthoslicer_rangercopy import _clip_pair
from .multichannel_timeseries import multichannel_timeseries_view


hv.extension("bokeh")
pn.extension()


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
        t_full = np.asarray(slab.t.values, dtype=float).reshape(-1)

        yx = self._selected_yx_indices()
        if len(yx) == 0:
            return hv.Text(0.5, 0.5, "No channels selected").opts(height=self.ts_height, width=self.xy_width)

        # Downsample once and apply consistently across channels.
        if self.max_trace_points is not None and int(self.max_trace_points) > 0 and t_full.size > int(self.max_trace_points):
            step = int(np.ceil(t_full.size / int(self.max_trace_points)))
        else:
            step = 1
        t = t_full[::step]

        # shape (time, channel)
        traces: list[np.ndarray] = []
        for (r, c) in yx:
            v = np.asarray(slab.isel(y=r, x=c).values, dtype=float).reshape(-1)[::step]
            traces.append(v)
        x_tch = np.stack(traces, axis=1) if len(traces) else np.zeros((t.size, 0), dtype=float)

        labels = self._selected_channel_labels(yx)

        # Build view using the shared multichannel helper.
        tb = self.param["t_window"].bounds
        boundsx = _clip_pair(*self.t_window, tb)
        parts = multichannel_timeseries_view(
            t,
            x_tch,
            title=f"Time series @ {self._fmt_val('z', self.z)}",
            channel_labels=labels,
            ylabel="Channel (stacked)",
            boundsx=boundsx,
            width=int(self.xy_width),
            overlay_height=int(self.ts_height),
            minimap_height=int(self.ts_minimap_height),
            responsive=True,
            return_parts=True,
        )
        layout = parts["layout"]
        minimap = parts["minimap"]
        self._ts_rtl = parts["rtlink"]  # keep alive

        def _on_ts_bounds(boundsx_):
            if boundsx_:
                self.t_window = _clip_pair(float(boundsx_[0]), float(boundsx_[1]), tb)

        # Keep t_window in sync with the RangeTool selection itself (source plot),
        # not the target plot x-range. This avoids RangeToolLink <-> RangeX loops.
        self._ts_bx = streams.BoundsX(source=minimap, boundsx=boundsx)
        self._ts_bx.add_subscriber(lambda **kw: _on_ts_bounds(kw.get("boundsx")))

        return layout

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
        Panel app laid out like :meth:`~cogpy.core.plot.orthoslicer_bursts.OrthoSlicerRangerBursts.panel_app`
        with an additional multichannel time-series panel.

        Layout
        ------
        - Optional collapsible left sidebar ("Controls")
        - Three main columns:
          1) TZ
          2) XY + burst navigator/table
          3) Multichannel time-series (selected subgrid)
        """
        tz_total_height = int(
            self.tz_height
            + self.ranger_height
            + (self.tz_trace_height if "burst_trace" in list(getattr(self, "tz_time_panels", [])) else 1)
            + (self.tz_rate_height if "burst_rate" in list(getattr(self, "tz_time_panels", [])) else 1)
            + 120
        )
        xy_total_height = int(self.xy_height + 80)

        tz_pane = pn.pane.HoloViews(
            self._tz_layout,
            width=int(self.tz_width),
            height=tz_total_height,
            sizing_mode="fixed",
        )
        xy_pane = pn.pane.HoloViews(
            self.view_xy,
            width=int(self.xy_width),
            height=xy_total_height,
            sizing_mode="fixed",
        )

        bursts_panel = pn.Column(
            pn.pane.Markdown("### Burst"),
            self.burst_nav(),
            self.bursts_table(),
            width=int(self.xy_width),
        )

        ts_panel = pn.Column(
            pn.pane.HoloViews(
                self.view_timeseries,
                width=int(self.xy_width),
                sizing_mode="fixed",
            ),
            width=int(self.xy_width),
        )

        tz_col = pn.Column(
            pn.pane.Markdown("### TZ"),
            tz_pane,
            width=int(self.tz_width),
            margin=0,
        )
        xy_col = pn.Column(
            pn.pane.Markdown("### XY"),
            xy_pane,
            pn.layout.Divider(),
            bursts_panel,
            width=int(self.xy_width),
            margin=0,
        )

        ts_controls = pn.Card(
            pn.Column(
                pn.pane.Markdown("Select a subgrid of channels (y,x)."),
                self.timeseries_controls(),
            ),
            title="Time Series",
            collapsible=True,
            collapsed=False,
            width=360,
        )

        controls_sidebar = pn.Column(
            pn.Card(
                self.controls_panel(),
                title="Slicer",
                collapsible=True,
                collapsed=False,
                width=360,
            ),
            ts_controls,
            width=380,
        )
        controls_sidebar.visible = False

        controls_toggle = pn.widgets.Toggle(name="Controls", value=False, button_type="primary", width=110)

        def _set_sidebar_visible(event):
            controls_sidebar.visible = bool(event.new)

        controls_toggle.param.watch(_set_sidebar_visible, "value")

        header = pn.Row(controls_toggle, pn.Spacer())

        main = pn.Row(
            tz_col,
            xy_col,
            pn.Column(
                pn.pane.Markdown("### Time Series (selected subgrid)"),
                ts_panel,
                width=int(self.xy_width),
                margin=0,
            ),
            margin=0,
            styles={"gap": "12px"},
        )

        return pn.Column(
            header,
            pn.Row(controls_sidebar, main, margin=0, styles={"gap": "12px"}),
            sizing_mode="stretch_both",
            margin=0,
        )


def main() -> None:
    """
    Debug entrypoint: run a small demo app.

    This intentionally mirrors the docstring example but lives in a callable so
    it can be used via ``python -m cogpy.core.plot.orthoslicer_bursts_timeseries``.
    """
    from cogpy.datasets.tensor import make_flat_blob_dataset, detect_bursts_hmaxima

    pn.extension()
    hv.extension("bokeh")

    da = make_flat_blob_dataset(duration=2.0, nt=80, n_peaks=5, seed=0)
    bursts = detect_bursts_hmaxima(da, h_quantile=0.99)

    dx = ("ml", hv.Dimension("x", label="Medial-Lateral", unit="mm"))
    dy = ("ap", hv.Dimension("y", label="Anterior-Posterior", unit="mm"))
    dt = ("time", hv.Dimension("t", label="Time", unit="s"))
    dz = ("freq", hv.Dimension("z", label="Frequency", unit="Hz"))

    slicer = OrthoSlicerRangerBurstsTimeseries(da, bursts=bursts, dt=dt, dz=dz, dy=dy, dx=dx)
    slicer.tz_logy = True

    # Debug: print tap events to the console.
    slicer.tap_xy.add_subscriber(lambda **kw: print("tap_xy", kw))
    slicer.tap_tz.add_subscriber(lambda **kw: print("tap_tz", kw))

    slicer.panel_app().show()


if __name__ == "__main__":
    main()
