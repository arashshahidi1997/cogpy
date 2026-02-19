"""
OrthoSlicerRanger with burst navigation + overlays.

This module adds a small UI layer on top of
:class:`cogpy.core.plot.orthoslicer_rangercopy.OrthoSlicerRanger`:

- A burst table (``pandas.DataFrame``) rendered in Panel.
- A burst navigator (prev/next + a ``burst_id`` parameter).
- When a burst is selected, the slicer jumps to that burst's coordinates and
  overlays a marker on both:
  - XY view at (x, y) for the burst
  - TZ view at (t, z) for the burst

Expected bursts table schema (minimum):
``burst_id, x, y, t, z, value``.

Example
-------
>>> import holoviews as hv
>>> import panel as pn
>>> from cogpy.datasets.tensor import make_dataset, detect_bursts_hmaxima
>>> from cogpy.core.plot.orthoslicer_bursts import OrthoSlicerRangerBursts
>>>
>>> pn.extension()
>>> hv.extension("bokeh")
>>>
>>> da = make_dataset(duration=2.0, nt=80, seed=0)
>>> bursts = detect_bursts_hmaxima(da, h_quantile=0.9)
>>> dx = ("ml", hv.Dimension("x", label="Medial-Lateral", unit="mm"))
>>> dy = ("ap", hv.Dimension("y", label="Anterior-Posterior", unit="mm"))
>>> dt = ("time", hv.Dimension("t", label="Time", unit="s"))
>>> dz = ("freq", hv.Dimension("z", label="Frequency", unit="Hz"))
>>>
>>> slicer = OrthoSlicerRangerBursts(
...     da,
...     bursts=bursts,
...     dt=dt,
...     dz=dz,
...     dy=dy,
...     dx=dx,
... )
>>> slicer.tz_logy = True  # log-scale frequency axis (plotting only)
>>> slicer.panel_app().show()

Example (Flat Background + Random Blobs)
---------------------------------------
>>> import holoviews as hv
>>> import panel as pn
>>> from cogpy.datasets.tensor import make_flat_blob_dataset, detect_bursts_hmaxima
>>> from cogpy.core.plot.orthoslicer_bursts import OrthoSlicerRangerBursts
>>>
>>> pn.extension()
>>> hv.extension("bokeh")
>>>
>>> da = make_flat_blob_dataset(
...     duration=2.0,
...     nt=80,
...     n_peaks=5,
...     peak_dist="uniform",
...     blob_sigma_idx=(0.8, 0.8, 2.0, 0.8),
...     blob_amp=10.0,
...     seed=0,
... )
>>> bursts = detect_bursts_hmaxima(da, h_quantile=0.99)
>>>
>>> dx = ("ml", hv.Dimension("x", label="Medial-Lateral", unit="mm"))
>>> dy = ("ap", hv.Dimension("y", label="Anterior-Posterior", unit="mm"))
>>> dt = ("time", hv.Dimension("t", label="Time", unit="s"))
>>> dz = ("freq", hv.Dimension("z", label="Frequency", unit="Hz"))
>>>
>>> slicer = OrthoSlicerRangerBursts(da, bursts=bursts, dt=dt, dz=dz, dy=dy, dx=dx)
>>> slicer.tz_logy = True
>>> slicer.panel_app().show()
"""

from __future__ import annotations

import pandas as pd
import holoviews as hv
import panel as pn
import param

from .orthoslicer_rangercopy import OrthoSlicerRanger, _clip_pair


class OrthoSlicerRangerBursts(OrthoSlicerRanger):
    burst_id = param.Integer(default=None, allow_None=True, label="Burst id")
    # Make TZ (time-freq) a bit wider by default for easier tapping/reading.
    tz_width = param.Integer(default=620, bounds=(200, None), label="TZ Width")
    tz_height = param.Integer(default=320, bounds=(200, None), label="TZ Height")
    xy_width = param.Integer(default=520, bounds=(200, None), label="XY Width")
    xy_height = param.Integer(default=320, bounds=(200, None), label="XY Height")
    ranger_height = param.Integer(default=120, bounds=(60, None), label="Ranger Height")

    follow_burst = param.Boolean(
        default=False,
        label="Follow burst",
        doc="If True, selecting a burst also jumps the slicer crosshair to it.",
    )

    burst_x = param.Number(allow_None=True, default=None)
    burst_y = param.Number(allow_None=True, default=None)
    burst_t = param.Number(allow_None=True, default=None)
    burst_z = param.Number(allow_None=True, default=None)

    def __init__(self, *args, bursts: pd.DataFrame | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.bursts = bursts if bursts is not None else pd.DataFrame()

        if len(self.bursts) > 0:
            if "burst_id" not in self.bursts.columns:
                self.bursts = self.bursts.copy()
                self.bursts["burst_id"] = range(len(self.bursts))
            self.bursts = self.bursts.sort_values("burst_id").reset_index(drop=True)
            self.param["burst_id"].bounds = (0, int(self.bursts["burst_id"].max()))
            self.burst_id = int(self.bursts["burst_id"].iloc[0])
            self._set_burst_marker(self.burst_id)
            if self.follow_burst:
                self._jump_to_burst(self.burst_id)

        self.param.watch(self._on_burst_id, "burst_id")

    # --------------------------- burst helpers ---------------------------
    def _burst_row(self, burst_id: int) -> pd.Series:
        if len(self.bursts) == 0:
            raise ValueError("No bursts available")
        m = self.bursts["burst_id"] == burst_id
        if not bool(m.any()):
            raise KeyError(f"burst_id {burst_id} not found")
        return self.bursts.loc[m].iloc[0]

    def _set_burst_marker(self, burst_id: int) -> None:
        row = self._burst_row(int(burst_id))
        self.param.update(
            burst_x=float(row["x"]),
            burst_y=float(row["y"]),
            burst_t=float(row["t"]),
            burst_z=float(row["z"]),
        )

    def _jump_to_burst(self, burst_id: int) -> None:
        row = self._burst_row(int(burst_id))
        self.param.update(
            x=float(row["x"]),
            y=float(row["y"]),
            t=float(row["t"]),
            z=float(row["z"]),
        )

    def _on_burst_id(self, event) -> None:
        if event.new is None or len(self.bursts) == 0:
            return
        self._set_burst_marker(int(event.new))
        if self.follow_burst:
            self._jump_to_burst(int(event.new))

    # --------------------------- overlays ---------------------------
    def _burst_point_xy(self):
        if self.burst_id is None or self.burst_x is None or self.burst_y is None:
            return hv.Points([])
        return hv.Points(
            [(self.burst_x, self.burst_y)],
            kdims=[self.hvdims["x"], self.hvdims["y"]],
        ).opts(color="red", marker="circle", size=10, line_width=2, alpha=0.9)

    def _burst_point_tz(self):
        if self.burst_id is None or self.burst_t is None or self.burst_z is None:
            return hv.Points([])
        return hv.Points(
            [(self.burst_t, self.burst_z)],
            kdims=[self.hvdims["t"], self.hvdims["z"]],
        ).opts(color="red", marker="circle", size=10, line_width=2, alpha=0.9)

    # --------------------------- views ---------------------------
    @param.depends(
        "t",
        "z",
        "x",
        "y",
        "clim",
        "use_datashader",
        "t_window",
        "burst_id",
        "burst_t",
        "burst_z",
    )
    def view_tz(self):
        tz_overlay = self._tz_target_dm * self._burst_point_tz()
        # The displayed object is an overlay; ensure Tap listens on that exact object.
        # Otherwise the Tap stream may remain bound to the underlying target DynamicMap
        # and clicks won't propagate in some Panel/HoloViews compositions.
        self.tap_tz.source = tz_overlay
        # Keep t_window syncing tied to the actually displayed object.
        if hasattr(self, "_rx"):
            self._rx.source = tz_overlay

        curve = self._range_curve.opts(
            width=int(self.tz_width),
            height=int(self.ranger_height),
            xlim=self.t_window,
        )
        return (tz_overlay + curve).cols(1).opts(shared_axes=True)

    @param.depends(
        "t",
        "z",
        "x",
        "y",
        "clim",
        "use_datashader",
        "burst_id",
        "burst_x",
        "burst_y",
    )
    def view_xy(self):
        base = super().view_xy()
        return base * self._burst_point_xy()

    # --------------------------- panel UI ---------------------------
    def bursts_table(self):
        if len(self.bursts) == 0:
            return pn.pane.Markdown("_No bursts detected._")

        Tabulator = getattr(pn.widgets, "Tabulator", None)
        if Tabulator is not None:
            return Tabulator(self.bursts, height=250, pagination="remote", page_size=10)
        return pn.widgets.DataFrame(self.bursts, height=250, autosize_mode="fit_columns")

    def burst_nav(self):
        prev_btn = pn.widgets.Button(name="Prev", button_type="primary", width=80)
        next_btn = pn.widgets.Button(name="Next", button_type="primary", width=80)
        goto_btn = pn.widgets.Button(
            name="Go to burst", button_type="default", width=110
        )

        def _prev(_):
            if self.burst_id is None:
                return
            lo, hi = self.param["burst_id"].bounds
            self.burst_id = int(max(lo, self.burst_id - 1))

        def _next(_):
            if self.burst_id is None:
                return
            lo, hi = self.param["burst_id"].bounds
            self.burst_id = int(min(hi, self.burst_id + 1))

        def _goto(_):
            if self.burst_id is None or len(self.bursts) == 0:
                return
            self._jump_to_burst(int(self.burst_id))

        prev_btn.on_click(_prev)
        next_btn.on_click(_next)
        goto_btn.on_click(_goto)

        return pn.Row(
            pn.Param(
                self,
                parameters=["burst_id", "follow_burst"],
                show_name=False,
                width=240,
            ),
            prev_btn,
            next_btn,
            goto_btn,
        )

    def controls_panel(self):
        return pn.Column(
            super().controls_panel(),
        )

    def panel_app(self):
        tz_total_height = int(self.tz_height + self.ranger_height + 80)
        xy_total_height = int(self.xy_height + 80)
        tz_pane = pn.pane.HoloViews(
            self.view_tz,
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
            sizing_mode="fixed",
            width=int(self.xy_width),
        )

        left_col = pn.Column(
            pn.pane.Markdown("### TZ"),
            tz_pane,
            pn.Column(self.controls_panel(), sizing_mode="fixed", width=int(self.tz_width)),
            sizing_mode="fixed",
            width=int(self.tz_width),
            margin=0,
        )
        right_col = pn.Column(
            pn.pane.Markdown("### XY"),
            xy_pane,
            pn.layout.Divider(),
            bursts_panel,
            sizing_mode="fixed",
            width=int(self.xy_width),
            margin=0,
        )
        return pn.Row(
            left_col,
            right_col,
            sizing_mode="fixed",
            margin=0,
            styles={"gap": "12px"},
        )
