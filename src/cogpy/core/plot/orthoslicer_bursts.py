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

import warnings

warnings.warn(
    f"{__name__} is deprecated. "
    "For new projects, use TensorScope (cogpy.core.tensorscope). "
    "For maintenance of existing orthoslicer code, prefer orthoslicer_rangercopy.py.",
    DeprecationWarning,
    stacklevel=2,
)

import pandas as pd
import holoviews as hv
import panel as pn
import param
from holoviews import streams
import numpy as np

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
        default=True,
        label="Follow burst",
        doc="If True, selecting a burst jumps the (x,y,z) crosshairs to it and recenters the time window around the burst time (without changing `t`).",
    )
    show_all_bursts_tz = param.Boolean(
        default=True,
        label="Show bursts (TZ)",
        doc="If True, show all burst (t,z) peaks on the TZ plot, regardless of current (x,y).",
    )

    tz_time_panels = param.ListSelector(
        default=["burst_trace", "burst_rate"],
        objects=["burst_trace", "burst_rate"],
        label="TZ time panels",
        doc="Additional time plots to show under the overview curve.",
    )
    tz_trace_height = param.Integer(default=140, bounds=(60, None), label="Trace height")
    tz_rate_height = param.Integer(default=140, bounds=(60, None), label="Rate height")
    burst_trace_bandwidth_hz = param.Number(
        default=0.0,
        bounds=(0.0, None),
        label="Trace BW (Hz)",
        doc="If >0, mean over z in [burst_z±BW/2] to approximate a bandpassed energy trace.",
    )
    burst_rate_sigma_s = param.Number(
        default=0.15,
        bounds=(0.0, None),
        label="Rate σ (s)",
        doc="Gaussian smoothing sigma for burst rate over time (seconds). 0 disables smoothing.",
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
        self.param.watch(self._on_follow_burst, "follow_burst")
        self._build_tz_display()
        self._build_tz_time_panels()
        # Refresh stable layout to include extra panels.
        self._tz_layout = self._build_tz_layout()

    def _on_follow_burst(self, event) -> None:
        if not bool(event.new):
            return
        if self.burst_id is None or len(self.bursts) == 0:
            return
        # When follow_burst is enabled, immediately jump (and recenter time window)
        # to the currently selected burst.
        self._jump_to_burst(int(self.burst_id))

    def _build_tz_display(self) -> None:
        """
        Build a stable TZ overlay DynamicMap including the burst marker.

        Using a stable DynamicMap here keeps Bokeh stream wiring (Tap/RangeX)
        more reliable than dynamically multiplying Element overlays inside
        `view_tz()` on each render.
        """
        burst_param_names = ["burst_id", "burst_t", "burst_z", "show_all_bursts_tz"]
        burst_param_names = [p for p in burst_param_names if p in self.param]
        self._burst_params = streams.Params(self, parameters=burst_param_names)
        self._all_bursts_point_dm = hv.DynamicMap(
            lambda **_: self._all_bursts_points_tz(), streams=[self._burst_params]
        )
        self._burst_point_dm = hv.DynamicMap(
            lambda **_: self._burst_point_tz(), streams=[self._burst_params]
        )
        # Layer order: image + crosshair (parent) -> all bursts -> selected burst.
        self._tz_display_dm = (
            self._tz_display_dm * self._all_bursts_point_dm * self._burst_point_dm
        )
        # Rebuild the stable TZ layout now that we've extended _tz_display_dm.
        self._tz_layout = self._build_tz_layout()

    def _build_tz_time_panels(self) -> None:
        panel_param_names = [
            "burst_id",
            "burst_x",
            "burst_y",
            "burst_t",
            "burst_z",
            "tz_time_panels",
            "t_window",
            "tz_width",
            "tz_trace_height",
            "tz_rate_height",
            "burst_trace_bandwidth_hz",
            "burst_rate_sigma_s",
        ]
        panel_param_names = [p for p in panel_param_names if p in self.param]
        self._tz_panel_params = streams.Params(self, parameters=panel_param_names)

        self._tz_track_dms = {
            "burst_trace": hv.DynamicMap(self._burst_trace_curve, streams=[self._tz_panel_params]),
            "burst_rate": hv.DynamicMap(self._burst_rate_curve, streams=[self._tz_panel_params]),
        }

    def _burst_trace_curve(self, **_):
        if "burst_trace" not in list(getattr(self, "tz_time_panels", [])):
            return hv.Curve([]).opts(height=1, width=int(self.tz_width), toolbar="disable")
        if self.burst_x is None or self.burst_y is None or self.burst_z is None:
            return hv.Text(0.5, 0.5, "No selected burst").opts(height=int(self.tz_trace_height), width=int(self.tz_width))

        bw = float(getattr(self, "burst_trace_bandwidth_hz", 0.0) or 0.0)
        if bw > 0:
            z0 = float(self.burst_z) - bw / 2.0
            z1 = float(self.burst_z) + bw / 2.0
            slab = self.array.sel(x=self.burst_x, y=self.burst_y, method="nearest").sel(z=slice(z0, z1)).mean("z")
            title = f"Band energy @ (x,y)=({self._fmt_val('x', self.burst_x)}, {self._fmt_val('y', self.burst_y)}), z∈[{z0:.2f},{z1:.2f}]"
        else:
            slab = self.array.sel(x=self.burst_x, y=self.burst_y, z=self.burst_z, method="nearest")
            title = f"Trace @ (x,y,z)=({self._fmt_val('x', self.burst_x)}, {self._fmt_val('y', self.burst_y)}, {self._fmt_val('z', self.burst_z)})"

        t = np.asarray(slab.t.values, dtype=float).reshape(-1)
        v = np.asarray(slab.values, dtype=float).reshape(-1)
        curve = hv.Curve((t, v), kdims=[self.hvdims["t"]], vdims=[self.vdim]).opts(
            width=int(self.tz_width),
            height=int(self.tz_trace_height),
            title=title,
            tools=["pan", "wheel_zoom", "reset"],
            active_tools=["wheel_zoom"],
            framewise=False,
        )
        if self.burst_t is not None:
            curve = curve * hv.VLine(float(self.burst_t)).opts(color="red", line_width=2, alpha=0.8)

        tb = self.param["t_window"].bounds
        xlim = _clip_pair(*self.t_window, tb)
        return curve.opts(xlim=xlim)

    def _burst_rate_curve(self, **_):
        if "burst_rate" not in list(getattr(self, "tz_time_panels", [])):
            return hv.Curve([]).opts(height=1, width=int(self.tz_width), toolbar="disable")
        if len(self.bursts) == 0 or "t" not in self.bursts.columns:
            return hv.Text(0.5, 0.5, "No bursts").opts(height=int(self.tz_rate_height), width=int(self.tz_width))

        tgrid = np.asarray(self.array.t.values, dtype=float).reshape(-1)
        if tgrid.size < 2:
            return hv.Curve([]).opts(height=1, width=int(self.tz_width), toolbar="disable")
        dt = float(np.nanmedian(np.diff(tgrid)))
        dt = dt if dt > 0 else 1.0

        # Histogram bursts onto the array time grid.
        bt = np.asarray(self.bursts["t"].values, dtype=float).reshape(-1)
        idx = np.searchsorted(tgrid, bt, side="left")
        idx = np.clip(idx, 0, tgrid.size - 1)
        counts = np.zeros_like(tgrid, dtype=float)
        np.add.at(counts, idx, 1.0)

        # Convert to rate (bursts per second) and smooth.
        rate = counts / dt
        sigma_s = float(getattr(self, "burst_rate_sigma_s", 0.0) or 0.0)
        if sigma_s > 0:
            sigma = max(1e-9, sigma_s / dt)
            rad = int(max(3, np.ceil(4.0 * sigma)))
            x = np.arange(-rad, rad + 1, dtype=float)
            k = np.exp(-0.5 * (x / sigma) ** 2)
            k = k / np.sum(k)
            rate = np.convolve(rate, k, mode="same")

        curve = hv.Curve((tgrid, rate), kdims=[self.hvdims["t"]], vdims=[hv.Dimension("rate", label="Burst rate", unit="1/s")]).opts(
            width=int(self.tz_width),
            height=int(self.tz_rate_height),
            title="Burst rate (smoothed)",
            tools=["pan", "wheel_zoom", "reset"],
            active_tools=["wheel_zoom"],
            framewise=False,
        )
        if self.burst_t is not None:
            curve = curve * hv.VLine(float(self.burst_t)).opts(color="red", line_width=2, alpha=0.6)

        tb = self.param["t_window"].bounds
        xlim = _clip_pair(*self.t_window, tb)
        return curve.opts(xlim=xlim)

    def _tz_extra_panels(self):
        track_dms = getattr(self, "_tz_track_dms", {}) or {}
        out = []
        for track_id in list(getattr(self, "tz_time_panels", [])):
            dm = track_dms.get(track_id)
            if dm is not None:
                out.append(dm)
        return out

    def _all_bursts_points_tz(self):
        if not bool(getattr(self, "show_all_bursts_tz", True)) or len(self.bursts) == 0:
            return hv.Points([])
        pts = hv.Points(
            self.bursts[["t", "z"]],
            kdims=[self.hvdims["t"], self.hvdims["z"]],
        )
        # Important: keep this layer passive (no tools) so it doesn't interfere
        # with the Tap stream bound to the TZ image layer.
        return pts.opts(color="#ffb000", size=6, alpha=0.45, line_width=0, tools=[])

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
        updates = dict(
            x=float(row["x"]),
            y=float(row["y"]),
            z=float(row["z"]),
        )
        # Recenter the time window so the RangeTool selection (and TZ x-range)
        # jumps to the burst time, but do not force the global time cursor `t`
        # to jump as well.
        tb = self.param["t_window"].bounds
        tc = float(row["t"])
        if bool(getattr(self, "reset_zoom_on_navigation", False)):
            updates["t_window"] = self._default_window_around(tc)
        else:
            w = float(self.t_window[1] - self.t_window[0])
            w = w if w > 0 else (tb[1] - tb[0]) * 0.1
            updates["t_window"] = _clip_pair(tc - 0.5 * w, tc + 0.5 * w, tb)

        self.param.update(**updates)

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
        ).opts(color="red", marker="circle", size=10, line_width=2, alpha=0.9, tools=[])

    def _burst_point_tz(self):
        if self.burst_id is None or self.burst_t is None or self.burst_z is None:
            return hv.Points([])
        return hv.Points(
            [(self.burst_t, self.burst_z)],
            kdims=[self.hvdims["t"], self.hvdims["z"]],
        ).opts(color="red", marker="circle", size=10, line_width=2, alpha=0.9, tools=[])

    # --------------------------- views ---------------------------
    @param.depends(
        "t",
        "z",
        "x",
        "y",
        "clim",
        "use_datashader",
        "t_window",
        "tz_width",
        "ranger_height",
        "burst_id",
        "burst_t",
        "burst_z",
    )
    def view_tz(self):
        return self._tz_layout

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
                parameters=["burst_id", "follow_burst", "show_all_bursts_tz"],
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
            pn.layout.Divider(),
            pn.pane.Markdown("### Time Panels"),
            pn.Param(
                self,
                parameters=[
                    "tz_time_panels",
                    "tz_trace_height",
                    "tz_rate_height",
                    "burst_trace_bandwidth_hz",
                    "burst_rate_sigma_s",
                ],
                show_name=False,
                width=int(self.tz_width),
            ),
        )

    def panel_app(self):
        tz_total_height = int(
            self.tz_height
            + self.ranger_height
            + (self.tz_trace_height if "burst_trace" in list(self.tz_time_panels) else 1)
            + (self.tz_rate_height if "burst_rate" in list(self.tz_time_panels) else 1)
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


def main() -> None:
    """
    Debug entrypoint: run a small demo app.

    This intentionally mirrors the docstring example but lives in a callable so
    it can be used via ``python -m cogpy.core.plot.orthoslicer_bursts``.
    """
    from cogpy.datasets.tensor import make_flat_blob_dataset, detect_bursts_hmaxima

    pn.extension()
    hv.extension("bokeh")

    da = make_flat_blob_dataset(
        duration=2.0,
        nt=80,
        n_peaks=5,
        peak_dist="uniform",
        blob_sigma_idx=(0.8, 0.8, 2.0, 0.8),
        blob_amp=10.0,
        seed=0,
    )
    bursts = detect_bursts_hmaxima(da, h_quantile=0.99)

    dx = ("ml", hv.Dimension("x", label="Medial-Lateral", unit="mm"))
    dy = ("ap", hv.Dimension("y", label="Anterior-Posterior", unit="mm"))
    dt = ("time", hv.Dimension("t", label="Time", unit="s"))
    dz = ("freq", hv.Dimension("z", label="Frequency", unit="Hz"))

    slicer = OrthoSlicerRangerBursts(da, bursts=bursts, dt=dt, dz=dz, dy=dy, dx=dx)
    slicer.tz_logy = True

    # Debug: print tap events to the console.
    slicer.tap_xy.add_subscriber(lambda **kw: print("tap_xy", kw))
    slicer.tap_tz.add_subscriber(lambda **kw: print("tap_tz", kw))

    slicer.panel_app().show()


if __name__ == "__main__":
    main()
