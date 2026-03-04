"""
Interactive orthoslicer with a linked time-window selector (experimental).

This module defines :class:`~cogpy.core.plot.orthoslicer_ranger.OrthoSlicerRanger`,
which extends the basic “XY + TZ orthoslice” idea by adding a RangeTool-based
time-window selection:

- A 1D ``rangeslider_sig`` curve (typically a summary over time) acts as the
  RangeTool *source*.
- The TZ view acts as the RangeTool *target* so the selected x-range maps to a
  ``t_window`` parameter.

How it differs from the other orthoslicer modules:

- Unlike ``orthoslicer.py`` / ``orthoslicer_facet.py``, it introduces a linked
  x-range interaction (RangeToolLink) to pick a time window.
- Unlike ``orthoslicer_zoom.py``, the primary focus is time-window selection,
  not persisting general zoom state.
- This file appears to be an *in-progress* refactor: ``panel_app()`` references
  ``view_source`` / ``view_target`` methods that are not implemented here.
  For a more robust implementation of the same concept, prefer
  ``orthoslicer_rangercopy.py``.
"""

import warnings

warnings.warn(
    f"{__name__} is deprecated. "
    "For new projects, use TensorScope (cogpy.core.plot.tensorscope). "
    "For maintenance of existing orthoslicer code, prefer orthoslicer_rangercopy.py.",
    DeprecationWarning,
    stacklevel=2,
)

import numpy as np
import xarray as xr
import holoviews as hv
import panel as pn
import param
import datashader as ds
from holoviews.operation.datashader import rasterize
from holoviews import streams
from holoviews.plotting.links import RangeToolLink

# If you have this locally, keep it; otherwise replace with your own time player
from ..plot.time_player import PlayerWithRealTime

hv.extension("bokeh")
pn.extension()


def _clip_pair(lo, hi, bounds):
    b0, b1 = bounds
    # clip, then ensure strictly ordered
    c0 = min(max(lo, b0), b1)
    c1 = min(max(hi, b0), b1)
    if c0 == c1:  # widen a hair to avoid zero-width
        eps = (b1 - b0) * 1e-9 if b1 > b0 else 1e-9
        c1 = min(c0 + eps, b1)
    return (c0, c1)


def nearest_sel(dimx, val):
    return dimx.sel({dimx.name: val}, method="nearest").item()


class OrthoSlicerRanger(param.Parameterized):
    """
    Panel/HoloViews refactor exposing:
      - view_source()   -> range-slider curve (source for RangeTool)
      - view_target()   -> TZ orthoslice (target), linked to source
      - panel()         -> full Panel app: controls + composite
    Extra:
      - view_xy()       -> XY orthoslice with crosshair & histogram

    Parameters:
      use_datashader : toggle datashader
      t, x, y, z     : slicing coords (kept numeric; discrete sliders provided)
      clim           : color limits
    """

    # --- parameters (Panel will render these) ---
    use_datashader = param.Boolean(True, label="Use Datashader")
    t = param.Number(label="Time")
    x = param.Number(label="X")
    y = param.Number(label="Y")
    z = param.Number(label="Z")
    clim = param.Range(label="Color Limits")
    t_window = param.Range(label="Time window")  # selection on X (time)

    # layout toggles
    merge_tools = param.Boolean(default=False)

    def __init__(
        self,
        array,
        rangeslider_sig=None,
        dt=None,
        dz=None,
        dy=None,
        dx=None,
        clim_quantile=True,
        **params,
    ):
        """
        array: xr.DataArray, dims include whatever you map to (t,z,y,x)
        rangeslider_sig: xr.DataArray (1D over original time), drives the RangeTool
        dt/dz/dy/dx: tuples (original_dim_name, hv.Dimension(...))
        """
        super().__init__(**params)

        if not isinstance(array, xr.DataArray):
            raise ValueError("Input must be an xarray.DataArray.")

        self.array = array
        self.rangeslider_sig = rangeslider_sig
        self.clim_quantile = clim_quantile

        # dimension metadata
        self.dt, self.dz, self.dy, self.dx = dt, dz, dy, dx
        self.vdim = hv.Dimension(
            array.name or "val",
            label=(array.name or "Value"),
            unit=array.attrs.get("units", ""),
        )

        # tap streams
        self.tap_xy = streams.Tap()  # we'll set .source later in view_xy
        self.tap_tz = streams.Tap()

        # Bind tap_tz immediately to the stable DynamicMap (we'll create it in _build_core)
        # self.tap_tz = None  # will be created in _build_core
        # self.tap_tz = streams.Tap()

        # setup all internals
        self._setup_all()

    # --------------------------- setup pipeline ------------------------------
    def _setup_all(self):
        self._prepare_meta()
        self._standardize_coords()
        self._set_param_bounds()
        self._set_contrast_limits()
        self._set_controls()

    def _prepare_meta(self):
        dims = dict(x=self.dx, y=self.dy, z=self.dz, t=self.dt)
        self._rename_map, self._original_name, self.hvdims = {}, {}, {}
        for std_name, spec in dims.items():
            if spec is None:
                continue
            src, hvdim = spec
            self._rename_map[src] = std_name
            self._original_name[std_name] = src
            self.hvdims[std_name] = hvdim
            # propagate labels to param widgets if present
            if hvdim.label and std_name in self.param:
                self.param[std_name].label = hvdim.label

        self._units = {k: v.unit for k, v in self.hvdims.items()}
        self._labels = {k: v.label for k, v in self.hvdims.items()}

        # handy aliases
        self.ox, self.oy, self.oz, self.ot = (
            self._original_name.get("x", "x"),
            self._original_name.get("y", "y"),
            self._original_name.get("z", "z"),
            self._original_name.get("t", "t"),
        )

    def _standardize_coords(self):
        # rename & transpose to the canonical order
        self.array = self.array.rename(self._rename_map).transpose("t", "z", "y", "x")
        self.array.name = "val"

        # rename time of the range-signal, keep it 1D on t
        if self.rangeslider_sig is None:
            # fallback: build a simple mean signal over the volume
            self.rangeslider_sig = self.array.mean(dim=("z", "y", "x"))
        if isinstance(self.rangeslider_sig, xr.DataArray):
            if self.dt is not None:
                src_t = self.dt[0]
                if src_t in self.rangeslider_sig.dims:
                    self.rangeslider_sig = self.rangeslider_sig.rename(
                        {src_t: self._rename_map.get(src_t, "t")}
                    )

        # update param labels from hvdims (nice display names)
        for dim in ("t", "z", "x", "y"):
            if dim in self.param and dim in self._labels:
                self.param[dim].label = self._labels[dim] or dim

    def _get_bounds(self):
        b = {
            d: (float(self.array[d].min().item()), float(self.array[d].max().item()))
            for d in self.array.dims
        }
        b[self.array.name] = (
            float(self.array.min().item()),
            float(self.array.max().item()),
        )
        return b

    def _set_param_bounds(self):
        self.bounds = self._get_bounds()
        for dim in ("t", "x", "y", "z"):
            lo, hi = self.bounds[dim]
            self.param[dim].bounds = (lo, hi)
            setattr(self, dim, lo)

        # --- set t_window.bounds FIRST
        t_bounds = self.param["t"].bounds
        self.param["t_window"].bounds = t_bounds

        # initial window using percentiles, but CLIP it
        t0, t1 = np.percentile(self.array.t.values, [10, 30])
        self.t_window = _clip_pair(float(t0), float(t1), t_bounds)

    def _set_contrast_limits(self):
        vmin, vmax = self.bounds["val"]
        if self.clim_quantile:
            vals = self.array.values
            clim_init = (float(np.percentile(vals, 2)), float(np.percentile(vals, 98)))
        else:
            clim_init = (float(vmin), float(vmax))
        self.clim = clim_init
        self.param["clim"].bounds = (float(vmin), float(vmax))
        self.param["clim"].default = clim_init

    # ----------------------------- core plots --------------------------------
    def _fmt_val(self, dim, val):
        u = self._units.get(dim, "")
        return f"{val:.2f} {u}" if u else f"{val:.2f}"

    def _title_xy(self):
        return f"{self.ox}-{self.oy} @ {self.oz}={self._fmt_val('z', self.z)}, {self.ot}={self._fmt_val('t', self.t)}"

    def _title_tz(self):
        return f"{self.ot}-{self.oz} @ {self.ox}={self._fmt_val('x', self.x)}, {self.oy}={self._fmt_val('y', self.y)}"

    # --------------------------- public HV views ------------------------------
    @param.depends("t", "z", "x", "y", "clim", "use_datashader")
    def view_tz(self):

        if hasattr(self, "_rtl"):
            # readout the live bounds:
            self.t_window = self._rx.x_range

        width, height, ranger_hegiht = 400, 300, 120

        # base
        img = self.array.sel(x=self.x, y=self.y, method="nearest")
        base = hv.Image(
            img, kdims=[self.hvdims["t"], self.hvdims["z"]], vdims=[self.vdim]
        )
        if self.use_datashader:
            base = rasterize(
                base,
                aggregator=ds.mean("val"),
                width=width,
                height=height,
                dynamic=False,
            )
        base = base.opts(
            cmap="Viridis",
            colorbar=True,
            framewise=True,
            title=self._title_tz(),
            clim=self.clim,
            width=width,
            height=height,
            tools=["tap", "pan", "wheel_zoom", "box_zoom", "reset"],
            active_tools=["tap"],
        )

        # crosshair
        cross = hv.VLine(self.t).opts(
            color="white", line_width=1, alpha=0.8
        ) * hv.HLine(self.z).opts(color="white", line_width=1, alpha=0.8)

        # bind tap to base (not the overlay)
        self.tap_tz.source = base

        # RangeToolLink
        self._range_curve = hv.Curve(self.rangeslider_sig, kdims=["t"]).opts(
            width=width, height=ranger_hegiht, tools=[]
        )
        tb = self.param["t_window"].bounds

        # Range x to keep track of the bounds
        self._rx = streams.RangeX(source=base, x_range=self.t_window)
        self._rtl = RangeToolLink(
            self._range_curve, base, axes=["x"], boundsx=self.t_window
        )
        return (base.hist() * cross + self._range_curve).cols(1)

    @param.depends("t", "z", "x", "y", "clim", "use_datashader")
    def view_xy(self):
        width, height = 400, 300
        img = self.array.sel(t=self.t, z=self.z, method="nearest")
        base = hv.Image(
            img, kdims=[self.hvdims["x"], self.hvdims["y"]], vdims=[self.vdim]
        )
        if self.use_datashader:
            base = rasterize(
                base,
                aggregator=ds.mean("val"),
                width=width,
                height=height,
                dynamic=False,
            )
        base = base.opts(
            cmap="Viridis",
            colorbar=True,
            framewise=True,
            title=self._title_xy(),
            clim=self.clim,
            width=width,
            height=height,
            tools=["tap", "pan", "wheel_zoom", "box_zoom", "reset"],
            active_tools=["tap"],
        )
        cross = hv.VLine(self.x).opts(
            color="white", line_width=1, alpha=0.8
        ) * hv.HLine(self.y).opts(color="white", line_width=1, alpha=0.8)

        # bind tap to THIS base each re-render
        self.tap_xy.source = base

        return base.hist() * cross

    # ------------------------------- controls --------------------------------
    def _set_controls(self):
        # discrete coordinate options
        x_vals = list(map(float, self.array.x.values))
        y_vals = list(map(float, self.array.y.values))
        z_vals = list(map(float, self.array.z.values))

        # param controls (x, y, z, clim, datashader)
        self.param_controls = pn.Param(
            self,
            parameters=["x", "y", "z", "clim", "use_datashader", "merge_tools"],
            widgets={
                "x": pn.widgets.DiscreteSlider(options=x_vals, name=self.ox),
                "y": pn.widgets.DiscreteSlider(options=y_vals, name=self.oy),
                "z": pn.widgets.DiscreteSlider(options=z_vals, name=self.oz),
                "clim": pn.widgets.RangeSlider,
                "use_datashader": pn.widgets.Checkbox,
            },
            show_name=False,
            width=400,
        )

        # time player (kept bi-directional with self.t)
        self._set_t_player()

        # tap callbacks
        self.tap_xy.add_subscriber(self._on_tap_xy)
        self.tap_tz.add_subscriber(self._on_tap_tz)

    def _set_t_player(self):
        self.t_player_widget = PlayerWithRealTime(
            self.array.t.values, interval_bounds=(50, 2000)
        )
        self.t_player = self.t_player_widget.t_player
        # keep player <-> param.t in sync
        self.t_player.param.watch(lambda e: setattr(self, "t", e.new), "value")
        self.param.watch(lambda e: setattr(self.t_player, "value", e.new), "t")

    # ------------------------------ taps -------------------------------------
    def _on_tap_xy(self, x, y):
        if x is not None and y is not None:
            self.param.update(
                x=nearest_sel(self.array.x, x),
                y=nearest_sel(self.array.y, y),
            )

    # --- IMPORTANT: accept **kwargs for Tap callbacks
    def _on_tap_tz(self, **kwargs):
        x, y = kwargs.get("x"), kwargs.get("y")
        if x is not None and y is not None:
            self.param.update(
                t=nearest_sel(self.array.t, x),
                z=nearest_sel(self.array.z, y),
            )

    # ------------------------------ panel app --------------------------------
    def controls_panel(self):
        return pn.Column(
            pn.pane.Markdown("### Time"),
            self.t_player_widget.view,
            pn.layout.Divider(),
            pn.pane.Markdown("### Controls"),
            self.param_controls,
        )

    def panel_app(self):
        """
        Full Panel application
        Left: TZ target (linked to range-curve)
        └─ Range curve (source for RangeTool)
        └─ Controls (bottom)
        Right: XY orthoslice
        """
        # Dynamic panes: pass callables (no parentheses)
        target_pane_tz = pn.pane.HoloViews(
            self.view_target, width=400, height=300, sizing_mode="fixed"
        )
        source_pane_tz = pn.pane.HoloViews(
            self.view_source(), width=400, height=120, sizing_mode="fixed"
        )
        xy_pane = pn.pane.HoloViews(
            self.view_xy, width=400, height=300, sizing_mode="fixed"
        )

        left_col = pn.Column(
            target_pane_tz,
            source_pane_tz,
            self.controls_panel(),
            sizing_mode="stretch_width",
        )

        right_col = pn.Column(
            xy_pane,
            sizing_mode="stretch_width",
        )

        return pn.Row(left_col, right_col, sizing_mode="stretch_width")

    def panel_app2(self):
        """
        Full Panel application: controls + a Tabs view.
        Tab 1: Linked TZ target over range-curve (composite).
        Tab 2: XY orthoslice.
        """
        linked = self.composite_hv()
        xyview = self.view_xy

        app = pn.Column(
            pn.Row(
                pn.pane.HoloViews(linked, sizing_mode="stretch_width"),
                pn.pane.HoloViews(xyview, sizing_mode="stretch_width"),
            ),
            self.controls_panel(),
            sizing_mode="stretch_width",
        )

        return app

    def panel_app3(self):
        tzview = self.view_tz
        xyview = self.view_xy

        app = pn.Column(
            pn.Row(
                pn.pane.HoloViews(tzview, sizing_mode="stretch_width"),
                pn.pane.HoloViews(xyview, sizing_mode="stretch_width"),
            ),
            self.controls_panel(),
            sizing_mode="stretch_width",
        )
        return app

    def show(self):
        self.panel_app().show()
