"""
Interactive orthoslicer with a linked time-window view (no RangeTool).

This module defines :class:`~cogpy.core.plot.orthoslicer_rangercopy.OrthoSlicerRanger`,
a stabilized orthoslicer concept that shows a 1D time summary curve underneath
the time–frequency view with a shared time axis.

Key differences vs. ``orthoslicer_ranger.py``:

- The core plotting plumbing is built once in ``_build_core()`` using stable
  HoloViews ``DynamicMap`` objects and streams, instead of being recreated
  inside the view functions on every parameter update.
- ``panel_app()`` uses ``view_tz`` / ``view_xy`` consistently.

Compared to the other orthoslicer modules:

- Adds a linked time-window (``t_window``) that tracks the visible TZ x-range.
- Does not focus on general zoom persistence (see ``orthoslicer_zoom.py``) or
  faceting/local sampling helpers (see ``orthoslicer_facet.py``).

Example
-------
>>> import numpy as np
>>> import xarray as xr
>>> import holoviews as hv
>>> import panel as pn
>>> from cogpy.datasets.tensor import make_dataset
>>> from cogpy.core.plot.orthoslicer_rangercopy import OrthoSlicerRanger
>>>
>>> pn.extension()
>>> hv.extension("bokeh")
>>>
>>> da = make_dataset()
>>> da = xr.concat([da] * 100, dim="time")
>>> fs = float(da["time"].values[1] - da["time"].values[0])
>>> da = da.assign_coords({"time": fs * np.arange(da.sizes["time"])})
>>>
>>> dx = ("ml", hv.Dimension("x", label="Medial-Lateral", unit="mm"))
>>> dy = ("ap", hv.Dimension("y", label="Anterior-Posterior", unit="mm"))
>>> dt = ("time", hv.Dimension("t", label="Time", unit="s"))
>>> dz = ("freq", hv.Dimension("z", label="Frequency", unit="Hz"))
>>>
>>> # 1D signal plotted under the TZ view (defaults to a mean if omitted)
>>> rangeslider_sig = da.mean(dim=("freq", "ap", "ml"))
>>>
>>> slicer = OrthoSlicerRanger(
...     da, rangeslider_sig=rangeslider_sig, dt=dt, dz=dz, dy=dy, dx=dx
... )
>>> slicer.panel_app().show()
"""

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


def _safe_quantile_clim(values: np.ndarray, qlo: float, qhi: float, fallback):
    vals = np.asarray(values, dtype=float)
    if vals.size == 0:
        return fallback
    try:
        lo, hi = np.nanpercentile(vals, [qlo, qhi])
        lo, hi = float(lo), float(hi)
    except Exception:
        return fallback
    if not np.isfinite(lo) or not np.isfinite(hi):
        return fallback
    if lo == hi:
        eps = abs(lo) * 1e-9 if lo != 0 else 1e-9
        hi = lo + eps
    return (lo, hi)


def _set_active_scroll_wheel_zoom(plot, _element):
    """
    Make scroll-wheel zoom the active scroll tool without disabling Tap.

    Bokeh distinguishes between "active_tap" and "active_scroll". HoloViews'
    `active_tools` doesn't reliably configure scroll vs. tap when both are listed,
    so we set the scroll tool explicitly via a hook.
    """
    try:
        from bokeh.models import WheelZoomTool

        fig = plot.state
        wz = next((t for t in fig.toolbar.tools if isinstance(t, WheelZoomTool)), None)
        if wz is not None:
            fig.toolbar.active_scroll = wz
    except Exception:
        return


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
    tz_logy = param.Boolean(default=False, label="Log Frequency Axis")
    use_range_tool = param.Boolean(default=True, label="Range Selector")

    # Adaptive color scaling (separate for TZ vs XY)
    tz_autoscale = param.Boolean(
        default=False,
        label="Auto clim (TZ)",
        doc="Autoscale TZ color limits from current TZ slice quantiles.",
    )
    xy_autoscale = param.Boolean(
        default=False,
        label="Auto clim (XY)",
        doc="Autoscale XY color limits from current XY slice quantiles.",
    )
    tz_clim_quantiles = param.Range(
        default=(2.0, 98.0),
        bounds=(0.0, 100.0),
        label="TZ quantiles (%)",
        doc="Low/high percentiles for TZ autoscaling.",
    )
    xy_clim_quantiles = param.Range(
        default=(2.0, 98.0),
        bounds=(0.0, 100.0),
        label="XY quantiles (%)",
        doc="Low/high percentiles for XY autoscaling.",
    )

    # Navigation + zoom policy
    nav_step_s = param.Number(
        default=0.25,
        bounds=(0.0, None),
        label="Step (s)",
        doc="Time step for navigation buttons/shortcuts.",
    )
    default_window_fraction = param.Number(
        default=0.25,
        bounds=(0.01, 1.0),
        label="Default window frac",
        doc="Fraction of full time range used by Reset view.",
    )
    reset_zoom_on_navigation = param.Boolean(
        default=False,
        label="Reset view on nav",
        doc="If True, navigation resets t_window around the new time.",
    )
    enable_shortcuts = param.Boolean(
        default=False,
        label="Keyboard shortcuts",
        doc="If True, enable basic keyboard shortcuts when running as a Panel server.",
    )

    # plot sizing (used for both HoloViews opts and datashader rasterization)
    tz_width = param.Integer(default=400, bounds=(200, None), label="TZ Width")
    tz_height = param.Integer(default=300, bounds=(200, None), label="TZ Height")
    xy_width = param.Integer(default=400, bounds=(200, None), label="XY Width")
    xy_height = param.Integer(default=300, bounds=(200, None), label="XY Height")
    ranger_height = param.Integer(default=120, bounds=(60, None), label="Ranger Height")

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

        # Navigation history (t_window snapshots)
        self._t_window_history: list[tuple[float, float]] = []
        self._t_window_history_idx: int = -1
        self._shortcuts_registered: bool = False

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
        self._build_core()  # ← add this call
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
            # If user provided a signal in original coordinates, rename its time dim.
            # If we created it from the standardized array, it's already on "t".
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
        self._push_window_history(self.t_window)

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

    def _build_core(self):
        # 1) params stream -> drives everything
        tz_param_names = [
            "x",
            "y",
            "t",
            "z",
            "clim",
            "use_datashader",
            "tz_logy",
            "tz_autoscale",
            "tz_clim_quantiles",
            "tz_width",
            "tz_height",
        ]
        # Be robust to older class objects in long-running notebook kernels where
        # parameters may not exist yet.
        tz_param_names = [p for p in tz_param_names if p in self.param]
        self._tz_params = streams.Params(self, parameters=tz_param_names)

        # 2) STABLE DynamicMaps (never recreate in views)
        #
        # IMPORTANT: Bind interactive streams (Tap/RangeX) to a *single Element*
        # DynamicMap (the image). Using an Overlay/Layout as a stream source is
        # fragile in HoloViews and can cause Tap events to never reach Python.
        self._tz_img_dm = hv.DynamicMap(self._tz_img, streams=[self._tz_params])
        self._tz_xhair_dm = hv.DynamicMap(self._tz_crosshair, streams=[self._tz_params])
        self._tz_display_dm = self._tz_img_dm * self._tz_xhair_dm

        # 3) Track visible x-range; keep t_window in sync
        self._rx = streams.RangeX(source=self._tz_img_dm)

        def _on_range_x(x_range):
            if x_range:
                tb = self.param["t_window"].bounds
                self.t_window = _clip_pair(float(x_range[0]), float(x_range[1]), tb)

        self._rx.add_subscriber(lambda **kw: _on_range_x(kw.get("x_range")))

        # 4) Static curve – one Bokeh figure, never recreated.
        #    DO NOT set xlim here: shared_axes=True in the Layout links the
        #    x-ranges of both Bokeh figures entirely in the browser.  As soon
        #    as you set an explicit xlim the Bokeh range is locked and the
        #    shared-axes link stops working.
        self._range_curve = hv.Curve(self.rangeslider_sig, kdims=["t"]).opts(
            width=int(self.tz_width),
            height=int(self.ranger_height),
            tools=[],
            default_tools=[],
            framewise=False,
        )
        if bool(getattr(self, "use_range_tool", False)):
            # In RangeTool mode, keep the overview curve full-range so the
            # selector box remains meaningful.
            self._range_curve = self._range_curve.opts(xlim=self.param["t"].bounds)

        # 5) Optional RangeToolLink: pick a time window on the curve and link it
        #    to the TZ image x-range. Keep a reference to avoid GC.
        self._rtl = None
        if bool(getattr(self, "use_range_tool", False)):
            self._rtl = RangeToolLink(
                self._range_curve,
                self._tz_img_dm,
                axes=["x"],
                boundsx=self.t_window,
            )
            # Keep the selector bounds synced to the current t_window param even
            # though the layout is stable and view_tz() is non-reactive.
            def _sync_bounds(_event=None):
                tb = self.param["t_window"].bounds
                self._rtl.boundsx = _clip_pair(*self.t_window, tb)

            self.param.watch(lambda _e: _sync_bounds(), "t_window")

        # 6) Tap bound to the clickable image layer (not the overlay)
        self.tap_tz = streams.Tap(source=self._tz_img_dm, x=None, y=None)
        self.tap_tz.add_subscriber(self._on_tap_tz)

        # 7) Build the stable TZ composite once. The DynamicMaps update in-place
        # (crosshair, image slices, etc.) without needing Panel to replace the
        # underlying Bokeh model (which can break stream callbacks).
        self._tz_layout = self._build_tz_layout()

        # ----------------------------- XY core -----------------------------
        xy_param_names = [
            "t",
            "z",
            "x",
            "y",
            "clim",
            "use_datashader",
            "xy_autoscale",
            "xy_clim_quantiles",
            "xy_width",
            "xy_height",
        ]
        xy_param_names = [p for p in xy_param_names if p in self.param]
        self._xy_params = streams.Params(self, parameters=xy_param_names)

        # Stable XY DynamicMaps (image is the only clickable source).
        self._xy_img_dm = hv.DynamicMap(self._xy_img, streams=[self._xy_params])
        self._xy_xhair_dm = hv.DynamicMap(self._xy_crosshair, streams=[self._xy_params])
        self._xy_display_dm = self._xy_img_dm * self._xy_xhair_dm

        # Bind Tap to the clickable image layer once (never re-source in view_xy).
        self.tap_xy = streams.Tap(source=self._xy_img_dm, x=None, y=None)
        self.tap_xy.add_subscriber(self._on_tap_xy)

        # Stable XY composite: histogram + crosshair overlay.
        # NOTE: keep Tap bound to _xy_img_dm, not the composed layout.
        self._xy_layout = (self._xy_img_dm.hist() * self._xy_xhair_dm)

    def _build_tz_layout(self):
        """
        Build the TZ composite (TZ view + overview curve) as a stable object.

        Returning brand-new Layout objects on every param update can lead Panel
        to replace the Bokeh model repeatedly. That replacement is a common
        source of broken HoloViews streams/callbacks (e.g., Tap/RangeX), showing
        up as 'NoneType has no attribute comm' in Bokeh/holoviews callbacks.
        """
        panels = [self._tz_display_dm, self._range_curve]
        panels.extend(self._tz_extra_panels())

        layout = hv.Layout(panels).cols(1)

        if getattr(self, "_rtl", None) is not None and bool(getattr(self, "use_range_tool", False)):
            # RangeTool mode: selector box on the curve controls TZ x-range.
            # Merge toolbars so RangeTool stays attached to the curve only.
            return layout.opts(merge_tools=True)

        # Shared-axes mode: curve mirrors TZ x-range client-side (no selector).
        return layout.opts(shared_axes=True)

    def _tz_extra_panels(self):
        """Extra time panels shown underneath the overview curve."""
        return []

    # ----------------------------- core plots --------------------------------
    def _fmt_val(self, dim, val):
        u = self._units.get(dim, "")
        return f"{val:.2f} {u}" if u else f"{val:.2f}"

    def _title_xy(self):
        return f"{self.ox}-{self.oy} @ {self.oz}={self._fmt_val('z', self.z)}, {self.ot}={self._fmt_val('t', self.t)}"

    def _title_tz(self):
        return f"{self.ot}-{self.oz} @ {self.ox}={self._fmt_val('x', self.x)}, {self.oy}={self._fmt_val('y', self.y)}"

    def _tz_crosshair(self, **_):
        """Crosshair overlay for the TZ view (does not affect axes or tools)."""
        v = hv.VLine(self.t).opts(line_width=2, alpha=0.9, color="white")
        h = hv.HLine(self.z).opts(line_width=2, alpha=0.9, color="white")
        # return as an Overlay so it composes cleanly with the base image DM
        return v * h

    def _tz_img(self, **_):
        """Return ONLY the TZ image for linking & tapping (no crosshair here)."""
        width, height = int(self.tz_width), int(self.tz_height)

        img = self.array.sel(x=self.x, y=self.y, method="nearest")
        base = hv.Image(
            img, kdims=[self.hvdims["t"], self.hvdims["z"]], vdims=[self.vdim]
        )

        clim = self.clim
        if bool(getattr(self, "tz_autoscale", False)):
            qlo, qhi = self.tz_clim_quantiles
            clim = _safe_quantile_clim(img.values, float(qlo), float(qhi), fallback=clim)

        if self.use_datashader:
            base = rasterize(
                base,
                aggregator=ds.mean("val"),
                width=width,
                height=height,
                dynamic=False,
            )

        return base.opts(
            cmap="Viridis",
            colorbar=True,
            title=self._title_tz(),
            clim=clim,
            width=width,
            height=height,
            framewise=False,  # keep ranges stable across frames
            logy=bool(self.tz_logy),
            tools=["tap", "pan", "wheel_zoom", "box_zoom", "reset"],
            # Keep tap active for crosshair movement; configure scroll-wheel zoom via hook.
            active_tools=["tap"],
            hooks=[_set_active_scroll_wheel_zoom],
        )

    def _xy_crosshair(self, **_):
        """Crosshair overlay for the XY view (does not affect axes or tools)."""
        v = hv.VLine(self.x).opts(line_width=2, alpha=0.9, color="white")
        h = hv.HLine(self.y).opts(line_width=2, alpha=0.9, color="white")
        return v * h

    def _xy_img(self, **_):
        """Return ONLY the XY image for linking & tapping (no crosshair here)."""
        width, height = int(self.xy_width), int(self.xy_height)

        img = self.array.sel(t=self.t, z=self.z, method="nearest")
        base = hv.Image(
            img, kdims=[self.hvdims["x"], self.hvdims["y"]], vdims=[self.vdim]
        )

        clim = self.clim
        if bool(getattr(self, "xy_autoscale", False)):
            qlo, qhi = self.xy_clim_quantiles
            clim = _safe_quantile_clim(img.values, float(qlo), float(qhi), fallback=clim)

        if self.use_datashader:
            base = rasterize(
                base,
                aggregator=ds.mean("val"),
                width=width,
                height=height,
                dynamic=False,
            )

        return base.opts(
            cmap="Viridis",
            colorbar=True,
            framewise=False,
            title=self._title_xy(),
            clim=clim,
            width=width,
            height=height,
            tools=["tap", "pan", "wheel_zoom", "box_zoom", "reset"],
            active_tools=["tap"],
            hooks=[_set_active_scroll_wheel_zoom],
        )

    # --------------------------- public HV views ------------------------------
    @param.depends(
        "t",
        "z",
        "x",
        "y",
        "clim",
        "use_datashader",
        "tz_width",
        "ranger_height",
    )
    def view_tz(self):
        return self._tz_layout

    def view_xy(self):
        return self._xy_layout

    # ------------------------------- controls --------------------------------
    def _set_controls(self):
        # discrete coordinate options
        x_vals = list(map(float, self.array.x.values))
        y_vals = list(map(float, self.array.y.values))
        z_vals = list(map(float, self.array.z.values))

        # param controls (x, y, z, clim, datashader)
        control_params = [
            "x",
            "y",
            "z",
            "clim",
            "tz_autoscale",
            "tz_clim_quantiles",
            "xy_autoscale",
            "xy_clim_quantiles",
            "use_datashader",
            "tz_logy",
            "merge_tools",
        ]
        control_params = [p for p in control_params if p in self.param]
        widgets = {
            "x": pn.widgets.DiscreteSlider(options=x_vals, name=self.ox),
            "y": pn.widgets.DiscreteSlider(options=y_vals, name=self.oy),
            "z": pn.widgets.DiscreteSlider(options=z_vals, name=self.oz),
            "clim": pn.widgets.RangeSlider,
            "tz_autoscale": pn.widgets.Checkbox,
            "xy_autoscale": pn.widgets.Checkbox,
            "tz_clim_quantiles": pn.widgets.RangeSlider,
            "xy_clim_quantiles": pn.widgets.RangeSlider,
            "use_datashader": pn.widgets.Checkbox,
        }
        if "tz_logy" in control_params:
            widgets["tz_logy"] = pn.widgets.Checkbox

        self.param_controls = pn.Param(
            self,
            parameters=control_params,
            widgets=widgets,
            show_name=False,
            width=400,
        )

        # time player (kept bi-directional with self.t)
        self._set_t_player()

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

    # ------------------------- navigation helpers ---------------------------
    def _t_bounds(self) -> tuple[float, float]:
        return tuple(map(float, self.param["t"].bounds))

    def _default_window_around(self, t_center: float) -> tuple[float, float]:
        t0, t1 = self._t_bounds()
        width = float(self.default_window_fraction) * float(t1 - t0)
        width = max(width, (t1 - t0) * 1e-9)
        return _clip_pair(float(t_center) - 0.5 * width, float(t_center) + 0.5 * width, (t0, t1))

    def _push_window_history(self, t_window: tuple[float, float]) -> None:
        tw = (float(t_window[0]), float(t_window[1]))
        if self._t_window_history_idx >= 0 and self._t_window_history:
            cur = self._t_window_history[self._t_window_history_idx]
            if abs(cur[0] - tw[0]) < 1e-12 and abs(cur[1] - tw[1]) < 1e-12:
                return
        if self._t_window_history_idx < len(self._t_window_history) - 1:
            self._t_window_history = self._t_window_history[: self._t_window_history_idx + 1]
        self._t_window_history.append(tw)
        self._t_window_history_idx = len(self._t_window_history) - 1

    def history_back(self) -> None:
        if self._t_window_history_idx <= 0:
            return
        self._t_window_history_idx -= 1
        self.t_window = self._t_window_history[self._t_window_history_idx]

    def history_forward(self) -> None:
        if self._t_window_history_idx >= len(self._t_window_history) - 1:
            return
        self._t_window_history_idx += 1
        self.t_window = self._t_window_history[self._t_window_history_idx]

    def goto(self, t: float) -> None:
        self.t = nearest_sel(self.array.t, float(t))
        if bool(self.reset_zoom_on_navigation):
            self.t_window = self._default_window_around(float(self.t))
        self._push_window_history(self.t_window)

    def advance(self, dt: float | None = None) -> None:
        step = float(self.nav_step_s if dt is None else dt)
        self.goto(float(self.t) + step)

    def back(self, dt: float | None = None) -> None:
        step = float(self.nav_step_s if dt is None else dt)
        self.goto(float(self.t) - step)

    def reset_view(self) -> None:
        self.t_window = self._default_window_around(float(self.t))
        self._push_window_history(self.t_window)

    # ------------------------- UI helpers ---------------------------
    def _help_markdown(self) -> str:
        return (
            "### Help\n"
            "- **XY**: click to move (x,y) crosshair.\n"
            "- **TZ**: click to move (t,z) crosshair.\n"
            "- **Range selector**: drag/resize the box on the overview curve to change the TZ time window.\n"
            "- **Navigation**: Prev/Next step by `Step (s)`.\n"
            "- **Shortcuts** (optional): left/right arrows step time, `r` resets view, `b` goes back in history.\n"
        )

    def _nav_controls(self):
        back_btn = pn.widgets.Button(name="◀", width=40)
        fwd_btn = pn.widgets.Button(name="▶", width=40)
        prev_btn = pn.widgets.Button(name="Prev", width=60)
        next_btn = pn.widgets.Button(name="Next", width=60)
        reset_btn = pn.widgets.Button(name="Reset view", button_type="primary", width=110)

        back_btn.on_click(lambda _e: self.history_back())
        fwd_btn.on_click(lambda _e: self.history_forward())
        prev_btn.on_click(lambda _e: self.back())
        next_btn.on_click(lambda _e: self.advance())
        reset_btn.on_click(lambda _e: self.reset_view())

        return pn.Row(
            back_btn,
            fwd_btn,
            prev_btn,
            next_btn,
            pn.Param(self, parameters=["nav_step_s", "reset_zoom_on_navigation"], show_name=False, width=260),
            reset_btn,
        )

    def _maybe_register_shortcuts(self):
        if not bool(self.enable_shortcuts) or self._shortcuts_registered:
            return
        try:
            if pn.state.curdoc is None:
                return

            def _on_key(event):
                key = event.key
                if key == "ArrowLeft":
                    self.back()
                elif key == "ArrowRight":
                    self.advance()
                elif key in ("r", "R"):
                    self.reset_view()
                elif key in ("b", "B"):
                    self.history_back()

            pn.state.on_keydown(_on_key)
            self._shortcuts_registered = True
        except Exception:
            return

    # ------------------------------ panel app --------------------------------
    def controls_panel(self):
        help_card = pn.Card(
            pn.pane.Markdown(self._help_markdown()),
            title="Help",
            collapsible=True,
            collapsed=True,
            sizing_mode="stretch_width",
        )
        return pn.Column(
            pn.pane.Markdown("### Time"),
            self.t_player_widget.view,
            pn.layout.Divider(),
            pn.pane.Markdown("### Navigation"),
            self._nav_controls(),
            pn.layout.Divider(),
            pn.pane.Markdown("### Controls"),
            self.param_controls,
            pn.layout.Divider(),
            pn.Param(self, parameters=["enable_shortcuts"], show_name=False, width=200),
            help_card,
        )

    def panel_app(self):
        """
        Full Panel application
        Left: TZ target + range curve (linked)
        └─ Controls (bottom)
        Right: XY orthoslice
        """
        self._maybe_register_shortcuts()
        tz_pane = pn.pane.HoloViews(self._tz_layout, width=400, height=420, sizing_mode="fixed")
        xy_pane = pn.pane.HoloViews(
            self.view_xy, width=400, height=300, sizing_mode="fixed"
        )

        left_col = pn.Column(
            tz_pane,
            self.controls_panel(),
            sizing_mode="stretch_width",
        )
        right_col = pn.Column(xy_pane, sizing_mode="stretch_width")
        return pn.Row(left_col, right_col, sizing_mode="stretch_width")

    def show(self):
        self.panel_app().show()
